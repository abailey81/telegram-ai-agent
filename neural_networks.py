"""
Custom Neural Network Architectures for Conversation Analysis.

Implements domain-specific deep learning models:

1. TextCNN - Convolutional Neural Network for text classification
   - Multi-kernel convolutions (sizes 2,3,4,5) for n-gram capture
   - Dropout regularization, BatchNorm
   - Used for: emotion detection, intent classification

2. ConversationLSTM - Bidirectional LSTM for sequence modeling
   - Captures temporal conversation dynamics
   - Attention mechanism over hidden states
   - Used for: conversation stage prediction, momentum detection

3. EmotionAttentionNet - Self-attention network for emotion analysis
   - Multi-head self-attention on word embeddings
   - Residual connections, layer normalization
   - Used for: nuanced emotion detection with context

4. ConversationTransformer - Lightweight transformer for conversation flow
   - Positional encoding for message order
   - Cross-attention between user messages
   - Used for: predicting conversation trajectory

5. EnsemblePredictor - Combines all models for robust prediction
   - Weighted voting across architectures
   - Confidence calibration

All models use PyTorch and support CPU/GPU/MPS inference.
Training uses the data from training/training_data.py
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

nn_logger = logging.getLogger("neural_networks")
nn_logger.setLevel(logging.INFO)

NEURAL_MODEL_DIR = Path(__file__).parent / "trained_models" / "neural"
NEURAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Check PyTorch availability
_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    _torch_available = True
except ImportError:
    nn_logger.warning("PyTorch not available - neural network models disabled")


if _torch_available:

    # ═══════════════════════════════════════════════════════════════
    #  DATASET
    # ═══════════════════════════════════════════════════════════════

    class TextDataset(Dataset):
        """Dataset for text classification using pre-computed embeddings."""

        def __init__(self, embeddings, labels):
            self.embeddings = torch.FloatTensor(embeddings)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    class SequenceDataset(Dataset):
        """Dataset for sequence classification (conversation-level)."""

        def __init__(self, sequences, labels, max_len=20):
            """
            sequences: list of lists of embeddings (one list per conversation)
            labels: list of labels
            """
            self.max_len = max_len
            self.labels = torch.LongTensor(labels)
            self.sequences = []
            self.lengths = []

            for seq in sequences:
                if len(seq) > max_len:
                    seq = seq[-max_len:]  # take most recent
                self.lengths.append(len(seq))
                # Pad to max_len
                if len(seq) < max_len:
                    pad_size = max_len - len(seq)
                    padding = [[0.0] * len(seq[0])] * pad_size
                    seq = padding + seq
                self.sequences.append(seq)

            self.sequences = torch.FloatTensor(self.sequences)
            self.lengths = torch.LongTensor(self.lengths)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.sequences[idx], self.lengths[idx], self.labels[idx]

    # ═══════════════════════════════════════════════════════════════
    #  1. TextCNN - Convolutional Neural Network
    # ═══════════════════════════════════════════════════════════════

    class TextCNN(nn.Module):
        """Multi-kernel CNN for text classification from embeddings.

        Architecture:
        - Input: sentence embedding (384-dim from MiniLM)
        - Reshape to 1D signal
        - Parallel convolution with kernel sizes [2, 3, 4, 5]
        - Max-over-time pooling per kernel
        - Concatenate + Dense layers
        - Output: class probabilities

        This captures local patterns at different scales in the
        embedding representation.
        """

        def __init__(
            self,
            input_dim: int = 384,
            num_classes: int = 12,
            num_filters: int = 128,
            kernel_sizes: List[int] = None,
            dropout: float = 0.3,
        ):
            super().__init__()
            if kernel_sizes is None:
                kernel_sizes = [2, 3, 4, 5]

            self.input_dim = input_dim
            self.num_classes = num_classes

            # Project embedding to a sequence-like representation
            # Reshape 384-dim vector into (seq_len, channels) for Conv1d
            self.seq_len = 24  # 384 / 16 = 24 time steps
            self.channels = input_dim // self.seq_len  # 16 channels

            # Conv layers with different kernel sizes
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(self.channels, num_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for k in kernel_sizes
            ])

            total_filters = num_filters * len(kernel_sizes)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(total_filters, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            # x: (batch_size, 384)
            batch_size = x.size(0)

            # Reshape to (batch, channels, seq_len)
            x = x.view(batch_size, self.channels, self.seq_len)

            # Apply each conv + pool
            conv_outputs = [conv(x).squeeze(-1) for conv in self.convs]

            # Concatenate: (batch, total_filters)
            x = torch.cat(conv_outputs, dim=1)

            # Classify
            return self.classifier(x)

    # ═══════════════════════════════════════════════════════════════
    #  2. ConversationLSTM - Bidirectional LSTM with Attention
    # ═══════════════════════════════════════════════════════════════

    class AttentionLayer(nn.Module):
        """Bahdanau-style attention over LSTM hidden states."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, lstm_output, lengths=None):
            # lstm_output: (batch, seq_len, hidden_dim)
            attn_weights = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

            # Mask padding
            if lengths is not None:
                max_len = lstm_output.size(1)
                mask = torch.arange(max_len, device=lstm_output.device).unsqueeze(0)
                mask = mask >= (max_len - lengths.unsqueeze(1))
                attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)

            # Weighted sum
            context = torch.bmm(
                attn_weights.unsqueeze(1), lstm_output
            ).squeeze(1)  # (batch, hidden_dim)

            return context, attn_weights

    class ConversationLSTM(nn.Module):
        """Bidirectional LSTM for conversation sequence modeling.

        Architecture:
        - Input: sequence of message embeddings (384-dim each)
        - Bidirectional LSTM (2 layers)
        - Self-attention over hidden states
        - Dense classification head

        Captures temporal conversation dynamics:
        - Message order and flow
        - Building momentum or cooling down
        - Emotional arcs across messages
        """

        def __init__(
            self,
            input_dim: int = 384,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 7,
            dropout: float = 0.3,
            bidirectional: bool = True,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Attention
            self.attention = AttentionLayer(hidden_dim * self.num_directions)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x, lengths=None):
            # x: (batch, seq_len, input_dim)
            batch_size, seq_len, _ = x.size()

            # Project input
            x = self.input_proj(x)

            # LSTM forward
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * num_directions)

            # Attention pooling
            context, attn_weights = self.attention(lstm_out, lengths)

            # Classify
            return self.classifier(context)

    # ═══════════════════════════════════════════════════════════════
    #  3. EmotionAttentionNet - Multi-Head Self-Attention
    # ═══════════════════════════════════════════════════════════════

    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention mechanism."""

        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            assert embed_dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(self, x):
            batch_size, seq_len, embed_dim = x.size()

            # Project Q, K, V
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores
            attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Apply attention to values
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

            return self.out_proj(out)

    class EmotionAttentionNet(nn.Module):
        """Self-attention network for nuanced emotion analysis.

        Architecture:
        - Input: sentence embedding (384-dim)
        - Reshape to pseudo-sequence
        - Multi-head self-attention (4 heads)
        - Residual connections + LayerNorm
        - Feed-forward network
        - Global average pooling
        - Classification head

        Captures complex emotional patterns by allowing
        different parts of the embedding to attend to each other.
        """

        def __init__(
            self,
            input_dim: int = 384,
            num_classes: int = 11,
            num_heads: int = 4,
            ff_dim: int = 512,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.seq_len = 24
            self.embed_dim = input_dim // self.seq_len  # 16

            # Project to larger embedding space
            self.input_proj = nn.Linear(self.embed_dim, 64)
            work_dim = 64

            # Transformer-like blocks
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    "attn": MultiHeadSelfAttention(work_dim, num_heads, dropout),
                    "norm1": nn.LayerNorm(work_dim),
                    "ff": nn.Sequential(
                        nn.Linear(work_dim, ff_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, work_dim),
                        nn.Dropout(dropout),
                    ),
                    "norm2": nn.LayerNorm(work_dim),
                }))

            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(work_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            batch_size = x.size(0)
            # Reshape: (batch, 384) -> (batch, 24, 16)
            x = x.view(batch_size, self.seq_len, -1)
            x = self.input_proj(x)  # (batch, 24, 64)

            # Apply transformer blocks
            for layer in self.layers:
                # Self-attention with residual
                attn_out = layer["attn"](layer["norm1"](x))
                x = x + attn_out

                # Feed-forward with residual
                ff_out = layer["ff"](layer["norm2"](x))
                x = x + ff_out

            # Global average pooling
            x = x.mean(dim=1)  # (batch, 64)

            return self.classifier(x)

    # ═══════════════════════════════════════════════════════════════
    #  4. ConversationTransformer - Lightweight Transformer
    # ═══════════════════════════════════════════════════════════════

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for message position."""

        def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class ConversationTransformer(nn.Module):
        """Lightweight transformer for conversation flow prediction.

        Architecture:
        - Input: sequence of message embeddings with sender info
        - Positional encoding for message order
        - 2-layer transformer encoder
        - CLS token pooling
        - Classification head

        Understands the flow and dynamics of a conversation
        by attending to relationships between all messages.
        """

        def __init__(
            self,
            input_dim: int = 384,
            d_model: int = 128,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            dim_feedforward: int = 256,
            num_classes: int = 7,
            dropout: float = 0.2,
            max_seq_len: int = 30,
        ):
            super().__init__()

            # Input projection (embedding + sender feature -> d_model)
            self.input_proj = nn.Linear(input_dim + 1, d_model)  # +1 for sender flag

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

            # CLS token (learnable)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_encoder_layers
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x, sender_flags=None):
            """
            x: (batch, seq_len, input_dim) - message embeddings
            sender_flags: (batch, seq_len, 1) - 0=them, 1=me
            """
            batch_size = x.size(0)

            # Append sender flag
            if sender_flags is not None:
                x = torch.cat([x, sender_flags], dim=-1)
            else:
                # Default: add zero column
                zeros = torch.zeros(batch_size, x.size(1), 1, device=x.device)
                x = torch.cat([x, zeros], dim=-1)

            # Project to d_model
            x = self.input_proj(x)

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add positional encoding
            x = self.pos_encoder(x)

            # Transformer encoding
            x = self.transformer_encoder(x)

            # Extract CLS token output
            cls_output = x[:, 0]

            return self.classifier(cls_output)

    # ═══════════════════════════════════════════════════════════════
    #  5. EnsemblePredictor
    # ═══════════════════════════════════════════════════════════════

    class EnsemblePredictor:
        """Combines predictions from multiple neural network models.

        Uses weighted soft voting for robust predictions.
        Each model contributes based on its validation accuracy.
        """

        def __init__(self):
            self.models: Dict[str, Tuple[nn.Module, float]] = {}
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )

        def add_model(self, name: str, model: nn.Module, weight: float = 1.0):
            """Add a model to the ensemble."""
            model.eval()
            model.to(self.device)
            self.models[name] = (model, weight)

        def predict(self, x: torch.Tensor) -> Dict[str, Any]:
            """Get ensemble prediction."""
            if not self.models:
                return {"error": "No models in ensemble"}

            x = x.to(self.device)
            all_probs = []
            weights = []

            with torch.no_grad():
                for name, (model, weight) in self.models.items():
                    try:
                        logits = model(x)
                        probs = F.softmax(logits, dim=-1)
                        all_probs.append(probs * weight)
                        weights.append(weight)
                    except Exception as e:
                        nn_logger.warning(f"Model {name} failed: {e}")

            if not all_probs:
                return {"error": "All models failed"}

            # Weighted average
            total_weight = sum(weights)
            ensemble_probs = sum(all_probs) / total_weight

            # Get prediction
            pred_idx = ensemble_probs.argmax(dim=-1).item()
            confidence = ensemble_probs[0, pred_idx].item()

            return {
                "prediction_idx": pred_idx,
                "confidence": round(confidence, 4),
                "all_probabilities": ensemble_probs[0].cpu().tolist(),
            }


# ═══════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def train_neural_models(
    task_name: str = "all",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hparams: Optional[Dict[str, Any]] = None,
    data_override: Optional[Dict[str, List[Tuple]]] = None,
    model_types: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
):
    """Train neural network models on the training data.

    Args:
        task_name: "romantic_intent", "conversation_stage",
                   "emotional_tone", or "all"
        epochs: number of training epochs
        batch_size: training batch size
        learning_rate: initial learning rate
        hparams: Optional hyperparameter overrides dict. Supported keys:
            - epochs, batch_size, learning_rate, dropout
            - num_filters, kernel_sizes (TextCNN)
            - num_heads, num_layers, ff_dim (EmotionAttentionNet)
            - weight_decay, label_smoothing
            - optimizer ("adamw", "adam", "sgd")
            - scheduler ("cosine", "step", "plateau")
        data_override: Optional dict of {task: [(text, label), ...]} to use
            instead of get_all_data(). Used by autoresearch.
        model_types: Optional list of model types to train (e.g., ["textcnn"]).
            Default: ["textcnn", "emotion_attn"]
        save_dir: Optional directory to save models to (default: NEURAL_MODEL_DIR)

    Returns:
        Dict of {task: {model_name: {"val_acc": float, "save_path": str}}}
    """
    if not _torch_available:
        nn_logger.error("PyTorch not available. Install with: pip install torch")
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import numpy as np
    except ImportError as e:
        nn_logger.error(f"Missing dependency: {e}")
        return {}

    # Apply hparams overrides
    hp = hparams or {}
    epochs = hp.get("epochs", epochs)
    batch_size = hp.get("batch_size", batch_size)
    learning_rate = hp.get("learning_rate", learning_rate)
    dropout = hp.get("dropout", 0.3)
    num_filters = hp.get("num_filters", 128)
    kernel_sizes = hp.get("kernel_sizes", [2, 3, 4, 5])
    num_heads = hp.get("num_heads", 4)
    num_layers = hp.get("num_layers", 2)
    ff_dim = hp.get("ff_dim", 512)
    weight_decay = hp.get("weight_decay", 0.01)
    label_smoothing = hp.get("label_smoothing", 0.0)
    optimizer_name = hp.get("optimizer", "adamw")
    scheduler_name = hp.get("scheduler", "cosine")

    out_dir = Path(save_dir) if save_dir else NEURAL_MODEL_DIR

    if data_override:
        all_data = data_override
    else:
        from training.training_data import get_all_data
        all_data = get_all_data()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    nn_logger.info(f"Training on device: {device}")

    # Load embedder
    nn_logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    tasks = [task_name] if task_name != "all" else list(all_data.keys())
    all_results = {}

    for task in tasks:
        if task not in all_data:
            nn_logger.warning(f"Unknown task: {task}")
            continue

        data = all_data[task]
        nn_logger.info(f"\n{'='*60}")
        nn_logger.info(f"Training neural models for: {task}")
        nn_logger.info(f"{'='*60}")

        texts = [t for t, _ in data]
        labels = [l for _, l in data]

        # Encode
        nn_logger.info(f"Encoding {len(texts)} texts...")
        embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        le = LabelEncoder()
        y = le.fit_transform(labels)
        num_classes = len(le.classes_)

        nn_logger.info(f"Classes ({num_classes}): {list(le.classes_)}")

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, y, test_size=0.15, stratify=y, random_state=42
        )

        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Models to train (filtered by model_types if specified)
        available_models = {
            "textcnn": TextCNN(
                input_dim=384, num_classes=num_classes,
                num_filters=num_filters, kernel_sizes=kernel_sizes, dropout=dropout,
            ),
            "emotion_attn": EmotionAttentionNet(
                input_dim=384, num_classes=num_classes,
                num_heads=num_heads, num_layers=num_layers,
                ff_dim=ff_dim, dropout=dropout,
            ),
        }
        if model_types:
            models_to_train = {k: v for k, v in available_models.items() if k in model_types}
        else:
            models_to_train = available_models

        best_models = {}

        for model_name, model in models_to_train.items():
            nn_logger.info(f"\n  Training {model_name}...")
            model = model.to(device)

            # Configurable optimizer
            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    momentum=0.9, weight_decay=weight_decay,
                )
            elif optimizer_name == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
            else:  # adamw (default)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )

            # Configurable scheduler
            if scheduler_name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
            elif scheduler_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", patience=5, factor=0.5,
                )
            else:  # cosine (default)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            # Class weights for imbalanced data
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / (class_counts + 1)
            class_weights = class_weights / class_weights.sum() * num_classes
            weight_tensor = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=label_smoothing,
            )

            best_val_acc = 0.0
            best_state = None
            patience = 10
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()

                if scheduler_name != "plateau":
                    scheduler.step()

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                train_acc = train_correct / max(train_total, 1)
                val_acc = val_correct / max(val_total, 1)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    nn_logger.info(
                        f"    Epoch {epoch+1}/{epochs}: "
                        f"train_loss={train_loss/len(train_loader):.4f}, "
                        f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
                    )

                # Plateau scheduler needs val metric
                if scheduler_name == "plateau":
                    scheduler.step(val_acc)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model state
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        nn_logger.info(f"    Early stopping at epoch {epoch+1}")
                        break

            # Restore best model
            if best_state:
                model.load_state_dict(best_state)
                model.to(device)

            nn_logger.info(f"    Best validation accuracy: {best_val_acc:.4f}")
            best_models[model_name] = (model, best_val_acc)

            # Save model
            save_path = out_dir / f"{task}_{model_name}.pt"
            torch.save({
                "model_state_dict": model.cpu().state_dict(),
                "model_class": model_name,
                "num_classes": num_classes,
                "classes": list(le.classes_),
                "val_accuracy": best_val_acc,
                "input_dim": 384,
                "hparams": hp,
            }, save_path)
            nn_logger.info(f"    Saved to {save_path}")

        # Save label encoder
        import joblib
        le_path = out_dir / f"{task}_label_encoder.joblib"
        joblib.dump(le, le_path)

        # Save ensemble metadata
        task_results = {}
        meta = {
            "task": task,
            "models": {},
            "classes": list(le.classes_),
            "num_classes": num_classes,
        }
        for name, (_, acc) in best_models.items():
            meta["models"][name] = {"val_accuracy": acc}
            task_results[name] = {
                "val_acc": acc,
                "save_path": str(out_dir / f"{task}_{name}.pt"),
            }
        meta_path = out_dir / f"{task}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        all_results[task] = task_results

    nn_logger.info("\nNeural network training complete!")
    return all_results


def load_neural_model(task: str, model_name: str) -> Optional[Any]:
    """Load a trained neural network model."""
    if not _torch_available:
        return None

    model_path = NEURAL_MODEL_DIR / f"{task}_{model_name}.pt"
    if not model_path.exists():
        return None

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        hp = checkpoint.get("hparams", {})

        if model_name == "textcnn":
            model = TextCNN(
                input_dim=checkpoint["input_dim"],
                num_classes=checkpoint["num_classes"],
                num_filters=hp.get("num_filters", 128),
                kernel_sizes=hp.get("kernel_sizes", [2, 3, 4, 5]),
                dropout=hp.get("dropout", 0.3),
            )
        elif model_name == "emotion_attn":
            model = EmotionAttentionNet(
                input_dim=checkpoint["input_dim"],
                num_classes=checkpoint["num_classes"],
                num_heads=hp.get("num_heads", 4),
                num_layers=hp.get("num_layers", 2),
                ff_dim=hp.get("ff_dim", 512),
                dropout=hp.get("dropout", 0.3),
            )
        else:
            nn_logger.warning(f"Unknown model type: {model_name}")
            return None

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return {
            "model": model,
            "classes": checkpoint["classes"],
            "val_accuracy": checkpoint["val_accuracy"],
        }
    except Exception as e:
        nn_logger.error(f"Failed to load {task}/{model_name}: {e}")
        return None


# Cache for loaded neural models to avoid reloading from disk every prediction
_neural_model_cache: Dict[str, dict] = {}


def predict_with_neural(
    task: str,
    embedding,
    model_name: str = "textcnn",
) -> Optional[Dict[str, Any]]:
    """Make a prediction using a trained neural network model.

    Args:
        task: "romantic_intent", "conversation_stage", "emotional_tone"
        embedding: numpy array (384,) - sentence embedding
        model_name: "textcnn" or "emotion_attn"

    Returns: {"label": str, "confidence": float, "all_probabilities": dict}
    """
    if not _torch_available:
        return None

    cache_key = f"{task}_{model_name}"
    if cache_key in _neural_model_cache:
        loaded = _neural_model_cache[cache_key]
    else:
        loaded = load_neural_model(task, model_name)
        if loaded is not None:
            _neural_model_cache[cache_key] = loaded
    if loaded is None:
        return None

    model = loaded["model"]
    classes = loaded["classes"]

    try:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        model = model.to(device)

        x = torch.FloatTensor(embedding).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        all_probs = {
            classes[i]: round(probs[i].item(), 4) for i in range(len(classes))
        }

        return {
            "label": classes[pred_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "model": model_name,
        }
    except Exception as e:
        nn_logger.error(f"Neural prediction failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train neural network models")
    parser.add_argument("--task", default="all", help="Task to train (or 'all')")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_neural_models(
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
