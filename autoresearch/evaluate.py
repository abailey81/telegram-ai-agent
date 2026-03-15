"""
Autoresearch Evaluation — Metrics computation for model quality.

Computes per-task and composite scores used to compare experiments.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import TASK_WEIGHTS

logger = logging.getLogger("autoresearch.evaluate")


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """Compute accuracy, macro F1, and weighted F1."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "val_acc": round(acc, 6),
        "val_f1_macro": round(f1_macro, 6),
        "val_f1_weighted": round(f1_weighted, 6),
    }


def compute_composite_score(
    task_scores: Dict[str, Dict[str, float]],
    metric_key: str = "val_f1_weighted",
) -> float:
    """Compute weighted composite score across tasks.

    Args:
        task_scores: {task_name: {metric_name: value}}
        metric_key: Which metric to use for composite (default: val_f1_weighted)

    Returns:
        Weighted composite score (higher = better).
    """
    # Compute weighted average of per-task scores.
    # When only a subset of tasks is present, we compute a weighted average
    # over the present tasks only (no inflation, no deflation).
    # This means single-task and multi-task experiments are comparable.
    score = 0.0
    total_weight = 0.0
    for task, weight in TASK_WEIGHTS.items():
        if task in task_scores and metric_key in task_scores[task]:
            score += weight * task_scores[task][metric_key]
            total_weight += weight

    if total_weight > 0:
        score /= total_weight  # weighted average (always in [0, 1])

    return round(score, 6)


def evaluate_model(
    model_path: str,
    task: str,
    data: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Evaluate a saved model on validation data.

    Args:
        model_path: Path to .pt model file.
        task: Task name (romantic_intent, etc.)
        data: Optional override data. If None, uses get_all_data().

    Returns:
        Dict with val_acc, val_f1_macro, val_f1_weighted.
    """
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from neural_networks import TextCNN, EmotionAttentionNet

    if data is None:
        from training.training_data import get_all_data
        all_data = get_all_data()
        data = all_data.get(task, [])

    if not data:
        logger.warning(f"No data for task {task}")
        return {"val_acc": 0.0, "val_f1_macro": 0.0, "val_f1_weighted": 0.0}

    # Load model — use saved hparams for architecture reconstruction
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_class = checkpoint.get("model_class", "textcnn")
    num_classes = checkpoint["num_classes"]
    hp = checkpoint.get("hparams", {})

    if model_class == "textcnn":
        model = TextCNN(
            input_dim=384, num_classes=num_classes,
            num_filters=hp.get("num_filters", 128),
            kernel_sizes=hp.get("kernel_sizes", [2, 3, 4, 5]),
            dropout=hp.get("dropout", 0.3),
        )
    elif model_class == "emotion_attn":
        model = EmotionAttentionNet(
            input_dim=384, num_classes=num_classes,
            num_heads=hp.get("num_heads", 4),
            num_layers=hp.get("num_layers", 2),
            ff_dim=hp.get("ff_dim", 512),
            dropout=hp.get("dropout", 0.3),
        )
    else:
        logger.error(f"Unknown model class: {model_class}")
        return {"val_acc": 0.0, "val_f1_macro": 0.0, "val_f1_weighted": 0.0}

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use label mapping from checkpoint to ensure consistency with training
    saved_classes = checkpoint.get("classes", [])

    # Prepare data — filter to only classes the model knows about
    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    if saved_classes:
        # Use the saved class ordering from training, not a re-fit LabelEncoder
        class_to_idx = {c: i for i, c in enumerate(saved_classes)}
        valid_indices = [i for i, l in enumerate(labels) if l in class_to_idx]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        y = np.array([class_to_idx[l] for l in labels])
    else:
        le = LabelEncoder()
        y = le.fit_transform(labels)

    if len(texts) < 10:
        logger.warning(f"Too few valid examples for task {task}: {len(texts)}")
        return {"val_acc": 0.0, "val_f1_macro": 0.0, "val_f1_weighted": 0.0}

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # Use same split as training
    _, X_val, _, y_val = train_test_split(
        embeddings, y, test_size=0.15, stratify=y, random_state=42
    )

    # Predict
    X_val_tensor = torch.FloatTensor(X_val)
    with torch.no_grad():
        outputs = model(X_val_tensor)
        _, predictions = outputs.max(1)

    return compute_metrics(y_val.tolist(), predictions.tolist(), num_classes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        task = sys.argv[2] if len(sys.argv) > 2 else "emotional_tone"
        metrics = evaluate_model(model_path, task)
        print(json.dumps(metrics, indent=2))
    else:
        print("Usage: python -m autoresearch.evaluate <model_path> [task]")
