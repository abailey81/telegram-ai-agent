"""
Deep Learning Model Manager for Advanced NLP.

Handles lazy-loading, caching, and inference for all transformer models.
Supports CPU and GPU inference with automatic device detection.
Graceful degradation if models aren't available.

Models used:
- Sentiment: distilbert-base-uncased-finetuned-sst-2-english
- Emotion: j-hartmann/emotion-english-distilroberta-base
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Zero-shot: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
- Custom classifiers: sklearn models trained on sentence embeddings
"""

import os
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

dl_logger = logging.getLogger("dl_models")
dl_logger.setLevel(logging.INFO)

# Model cache directory
MODEL_CACHE_DIR = Path(__file__).parent / ".model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

CUSTOM_MODEL_DIR = Path(__file__).parent / "trained_models"
CUSTOM_MODEL_DIR.mkdir(exist_ok=True)


class ModelManager:
    """Singleton manager for all deep learning models.

    Handles lazy loading, caching, and thread-safe access to models.
    Falls back gracefully when models or libraries aren't available.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        self._custom_classifiers: Dict[str, Any] = {}
        self._device = None
        self._torch_available = False
        self._transformers_available = False
        self._sentence_transformers_available = False
        self._sklearn_available = False

        self._check_dependencies()
        dl_logger.info(
            f"ModelManager initialized | torch={self._torch_available} | "
            f"transformers={self._transformers_available} | "
            f"sentence-transformers={self._sentence_transformers_available} | "
            f"sklearn={self._sklearn_available} | device={self._device}"
        )

    def _check_dependencies(self):
        """Check which ML libraries are available."""
        try:
            import torch

            self._torch_available = True
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

        try:
            import transformers  # noqa: F401

            self._transformers_available = True
        except ImportError:
            pass

        try:
            import sentence_transformers  # noqa: F401

            self._sentence_transformers_available = True
        except ImportError:
            pass

        try:
            import sklearn  # noqa: F401

            self._sklearn_available = True
        except ImportError:
            pass

    @property
    def is_available(self) -> bool:
        """Check if any DL capability is available."""
        return self._torch_available and self._transformers_available

    @property
    def has_embeddings(self) -> bool:
        """Check if sentence embeddings are available."""
        return self._sentence_transformers_available

    @property
    def has_custom_classifiers(self) -> bool:
        """Check if sklearn classifiers can be loaded."""
        return self._sklearn_available

    @property
    def device(self) -> str:
        return self._device or "cpu"

    # ─── Sentiment Model ────────────────────────────────────────

    def get_sentiment_pipeline(self):
        """Get the sentiment analysis pipeline (distilbert-based)."""
        if "sentiment" not in self._pipelines:
            if not self.is_available:
                return None
            try:
                from transformers import pipeline

                self._pipelines["sentiment"] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self._device == "cuda" else -1,
                    truncation=True,
                    max_length=512,
                )
                dl_logger.info("Loaded sentiment model: distilbert-base-uncased-finetuned-sst-2-english")
            except Exception as e:
                dl_logger.error(f"Failed to load sentiment model: {e}")
                self._pipelines["sentiment"] = None
        return self._pipelines.get("sentiment")

    def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Run transformer-based sentiment analysis.

        Returns: {"label": "POSITIVE"|"NEGATIVE", "score": 0.0-1.0} or None
        """
        pipe = self.get_sentiment_pipeline()
        if pipe is None:
            return None
        try:
            result = pipe(text[:512])[0]
            return {
                "label": result["label"].lower(),
                "score": round(result["score"], 4),
            }
        except Exception as e:
            dl_logger.error(f"Sentiment analysis failed: {e}")
            return None

    # ─── Emotion Detection Model ────────────────────────────────

    def get_emotion_pipeline(self):
        """Get the emotion detection pipeline (7 emotions)."""
        if "emotion" not in self._pipelines:
            if not self.is_available:
                return None
            try:
                from transformers import pipeline

                self._pipelines["emotion"] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=0 if self._device == "cuda" else -1,
                    truncation=True,
                    max_length=512,
                )
                dl_logger.info(
                    "Loaded emotion model: j-hartmann/emotion-english-distilroberta-base"
                )
            except Exception as e:
                dl_logger.error(f"Failed to load emotion model: {e}")
                self._pipelines["emotion"] = None
        return self._pipelines.get("emotion")

    def detect_emotions(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Detect emotions with confidence scores.

        Returns sorted list: [{"label": "joy", "score": 0.85}, ...] or None
        Emotions: anger, disgust, fear, joy, neutral, sadness, surprise
        """
        pipe = self.get_emotion_pipeline()
        if pipe is None:
            return None
        try:
            results = pipe(text[:512])[0]
            # Sort by score descending
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            return [{"label": r["label"], "score": round(r["score"], 4)} for r in results]
        except Exception as e:
            dl_logger.error(f"Emotion detection failed: {e}")
            return None

    # ─── Sentence Embeddings ────────────────────────────────────

    def get_embedder(self):
        """Get the sentence-transformers embedding model."""
        if "embedder" not in self._models:
            if not self._sentence_transformers_available:
                return None
            try:
                from sentence_transformers import SentenceTransformer

                self._models["embedder"] = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device=self._device,
                )
                dl_logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                dl_logger.error(f"Failed to load embedding model: {e}")
                self._models["embedder"] = None
        return self._models.get("embedder")

    def embed(self, texts: List[str]) -> Optional[Any]:
        """Compute sentence embeddings for a list of texts.

        Returns numpy array of shape (n_texts, 384) or None.
        """
        embedder = self.get_embedder()
        if embedder is None:
            return None
        try:
            import numpy as np

            embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            dl_logger.error(f"Embedding failed: {e}")
            return None

    def embed_single(self, text: str) -> Optional[Any]:
        """Compute embedding for a single text. Returns 1D numpy array or None."""
        result = self.embed([text])
        if result is not None:
            return result[0]
        return None

    def cosine_similarity(self, emb1, emb2) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            import numpy as np

            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot / (norm1 * norm2))
        except Exception:
            return 0.0

    # ─── Zero-Shot Classification ───────────────────────────────

    def get_zero_shot_pipeline(self):
        """Get the zero-shot classification pipeline."""
        if "zero_shot" not in self._pipelines:
            if not self.is_available:
                return None
            try:
                from transformers import pipeline

                self._pipelines["zero_shot"] = pipeline(
                    "zero-shot-classification",
                    model="valhalla/distilbart-mnli-12-1",
                    device=0 if self._device == "cuda" else -1,
                )
                dl_logger.info("Loaded zero-shot model: valhalla/distilbart-mnli-12-1")
            except Exception as e:
                dl_logger.error(f"Failed to load zero-shot model: {e}")
                self._pipelines["zero_shot"] = None
        return self._pipelines.get("zero_shot")

    def zero_shot_classify(
        self, text: str, candidate_labels: List[str], multi_label: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Classify text into candidate labels using zero-shot NLI.

        Returns: {"labels": [...], "scores": [...]} or None
        """
        pipe = self.get_zero_shot_pipeline()
        if pipe is None:
            return None
        try:
            result = pipe(text[:512], candidate_labels, multi_label=multi_label)
            return {
                "labels": result["labels"],
                "scores": [round(s, 4) for s in result["scores"]],
                "top_label": result["labels"][0],
                "top_score": round(result["scores"][0], 4),
            }
        except Exception as e:
            dl_logger.error(f"Zero-shot classification failed: {e}")
            return None

    # ─── Custom Trained Classifiers ─────────────────────────────

    def load_custom_classifier(self, name: str) -> Optional[Any]:
        """Load a custom sklearn classifier from disk."""
        if name in self._custom_classifiers:
            return self._custom_classifiers[name]

        if not self._sklearn_available:
            return None

        model_path = CUSTOM_MODEL_DIR / f"{name}.joblib"
        if not model_path.exists():
            dl_logger.debug(f"Custom classifier '{name}' not found at {model_path}")
            return None

        try:
            import joblib

            loaded = joblib.load(model_path)
            # train_classifiers saves a dict with "classifier" and "label_encoder" keys
            if isinstance(loaded, dict) and "classifier" in loaded:
                clf = loaded["classifier"]
                le = loaded.get("label_encoder")
            else:
                clf = loaded
                le = None
            self._custom_classifiers[name] = {"clf": clf, "label_encoder": le}
            dl_logger.info(f"Loaded custom classifier: {name}")
            return self._custom_classifiers[name]
        except Exception as e:
            dl_logger.error(f"Failed to load custom classifier '{name}': {e}")
            return None

    def predict_with_custom(
        self, name: str, text: str
    ) -> Optional[Dict[str, Any]]:
        """Predict using a custom classifier (embedding + sklearn).

        Returns: {"label": str, "confidence": float, "all_probabilities": dict} or None
        """
        loaded = self.load_custom_classifier(name)
        if loaded is None:
            return None

        # Extract classifier and optional label encoder
        if isinstance(loaded, dict) and "clf" in loaded:
            clf = loaded["clf"]
            le = loaded.get("label_encoder")
        else:
            clf = loaded
            le = None

        embedding = self.embed_single(text)
        if embedding is None:
            return None

        try:
            import numpy as np

            embedding_2d = embedding.reshape(1, -1)
            raw_prediction = clf.predict(embedding_2d)[0]

            # Decode numeric label back to text using label encoder
            if le is not None:
                prediction = le.inverse_transform([raw_prediction])[0]
            else:
                prediction = raw_prediction

            # Get probabilities if available
            probabilities = {}
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(embedding_2d)[0]
                classes = clf.classes_
                if le is not None:
                    decoded_classes = le.inverse_transform(classes)
                else:
                    decoded_classes = classes
                probabilities = {
                    str(c): round(float(p), 4) for c, p in zip(decoded_classes, probs)
                }

            confidence = probabilities.get(str(prediction), 0.0)
            if confidence == 0.0 and hasattr(clf, "decision_function"):
                # For SVM without probability
                confidence = 0.5  # default

            return {
                "label": str(prediction),
                "confidence": confidence,
                "all_probabilities": probabilities,
            }
        except Exception as e:
            dl_logger.error(f"Custom classifier prediction failed for '{name}': {e}")
            return None

    # ─── Batch Operations ───────────────────────────────────────

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch for efficiency.

        Returns list of analysis results with sentiment + emotions.
        """
        results = []
        for text in texts:
            result = {
                "text": text[:100],
                "sentiment": self.analyze_sentiment(text),
                "emotions": self.detect_emotions(text),
            }
            results.append(result)
        return results

    # ─── Model Status ──────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get status of all models and capabilities."""
        custom_models = []
        if self._sklearn_available:
            for path in CUSTOM_MODEL_DIR.glob("*.joblib"):
                custom_models.append(path.stem)

        return {
            "device": self.device,
            "torch_available": self._torch_available,
            "transformers_available": self._transformers_available,
            "sentence_transformers_available": self._sentence_transformers_available,
            "sklearn_available": self._sklearn_available,
            "loaded_pipelines": list(self._pipelines.keys()),
            "loaded_models": list(self._models.keys()),
            "loaded_custom_classifiers": list(self._custom_classifiers.keys()),
            "available_custom_models": custom_models,
            "model_cache_dir": str(MODEL_CACHE_DIR),
            "custom_model_dir": str(CUSTOM_MODEL_DIR),
        }

    def preload_all(self):
        """Preload all models (call during startup for faster first inference)."""
        dl_logger.info("Preloading all models...")
        self.get_sentiment_pipeline()
        self.get_emotion_pipeline()
        self.get_embedder()
        self.get_zero_shot_pipeline()

        # Load all custom classifiers (skip label encoders)
        if self._sklearn_available:
            for path in CUSTOM_MODEL_DIR.glob("*.joblib"):
                if "_label_encoder" in path.stem:
                    continue
                self.load_custom_classifier(path.stem)

        dl_logger.info("All models preloaded.")

    def unload_all(self):
        """Unload all models to free memory."""
        self._models.clear()
        self._tokenizers.clear()
        self._pipelines.clear()
        self._custom_classifiers.clear()

        if self._torch_available:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass

        # Also clear the neural model cache in neural_networks module
        try:
            from neural_networks import _neural_model_cache
            _neural_model_cache.clear()
        except Exception:
            pass

        dl_logger.info("All models unloaded.")

    def reload_model(self, name: str):
        """Clear a specific cached model, forcing re-load from disk on next use.

        Used by autoresearch to hot-swap improved models without restarting.
        """
        removed = False
        for cache in (self._models, self._pipelines, self._custom_classifiers):
            if name in cache:
                del cache[name]
                removed = True
        if name in self._tokenizers:
            del self._tokenizers[name]
            removed = True
        # Also clear neural model cache for this model
        try:
            from neural_networks import _neural_model_cache
            if name in _neural_model_cache:
                del _neural_model_cache[name]
                removed = True
        except Exception:
            pass
        if removed:
            dl_logger.info(f"Reloaded model cache for: {name}")
        else:
            dl_logger.debug(f"Model {name} not in cache (nothing to reload)")

    def swap_model(self, name: str, new_path: str):
        """Atomically swap a production model file and reload from cache.

        Args:
            name: Model identifier (e.g., "romantic_intent_textcnn")
            new_path: Path to the new .pt or .joblib model file.
        """
        import shutil
        new_file = Path(new_path)
        if not new_file.exists():
            dl_logger.error(f"New model file not found: {new_path}")
            return False

        # Determine destination
        suffix = new_file.suffix
        dest = CUSTOM_MODEL_DIR / f"{name}{suffix}"

        # Backup existing
        if dest.exists():
            backup = CUSTOM_MODEL_DIR / f"{name}{suffix}.backup"
            shutil.copy2(dest, backup)

        # Copy new model
        shutil.copy2(new_file, dest)
        dl_logger.info(f"Swapped model {name}: {new_path} -> {dest}")

        # Clear cache so next inference loads the new model
        self.reload_model(name)
        return True


# ─── Module-Level Singleton Access ──────────────────────────────

_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager singleton."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
