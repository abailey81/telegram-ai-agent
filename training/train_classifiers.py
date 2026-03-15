"""
Training Script for Custom NLP Classifiers.

Trains three types of classifiers:
1. Embedding + sklearn classifiers (fast, reliable baseline)
2. Evaluates models and saves the best performers

Usage:
    python -m training.train_classifiers          # Train all classifiers
    python -m training.train_classifiers --eval    # Train + evaluate
    python -m training.train_classifiers --stats   # Show data statistics only

Models are saved to trained_models/ as .joblib files for use by dl_models.py.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
logger = logging.getLogger("train")

TRAINED_MODEL_DIR = Path(__file__).parent.parent / "trained_models"
TRAINED_MODEL_DIR.mkdir(exist_ok=True)


def train_sklearn_classifiers():
    """Train sklearn classifiers on sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.ensemble import (
            RandomForestClassifier,
            GradientBoostingClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report
        import joblib
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install sentence-transformers scikit-learn joblib numpy")
        return

    from training.training_data import get_all_data, get_data_stats

    # Show data stats
    stats = get_data_stats()
    logger.info(f"Total training examples: {stats['total_examples_all']}")
    for name, s in stats.items():
        if isinstance(s, dict) and "total_examples" in s:
            logger.info(f"  {name}: {s['total_examples']} examples, {s['num_classes']} classes")

    # Load embedding model
    logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_data = get_all_data()

    for task_name, data in all_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {task_name}")
        logger.info(f"{'='*60}")

        texts = [text for text, _ in data]
        labels = [label for _, label in data]

        # Encode texts
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        logger.info(f"Classes: {list(le.classes_)}")

        # Train multiple classifiers and pick the best
        classifiers = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                C=10.0,
                class_weight="balanced",
                solver="lbfgs",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
            "svm": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        }

        best_clf_name = None
        best_score = 0.0
        best_clf = None
        cv_results = {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for clf_name, clf in classifiers.items():
            logger.info(f"  Training {clf_name}...")
            try:
                scores = cross_val_score(clf, embeddings, y, cv=cv, scoring="accuracy", n_jobs=-1)
                mean_score = scores.mean()
                std_score = scores.std()
                cv_results[clf_name] = {
                    "mean_accuracy": round(mean_score, 4),
                    "std": round(std_score, 4),
                    "scores": [round(s, 4) for s in scores],
                }
                logger.info(
                    f"    {clf_name}: accuracy={mean_score:.4f} (+/- {std_score:.4f})"
                )

                if mean_score > best_score:
                    best_score = mean_score
                    best_clf_name = clf_name
                    best_clf = clf
            except Exception as e:
                logger.error(f"    {clf_name} failed: {e}")

        if best_clf is None:
            logger.error(f"No classifier succeeded for {task_name}")
            continue

        logger.info(f"\n  Best classifier: {best_clf_name} (accuracy={best_score:.4f})")

        # Train the best classifier on all data
        logger.info(f"  Training final {best_clf_name} on all data...")
        best_clf.fit(embeddings, y)

        # Full evaluation
        y_pred = best_clf.predict(embeddings)
        report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
        logger.info(f"  Training accuracy: {report['accuracy']:.4f}")

        # Print per-class metrics
        for cls_name in le.classes_:
            cls_metrics = report[cls_name]
            logger.info(
                f"    {cls_name}: precision={cls_metrics['precision']:.3f} "
                f"recall={cls_metrics['recall']:.3f} f1={cls_metrics['f1-score']:.3f}"
            )

        # Save the pipeline (classifier + label encoder)
        pipeline = {
            "classifier": best_clf,
            "label_encoder": le,
            "classifier_type": best_clf_name,
            "cv_accuracy": best_score,
            "n_classes": len(le.classes_),
            "classes": list(le.classes_),
            "n_training_examples": len(texts),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings.shape[1],
        }

        # Save model
        model_path = TRAINED_MODEL_DIR / f"{task_name}.joblib"
        joblib.dump(pipeline, model_path)
        logger.info(f"  Saved model to {model_path}")

        # Save metadata
        meta_path = TRAINED_MODEL_DIR / f"{task_name}_meta.json"
        meta = {
            "task": task_name,
            "best_classifier": best_clf_name,
            "cv_accuracy": best_score,
            "all_cv_results": cv_results,
            "n_classes": len(le.classes_),
            "classes": list(le.classes_),
            "n_training_examples": len(texts),
            "training_report": {
                k: v
                for k, v in report.items()
                if isinstance(v, dict)
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"  Saved metadata to {meta_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Models saved to: {TRAINED_MODEL_DIR}")
    logger.info("=" * 60)


def show_stats():
    """Show training data statistics."""
    from training.training_data import get_data_stats

    stats = get_data_stats()
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train custom NLP classifiers")
    parser.add_argument("--stats", action="store_true", help="Show data statistics only")
    parser.add_argument("--eval", action="store_true", help="Train with full evaluation")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    train_sklearn_classifiers()


if __name__ == "__main__":
    main()
