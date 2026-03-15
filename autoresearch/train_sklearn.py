"""
Autoresearch — sklearn classifier experiments.

Trains sklearn classifiers (SVM, RF, LR, GBT) with randomized hyperparameters
on the same 3 tasks as neural networks, using harvested + curated data.

Returns metrics in the same format as train.py for unified comparison.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import (
    BUDGET_SECONDS, MODEL_VERSIONS_DIR, SKLEARN_PARAM_SPACE,
    SKLEARN_TASKS, SKLEARN_MODELS_DIR,
)

logger = logging.getLogger("autoresearch.train_sklearn")


def _sample_sklearn_hparams() -> dict:
    """Sample random hyperparameters for a random sklearn classifier type."""
    clf_type = random.choice(list(SKLEARN_PARAM_SPACE.keys()))
    space = SKLEARN_PARAM_SPACE[clf_type]

    hp = {"classifier_type": clf_type}
    for param, values in space.items():
        hp[param] = random.choice(values)

    hp["task"] = random.choice(SKLEARN_TASKS)
    hp["use_harvested_data"] = random.choice([True, True, True, False])  # 75% harvested
    hp["harvested_weight"] = random.choice([0.3, 0.5, 0.7, 1.0])

    return hp


def _build_classifier(hp: dict):
    """Build sklearn classifier from hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    clf_type = hp["classifier_type"]

    if clf_type == "svm":
        return SVC(
            kernel=hp.get("kernel", "rbf"),
            C=hp.get("C", 10.0),
            gamma=hp.get("gamma", "scale"),
            class_weight=hp.get("class_weight", "balanced"),
            probability=True,
            random_state=42,
        )
    elif clf_type == "logistic_regression":
        return LogisticRegression(
            C=hp.get("C", 10.0),
            solver=hp.get("solver", "lbfgs"),
            max_iter=hp.get("max_iter", 2000),
            class_weight=hp.get("class_weight", "balanced"),
        )
    elif clf_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth"),
            min_samples_split=hp.get("min_samples_split", 2),
            class_weight=hp.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )
    elif clf_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", 5),
            learning_rate=hp.get("learning_rate", 0.1),
            subsample=hp.get("subsample", 1.0),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def run_sklearn_experiment(hp: dict = None, budget: int = BUDGET_SECONDS) -> dict:
    """Run a single sklearn classifier experiment.

    Returns dict with metrics in the same format as neural network experiments.
    """
    import numpy as np
    import joblib
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import f1_score

    hp = hp or _sample_sklearn_hparams()
    task = hp.get("task", "emotional_tone")
    use_harvested = hp.get("use_harvested_data", True)
    harvested_weight = hp.get("harvested_weight", 0.5)

    start_time = time.time()
    exp_id = f"sklearn_{int(time.time())}"

    try:
        # Load data
        if use_harvested:
            from autoresearch.prepare import prepare_dataset
            all_data = prepare_dataset(refresh_harvest=True, harvested_weight=harvested_weight)
        else:
            from training.training_data import get_all_data
            all_data = get_all_data()

        data = all_data.get(task, [])
        if not data or len(data) < 20:
            return {
                "experiment_id": exp_id, "hparams": hp,
                "composite_score": 0.0, "training_time_s": round(time.time() - start_time, 1),
                "status": f"error: insufficient data for {task} ({len(data or [])} examples)",
                "experiment_type": "sklearn",
            }

        texts = [t for t, _ in data]
        labels = [l for _, l in data]

        # Embed with caching — encoding 6000+ texts takes 2+ min, reuse when possible
        cache_dir = Path(__file__).parent / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"{task}_{len(texts)}_{use_harvested}_{harvested_weight}"
        cache_path = cache_dir / f"embeddings_{cache_key}.npz"

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            if list(cached["texts"]) == texts:
                embeddings = cached["embeddings"]
                logger.info(f"Loaded cached embeddings for {task} ({len(texts)} examples)")
            else:
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                np.savez_compressed(cache_path, embeddings=embeddings, texts=np.array(texts))
        else:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            np.savez_compressed(cache_path, embeddings=embeddings, texts=np.array(texts))
            logger.info(f"Cached embeddings for {task} ({len(texts)} examples)")

        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Check budget
        if time.time() - start_time > budget:
            return {
                "experiment_id": exp_id, "hparams": hp,
                "composite_score": 0.0, "training_time_s": round(time.time() - start_time, 1),
                "status": "budget_exceeded", "experiment_type": "sklearn",
            }

        # Build classifier and cross-validate
        clf = _build_classifier(hp)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_acc = cross_val_score(clf, embeddings, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_f1 = cross_val_score(clf, embeddings, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

        mean_acc = float(cv_acc.mean())
        mean_f1 = float(cv_f1.mean())

        # Train final model on all data
        clf.fit(embeddings, y)

        elapsed = time.time() - start_time

        # Build task_scores in same format as neural experiments
        task_scores = {
            task: {
                "val_acc": round(mean_acc, 6),
                "val_f1_macro": round(mean_f1, 6),  # approximation
                "val_f1_weighted": round(mean_f1, 6),
            }
        }

        # Compute composite score (single task = its own score)
        from autoresearch.evaluate import compute_composite_score
        composite = compute_composite_score(task_scores)

        # Save experiment artifacts
        exp_dir = MODEL_VERSIONS_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        pipeline = {
            "classifier": clf,
            "label_encoder": le,
            "classifier_type": hp["classifier_type"],
            "cv_accuracy": mean_acc,
            "cv_f1_weighted": mean_f1,
            "n_classes": len(le.classes_),
            "classes": list(le.classes_),
            "n_training_examples": len(texts),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": embeddings.shape[1],
        }
        joblib.dump(pipeline, exp_dir / f"{task}.joblib")

        with open(exp_dir / "hparams.json", "w") as f:
            json.dump(hp, f, indent=2)

        return {
            "experiment_id": exp_id,
            "hparams": hp,
            "task_scores": task_scores,
            "composite_score": composite,
            "training_time_s": round(elapsed, 1),
            "model_dir": str(exp_dir),
            "status": "completed",
            "experiment_type": "sklearn",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"sklearn experiment failed: {e}")
        return {
            "experiment_id": exp_id, "hparams": hp,
            "composite_score": 0.0, "training_time_s": round(elapsed, 1),
            "status": f"error: {str(e)[:100]}", "experiment_type": "sklearn",
        }


def promote_sklearn_model(result: dict):
    """Promote winning sklearn model to production (trained_models/)."""
    import shutil

    exp_dir = Path(result.get("model_dir", ""))
    task = result.get("hparams", {}).get("task", "")

    if not exp_dir.exists() or not task:
        logger.warning(f"Cannot promote: dir={exp_dir}, task={task}")
        return

    SKLEARN_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    src = exp_dir / f"{task}.joblib"
    if src.exists():
        dest = SKLEARN_MODELS_DIR / f"{task}.joblib"
        # Backup current
        if dest.exists():
            backup = SKLEARN_MODELS_DIR / f"{task}.joblib.backup"
            shutil.copy2(dest, backup)
            logger.info(f"Backed up {dest.name}")
        shutil.copy2(src, dest)
        logger.info(f"Promoted sklearn {task} to production")

        # Try hot-reload
        try:
            from dl_models import ModelManager
            mm = ModelManager()
            mm.reload_model(task)
            logger.info(f"Hot-reloaded sklearn model: {task}")
        except Exception as e:
            logger.debug(f"Hot-reload skipped: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    hp = _sample_sklearn_hparams()
    logger.info(f"Sampled hparams: {json.dumps(hp, indent=2)}")
    result = run_sklearn_experiment(hp)
    print("\n=== SKLEARN EXPERIMENT RESULT ===")
    print(json.dumps(result, indent=2, default=str))
