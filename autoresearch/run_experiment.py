"""
Autoresearch Experiment Loop — The autonomous improvement engine.

Now covers the ENTIRE project, not just neural networks:
  - neural:       TextCNN / EmotionAttentionNet hyperparameters
  - sklearn:      sklearn classifier hyperparameter search
  - rl_params:    RL engine reward weights, decay, exploration
  - engine_params: Conversation + emotional intelligence thresholds
  - voice:        Voice engine TTS parameter optimization

Experiments rotate through types based on weighted probability.
Each type tracks its own best score; improvements auto-promote.

Usage:
    uv run python -m autoresearch.run_experiment              # Run default 10 experiments
    uv run python -m autoresearch.run_experiment --n 5         # Run 5 experiments
    uv run python -m autoresearch.run_experiment --type neural # Only neural experiments
    uv run python -m autoresearch.run_experiment --results     # Show results table
"""

import csv
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import (
    BUDGET_SECONDS, DEFAULT_N_EXPERIMENTS, RESULTS_FILE,
    MODEL_VERSIONS_DIR, ROLLBACK_DIR, NEURAL_MODELS_DIR,
    MAX_MODEL_VERSIONS, PROGRAM_FILE,
    EXPERIMENT_TYPES, EXPERIMENT_TYPE_WEIGHTS,
)

logger = logging.getLogger("autoresearch.experiment")

RESULTS_HEADER = [
    "timestamp", "experiment_id", "experiment_type", "task", "model_type",
    "hparams_json", "val_acc", "composite_score",
    "training_time_s", "status", "notes",
]


def _ensure_results_file():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(RESULTS_HEADER)
    else:
        # Check if old format (no experiment_type column) — migrate if needed
        with open(RESULTS_FILE, "r") as f:
            first_line = f.readline().strip()
        if "experiment_type" not in first_line:
            # Old format — read all, rewrite with new header
            rows = []
            with open(RESULTS_FILE, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                old_header = next(reader)
                for row in reader:
                    # Insert "neural" as experiment_type at position 2
                    new_row = row[:2] + ["neural"] + row[2:]
                    rows.append(new_row)
            with open(RESULTS_FILE, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(RESULTS_HEADER)
                writer.writerows(rows)
            logger.info("Migrated results.tsv to new format with experiment_type column")


def _log_result(result: dict, notes: str = ""):
    """Append experiment result to results.tsv."""
    _ensure_results_file()

    task_scores = result.get("task_scores", {})
    best_acc = max(
        (s.get("val_acc", 0) for s in task_scores.values()),
        default=0,
    )

    exp_type = result.get("experiment_type", "neural")
    hp = result.get("hparams", {})

    row = [
        datetime.now().isoformat(),
        result.get("experiment_id", ""),
        exp_type,
        hp.get("task", "all"),
        hp.get("model_type", hp.get("classifier_type", exp_type)),
        json.dumps(hp, default=str),
        f"{best_acc:.6f}",
        f"{result.get('composite_score', 0):.6f}",
        f"{result.get('training_time_s', 0):.1f}",
        result.get("status", "unknown"),
        notes,
    ]

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(row)


def get_results(limit: int = 20) -> List[dict]:
    """Read recent results from results.tsv."""
    _ensure_results_file()
    if not RESULTS_FILE.exists():
        return []

    results = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(dict(row))

    return results[-limit:]


def get_best_score(experiment_type: str = None) -> float:
    """Get the best composite score, optionally filtered by experiment type."""
    results = get_results(limit=5000)
    if not results:
        return 0.0

    if experiment_type:
        results = [r for r in results if r.get("experiment_type", "neural") == experiment_type]

    if not results:
        return 0.0
    return max(float(r.get("composite_score", 0)) for r in results)


# ─── Promotion Functions ─────────────────────────────────────────────

def _promote_neural_model(result: dict):
    """Promote neural network model to production."""
    exp_dir = Path(result.get("model_dir", ""))
    if not exp_dir.exists():
        logger.warning(f"Experiment dir not found: {exp_dir}")
        return

    NEURAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)

    for f in NEURAL_MODELS_DIR.glob("*.pt"):
        shutil.copy2(f, ROLLBACK_DIR / f.name)
        logger.info(f"Backed up {f.name} to rollback/")

    for f in exp_dir.glob("*.pt"):
        shutil.copy2(f, NEURAL_MODELS_DIR / f.name)
        logger.info(f"Promoted {f.name} to production")

    for f in exp_dir.glob("*.joblib"):
        shutil.copy2(f, NEURAL_MODELS_DIR / f.name)
    for f in exp_dir.glob("*_meta.json"):
        shutil.copy2(f, NEURAL_MODELS_DIR / f.name)

    try:
        from dl_models import ModelManager
        mm = ModelManager()
        for f in exp_dir.glob("*.pt"):
            mm.reload_model(f.stem)
        logger.info("Hot-reloaded neural models in live system")
    except Exception as e:
        logger.debug(f"Hot-reload skipped: {e}")


def _promote_result(result: dict):
    """Route promotion to the correct handler based on experiment type."""
    exp_type = result.get("experiment_type", "neural")

    if exp_type == "neural":
        _promote_neural_model(result)
    elif exp_type == "sklearn":
        from autoresearch.train_sklearn import promote_sklearn_model
        promote_sklearn_model(result)
    elif exp_type == "rl_params":
        from autoresearch.optimize_rl import promote_rl_params
        promote_rl_params(result)
    elif exp_type == "engine_params":
        from autoresearch.optimize_engines import promote_engine_params
        promote_engine_params(result)
    elif exp_type == "voice":
        # Voice experiments self-promote via voice_experiment.py
        logger.info("Voice experiment promotion handled internally")


def _cleanup_failed(result: dict):
    """Clean up failed experiment artifacts."""
    model_dir = result.get("model_dir", "")
    if not model_dir:
        return
    exp_dir = Path(model_dir)
    if (
        exp_dir.exists()
        and exp_dir.is_dir()
        and str(exp_dir).startswith(str(MODEL_VERSIONS_DIR))
        and (exp_dir.name.startswith("exp_") or exp_dir.name.startswith("sklearn_"))
    ):
        shutil.rmtree(exp_dir)


def _prune_old_versions():
    """Remove oldest model versions beyond MAX_MODEL_VERSIONS."""
    prefixes = ("exp_", "sklearn_")
    versions = sorted(
        [d for d in MODEL_VERSIONS_DIR.iterdir()
         if d.is_dir() and any(d.name.startswith(p) for p in prefixes)],
        key=lambda d: d.stat().st_mtime,
    )
    if len(versions) > MAX_MODEL_VERSIONS:
        for old in versions[:len(versions) - MAX_MODEL_VERSIONS]:
            shutil.rmtree(old)
            logger.info(f"Pruned old version: {old.name}")


# ─── Experiment Runners ──────────────────────────────────────────────

def _pick_experiment_type(force_type: str = None) -> str:
    """Pick next experiment type using weighted random selection."""
    if force_type and force_type in EXPERIMENT_TYPES:
        return force_type

    types = list(EXPERIMENT_TYPE_WEIGHTS.keys())
    weights = [EXPERIMENT_TYPE_WEIGHTS[t] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def _run_single_experiment(exp_type: str, budget: int) -> dict:
    """Run a single experiment of the given type."""

    if exp_type == "neural":
        from autoresearch.train import HPARAMS, run_training
        hp = _generate_neural_hparams(dict(HPARAMS))
        result = run_training(hp, budget)
        result["experiment_type"] = "neural"
        return result

    elif exp_type == "sklearn":
        from autoresearch.train_sklearn import run_sklearn_experiment
        return run_sklearn_experiment(budget=budget)

    elif exp_type == "rl_params":
        from autoresearch.optimize_rl import run_rl_experiment
        return run_rl_experiment()

    elif exp_type == "engine_params":
        from autoresearch.optimize_engines import run_engine_experiment
        return run_engine_experiment()

    elif exp_type == "voice":
        try:
            from autoresearch.voice_experiment import run_voice_experiment
            return run_voice_experiment()
        except ImportError:
            logger.warning("Voice experiment module not available, skipping")
            return {
                "experiment_id": f"voice_{int(time.time())}",
                "composite_score": 0.0,
                "training_time_s": 0.0,
                "status": "skipped: voice module unavailable",
                "experiment_type": "voice",
                "hparams": {},
            }

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


def _generate_neural_hparams(base: dict) -> dict:
    """Generate a single neural network hparam variant."""
    hp = dict(base)

    modification = random.choice([
        "architecture", "optimizer", "data", "regularization", "schedule",
    ])

    if modification == "architecture":
        hp["model_type"] = random.choice(["textcnn", "emotion_attn"])
        if hp["model_type"] == "textcnn":
            hp["num_filters"] = random.choice([64, 128, 256, 512])
            hp["kernel_sizes"] = random.choice([
                [2, 3, 4, 5], [3, 4, 5, 6], [2, 3, 5, 7], [3, 5, 7],
            ])
        else:
            hp["num_heads"] = random.choice([2, 4, 8])
            hp["num_layers"] = random.choice([1, 2, 3, 4])
            hp["ff_dim"] = random.choice([256, 512, 1024])
    elif modification == "optimizer":
        hp["optimizer"] = random.choice(["adamw", "adam", "sgd"])
        hp["learning_rate"] = random.choice([0.0001, 0.0005, 0.001, 0.002, 0.005])
        hp["weight_decay"] = random.choice([0.001, 0.01, 0.05, 0.1])
    elif modification == "data":
        hp["use_harvested_data"] = random.choice([True, False])
        hp["harvested_weight"] = random.choice([0.3, 0.5, 0.7, 1.0])
    elif modification == "regularization":
        hp["dropout"] = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        hp["label_smoothing"] = random.choice([0.0, 0.05, 0.1, 0.15])
    elif modification == "schedule":
        hp["scheduler"] = random.choice(["cosine", "step", "plateau"])
        hp["epochs"] = random.choice([15, 20, 25, 30])

    # Don't use "all" — it trains 3 tasks sequentially, always exceeds budget.
    # Focus experiments on individual tasks for faster iteration.
    tasks = ["emotional_tone", "romantic_intent", "conversation_stage"]
    hp["task"] = random.choice(tasks)

    return hp


# ─── Main Loop ───────────────────────────────────────────────────────

def run_experiment_loop(
    n_experiments: int = DEFAULT_N_EXPERIMENTS,
    budget_seconds: int = BUDGET_SECONDS,
    base_hparams: Optional[dict] = None,
    experiment_type: str = None,
):
    """Run the autonomous experiment loop across ALL project components.

    Args:
        n_experiments: Number of experiments to run.
        budget_seconds: Time budget per experiment in seconds.
        base_hparams: Base hyperparameters for neural experiments.
        experiment_type: Force a specific experiment type (None = rotate all).
    """
    random.seed(int(time.time()))
    _ensure_results_file()

    # Track best scores PER experiment type
    best_scores = {t: get_best_score(t) for t in EXPERIMENT_TYPES}

    logger.info(f"Starting {n_experiments} experiments (budget: {budget_seconds}s each)")
    logger.info(f"Experiment types: {', '.join(EXPERIMENT_TYPES)}")
    logger.info(f"Best scores: {json.dumps({k: f'{v:.4f}' for k, v in best_scores.items()})}")

    wins = {t: 0 for t in EXPERIMENT_TYPES}
    total_by_type = {t: 0 for t in EXPERIMENT_TYPES}

    for i in range(n_experiments):
        exp_type = _pick_experiment_type(experiment_type)
        total_by_type[exp_type] = total_by_type.get(exp_type, 0) + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment {i+1}/{n_experiments} [{exp_type.upper()}]")
        logger.info(f"{'='*60}")

        result = _run_single_experiment(exp_type, budget_seconds)
        score = result.get("composite_score", 0)
        best = best_scores.get(exp_type, 0)

        is_improvement = score > best and result.get("status") == "completed"
        notes = ""

        if is_improvement:
            wins[exp_type] = wins.get(exp_type, 0) + 1
            improvement = score - best
            notes = f"NEW BEST [{exp_type}] (+{improvement:.4f})"
            logger.info(f"NEW BEST [{exp_type}]: {score:.4f} (was {best:.4f}, +{improvement:.4f})")
            best_scores[exp_type] = score
            _promote_result(result)
        else:
            notes = f"no improvement [{exp_type}] (best={best:.4f})"
            logger.info(f"[{exp_type}] Score {score:.4f} did not beat best {best:.4f}")
            _cleanup_failed(result)

        _log_result(result, notes)

    _prune_old_versions()

    total_wins = sum(wins.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT LOOP COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total improvements: {total_wins}/{n_experiments}")
    for t in EXPERIMENT_TYPES:
        if total_by_type.get(t, 0) > 0:
            logger.info(f"  {t}: {wins.get(t, 0)}/{total_by_type[t]} wins (best: {best_scores.get(t, 0):.4f})")

    return {
        "total": n_experiments,
        "wins": total_wins,
        "wins_by_type": wins,
        "best_scores": best_scores,
    }


def show_results():
    """Display results table."""
    results = get_results(limit=30)
    if not results:
        print("No results yet. Run experiments first.")
        return

    print(f"\n{'='*120}")
    print(f"{'Timestamp':<20} {'ID':<15} {'Type':<12} {'Task':<18} {'Model':<12} {'Score':<10} {'Time':<8} {'Status':<12} {'Notes'}")
    print(f"{'='*120}")
    for r in results:
        ts = r.get("timestamp", "")[:19]
        exp_id = r.get("experiment_id", "")[:14]
        exp_type = r.get("experiment_type", "neural")[:11]
        task = r.get("task", "")[:17]
        model = r.get("model_type", "")[:11]
        score = r.get("composite_score", "0")[:9]
        time_s = r.get("training_time_s", "0")[:7]
        status = r.get("status", "")[:11]
        notes = r.get("notes", "")[:30]
        print(f"{ts:<20} {exp_id:<15} {exp_type:<12} {task:<18} {model:<12} {score:<10} {time_s:<8} {status:<12} {notes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if "--results" in sys.argv:
        show_results()
    else:
        n = DEFAULT_N_EXPERIMENTS
        budget = BUDGET_SECONDS
        exp_type = None

        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--n" and i + 2 <= len(sys.argv[1:]):
                n = int(sys.argv[i + 2])
            elif arg == "--budget" and i + 2 <= len(sys.argv[1:]):
                budget = int(sys.argv[i + 2])
            elif arg == "--type" and i + 2 <= len(sys.argv[1:]):
                exp_type = sys.argv[i + 2]

        run_experiment_loop(
            n_experiments=n,
            budget_seconds=budget,
            experiment_type=exp_type,
        )
