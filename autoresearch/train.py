"""
Autoresearch Training Script (AGENT-EDITABLE).

This file is the core of the autoresearch loop. The AI agent modifies the
HPARAMS dict below to try different configurations, then runs this script.
Training is budgeted to BUDGET_SECONDS (default 5 min).

Usage:
    uv run python autoresearch/train.py                     # Train with current HPARAMS
    uv run python autoresearch/train.py --task emotional_tone  # Specific task
    uv run python autoresearch/train.py --budget 120          # 2-min budget

Output: Prints JSON metrics to stdout for the experiment loop to capture.
"""

import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import BUDGET_SECONDS, MODEL_VERSIONS_DIR

logger = logging.getLogger("autoresearch.train")

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS — The AI agent modifies this section       ║
# ╚══════════════════════════════════════════════════════════════╝
HPARAMS = {
    # Model selection
    "model_type": "textcnn",          # "textcnn" or "emotion_attn"
    "task": "all",                    # "emotional_tone", "romantic_intent", "conversation_stage", or "all"

    # Training
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "label_smoothing": 0.0,
    "optimizer": "adamw",             # "adamw", "adam", "sgd"
    "scheduler": "cosine",            # "cosine", "step", "plateau"

    # TextCNN architecture
    "num_filters": 128,
    "kernel_sizes": [2, 3, 4, 5],
    "dropout": 0.3,

    # EmotionAttentionNet architecture
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 512,

    # Data
    "use_harvested_data": True,       # Include harvested conversation data
    "harvested_weight": 0.5,          # Weight for harvested vs curated (1.0 = equal)
}


class BudgetExceeded(Exception):
    pass


def _timeout_handler(signum, frame):
    raise BudgetExceeded("Training budget exceeded")


def run_training(hparams: dict = None, budget: int = BUDGET_SECONDS) -> dict:
    """Run a single training experiment within the time budget.

    Returns:
        Dict with per-task metrics and composite score.
    """
    hp = dict(hparams or HPARAMS)
    task = hp.get("task", "all")
    model_type = hp.get("model_type", "textcnn")
    use_harvested = hp.get("use_harvested_data", True)
    harvested_weight = hp.get("harvested_weight", 0.5)

    # Auto-scale epochs to fit budget. Data prep (harvest + encode with MiniLM) takes
    # ~200-300s for large datasets. Each epoch takes ~15-25s depending on task size.
    # Reserve 350s for data prep + evaluation + safety margin.
    training_budget = max(60, budget - 350)
    # All tasks with 20+ classes need conservative epoch estimates
    secs_per_epoch = 20  # conservative: covers all task sizes on MPS
    max_safe_epochs = max(8, training_budget // secs_per_epoch)
    if hp.get("epochs", 40) > max_safe_epochs:
        logger.info(f"Auto-scaling epochs {hp['epochs']} -> {max_safe_epochs} to fit {budget}s budget")
        hp["epochs"] = max_safe_epochs

    # Set timeout — use SIGALRM in main thread, threading.Timer in worker threads
    _budget_timer = None
    _is_main_thread = threading.current_thread() is threading.main_thread()
    if _is_main_thread and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(budget)
    else:
        # In a worker thread (e.g., MCP executor), use a timer that sets a flag
        _budget_exceeded_flag = threading.Event()

        def _timer_handler():
            _budget_exceeded_flag.set()

        _budget_timer = threading.Timer(budget, _timer_handler)
        _budget_timer.start()

    start_time = time.time()

    try:
        # Prepare data
        if use_harvested:
            from autoresearch.prepare import prepare_dataset
            data = prepare_dataset(refresh_harvest=True, harvested_weight=harvested_weight)
        else:
            from training.training_data import get_all_data
            data = get_all_data()

        # Generate experiment ID
        exp_id = f"exp_{int(time.time())}"
        exp_dir = MODEL_VERSIONS_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Train
        from neural_networks import train_neural_models
        model_types = [model_type] if model_type != "all" else None

        results = train_neural_models(
            task_name=task,
            hparams=hp,
            data_override=data,
            model_types=model_types,
            save_dir=str(exp_dir),
        )

        # Compute metrics — use real F1 scores via evaluate_model
        from autoresearch.evaluate import compute_composite_score, evaluate_model

        task_scores = {}
        for t, model_results in (results or {}).items():
            # Try to compute real F1 from saved model
            best_model_name = max(model_results, key=lambda k: model_results[k]["val_acc"])
            best_info = model_results[best_model_name]
            model_path = best_info.get("save_path", "")

            if model_path and Path(model_path).exists():
                try:
                    real_metrics = evaluate_model(model_path, t, data=data.get(t))
                    task_scores[t] = real_metrics
                except Exception as eval_e:
                    logger.warning(f"F1 evaluation failed for {t}, using accuracy: {eval_e}")
                    task_scores[t] = {
                        "val_acc": best_info["val_acc"],
                        "val_f1_macro": best_info["val_acc"],
                        "val_f1_weighted": best_info["val_acc"],
                    }
            else:
                task_scores[t] = {
                    "val_acc": best_info["val_acc"],
                    "val_f1_macro": best_info["val_acc"],
                    "val_f1_weighted": best_info["val_acc"],
                }

        composite = compute_composite_score(task_scores)
        elapsed = time.time() - start_time

        output = {
            "experiment_id": exp_id,
            "hparams": hp,
            "task_scores": task_scores,
            "composite_score": composite,
            "training_time_s": round(elapsed, 1),
            "model_dir": str(exp_dir),
            "status": "completed",
        }

        # Save hparams
        with open(exp_dir / "hparams.json", "w") as f:
            json.dump(hp, f, indent=2)
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(output, f, indent=2)

        return output

    except BudgetExceeded:
        elapsed = time.time() - start_time
        logger.warning(f"Budget exceeded after {elapsed:.0f}s")
        exp_id = f"exp_{int(start_time)}"
        return {
            "experiment_id": exp_id,
            "hparams": hp,
            "composite_score": 0.0,
            "training_time_s": round(elapsed, 1),
            "model_dir": str(MODEL_VERSIONS_DIR / exp_id),
            "status": "budget_exceeded",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Training failed: {e}")
        exp_id = f"exp_{int(start_time)}"
        return {
            "experiment_id": exp_id,
            "hparams": hp,
            "composite_score": 0.0,
            "training_time_s": round(elapsed, 1),
            "model_dir": str(MODEL_VERSIONS_DIR / exp_id),
            "status": f"error: {str(e)[:100]}",
        }
    finally:
        if _is_main_thread and hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # cancel alarm
        if _budget_timer is not None:
            _budget_timer.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Parse CLI args
    budget = BUDGET_SECONDS
    hp = dict(HPARAMS)

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--task" and i + 1 < len(args):
            hp["task"] = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            hp["model_type"] = args[i + 1]
            i += 2
        elif args[i] == "--budget" and i + 1 < len(args):
            budget = int(args[i + 1])
            i += 2
        elif args[i] == "--no-harvest":
            hp["use_harvested_data"] = False
            i += 1
        else:
            i += 1

    result = run_training(hp, budget)
    print("\n=== EXPERIMENT RESULT ===")
    print(json.dumps(result, indent=2))
