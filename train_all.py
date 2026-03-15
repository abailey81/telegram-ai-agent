"""
Master Training Script - One command to train everything.

Trains all models in the correct order:
1. sklearn classifiers (fast, reliable baselines)
2. Neural network models (TextCNN, EmotionAttentionNet)
3. Validates all saved models
4. Reports final status

Usage:
    python train_all.py              # Train everything
    python train_all.py --sklearn    # Train only sklearn classifiers
    python train_all.py --neural     # Train only neural networks
    python train_all.py --validate   # Validate saved models only
    python train_all.py --status     # Show model status
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN-ALL] %(message)s")
logger = logging.getLogger("train_all")

TRAINED_MODEL_DIR = Path(__file__).parent / "trained_models"
NEURAL_MODEL_DIR = TRAINED_MODEL_DIR / "neural"


def train_sklearn():
    """Train sklearn classifiers."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Training sklearn classifiers")
    logger.info("=" * 70)
    start = time.time()
    try:
        from training.train_classifiers import train_sklearn_classifiers
        train_sklearn_classifiers()
        elapsed = time.time() - start
        logger.info(f"sklearn training completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"sklearn training failed: {e}")
        return False


def train_neural():
    """Train neural network models."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Training neural network models")
    logger.info("=" * 70)
    start = time.time()
    try:
        from neural_networks import train_neural_models
        train_neural_models(task_name="all", epochs=50, batch_size=32, learning_rate=0.001)
        elapsed = time.time() - start
        logger.info(f"Neural network training completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"Neural network training failed: {e}")
        return False


def validate_models():
    """Validate all saved models by running test predictions."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Validating saved models")
    logger.info("=" * 70)

    test_texts = [
        "I love you so much, you mean the world to me",
        "Hey what's up?",
        "I'm really angry right now, you never listen",
        "Can we meet for dinner tonight?",
        "You're the best thing that ever happened to me",
        "I'm feeling sad and lonely today",
        "Haha that's hilarious!",
        "We need to talk about something serious",
    ]

    results = {"sklearn": {}, "neural": {}, "transformers": {}}
    all_passed = True

    # Validate sklearn models
    sklearn_models = list(TRAINED_MODEL_DIR.glob("*.joblib"))
    if sklearn_models:
        logger.info(f"\n  Found {len(sklearn_models)} sklearn models")
        try:
            from dl_models import get_model_manager
            mm = get_model_manager()

            for model_path in sklearn_models:
                name = model_path.stem
                prediction = mm.predict_with_custom(name, test_texts[0])
                if prediction:
                    results["sklearn"][name] = {
                        "status": "OK",
                        "test_prediction": prediction["label"],
                        "confidence": prediction["confidence"],
                    }
                    logger.info(f"    {name}: OK (predicted '{prediction['label']}' "
                                f"with {prediction['confidence']:.0%} confidence)")
                else:
                    results["sklearn"][name] = {"status": "FAILED"}
                    logger.error(f"    {name}: FAILED")
                    all_passed = False
        except Exception as e:
            logger.error(f"  sklearn validation error: {e}")
            all_passed = False
    else:
        logger.warning("  No sklearn models found")

    # Validate neural models
    neural_models = list(NEURAL_MODEL_DIR.glob("*.pt")) if NEURAL_MODEL_DIR.exists() else []
    if neural_models:
        logger.info(f"\n  Found {len(neural_models)} neural models")
        try:
            from neural_networks import predict_with_neural
            from dl_models import get_model_manager
            mm = get_model_manager()

            known_model_types = ["textcnn", "emotion_attn"]
            for model_path in neural_models:
                name = model_path.stem
                model_type = None
                task = None
                for mt in sorted(known_model_types, key=len, reverse=True):
                    if name.endswith(f"_{mt}"):
                        model_type = mt
                        task = name[: -(len(mt) + 1)]
                        break
                if not model_type or not task:
                    continue

                embedding = mm.embed_single(test_texts[0])
                if embedding is not None:
                    prediction = predict_with_neural(task, embedding, model_type)
                    if prediction:
                        results["neural"][name] = {
                            "status": "OK",
                            "test_prediction": prediction["label"],
                            "confidence": prediction["confidence"],
                        }
                        logger.info(f"    {name}: OK (predicted '{prediction['label']}' "
                                    f"with {prediction['confidence']:.0%} confidence)")
                    else:
                        results["neural"][name] = {"status": "FAILED"}
                        logger.error(f"    {name}: FAILED")
                        all_passed = False
        except Exception as e:
            logger.error(f"  Neural validation error: {e}")
            all_passed = False
    else:
        logger.warning("  No neural models found")

    # Validate transformer pipelines
    logger.info("\n  Validating transformer pipelines...")
    try:
        from advanced_nlp import dl_sentiment, dl_emotions, dl_classify_intent

        sent = dl_sentiment(test_texts[0])
        if sent:
            results["transformers"]["sentiment"] = {"status": "OK", "result": sent["sentiment"]}
            logger.info(f"    Sentiment: OK ({sent['sentiment']}, {sent['confidence']:.0%})")
        else:
            results["transformers"]["sentiment"] = {"status": "UNAVAILABLE"}
            logger.warning("    Sentiment: UNAVAILABLE (install transformers + torch)")

        emo = dl_emotions(test_texts[0])
        if emo:
            results["transformers"]["emotions"] = {"status": "OK", "result": emo["primary_emotion"]}
            logger.info(f"    Emotions: OK ({emo['primary_emotion']}, {emo['emotional_intensity']:.0%})")
        else:
            results["transformers"]["emotions"] = {"status": "UNAVAILABLE"}
            logger.warning("    Emotions: UNAVAILABLE")

        intent = dl_classify_intent(test_texts[0])
        if intent:
            results["transformers"]["intent"] = {"status": "OK", "result": intent["primary_intent"]}
            logger.info(f"    Intent: OK ({intent['primary_intent']}, {intent['confidence']:.0%})")
        else:
            results["transformers"]["intent"] = {"status": "UNAVAILABLE"}
            logger.warning("    Intent: UNAVAILABLE")

    except Exception as e:
        logger.error(f"  Transformer validation error: {e}")

    # Summary
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("VALIDATION: ALL MODELS PASSED")
    else:
        logger.warning("VALIDATION: SOME MODELS FAILED (see above)")
    logger.info("=" * 70)

    return results


def show_status():
    """Show status of all models and dependencies."""
    logger.info("=" * 70)
    logger.info("MODEL STATUS")
    logger.info("=" * 70)

    # Check dependencies
    deps = {}
    for lib in ["torch", "transformers", "sentence_transformers", "sklearn", "numpy", "joblib"]:
        try:
            mod = __import__(lib)
            version = getattr(mod, "__version__", "unknown")
            deps[lib] = version
            logger.info(f"  {lib}: {version}")
        except ImportError:
            deps[lib] = "NOT INSTALLED"
            logger.warning(f"  {lib}: NOT INSTALLED")

    # Check device
    try:
        import torch
        if torch.cuda.is_available():
            device = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        else:
            device = "CPU"
        logger.info(f"\n  Compute device: {device}")
    except ImportError:
        logger.info("\n  Compute device: CPU (torch not installed)")

    # Count models
    sklearn_count = len(list(TRAINED_MODEL_DIR.glob("*.joblib"))) if TRAINED_MODEL_DIR.exists() else 0
    neural_count = len(list(NEURAL_MODEL_DIR.glob("*.pt"))) if NEURAL_MODEL_DIR.exists() else 0

    logger.info(f"\n  sklearn models: {sklearn_count}")
    logger.info(f"  Neural models: {neural_count}")

    # Show metadata for each model
    for meta_path in sorted(TRAINED_MODEL_DIR.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        logger.info(f"\n  Model: {meta.get('task', meta_path.stem)}")
        logger.info(f"    Best classifier: {meta.get('best_classifier', 'N/A')}")
        logger.info(f"    CV accuracy: {meta.get('cv_accuracy', 0):.4f}")
        logger.info(f"    Classes: {meta.get('classes', [])}")
        logger.info(f"    Training examples: {meta.get('n_training_examples', 0)}")

    if NEURAL_MODEL_DIR.exists():
        for meta_path in sorted(NEURAL_MODEL_DIR.glob("*_meta.json")):
            meta = json.loads(meta_path.read_text())
            logger.info(f"\n  Neural Model: {meta.get('task', meta_path.stem)}")
            for mname, mdata in meta.get("models", {}).items():
                logger.info(f"    {mname}: val_accuracy={mdata.get('val_accuracy', 0):.4f}")

    return deps


def main():
    parser = argparse.ArgumentParser(
        description="Master training script for all ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py              Train everything (sklearn + neural)
  python train_all.py --sklearn    Train only sklearn classifiers
  python train_all.py --neural     Train only neural networks
  python train_all.py --validate   Validate all saved models
  python train_all.py --status     Show dependency and model status
        """,
    )
    parser.add_argument("--sklearn", action="store_true", help="Train sklearn classifiers only")
    parser.add_argument("--neural", action="store_true", help="Train neural networks only")
    parser.add_argument("--validate", action="store_true", help="Validate saved models")
    parser.add_argument("--status", action="store_true", help="Show model status")
    parser.add_argument("--autoresearch", action="store_true",
                        help="Run autoresearch experiment loop (autonomous full-project improvement)")
    parser.add_argument("--harvest", action="store_true",
                        help="Harvest conversation data for training (no training)")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of autoresearch experiments (default: 10)")
    parser.add_argument("--budget", type=int, default=300,
                        help="Training budget per experiment in seconds (default: 300)")
    parser.add_argument("--type", type=str, default=None,
                        help="Force experiment type: neural, sklearn, rl_params, engine_params, voice")
    args = parser.parse_args()

    total_start = time.time()

    if args.status:
        show_status()
        # Also show autoresearch results if available
        try:
            from autoresearch.run_experiment import show_results
            show_results()
        except Exception:
            pass
        return

    if args.harvest:
        logger.info("=" * 70)
        logger.info("HARVESTING CONVERSATION DATA")
        logger.info("=" * 70)
        try:
            from autoresearch.harvest import harvest_all
            results = harvest_all()
            for task, data in results.items():
                logger.info(f"  {task}: {len(data)} examples harvested")
        except Exception as e:
            logger.error(f"Harvesting failed: {e}")
        return

    if args.autoresearch:
        type_desc = f" [{args.type}]" if args.type else " [all types]"
        logger.info("=" * 70)
        logger.info(f"AUTORESEARCH{type_desc}: Running {args.n} experiments ({args.budget}s budget each)")
        logger.info("=" * 70)
        try:
            from autoresearch.run_experiment import run_experiment_loop
            result = run_experiment_loop(
                n_experiments=args.n,
                budget_seconds=args.budget,
                experiment_type=args.type,
            )
            total_wins = result.get("wins", 0)
            logger.info(f"Autoresearch complete: {total_wins}/{result['total']} improvements")
            for t, score in result.get("best_scores", {}).items():
                wins = result.get("wins_by_type", {}).get(t, 0)
                if score > 0:
                    logger.info(f"  {t}: best={score:.4f} ({wins} wins)")
        except Exception as e:
            logger.error(f"Autoresearch failed: {e}")
        return

    if args.validate:
        validate_models()
        return

    if args.sklearn:
        train_sklearn()
        validate_models()
        return

    if args.neural:
        train_neural()
        validate_models()
        return

    # Train everything
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 70)

    show_status()

    sklearn_ok = train_sklearn()
    neural_ok = train_neural()

    validate_models()

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal training time: {total_elapsed:.1f}s")
    logger.info(f"sklearn: {'OK' if sklearn_ok else 'FAILED'}")
    logger.info(f"Neural: {'OK' if neural_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
