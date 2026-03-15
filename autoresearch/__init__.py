"""
Autoresearch — Autonomous Full-Project Improvement Framework
Adapted from Karpathy's autoresearch for the Telegram AI Agent.

Optimizes EVERYTHING, not just neural networks:
  - Neural networks (TextCNN, EmotionAttentionNet)
  - sklearn classifiers (SVM, RF, LR, GBT)
  - RL engine parameters (reward weights, decay, exploration)
  - Conversation + emotional intelligence thresholds
  - Voice engine TTS parameters

Usage:
    from autoresearch import run_experiment_loop, harvest_data, get_results

    # Harvest conversation data into training examples
    harvest_data()

    # Run N autonomous experiments across ALL components
    run_experiment_loop(n_experiments=100, budget_seconds=300)

    # Run specific experiment type only
    run_experiment_loop(n_experiments=10, experiment_type="sklearn")

    # Check results
    results = get_results(limit=20)
"""


def harvest_data():
    from autoresearch.harvest import harvest_all
    return harvest_all()


def run_experiment_loop(**kwargs):
    from autoresearch.run_experiment import run_experiment_loop as _run
    return _run(**kwargs)


def get_results(**kwargs):
    from autoresearch.run_experiment import get_results as _get
    return _get(**kwargs)


def run_sklearn_experiment(**kwargs):
    from autoresearch.train_sklearn import run_sklearn_experiment as _run
    return _run(**kwargs)


def run_rl_experiment(**kwargs):
    from autoresearch.optimize_rl import run_rl_experiment as _run
    return _run(**kwargs)


def run_engine_experiment(**kwargs):
    from autoresearch.optimize_engines import run_engine_experiment as _run
    return _run(**kwargs)


__all__ = [
    "harvest_data",
    "run_experiment_loop",
    "get_results",
    "run_sklearn_experiment",
    "run_rl_experiment",
    "run_engine_experiment",
]
