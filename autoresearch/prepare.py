"""
Autoresearch Data Preparation (READ-ONLY by the experiment agent).

Merges harvested conversation data with curated training data.
Produces a unified dataset ready for train.py.

Usage:
    python -m autoresearch.prepare          # Harvest + merge
    python -m autoresearch.prepare --stats  # Show data statistics
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import DATA_DIR, HARVESTED_WEIGHT
from autoresearch.harvest import harvest_all

logger = logging.getLogger("autoresearch.prepare")


def load_harvested_data() -> Dict[str, List[Tuple[str, str]]]:
    """Load previously harvested data from disk."""
    results = {}
    for task in ("emotional_tone", "romantic_intent", "conversation_stage"):
        path = DATA_DIR / f"harvested_{task}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[task] = [tuple(item) for item in json.load(f)]
        else:
            results[task] = []
    return results


def prepare_dataset(
    refresh_harvest: bool = True,
    harvested_weight: float = HARVESTED_WEIGHT,
) -> Dict[str, List[Tuple[str, str]]]:
    """Prepare the full training dataset: curated + harvested.

    Args:
        refresh_harvest: If True, re-harvest from rl_data/ before merging.
        harvested_weight: Weight multiplier for harvested examples.
            1.0 = equal weight with curated data.
            0.5 = harvested examples count as half.
            Values < 1.0 are implemented by random subsampling.

    Returns:
        Dict mapping task name to list of (text, label) tuples.
    """
    from training.training_data import get_all_data

    # Get curated data
    curated = get_all_data()

    # Harvest or load
    if refresh_harvest:
        harvested = harvest_all()
    else:
        harvested = load_harvested_data()

    # Merge
    merged = {}
    for task, curated_data in curated.items():
        task_harvested = harvested.get(task, [])

        # Apply weight by subsampling harvested data
        if harvested_weight < 1.0 and task_harvested:
            import random
            n_keep = max(1, int(len(task_harvested) * harvested_weight))
            random.seed(42)  # reproducible
            task_harvested = random.sample(task_harvested, min(n_keep, len(task_harvested)))

        combined = list(curated_data) + task_harvested
        merged[task] = combined
        logger.info(
            f"{task}: {len(curated_data)} curated + {len(task_harvested)} harvested "
            f"= {len(combined)} total"
        )

    return merged


def show_stats():
    """Show dataset statistics."""
    from training.training_data import get_all_data, get_data_stats

    print("\n=== Curated Data ===")
    stats = get_data_stats()
    for task, info in stats.items():
        if task == "total_examples_all":
            continue
        print(f"\n{task}:")
        print(f"  Total: {info['total_examples']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Min/class: {info['min_per_class']}, Max/class: {info['max_per_class']}")

    print(f"\nTotal curated: {stats['total_examples_all']}")

    print("\n=== Harvested Data ===")
    harvested = load_harvested_data()
    for task, data in harvested.items():
        labels = {}
        for _, label in data:
            labels[label] = labels.get(label, 0) + 1
        print(f"\n{task}: {len(data)} examples")
        if labels:
            print(f"  Labels: {dict(sorted(labels.items()))}")

    stats_path = DATA_DIR / "harvest_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            h_stats = json.load(f)
        print(f"\nSource: {h_stats.get('total_experiences', 0)} RL experiences")
        print(f"Reward threshold: >= {h_stats.get('min_reward_threshold', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    if "--stats" in sys.argv:
        show_stats()
    else:
        dataset = prepare_dataset()
        print("\n=== Prepared Dataset ===")
        for task, data in dataset.items():
            print(f"  {task}: {len(data)} examples")
