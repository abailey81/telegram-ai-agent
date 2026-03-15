"""
Autoresearch Data Harvester
Converts real conversation logs into labeled training data.

Sources:
1. RL experience replay (rl_data/*_experiences.json) — richest source
2. Emotion timeline logs (engine_data/emotional/*_emotions.json)

Output: JSON files in autoresearch/data/ matching training_data.py format.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

from autoresearch.config import (
    RL_DATA_DIR, ENGINE_DATA_DIR, DATA_DIR,
    MIN_REWARD_THRESHOLD, MIN_TEXT_LENGTH,
    CONTEXT_TO_INTENT, STATE_TO_STAGE,
)

logger = logging.getLogger("autoresearch.harvest")

# === Label Mappings ===
# Map RL emotional_tone values to training data emotional_tone labels.
# Valid training labels: anger, angry, anxious, bored, confident, desire, excited,
# excitement, fear, frustration, grateful, gratitude, happy, joy, love, neutral,
# nostalgic, playful, sad, sadness, surprise, tenderness, tired
EMOTION_MAP = {
    "neutral": "neutral",
    "joy": "joy",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "anger",      # closest match
    "sadness": "sadness",    # exact match (not "sad")
    "sad": "sad",            # exact match
    "happy": "happy",        # exact match
    "love": "love",
    "excited": "excited",    # exact match
    "playful": "playful",
}

# Map context_key prefixes to romantic_intent labels.
# Valid training labels: advice_seeking, angry, apology, casual, curious, distant,
# flirty, goodbye, grateful, greeting, hurt, jealous, opinion, planning, plans,
# playful, romantic, sad, serious, sharing, sincere, small_talk, supportive,
# testing, venting
CONTEXT_PREFIX_TO_INTENT = {
    "light_neutral": "casual",
    "light_joy": "playful",
    "light_positive": "casual",       # "friendly" not valid -> casual
    "light_surprise": "curious",      # exact match
    "light_anger": "angry",
    "light_disgust": "distant",       # exact match
    "light_fear": "supportive",       # "anxious" not valid -> supportive
    "tense_neutral": "serious",       # exact match
    "tense_anger": "angry",
    "tense_surprise": "curious",      # exact match
    "deep_emotional": "sharing",      # exact match
    "deep_positive": "romantic",
    "deep_negative": "supportive",
    "flirty": "flirty",
    "conflict": "venting",            # "angry" less specific -> venting
    "reconnect": "sincere",
    "planning": "planning",           # exact match
}

# Map conversation_state to conversation_stage labels.
# Valid training labels: advising, brainstorming, closing, conflict, cooling_down,
# debating, deep, deep_conversation, flirting, flowing, makeup, opening,
# resolution, small_talk, storytelling, topic_discussion, venting, warming_up
CONV_STATE_TO_STAGE = {
    "flowing": "flowing",
    "warming_up": "warming_up",
    "cooling_down": "cooling_down",
    "conflict": "conflict",
    "deep": "deep",
}


def _load_rl_experiences() -> List[dict]:
    """Load all RL experience records from rl_data/."""
    experiences = []
    if not RL_DATA_DIR.exists():
        logger.warning(f"rl_data directory not found: {RL_DATA_DIR}")
        return experiences

    for f in RL_DATA_DIR.iterdir():
        if not f.name.endswith("_experiences.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    experiences.extend(data)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Loaded {len(experiences)} RL experiences from {RL_DATA_DIR}")
    return experiences


def _harvest_emotional_tone(experiences: List[dict]) -> List[Tuple[str, str]]:
    """Extract emotional_tone training data from RL experiences."""
    results = []
    for exp in experiences:
        reward = exp.get("reward", 0)
        if reward < MIN_REWARD_THRESHOLD:
            continue

        tone = exp.get("emotional_tone", "")
        mapped = EMOTION_MAP.get(tone)
        if not mapped:
            continue

        # Use both our message and their response as training examples
        for key in ("our_message_preview", "their_response_preview"):
            text = exp.get(key, "")
            if text and len(text) >= MIN_TEXT_LENGTH:
                results.append((text, mapped))

    logger.info(f"Harvested {len(results)} emotional_tone examples")
    return results


def _harvest_romantic_intent(experiences: List[dict]) -> List[Tuple[str, str]]:
    """Extract romantic_intent training data from RL experiences."""
    results = []
    for exp in experiences:
        reward = exp.get("reward", 0)
        if reward < MIN_REWARD_THRESHOLD:
            continue

        context_key = exp.get("context_key", "")
        # Strip the length suffix (e.g., "light_neutral_short" -> "light_neutral")
        parts = context_key.rsplit("_", 1)
        prefix = parts[0] if len(parts) > 1 and parts[1] in ("short", "medium", "long") else context_key

        mapped = CONTEXT_PREFIX_TO_INTENT.get(prefix)
        if not mapped:
            continue

        for key in ("our_message_preview", "their_response_preview"):
            text = exp.get(key, "")
            if text and len(text) >= MIN_TEXT_LENGTH:
                results.append((text, mapped))

    logger.info(f"Harvested {len(results)} romantic_intent examples")
    return results


def _harvest_conversation_stage(experiences: List[dict]) -> List[Tuple[str, str]]:
    """Extract conversation_stage training data from RL experiences."""
    results = []
    for exp in experiences:
        reward = exp.get("reward", 0)
        if reward < MIN_REWARD_THRESHOLD:
            continue

        state = exp.get("conversation_state", "")
        mapped = CONV_STATE_TO_STAGE.get(state)
        if not mapped:
            continue

        for key in ("our_message_preview", "their_response_preview"):
            text = exp.get(key, "")
            if text and len(text) >= MIN_TEXT_LENGTH:
                results.append((text, mapped))

    logger.info(f"Harvested {len(results)} conversation_stage examples")
    return results


def _harvest_from_emotion_timeline() -> List[Tuple[str, str]]:
    """Extract emotional_tone examples from emotion timeline logs.

    These logs have emotion labels but no message text directly.
    We can only use them if cross-referenced with other sources.
    For now, returns empty — placeholder for future enrichment.
    """
    # Emotion timeline has: emotion, intensity, valence, arousal, timestamp
    # But no message text attached. Would need message history correlation.
    return []


def _deduplicate(
    harvested: List[Tuple[str, str]],
    existing: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Remove harvested examples that duplicate existing training data."""
    existing_texts: Set[str] = {t.lower().strip() for t, _ in existing}
    deduped = []
    seen: Set[str] = set()
    for text, label in harvested:
        key = text.lower().strip()
        if key not in existing_texts and key not in seen:
            deduped.append((text, label))
            seen.add(key)
    removed = len(harvested) - len(deduped)
    if removed > 0:
        logger.info(f"Deduplicated: removed {removed} duplicates")
    return deduped


def harvest_all() -> Dict[str, List[Tuple[str, str]]]:
    """Run full harvesting pipeline. Returns harvested data per task."""
    experiences = _load_rl_experiences()

    results = {
        "emotional_tone": _harvest_emotional_tone(experiences),
        "romantic_intent": _harvest_romantic_intent(experiences),
        "conversation_stage": _harvest_conversation_stage(experiences),
    }

    # Add emotion timeline data
    timeline_data = _harvest_from_emotion_timeline()
    if timeline_data:
        results["emotional_tone"].extend(timeline_data)

    # Deduplicate against existing training data
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from training.training_data import get_all_data
        existing = get_all_data()
        for task in results:
            if task in existing:
                results[task] = _deduplicate(results[task], existing[task])
    except ImportError:
        logger.warning("Could not import training_data for deduplication")

    # Save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for task, data in results.items():
        output_path = DATA_DIR / f"harvested_{task}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        stats[task] = len(data)
        logger.info(f"Saved {len(data)} harvested examples to {output_path}")

    # Save stats
    stats_path = DATA_DIR / "harvest_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_experiences": len(experiences),
            "harvested_counts": stats,
            "min_reward_threshold": MIN_REWARD_THRESHOLD,
            "min_text_length": MIN_TEXT_LENGTH,
        }, f, indent=2)

    total = sum(stats.values())
    logger.info(f"Harvesting complete: {total} total examples from {len(experiences)} experiences")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    results = harvest_all()
    for task, data in results.items():
        print(f"  {task}: {len(data)} examples")
        if data:
            print(f"    Sample: {data[0]}")
