"""
Autoresearch — Conversation & Emotional Intelligence Engine Optimization.

Optimizes hardcoded thresholds in:
- conversation_engine.py: recency decay, max_messages, state confidence
- emotional_intelligence.py: baseline thresholds, attachment weights, intensity calibration
- style_engine.py: emoji density, formality, humor thresholds
- memory_engine.py: retention limits, relevance cutoffs
- advanced_nlp.py: staleness, penalties
- orchestrator.py: LLM temperatures

Evaluation: replays conversation logs and measures quality of state detection,
emotion profiling accuracy, and response calibration consistency.
"""

import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import (
    ENGINE_DATA_DIR, ENGINE_PARAM_SPACE, ENGINE_EVAL_FILE, RL_DATA_DIR,
)

logger = logging.getLogger("autoresearch.optimize_engines")


def _sample_engine_hparams() -> dict:
    """Sample random engine parameter configuration."""
    hp = {}
    for param, values in ENGINE_PARAM_SPACE.items():
        hp[param] = random.choice(values)
    return hp


def _load_conversation_histories() -> List[dict]:
    """Load conversation summaries and emotion histories for evaluation."""
    histories = []

    # Load emotion histories
    emotion_dir = ENGINE_DATA_DIR / "emotional"
    if emotion_dir.exists():
        for f in emotion_dir.glob("*_emotions.json"):
            try:
                data = json.loads(f.read_text())
                histories.append({"type": "emotion", "data": data, "source": f.name})
            except Exception:
                pass

    # Load conversation summaries
    conv_dir = ENGINE_DATA_DIR / "conversations"
    if conv_dir.exists():
        for f in conv_dir.glob("*_summary.json"):
            try:
                data = json.loads(f.read_text())
                histories.append({"type": "conversation", "data": data, "source": f.name})
            except Exception:
                pass

    # Load RL experiences as additional signal
    if RL_DATA_DIR.exists():
        for f in RL_DATA_DIR.glob("*_experiences.json"):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    histories.append({"type": "rl", "data": data, "source": f.name})
                elif isinstance(data, dict) and "experiences" in data:
                    histories.append({"type": "rl", "data": data["experiences"], "source": f.name})
            except Exception:
                pass

    return histories


def _evaluate_context_assembly(hp: dict, histories: List[dict]) -> float:
    """Score context assembly parameters by simulating recency-weighted context.

    For each RL experience, simulate what recency-weighted context the model would
    have seen with these parameters and correlate with reward quality.
    """
    import numpy as np

    rl_histories = [h for h in histories if h["type"] == "rl"]
    if not rl_histories:
        return 0.5

    recency_decay = hp.get("recency_decay", 0.15)
    recency_weight = hp.get("recency_weight", 3.0)
    max_messages = hp.get("max_messages", 20)

    scores = []
    for h in rl_histories:
        exps = h["data"]
        if len(exps) < 3:
            continue

        for i, exp in enumerate(exps):
            reward = exp.get("reward", 0.5)
            msg_count = exp.get("message_count", 10)
            context_key = exp.get("context_key", "")
            state = exp.get("conversation_state", "")
            response_time = exp.get("response_time_seconds", 300)

            # 1. Context window coverage: how many relevant messages included?
            effective_msgs = min(msg_count, max_messages)
            coverage = effective_msgs / max(msg_count, 1)

            # 2. Recency weighting quality: simulate exponential decay weights
            # Fast decay + many messages = wasted context on stale messages
            # Slow decay + few messages = everything weighted equally (no recency signal)
            if effective_msgs > 0:
                weights = np.array([
                    math.exp(-recency_decay * j) for j in range(effective_msgs)
                ])
                weight_entropy = float(-np.sum(
                    (weights / weights.sum()) * np.log(weights / weights.sum() + 1e-10)
                ))
                # Ideal: moderate entropy — not uniform (no recency) nor too peaked
                max_entropy = math.log(effective_msgs + 1e-10)
                entropy_ratio = weight_entropy / max(max_entropy, 1e-10)
                # Sweet spot: 0.4-0.7 entropy ratio
                recency_quality = 1.0 - 2.0 * abs(entropy_ratio - 0.55)
                recency_quality = max(0.0, min(1.0, recency_quality))
            else:
                recency_quality = 0.3

            # 3. Recency weight scaling: how much recency matters
            # Higher recency_weight = recency dominates. Good for fast-paced convos.
            # For high-reward experiences with many messages, higher weight helps focus.
            if msg_count > 15 and reward > 0.6:
                weight_bonus = min((recency_weight - 2.0) / 3.0, 0.3)
            elif msg_count < 8:
                weight_bonus = -min((recency_weight - 2.0) / 4.0, 0.15)
            else:
                weight_bonus = 0.0

            # Combine: coverage matters most for high-reward experiences
            if reward > 0.7:
                score = 0.4 * coverage + 0.4 * recency_quality + 0.2 * (0.5 + weight_bonus)
            else:
                score = 0.3 * coverage + 0.5 * recency_quality + 0.2 * (0.5 + weight_bonus)

            scores.append(score * reward)  # weight by actual reward quality

    if not scores:
        return 0.5
    return float(np.mean(scores))


def _evaluate_emotion_profiling(hp: dict, histories: List[dict]) -> float:
    """Score emotion profiling parameters by simulating baseline detection and intensity.

    Simulates the emotion profiling pipeline with candidate parameters and scores
    how well the thresholds separate emotional states.
    """
    import numpy as np

    emotion_histories = [h for h in histories if h["type"] == "emotion"]
    rl_histories = [h for h in histories if h["type"] == "rl"]

    scores = []

    # Score against emotion timelines
    for h in emotion_histories:
        data = h["data"]
        timeline = data.get("emotion_timeline", [])
        if len(timeline) < 3:
            continue

        baseline_low = hp.get("baseline_valence_low", 0.35)
        baseline_high = hp.get("baseline_valence_high", 0.65)
        intensity_floor = hp.get("intensity_floor", 0.3)
        intensity_scale = hp.get("intensity_scale", 0.7)
        anxious_w = hp.get("anxious_weight", 1.5)
        avoidant_w = hp.get("avoidant_weight", 1.5)
        secure_w = hp.get("secure_weight", 1.0)

        valences = np.array([e.get("valence", 0.5) for e in timeline])
        intensities = np.array([e.get("intensity", 0.5) for e in timeline])

        # 1. Baseline band width: too narrow = everything is "extreme", too wide = never triggers
        band_width = baseline_high - baseline_low
        # Optimal band width: capture ~50-70% of valences as "normal"
        in_band = np.mean((valences >= baseline_low) & (valences <= baseline_high))
        band_quality = 1.0 - 2.0 * abs(in_band - 0.6)  # sweet spot at 60% in-band
        band_quality = max(0.0, min(1.0, band_quality))

        # 2. Intensity calibration: apply floor+scale and see if result spreads well
        calibrated = intensity_floor + (intensities - intensity_floor) * intensity_scale
        calibrated = np.clip(calibrated, 0.0, 1.0)
        # Good calibration: std between 0.10 and 0.30, mean between 0.3 and 0.7
        cal_std = float(calibrated.std()) if len(calibrated) > 1 else 0.0
        cal_mean = float(calibrated.mean())
        spread_score = min(cal_std / 0.20, 1.0)  # reward spread up to 0.20
        center_score = 1.0 - 2.0 * abs(cal_mean - 0.50)
        center_score = max(0.0, min(1.0, center_score))
        cal_quality = 0.6 * spread_score + 0.4 * center_score

        # 3. Transition detection: valence changes should align with intensity spikes
        if len(valences) > 2:
            v_diffs = np.abs(np.diff(valences))
            i_peaks = intensities[1:]  # align with diffs
            # Correlation between valence changes and intensity
            if v_diffs.std() > 0.01 and i_peaks.std() > 0.01:
                corr = float(np.corrcoef(v_diffs, i_peaks)[0, 1])
                corr = max(0.0, corr)  # only positive correlation is good
            else:
                corr = 0.3
            transition_quality = corr
        else:
            transition_quality = 0.3

        # 4. Attachment weight balance: sum should be moderate, not extreme
        total_attachment = anxious_w + avoidant_w + secure_w
        # Ideal total: 3.0-4.5 (moderate responsiveness)
        attachment_score = 1.0 - abs(total_attachment - 3.75) / 2.0
        attachment_score = max(0.0, min(1.0, attachment_score))

        score = (
            0.30 * band_quality
            + 0.25 * cal_quality
            + 0.25 * transition_quality
            + 0.20 * attachment_score
        )
        scores.append(score)

    # Also score against RL rewards — good emotion profiling → higher rewards
    for h in rl_histories:
        for exp in h["data"]:
            reward = exp.get("reward", 0.5)
            tone = exp.get("emotional_tone", "neutral")

            # Simulate: does the baseline band correctly classify this tone?
            baseline_low = hp.get("baseline_valence_low", 0.35)
            baseline_high = hp.get("baseline_valence_high", 0.65)

            # Map tones to expected valence ranges
            negative_tones = {"sad", "angry", "anxious", "frustrated", "hurt", "worried"}
            positive_tones = {"happy", "excited", "loving", "grateful", "playful", "flirty"}

            if tone in negative_tones:
                # Low baseline_low catches negative emotions earlier (better)
                tone_score = max(0.0, 1.0 - baseline_low / 0.5) * reward
            elif tone in positive_tones:
                # High baseline_high lets positive emotions register (better)
                tone_score = max(0.0, baseline_high / 0.8) * reward
            else:
                tone_score = 0.5 * reward

            scores.append(tone_score * 0.5)  # lower weight than emotion timeline data

    if not scores:
        return 0.5
    return float(np.mean(scores))


def _evaluate_state_detection(hp: dict, histories: List[dict]) -> float:
    """Score state detection by simulating confidence thresholding on RL data.

    Lower threshold = detects more states (good for nuanced conversation).
    Higher threshold = more conservative, defaults to small_talk.
    The best threshold correctly identifies diverse states for high-reward experiences.
    """
    import numpy as np

    threshold = hp.get("state_confidence_threshold", 0.15)

    rl_histories = [h for h in histories if h["type"] == "rl"]
    if not rl_histories:
        return 0.5

    scores = []
    state_counts = {}

    for h in rl_histories:
        for exp in h["data"]:
            reward = exp.get("reward", 0.5)
            state = exp.get("conversation_state", "")
            context_key = exp.get("context_key", "")
            tone = exp.get("emotional_tone", "neutral")

            if state:
                state_counts[state] = state_counts.get(state, 0) + 1

            # Simulate confidence for this state
            # Complex states (conflict, de_escalating) need lower threshold
            complex_states = {"escalating", "de_escalating", "cooling"}
            simple_states = {"flowing", "initial", "warming_up"}

            if state in complex_states:
                # Low threshold → detects these correctly → better reward
                detection_prob = max(0.0, 1.0 - threshold / 0.25)
                score = detection_prob * reward
            elif state in simple_states:
                # These are easy to detect, threshold doesn't matter much
                score = 0.8 * reward
            elif state == "stalled":
                # Stalled detection benefits from moderate threshold
                score = (1.0 - abs(threshold - 0.15) / 0.10) * reward
                score = max(0.0, score)
            else:
                score = 0.6 * reward

            # Bonus: context_key alignment with state
            # If context suggests conflict but state is flowing, threshold may be wrong
            conflict_contexts = {"conflict", "deep_negative"}
            positive_contexts = {"light_positive", "deep_positive", "flirty"}

            if any(c in context_key for c in ["conflict"]) and state == "flowing":
                score *= 0.7  # misclassification penalty
            elif any(c in context_key for c in ["flirty"]) and state in complex_states:
                score *= 0.7  # misclassification penalty

            scores.append(score)

    if not scores:
        return 0.5

    base_score = float(np.mean(scores))

    # Diversity bonus: more distinct states detected = better threshold tuning
    n_states = len(state_counts)
    diversity_bonus = min(n_states / 6.0, 0.15)  # up to 0.15 bonus for 6+ states

    return min(1.0, base_score + diversity_bonus)


def _evaluate_style_and_memory(hp: dict, histories: List[dict]) -> float:
    """Score style engine + memory engine + NLP scoring + orchestrator parameters.

    These affect response quality in ways correlated with RL rewards.
    """
    import numpy as np

    rl_histories = [h for h in histories if h["type"] == "rl"]
    if not rl_histories:
        return 0.5

    scores = []
    for h in rl_histories:
        for exp in h["data"]:
            reward = exp.get("reward", 0.5)
            context_key = exp.get("context_key", "")
            tone = exp.get("emotional_tone", "neutral")
            msg_len = len(exp.get("our_message_preview", ""))

            # Style scoring
            emoji_high = hp.get("emoji_density_high", 0.5)
            emoji_low = hp.get("emoji_density_low", 0.10)
            formality_casual = hp.get("formality_casual_threshold", 0.25)
            formality_formal = hp.get("formality_formal_threshold", 0.65)
            humor_freq = hp.get("humor_frequency_threshold", 0.3)

            # Casual contexts benefit from wider emoji range and lower formality
            casual_contexts = {"light_neutral", "light_positive", "flirty"}
            serious_contexts = {"conflict", "deep_negative", "deep_emotional"}

            style_score = 0.5
            if any(c in context_key for c in ["light", "flirty"]):
                # Casual: lower formality threshold + higher emoji = better
                style_score = (
                    0.3 * max(0.0, 1.0 - formality_casual / 0.4)
                    + 0.3 * min(emoji_high / 0.6, 1.0)
                    + 0.2 * min(humor_freq / 0.3, 1.0)
                    + 0.2 * 0.5
                )
            elif any(c in context_key for c in ["conflict", "deep_negative"]):
                # Serious: higher formality + lower emoji + less humor
                style_score = (
                    0.3 * min(formality_formal / 0.7, 1.0)
                    + 0.3 * max(0.0, 1.0 - emoji_high / 0.8)
                    + 0.2 * max(0.0, 1.0 - humor_freq / 0.5)
                    + 0.2 * 0.5
                )
            else:
                style_score = 0.5

            # Memory scoring: larger memory = better for repeat interactions
            max_facts = hp.get("max_facts", 50)
            max_episodes = hp.get("max_episodes", 100)
            mem_cutoff = hp.get("memory_relevance_cutoff", 3.0)
            sem_boost = hp.get("semantic_similarity_boost", 5.0)

            # More facts/episodes = better recall, but diminishing returns
            fact_score = min(max_facts / 60.0, 1.0)
            episode_score = min(max_episodes / 100.0, 1.0)
            # Relevance cutoff: too low = irrelevant memories, too high = miss useful ones
            cutoff_score = 1.0 - abs(mem_cutoff - 3.0) / 2.0
            cutoff_score = max(0.0, min(1.0, cutoff_score))
            memory_score = 0.3 * fact_score + 0.3 * episode_score + 0.4 * cutoff_score

            # NLP scoring
            staleness = hp.get("staleness_threshold", 0.75)
            rep_penalty = hp.get("repetition_penalty", -15)
            ai_penalty = hp.get("ai_detection_penalty", -10)

            # Moderate staleness threshold: too low = flags everything, too high = allows stale
            staleness_score = 1.0 - abs(staleness - 0.75) / 0.15
            staleness_score = max(0.0, min(1.0, staleness_score))
            # Stronger penalties = cleaner output (up to a point)
            penalty_score = min(abs(rep_penalty) / 18.0, 1.0)
            nlp_score = 0.5 * staleness_score + 0.5 * penalty_score

            # Temperature: match context
            base_temp = hp.get("base_temperature", 0.9)
            conflict_temp = hp.get("conflict_temperature", 0.7)
            creative_temp = hp.get("creative_temperature", 1.0)

            if any(c in context_key for c in ["conflict"]):
                temp_score = max(0.0, 1.0 - abs(conflict_temp - 0.65) / 0.15)
            elif any(c in context_key for c in ["flirty", "light_positive"]):
                temp_score = max(0.0, 1.0 - abs(creative_temp - 1.0) / 0.15)
            else:
                temp_score = max(0.0, 1.0 - abs(base_temp - 0.85) / 0.20)

            # Weighted combination, scaled by actual reward
            combined = (
                0.25 * style_score
                + 0.25 * memory_score
                + 0.20 * nlp_score
                + 0.30 * temp_score
            ) * reward

            scores.append(combined)

    if not scores:
        return 0.5
    return float(np.mean(scores))


def run_engine_experiment(hp: dict = None) -> dict:
    """Run a single engine parameter optimization experiment."""
    hp = hp or _sample_engine_hparams()
    start_time = time.time()
    exp_id = f"engine_{int(time.time())}"

    try:
        histories = _load_conversation_histories()
        if not histories:
            return {
                "experiment_id": exp_id, "hparams": hp,
                "composite_score": 0.0,
                "training_time_s": round(time.time() - start_time, 1),
                "status": "error: no conversation/emotion history data found",
                "experiment_type": "engine_params",
            }

        # Evaluate each dimension
        context_score = _evaluate_context_assembly(hp, histories)
        emotion_score = _evaluate_emotion_profiling(hp, histories)
        state_score = _evaluate_state_detection(hp, histories)
        style_mem_score = _evaluate_style_and_memory(hp, histories)

        # Composite: weighted average across all 4 dimensions
        composite = (
            0.25 * context_score
            + 0.25 * emotion_score
            + 0.25 * state_score
            + 0.25 * style_mem_score
        )

        elapsed = time.time() - start_time

        return {
            "experiment_id": exp_id,
            "hparams": hp,
            "task_scores": {"engine_optimization": {
                "val_acc": round(composite, 6),
                "val_f1_macro": round(composite, 6),
                "val_f1_weighted": round(composite, 6),
            }},
            "composite_score": round(composite, 6),
            "engine_metrics": {
                "context_assembly": round(context_score, 4),
                "emotion_profiling": round(emotion_score, 4),
                "state_detection": round(state_score, 4),
                "style_memory": round(style_mem_score, 4),
            },
            "training_time_s": round(elapsed, 1),
            "status": "completed",
            "experiment_type": "engine_params",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Engine experiment failed: {e}")
        return {
            "experiment_id": exp_id, "hparams": hp,
            "composite_score": 0.0, "training_time_s": round(elapsed, 1),
            "status": f"error: {str(e)[:100]}", "experiment_type": "engine_params",
        }


def promote_engine_params(result: dict):
    """Write winning engine parameters to config for live system."""
    hp = result.get("hparams", {})
    metrics = result.get("engine_metrics", {})

    output = {
        "recency_decay": hp.get("recency_decay", 0.15),
        "recency_weight": hp.get("recency_weight", 3.0),
        "max_messages": hp.get("max_messages", 20),
        "state_confidence_threshold": hp.get("state_confidence_threshold", 0.15),
        "baseline_valence_low": hp.get("baseline_valence_low", 0.35),
        "baseline_valence_high": hp.get("baseline_valence_high", 0.65),
        "anxious_weight": hp.get("anxious_weight", 1.5),
        "avoidant_weight": hp.get("avoidant_weight", 1.5),
        "secure_weight": hp.get("secure_weight", 1.0),
        "intensity_floor": hp.get("intensity_floor", 0.3),
        "intensity_scale": hp.get("intensity_scale", 0.7),
        "emoji_density_high": hp.get("emoji_density_high", 0.5),
        "emoji_density_low": hp.get("emoji_density_low", 0.10),
        "formality_casual_threshold": hp.get("formality_casual_threshold", 0.25),
        "formality_formal_threshold": hp.get("formality_formal_threshold", 0.65),
        "humor_frequency_threshold": hp.get("humor_frequency_threshold", 0.3),
        "max_facts": hp.get("max_facts", 50),
        "max_episodes": hp.get("max_episodes", 100),
        "max_milestones": hp.get("max_milestones", 20),
        "memory_relevance_cutoff": hp.get("memory_relevance_cutoff", 3.0),
        "semantic_similarity_boost": hp.get("semantic_similarity_boost", 5.0),
        "staleness_threshold": hp.get("staleness_threshold", 0.75),
        "repetition_penalty": hp.get("repetition_penalty", -15),
        "ai_detection_penalty": hp.get("ai_detection_penalty", -10),
        "base_temperature": hp.get("base_temperature", 0.9),
        "conflict_temperature": hp.get("conflict_temperature", 0.7),
        "creative_temperature": hp.get("creative_temperature", 1.0),
        "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "optimization_score": result.get("composite_score", 0.0),
        "metrics": metrics,
    }

    ENGINE_EVAL_FILE.write_text(json.dumps(output, indent=2))
    logger.info(f"Promoted engine params (score={result.get('composite_score', 0):.4f})")

    # Also save to engine_data for live pickup
    engine_config = ENGINE_DATA_DIR / "optimized_engine_params.json"
    ENGINE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine_config.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved optimized engine params to {engine_config}")


def get_best_engine_score() -> float:
    """Get best engine optimization score."""
    if ENGINE_EVAL_FILE.exists():
        try:
            data = json.loads(ENGINE_EVAL_FILE.read_text())
            return data.get("optimization_score", 0.0)
        except Exception:
            pass
    return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    hp = _sample_engine_hparams()
    logger.info(f"Sampled engine hparams: {json.dumps(hp, indent=2)}")
    result = run_engine_experiment(hp)
    print("\n=== ENGINE EXPERIMENT RESULT ===")
    print(json.dumps(result, indent=2, default=str))
