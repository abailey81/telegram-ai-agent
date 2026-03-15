"""
Reinforcement Learning Engine for Adaptive Response Strategy Selection.

Implements Thompson Sampling contextual bandits with implicit reward shaping
to learn optimal response strategies from conversation outcomes.

Architecture:
1. Contextual Multi-Armed Bandit (Thompson Sampling)
   - 8 response strategies as arms
   - Conversation state as context (emotion, stage, style, history)
   - Beta distribution priors updated from implicit rewards

2. Implicit Reward Shaping
   - No explicit user feedback needed
   - Rewards derived from: response time, message length, emotional valence,
     emoji reactions, engagement signals, conversation continuation

3. Experience Replay Buffer
   - Stores (state, action, reward, next_state) tuples
   - Enables offline analysis and batch policy updates

4. Strategy-Aware Prompt Injection
   - Selected strategy is injected into the system prompt
   - Strategy guides tone, approach, and response structure

Based on:
- Thompson Sampling for contextual bandits (Agrawal & Goyal, 2013)
- Reward shaping for dialogue systems (Li et al., 2016)
- Implicit feedback signals from messaging (Hancock et al., 2019)
"""

import json
import logging
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

rl_logger = logging.getLogger("rl_engine")
rl_logger.setLevel(logging.INFO)

# Persistent storage
RL_DATA_DIR = Path("rl_data")
RL_DATA_DIR.mkdir(exist_ok=True)

# Auto-pickup: load optimized params from autoresearch (if available)
_OPTIMIZED_RL_PARAMS = None
_OPTIMIZED_RL_PARAMS_MTIME = 0


def _load_optimized_rl_params() -> Optional[dict]:
    """Load optimized RL params from autoresearch (auto-pickup on change)."""
    global _OPTIMIZED_RL_PARAMS, _OPTIMIZED_RL_PARAMS_MTIME
    params_file = RL_DATA_DIR / "optimized_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_RL_PARAMS_MTIME:
            _OPTIMIZED_RL_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_RL_PARAMS_MTIME = mtime
            rl_logger.info(f"Auto-loaded optimized RL params (score={_OPTIMIZED_RL_PARAMS.get('optimization_score', '?')})")
        return _OPTIMIZED_RL_PARAMS
    except Exception as e:
        rl_logger.debug(f"Could not load optimized RL params: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  1. RESPONSE STRATEGIES (ARMS)
# ═══════════════════════════════════════════════════════════════

RESPONSE_STRATEGIES = {
    "mirror_energy": {
        "description": "Match their emotional intensity and style exactly",
        "prompt_directive": (
            "Mirror their energy level precisely. If they're excited, be excited. "
            "If they're aggressive, be aggressive back. If they're chill, be chill. "
            "Match their message length, emoji usage, and capitalization patterns. "
            "If they come at you hard, come back harder."
        ),
        "best_for": ["casual", "greeting", "small_talk"],
    },
    "emotional_support": {
        "description": "Validate feelings without being therapist-like",
        "prompt_directive": (
            "Be real — not a therapist. "
            "React like a human: 'damn thats rough', 'wtf happened', 'that sucks'. "
            "React first, ask second. "
            "Don't try to fix it, don't be soft about it — just be real."
        ),
        "best_for": ["emotional_sharing", "venting", "supporting", "sad"],
    },
    "playful_deflection": {
        "description": "Lighten the mood with humor and teasing",
        "prompt_directive": (
            "Lighten the mood. Use humor, light teasing, or change the subject to "
            "something fun. Don't force it if they're genuinely upset — but if the "
            "vibe allows it, be playful and spontaneous. Drop a funny observation or callback."
        ),
        "best_for": ["playful", "flirting", "brainstorming"],
    },
    "direct_engagement": {
        "description": "Answer directly and ask a follow-up question",
        "prompt_directive": (
            "Be direct and engaged. Answer their question or respond to their point "
            "clearly, then ask a follow-up that shows genuine interest. "
            "Keep it conversational, not interview-like. One question max."
        ),
        "best_for": ["curious", "topic_discussion", "debating", "advising"],
    },
    "vulnerable_sharing": {
        "description": "Open up personally to deepen connection",
        "prompt_directive": (
            "Share something personal and genuine that relates to what they said. "
            "Be a bit vulnerable — mention a real feeling, memory, or experience. "
            "This builds intimacy. Don't overdo it — one personal detail is enough."
        ),
        "best_for": ["deepening", "reconnecting", "emotional_sharing"],
    },
    "challenge_tease": {
        "description": "Push back playfully, create tension",
        "prompt_directive": (
            "Push back a little. Tease them, challenge their opinion, or create "
            "playful tension. Be confident but not mean. This creates chemistry. "
            "Example: 'wait you actually think that?? nah' or 'hmm idk about that one chief'"
        ),
        "best_for": ["flirting", "debating", "testing", "playful"],
    },
    "hold_ground": {
        "description": "Stand firm when things are heated",
        "prompt_directive": (
            "Stand your ground. If they're coming at you, fire back. "
            "Don't fold, don't apologize if you didn't do anything wrong. "
            "Be direct and blunt: 'nah thats not what happened' energy. "
            "Short, sharp responses. Match their intensity."
        ),
        "best_for": ["conflict", "de_escalating", "venting"],
    },
    "enthusiastic_amplification": {
        "description": "Amp up their positive energy",
        "prompt_directive": (
            "Amplify their excitement. If they're happy about something, be even happier. "
            "If they're sharing good news, react BIG first: 'YOOO no way!!', "
            "'wait fr?? thats insane', 'LETS GOOO'. Then ask for details. "
            "Ride the wave of their positive energy."
        ),
        "best_for": ["celebrating", "greeting", "storytelling"],
    },
}

STRATEGY_NAMES = list(RESPONSE_STRATEGIES.keys())
NUM_STRATEGIES = len(STRATEGY_NAMES)


# ═══════════════════════════════════════════════════════════════
#  2. CONTEXT FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

# Map conversation states to context buckets for generalization
STATE_BUCKETS = {
    "greeting": "light",
    "small_talk": "light",
    "checking_in": "light",
    "closing": "light",
    "deepening": "intimate",
    "emotional_sharing": "intimate",
    "supporting": "intimate",
    "reconnecting": "intimate",
    "flirting": "playful",
    "celebrating": "playful",
    "brainstorming": "playful",
    "storytelling": "playful",
    "planning": "practical",
    "topic_discussion": "practical",
    "debating": "practical",
    "advising": "practical",
    "conflict": "tense",
    "de_escalating": "tense",
    "venting": "tense",
}

EMOTION_VALENCE = {
    "joy": 1.0, "love": 1.0, "excitement": 0.9, "gratitude": 0.8,
    "tenderness": 0.7, "desire": 0.6, "playful": 0.5, "surprise": 0.3,
    "neutral": 0.0, "fear": -0.3, "sadness": -0.5, "frustration": -0.6,
    "anger": -0.8,
}


def extract_context_features(
    conversation_state: str,
    emotional_tone: str,
    sentiment_score: float,
    message_length: int,
    time_of_day: int,
    relationship_health: float = 0.5,
    recent_rewards: List[float] = None,
) -> Dict[str, Any]:
    """Extract context features for the bandit.

    Returns a context dict that can be used to select
    the appropriate strategy for this situation.
    """
    state_bucket = STATE_BUCKETS.get(conversation_state, "light")
    emotion_valence = EMOTION_VALENCE.get(emotional_tone, 0.0)

    # Time buckets: night(0-6), morning(6-12), afternoon(12-18), evening(18-24)
    if time_of_day < 6:
        time_bucket = "night"
    elif time_of_day < 12:
        time_bucket = "morning"
    elif time_of_day < 18:
        time_bucket = "afternoon"
    else:
        time_bucket = "evening"

    # Message length bucket
    if message_length < 20:
        length_bucket = "short"
    elif message_length < 80:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    # Recent reward trend
    reward_trend = 0.0
    if recent_rewards and len(recent_rewards) >= 2:
        recent = recent_rewards[-3:]
        reward_trend = sum(recent) / len(recent)

    return {
        "conversation_state": conversation_state,
        "state_bucket": state_bucket,
        "emotional_tone": emotional_tone,
        "emotion_valence": emotion_valence,
        "sentiment_score": sentiment_score,
        "time_bucket": time_bucket,
        "length_bucket": length_bucket,
        "relationship_health": relationship_health,
        "reward_trend": reward_trend,
        # Composite context key for bandit lookup
        "context_key": f"{state_bucket}_{emotional_tone}_{length_bucket}",
    }


# ═══════════════════════════════════════════════════════════════
#  3. THOMPSON SAMPLING CONTEXTUAL BANDIT
# ═══════════════════════════════════════════════════════════════

class ThompsonSamplingBandit:
    """Contextual multi-armed bandit using Thompson Sampling.

    Maintains separate Beta(alpha, beta) distributions for each
    (context, strategy) pair. Uses Thompson Sampling to balance
    exploration and exploitation.

    Alpha = successes + 1 (prior)
    Beta = failures + 1 (prior)

    Higher alpha → strategy worked well in this context
    Higher beta → strategy failed in this context
    """

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.params_file = RL_DATA_DIR / f"{chat_id}_bandit.json"

        # params[context_key][strategy_name] = {"alpha": float, "beta": float}
        self.params: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._load()

    def _load(self):
        """Load learned parameters from disk."""
        if self.params_file.exists():
            try:
                data = json.loads(self.params_file.read_text())
                self.params = data.get("params", {})
            except (json.JSONDecodeError, KeyError):
                self.params = {}

    def _save(self):
        """Persist parameters to disk."""
        data = {
            "chat_id": self.chat_id,
            "params": self.params,
            "last_updated": datetime.now().isoformat(),
            "total_updates": sum(
                p["alpha"] + p["beta"] - 2  # subtract priors
                for ctx in self.params.values()
                for p in ctx.values()
            ),
        }
        self.params_file.write_text(json.dumps(data, indent=2))

    def _get_params(self, context_key: str, strategy: str) -> Dict[str, float]:
        """Get Beta distribution parameters, initializing with informed priors."""
        if context_key not in self.params:
            self.params[context_key] = {}

        if strategy not in self.params[context_key]:
            # Informed prior: strategies that match context start with slight advantage
            strategy_info = RESPONSE_STRATEGIES.get(strategy, {})
            best_for = strategy_info.get("best_for", [])

            # Check if this context matches the strategy's best-for states
            state_bucket = context_key.split("_")[0] if "_" in context_key else context_key
            bucket_states = [
                s for s, b in STATE_BUCKETS.items() if b == state_bucket
            ]
            # match_bonus is auto-tuned by autoresearch
            _opt = _load_optimized_rl_params()
            _mb = (_opt or {}).get("match_bonus", 1.0)
            match_bonus = 0.0
            for bf in best_for:
                if bf in bucket_states or bf == state_bucket:
                    match_bonus = _mb
                    break

            self.params[context_key][strategy] = {
                "alpha": 1.0 + match_bonus,  # Prior successes
                "beta": 1.0,                  # Prior failures
            }

        return self.params[context_key][strategy]

    def select_strategy(
        self,
        context: Dict[str, Any],
        temperature: float = 1.0,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Select a strategy using Thompson Sampling.

        Args:
            context: Context features from extract_context_features()
            temperature: Controls exploration (>1 = more exploration)

        Returns:
            (strategy_name, sampled_value, all_samples)
        """
        context_key = context["context_key"]
        samples = {}

        for strategy in STRATEGY_NAMES:
            params = self._get_params(context_key, strategy)
            # Higher temperature = more exploration (wider Beta distribution)
            # Divide by temperature to flatten the distribution
            alpha = params["alpha"] / max(temperature, 0.01)
            beta_val = params["beta"] / max(temperature, 0.01)

            # Thompson Sampling: sample from Beta distribution
            try:
                sample = random.betavariate(max(alpha, 0.01), max(beta_val, 0.01))
            except ValueError:
                sample = 0.5

            # Context-aware bonus: boost strategies that match emotional valence
            valence = context.get("emotion_valence", 0.0)
            strategy_info = RESPONSE_STRATEGIES[strategy]

            if valence < -0.3:  # Negative emotions
                if strategy in ("emotional_support", "hold_ground", "vulnerable_sharing"):
                    sample *= 1.15
            elif valence > 0.5:  # Positive emotions
                if strategy in ("enthusiastic_amplification", "playful_deflection", "challenge_tease"):
                    sample *= 1.15

            samples[strategy] = round(sample, 4)

        # Select highest-sampled strategy
        best_strategy = max(samples, key=samples.get)
        return best_strategy, samples[best_strategy], samples

    def update(
        self,
        context_key: str,
        strategy: str,
        reward: float,
    ):
        """Update Beta distribution parameters based on observed reward.

        Reward should be in [0, 1] range.
        We use fractional updates to allow continuous reward values.
        """
        params = self._get_params(context_key, strategy)

        # Fractional update: reward of 0.7 adds 0.7 to alpha and 0.3 to beta
        reward = max(0.0, min(1.0, reward))
        params["alpha"] += reward
        params["beta"] += (1.0 - reward)

        # Decay old observations to prevent stale policies (exponential forgetting)
        # Decay rate and trigger are auto-tuned by autoresearch
        optimized = _load_optimized_rl_params()
        _decay_trigger = (optimized or {}).get("decay_trigger", 50)
        total = params["alpha"] + params["beta"]
        last_decay_total = params.get("_last_decay_total", _decay_trigger)
        if total >= last_decay_total + _decay_trigger:
            decay = (optimized or {}).get("decay_rate", 0.95)
            params["alpha"] = max(1.0, params["alpha"] * decay)
            params["beta"] = max(1.0, params["beta"] * decay)
            params["_last_decay_total"] = params["alpha"] + params["beta"]

        self.params[context_key][strategy] = params
        self._save()


# ═══════════════════════════════════════════════════════════════
#  4. IMPLICIT REWARD CALCULATOR
# ═══════════════════════════════════════════════════════════════

# Reward signal weights (defaults — autoresearch may override via optimized_params.json)
_DEFAULT_REWARD_WEIGHTS = {
    "response_received": 0.25,      # They replied at all
    "response_speed": 0.10,         # How fast they replied
    "length_maintenance": 0.15,     # Message length ratio
    "emotional_valence": 0.20,      # Emotional improvement
    "engagement_signals": 0.15,     # Questions, enthusiasm
    "emoji_sentiment": 0.10,        # Positive emoji usage
    "conversation_continuation": 0.05,  # Multi-turn continuation
}


def _get_reward_weights() -> dict:
    """Get reward weights — uses autoresearch-optimized values if available."""
    optimized = _load_optimized_rl_params()
    if optimized and "reward_weights" in optimized:
        return optimized["reward_weights"]
    return _DEFAULT_REWARD_WEIGHTS


# Keep REWARD_WEIGHTS as module-level for backward compat
REWARD_WEIGHTS = _DEFAULT_REWARD_WEIGHTS


def calculate_implicit_reward(
    our_message: str,
    their_response: Optional[str],
    response_delay_seconds: Optional[float],
    our_emotion: str,
    their_emotion: str,
    their_message_before: Optional[str] = None,
    conversation_continued: bool = False,
) -> Dict[str, Any]:
    """Calculate reward from implicit signals — no explicit feedback needed.

    Returns detailed reward breakdown and final composite score [0, 1].
    """
    signals = {}

    # 1. Response received (most important signal)
    if their_response is None:
        # No response = strong negative signal
        return {
            "total_reward": 0.1,
            "signals": {"response_received": 0.0},
            "explanation": "No response received",
        }
    signals["response_received"] = 1.0

    their_lower = their_response.lower().strip()
    their_len = len(their_response)
    our_len = len(our_message)

    # 2. Response speed (faster = more engaged)
    if response_delay_seconds is not None:
        if response_delay_seconds < 60:
            signals["response_speed"] = 1.0
        elif response_delay_seconds < 300:
            signals["response_speed"] = 0.7
        elif response_delay_seconds < 900:
            signals["response_speed"] = 0.4
        elif response_delay_seconds < 3600:
            signals["response_speed"] = 0.2
        else:
            signals["response_speed"] = 0.1
    else:
        signals["response_speed"] = 0.5  # Unknown

    # 3. Message length maintenance
    if our_len > 0:
        ratio = their_len / max(our_len, 1)
        if ratio > 1.5:
            signals["length_maintenance"] = 1.0  # They wrote MORE
        elif ratio > 0.8:
            signals["length_maintenance"] = 0.8  # Similar length
        elif ratio > 0.3:
            signals["length_maintenance"] = 0.5  # Shorter but engaged
        else:
            signals["length_maintenance"] = 0.2  # Very short (disengaged)
    else:
        signals["length_maintenance"] = 0.5

    # Penalize extremely short responses
    if their_len <= 2 and their_lower in ("k", "ok", ".", "..",
                                            "ок", "ладно", "пофиг", "мне всё равно",
                                            "хз", "норм", "забей", "да", "ну"):
        signals["length_maintenance"] = 0.1

    # 4. Emotional valence change
    our_valence = EMOTION_VALENCE.get(our_emotion, 0.0)
    their_valence = EMOTION_VALENCE.get(their_emotion, 0.0)
    valence_delta = their_valence - our_valence

    if valence_delta > 0.3:
        signals["emotional_valence"] = 1.0  # They got happier
    elif valence_delta > 0:
        signals["emotional_valence"] = 0.7
    elif valence_delta > -0.2:
        signals["emotional_valence"] = 0.5  # Stable
    elif valence_delta > -0.5:
        signals["emotional_valence"] = 0.3
    else:
        signals["emotional_valence"] = 0.1  # They got much sadder/angrier

    # 5. Engagement signals
    engagement_score = 0.5  # baseline

    # Questions show curiosity
    if "?" in their_response:
        engagement_score += 0.2

    # Enthusiasm markers
    enthusiasm_markers = ["!!", "HAHA", "LOL", "OMG", "YOOO", "YES",
                          "hahaha", "lmaooo", "nooo", "wait", "tell me",
                          "seriously??", "fr??", "no way",
                          "ахахах", "хахаха", "ого", "вау", "серьёзно",
                          "ну серьёзно", "погоди", "расскажи", "ты шутишь",
                          "не может быть", "ааа", "офигеть"]
    if any(m.lower() in their_lower for m in enthusiasm_markers):
        engagement_score += 0.2

    # Personal sharing (they opened up)
    personal_markers = ["i feel", "im feeling", "i think", "honestly",
                        "tbh", "ngl", "real talk", "between us",
                        "мне кажется", "я чувствую", "я думаю", "честно говоря",
                        "по правде", "между нами", "если честно", "я считаю"]
    if any(m in their_lower for m in personal_markers):
        engagement_score += 0.15

    # Affection markers
    affection = ["❤", "🥰", "😍", "💕", "😘", "love", "miss u",
                  "miss you", "babe", "baby",
                  "люблю", "скучаю", "малыш", "зайка", "солнышко",
                  "котик", "родной", "родная"]
    if any(a in their_lower for a in affection):
        engagement_score += 0.15

    signals["engagement_signals"] = min(1.0, engagement_score)

    # 6. Emoji sentiment
    positive_emoji = {"❤", "🥰", "😍", "💕", "😘", "😊", "🥺", "😂", "🤣",
                      "😭", "💖", "💗", "✨", "🫶", "💀", "🔥", "❤️"}
    negative_emoji = {"😑", "😐", "🙄", "😒", "👎", "💔"}

    pos_count = sum(1 for e in positive_emoji if e in their_response)
    neg_count = sum(1 for e in negative_emoji if e in their_response)

    if pos_count > neg_count:
        signals["emoji_sentiment"] = min(1.0, 0.6 + pos_count * 0.1)
    elif neg_count > pos_count:
        signals["emoji_sentiment"] = max(0.1, 0.4 - neg_count * 0.1)
    else:
        signals["emoji_sentiment"] = 0.5

    # 7. Conversation continuation
    signals["conversation_continuation"] = 1.0 if conversation_continued else 0.3

    # Negative signal overrides
    # If they're clearly disengaged, cap the reward
    disengagement = ["bye", "whatever", "fine", "leave me alone",
                     "stop", "dont text me", "go away",
                     "пока", "ладно", "оставь меня", "хватит",
                     "не пиши мне", "уходи", "отстань", "отвали"]
    if any(d in their_lower for d in disengagement):
        for key in signals:
            signals[key] = min(signals[key], 0.3)

    # Calculate weighted composite (auto-pickup optimized weights)
    total = 0.0
    active_weights = _get_reward_weights()
    for signal_name, weight in active_weights.items():
        total += signals.get(signal_name, 0.5) * weight

    total = max(0.0, min(1.0, total))

    return {
        "total_reward": round(total, 4),
        "signals": {k: round(v, 4) for k, v in signals.items()},
        "explanation": _explain_reward(signals, total),
    }


def _explain_reward(signals: Dict[str, float], total: float) -> str:
    """Generate a human-readable reward explanation."""
    best = max(signals, key=signals.get)
    worst = min(signals, key=signals.get)

    if total > 0.7:
        return f"Strong positive response (best signal: {best}={signals[best]:.0%})"
    elif total > 0.5:
        return f"Moderate engagement (weakest: {worst}={signals[worst]:.0%})"
    elif total > 0.3:
        return f"Low engagement (weakest: {worst}={signals[worst]:.0%})"
    else:
        return f"Poor response (worst signal: {worst}={signals[worst]:.0%})"


# ═══════════════════════════════════════════════════════════════
#  5. EXPERIENCE REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """Stores (state, action, reward, next_state) tuples for learning.

    Persisted to disk per chat. Supports:
    - Recording experiences
    - Batch retrieval for offline analysis
    - Statistics and insights
    """

    MAX_BUFFER_SIZE = 500

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.buffer_file = RL_DATA_DIR / f"{chat_id}_experiences.json"
        self.experiences: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.buffer_file.exists():
            try:
                self.experiences = json.loads(self.buffer_file.read_text())
            except (json.JSONDecodeError, TypeError):
                self.experiences = []

    def _save(self):
        self.buffer_file.write_text(json.dumps(
            self.experiences[-self.MAX_BUFFER_SIZE:],
            indent=2, default=str,
        ))

    def record(
        self,
        context: Dict[str, Any],
        strategy: str,
        reward_info: Dict[str, Any],
        our_message: str,
        their_response: Optional[str],
    ):
        """Record a complete experience tuple."""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "context_key": context.get("context_key", "unknown"),
            "conversation_state": context.get("conversation_state", "unknown"),
            "emotional_tone": context.get("emotional_tone", "neutral"),
            "strategy": strategy,
            "reward": reward_info.get("total_reward", 0.5),
            "reward_signals": reward_info.get("signals", {}),
            "our_message_preview": our_message[:80],
            "their_response_preview": (their_response or "")[:80],
        }
        self.experiences.append(experience)

        # Trim buffer
        if len(self.experiences) > self.MAX_BUFFER_SIZE:
            self.experiences = self.experiences[-self.MAX_BUFFER_SIZE:]

        self._save()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent experiences."""
        return self.experiences[-n:]

    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregate statistics per strategy."""
        stats = defaultdict(lambda: {"count": 0, "total_reward": 0.0, "rewards": []})

        for exp in self.experiences:
            strategy = exp["strategy"]
            reward = exp["reward"]
            stats[strategy]["count"] += 1
            stats[strategy]["total_reward"] += reward
            stats[strategy]["rewards"].append(reward)

        result = {}
        for strategy, data in stats.items():
            avg = data["total_reward"] / max(data["count"], 1)
            rewards = data["rewards"]
            # Standard deviation
            if len(rewards) > 1:
                mean = sum(rewards) / len(rewards)
                variance = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
                std = variance ** 0.5
            else:
                std = 0.0

            result[strategy] = {
                "count": data["count"],
                "avg_reward": round(avg, 4),
                "std_reward": round(std, 4),
                "best_reward": round(max(rewards), 4) if rewards else 0.0,
                "worst_reward": round(min(rewards), 4) if rewards else 0.0,
            }

        return result

    def get_recent_rewards(self, n: int = 10) -> List[float]:
        """Get recent reward values for trend analysis."""
        return [exp["reward"] for exp in self.experiences[-n:]]


# ═══════════════════════════════════════════════════════════════
#  6. MAIN RL ENGINE (ORCHESTRATOR)
# ═══════════════════════════════════════════════════════════════

# In-memory cache for bandits and buffers
_bandits: Dict[int, ThompsonSamplingBandit] = {}
_buffers: Dict[int, ExperienceBuffer] = {}
_pending_actions: Dict[int, Dict[str, Any]] = {}


def _get_bandit(chat_id: int) -> ThompsonSamplingBandit:
    if chat_id not in _bandits:
        _bandits[chat_id] = ThompsonSamplingBandit(chat_id)
    return _bandits[chat_id]


def _get_buffer(chat_id: int) -> ExperienceBuffer:
    if chat_id not in _buffers:
        _buffers[chat_id] = ExperienceBuffer(chat_id)
    return _buffers[chat_id]


def select_response_strategy(
    chat_id: int,
    conversation_state: str = "small_talk",
    emotional_tone: str = "neutral",
    sentiment_score: float = 0.0,
    message_length: int = 50,
    relationship_health: float = 0.5,
    incoming_text: str = "",
) -> Dict[str, Any]:
    """Select the optimal response strategy for the current context.

    This is the main entry point — called BEFORE generating a reply.
    Returns the strategy name and its prompt directive to inject.
    """
    bandit = _get_bandit(chat_id)
    buffer = _get_buffer(chat_id)

    # Extract context features
    hour = datetime.now().hour
    recent_rewards = buffer.get_recent_rewards(10)

    context = extract_context_features(
        conversation_state=conversation_state,
        emotional_tone=emotional_tone,
        sentiment_score=sentiment_score,
        message_length=message_length,
        time_of_day=hour,
        relationship_health=relationship_health,
        recent_rewards=recent_rewards,
    )

    # Thompson Sampling selection
    strategy_name, confidence, all_samples = bandit.select_strategy(context)
    strategy_info = RESPONSE_STRATEGIES[strategy_name]

    # Store pending action for later reward calculation
    _pending_actions[chat_id] = {
        "timestamp": time.time(),
        "context": context,
        "strategy": strategy_name,
        "incoming_text": incoming_text,
        "emotional_tone": emotional_tone,
    }

    result = {
        "strategy": strategy_name,
        "description": strategy_info["description"],
        "prompt_directive": strategy_info["prompt_directive"],
        "confidence": confidence,
        "all_scores": all_samples,
        "context_key": context["context_key"],
        "exploration_ratio": _compute_exploration_ratio(bandit, context["context_key"]),
    }

    rl_logger.info(
        f"RL strategy for {chat_id}: {strategy_name} "
        f"(conf={confidence:.2f}, ctx={context['context_key']})"
    )

    return result


def _compute_exploration_ratio(bandit: ThompsonSamplingBandit, context_key: str) -> float:
    """How much are we still exploring vs exploiting?

    High ratio = still learning, low ratio = confident in policy.
    """
    if context_key not in bandit.params:
        return 1.0

    total_observations = sum(
        p["alpha"] + p["beta"] - 2
        for p in bandit.params[context_key].values()
    )

    # After ~100 observations, exploration drops to ~0.1
    return max(0.05, 1.0 / (1.0 + total_observations / 10.0))


def record_outcome(
    chat_id: int,
    our_message: str,
    their_response: Optional[str],
    their_emotion: str = "neutral",
    response_delay_seconds: Optional[float] = None,
    conversation_continued: bool = False,
) -> Optional[Dict[str, Any]]:
    """Record the outcome of our reply and update the bandit.

    Called AFTER we observe their next message (or lack thereof).
    This closes the feedback loop.
    """
    pending = _pending_actions.pop(chat_id, None)
    if not pending:
        rl_logger.debug(f"No pending action for chat {chat_id}")
        return None

    bandit = _get_bandit(chat_id)
    buffer = _get_buffer(chat_id)

    # Calculate implicit reward
    reward_info = calculate_implicit_reward(
        our_message=our_message,
        their_response=their_response,
        response_delay_seconds=response_delay_seconds,
        our_emotion=pending["emotional_tone"],
        their_emotion=their_emotion,
        conversation_continued=conversation_continued,
    )

    # Update bandit
    context_key = pending["context"]["context_key"]
    strategy = pending["strategy"]
    bandit.update(context_key, strategy, reward_info["total_reward"])

    # Record experience
    buffer.record(
        context=pending["context"],
        strategy=strategy,
        reward_info=reward_info,
        our_message=our_message,
        their_response=their_response,
    )

    rl_logger.info(
        f"RL outcome for {chat_id}: strategy={strategy}, "
        f"reward={reward_info['total_reward']:.3f} ({reward_info['explanation']})"
    )

    return {
        "strategy_used": strategy,
        "reward": reward_info["total_reward"],
        "signals": reward_info["signals"],
        "explanation": reward_info["explanation"],
        "total_experiences": len(buffer.experiences),
    }


# ═══════════════════════════════════════════════════════════════
#  7. STRATEGY PROMPT FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_strategy_for_prompt(strategy_result: Dict[str, Any]) -> str:
    """Format the selected strategy as a system prompt injection.

    This tells the model HOW to respond based on RL-selected strategy.
    """
    strategy_name = strategy_result["strategy"]
    directive = strategy_result["prompt_directive"]
    confidence = strategy_result.get("confidence", 0.5)
    exploration = strategy_result.get("exploration_ratio", 0.5)

    lines = [
        f"Strategy: {strategy_name.replace('_', ' ').title()}",
        f"Approach: {directive}",
    ]

    # If we're still exploring, add a note
    if exploration > 0.5:
        lines.append("Note: Still calibrating for this conversation context — try this approach.")
    elif confidence > 0.8:
        lines.append("Note: This approach has worked well in similar situations.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  8. ANALYTICS & INSIGHTS
# ═══════════════════════════════════════════════════════════════

def get_rl_insights(chat_id: int) -> Dict[str, Any]:
    """Get learning insights for a specific chat.

    Returns strategy statistics, best/worst performers,
    learning progress, and recommendations.
    """
    buffer = _get_buffer(chat_id)
    bandit = _get_bandit(chat_id)

    stats = buffer.get_strategy_stats()
    recent = buffer.get_recent(20)
    total_exp = len(buffer.experiences)

    # Find best and worst strategies
    if stats:
        best = max(stats.items(), key=lambda x: x[1]["avg_reward"])
        worst = min(stats.items(), key=lambda x: x[1]["avg_reward"])
    else:
        best = worst = None

    # Learning curve (reward trend over time)
    all_rewards = [exp["reward"] for exp in buffer.experiences]
    if len(all_rewards) >= 20:
        first_half = all_rewards[:len(all_rewards)//2]
        second_half = all_rewards[len(all_rewards)//2:]
        improvement = (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))
    else:
        improvement = 0.0

    # Context coverage
    contexts_seen = set()
    for exp in buffer.experiences:
        contexts_seen.add(exp.get("context_key", "unknown"))

    return {
        "chat_id": chat_id,
        "total_experiences": total_exp,
        "strategy_stats": stats,
        "best_strategy": {
            "name": best[0],
            "avg_reward": best[1]["avg_reward"],
            "count": best[1]["count"],
        } if best else None,
        "worst_strategy": {
            "name": worst[0],
            "avg_reward": worst[1]["avg_reward"],
            "count": worst[1]["count"],
        } if worst else None,
        "learning_improvement": round(improvement, 4),
        "contexts_seen": len(contexts_seen),
        "recent_avg_reward": round(
            sum(r["reward"] for r in recent) / max(len(recent), 1), 4
        ) if recent else 0.0,
        "exploration_status": "exploring" if total_exp < 50 else "exploiting",
    }


def get_all_chat_insights() -> Dict[int, Dict[str, Any]]:
    """Get RL insights across all chats."""
    results = {}
    for f in RL_DATA_DIR.glob("*_experiences.json"):
        try:
            chat_id = int(f.stem.replace("_experiences", ""))
            results[chat_id] = get_rl_insights(chat_id)
        except (ValueError, Exception):
            continue
    return results
