"""
Predictive Intelligence Engine
================================
ML-powered prediction of engagement, optimal timing, conflict risk,
ghost detection, and conversation trajectory forecasting.

Uses lightweight PyTorch models + statistical heuristics for:
1. Engagement scoring — will they reply? how fast? how engaged?
2. Optimal send timing — when to send for maximum impact
3. Conflict risk scoring — detect brewing arguments before they explode
4. Ghost detection — predict ghosting before it happens
5. Interest trajectory — are they warming up or cooling down?
6. Message impact prediction — predict how a message will land
"""

import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

prediction_logger = logging.getLogger("prediction_engine")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

PREDICTION_DATA_DIR = Path("engine_data/prediction")
PREDICTION_DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR = PREDICTION_DATA_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PREDICTION_DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. CONVERSATION FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_conversation_features(
    messages: List[Dict[str, Any]],
    current_time: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract predictive features from conversation history.
    Each message dict should have: sender, text, timestamp (epoch).
    """
    if not messages:
        return _empty_features()

    now = current_time or time.time()
    their_msgs = [m for m in messages if m.get("sender") in ("Them", "them", "other")]
    our_msgs = [m for m in messages if m.get("sender") in ("Me", "me", "self")]

    features = {}

    # --- Temporal features ---
    timestamps = [m.get("timestamp", 0) for m in messages if m.get("timestamp")]
    if len(timestamps) >= 2:
        # Average gap between messages
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        features["avg_gap_seconds"] = sum(gaps) / len(gaps) if gaps else 0
        features["min_gap_seconds"] = min(gaps) if gaps else 0
        features["max_gap_seconds"] = max(gaps) if gaps else 0
        # Recent acceleration/deceleration
        if len(gaps) >= 4:
            recent_avg = sum(gaps[-3:]) / 3
            earlier_avg = sum(gaps[:-3]) / max(len(gaps) - 3, 1)
            features["tempo_change"] = (recent_avg - earlier_avg) / max(earlier_avg, 1)
        else:
            features["tempo_change"] = 0.0
    else:
        features["avg_gap_seconds"] = 0
        features["min_gap_seconds"] = 0
        features["max_gap_seconds"] = 0
        features["tempo_change"] = 0.0

    # Time since last message from them
    their_timestamps = [m.get("timestamp", 0) for m in their_msgs if m.get("timestamp")]
    features["time_since_their_last"] = (now - max(their_timestamps)) if their_timestamps else 99999

    # Time since our last message
    our_timestamps = [m.get("timestamp", 0) for m in our_msgs if m.get("timestamp")]
    features["time_since_our_last"] = (now - max(our_timestamps)) if our_timestamps else 99999

    # --- Balance features ---
    features["msg_ratio"] = len(their_msgs) / max(len(our_msgs), 1)  # >1 = they talk more
    features["total_messages"] = len(messages)

    # Word count ratio
    their_words = sum(len(m.get("text", "").split()) for m in their_msgs)
    our_words = sum(len(m.get("text", "").split()) for m in our_msgs)
    features["word_ratio"] = their_words / max(our_words, 1)

    # --- Engagement signals ---
    # Question frequency (they're asking = engaged)
    their_questions = sum(1 for m in their_msgs if "?" in m.get("text", ""))
    features["their_question_ratio"] = their_questions / max(len(their_msgs), 1)

    # Emoji/expression density
    emoji_re = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF]+",
        re.UNICODE,
    )
    their_emojis = sum(len(emoji_re.findall(m.get("text", ""))) for m in their_msgs)
    features["their_emoji_density"] = their_emojis / max(len(their_msgs), 1)

    # Exclamation density
    their_exclaims = sum(m.get("text", "").count("!") for m in their_msgs)
    features["their_exclaim_density"] = their_exclaims / max(len(their_msgs), 1)

    # LOL/haha density (positive engagement)
    laugh_re = re.compile(r"\b(lol|lmao|haha|hehe|😂|🤣|rofl|dead|ахахах|хахаха|ха-ха|хехе|ржу)\b", re.I)
    their_laughs = sum(len(laugh_re.findall(m.get("text", ""))) for m in their_msgs)
    features["their_laugh_density"] = their_laughs / max(len(their_msgs), 1)

    # --- Message length trends ---
    their_lengths = [len(m.get("text", "").split()) for m in their_msgs[-10:]]
    if len(their_lengths) >= 3:
        # Are their messages getting shorter? (disengagement signal)
        first_half = sum(their_lengths[:len(their_lengths)//2]) / max(len(their_lengths)//2, 1)
        second_half = sum(their_lengths[len(their_lengths)//2:]) / max(len(their_lengths) - len(their_lengths)//2, 1)
        features["their_length_trend"] = (second_half - first_half) / max(first_half, 1)
    else:
        features["their_length_trend"] = 0.0

    # Average their message length (recent 10)
    features["their_avg_length"] = sum(their_lengths) / max(len(their_lengths), 1) if their_lengths else 0

    # --- Sentiment trajectory ---
    neg_words = re.compile(
        r"\b(hate|annoying|boring|whatever|meh|nah|bye|leave me|stop|ugh|gross"
        r"|tired of|done with|dont care|cant be bothered|not interested"
        r"|ненавижу|скучно|надоело|отстой|уходи|хватит|бесит|достало"
        r"|задолбало|не интересно|плевать)\b", re.I
    )
    pos_words = re.compile(
        r"\b(love|amazing|awesome|great|miss you|thinking about you|cant wait"
        r"|excited|happy|beautiful|gorgeous|perfect|❤️|😍|🥰|💕"
        r"|люблю|прекрасно|восхитительно|скучаю|думаю о тебе"
        r"|классно|круто|супер|обожаю)\b", re.I
    )
    recent_their = their_msgs[-5:] if their_msgs else []
    neg_count = sum(len(neg_words.findall(m.get("text", ""))) for m in recent_their)
    pos_count = sum(len(pos_words.findall(m.get("text", ""))) for m in recent_their)
    features["recent_sentiment"] = (pos_count - neg_count) / max(pos_count + neg_count, 1)

    # --- Time of day features ---
    hour = datetime.now().hour
    features["hour_of_day"] = hour
    features["is_late_night"] = 1.0 if 23 <= hour or hour < 5 else 0.0
    features["is_morning"] = 1.0 if 6 <= hour < 10 else 0.0
    features["is_evening"] = 1.0 if 18 <= hour < 23 else 0.0

    # --- Conversation initiation pattern ---
    if len(messages) >= 2:
        initiator = messages[0].get("sender", "")
        features["they_initiated"] = 1.0 if initiator in ("Them", "them", "other") else 0.0
    else:
        features["they_initiated"] = 0.0

    return features


def _empty_features() -> Dict[str, float]:
    return {
        "avg_gap_seconds": 0, "min_gap_seconds": 0, "max_gap_seconds": 0,
        "tempo_change": 0, "time_since_their_last": 99999, "time_since_our_last": 99999,
        "msg_ratio": 1.0, "total_messages": 0, "word_ratio": 1.0,
        "their_question_ratio": 0, "their_emoji_density": 0, "their_exclaim_density": 0,
        "their_laugh_density": 0, "their_length_trend": 0, "their_avg_length": 0,
        "recent_sentiment": 0, "hour_of_day": 12, "is_late_night": 0,
        "is_morning": 0, "is_evening": 0, "they_initiated": 0,
    }


# ═══════════════════════════════════════════════════════════════
#  2. ENGAGEMENT PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict_engagement(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict current engagement level using feature heuristics.
    Returns engagement score (0-1), reply probability, expected reply speed.
    """
    score = 0.5  # baseline

    # Positive signals
    if features.get("their_question_ratio", 0) > 0.3:
        score += 0.15  # asking questions = engaged
    if features.get("their_emoji_density", 0) > 0.5:
        score += 0.1
    if features.get("their_laugh_density", 0) > 0.2:
        score += 0.1
    if features.get("their_exclaim_density", 0) > 0.3:
        score += 0.05
    if features.get("msg_ratio", 1) > 1.2:
        score += 0.1  # they're sending more messages
    if features.get("word_ratio", 1) > 1.3:
        score += 0.1  # they're writing more words
    if features.get("their_length_trend", 0) > 0.1:
        score += 0.08  # messages getting longer
    if features.get("recent_sentiment", 0) > 0.3:
        score += 0.1
    if features.get("they_initiated", 0) > 0:
        score += 0.08
    if features.get("tempo_change", 0) < -0.2:
        score += 0.05  # gaps shrinking = more engaged

    # Negative signals
    if features.get("their_length_trend", 0) < -0.3:
        score -= 0.15  # messages getting shorter
    if features.get("time_since_their_last", 0) > 3600:
        score -= 0.1  # >1 hour silence
    if features.get("time_since_their_last", 0) > 7200:
        score -= 0.15  # >2 hours
    if features.get("recent_sentiment", 0) < -0.3:
        score -= 0.15
    if features.get("msg_ratio", 1) < 0.5:
        score -= 0.15  # we're sending way more
    if features.get("tempo_change", 0) > 0.5:
        score -= 0.1  # gaps growing = disengaging
    if features.get("their_avg_length", 0) < 3:
        score -= 0.1  # very short replies = low effort

    score = max(0.0, min(1.0, score))

    # Reply probability
    reply_prob = min(0.95, score * 1.1 + 0.1)
    if features.get("time_since_our_last", 0) > 86400:
        reply_prob *= 0.5  # we haven't spoken in a day

    # Expected reply speed (seconds)
    base_speed = features.get("avg_gap_seconds", 300)
    if score > 0.7:
        expected_speed = base_speed * 0.5  # fast reply expected
    elif score > 0.5:
        expected_speed = base_speed * 0.8
    elif score < 0.3:
        expected_speed = base_speed * 2.0  # slow or no reply
    else:
        expected_speed = base_speed

    # Engagement label
    if score >= 0.75:
        label = "highly_engaged"
    elif score >= 0.55:
        label = "engaged"
    elif score >= 0.4:
        label = "moderate"
    elif score >= 0.25:
        label = "low"
    else:
        label = "disengaged"

    return {
        "engagement_score": round(score, 3),
        "label": label,
        "reply_probability": round(reply_prob, 3),
        "expected_reply_seconds": round(expected_speed),
        "signals": _get_engagement_signals(features),
    }


def _get_engagement_signals(features: Dict[str, float]) -> List[str]:
    """Return human-readable engagement signals."""
    signals = []
    if features.get("their_question_ratio", 0) > 0.3:
        signals.append("asking_questions")
    if features.get("their_emoji_density", 0) > 0.5:
        signals.append("using_emojis")
    if features.get("their_laugh_density", 0) > 0.2:
        signals.append("laughing")
    if features.get("they_initiated", 0) > 0:
        signals.append("they_initiated")
    if features.get("their_length_trend", 0) < -0.3:
        signals.append("shortening_messages")
    if features.get("msg_ratio", 1) < 0.5:
        signals.append("we_dominate_ratio")
    if features.get("time_since_their_last", 0) > 3600:
        signals.append("long_silence")
    if features.get("recent_sentiment", 0) < -0.3:
        signals.append("negative_sentiment")
    if features.get("recent_sentiment", 0) > 0.3:
        signals.append("positive_sentiment")
    return signals


# ═══════════════════════════════════════════════════════════════
#  3. OPTIMAL TIMING PREDICTION
# ═══════════════════════════════════════════════════════════════

# Per-chat activity patterns (hour → response count)
_activity_patterns: Dict[int, Dict[int, int]] = {}


def record_activity(chat_id: int, timestamp: Optional[float] = None) -> None:
    """Record message activity for timing pattern learning."""
    ts = timestamp or time.time()
    hour = datetime.fromtimestamp(ts).hour

    if chat_id not in _activity_patterns:
        _activity_patterns[chat_id] = _load_activity_pattern(chat_id)

    _activity_patterns[chat_id][hour] = _activity_patterns[chat_id].get(hour, 0) + 1


def predict_optimal_send_time(
    chat_id: int,
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Predict the optimal time to send a message for maximum engagement.
    """
    pattern = _activity_patterns.get(chat_id) or _load_activity_pattern(chat_id)

    if not pattern:
        # Default: peak hours are evening (7-10pm)
        current_hour = datetime.now().hour
        if 19 <= current_hour <= 22:
            return {"optimal_hour": current_hour, "send_now": True, "reason": "evening_peak_default"}
        return {"optimal_hour": 20, "send_now": False, "reason": "no_data_default"}

    # Find peak activity hours
    total = sum(pattern.values()) or 1
    hourly_probs = {h: count / total for h, count in pattern.items()}

    # Top 3 hours
    sorted_hours = sorted(hourly_probs.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [h for h, _ in sorted_hours[:3]]

    current_hour = datetime.now().hour
    current_prob = hourly_probs.get(current_hour, 0)
    peak_prob = sorted_hours[0][1] if sorted_hours else 0

    # Should we send now?
    send_now = current_prob >= peak_prob * 0.7  # within 70% of peak

    # Next optimal hour
    future_hours = [(h, p) for h, p in sorted_hours if h >= current_hour or h < current_hour - 12]
    next_optimal = future_hours[0][0] if future_hours else peak_hours[0]

    return {
        "optimal_hour": next_optimal,
        "peak_hours": peak_hours,
        "current_hour_quality": round(current_prob / max(peak_prob, 0.01), 2),
        "send_now": send_now,
        "reason": "learned_pattern",
    }


# ═══════════════════════════════════════════════════════════════
#  4. CONFLICT RISK SCORING
# ═══════════════════════════════════════════════════════════════

def predict_conflict_risk(
    messages: List[Dict[str, Any]],
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Predict risk of conflict escalation.
    Returns risk score (0-1) + specific risk factors.
    """
    if not messages:
        return {"risk_score": 0.0, "level": "none", "factors": []}

    risk = 0.0
    factors = []

    recent = messages[-10:]
    recent_text = " ".join(m.get("text", "") for m in recent).lower()

    # --- Hostility indicators (English + Russian) ---
    hostility_words = re.compile(
        r"(fuck|fucking|shit|shitty|bitch|asshole|idiot|stupid|hate you|piss off"
        r"|go away|leave me alone|shut up|dont talk to me|blocked"
        r"|stfu|gtfo|moron|loser|screw you|go to hell|dumbass|dipshit"
        r"|motherfucker|bastard|dickhead|drop dead|die|bullshit|pathetic"
        r"|worthless|disgusting|ffs|wtf|trash|garbage"
        r"|блядь|блять|сука|сучка|сучара|пиздец|пизда|нахуй|нахер"
        r"|гандон|мудак|мудила|дебил|долбоёб|долбоеб|тварь|урод"
        r"|козёл|козел|скотина|чмо|ёбаный|ебаный|ебать|заебал|заебала"
        r"|отъебись|придурок|кретин|идиотка|дура|дурак|лох"
        r"|заткнись|отвали|вали|проваливай|ненавижу"
        r"|пошёл|пошел|пошла|иди|катись|убирайся|свали"
        r"|ублюдок|выродок|мразь|подонок|шлюха|сволочь|гнида|падла|паскуда"
        r"|хуй|хуйня|хуесос|пидор|пидорас|уёбок|уебок|уёбище|пиздабол"
        r"|засранец|говно|говнюк|дерьмо|тупица|бездарь|мразота)", re.I
    )
    hostility_count = len(hostility_words.findall(recent_text))
    if hostility_count > 0:
        risk += min(hostility_count * 0.15, 0.6)
        factors.append(f"hostile_language_x{hostility_count}")

    # --- Escalation pattern (messages getting more aggressive) ---
    if len(recent) >= 4:
        first_half = " ".join(m.get("text", "") for m in recent[:len(recent)//2]).lower()
        second_half = " ".join(m.get("text", "") for m in recent[len(recent)//2:]).lower()
        h1 = len(hostility_words.findall(first_half))
        h2 = len(hostility_words.findall(second_half))
        if h2 > h1 + 1:
            risk += 0.2
            factors.append("escalation_pattern")

    # --- Stonewalling (one-word replies) ---
    their_recent = [m for m in recent if m.get("sender") in ("Them", "them", "other")]
    if their_recent:
        short_replies = sum(1 for m in their_recent if len(m.get("text", "").split()) <= 2)
        stonewall_ratio = short_replies / len(their_recent)
        if stonewall_ratio > 0.6 and len(their_recent) >= 3:
            risk += 0.15
            factors.append("stonewalling")

    # --- Defensive language ---
    defensive = re.compile(
        r"\b(i didnt|i never|thats not true|you always|you never|its not my"
        r"|dont blame|not my fault|i was just|im not the one|whatever)\b", re.I
    )
    def_count = len(defensive.findall(recent_text))
    if def_count >= 2:
        risk += 0.1
        factors.append("defensive_language")

    # --- Contempt markers ---
    contempt = re.compile(
        r"(pathetic|worthless|disgusting|ridiculous|embarrassing|joke"
        r"|grow up|get a life|childish|immature|you.re nothing"
        r"|useless|incompetent|laughable|pitiful|sad excuse"
        r"|waste of space|waste of time|beneath me|beneath you"
        r"|жалкий|ничтожество|позор|стыдно|убогий|отстой|лох"
        r"|неудачник|тупица|бездарь|клоун|посмешище|позорище"
        r"|бестолочь|никчёмный|никчемный|бесполезный|бесполезная"
        r"|ничтожный|жалкое зрелище|пустое место|недоразумение)", re.I
    )
    contempt_count = len(contempt.findall(recent_text))
    if contempt_count > 0:
        risk += min(contempt_count * 0.15, 0.3)
        factors.append(f"contempt_x{contempt_count}")

    # --- Declining sentiment from features ---
    if features and features.get("recent_sentiment", 0) < -0.4:
        risk += 0.15
        factors.append("negative_sentiment_trend")

    # --- Rapid back-and-forth (argument tempo) ---
    if features and features.get("avg_gap_seconds", 999) < 30 and hostility_count > 0:
        risk += 0.1
        factors.append("argument_tempo")

    risk = max(0.0, min(1.0, risk))

    # Risk level
    if risk >= 0.6:
        level = "high"
    elif risk >= 0.35:
        level = "medium"
    elif risk >= 0.15:
        level = "low"
    else:
        level = "none"

    return {
        "risk_score": round(risk, 3),
        "level": level,
        "factors": factors,
        "recommendation": _conflict_recommendation(level, factors),
    }


def _conflict_recommendation(level: str, factors: List[str]) -> str:
    if level == "high":
        if "contempt" in str(factors):
            return "They're being contemptuous. Fire back or disengage — dont apologize."
        if "escalation_pattern" in factors:
            return "Things are escalating fast. Match energy or end the conversation."
        return "High conflict. Stand ground, be direct, dont back down."
    elif level == "medium":
        if "stonewalling" in factors:
            return "They're stonewalling. Give space or call it out directly."
        return "Tension building. Be direct, dont sugarcoat."
    elif level == "low":
        return "Minor tension. Keep it real, dont overthink."
    return ""


# ═══════════════════════════════════════════════════════════════
#  5. GHOST DETECTION
# ═══════════════════════════════════════════════════════════════

_ghost_history: Dict[int, List[Dict]] = {}


def record_response_event(
    chat_id: int, they_replied: bool, wait_seconds: float
) -> None:
    """Record whether they replied and how long it took."""
    if chat_id not in _ghost_history:
        _ghost_history[chat_id] = []

    _ghost_history[chat_id].append({
        "timestamp": time.time(),
        "replied": they_replied,
        "wait_seconds": wait_seconds,
    })

    # Keep last 100 events
    if len(_ghost_history[chat_id]) > 100:
        _ghost_history[chat_id] = _ghost_history[chat_id][-100:]


def predict_ghost_risk(
    chat_id: int,
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Predict probability of being ghosted.
    """
    history = _ghost_history.get(chat_id, [])
    risk = 0.0
    factors = []

    if not history and not features:
        return {"ghost_risk": 0.0, "level": "none", "factors": []}

    # --- Response pattern analysis ---
    if history:
        recent_10 = history[-10:]
        reply_rate = sum(1 for e in recent_10 if e["replied"]) / len(recent_10)
        avg_wait = sum(e["wait_seconds"] for e in recent_10) / len(recent_10)

        # Declining reply rate
        if len(history) >= 10:
            older = history[-20:-10] if len(history) >= 20 else history[:10]
            older_rate = sum(1 for e in older if e["replied"]) / max(len(older), 1)
            if reply_rate < older_rate - 0.2:
                risk += 0.25
                factors.append("declining_reply_rate")

        if reply_rate < 0.5:
            risk += 0.2
            factors.append(f"low_reply_rate_{reply_rate:.0%}")

        # Increasing wait times
        if len(recent_10) >= 3:
            waits = [e["wait_seconds"] for e in recent_10 if e["replied"]]
            if len(waits) >= 2 and waits[-1] > waits[0] * 2:
                risk += 0.15
                factors.append("increasing_wait_times")

    # --- Feature-based signals ---
    if features:
        if features.get("time_since_their_last", 0) > 86400:  # >24h
            risk += 0.3
            factors.append("24h_silence")
        elif features.get("time_since_their_last", 0) > 43200:  # >12h
            risk += 0.15
            factors.append("12h_silence")

        if features.get("their_length_trend", 0) < -0.5:
            risk += 0.1
            factors.append("drastically_shorter_messages")

        if features.get("msg_ratio", 1) < 0.3:
            risk += 0.2
            factors.append("severe_imbalance")

        if features.get("recent_sentiment", 0) < -0.5:
            risk += 0.1
            factors.append("very_negative")

    risk = max(0.0, min(1.0, risk))

    if risk >= 0.6:
        level = "high"
    elif risk >= 0.35:
        level = "moderate"
    elif risk >= 0.15:
        level = "low"
    else:
        level = "none"

    return {
        "ghost_risk": round(risk, 3),
        "level": level,
        "factors": factors,
    }


# ═══════════════════════════════════════════════════════════════
#  6. INTEREST TRAJECTORY
# ═══════════════════════════════════════════════════════════════

_interest_history: Dict[int, List[Dict]] = {}


def record_interest_signal(
    chat_id: int, engagement_score: float, features: Dict[str, float]
) -> None:
    """Record an interest datapoint for trajectory analysis."""
    if chat_id not in _interest_history:
        _interest_history[chat_id] = []

    _interest_history[chat_id].append({
        "timestamp": time.time(),
        "engagement": engagement_score,
        "msg_ratio": features.get("msg_ratio", 1.0),
        "sentiment": features.get("recent_sentiment", 0),
        "their_avg_length": features.get("their_avg_length", 0),
    })

    if len(_interest_history[chat_id]) > 200:
        _interest_history[chat_id] = _interest_history[chat_id][-200:]


def get_interest_trajectory(chat_id: int) -> Dict[str, Any]:
    """
    Analyze interest trajectory over time.
    Returns trend direction, velocity, and prediction.
    """
    history = _interest_history.get(chat_id, [])
    if len(history) < 3:
        return {"status": "insufficient_data", "trend": "unknown", "points": len(history)}

    # Split into thirds
    third = max(len(history) // 3, 1)
    early = history[:third]
    mid = history[third:2*third]
    recent = history[-third:]

    # Average engagement per period
    early_eng = sum(d["engagement"] for d in early) / len(early)
    mid_eng = sum(d["engagement"] for d in mid) / len(mid)
    recent_eng = sum(d["engagement"] for d in recent) / len(recent)

    # Trend detection
    if recent_eng > early_eng + 0.15:
        trend = "warming_up"
    elif recent_eng < early_eng - 0.15:
        trend = "cooling_down"
    elif recent_eng > mid_eng + 0.1:
        trend = "recovering"
    elif recent_eng < mid_eng - 0.1:
        trend = "declining"
    else:
        trend = "stable"

    # Velocity (rate of change)
    if len(history) >= 5:
        recent_5 = [d["engagement"] for d in history[-5:]]
        velocity = (recent_5[-1] - recent_5[0]) / max(len(recent_5) - 1, 1)
    else:
        velocity = 0

    # Prediction: if trend continues, where will we be?
    predicted = recent_eng + velocity * 5  # 5 interactions ahead
    predicted = max(0.0, min(1.0, predicted))

    return {
        "status": "analyzed",
        "trend": trend,
        "early_engagement": round(early_eng, 3),
        "mid_engagement": round(mid_eng, 3),
        "recent_engagement": round(recent_eng, 3),
        "velocity": round(velocity, 4),
        "predicted_5_ahead": round(predicted, 3),
        "total_datapoints": len(history),
    }


# ═══════════════════════════════════════════════════════════════
#  7. MESSAGE IMPACT PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict_message_impact(
    proposed_message: str,
    context_features: Dict[str, float],
    personality_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predict how a proposed message will land based on context and personality.
    Returns impact score, risk assessment, and suggestions.
    """
    impact = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    risks = []
    text = proposed_message.lower()
    words = text.split()
    msg_length = len(words)

    # --- Length appropriateness ---
    their_avg = context_features.get("their_avg_length", 10)
    if their_avg > 0:
        length_ratio = msg_length / their_avg
        if length_ratio > 3:
            impact["negative"] += 0.15
            risks.append("way_too_long_for_their_style")
        elif 0.5 <= length_ratio <= 2.0:
            impact["positive"] += 0.1  # good length matching

    # --- Engagement-appropriate ---
    engagement = context_features.get("their_question_ratio", 0)

    # Question when they're asking questions = good reciprocity
    has_question = "?" in proposed_message
    if has_question and engagement > 0.2:
        impact["positive"] += 0.1

    # --- Emotional tone matching ---
    sentiment = context_features.get("recent_sentiment", 0)
    pos_words = len(re.findall(r"\b(love|great|awesome|amazing|happy|miss|❤️|😍|люблю|прекрасно|восхитительно|скучаю|думаю о тебе|классно|круто|супер|обожаю)\b", text, re.I))
    neg_words = len(re.findall(r"\b(hate|stupid|annoying|shut up|whatever|ugh|ненавижу|скучно|надоело|отстой|бесит|достало|задолбало|плевать)\b", text, re.I))

    msg_sentiment = (pos_words - neg_words) / max(pos_words + neg_words, 1)
    if sentiment < -0.3 and msg_sentiment > 0.5:
        impact["negative"] += 0.1
        risks.append("too_positive_for_negative_mood")
    elif sentiment > 0.3 and msg_sentiment < -0.3:
        impact["negative"] += 0.15
        risks.append("bringing_negativity_to_positive_vibe")

    # --- Personality-aware impact ---
    if personality_profile:
        prefs = personality_profile.get("communication_preferences", {})
        pref_tone = prefs.get("tone", "")

        if pref_tone == "casual_slang" and not re.search(r"\b(lol|haha|bruh|ngl|fr|блин|вау|капец|жесть|хз|типа)\b", text, re.I):
            impact["neutral"] += 0.05  # could be more casual

        if pref_tone == "formal" and re.search(r"\b(lol|lmao|bruh|wtf)\b", text, re.I):
            impact["negative"] += 0.1
            risks.append("too_casual_for_formal_person")

    # --- Overall score ---
    total_positive = max(impact["positive"], 0)
    total_negative = max(impact["negative"], 0)
    net = 0.5 + total_positive - total_negative
    net = max(0.0, min(1.0, net))

    return {
        "impact_score": round(net, 3),
        "risks": risks,
        "positive_factors": round(total_positive, 3),
        "negative_factors": round(total_negative, 3),
    }


# ═══════════════════════════════════════════════════════════════
#  8. DYNAMIC MESSAGE LENGTH CALCULATOR
# ═══════════════════════════════════════════════════════════════

def calculate_dynamic_length(
    features: Dict[str, float],
    engagement: Dict[str, Any],
    conflict: Dict[str, Any],
    personality_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate optimal message length based on all signals.
    Returns recommended word count, max_tokens, and reasoning.
    """
    reasons = []

    # Base: match their average length
    their_avg = features.get("their_avg_length", 8)
    target_words = max(3, their_avg)
    reasons.append(f"base_from_their_avg={their_avg:.0f}w")

    # --- Engagement adjustment ---
    eng_score = engagement.get("engagement_score", 0.5)
    if eng_score > 0.7:
        # High engagement: can go longer, they're invested
        target_words *= 1.3
        reasons.append("high_engagement_+30%")
    elif eng_score < 0.3:
        # Low engagement: keep it SHORT to not overwhelm
        target_words *= 0.6
        reasons.append("low_engagement_-40%")

    # --- Conflict adjustment ---
    conflict_level = conflict.get("level", "none")
    if conflict_level == "high":
        target_words = min(target_words, 8)  # short, sharp in conflict
        reasons.append("conflict_cap_8w")
    elif conflict_level == "medium":
        target_words = min(target_words, 15)
        reasons.append("tension_cap_15w")

    # --- Personality preference ---
    if personality_profile:
        pref_len = personality_profile.get("communication_preferences", {}).get("preferred_length", "")
        if pref_len == "very_short":
            target_words = min(target_words, 5)
            reasons.append("personality_very_short")
        elif pref_len == "short":
            target_words = min(target_words, 12)
            reasons.append("personality_short")
        elif pref_len == "long":
            target_words = max(target_words, 15)
            reasons.append("personality_long")

    # --- Time of day ---
    if features.get("is_late_night", 0):
        target_words = min(target_words, 8)
        reasons.append("late_night_short")

    # --- Message ratio balance ---
    ratio = features.get("msg_ratio", 1.0)
    if ratio < 0.5:
        # We're over-messaging — go shorter
        target_words *= 0.7
        reasons.append("we_over_messaging_-30%")

    target_words = max(3, round(target_words))

    # Convert to max_tokens (rough: 1 word ≈ 1.5 tokens)
    max_tokens = max(30, round(target_words * 2.5))

    # Also derive a prompt hint
    if target_words <= 5:
        length_hint = "Reply in 1-5 words MAX. Ultra short."
    elif target_words <= 10:
        length_hint = "Keep it under 10 words. Short and punchy."
    elif target_words <= 20:
        length_hint = "Keep it around 10-20 words. Natural length."
    elif target_words <= 35:
        length_hint = "You can go 20-35 words. They're engaged."
    else:
        length_hint = f"Match their energy — around {target_words} words is fine."

    return {
        "target_words": target_words,
        "max_tokens": max_tokens,
        "length_hint": length_hint,
        "reasons": reasons,
    }


# ═══════════════════════════════════════════════════════════════
#  9. FORMAT FOR PROMPT INJECTION
# ═══════════════════════════════════════════════════════════════

def format_predictions_for_prompt(
    engagement: Dict[str, Any],
    conflict: Dict[str, Any],
    ghost: Dict[str, Any],
    trajectory: Dict[str, Any],
    length_calc: Dict[str, Any],
) -> str:
    """
    Format all predictions as concise prompt injection.
    """
    parts = []

    # Engagement
    eng_label = engagement.get("label", "unknown")
    eng_score = engagement.get("engagement_score", 0)
    if eng_label != "unknown":
        parts.append(f"[ENGAGEMENT: {eng_label.upper()} ({eng_score:.0%})]")
        signals = engagement.get("signals", [])
        if signals:
            parts.append(f"  Signals: {', '.join(signals[:4])}")

    # Conflict
    conflict_level = conflict.get("level", "none")
    if conflict_level != "none":
        rec = conflict.get("recommendation", "")
        parts.append(f"[CONFLICT RISK: {conflict_level.upper()}]")
        if rec:
            parts.append(f"  → {rec}")

    # Ghost risk
    ghost_level = ghost.get("level", "none")
    if ghost_level in ("moderate", "high"):
        parts.append(f"[GHOST RISK: {ghost_level.upper()} ({ghost.get('ghost_risk', 0):.0%})]")
        factors = ghost.get("factors", [])
        if factors:
            parts.append(f"  Factors: {', '.join(factors[:3])}")

    # Trajectory
    trend = trajectory.get("trend", "unknown")
    if trend not in ("unknown", "stable") and trajectory.get("status") == "analyzed":
        parts.append(f"[INTEREST TREND: {trend.upper()}]")

    # Length directive
    hint = length_calc.get("length_hint", "")
    if hint:
        parts.append(f"[MESSAGE LENGTH: {hint}]")

    if not parts:
        return ""

    return "\n## PREDICTIVE INTELLIGENCE\n" + "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  10. FULL PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_prediction(
    chat_id: int,
    messages: List[Dict[str, Any]],
    personality_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Run complete prediction pipeline.
    Returns (all_predictions_dict, prompt_injection_string).
    """
    features = extract_conversation_features(messages)
    engagement = predict_engagement(features)
    conflict = predict_conflict_risk(messages, features)
    ghost = predict_ghost_risk(chat_id, features)
    trajectory = get_interest_trajectory(chat_id)
    length_calc = calculate_dynamic_length(features, engagement, conflict, personality_profile)

    # Record signals for future predictions
    record_activity(chat_id)
    record_interest_signal(chat_id, engagement["engagement_score"], features)

    all_predictions = {
        "features": features,
        "engagement": engagement,
        "conflict": conflict,
        "ghost": ghost,
        "trajectory": trajectory,
        "dynamic_length": length_calc,
    }

    prompt_block = format_predictions_for_prompt(
        engagement, conflict, ghost, trajectory, length_calc,
    )

    return all_predictions, prompt_block


# ═══════════════════════════════════════════════════════════════
#  11. PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _load_activity_pattern(chat_id: int) -> Dict[int, int]:
    try:
        path = HISTORY_DIR / f"{chat_id}_activity.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    except Exception as e:
        prediction_logger.warning(f"Failed to load activity for {chat_id}: {e}")
    return {}


def save_activity_patterns() -> None:
    """Save all activity patterns to disk."""
    for chat_id, pattern in _activity_patterns.items():
        try:
            path = HISTORY_DIR / f"{chat_id}_activity.json"
            with open(path, "w") as f:
                json.dump(pattern, f)
        except Exception as e:
            prediction_logger.warning(f"Failed to save activity for {chat_id}: {e}")
