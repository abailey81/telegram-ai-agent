"""
Advanced Personality Profiling Engine
======================================
Real-time personality inference from conversation patterns using
Big Five (OCEAN), Dark Triad, communication style DNA, and
personality evolution tracking over time.

Features:
1. Big Five (OCEAN) scoring from linguistic cues
2. Dark Triad detection (Machiavellianism, Narcissism, Psychopathy)
3. Communication Style DNA — fingerprint of HOW they talk
4. Personality evolution — track shifts over weeks/months
5. Compatibility scoring between bot persona and target
6. Adaptive persona tuning — adjust bot behavior to complement
"""

import json
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

personality_logger = logging.getLogger("personality_engine")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

PERSONALITY_DATA_DIR = Path("engine_data/personality")
PERSONALITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DIR = PERSONALITY_DATA_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
EVOLUTION_DIR = PERSONALITY_DATA_DIR / "evolution"
EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
STYLE_DNA_DIR = PERSONALITY_DATA_DIR / "style_dna"
STYLE_DNA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. LINGUISTIC MARKERS FOR BIG FIVE (OCEAN)
# ═══════════════════════════════════════════════════════════════

# Based on Pennebaker's LIWC research + Schwartz et al. (2013)
OCEAN_MARKERS = {
    "openness": {
        "high": [
            r"\b(imagine|wonder|what if|curious|fascin|creative|idea|concept|abstract"
            r"|philosophy|art|explore|discover|novel|unique|original|insight|perspective"
            r"|theory|hypothe|innovat|experiment|vision)\b",
        ],
        "low": [
            r"\b(normal|usual|traditional|always been|same as|everyone does"
            r"|obviously|clearly|common sense|simple|basic|standard|regular)\b",
        ],
    },
    "conscientiousness": {
        "high": [
            r"\b(plan|schedule|organiz|careful|detail|precise|thorough|efficient"
            r"|deadline|on time|prepared|systematic|discipline|responsib|commit"
            r"|goal|achievement|accomplish|structure|method)\b",
        ],
        "low": [
            r"\b(whatever|idc|don.t care|yolo|spontaneous|last minute|forgot"
            r"|oops|my bad|lazy|procrastinat|chill|go with the flow|wing it"
            r"|meh|eh|nah)\b",
        ],
    },
    "extraversion": {
        "high": [
            r"\b(party|friends|hangout|everyone|social|fun|excit|amazing|awesome"
            r"|love it|omg|haha|lol|lmao|lets go|vibes|energy|people|crowd"
            r"|club|festival|group|together)\b",
        ],
        "low": [
            r"\b(alone|quiet|introvert|recharge|peaceful|solitude|by myself"
            r"|staying in|prefer not|rather not|small group|one on one|private"
            r"|home|book|silent)\b",
        ],
    },
    "agreeableness": {
        "high": [
            r"\b(help|support|kind|nice|sweet|care|please|thank|sorry|understand"
            r"|forgive|generous|cooperat|trust|gentle|compas|empathy|harmony"
            r"|fair|considerate)\b",
        ],
        "low": [
            r"\b(disagree|wrong|stupid|idiot|annoying|hate|competition|better than"
            r"|dominate|win|argument|fight|confront|challenge|attack|blame"
            r"|your fault|pathetic|weak)\b",
        ],
    },
    "neuroticism": {
        "high": [
            r"\b(worry|anxious|stress|nervous|scared|afraid|panic|overwhelm"
            r"|can.t handle|falling apart|depressed|sad|cry|upset|frustrat"
            r"|angry|furious|irritat|insecure|doubt|uncertain)\b",
        ],
        "low": [
            r"\b(calm|relax|chill|no worries|all good|fine|stable|steady|confident"
            r"|secure|comfortable|peace|balanced|composed|resilient|strong"
            r"|unbothered|whatever happens)\b",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
#  2. DARK TRIAD MARKERS
# ═══════════════════════════════════════════════════════════════

DARK_TRIAD_MARKERS = {
    "machiavellianism": {
        "indicators": [
            r"\b(manipulat|strategic|advantage|leverage|useful|use them"
            r"|play the game|chess|move|calculated|agenda|ulterior|exploit"
            r"|power|influence|control|persuad|convince|scheme)\b",
        ],
        "weight": 1.0,
    },
    "narcissism": {
        "indicators": [
            r"\b(i.m the best|better than|superior|deserve|entitled|special"
            r"|admire me|look at me|my achievement|i always|i never fail"
            r"|they.re jealous|beneath me|worship|greatest|genius|perfect)\b",
        ],
        "weight": 1.0,
    },
    "psychopathy": {
        "indicators": [
            r"\b(don.t care about|feelings are weak|boring|no remorse"
            r"|thrill|risk|danger|impulsive|reckless|cold|detached"
            r"|manipulat.*emotion|fake cry|pretend to care|shallow)\b",
        ],
        "weight": 1.2,
    },
}


# ═══════════════════════════════════════════════════════════════
#  3. COMMUNICATION STYLE DNA
# ═══════════════════════════════════════════════════════════════

def extract_style_dna(messages: List[str]) -> Dict[str, Any]:
    """
    Extract a communication 'fingerprint' from messages.
    Returns quantified style dimensions.
    """
    if not messages:
        return _empty_style_dna()

    total_msgs = len(messages)
    all_text = " ".join(messages)
    all_words = all_text.split()
    total_words = max(len(all_words), 1)

    # --- Message length distribution ---
    lengths = [len(m.split()) for m in messages]
    avg_length = sum(lengths) / total_msgs
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / total_msgs

    # --- Vocabulary richness (type-token ratio) ---
    unique_words = len(set(w.lower() for w in all_words))
    ttr = unique_words / total_words

    # --- Emoji density ---
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
        re.UNICODE,
    )
    emoji_count = sum(len(emoji_pattern.findall(m)) for m in messages)
    emoji_density = emoji_count / total_msgs

    # --- Question frequency ---
    question_msgs = sum(1 for m in messages if "?" in m)
    question_ratio = question_msgs / total_msgs

    # --- Exclamation frequency ---
    exclaim_msgs = sum(1 for m in messages if "!" in m)
    exclaim_ratio = exclaim_msgs / total_msgs

    # --- Caps usage (shouting) ---
    caps_words = sum(1 for w in all_words if w.isupper() and len(w) > 1)
    caps_ratio = caps_words / total_words

    # --- Abbreviation / slang density ---
    slang_patterns = re.compile(
        r"\b(lol|lmao|omg|wtf|bruh|fr|ngl|tbh|imo|smh|ikr|idk|wyd|hbu"
        r"|rn|nvm|dm|fyi|ong|lowkey|highkey|nah|yea|yep|nope|aight"
        r"|cuz|tryna|gonna|wanna|gotta|kinda|sorta|finna|boutta)\b",
        re.IGNORECASE,
    )
    slang_count = len(slang_patterns.findall(all_text))
    slang_density = slang_count / total_words

    # --- Formality index (formal words vs slang) ---
    formal_patterns = re.compile(
        r"\b(however|furthermore|therefore|nevertheless|accordingly|consequently"
        r"|regarding|concerning|perhaps|indeed|certainly|quite|rather)\b",
        re.IGNORECASE,
    )
    formal_count = len(formal_patterns.findall(all_text))
    formality = formal_count / max(formal_count + slang_count, 1)

    # --- Response speed proxy (rapid-fire short msgs vs long thoughtful) ---
    short_msgs = sum(1 for l in lengths if l <= 3)
    rapid_fire_ratio = short_msgs / total_msgs

    # --- Emotional expressiveness ---
    emotion_words = re.compile(
        r"\b(love|hate|angry|happy|sad|excited|scared|nervous|amazing|terrible"
        r"|wonderful|awful|beautiful|ugly|gorgeous|horrible|fantastic|miserable)\b",
        re.IGNORECASE,
    )
    emotion_count = len(emotion_words.findall(all_text))
    emotional_expressiveness = emotion_count / total_words

    # --- Sarcasm indicators ---
    sarcasm_patterns = re.compile(
        r"(\.{3,}|sure\.{2,}|oh really|wow\s+just\s+wow|great\.{2,}|nice\.{2,}"
        r"|okay then|lol okay|suuure|riiiight|totally|yea right)",
        re.IGNORECASE,
    )
    sarcasm_hits = len(sarcasm_patterns.findall(all_text))
    sarcasm_density = sarcasm_hits / total_msgs

    # --- Self-reference ratio (I/me/my vs you/your) ---
    self_refs = len(re.findall(r"\b(i|me|my|mine|myself|i'm|i've|i'll|i'd)\b", all_text, re.I))
    other_refs = len(re.findall(r"\b(you|your|yours|yourself|you're|you've|you'll)\b", all_text, re.I))
    self_focus = self_refs / max(self_refs + other_refs, 1)

    return {
        "avg_msg_length": round(avg_length, 1),
        "length_variance": round(length_variance, 1),
        "vocabulary_richness": round(ttr, 3),
        "emoji_density": round(emoji_density, 2),
        "question_ratio": round(question_ratio, 3),
        "exclamation_ratio": round(exclaim_ratio, 3),
        "caps_ratio": round(caps_ratio, 3),
        "slang_density": round(slang_density, 3),
        "formality": round(formality, 3),
        "rapid_fire_ratio": round(rapid_fire_ratio, 3),
        "emotional_expressiveness": round(emotional_expressiveness, 3),
        "sarcasm_density": round(sarcasm_density, 3),
        "self_focus": round(self_focus, 3),
        "total_messages_analyzed": total_msgs,
    }


def _empty_style_dna() -> Dict[str, Any]:
    return {
        "avg_msg_length": 0, "length_variance": 0, "vocabulary_richness": 0,
        "emoji_density": 0, "question_ratio": 0, "exclamation_ratio": 0,
        "caps_ratio": 0, "slang_density": 0, "formality": 0,
        "rapid_fire_ratio": 0, "emotional_expressiveness": 0,
        "sarcasm_density": 0, "self_focus": 0, "total_messages_analyzed": 0,
    }


# ═══════════════════════════════════════════════════════════════
#  4. BIG FIVE SCORER
# ═══════════════════════════════════════════════════════════════

def score_big_five(messages: List[str]) -> Dict[str, float]:
    """
    Score Big Five personality traits from message history.
    Returns dict with O, C, E, A, N scores from 0.0 to 1.0.
    """
    if not messages:
        return {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                "agreeableness": 0.5, "neuroticism": 0.5}

    all_text = " ".join(messages).lower()
    scores = {}

    for trait, markers in OCEAN_MARKERS.items():
        high_count = 0
        low_count = 0
        for pattern in markers["high"]:
            high_count += len(re.findall(pattern, all_text, re.IGNORECASE))
        for pattern in markers["low"]:
            low_count += len(re.findall(pattern, all_text, re.IGNORECASE))

        total = high_count + low_count
        if total == 0:
            scores[trait] = 0.5  # neutral
        else:
            raw = high_count / total
            # Smooth toward center (avoid extreme scores from few datapoints)
            n_msgs = len(messages)
            confidence = min(n_msgs / 50, 1.0)  # full confidence at 50+ messages
            scores[trait] = round(0.5 + (raw - 0.5) * confidence, 3)

    return scores


# ═══════════════════════════════════════════════════════════════
#  5. DARK TRIAD SCORER
# ═══════════════════════════════════════════════════════════════

def score_dark_triad(messages: List[str]) -> Dict[str, float]:
    """
    Score Dark Triad traits. Returns 0.0-1.0 for each.
    High scores indicate strong presence of the trait.
    """
    if not messages:
        return {"machiavellianism": 0.0, "narcissism": 0.0, "psychopathy": 0.0}

    all_text = " ".join(messages).lower()
    total_words = max(len(all_text.split()), 1)
    scores = {}

    for trait, config in DARK_TRIAD_MARKERS.items():
        hit_count = 0
        for pattern in config["indicators"]:
            hit_count += len(re.findall(pattern, all_text, re.IGNORECASE))

        # Normalize: hits per 100 words, scaled by weight
        density = (hit_count / total_words) * 100 * config["weight"]
        # Sigmoid normalization to 0-1 range
        score = 1.0 / (1.0 + math.exp(-0.5 * (density - 3)))
        scores[trait] = round(score, 3)

    return scores


# ═══════════════════════════════════════════════════════════════
#  6. ATTACHMENT STYLE DETECTION
# ═══════════════════════════════════════════════════════════════

ATTACHMENT_INDICATORS = {
    "secure": {
        "patterns": [
            r"\b(trust|comfortable|safe|close|open up|share|connect|support"
            r"|together|reliable|consistent|honest|vulnerable)\b",
        ],
        "behaviors": {
            "balanced_self_focus": (0.35, 0.65),  # moderate self-reference
            "moderate_emotion": (0.01, 0.05),       # expressive but not volatile
            "question_engagement": (0.1, 0.4),      # asks but not anxiously
        },
    },
    "anxious": {
        "patterns": [
            r"\b(miss you|where are you|why aren.t you|please respond|are you mad"
            r"|don.t leave|need you|can.t without you|promise me|are we ok"
            r"|do you still|worried about us|scared you.ll|clingy)\b",
        ],
        "behaviors": {
            "high_question_ratio": (0.3, 1.0),
            "high_emotion": (0.05, 1.0),
            "short_response_gap": True,
        },
    },
    "avoidant": {
        "patterns": [
            r"\b(need space|too much|overwhelming|back off|independent|don.t need"
            r"|fine alone|not ready|slow down|too fast|whatever|doesn.t matter"
            r"|i.m good|don.t worry about it)\b",
        ],
        "behaviors": {
            "low_emotion": (0.0, 0.01),
            "low_question": (0.0, 0.05),
            "high_formality": (0.3, 1.0),
        },
    },
    "disorganized": {
        "patterns": [
            r"\b(i love you.*i hate you|come here.*go away|need you.*leave me"
            r"|hot and cold|push.*pull|confused about us|don.t know what i want"
            r"|one minute.*next minute)\b",
        ],
        "behaviors": {
            "high_variance": True,
            "emotional_swings": True,
        },
    },
}


def detect_attachment_style(
    messages: List[str], style_dna: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Detect attachment style from messages + style DNA.
    Returns primary style + confidence + secondary style.
    """
    if not messages:
        return {"primary": "unknown", "confidence": 0.0, "scores": {}}

    all_text = " ".join(messages).lower()
    scores = {}

    for style, config in ATTACHMENT_INDICATORS.items():
        score = 0.0
        # Pattern matching
        for pattern in config["patterns"]:
            hits = len(re.findall(pattern, all_text, re.IGNORECASE))
            score += hits * 2.0

        # Behavioral indicators from style DNA
        if style_dna:
            for behavior, threshold in config.get("behaviors", {}).items():
                if isinstance(threshold, tuple):
                    lo, hi = threshold
                    dna_key = behavior.replace("high_", "").replace("low_", "").replace("balanced_", "")
                    # Map behavior name to DNA key
                    dna_map = {
                        "self_focus": "self_focus",
                        "emotion": "emotional_expressiveness",
                        "question_ratio": "question_ratio",
                        "question": "question_ratio",
                        "formality": "formality",
                        "question_engagement": "question_ratio",
                        "moderate_emotion": "emotional_expressiveness",
                    }
                    mapped_key = dna_map.get(dna_key, dna_key)
                    val = style_dna.get(mapped_key, 0)
                    if lo <= val <= hi:
                        score += 1.5
                elif threshold is True:
                    if behavior == "high_variance" and style_dna.get("length_variance", 0) > 50:
                        score += 2.0
                    elif behavior == "emotional_swings":
                        expr = style_dna.get("emotional_expressiveness", 0)
                        if expr > 0.03 and style_dna.get("length_variance", 0) > 30:
                            score += 2.0
                    elif behavior == "short_response_gap":
                        if style_dna.get("rapid_fire_ratio", 0) > 0.4:
                            score += 1.5

        scores[style] = round(score, 2)

    # Normalize
    total = sum(scores.values()) or 1.0
    for k in scores:
        scores[k] = round(scores[k] / total, 3)

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = ranked[0]
    secondary = ranked[1] if len(ranked) > 1 else ("unknown", 0.0)

    return {
        "primary": primary[0],
        "primary_confidence": primary[1],
        "secondary": secondary[0],
        "secondary_confidence": secondary[1],
        "scores": scores,
    }


# ═══════════════════════════════════════════════════════════════
#  7. FULL PERSONALITY PROFILE
# ═══════════════════════════════════════════════════════════════

# In-memory profile cache
_profile_cache: Dict[int, Dict] = {}


def build_personality_profile(
    chat_id: int, their_messages: List[str]
) -> Dict[str, Any]:
    """
    Build a complete personality profile from message history.
    Caches results and updates incrementally.
    """
    if not their_messages:
        return _empty_profile()

    # Score all dimensions
    big_five = score_big_five(their_messages)
    dark_triad = score_dark_triad(their_messages)
    style_dna = extract_style_dna(their_messages)
    attachment = detect_attachment_style(their_messages, style_dna)

    # Derive personality archetype
    archetype = _derive_archetype(big_five, dark_triad, style_dna)

    # Derive communication preferences
    comm_prefs = _derive_communication_preferences(big_five, style_dna, attachment)

    profile = {
        "chat_id": chat_id,
        "big_five": big_five,
        "dark_triad": dark_triad,
        "style_dna": style_dna,
        "attachment_style": attachment,
        "archetype": archetype,
        "communication_preferences": comm_prefs,
        "messages_analyzed": len(their_messages),
        "last_updated": time.time(),
    }

    # Cache
    _profile_cache[chat_id] = profile

    # Save to disk
    _save_profile(chat_id, profile)

    return profile


def _empty_profile() -> Dict[str, Any]:
    return {
        "big_five": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                      "agreeableness": 0.5, "neuroticism": 0.5},
        "dark_triad": {"machiavellianism": 0.0, "narcissism": 0.0, "psychopathy": 0.0},
        "style_dna": _empty_style_dna(),
        "attachment_style": {"primary": "unknown", "confidence": 0.0},
        "archetype": "unknown",
        "communication_preferences": {},
        "messages_analyzed": 0,
    }


def _derive_archetype(
    big_five: Dict[str, float],
    dark_triad: Dict[str, float],
    style_dna: Dict[str, Any],
) -> str:
    """Derive a human-readable personality archetype label."""
    o = big_five.get("openness", 0.5)
    c = big_five.get("conscientiousness", 0.5)
    e = big_five.get("extraversion", 0.5)
    a = big_five.get("agreeableness", 0.5)
    n = big_five.get("neuroticism", 0.5)

    mach = dark_triad.get("machiavellianism", 0)
    narc = dark_triad.get("narcissism", 0)

    # Check for strong dark triad first
    if mach > 0.6 and narc > 0.6:
        return "power_player"
    if narc > 0.7:
        return "spotlight_seeker"
    if mach > 0.7:
        return "strategic_operator"

    # Big Five archetypes
    if e > 0.7 and a > 0.6 and o > 0.6:
        return "social_butterfly"
    if e > 0.7 and a < 0.4:
        return "dominant_extrovert"
    if e < 0.3 and o > 0.6:
        return "quiet_thinker"
    if e < 0.3 and a > 0.6:
        return "gentle_introvert"
    if c > 0.7 and n < 0.3:
        return "steady_achiever"
    if n > 0.7 and a > 0.6:
        return "sensitive_soul"
    if n > 0.7 and a < 0.4:
        return "volatile_reactor"
    if o > 0.7 and c < 0.3:
        return "free_spirit"
    if a > 0.7 and c > 0.6:
        return "reliable_nurturer"
    if a < 0.3 and e > 0.5:
        return "challenger"

    return "balanced"


def _derive_communication_preferences(
    big_five: Dict[str, float],
    style_dna: Dict[str, Any],
    attachment: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive how this person prefers to communicate."""
    prefs = {}

    e = big_five.get("extraversion", 0.5)
    a = big_five.get("agreeableness", 0.5)
    n = big_five.get("neuroticism", 0.5)
    o = big_five.get("openness", 0.5)

    # Message length preference
    if style_dna.get("avg_msg_length", 0) < 5:
        prefs["preferred_length"] = "very_short"
    elif style_dna.get("avg_msg_length", 0) < 12:
        prefs["preferred_length"] = "short"
    elif style_dna.get("avg_msg_length", 0) < 25:
        prefs["preferred_length"] = "medium"
    else:
        prefs["preferred_length"] = "long"

    # Tone preference
    if style_dna.get("slang_density", 0) > 0.05:
        prefs["tone"] = "casual_slang"
    elif style_dna.get("formality", 0) > 0.5:
        prefs["tone"] = "formal"
    elif e > 0.6:
        prefs["tone"] = "energetic_casual"
    else:
        prefs["tone"] = "relaxed_casual"

    # Emoji preference
    emoji_d = style_dna.get("emoji_density", 0)
    if emoji_d > 1.0:
        prefs["emoji_level"] = "heavy"
    elif emoji_d > 0.3:
        prefs["emoji_level"] = "moderate"
    elif emoji_d > 0.05:
        prefs["emoji_level"] = "light"
    else:
        prefs["emoji_level"] = "minimal"

    # Depth preference
    if o > 0.6 and style_dna.get("avg_msg_length", 0) > 15:
        prefs["depth"] = "deep"
    elif o < 0.4 and style_dna.get("avg_msg_length", 0) < 8:
        prefs["depth"] = "surface"
    else:
        prefs["depth"] = "moderate"

    # Response speed expectation
    if style_dna.get("rapid_fire_ratio", 0) > 0.5:
        prefs["speed_expectation"] = "instant"
    elif style_dna.get("rapid_fire_ratio", 0) > 0.3:
        prefs["speed_expectation"] = "quick"
    else:
        prefs["speed_expectation"] = "relaxed"

    # Conflict style
    att = attachment.get("primary", "unknown")
    if a < 0.3:
        prefs["conflict_style"] = "confrontational"
    elif att == "avoidant":
        prefs["conflict_style"] = "withdrawing"
    elif att == "anxious":
        prefs["conflict_style"] = "escalating"
    elif a > 0.7:
        prefs["conflict_style"] = "accommodating"
    else:
        prefs["conflict_style"] = "direct"

    # Validation need
    if n > 0.6 or att == "anxious":
        prefs["validation_need"] = "high"
    elif n < 0.3 and att == "secure":
        prefs["validation_need"] = "low"
    else:
        prefs["validation_need"] = "moderate"

    return prefs


# ═══════════════════════════════════════════════════════════════
#  8. PERSONALITY EVOLUTION TRACKING
# ═══════════════════════════════════════════════════════════════

_evolution_cache: Dict[int, List[Dict]] = {}


def record_personality_snapshot(chat_id: int, profile: Dict[str, Any]) -> None:
    """Record a timestamped personality snapshot for evolution tracking."""
    snapshot = {
        "timestamp": time.time(),
        "big_five": profile.get("big_five", {}),
        "dark_triad": profile.get("dark_triad", {}),
        "archetype": profile.get("archetype", "unknown"),
        "attachment": profile.get("attachment_style", {}).get("primary", "unknown"),
        "messages_analyzed": profile.get("messages_analyzed", 0),
    }

    if chat_id not in _evolution_cache:
        _evolution_cache[chat_id] = _load_evolution(chat_id)

    _evolution_cache[chat_id].append(snapshot)

    # Keep last 200 snapshots
    if len(_evolution_cache[chat_id]) > 200:
        _evolution_cache[chat_id] = _evolution_cache[chat_id][-200:]

    _save_evolution(chat_id)


def get_personality_evolution(chat_id: int) -> Dict[str, Any]:
    """
    Analyze how personality has evolved over time.
    Returns trends, shifts, and stability metrics.
    """
    if chat_id not in _evolution_cache:
        _evolution_cache[chat_id] = _load_evolution(chat_id)

    snapshots = _evolution_cache[chat_id]
    if len(snapshots) < 2:
        return {"status": "insufficient_data", "snapshots": len(snapshots)}

    # Compare earliest third vs latest third
    third = max(len(snapshots) // 3, 1)
    early = snapshots[:third]
    recent = snapshots[-third:]

    trends = {}
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        early_avg = sum(s["big_five"].get(trait, 0.5) for s in early) / len(early)
        recent_avg = sum(s["big_five"].get(trait, 0.5) for s in recent) / len(recent)
        delta = recent_avg - early_avg
        if abs(delta) > 0.1:
            direction = "increasing" if delta > 0 else "decreasing"
        else:
            direction = "stable"
        trends[trait] = {
            "early_avg": round(early_avg, 3),
            "recent_avg": round(recent_avg, 3),
            "delta": round(delta, 3),
            "direction": direction,
        }

    # Archetype stability
    early_archetypes = [s["archetype"] for s in early]
    recent_archetypes = [s["archetype"] for s in recent]
    archetype_stable = len(set(recent_archetypes)) <= 2

    # Attachment shift
    early_att = [s.get("attachment", "unknown") for s in early]
    recent_att = [s.get("attachment", "unknown") for s in recent]

    return {
        "status": "analyzed",
        "total_snapshots": len(snapshots),
        "time_span_hours": round((snapshots[-1]["timestamp"] - snapshots[0]["timestamp"]) / 3600, 1),
        "big_five_trends": trends,
        "archetype_stable": archetype_stable,
        "early_archetype": max(set(early_archetypes), key=early_archetypes.count),
        "recent_archetype": max(set(recent_archetypes), key=recent_archetypes.count),
        "early_attachment": max(set(early_att), key=early_att.count),
        "recent_attachment": max(set(recent_att), key=recent_att.count),
    }


# ═══════════════════════════════════════════════════════════════
#  9. COMPATIBILITY SCORING
# ═══════════════════════════════════════════════════════════════

def compute_compatibility(
    their_profile: Dict[str, Any],
    bot_persona: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute compatibility between target's personality and bot persona.
    Uses complementarity theory (some traits attract, some repel).
    """
    # Default bot persona (slightly high E, high A, moderate everything else)
    if bot_persona is None:
        bot_persona = {
            "openness": 0.65, "conscientiousness": 0.5, "extraversion": 0.6,
            "agreeableness": 0.7, "neuroticism": 0.3,
        }

    their_b5 = their_profile.get("big_five", {})
    if not their_b5:
        return {"overall": 0.5, "details": {}}

    details = {}
    weighted_sum = 0
    total_weight = 0

    # Complementarity rules
    rules = {
        "openness": {"mode": "similar", "weight": 1.0},        # similar O = good
        "conscientiousness": {"mode": "similar", "weight": 0.8},
        "extraversion": {"mode": "complement", "weight": 1.2},  # opposites can attract
        "agreeableness": {"mode": "both_high", "weight": 1.5},  # both high = harmony
        "neuroticism": {"mode": "both_low", "weight": 1.3},     # both low = stable
    }

    for trait, rule in rules.items():
        their_val = their_b5.get(trait, 0.5)
        bot_val = bot_persona.get(trait, 0.5)
        weight = rule["weight"]

        if rule["mode"] == "similar":
            score = 1.0 - abs(their_val - bot_val)
        elif rule["mode"] == "complement":
            # Some difference is attractive, too much is bad
            diff = abs(their_val - bot_val)
            score = 1.0 - abs(diff - 0.3)  # optimal diff is ~0.3
        elif rule["mode"] == "both_high":
            score = (their_val + bot_val) / 2
        elif rule["mode"] == "both_low":
            score = 1.0 - (their_val + bot_val) / 2
        else:
            score = 0.5

        score = max(0.0, min(1.0, score))
        details[trait] = {"score": round(score, 3), "weight": weight}
        weighted_sum += score * weight
        total_weight += weight

    overall = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.5

    return {
        "overall": overall,
        "label": _compatibility_label(overall),
        "details": details,
    }


def _compatibility_label(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    elif score >= 0.65:
        return "good"
    elif score >= 0.5:
        return "moderate"
    elif score >= 0.35:
        return "challenging"
    else:
        return "poor"


# ═══════════════════════════════════════════════════════════════
#  10. ADAPTIVE PERSONA TUNING
# ═══════════════════════════════════════════════════════════════

def generate_persona_adjustments(
    their_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate specific persona adjustments for the bot based on
    the target's personality. Returns prompt injection directives.
    """
    prefs = their_profile.get("communication_preferences", {})
    b5 = their_profile.get("big_five", {})
    dark = their_profile.get("dark_triad", {})
    att = their_profile.get("attachment_style", {})
    archetype = their_profile.get("archetype", "unknown")

    directives = []

    # --- Length matching ---
    pref_len = prefs.get("preferred_length", "short")
    if pref_len == "very_short":
        directives.append("Keep responses VERY short (1-5 words). They text in fragments.")
    elif pref_len == "short":
        directives.append("Keep responses short (5-12 words). Match their brief style.")
    elif pref_len == "long":
        directives.append("They write longer messages — you can match with moderate length (15-30 words).")

    # --- Tone matching ---
    tone = prefs.get("tone", "relaxed_casual")
    if tone == "casual_slang":
        directives.append("Use heavy slang and abbreviations. They're very informal.")
    elif tone == "formal":
        directives.append("They're more formal. Reduce slang, be articulate.")
    elif tone == "energetic_casual":
        directives.append("Match their high energy. Exclamation marks ok. Be enthusiastic.")

    # --- Emoji matching ---
    emoji_lvl = prefs.get("emoji_level", "light")
    if emoji_lvl == "heavy":
        directives.append("They use lots of emojis — mirror with moderate emoji use.")
    elif emoji_lvl == "minimal":
        directives.append("They rarely use emojis — keep yours minimal too.")

    # --- Depth matching ---
    depth = prefs.get("depth", "moderate")
    if depth == "deep":
        directives.append("They enjoy deeper topics. Don't shy away from meaningful convos.")
    elif depth == "surface":
        directives.append("Keep it light. They prefer surface-level banter.")

    # --- Dark triad adjustments ---
    if dark.get("narcissism", 0) > 0.5:
        directives.append("They have narcissistic tendencies — don't be a pushover but acknowledge their ego occasionally.")
    if dark.get("machiavellianism", 0) > 0.5:
        directives.append("They're strategic/manipulative. Be sharp, don't get played.")

    # --- Attachment-based adjustments ---
    att_primary = att.get("primary", "unknown")
    if att_primary == "anxious":
        directives.append("They get anxious about communication gaps. Be responsive and reassuring when needed, but dont overdo it.")
    elif att_primary == "avoidant":
        directives.append("They need space. Don't be clingy or overly warm. Match their emotional distance.")
    elif att_primary == "disorganized":
        directives.append("Their attachment is chaotic — hot and cold. Stay steady, dont mirror their chaos.")

    # --- Speed expectation ---
    speed = prefs.get("speed_expectation", "relaxed")
    if speed == "instant":
        directives.append("They expect fast replies. Keep the energy up.")

    # --- Archetype-specific ---
    archetype_tips = {
        "social_butterfly": "They thrive on social energy. Be fun and engaging.",
        "dominant_extrovert": "They like to lead. Let them sometimes, but stand your ground.",
        "quiet_thinker": "They value thoughtful exchanges. Quality over quantity.",
        "gentle_introvert": "Be warm but not overwhelming. Respect their pace.",
        "volatile_reactor": "They're emotionally volatile. Don't pour fuel on fire unless matching aggression.",
        "challenger": "They push back. Give them something to push against.",
        "free_spirit": "Be spontaneous and open. They hate rigidity.",
        "power_player": "They respect strength. Never show weakness.",
        "spotlight_seeker": "They want attention. Give some, but dont be a simp.",
    }
    if archetype in archetype_tips:
        directives.append(archetype_tips[archetype])

    return {
        "directives": directives,
        "archetype": archetype,
        "priority_adjustments": {
            "length": pref_len,
            "tone": tone,
            "emoji": emoji_lvl,
            "depth": depth,
            "speed": speed,
        },
    }


# ═══════════════════════════════════════════════════════════════
#  11. FORMAT FOR PROMPT INJECTION
# ═══════════════════════════════════════════════════════════════

def format_personality_for_prompt(
    profile: Dict[str, Any],
    adjustments: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format personality analysis as a prompt injection block.
    Concise, actionable, no therapy-speak.
    """
    parts = []

    # Archetype
    archetype = profile.get("archetype", "unknown")
    if archetype != "unknown":
        parts.append(f"[THEIR PERSONALITY TYPE: {archetype.replace('_', ' ').upper()}]")

    # Key Big Five traits (only notable ones)
    b5 = profile.get("big_five", {})
    notable = []
    for trait, val in b5.items():
        if val > 0.65:
            notable.append(f"high {trait}")
        elif val < 0.35:
            notable.append(f"low {trait}")
    if notable:
        parts.append(f"Key traits: {', '.join(notable)}")

    # Dark triad warning
    dark = profile.get("dark_triad", {})
    high_dark = [t for t, v in dark.items() if v > 0.5]
    if high_dark:
        parts.append(f"WARNING — dark traits detected: {', '.join(high_dark)}. Stay sharp.")

    # Attachment
    att = profile.get("attachment_style", {})
    if att.get("primary") and att["primary"] != "unknown":
        parts.append(f"Attachment: {att['primary']} (confidence: {att.get('primary_confidence', 0):.0%})")

    # Communication preferences
    prefs = profile.get("communication_preferences", {})
    if prefs:
        pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items() if v)
        parts.append(f"Their comms style: {pref_str}")

    # Persona adjustments (directives)
    if adjustments:
        directives = adjustments.get("directives", [])
        if directives:
            parts.append("ADAPT YOUR STYLE:")
            for d in directives[:6]:  # cap at 6 to avoid prompt bloat
                parts.append(f"  → {d}")

    if not parts:
        return ""

    return "\n## PERSONALITY PROFILE\n" + "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  12. DISK PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _save_profile(chat_id: int, profile: Dict[str, Any]) -> None:
    try:
        path = PROFILES_DIR / f"{chat_id}.json"
        # Make JSON-serializable
        serializable = json.loads(json.dumps(profile, default=str))
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        personality_logger.warning(f"Failed to save profile for {chat_id}: {e}")


def load_profile(chat_id: int) -> Optional[Dict[str, Any]]:
    """Load a cached personality profile from disk."""
    if chat_id in _profile_cache:
        return _profile_cache[chat_id]
    try:
        path = PROFILES_DIR / f"{chat_id}.json"
        if path.exists():
            with open(path) as f:
                profile = json.load(f)
            _profile_cache[chat_id] = profile
            return profile
    except Exception as e:
        personality_logger.warning(f"Failed to load profile for {chat_id}: {e}")
    return None


def _save_evolution(chat_id: int) -> None:
    try:
        path = EVOLUTION_DIR / f"{chat_id}.json"
        data = _evolution_cache.get(chat_id, [])
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        personality_logger.warning(f"Failed to save evolution for {chat_id}: {e}")


def _load_evolution(chat_id: int) -> List[Dict]:
    try:
        path = EVOLUTION_DIR / f"{chat_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        personality_logger.warning(f"Failed to load evolution for {chat_id}: {e}")
    return []


# ═══════════════════════════════════════════════════════════════
#  13. CONVENIENCE: FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def analyze_personality(
    chat_id: int, their_messages: List[str]
) -> Tuple[Dict[str, Any], str]:
    """
    Full pipeline: build profile → generate adjustments → format for prompt.
    Returns (profile_dict, prompt_injection_string).
    """
    profile = build_personality_profile(chat_id, their_messages)

    # Record evolution snapshot (every 10 messages)
    if profile.get("messages_analyzed", 0) % 10 == 0:
        record_personality_snapshot(chat_id, profile)

    adjustments = generate_persona_adjustments(profile)
    prompt_block = format_personality_for_prompt(profile, adjustments)

    return profile, prompt_block
