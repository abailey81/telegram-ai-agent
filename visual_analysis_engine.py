"""
Advanced Visual Analysis Engine
=================================
Deep contextual understanding of images, GIFs, stickers, and video
thumbnails. Goes beyond "what is in the image" to understand
"what does sending this MEAN in this conversation context."

Features:
1. Sticker Emotion Decoding — maps sticker packs + emojis to emotional intent
2. GIF Sentiment Analysis — understands meme/reaction GIFs contextually
3. Image Context Analysis — what sending this photo means (selfie = seeking validation, etc.)
4. Media Pattern Tracking — track what types of media they send over time
5. Multi-modal Conversation Analysis — combine text + media for deeper understanding
6. Visual Reply Intelligence — suggest appropriate media responses
7. Meme/Reference Detection — detect meme formats and cultural references
"""

import json
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

visual_logger = logging.getLogger("visual_analysis")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

VISUAL_DATA_DIR = Path("engine_data/visual")
VISUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_PATTERNS_DIR = VISUAL_DATA_DIR / "patterns"
MEDIA_PATTERNS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. STICKER EMOTION DECODING
# ═══════════════════════════════════════════════════════════════

# Emoji → emotional intent mapping (sticker alt text is often an emoji)
STICKER_EMOJI_INTENT = {
    # Positive
    "❤️": {"emotion": "love", "intent": "affection", "energy": "warm", "weight": 0.9},
    "😍": {"emotion": "love", "intent": "attraction", "energy": "hot", "weight": 0.85},
    "🥰": {"emotion": "love", "intent": "adoration", "energy": "warm", "weight": 0.85},
    "😘": {"emotion": "love", "intent": "flirting", "energy": "warm", "weight": 0.8},
    "💕": {"emotion": "love", "intent": "affection", "energy": "warm", "weight": 0.75},
    "🥺": {"emotion": "pleading", "intent": "seeking_sympathy", "energy": "soft", "weight": 0.7},
    "🤗": {"emotion": "warmth", "intent": "comfort", "energy": "warm", "weight": 0.7},

    # Humor
    "😂": {"emotion": "amusement", "intent": "laughing", "energy": "high", "weight": 0.8},
    "🤣": {"emotion": "amusement", "intent": "laughing_hard", "energy": "high", "weight": 0.85},
    "💀": {"emotion": "amusement", "intent": "dead_laughing", "energy": "high", "weight": 0.9},
    "😭": {"emotion": "amused_crying", "intent": "overwhelmed_funny", "energy": "high", "weight": 0.8},
    "🤡": {"emotion": "self_deprecation", "intent": "clowning", "energy": "medium", "weight": 0.7},

    # Negative
    "😤": {"emotion": "frustration", "intent": "annoyed", "energy": "hot", "weight": 0.75},
    "😡": {"emotion": "anger", "intent": "angry", "energy": "boiling", "weight": 0.85},
    "🙄": {"emotion": "contempt", "intent": "eye_roll", "energy": "cold", "weight": 0.8},
    "😒": {"emotion": "displeasure", "intent": "unimpressed", "energy": "cold", "weight": 0.75},
    "😐": {"emotion": "neutral", "intent": "deadpan", "energy": "flat", "weight": 0.6},
    "😑": {"emotion": "annoyance", "intent": "not_amused", "energy": "cold", "weight": 0.7},

    # Surprise/shock
    "😱": {"emotion": "shock", "intent": "shocked", "energy": "high", "weight": 0.8},
    "😳": {"emotion": "embarrassment", "intent": "flustered", "energy": "medium", "weight": 0.7},
    "👀": {"emotion": "curiosity", "intent": "watching", "energy": "medium", "weight": 0.65},
    "🤯": {"emotion": "amazement", "intent": "mind_blown", "energy": "high", "weight": 0.8},

    # Flirty/suggestive
    "😏": {"emotion": "suggestive", "intent": "flirty_smirk", "energy": "warm", "weight": 0.8},
    "🔥": {"emotion": "attraction", "intent": "hot_compliment", "energy": "hot", "weight": 0.85},
    "😈": {"emotion": "mischief", "intent": "playfully_evil", "energy": "hot", "weight": 0.75},
    "💋": {"emotion": "romantic", "intent": "kiss", "energy": "warm", "weight": 0.85},

    # Neutral/acknowledgment
    "👍": {"emotion": "agreement", "intent": "acknowledgment", "energy": "low", "weight": 0.4},
    "👌": {"emotion": "agreement", "intent": "ok", "energy": "low", "weight": 0.4},
    "🤷": {"emotion": "indifference", "intent": "whatever", "energy": "flat", "weight": 0.5},
    "✅": {"emotion": "completion", "intent": "done", "energy": "low", "weight": 0.3},

    # Sadness
    "😢": {"emotion": "sadness", "intent": "crying", "energy": "low", "weight": 0.75},
    "😔": {"emotion": "sadness", "intent": "disappointed", "energy": "low", "weight": 0.7},
    "💔": {"emotion": "heartbreak", "intent": "hurt", "energy": "low", "weight": 0.85},
}


def decode_sticker_intent(
    sticker_emoji: Optional[str] = None,
    sticker_set_name: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decode the emotional intent behind a sticker.
    """
    result = {
        "emotion": "unknown",
        "intent": "unknown",
        "energy": "medium",
        "confidence": 0.3,
        "contextual_meaning": "",
    }

    # Decode from emoji alt text
    if sticker_emoji and sticker_emoji in STICKER_EMOJI_INTENT:
        mapping = STICKER_EMOJI_INTENT[sticker_emoji]
        result.update(mapping)
        result["confidence"] = mapping.get("weight", 0.5)

    # Contextual meaning based on conversation
    if conversation_context and result["emotion"] != "unknown":
        ctx_lower = conversation_context.lower()
        # Sarcastic usage detection
        if result["emotion"] == "amusement" and any(w in ctx_lower for w in ("serious", "not funny", "angry", "mad")):
            result["contextual_meaning"] = "possibly_sarcastic_laugh"
            result["energy"] = "cold"
        elif result["emotion"] == "love" and any(w in ctx_lower for w in ("fight", "argue", "angry", "mad")):
            result["contextual_meaning"] = "making_up_attempt"
        elif result["intent"] == "eye_roll" and any(w in ctx_lower for w in ("joke", "haha", "lol")):
            result["contextual_meaning"] = "playful_eye_roll"
            result["energy"] = "warm"

    # Set name can give additional context
    if sticker_set_name:
        set_lower = sticker_set_name.lower() if sticker_set_name else ""
        if any(w in set_lower for w in ("love", "heart", "couple", "romantic")):
            result["sticker_category"] = "romantic"
        elif any(w in set_lower for w in ("angry", "rage", "mad")):
            result["sticker_category"] = "anger"
        elif any(w in set_lower for w in ("meme", "pepe", "troll", "doge")):
            result["sticker_category"] = "meme"
        elif any(w in set_lower for w in ("cute", "kawaii", "soft")):
            result["sticker_category"] = "cute"

    return result


# ═══════════════════════════════════════════════════════════════
#  2. GIF SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════

# Common GIF search terms → emotional intent
GIF_KEYWORD_INTENT = {
    # Positive reactions
    "happy": {"emotion": "joy", "energy": "high", "response_type": "match_energy"},
    "excited": {"emotion": "excitement", "energy": "high", "response_type": "match_energy"},
    "love": {"emotion": "love", "energy": "warm", "response_type": "reciprocate"},
    "hug": {"emotion": "affection", "energy": "warm", "response_type": "reciprocate"},
    "kiss": {"emotion": "romantic", "energy": "warm", "response_type": "reciprocate"},
    "dance": {"emotion": "joy", "energy": "high", "response_type": "celebrate"},
    "celebrate": {"emotion": "celebration", "energy": "high", "response_type": "celebrate"},
    "thank": {"emotion": "gratitude", "energy": "warm", "response_type": "acknowledge"},
    "welcome": {"emotion": "warmth", "energy": "medium", "response_type": "acknowledge"},

    # Humor
    "laugh": {"emotion": "amusement", "energy": "high", "response_type": "join_laughter"},
    "lol": {"emotion": "amusement", "energy": "high", "response_type": "join_laughter"},
    "funny": {"emotion": "amusement", "energy": "high", "response_type": "join_laughter"},
    "rofl": {"emotion": "amusement", "energy": "high", "response_type": "join_laughter"},
    "fail": {"emotion": "schadenfreude", "energy": "medium", "response_type": "tease"},

    # Negative
    "angry": {"emotion": "anger", "energy": "hot", "response_type": "acknowledge_emotion"},
    "mad": {"emotion": "anger", "energy": "hot", "response_type": "match_or_calm"},
    "sad": {"emotion": "sadness", "energy": "low", "response_type": "comfort"},
    "cry": {"emotion": "sadness", "energy": "low", "response_type": "comfort"},
    "frustrated": {"emotion": "frustration", "energy": "hot", "response_type": "acknowledge_emotion"},
    "annoyed": {"emotion": "annoyance", "energy": "medium", "response_type": "acknowledge_emotion"},
    "eye roll": {"emotion": "contempt", "energy": "cold", "response_type": "tease_back"},
    "whatever": {"emotion": "indifference", "energy": "flat", "response_type": "challenge"},
    "bye": {"emotion": "dismissal", "energy": "cold", "response_type": "acknowledge"},
    "done": {"emotion": "exasperation", "energy": "medium", "response_type": "acknowledge"},

    # Surprise
    "shocked": {"emotion": "shock", "energy": "high", "response_type": "explain_or_share"},
    "surprise": {"emotion": "surprise", "energy": "high", "response_type": "explain_or_share"},
    "omg": {"emotion": "shock", "energy": "high", "response_type": "join_reaction"},
    "wow": {"emotion": "amazement", "energy": "high", "response_type": "join_reaction"},
    "mind blown": {"emotion": "amazement", "energy": "high", "response_type": "join_reaction"},

    # Flirty
    "wink": {"emotion": "flirty", "energy": "warm", "response_type": "flirt_back"},
    "hot": {"emotion": "attraction", "energy": "hot", "response_type": "flirt_back"},
    "sexy": {"emotion": "attraction", "energy": "hot", "response_type": "flirt_back"},

    # Sarcasm
    "sarcasm": {"emotion": "sarcasm", "energy": "cold", "response_type": "match_sarcasm"},
    "sure": {"emotion": "skepticism", "energy": "cold", "response_type": "call_out_or_match"},
    "slow clap": {"emotion": "sarcasm", "energy": "cold", "response_type": "match_sarcasm"},
    "cool story": {"emotion": "dismissive", "energy": "cold", "response_type": "tease_back"},

    # Waiting
    "waiting": {"emotion": "impatience", "energy": "medium", "response_type": "respond_quickly"},
    "bored": {"emotion": "boredom", "energy": "flat", "response_type": "entertain"},
}


def analyze_gif_intent(
    gif_url: Optional[str] = None,
    gif_title: Optional[str] = None,
    search_query: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze the intent behind a GIF in conversation context.
    """
    result = {
        "emotion": "unknown",
        "energy": "medium",
        "response_type": "acknowledge",
        "confidence": 0.3,
        "decoded_meaning": "",
    }

    # Analyze from title or search query
    text_to_analyze = (gif_title or "") + " " + (search_query or "")
    text_lower = text_to_analyze.lower()

    best_match = None
    best_score = 0

    for keyword, intent in GIF_KEYWORD_INTENT.items():
        if keyword in text_lower:
            score = len(keyword)  # longer matches are more specific
            if score > best_score:
                best_score = score
                best_match = intent

    if best_match:
        result.update(best_match)
        result["confidence"] = min(0.9, 0.5 + best_score * 0.05)

    # Contextual adjustment
    if conversation_context:
        ctx_lower = conversation_context.lower()
        # If they send a "laugh" GIF after we said something
        if result["emotion"] == "amusement":
            result["decoded_meaning"] = "They found what you said funny — positive engagement signal"
        elif result["emotion"] in ("anger", "frustration"):
            if any(w in ctx_lower for w in ("sorry", "my bad", "apologize")):
                result["decoded_meaning"] = "Still angry despite apology — needs more"
            else:
                result["decoded_meaning"] = "Expressing frustration through GIF — take seriously"
        elif result["emotion"] in ("love", "romantic"):
            result["decoded_meaning"] = "Affectionate intent — reciprocate appropriately"
        elif result["emotion"] == "sarcasm":
            result["decoded_meaning"] = "Being sarcastic — match energy or call it out"

    return result


# ═══════════════════════════════════════════════════════════════
#  3. IMAGE CONTEXT ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_image_context(
    caption: str = "",
    is_selfie: bool = False,
    has_face: bool = False,
    is_screenshot: bool = False,
    is_food: bool = False,
    is_nature: bool = False,
    is_meme: bool = False,
    conversation_context: Optional[str] = None,
    time_of_day: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze the contextual meaning of sending an image.
    """
    analysis = {
        "intent": "sharing",
        "emotion": "neutral",
        "energy": "medium",
        "response_strategy": "",
        "significance": "normal",
        "decoded_meaning": "",
    }

    caption_lower = caption.lower() if caption else ""

    # Selfie analysis
    if is_selfie or has_face:
        analysis["intent"] = "seeking_validation"
        analysis["emotion"] = "confident"
        analysis["energy"] = "warm"
        analysis["response_strategy"] = "compliment_genuinely"
        analysis["significance"] = "high"
        analysis["decoded_meaning"] = (
            "Sending a selfie = seeking your reaction. "
            "Compliment specifically (not just 'nice'), react with 🔥 or 😍."
        )

        # Context modifiers for selfie
        if any(w in caption_lower for w in ("new hair", "new outfit", "dressed up", "going out")):
            analysis["decoded_meaning"] += " They made an effort — acknowledge it specifically."
        if any(w in caption_lower for w in ("bored", "just me", "nothing")):
            analysis["decoded_meaning"] += " Casual selfie = wanting attention/connection."
        if time_of_day and (22 <= time_of_day or time_of_day < 3):
            analysis["decoded_meaning"] += " Late night selfie = high personal engagement signal."
            analysis["significance"] = "very_high"

    # Screenshot analysis
    elif is_screenshot:
        analysis["intent"] = "sharing_reference"
        analysis["emotion"] = "neutral"
        analysis["energy"] = "medium"
        analysis["response_strategy"] = "engage_with_content"
        analysis["decoded_meaning"] = (
            "Screenshot = sharing something specific. React to the CONTENT, not the fact they screenshotted."
        )
        if any(w in caption_lower for w in ("look", "check this", "wtf", "omg")):
            analysis["energy"] = "high"
            analysis["decoded_meaning"] += " They're excited/shocked about it — match energy."

    # Food photo
    elif is_food:
        analysis["intent"] = "sharing_experience"
        analysis["emotion"] = "content"
        analysis["energy"] = "warm"
        analysis["response_strategy"] = "engage_naturally"
        analysis["decoded_meaning"] = (
            "Food photo = sharing daily life. React naturally ('looks good' / 'im hungry now')."
        )

    # Nature/scenery
    elif is_nature:
        analysis["intent"] = "sharing_mood"
        analysis["emotion"] = "peaceful"
        analysis["energy"] = "calm"
        analysis["response_strategy"] = "appreciate_and_connect"
        analysis["decoded_meaning"] = (
            "Nature/scenery = sharing a moment/mood. Appreciate it, maybe connect it to shared interests."
        )

    # Meme
    elif is_meme:
        analysis["intent"] = "humor_bonding"
        analysis["emotion"] = "playful"
        analysis["energy"] = "high"
        analysis["response_strategy"] = "laugh_and_engage"
        analysis["significance"] = "bonding"
        analysis["decoded_meaning"] = (
            "Meme = trying to make you laugh or share humor. Laugh (react 😂/💀) "
            "and engage — maybe send one back or comment on it."
        )
        if any(w in caption_lower for w in ("us", "me and you", "this is us", "literally")):
            analysis["decoded_meaning"] += " They're comparing your dynamic to the meme — HUGE bonding signal."
            analysis["significance"] = "very_high"

    # Generic image with caption
    else:
        if caption:
            if "?" in caption:
                analysis["intent"] = "seeking_opinion"
                analysis["response_strategy"] = "answer_and_engage"
            elif any(w in caption_lower for w in ("miss", "wish", "thinking")):
                analysis["intent"] = "expressing_longing"
                analysis["energy"] = "warm"
                analysis["response_strategy"] = "reciprocate_emotion"
        else:
            analysis["intent"] = "sharing"
            analysis["response_strategy"] = "react_and_comment"
            analysis["decoded_meaning"] = "Image without caption — they want your reaction to the visual."

    return analysis


# ═══════════════════════════════════════════════════════════════
#  4. MEDIA PATTERN TRACKING
# ═══════════════════════════════════════════════════════════════

_media_pattern_cache: Dict[int, List[Dict]] = {}


def record_media_event(
    chat_id: int,
    media_type: str,
    analysis: Dict[str, Any],
    sender: str = "Them",
) -> None:
    """Record a media event for pattern analysis."""
    if chat_id not in _media_pattern_cache:
        _media_pattern_cache[chat_id] = []

    _media_pattern_cache[chat_id].append({
        "timestamp": time.time(),
        "type": media_type,
        "sender": sender,
        "emotion": analysis.get("emotion", "unknown"),
        "intent": analysis.get("intent", "unknown"),
        "energy": analysis.get("energy", "medium"),
    })

    # Keep last 200
    if len(_media_pattern_cache[chat_id]) > 200:
        _media_pattern_cache[chat_id] = _media_pattern_cache[chat_id][-200:]


def analyze_media_patterns(chat_id: int) -> Dict[str, Any]:
    """Analyze media usage patterns over time."""
    events = _media_pattern_cache.get(chat_id, [])
    if not events:
        return {"status": "no_data"}

    their_events = [e for e in events if e["sender"] == "Them"]
    if not their_events:
        return {"status": "no_their_media"}

    # Type distribution
    type_counts = Counter(e["type"] for e in their_events)
    total = len(their_events)

    # Emotion distribution from media
    emotion_counts = Counter(e["emotion"] for e in their_events)

    # Energy distribution
    energy_counts = Counter(e["energy"] for e in their_events)

    # Frequency analysis
    if len(their_events) >= 2:
        gaps = []
        for i in range(1, len(their_events)):
            gap = their_events[i]["timestamp"] - their_events[i-1]["timestamp"]
            if 0 < gap < 86400 * 7:  # within a week
                gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
    else:
        avg_gap = 0

    # Trend: are they sending more media recently?
    if len(their_events) >= 10:
        first_half_ts = their_events[len(their_events)//2]["timestamp"] - their_events[0]["timestamp"]
        first_half_count = len(their_events) // 2
        second_half_ts = their_events[-1]["timestamp"] - their_events[len(their_events)//2]["timestamp"]
        second_half_count = len(their_events) - len(their_events) // 2

        first_rate = first_half_count / max(first_half_ts, 1) * 3600
        second_rate = second_half_count / max(second_half_ts, 1) * 3600

        if second_rate > first_rate * 1.5:
            media_trend = "increasing"
        elif second_rate < first_rate * 0.5:
            media_trend = "decreasing"
        else:
            media_trend = "stable"
    else:
        media_trend = "insufficient_data"

    # Selfie frequency (strong engagement signal)
    selfie_events = [e for e in their_events if e.get("intent") in ("seeking_validation", "selfie")]
    selfie_ratio = len(selfie_events) / max(total, 1)

    return {
        "status": "analyzed",
        "total_media": total,
        "type_distribution": dict(type_counts),
        "emotion_distribution": dict(emotion_counts),
        "energy_distribution": dict(energy_counts),
        "avg_gap_hours": round(avg_gap / 3600, 1) if avg_gap > 0 else None,
        "media_trend": media_trend,
        "selfie_ratio": round(selfie_ratio, 3),
        "most_common_type": type_counts.most_common(1)[0][0] if type_counts else None,
        "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else None,
    }


# ═══════════════════════════════════════════════════════════════
#  5. MULTI-MODAL CONVERSATION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_multimodal_context(
    text_messages: List[Dict],
    media_events: List[Dict],
    current_media_type: Optional[str] = None,
    current_media_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Combine text and media analysis for deeper understanding.
    """
    analysis = {
        "conversation_mode": "text_primary",
        "media_role": "supplementary",
        "emotional_coherence": "unknown",
        "engagement_signal": "neutral",
        "recommended_response_mode": "text",
    }

    if not media_events:
        return analysis

    recent_media = media_events[-5:] if media_events else []
    recent_text = text_messages[-5:] if text_messages else []

    # Determine conversation mode
    total_recent = len(recent_media) + len(recent_text)
    if total_recent > 0:
        media_ratio = len(recent_media) / total_recent
        if media_ratio > 0.6:
            analysis["conversation_mode"] = "media_heavy"
            analysis["recommended_response_mode"] = "mixed"  # text + reaction + maybe media back
        elif media_ratio > 0.3:
            analysis["conversation_mode"] = "mixed"
            analysis["recommended_response_mode"] = "text_with_reaction"

    # Emotional coherence: does media emotion match text emotion?
    if recent_media and recent_text:
        media_emotions = [e.get("emotion", "") for e in recent_media]
        text_emotions = []
        for msg in recent_text:
            t = msg.get("text", "").lower()
            if any(w in t for w in ("happy", "love", "great", "amazing", "haha", "lol")):
                text_emotions.append("positive")
            elif any(w in t for w in ("angry", "mad", "hate", "annoyed", "frustrated")):
                text_emotions.append("negative")
            elif any(w in t for w in ("sad", "cry", "miss", "lonely")):
                text_emotions.append("sad")
            else:
                text_emotions.append("neutral")

        # Check coherence
        media_positive = sum(1 for e in media_emotions if e in ("joy", "love", "amusement", "excitement", "romantic"))
        media_negative = sum(1 for e in media_emotions if e in ("anger", "sadness", "frustration", "contempt"))
        text_positive = text_emotions.count("positive")
        text_negative = text_emotions.count("negative")

        if (media_positive > media_negative and text_positive > text_negative):
            analysis["emotional_coherence"] = "coherent_positive"
        elif (media_negative > media_positive and text_negative > text_positive):
            analysis["emotional_coherence"] = "coherent_negative"
        elif (media_positive > media_negative and text_negative > text_positive):
            analysis["emotional_coherence"] = "incoherent"
            analysis["note"] = "Media is positive but text is negative — possible masking or sarcasm"
        elif (media_negative > media_positive and text_positive > text_negative):
            analysis["emotional_coherence"] = "incoherent"
            analysis["note"] = "Text is positive but media is negative — complex emotion state"
        else:
            analysis["emotional_coherence"] = "neutral"

    # Engagement signal from current media
    if current_media_analysis:
        sig = current_media_analysis.get("significance", "normal")
        if sig in ("high", "very_high"):
            analysis["engagement_signal"] = "high"
        intent = current_media_analysis.get("intent", "")
        if intent == "seeking_validation":
            analysis["engagement_signal"] = "very_high"
            analysis["recommended_response_mode"] = "enthusiastic_reaction"

    return analysis


# ═══════════════════════════════════════════════════════════════
#  6. VISUAL REPLY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

def suggest_media_response(
    media_analysis: Dict[str, Any],
    conversation_energy: str = "medium",
    personality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Suggest what media to respond with (if any).
    """
    suggestion = {
        "should_send_media": False,
        "media_type": None,
        "content_hint": "",
        "emoji_reaction": None,
    }

    intent = media_analysis.get("intent", "unknown")
    emotion = media_analysis.get("emotion", "unknown")
    energy = media_analysis.get("energy", "medium")

    # Selfie → compliment + reaction
    if intent == "seeking_validation":
        suggestion["emoji_reaction"] = random.choice(["🔥", "😍", "❤️"])
        suggestion["content_hint"] = "React with fire/heart emoji, then compliment specifically"
        suggestion["should_send_media"] = random.random() < 0.2  # sometimes send selfie back
        if suggestion["should_send_media"]:
            suggestion["media_type"] = "selfie_back"

    # Meme → laugh + maybe meme back
    elif intent == "humor_bonding":
        suggestion["emoji_reaction"] = random.choice(["😂", "💀", "🤣"])
        if random.random() < 0.3:
            suggestion["should_send_media"] = True
            suggestion["media_type"] = "gif"
            suggestion["content_hint"] = "Send a funny reaction GIF"

    # Sad/emotional → comforting reaction
    elif emotion in ("sadness", "hurt", "heartbreak"):
        suggestion["emoji_reaction"] = "❤️"
        suggestion["content_hint"] = "React with heart, respond with genuine words"

    # Angry → match or acknowledge
    elif emotion in ("anger", "frustration"):
        suggestion["emoji_reaction"] = random.choice(["👀", "😬"])
        suggestion["content_hint"] = "Acknowledge their anger, dont dismiss"

    # Love/affectionate → reciprocate
    elif emotion in ("love", "romantic", "attraction"):
        suggestion["emoji_reaction"] = random.choice(["❤️", "😍", "🥰"])
        suggestion["content_hint"] = "Reciprocate the affectionate energy"

    return suggestion


# ═══════════════════════════════════════════════════════════════
#  7. UNIFIED MEDIA ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════

def analyze_visual_message(
    chat_id: int,
    media_type: str,
    caption: str = "",
    sticker_emoji: Optional[str] = None,
    sticker_set: Optional[str] = None,
    gif_title: Optional[str] = None,
    gif_query: Optional[str] = None,
    is_selfie: bool = False,
    has_face: bool = False,
    is_screenshot: bool = False,
    is_food: bool = False,
    is_nature: bool = False,
    is_meme: bool = False,
    duration: int = 0,
    is_round: bool = False,
    conversation_context: Optional[str] = None,
    personality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Unified pipeline: analyze any visual media and return
    comprehensive understanding + response guidance.
    """
    result = {
        "media_type": media_type,
        "analysis": {},
        "response_guidance": {},
        "prompt_injection": "",
    }

    # Route to appropriate analyzer
    if media_type == "sticker":
        analysis = decode_sticker_intent(sticker_emoji, sticker_set, conversation_context)
    elif media_type in ("gif", "animation"):
        analysis = analyze_gif_intent(None, gif_title, gif_query, conversation_context)
    elif media_type in ("photo", "image"):
        hour = datetime.now().hour
        analysis = analyze_image_context(
            caption, is_selfie, has_face, is_screenshot,
            is_food, is_nature, is_meme, conversation_context, hour,
        )
    elif media_type in ("voice_message", "audio"):
        analysis = {
            "intent": "personal_connection",
            "emotion": "intimate",
            "energy": "warm",
            "decoded_meaning": "Voice message = personal, intimate. Shows effort. Consider responding with voice or matching the emotional energy.",
            "significance": "high",
        }
    elif media_type in ("video", "video_message"):
        analysis = {
            "intent": "sharing_moment" if not is_round else "personal_connection",
            "emotion": "engaged",
            "energy": "high" if is_round else "medium",
            "decoded_meaning": (
                "Round video = very personal (like FaceTime)" if is_round
                else "Video = sharing a moment. React to the content."
            ),
            "significance": "high" if is_round else "normal",
        }
    else:
        analysis = {
            "intent": "sharing",
            "emotion": "neutral",
            "energy": "medium",
            "decoded_meaning": f"Media type: {media_type}. React naturally.",
        }

    result["analysis"] = analysis

    # Record for pattern tracking
    record_media_event(chat_id, media_type, analysis, "Them")

    # Get response suggestions
    result["response_guidance"] = suggest_media_response(
        analysis, analysis.get("energy", "medium"), personality,
    )

    # Build prompt injection
    parts = []
    decoded = analysis.get("decoded_meaning", "")
    if decoded:
        parts.append(f"[MEDIA INTENT: {decoded}]")

    emotion = analysis.get("emotion", "unknown")
    if emotion != "unknown":
        parts.append(f"Media emotion: {emotion}, energy: {analysis.get('energy', 'medium')}")

    response_strat = analysis.get("response_strategy", "")
    if response_strat:
        parts.append(f"Respond with: {response_strat.replace('_', ' ')}")

    sig = analysis.get("significance", "normal")
    if sig in ("high", "very_high"):
        parts.append(f"SIGNIFICANCE: {sig.upper()} — this media deserves a strong reaction")

    if parts:
        result["prompt_injection"] = "\n## VISUAL ANALYSIS\n" + "\n".join(parts)

    return result


# ═══════════════════════════════════════════════════════════════
#  8. FORMAT FOR PROMPT
# ═══════════════════════════════════════════════════════════════

def format_visual_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """Format the visual analysis for prompt injection."""
    return analysis.get("prompt_injection", "")


# ═══════════════════════════════════════════════════════════════
#  9. PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_media_patterns(chat_id: int) -> None:
    try:
        data = _media_pattern_cache.get(chat_id, [])
        path = MEDIA_PATTERNS_DIR / f"{chat_id}.json"
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        visual_logger.warning(f"Failed to save media patterns for {chat_id}: {e}")


def load_media_patterns(chat_id: int) -> List[Dict]:
    try:
        path = MEDIA_PATTERNS_DIR / f"{chat_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        visual_logger.warning(f"Failed to load media patterns for {chat_id}: {e}")
    return []
