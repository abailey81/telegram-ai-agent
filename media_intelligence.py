"""
Media Intelligence Engine — Advanced recognition for images, videos, audio,
voice messages, stickers, GIFs, emojis, and documents in Telegram conversations.

Provides sophisticated analysis of media messages for conversational context,
emotional intelligence, and conversation understanding.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("media_intelligence")


# ═══════════════════════════════════════════════════════════════
#  EMOJI INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

# Comprehensive emoji → emotion/intent mapping
EMOJI_EMOTION_MAP: Dict[str, Dict[str, Any]] = {
    # Love & Romance
    "❤️": {"emotion": "love", "intensity": 0.9, "intent": "romantic", "category": "heart"},
    "💕": {"emotion": "love", "intensity": 0.8, "intent": "romantic", "category": "heart"},
    "💗": {"emotion": "love", "intensity": 0.7, "intent": "romantic", "category": "heart"},
    "💖": {"emotion": "love", "intensity": 0.8, "intent": "romantic", "category": "heart"},
    "💓": {"emotion": "love", "intensity": 0.7, "intent": "romantic", "category": "heart"},
    "💞": {"emotion": "love", "intensity": 0.7, "intent": "romantic", "category": "heart"},
    "💘": {"emotion": "love", "intensity": 0.9, "intent": "romantic", "category": "heart"},
    "💝": {"emotion": "love", "intensity": 0.8, "intent": "romantic", "category": "heart"},
    "🖤": {"emotion": "love", "intensity": 0.6, "intent": "edgy_affection", "category": "heart"},
    "🤍": {"emotion": "love", "intensity": 0.5, "intent": "pure_affection", "category": "heart"},
    "🩷": {"emotion": "love", "intensity": 0.7, "intent": "romantic", "category": "heart"},
    "🩵": {"emotion": "love", "intensity": 0.5, "intent": "casual_affection", "category": "heart"},
    "💔": {"emotion": "sadness", "intensity": 0.9, "intent": "heartbreak", "category": "heart"},
    "🫶": {"emotion": "love", "intensity": 0.8, "intent": "supportive", "category": "gesture"},

    # Face — Happy/Positive
    "😊": {"emotion": "joy", "intensity": 0.6, "intent": "friendly", "category": "face"},
    "😄": {"emotion": "joy", "intensity": 0.7, "intent": "happy", "category": "face"},
    "😁": {"emotion": "joy", "intensity": 0.7, "intent": "happy", "category": "face"},
    "☺️": {"emotion": "joy", "intensity": 0.5, "intent": "content", "category": "face"},
    "🥰": {"emotion": "love", "intensity": 0.9, "intent": "romantic", "category": "face"},
    "😍": {"emotion": "love", "intensity": 0.9, "intent": "attraction", "category": "face"},
    "🤩": {"emotion": "joy", "intensity": 0.8, "intent": "admiration", "category": "face"},
    "😘": {"emotion": "love", "intensity": 0.8, "intent": "kiss", "category": "face"},
    "😚": {"emotion": "love", "intensity": 0.6, "intent": "shy_kiss", "category": "face"},
    "😙": {"emotion": "love", "intensity": 0.5, "intent": "casual_kiss", "category": "face"},
    "🥳": {"emotion": "joy", "intensity": 0.9, "intent": "celebration", "category": "face"},

    # Face — Flirty/Playful
    "😏": {"emotion": "desire", "intensity": 0.7, "intent": "flirty", "category": "face"},
    "😉": {"emotion": "playful", "intensity": 0.6, "intent": "flirty", "category": "face"},
    "😈": {"emotion": "desire", "intensity": 0.8, "intent": "naughty", "category": "face"},
    "😜": {"emotion": "playful", "intensity": 0.7, "intent": "silly", "category": "face"},
    "😝": {"emotion": "playful", "intensity": 0.7, "intent": "teasing", "category": "face"},
    "🤪": {"emotion": "playful", "intensity": 0.8, "intent": "wild", "category": "face"},
    "🫦": {"emotion": "desire", "intensity": 0.9, "intent": "seductive", "category": "face"},
    "👀": {"emotion": "curiosity", "intensity": 0.6, "intent": "attention", "category": "face"},

    # Face — Laughing
    "😂": {"emotion": "joy", "intensity": 0.8, "intent": "laughing", "category": "face"},
    "🤣": {"emotion": "joy", "intensity": 0.9, "intent": "laughing_hard", "category": "face"},
    "😹": {"emotion": "joy", "intensity": 0.7, "intent": "laughing", "category": "face"},
    "💀": {"emotion": "joy", "intensity": 0.9, "intent": "dead_laughing", "category": "face"},

    # Face — Sad/Crying
    "😢": {"emotion": "sadness", "intensity": 0.7, "intent": "sad", "category": "face"},
    "😭": {"emotion": "sadness", "intensity": 0.9, "intent": "crying", "category": "face"},
    "🥺": {"emotion": "sadness", "intensity": 0.6, "intent": "pleading", "category": "face"},
    "😔": {"emotion": "sadness", "intensity": 0.6, "intent": "pensive", "category": "face"},
    "😞": {"emotion": "sadness", "intensity": 0.7, "intent": "disappointed", "category": "face"},
    "😿": {"emotion": "sadness", "intensity": 0.6, "intent": "sad", "category": "face"},
    "😥": {"emotion": "sadness", "intensity": 0.5, "intent": "worried", "category": "face"},

    # Face — Angry
    "😤": {"emotion": "anger", "intensity": 0.7, "intent": "frustrated", "category": "face"},
    "😡": {"emotion": "anger", "intensity": 0.9, "intent": "angry", "category": "face"},
    "😠": {"emotion": "anger", "intensity": 0.8, "intent": "annoyed", "category": "face"},
    "🙄": {"emotion": "anger", "intensity": 0.5, "intent": "dismissive", "category": "face"},
    "💢": {"emotion": "anger", "intensity": 0.8, "intent": "rage", "category": "face"},

    # Face — Surprised
    "😱": {"emotion": "surprise", "intensity": 0.9, "intent": "shocked", "category": "face"},
    "😲": {"emotion": "surprise", "intensity": 0.7, "intent": "astonished", "category": "face"},
    "🤯": {"emotion": "surprise", "intensity": 0.9, "intent": "mind_blown", "category": "face"},
    "😳": {"emotion": "surprise", "intensity": 0.6, "intent": "embarrassed", "category": "face"},
    "😮": {"emotion": "surprise", "intensity": 0.5, "intent": "surprised", "category": "face"},

    # Face — Other
    "🤔": {"emotion": "neutral", "intensity": 0.3, "intent": "thinking", "category": "face"},
    "😬": {"emotion": "fear", "intensity": 0.4, "intent": "awkward", "category": "face"},
    "🫣": {"emotion": "fear", "intensity": 0.3, "intent": "shy", "category": "face"},
    "🤗": {"emotion": "love", "intensity": 0.6, "intent": "supportive", "category": "face"},
    "🥱": {"emotion": "neutral", "intensity": 0.2, "intent": "bored", "category": "face"},
    "😴": {"emotion": "neutral", "intensity": 0.1, "intent": "sleepy", "category": "face"},
    "🤮": {"emotion": "anger", "intensity": 0.6, "intent": "disgust", "category": "face"},
    "🤢": {"emotion": "anger", "intensity": 0.5, "intent": "disgust", "category": "face"},

    # Gestures
    "👍": {"emotion": "neutral", "intensity": 0.3, "intent": "approval", "category": "gesture"},
    "👎": {"emotion": "anger", "intensity": 0.4, "intent": "disapproval", "category": "gesture"},
    "🤝": {"emotion": "neutral", "intensity": 0.4, "intent": "agreement", "category": "gesture"},
    "🙏": {"emotion": "sadness", "intensity": 0.5, "intent": "pleading", "category": "gesture"},
    "👏": {"emotion": "joy", "intensity": 0.6, "intent": "applause", "category": "gesture"},
    "💪": {"emotion": "joy", "intensity": 0.6, "intent": "supportive", "category": "gesture"},
    "🤞": {"emotion": "fear", "intensity": 0.4, "intent": "hopeful", "category": "gesture"},
    "🫡": {"emotion": "neutral", "intensity": 0.3, "intent": "respectful", "category": "gesture"},

    # Symbols
    "✨": {"emotion": "joy", "intensity": 0.5, "intent": "sparkly", "category": "symbol"},
    "💫": {"emotion": "joy", "intensity": 0.5, "intent": "dreamy", "category": "symbol"},
    "🔥": {"emotion": "desire", "intensity": 0.8, "intent": "hot", "category": "symbol"},
    "💋": {"emotion": "love", "intensity": 0.8, "intent": "kiss", "category": "symbol"},
    "🎉": {"emotion": "joy", "intensity": 0.8, "intent": "celebration", "category": "symbol"},
    "⚡": {"emotion": "surprise", "intensity": 0.6, "intent": "energy", "category": "symbol"},
    "🌹": {"emotion": "love", "intensity": 0.8, "intent": "romantic", "category": "symbol"},
    "🌸": {"emotion": "tenderness", "intensity": 0.5, "intent": "gentle", "category": "symbol"},
    "🍷": {"emotion": "desire", "intensity": 0.5, "intent": "romantic", "category": "symbol"},
    "🍕": {"emotion": "neutral", "intensity": 0.2, "intent": "casual", "category": "symbol"},
    "☕": {"emotion": "neutral", "intensity": 0.2, "intent": "casual", "category": "symbol"},
}

# Emoji combination patterns — certain combinations carry specific meaning
EMOJI_COMBO_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": ["😂", "💀"], "meaning": "dying_of_laughter", "emotion": "joy", "intensity": 1.0},
    {"pattern": ["🥺", "👉", "👈"], "meaning": "shy_request", "emotion": "tenderness", "intensity": 0.7},
    {"pattern": ["❤️", "🔥"], "meaning": "passionate_love", "emotion": "desire", "intensity": 0.95},
    {"pattern": ["😤", "💢"], "meaning": "very_angry", "emotion": "anger", "intensity": 0.95},
    {"pattern": ["😢", "💔"], "meaning": "heartbroken", "emotion": "sadness", "intensity": 0.95},
    {"pattern": ["🥰", "😘"], "meaning": "very_loving", "emotion": "love", "intensity": 0.95},
    {"pattern": ["🔥", "🔥", "🔥"], "meaning": "extremely_hot", "emotion": "desire", "intensity": 1.0},
    {"pattern": ["😍", "❤️"], "meaning": "love_struck", "emotion": "love", "intensity": 0.9},
    {"pattern": ["😏", "😈"], "meaning": "naughty_intentions", "emotion": "desire", "intensity": 0.9},
    {"pattern": ["🙄", "😤"], "meaning": "annoyed_and_fed_up", "emotion": "anger", "intensity": 0.8},
    {"pattern": ["😭", "😭"], "meaning": "sobbing", "emotion": "sadness", "intensity": 0.95},
    {"pattern": ["❤️", "❤️", "❤️"], "meaning": "overwhelming_love", "emotion": "love", "intensity": 1.0},
]


def analyze_emojis(text: str) -> Dict[str, Any]:
    """
    Deep emoji analysis — extracts emotional meaning from emojis in text.

    Returns:
        - dominant_emotion: the primary emotion expressed
        - emotions: dict of emotion -> intensity scores
        - intent: primary communicative intent
        - emoji_density: ratio of emojis to text length
        - combo_meanings: any detected emoji combination patterns
        - sentiment_shift: how emojis modify the text sentiment
    """
    # Extract all emojis from text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0000200D"             # ZWJ
        "\U00002764"             # heart variants
        "]+",
        flags=re.UNICODE,
    )

    found_emojis = emoji_pattern.findall(text)
    # Also split individual emojis from clusters
    individual_emojis = []
    for cluster in found_emojis:
        for char in cluster:
            if char in EMOJI_EMOTION_MAP or ord(char) > 0x1F000:
                individual_emojis.append(char)

    # Handle variation selectors (e.g., ❤️ = ❤ + VS16)
    cleaned = text
    for e in found_emojis:
        cleaned = cleaned.replace(e, "")
    text_length = len(cleaned.strip())

    if not individual_emojis:
        return {
            "has_emojis": False,
            "dominant_emotion": None,
            "emotions": {},
            "intent": None,
            "emoji_density": 0.0,
            "combo_meanings": [],
            "sentiment_shift": 0.0,
            "emoji_count": 0,
        }

    # Score emotions from individual emojis
    emotion_scores: Dict[str, float] = {}
    intent_scores: Dict[str, int] = {}
    for e in individual_emojis:
        # Try the emoji with variation selector too
        emoji_info = EMOJI_EMOTION_MAP.get(e) or EMOJI_EMOTION_MAP.get(e + "\uFE0F")
        if emoji_info:
            emo = emoji_info["emotion"]
            intensity = emoji_info["intensity"]
            emotion_scores[emo] = max(emotion_scores.get(emo, 0), intensity)
            intent = emoji_info["intent"]
            intent_scores[intent] = intent_scores.get(intent, 0) + 1

    # Check combo patterns
    emoji_str = "".join(individual_emojis)
    combo_meanings = []
    for combo in EMOJI_COMBO_PATTERNS:
        pattern_str = "".join(combo["pattern"])
        if pattern_str in emoji_str:
            combo_meanings.append({
                "meaning": combo["meaning"],
                "emotion": combo["emotion"],
                "intensity": combo["intensity"],
            })
            # Boost the combo emotion
            emotion_scores[combo["emotion"]] = max(
                emotion_scores.get(combo["emotion"], 0), combo["intensity"]
            )

    # Determine dominant
    dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else None
    dominant_intent = max(intent_scores, key=intent_scores.get) if intent_scores else None

    # Calculate emoji density (emojis per word)
    word_count = max(len(cleaned.split()), 1)
    emoji_density = len(individual_emojis) / word_count

    # Sentiment shift: how much emojis change the base text
    positive_emojis = {"joy", "love", "tenderness", "playful", "desire"}
    negative_emojis = {"sadness", "anger", "fear"}
    pos_score = sum(v for k, v in emotion_scores.items() if k in positive_emojis)
    neg_score = sum(v for k, v in emotion_scores.items() if k in negative_emojis)
    sentiment_shift = (pos_score - neg_score) / max(pos_score + neg_score, 1)

    return {
        "has_emojis": True,
        "dominant_emotion": dominant_emotion,
        "emotions": emotion_scores,
        "intent": dominant_intent,
        "emoji_density": round(emoji_density, 3),
        "combo_meanings": combo_meanings,
        "sentiment_shift": round(sentiment_shift, 3),
        "emoji_count": len(individual_emojis),
        "emojis_found": individual_emojis[:20],  # cap for readability
    }


# ═══════════════════════════════════════════════════════════════
#  MEDIA TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

# Telegram media type → meaning in conversational context
MEDIA_CONTEXT_MAP: Dict[str, Dict[str, Any]] = {
    "MessageMediaPhoto": {
        "type": "photo",
        "intimacy_signal": 0.7,
        "engagement_level": "high",
        "typical_emotions": ["love", "joy", "desire", "tenderness"],
        "relationship_meaning": "sharing_visual_moments",
        "response_suggestions": [
            "compliment_specific_detail",
            "express_appreciation",
            "share_photo_back",
            "react_with_heart",
        ],
        "analysis_notes": "Photos indicate trust and willingness to share life visually",
    },
    "MessageMediaDocument": {
        "type": "document",
        "intimacy_signal": 0.3,
        "engagement_level": "medium",
        "typical_emotions": ["neutral"],
        "relationship_meaning": "practical_sharing",
        "response_suggestions": ["acknowledge", "discuss_content"],
        "analysis_notes": "Document sharing is practical, may contain GIFs or stickers",
    },
    "MessageMediaGeo": {
        "type": "location",
        "intimacy_signal": 0.8,
        "engagement_level": "high",
        "typical_emotions": ["love", "joy"],
        "relationship_meaning": "location_sharing_trust",
        "response_suggestions": [
            "acknowledge_location",
            "express_wanting_to_be_there",
            "share_your_location_back",
        ],
        "analysis_notes": "Location sharing = high trust signal, wanting the other person to know where they are",
    },
    "MessageMediaGeoLive": {
        "type": "live_location",
        "intimacy_signal": 0.9,
        "engagement_level": "very_high",
        "typical_emotions": ["love", "tenderness"],
        "relationship_meaning": "deep_trust_transparency",
        "response_suggestions": [
            "express_appreciation_for_trust",
            "share_location_back",
        ],
        "analysis_notes": "Live location = very high trust; they want the recipient to always know where they are",
    },
    "MessageMediaContact": {
        "type": "contact",
        "intimacy_signal": 0.5,
        "engagement_level": "medium",
        "typical_emotions": ["neutral"],
        "relationship_meaning": "practical_sharing",
        "response_suggestions": ["acknowledge", "save_contact"],
        "analysis_notes": "Contact sharing is practical",
    },
    "MessageMediaPoll": {
        "type": "poll",
        "intimacy_signal": 0.3,
        "engagement_level": "medium",
        "typical_emotions": ["playful", "neutral"],
        "relationship_meaning": "seeking_input_playful",
        "response_suggestions": ["participate", "discuss"],
        "analysis_notes": "Polls in DMs = playful engagement seeking the other person's opinion",
    },
    "MessageMediaWebPage": {
        "type": "link_preview",
        "intimacy_signal": 0.3,
        "engagement_level": "medium",
        "typical_emotions": ["neutral", "joy"],
        "relationship_meaning": "sharing_interests",
        "response_suggestions": ["discuss_content", "express_interest"],
        "analysis_notes": "Link sharing = wanting to share interests/conversation topics",
    },
    "MessageMediaDice": {
        "type": "dice",
        "intimacy_signal": 0.4,
        "engagement_level": "medium",
        "typical_emotions": ["playful", "joy"],
        "relationship_meaning": "playful_engagement",
        "response_suggestions": ["play_along", "send_dice_back"],
        "analysis_notes": "Dice/games = playful mood, wants fun interaction",
    },
}


def classify_media_type(media_type_name: str, message: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Classify a Telegram media type and return conversational context analysis.

    Args:
        media_type_name: The __name__ of the Telethon media type class
        message: Optional message dict with extra metadata
    """
    base_context = MEDIA_CONTEXT_MAP.get(media_type_name, {
        "type": "unknown",
        "intimacy_signal": 0.2,
        "engagement_level": "low",
        "typical_emotions": ["neutral"],
        "relationship_meaning": "unknown_media",
        "response_suggestions": ["acknowledge"],
        "analysis_notes": f"Unknown media type: {media_type_name}",
    })

    result = dict(base_context)

    # Enhance based on message metadata
    if message:
        caption = message.get("caption", "")
        if caption:
            emoji_analysis = analyze_emojis(caption)
            result["caption_emoji_analysis"] = emoji_analysis
            if emoji_analysis.get("dominant_emotion"):
                result["caption_emotion"] = emoji_analysis["dominant_emotion"]

    return result


# ═══════════════════════════════════════════════════════════════
#  VOICE MESSAGE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

def analyze_voice_message(duration_seconds: int, message_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyze a voice message based on its duration and conversation context.

    Voice messages carry rich emotional subtext in conversations:
    - Short (1-5s): quick reactions, casual, or lazy to type
    - Medium (5-30s): thoughtful response, explaining something
    - Long (30-120s): deep emotional content, venting, storytelling
    - Very long (120s+): serious conversation, major emotional event

    Args:
        duration_seconds: Voice message duration in seconds
        message_context: Optional conversation context
    """
    if duration_seconds <= 5:
        category = "quick_reaction"
        intimacy = 0.5
        emotional_intensity = 0.4
        likely_emotions = ["playful", "casual", "neutral"]
        meaning = "Quick vocal reaction — casual and comfortable"
        response_type = "brief_acknowledgment_or_voice_back"
    elif duration_seconds <= 15:
        category = "casual_voice"
        intimacy = 0.6
        emotional_intensity = 0.5
        likely_emotions = ["neutral", "joy", "casual"]
        meaning = "Comfortable enough to voice message — prefers vocal to typing"
        response_type = "voice_or_text_response"
    elif duration_seconds <= 30:
        category = "thoughtful_message"
        intimacy = 0.7
        emotional_intensity = 0.6
        likely_emotions = ["love", "tenderness", "sincere"]
        meaning = "Took time to express thoughts vocally — indicates emotional investment"
        response_type = "thoughtful_text_or_voice_response"
    elif duration_seconds <= 60:
        category = "emotional_sharing"
        intimacy = 0.8
        emotional_intensity = 0.8
        likely_emotions = ["love", "sadness", "anger", "sincerity"]
        meaning = "Significant emotional content — they need to be heard"
        response_type = "empathetic_response_acknowledge_effort"
    elif duration_seconds <= 120:
        category = "deep_emotional"
        intimacy = 0.9
        emotional_intensity = 0.9
        likely_emotions = ["love", "sadness", "anger", "fear", "sincerity"]
        meaning = "Deep emotional expression — likely venting, confessing, or working through something major"
        response_type = "supportive_patient_voice_message_back"
    else:
        category = "extended_emotional"
        intimacy = 0.95
        emotional_intensity = 0.95
        likely_emotions = ["love", "sadness", "anger", "sincerity"]
        meaning = "Extended voice message — major emotional event, needs serious attention"
        response_type = "call_them_or_long_supportive_response"

    return {
        "duration_seconds": duration_seconds,
        "category": category,
        "intimacy_signal": intimacy,
        "emotional_intensity": emotional_intensity,
        "likely_emotions": likely_emotions,
        "meaning": meaning,
        "suggested_response_type": response_type,
        "prefers_voice": duration_seconds > 10,
        "analysis": f"Voice message ({duration_seconds}s) — {meaning}",
    }


# ═══════════════════════════════════════════════════════════════
#  VIDEO MESSAGE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

def analyze_video_message(
    duration_seconds: int,
    is_round: bool = False,
    has_caption: bool = False,
    caption: str = "",
) -> Dict[str, Any]:
    """
    Analyze video messages including video notes (round/circle videos).

    Video notes (round videos) are particularly intimate in Telegram —
    they show the sender's face in the moment.

    Args:
        duration_seconds: Video duration
        is_round: Whether it's a round video note (shows face)
        has_caption: Whether the video has a caption
        caption: The caption text if any
    """
    if is_round:
        # Round video notes — very personal, showing face
        intimacy = 0.85
        emotional_intensity = 0.7
        meaning = "Video note showing their face — high intimacy and personal connection"
        response_suggestions = [
            "send_video_note_back",
            "compliment_their_appearance",
            "express_missing_their_face",
        ]
        if duration_seconds <= 10:
            category = "quick_face_check"
            meaning = "Quick video note — showing face, wanting visual connection"
        elif duration_seconds <= 30:
            category = "personal_video_note"
            meaning = "Personal video note — sharing a moment or reacting visually"
        else:
            category = "extended_video_note"
            meaning = "Extended video note — significant visual communication, emotionally invested"
            intimacy = 0.9
    else:
        # Regular video
        if duration_seconds <= 15:
            category = "short_video"
            intimacy = 0.5
            emotional_intensity = 0.4
            meaning = "Short video sharing — sharing something funny or interesting"
        elif duration_seconds <= 60:
            category = "medium_video"
            intimacy = 0.6
            emotional_intensity = 0.5
            meaning = "Medium video — sharing an experience or moment"
        else:
            category = "long_video"
            intimacy = 0.5
            emotional_intensity = 0.4
            meaning = "Long video — sharing content (may be forwarded)"
        response_suggestions = [
            "react_to_content",
            "discuss_video",
            "share_related_content",
        ]

    result = {
        "duration_seconds": duration_seconds,
        "is_round_video": is_round,
        "category": category,
        "intimacy_signal": intimacy,
        "emotional_intensity": emotional_intensity,
        "meaning": meaning,
        "response_suggestions": response_suggestions,
    }

    if has_caption and caption:
        result["caption_analysis"] = analyze_emojis(caption)

    return result


# ═══════════════════════════════════════════════════════════════
#  STICKER & GIF INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

# Common sticker/GIF emotional categories
STICKER_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "love": ["love", "heart", "kiss", "hug", "cuddle", "miss", "любовь", "люблю",
             "целую", "обнимаю", "скучаю"],
    "joy": ["happy", "dance", "celebrate", "yay", "party", "excited", "радость",
            "танец", "праздник", "ура"],
    "sadness": ["sad", "cry", "tears", "lonely", "miss", "грустно", "плачу",
                "слёзы", "одиноко"],
    "anger": ["angry", "mad", "rage", "furious", "annoyed", "злой", "бесит",
              "ярость"],
    "playful": ["funny", "joke", "lol", "haha", "silly", "prank", "смешно",
                "шутка", "хаха"],
    "surprise": ["wow", "omg", "shock", "surprise", "what", "вау", "шок",
                 "удивление"],
    "tenderness": ["cute", "sweet", "soft", "gentle", "adorable", "милый",
                   "нежный", "мягкий"],
}


def analyze_sticker(sticker_emoji: Optional[str] = None, sticker_set_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a sticker message in conversational context.

    Stickers are visual emotional expressions. In conversations:
    - Love stickers: expressing affection non-verbally
    - Funny stickers: lightening the mood, playful
    - Cute/animal stickers: tenderness, playful affection
    - Angry stickers: expressing frustration semi-playfully (less harsh than text)
    """
    result = {
        "type": "sticker",
        "intimacy_signal": 0.5,
        "engagement_level": "medium",
        "emotional_expression": "visual",
    }

    if sticker_emoji:
        emoji_info = EMOJI_EMOTION_MAP.get(sticker_emoji)
        if emoji_info:
            result["emotion"] = emoji_info["emotion"]
            result["intensity"] = emoji_info["intensity"]
            result["intent"] = emoji_info["intent"]
        else:
            result["emotion"] = "playful"
            result["intensity"] = 0.5
            result["intent"] = "expressive"

    if sticker_set_name:
        set_lower = sticker_set_name.lower()
        detected_emotion = "neutral"
        for emotion, keywords in STICKER_EMOTION_KEYWORDS.items():
            if any(kw in set_lower for kw in keywords):
                detected_emotion = emotion
                break
        result["set_emotion"] = detected_emotion
        result["sticker_set"] = sticker_set_name

    result["relationship_meaning"] = (
        "Stickers are a comfortable, visual way to express emotions. "
        "Using stickers indicates comfort in the conversation and desire "
        "for expressive, fun communication."
    )
    result["response_suggestions"] = [
        "send_sticker_back",
        "react_with_matching_emoji",
        "playful_text_response",
    ]

    return result


def analyze_gif(caption: str = "") -> Dict[str, Any]:
    """
    Analyze a GIF message in conversational context.

    GIFs are used for:
    - Humor and reaction (most common)
    - Expressing emotions visually
    - Playful or expressive communication
    - Inside jokes
    """
    result = {
        "type": "gif",
        "intimacy_signal": 0.4,
        "engagement_level": "medium",
        "typical_intent": "playful_expressive",
        "relationship_meaning": (
            "GIF sharing indicates playful mood and desire for visual humor. "
            "Frequent GIF use suggests comfort and established rapport."
        ),
        "response_suggestions": [
            "react_to_gif",
            "send_gif_back",
            "playful_text",
        ],
    }

    if caption:
        result["caption_analysis"] = analyze_emojis(caption)

    return result


# ═══════════════════════════════════════════════════════════════
#  PHOTO/IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_photo_context(
    caption: str = "",
    is_selfie_likely: bool = False,
    time_of_day: Optional[str] = None,
    conversation_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a photo message in conversational context.

    Different types of photos carry different emotional weight:
    - Selfies: high intimacy, wanting to be seen
    - Food photos: sharing daily life, casual
    - Scenery: sharing experiences
    - Screenshot: sharing information/humor
    - Group photos: showing social life

    Args:
        caption: Photo caption text
        is_selfie_likely: Whether this is likely a selfie
        time_of_day: Time context (morning, evening, etc.)
        conversation_stage: Current conversation stage if known
    """
    result = {
        "type": "photo",
        "engagement_level": "high",
    }

    if is_selfie_likely or (caption and any(w in caption.lower() for w in
            ["selfie", "me", "look", "outfit", "face", "hair", "я", "смотри", "как я"])):
        result["subtype"] = "selfie"
        result["intimacy_signal"] = 0.8
        result["emotional_intent"] = "seeking_validation_connection"
        result["likely_emotions"] = ["desire", "love", "playful"]
        result["relationship_meaning"] = (
            "Selfie sharing is a high-trust, high-intimacy action. "
            "They want you to see them and likely want a compliment. "
            "This shows they care about your perception of them."
        )
        result["response_suggestions"] = [
            "specific_compliment_about_appearance",
            "express_how_they_make_you_feel",
            "send_selfie_back",
            "heart_or_fire_reaction",
        ]
    elif caption and any(w in caption.lower() for w in
            ["food", "eat", "cook", "lunch", "dinner", "ем", "еда", "готовлю", "вкусно"]):
        result["subtype"] = "food"
        result["intimacy_signal"] = 0.4
        result["emotional_intent"] = "sharing_daily_life"
        result["likely_emotions"] = ["joy", "neutral"]
        result["relationship_meaning"] = "Sharing meals = sharing daily life, casual intimacy"
        result["response_suggestions"] = [
            "comment_on_food",
            "suggest_eating_together",
            "share_what_you_ate",
        ]
    else:
        result["subtype"] = "general"
        result["intimacy_signal"] = 0.6
        result["emotional_intent"] = "sharing_moment"
        result["likely_emotions"] = ["joy", "neutral", "love"]
        result["relationship_meaning"] = "Sharing visual moments = wanting you in their experience"
        result["response_suggestions"] = [
            "react_to_content",
            "ask_about_context",
            "express_appreciation",
        ]

    if caption:
        result["caption_analysis"] = analyze_emojis(caption)

    if time_of_day:
        if time_of_day in ("night", "late_night", "evening"):
            result["time_context"] = "Late-night photo sharing increases intimacy signal"
            result["intimacy_signal"] = min(result.get("intimacy_signal", 0.5) + 0.15, 1.0)
        elif time_of_day in ("morning", "early_morning"):
            result["time_context"] = "Morning photos = you're first on their mind"
            result["intimacy_signal"] = min(result.get("intimacy_signal", 0.5) + 0.1, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════
#  MEDIA PATTERN ANALYSIS (conversation-level)
# ═══════════════════════════════════════════════════════════════

def analyze_media_patterns(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze media usage patterns across a conversation for engagement insights.

    Looks at:
    - Media frequency: how often media is shared
    - Media types: what kinds of media dominate
    - Media timing: when media is shared
    - Reciprocity: who shares more media
    - Progression: is media sharing increasing (trust building)
    """
    media_messages = [m for m in messages if m.get("has_media")]
    total = len(messages)
    media_count = len(media_messages)

    if media_count == 0:
        return {
            "media_frequency": 0,
            "media_ratio": 0.0,
            "analysis": "No media in conversation — text-only communication",
            "relationship_signal": "neutral",
        }

    media_ratio = media_count / max(total, 1)

    # Count media types
    type_counts: Dict[str, int] = {}
    for m in media_messages:
        mt = m.get("media_type", "unknown")
        type_counts[mt] = type_counts.get(mt, 0) + 1

    dominant_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"

    # Analyze emoji usage across all messages
    all_emojis = []
    for m in messages:
        text = m.get("text", "")
        if text:
            ea = analyze_emojis(text)
            if ea.get("has_emojis"):
                all_emojis.extend(ea.get("emojis_found", []))

    emoji_frequency = len(all_emojis) / max(total, 1)

    # Determine relationship signal
    if media_ratio > 0.3:
        relationship_signal = "high_engagement"
        analysis = "Heavy media sharing indicates deep comfort and high engagement"
    elif media_ratio > 0.15:
        relationship_signal = "moderate_engagement"
        analysis = "Regular media sharing — comfortable rapport with visual communication"
    elif media_ratio > 0.05:
        relationship_signal = "light_engagement"
        analysis = "Occasional media sharing — primarily text-based communication"
    else:
        relationship_signal = "text_focused"
        analysis = "Minimal media sharing — conversation is text-centric"

    return {
        "media_frequency": media_count,
        "media_ratio": round(media_ratio, 3),
        "total_messages": total,
        "type_counts": type_counts,
        "dominant_media_type": dominant_type,
        "emoji_frequency": round(emoji_frequency, 3),
        "total_emojis_used": len(all_emojis),
        "relationship_signal": relationship_signal,
        "analysis": analysis,
    }


# ═══════════════════════════════════════════════════════════════
#  COMPREHENSIVE MEDIA ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_media_message(
    media_type: str,
    text: str = "",
    caption: str = "",
    duration: int = 0,
    is_round: bool = False,
    sticker_emoji: Optional[str] = None,
    sticker_set: Optional[str] = None,
    conversation_context: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive media message analysis — the main entry point.

    Analyzes any media message type with full conversational context:
    - Photos, selfies
    - Voice messages
    - Video messages, video notes
    - Stickers
    - GIFs
    - Emojis in text
    - Documents, locations, contacts

    Returns a rich analysis dict with emotional state, intimacy signals,
    contextual meaning, and response suggestions.
    """
    result: Dict[str, Any] = {
        "media_type": media_type,
        "timestamp": datetime.now().isoformat(),
    }

    # Always analyze text/caption for emojis
    full_text = text or caption or ""
    if full_text:
        result["text_analysis"] = analyze_emojis(full_text)

    # Type-specific analysis
    if "Photo" in media_type:
        result["media_analysis"] = analyze_photo_context(
            caption=caption or text,
        )
    elif media_type == "voice_message" or ("Document" in media_type and duration > 0):
        result["media_analysis"] = analyze_voice_message(
            duration_seconds=duration,
        )
    elif "Video" in media_type or is_round:
        result["media_analysis"] = analyze_video_message(
            duration_seconds=duration,
            is_round=is_round,
            has_caption=bool(caption),
            caption=caption,
        )
    elif media_type == "sticker":
        result["media_analysis"] = analyze_sticker(
            sticker_emoji=sticker_emoji,
            sticker_set_name=sticker_set,
        )
    elif media_type == "gif" or media_type == "animation":
        result["media_analysis"] = analyze_gif(caption=caption or text)
    else:
        result["media_analysis"] = classify_media_type(media_type)

    # Conversation-level patterns
    if conversation_context:
        result["conversation_patterns"] = analyze_media_patterns(conversation_context)

    # Synthesize overall emotional state
    media_analysis = result.get("media_analysis", {})
    text_analysis = result.get("text_analysis", {})

    # Determine overall emotion
    media_emotions = media_analysis.get("likely_emotions", media_analysis.get("typical_emotions", []))
    text_emotion = text_analysis.get("dominant_emotion")

    if text_emotion:
        result["dominant_emotion"] = text_emotion
    elif media_emotions:
        result["dominant_emotion"] = media_emotions[0]
    else:
        result["dominant_emotion"] = "neutral"

    result["intimacy_signal"] = media_analysis.get("intimacy_signal", 0.3)
    result["engagement_level"] = media_analysis.get("engagement_level", "medium")
    result["response_suggestions"] = media_analysis.get("response_suggestions", ["acknowledge"])

    return result


# ═══════════════════════════════════════════════════════════════
#  MEDIA-AWARE TEXT GENERATION CONTEXT
# ═══════════════════════════════════════════════════════════════

def build_media_context_for_reply(
    media_type: str,
    caption: str = "",
    duration: int = 0,
    is_round: bool = False,
    sticker_emoji: Optional[str] = None,
) -> str:
    """
    Build a natural-language context string about a media message
    that can be injected into the auto-reply system prompt.

    This tells the AI what kind of media was received and how to respond.
    """
    analysis = analyze_media_message(
        media_type=media_type,
        caption=caption,
        duration=duration,
        is_round=is_round,
        sticker_emoji=sticker_emoji,
    )

    media_info = analysis.get("media_analysis", {})
    lines = []

    if "Photo" in media_type:
        lines.append("[They sent a photo]")
        if caption:
            lines.append(f"Caption: {caption}")
        meaning = media_info.get("relationship_meaning", "sharing a visual moment")
        lines.append(f"Context: {meaning}")
        suggestions = media_info.get("response_suggestions", [])
        if suggestions:
            lines.append(f"Good responses: {', '.join(s.replace('_', ' ') for s in suggestions[:3])}")

    elif media_type == "voice_message":
        dur = duration
        lines.append(f"[They sent a voice message ({dur}s)]")
        meaning = media_info.get("meaning", "voice communication")
        lines.append(f"Context: {meaning}")
        lines.append(f"Emotional intensity: {media_info.get('emotional_intensity', 0.5):.0%}")
        lines.append(f"Suggested response: {media_info.get('suggested_response_type', 'acknowledge')}")

    elif is_round or "Video" in media_type:
        if is_round:
            lines.append(f"[They sent a video note/circle video ({duration}s)]")
            lines.append("Context: They showed you their face — this is personal and intimate")
            lines.append("Good response: Send a video note back or compliment them")
        else:
            lines.append(f"[They sent a video ({duration}s)]")
            if caption:
                lines.append(f"Caption: {caption}")

    elif media_type == "sticker":
        emoji_desc = sticker_emoji or "expressive"
        lines.append(f"[They sent a sticker ({emoji_desc})]")
        emotion = media_info.get("emotion", "playful")
        lines.append(f"Emotional tone: {emotion}")
        lines.append("Good response: React with matching energy, send sticker back, or playful text")

    elif media_type in ("gif", "animation"):
        lines.append("[They sent a GIF]")
        if caption:
            lines.append(f"Caption: {caption}")
        lines.append("Context: Playful/expressive mood — respond with matching energy")

    elif "Geo" in media_type:
        lines.append("[They shared their location]")
        lines.append("Context: High trust signal — they want you to know where they are")
        lines.append("Good response: Acknowledge, express wanting to be there, or share yours")

    else:
        lines.append(f"[They sent media: {media_type}]")
        if caption:
            lines.append(f"Caption: {caption}")

    # Add emoji context from caption
    text_analysis = analysis.get("text_analysis", {})
    if text_analysis.get("has_emojis"):
        dom_emo = text_analysis.get("dominant_emotion")
        if dom_emo:
            lines.append(f"Emoji mood: {dom_emo}")

    return "\n".join(lines)
