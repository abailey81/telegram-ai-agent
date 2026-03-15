"""
Media Response Brain
=====================
Unified contextual decision engine for ALL media responses:
reactions, GIFs, stickers, quote-replies, emoji guidance, voice notes.

Every decision is context-driven — nothing random. Each action gets a
contextual score (0.0-1.0) computed from conversation signals. Only
actions scoring above threshold fire. Mutual exclusion prevents spam.

Consumes signals from: NLP engine, emotional intelligence, personality
engine, visual analysis, autonomy engine, prediction engine.
"""

import logging
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

media_brain_logger = logging.getLogger("media_response_brain")

# ═══════════════════════════════════════════════════════════════
#  REACTION PALETTES — context-specific emoji selection
# ═══════════════════════════════════════════════════════════════

# Emotion → ranked reaction emojis (best fit first)
REACTION_PALETTE = {
    # Positive emotions
    "joy":        ["😂", "🔥", "💀", "❤️", "🎉"],
    "love":       ["❤️", "😍", "🥰", "❤️‍🔥", "😘"],
    "excitement": ["🔥", "🎉", "😍", "💯", "🤩"],
    "humor":      ["😂", "💀", "🤣", "😭", "👀"],
    "flirty":     ["😏", "🔥", "😍", "👀", "❤️‍🔥"],
    "agreement":  ["👍", "💯", "🔥", "✅", "👏"],
    "gratitude":  ["❤️", "🥺", "🙏", "😊", "💕"],
    "pride":      ["🔥", "💯", "👏", "🎉", "😤"],

    # Negative / complex emotions — be very careful
    "sadness":    ["❤️"],               # only heart for sadness
    "anger":      [],                    # NEVER react to anger
    "fear":       ["❤️"],               # just support
    "surprise":   ["👀", "😮"],          # minimal, relevant
    "disgust":    [],                    # don't react
    "sarcasm":    ["👀"],                # just eyes, subtle
    "contempt":   [],                    # don't react
    "frustration": [],                   # don't react

    # Media-specific
    "photo_selfie":  ["🔥", "😍", "❤️", "👀", "❤️‍🔥"],
    "photo_food":    ["😋", "🔥", "😍", "🤤"],
    "photo_nature":  ["😍", "🔥", "❤️", "✨"],
    "photo_general": ["👀", "🔥", "😍"],
    "voice_message": ["❤️", "🔥", "👍", "🥰"],
    "sticker":       ["😂", "❤️", "💀", "🔥"],
    "gif":           ["😂", "💀", "🔥"],
    "video":         ["🔥", "👀", "😍", "❤️"],
}

# GIF search queries mapped to specific contexts
GIF_CONTEXT_QUERIES = {
    # Emotion-based
    "joy":        ["happy dance", "celebration", "excited", "yay"],
    "humor":      ["laughing hard", "lol reaction", "dying laughing", "funny reaction"],
    "love":       ["love heart", "cute couple", "kiss", "heart eyes"],
    "surprise":   ["shocked reaction", "surprised pikachu", "omg", "mind blown"],
    "excitement": ["hype", "lets go", "pumped", "excited dance"],
    "sadness":    ["sad hug", "crying", "aww", "comfort"],
    "sarcasm":    ["slow clap", "sure jan", "oh really", "cool story"],
    "agreement":  ["nodding", "yes exactly", "facts", "preach"],
    "flirty":     ["wink", "flirty", "hey there", "smooth"],
    "anger":      ["angry reaction", "frustrated", "mad"],
    "disgust":    ["eww", "gross", "nope"],

    # Topic-based (override emotion-based when topic detected)
    "food":       ["yummy", "eating", "delicious", "hungry"],
    "work":       ["working hard", "office mood", "boss"],
    "gym":        ["workout", "gains", "gym motivation", "flex"],
    "music":      ["dancing", "vibing", "music mood"],
    "tired":      ["sleepy", "tired af", "exhausted", "need sleep"],
    "party":      ["party", "lets go party", "turn up"],
    "pets":       ["cute pet", "aww animal", "adorable"],
    "travel":     ["travel", "adventure", "explore"],
    "morning":    ["good morning", "waking up", "coffee"],
    "night":      ["goodnight", "sleepy", "sweet dreams"],
}

# Topic keywords for GIF query selection
TOPIC_KEYWORDS = {
    "food": ["eat", "food", "hungry", "dinner", "lunch", "cooking", "recipe",
             "ел", "еда", "голодный", "ужин", "обед", "готовить"],
    "work": ["work", "job", "boss", "meeting", "deadline", "office",
             "работа", "босс", "совещание"],
    "gym":  ["gym", "workout", "exercise", "run", "fit",
             "спортзал", "тренировка", "бегать"],
    "music": ["song", "music", "listen", "playlist", "concert",
              "песня", "музыка", "слушать"],
    "tired": ["tired", "exhausted", "sleepy", "cant sleep", "insomnia",
              "устал", "устала", "спать", "бессонница"],
    "party": ["party", "club", "drink", "bar", "celebration",
              "вечеринка", "клуб", "бар"],
    "pets":  ["dog", "cat", "pet", "puppy", "kitten",
              "собака", "кот", "кошка", "питомец"],
    "travel": ["travel", "trip", "flight", "vacation", "holiday",
               "путешествие", "поездка", "отпуск"],
    "morning": ["morning", "woke up", "coffee", "breakfast",
                "утро", "проснулся", "кофе", "завтрак"],
    "night": ["night", "sleep", "goodnight", "bedtime",
              "ночь", "спать", "спокойной"],
}


# ═══════════════════════════════════════════════════════════════
#  EMOJI GUIDANCE — context-aware rules for LLM emoji usage
# ═══════════════════════════════════════════════════════════════

EMOJI_DENSITY_RULES = {
    # stage → (min_emojis, max_emojis, style_note)
    "new":       (0, 1, "Minimal emojis — don't come on too strong"),
    "warming":   (0, 2, "Light emojis — match their usage"),
    "deep":      (1, 3, "Natural emojis — you know each other well"),
    "conflict":  (0, 0, "NO emojis during conflict — they feel dismissive"),
    "makeup":    (0, 1, "Careful — one ❤️ max, only if genuine"),
    "flirting":  (1, 3, "Flirty emojis welcome — 😏🔥😍"),
}

EMOTION_EMOJI_SETS = {
    "joy":        "😂💀🔥😭",
    "love":       "❤️😍🥰💕😘",
    "flirty":     "😏🔥😈👀😍",
    "sadness":    "❤️🥺",
    "anger":      "",  # no emojis in anger
    "excitement": "🔥🎉😍💯",
    "humor":      "😂💀🤣😭",
    "surprise":   "😮👀🤯",
    "neutral":    "👍",
    "sarcasm":    "😐👀💀",
}


# ═══════════════════════════════════════════════════════════════
#  CORE: CONTEXTUAL SCORING ENGINE
# ═══════════════════════════════════════════════════════════════

def _score_reaction(ctx: Dict[str, Any]) -> float:
    """Score whether to send a reaction emoji (0.0-1.0)."""
    score = 0.0
    media_type = ctx.get("media_type", "text")
    emotion = ctx.get("emotion", "neutral")
    stage = ctx.get("stage", "warming")
    temp = ctx.get("temperature", "neutral")
    text = ctx.get("text", "")
    engagement = ctx.get("engagement", 0.5)

    # Photo/video = high reaction value (it's expected/natural)
    if media_type in ("MessageMediaPhoto", "photo", "image"):
        score += 0.45
    elif media_type in ("video", "video_note", "round_video"):
        score += 0.40
    elif media_type in ("voice_message", "audio"):
        score += 0.30
    elif media_type in ("sticker",):
        score += 0.25
    elif media_type in ("gif", "animation"):
        score += 0.20

    # Strong emotions boost reaction likelihood
    emotion_boost = {
        "love": 0.25, "joy": 0.20, "humor": 0.20, "flirty": 0.25,
        "excitement": 0.20, "surprise": 0.15, "sadness": 0.15,
        "photo_selfie": 0.30, "photo_food": 0.15,
    }
    score += emotion_boost.get(emotion, 0.05)

    # Short messages are more "react-worthy" — long messages deserve text
    word_count = len(text.split())
    if word_count <= 3:
        score += 0.10
    elif word_count > 20:
        score -= 0.10

    # High engagement = react more naturally
    if engagement > 0.7:
        score += 0.08
    elif engagement < 0.3:
        score -= 0.10

    # Conflict = careful with reactions (can seem dismissive)
    if stage == "conflict" or temp in ("boiling", "hot"):
        score -= 0.30

    # They used emojis → mirror with reactions
    emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
    if emoji_count >= 2:
        score += 0.10

    return max(0.0, min(1.0, score))


def _score_gif(ctx: Dict[str, Any]) -> float:
    """Score whether to send a GIF (0.0-1.0)."""
    score = 0.0
    emotion = ctx.get("emotion", "neutral")
    stage = ctx.get("stage", "warming")
    temp = ctx.get("temperature", "neutral")
    media_type = ctx.get("media_type", "text")
    text = ctx.get("text", "")

    # Humor/excitement = GIF territory
    if emotion in ("humor", "joy"):
        score += 0.30
    elif emotion == "excitement":
        score += 0.25
    elif emotion == "surprise":
        score += 0.20
    elif emotion == "sarcasm":
        score += 0.15
    elif emotion in ("flirty",):
        score += 0.10

    # They sent a GIF → reciprocate
    if media_type in ("gif", "animation"):
        score += 0.30

    # Conflict / serious = no GIFs
    if stage == "conflict" or temp in ("boiling", "hot", "frozen"):
        score -= 0.40
    if stage == "deep" and emotion in ("sadness", "fear"):
        score -= 0.30

    # Casual conversation = GIFs fit
    if stage in ("warming", "flirting"):
        score += 0.05

    # Detection of specific GIF-worthy keywords
    gif_triggers = ["lol", "lmao", "haha", "😂", "💀", "omg", "no way",
                    "i cant", "im dead", "хаха", "ахах", "ору", "ржу"]
    if any(t in text.lower() for t in gif_triggers):
        score += 0.15

    return max(0.0, min(1.0, score))


def _score_sticker(ctx: Dict[str, Any]) -> float:
    """Score whether to send a sticker (0.0-1.0).
    HARD DISABLED — we cannot see what stickers look like, so random sticker
    search always produces irrelevant/stupid results. Emoji reactions are
    much better. Always returns 0.0."""
    return 0.0


def _score_quote_reply(ctx: Dict[str, Any]) -> float:
    """Score whether to quote-reply to a specific message (0.0-1.0)."""
    score = 0.0
    text = ctx.get("text", "")
    text_lower = text.lower()
    recent = ctx.get("recent_messages", [])

    # Direct question → natural to quote-reply
    if "?" in text:
        score += 0.30

    # Reference patterns → they're referring to something specific
    ref_patterns = [
        "about what you", "what you said", "you mentioned", "earlier you",
        "you were saying", "going back to", "regarding", "re:", "replying to",
        "то что ты", "ты говорил", "ты говорила", "насчет", "по поводу",
        "that thing", "what u said", "u said",
    ]
    if any(p in text_lower for p in ref_patterns):
        score += 0.45

    # Multiple rapid messages from them → quote the most relevant
    their_recent = [m for m in recent[-5:] if m.get("sender") in ("Them", "them", "other")]
    if len(their_recent) >= 3:
        score += 0.20

    # Long message → more likely to need targeted reply
    if len(text.split()) > 25:
        score += 0.10

    return max(0.0, min(1.0, score))


def _score_voice_note(ctx: Dict[str, Any]) -> float:
    """Score whether to send a voice note (0.0-1.0)."""
    score = 0.0
    media_type = ctx.get("media_type", "text")
    emotion = ctx.get("emotion", "neutral")
    stage = ctx.get("stage", "warming")
    hour = ctx.get("time_of_day", 12)

    # They sent voice → reciprocate
    if media_type in ("voice_message", "audio"):
        score += 0.35

    # Intimate moments
    if emotion == "love" and stage == "deep":
        score += 0.15

    # Late night + emotional = voice fits
    if (22 <= hour or hour < 2) and emotion in ("love", "sadness"):
        score += 0.10

    return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════
#  EMOJI PICKER — context-aware selection
# ═══════════════════════════════════════════════════════════════

def _pick_reaction_emoji(ctx: Dict[str, Any]) -> str:
    """Pick the best reaction emoji based on full context."""
    emotion = ctx.get("emotion", "neutral")
    media_type = ctx.get("media_type", "text")
    text = ctx.get("text", "")
    sticker_emoji = ctx.get("sticker_emoji")

    # Sticker → react with the sticker's own emoji if available
    if media_type == "sticker" and sticker_emoji:
        return sticker_emoji

    # Build palette key
    key = emotion

    # Photo sub-classification
    if media_type in ("MessageMediaPhoto", "photo", "image"):
        text_lower = text.lower()
        if any(w in text_lower for w in ("selfie", "me", "look", "фото", "это я")):
            key = "photo_selfie"
        elif any(w in text_lower for w in ("food", "eat", "cook", "еда", "готовлю")):
            key = "photo_food"
        elif any(w in text_lower for w in ("nature", "view", "sunset", "вид", "закат")):
            key = "photo_nature"
        else:
            key = "photo_general"

    if media_type == "voice_message":
        key = "voice_message"
    elif media_type in ("gif", "animation"):
        key = "gif"
    elif media_type == "video":
        key = "video"

    palette = REACTION_PALETTE.get(key, REACTION_PALETTE.get(emotion, ["👍"]))

    # Weighted random from palette (first items are preferred)
    weights = [1.0 / (i + 1) for i in range(len(palette))]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0
    for emoji, w in zip(palette, weights):
        cumulative += w
        if r <= cumulative:
            return emoji

    return palette[0]


def _build_gif_query(ctx: Dict[str, Any]) -> str:
    """Build a contextual GIF search query."""
    emotion = ctx.get("emotion", "neutral")
    text = ctx.get("text", "").lower()
    reply_text = ctx.get("reply_text", "").lower()

    # First: check for topic-based queries (most specific)
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            queries = GIF_CONTEXT_QUERIES.get(topic, [])
            if queries:
                return random.choice(queries)

    # Second: check for specific phrases that make great GIF queries
    phrase_gifs = [
        (r"(got the job|got hired|accepted|promoted)", "celebration excited"),
        (r"(broke up|breakup|dumped|single now)", "sad crying"),
        (r"(im so tired|exhausted|dead tired)", "exhausted sleepy"),
        (r"(i cant believe|no way|wtf)", "shocked surprised"),
        (r"(miss you|miss u|скучаю)", "miss you hug"),
        (r"(love you|люблю)", "love heart"),
        (r"(happy birthday|день рождения)", "happy birthday celebration"),
        (r"(thank you|thanks|спасибо)", "thank you appreciation"),
        (r"(im sorry|прости|извини)", "sorry apologize"),
        (r"(lets go|давай|поехали)", "lets go hype"),
    ]
    for pattern, query in phrase_gifs:
        if re.search(pattern, text, re.I):
            return query

    # Third: emotion-based fallback
    queries = GIF_CONTEXT_QUERIES.get(emotion, ["reaction"])
    return random.choice(queries)


def _pick_sticker_emoji(ctx: Dict[str, Any]) -> str:
    """Pick the best sticker search emoji based on context."""
    emotion = ctx.get("emotion", "neutral")
    sticker_emoji = ctx.get("sticker_emoji")

    # If they sent a sticker, use same emoji family
    if sticker_emoji:
        return sticker_emoji

    emoji_map = {
        "love": "❤️", "flirty": "😏", "humor": "😂", "joy": "😂",
        "sadness": "😢", "anger": "😤", "excitement": "🔥",
        "surprise": "😮", "sarcasm": "😏", "agreement": "👍",
    }
    return emoji_map.get(emotion, "😊")


# ═══════════════════════════════════════════════════════════════
#  EMOJI GUIDANCE BUILDER — tells LLM how to use emojis in text
# ═══════════════════════════════════════════════════════════════

def build_emoji_guidance(ctx: Dict[str, Any]) -> str:
    """Build specific emoji guidance for the LLM's text reply."""
    stage = ctx.get("stage", "warming")
    emotion = ctx.get("emotion", "neutral")
    text = ctx.get("text", "")
    their_emoji_count = sum(1 for c in text if ord(c) > 0x1F600)
    temp = ctx.get("temperature", "neutral")

    # Get density rules
    min_e, max_e, style = EMOJI_DENSITY_RULES.get(
        stage, (0, 2, "Natural emoji usage")
    )

    # Mirror their emoji density
    if their_emoji_count == 0:
        max_e = min(max_e, 1)  # they don't use emojis, limit ours
    elif their_emoji_count >= 3:
        min_e = max(min_e, 1)  # they use lots, we should too

    # Conflict override — no emojis
    if temp in ("boiling", "hot") or stage == "conflict":
        return "DO NOT use any emojis. This is serious."

    # Get appropriate emoji set
    emoji_set = EMOTION_EMOJI_SETS.get(emotion, "")
    if not emoji_set:
        return f"Keep emojis minimal (0-{max_e} max). {style}."

    lines = [f"Emoji usage: {min_e}-{max_e} emojis. {style}."]
    lines.append(f"Best emojis for this context: {emoji_set}")

    # Specific guidance based on emotion
    if emotion == "flirty":
        lines.append("Use 😏 or 🔥 for playful tension. Don't overdo hearts.")
    elif emotion == "love":
        lines.append("Hearts and warmth. Match their energy.")
    elif emotion == "humor":
        lines.append("Use 😂 or 💀 naturally. Don't add emojis to every sentence.")
    elif emotion == "sadness":
        lines.append("One ❤️ max. Emojis can feel dismissive when someone is sad.")

    return " ".join(lines)


# ═══════════════════════════════════════════════════════════════
#  MAIN: COMPUTE MEDIA RESPONSE
# ═══════════════════════════════════════════════════════════════

def compute_media_response(
    text: str,
    reply_text: str = "",
    nlp_analysis: Optional[Dict] = None,
    emotion: str = "neutral",
    emotion_score: float = 0.5,
    temperature: str = "neutral",
    stage: str = "warming",
    engagement: float = 0.5,
    media_type: str = "text",
    sticker_emoji: Optional[str] = None,
    personality: Optional[Dict] = None,
    recent_messages: Optional[List[Dict]] = None,
    our_last_media: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified media response brain. Computes what media to send based
    on full conversational context.

    Returns a MediaDecision dict with all media actions + reasoning.
    """
    hour = datetime.now().hour
    recent = recent_messages or []

    # Build context object for scoring functions
    ctx = {
        "text": text,
        "reply_text": reply_text,
        "nlp": nlp_analysis or {},
        "emotion": emotion,
        "emotion_score": emotion_score,
        "temperature": temperature,
        "stage": stage,
        "engagement": engagement,
        "media_type": media_type,
        "sticker_emoji": sticker_emoji,
        "personality": personality or {},
        "recent_messages": recent,
        "time_of_day": hour,
        "our_last_media": our_last_media,
    }

    # ── Score each action ──
    reaction_score = _score_reaction(ctx)
    gif_score = _score_gif(ctx)
    sticker_score = _score_sticker(ctx)
    quote_score = _score_quote_reply(ctx)
    voice_score = _score_voice_note(ctx)

    # ── Threshold: only fire if score > 0.4 ──
    threshold = 0.40

    decision = {
        "reaction": None,
        "gif": None,
        "sticker": None,
        "quote_reply_to": None,
        "emoji_guidance": "",
        "voice_note": False,
        "dice": None,
        "reasoning": "",
        "scores": {
            "reaction": round(reaction_score, 3),
            "gif": round(gif_score, 3),
            "sticker": round(sticker_score, 3),
            "quote_reply": round(quote_score, 3),
            "voice_note": round(voice_score, 3),
        },
    }

    reasons = []

    # ── Voice note (exclusive — overrides other media) ──
    if voice_score > threshold:
        decision["voice_note"] = True
        reasons.append(f"voice_note({voice_score:.2f})")

    # ── Quote reply (compatible with other actions) ──
    if quote_score > threshold:
        # Find the target message
        target_id = None
        if recent:
            try:
                from autonomy_engine import identify_relevant_reply_target
                target_id = identify_relevant_reply_target(text, recent)
            except ImportError:
                pass
        if target_id:
            decision["quote_reply_to"] = {
                "message_id": target_id,
                "reason": "contextual_reference",
            }
            reasons.append(f"quote_reply({quote_score:.2f})")

    # ── Reaction, GIF, Sticker — mutual exclusion (max 2) ──
    media_actions = []
    if reaction_score > threshold:
        media_actions.append(("reaction", reaction_score))
    if gif_score > threshold and not decision["voice_note"]:
        media_actions.append(("gif", gif_score))
    if sticker_score > threshold and not decision["voice_note"]:
        media_actions.append(("sticker", sticker_score))

    # Sort by score descending, take top 2
    media_actions.sort(key=lambda x: x[1], reverse=True)
    active_media = media_actions[:2]

    # Don't send GIF + sticker together (reaction + one of them is OK)
    if len(active_media) == 2:
        types = {a[0] for a in active_media}
        if "gif" in types and "sticker" in types:
            # Keep the higher-scored one
            active_media = [active_media[0]]

    for action_type, action_score in active_media:
        if action_type == "reaction":
            emoji = _pick_reaction_emoji(ctx)
            decision["reaction"] = {
                "emoji": emoji,
                "reason": f"{emotion}_reaction",
            }
            reasons.append(f"reaction={emoji}({action_score:.2f})")

        elif action_type == "gif":
            query = _build_gif_query(ctx)
            decision["gif"] = {
                "query": query,
                "reason": f"{emotion}_gif",
            }
            reasons.append(f"gif='{query}'({action_score:.2f})")

        elif action_type == "sticker":
            sticker_e = _pick_sticker_emoji(ctx)
            decision["sticker"] = {
                "emoji": sticker_e,
                "reason": f"{emotion}_sticker",
            }
            reasons.append(f"sticker={sticker_e}({action_score:.2f})")

    # ── Emoji guidance (always computed) ──
    decision["emoji_guidance"] = build_emoji_guidance(ctx)

    # ── Dice (very contextual — only during playful moments) ──
    if (stage in ("warming", "flirting") and emotion in ("humor", "excitement", "flirty")
            and engagement > 0.6 and random.random() < 0.08):
        dice_emoji = random.choice(["🎲", "🎯", "🏀", "🎳"])
        decision["dice"] = dice_emoji
        reasons.append(f"dice={dice_emoji}")

    # ── Avoid repetition ──
    if our_last_media:
        # Don't send same media type twice in a row
        if decision.get("gif") and our_last_media == "gif":
            decision["gif"] = None
            reasons.append("skipped_gif(repetition)")
        if decision.get("sticker") and our_last_media == "sticker":
            decision["sticker"] = None
            reasons.append("skipped_sticker(repetition)")

    decision["reasoning"] = " | ".join(reasons) if reasons else "text_only"

    media_brain_logger.info(
        f"Media brain: {decision['reasoning']} "
        f"[scores: r={reaction_score:.2f} g={gif_score:.2f} "
        f"s={sticker_score:.2f} q={quote_score:.2f} v={voice_score:.2f}]"
    )

    return decision


def should_react_only(ctx: Dict[str, Any]) -> bool:
    """Determine if we should ONLY react (no text reply).

    This replaces the random probability in autonomy_engine.
    """
    text = ctx.get("text", "")
    media_type = ctx.get("media_type", "text")
    engagement = ctx.get("engagement", 0.5)
    stage = ctx.get("stage", "warming")

    text_lower = text.strip().lower()

    # Sticker/GIF in casual conversation → react-only is natural
    if media_type in ("sticker", "gif") and stage not in ("conflict", "deep"):
        return random.random() < 0.30

    # Single emoji → react with emoji
    if len(text_lower) <= 2 and not text_lower.isalpha():
        return random.random() < 0.35

    # Just a laugh → react
    if text_lower in ("lol", "haha", "lmao", "😂", "🤣", "💀", "ахах", "хаха", "ору"):
        return random.random() < 0.25

    # Very low engagement → save energy, just react
    if engagement < 0.2 and len(text_lower.split()) <= 3:
        return random.random() < 0.20

    # Never react-only during conflict or important moments
    if stage in ("conflict", "deep"):
        return False

    return False
