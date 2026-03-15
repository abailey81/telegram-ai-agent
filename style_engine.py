"""
Adaptive Style Engine & Personality Profiler.

Implements sophisticated communication adaptation:

1. Real-Time Style Profiling - Analyze their texting patterns
2. Style Mirroring - Match their energy while keeping personality
3. Personality Sheet - Structured character definition
4. Communication Fingerprinting - Unique per-person adaptation
5. Style Drift Detection - Detect when tone shifts mid-conversation
6. Linguistic Accommodation - Research-backed style matching

Based on Communication Accommodation Theory (Giles, 1973)
and Linguistic Style Matching research.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

style_logger = logging.getLogger("style_engine")
style_logger.setLevel(logging.INFO)

STYLE_DATA_DIR = Path(__file__).parent / "engine_data" / "styles"
STYLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-pickup: load autoresearch-optimized engine parameters ──
_OPTIMIZED_STYLE_PARAMS = None
_OPTIMIZED_STYLE_PARAMS_MTIME = 0


def _load_optimized_style_params():
    """Load optimized style params from autoresearch (auto-pickup on file change)."""
    global _OPTIMIZED_STYLE_PARAMS, _OPTIMIZED_STYLE_PARAMS_MTIME
    params_file = Path(__file__).parent / "engine_data" / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_STYLE_PARAMS_MTIME:
            import json as _json
            _OPTIMIZED_STYLE_PARAMS = _json.loads(params_file.read_text())
            _OPTIMIZED_STYLE_PARAMS_MTIME = mtime
        return _OPTIMIZED_STYLE_PARAMS
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  1. REAL-TIME STYLE PROFILING
# ═══════════════════════════════════════════════════════════════

def profile_message_style(text: str) -> Dict[str, Any]:
    """Analyze the communication style of a single message.

    Returns quantified style dimensions for comparison and matching.
    """
    words = text.split()
    word_count = len(words)

    # Length analysis
    if word_count <= 3:
        length_category = "very_short"
    elif word_count <= 8:
        length_category = "short"
    elif word_count <= 20:
        length_category = "medium"
    elif word_count <= 40:
        length_category = "long"
    else:
        length_category = "very_long"

    # Emoji analysis
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001f900-\U0001f9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
        r"\U00002600-\U000026FF\U0000FE00-\U0000FE0F"
        r"\U00002764\U0000200D]+",
        flags=re.UNICODE,
    )
    emojis = emoji_pattern.findall(text)
    emoji_count = len(emojis)
    emoji_density = emoji_count / max(word_count, 1)

    # Formality analysis
    formal_words = [
        "therefore", "however", "furthermore", "nevertheless",
        "regarding", "accordingly", "indeed", "moreover",
        "additionally", "consequently",
        # Russian formal words
        "поэтому", "однако", "кроме того", "тем не менее",
        "относительно", "соответственно", "действительно",
        "более того", "следовательно", "вследствие",
    ]
    informal_markers = [
        "lol", "haha", "omg", "ngl", "tbh", "bruh",
        "nah", "yeah", "gonna", "wanna", "gotta", "kinda",
        "rn", "nvm", "idk", "imo", "btw", "ikr", "smh",
        # Russian informal markers
        "лол", "хаха", "ахах", "блин", "типа", "прикол",
        "чё", "чо", "ваще", "капец", "фиг", "нифига",
        "норм", "збс", "кста", "хз", "пофиг", "фигня",
    ]
    text_lower = text.lower()
    formal_count = sum(1 for w in formal_words if w in text_lower)
    informal_count = sum(1 for w in informal_markers if w in text_lower)

    # Russian formality via ты/вы distinction (takes priority)
    uses_vy = bool(re.search(r'\b[Вв]ы\b|\b[Вв]аш', text))
    uses_ty = bool(re.search(r'\b[Тт]ы\b|\b[Тт]ебе\b|\b[Тт]ебя\b|\b[Тт]вой\b', text))
    if uses_vy and not uses_ty:
        formality = "formal"
    else:
        if uses_ty:
            informal_count += 1  # boost informal detection

        if formal_count > informal_count:
            formality = "formal"
        elif informal_count > 2:
            formality = "very_casual"
        elif informal_count > 0:
            formality = "casual"
        else:
            formality = "neutral"

    # Punctuation style
    has_period = text.endswith(".")
    has_exclamation = "!" in text
    has_question = "?" in text
    has_ellipsis = "..." in text
    no_punctuation = not any(c in text for c in ".!?")

    punctuation_style = "minimal" if no_punctuation else "standard"
    if has_exclamation and text.count("!") > 1:
        punctuation_style = "expressive"
    elif has_ellipsis:
        punctuation_style = "trailing"

    # Capitalization style
    if text.isupper() and word_count > 2:
        caps_style = "all_caps"
    elif text.islower():
        caps_style = "all_lowercase"
    elif text[0].isupper() if text else False:
        caps_style = "proper"
    else:
        caps_style = "mixed"

    # Vocabulary complexity (average word length as proxy)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
    if avg_word_len > 6:
        vocabulary = "sophisticated"
    elif avg_word_len > 4.5:
        vocabulary = "moderate"
    else:
        vocabulary = "simple"

    # Humor detection
    humor_markers = [
        "haha", "hehe", "lol", "lmao", "rofl", "😂", "🤣",
        "jk", "kidding", "joking", "😏",
        # Russian humor markers
        "хаха", "ахахах", "хехе", "лол", "ржу", "угар",
        "прикол", "я шучу", "шутка",
    ]
    has_humor = any(h in text_lower for h in humor_markers)

    # Affection level
    affection_markers = [
        "❤", "🥰", "😍", "😘", "💕", "love", "babe", "baby",
        "darling", "sweetheart", "honey", "miss you", "xoxo",
        # Russian affection markers
        "люблю", "обожаю", "дорогой", "дорогая", "милый", "милая",
        "сладкий", "солнышко", "зайка", "котик", "скучаю", "целую",
    ]
    affection_count = sum(1 for m in affection_markers if m in text_lower)

    return {
        "word_count": word_count,
        "length_category": length_category,
        "emoji_count": emoji_count,
        "emoji_density": round(emoji_density, 3),
        "formality": formality,
        "punctuation_style": punctuation_style,
        "caps_style": caps_style,
        "vocabulary": vocabulary,
        "has_humor": has_humor,
        "affection_level": min(affection_count, 3),
        "has_question": has_question,
        "has_exclamation": has_exclamation,
    }


# ═══════════════════════════════════════════════════════════════
#  2. COMMUNICATION FINGERPRINT (PER-PERSON PROFILE)
# ═══════════════════════════════════════════════════════════════

def load_style_profile(chat_id: int) -> Dict[str, Any]:
    """Load the communication style profile for a chat partner."""
    path = STYLE_DATA_DIR / f"{chat_id}_style.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "avg_message_length": 0,
        "avg_word_count": 0,
        "emoji_frequency": 0.0,
        "formality_score": 0.5,
        "humor_frequency": 0.0,
        "affection_level": 0.0,
        "vocabulary_level": "moderate",
        "punctuation_preference": "standard",
        "caps_preference": "mixed",
        "question_frequency": 0.0,
        "messages_analyzed": 0,
        "style_summary": "",
        "last_updated": None,
    }


def save_style_profile(chat_id: int, profile: Dict[str, Any]):
    """Save style profile."""
    profile["last_updated"] = datetime.now().isoformat()
    path = STYLE_DATA_DIR / f"{chat_id}_style.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))


def update_style_profile(
    chat_id: int,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Update style profile from their messages using rolling averages."""
    profile = load_style_profile(chat_id)

    their_msgs = [m for m in messages if m.get("sender") == "Them" and m.get("text")]
    if not their_msgs:
        return profile

    # Analyze recent messages
    styles = [profile_message_style(m["text"]) for m in their_msgs[-30:]]
    n = len(styles)

    if n < 3:
        return profile

    # Compute aggregates
    profile["avg_word_count"] = round(sum(s["word_count"] for s in styles) / n, 1)
    profile["avg_message_length"] = round(
        sum(len(m["text"]) for m in their_msgs[-30:]) / n, 1
    )
    profile["emoji_frequency"] = round(
        sum(1 for s in styles if s["emoji_count"] > 0) / n, 3
    )
    profile["humor_frequency"] = round(
        sum(1 for s in styles if s["has_humor"]) / n, 3
    )
    profile["affection_level"] = round(
        sum(s["affection_level"] for s in styles) / n, 2
    )
    profile["question_frequency"] = round(
        sum(1 for s in styles if s["has_question"]) / n, 3
    )

    # Formality score (0=very casual, 1=very formal)
    formality_map = {"very_casual": 0.1, "casual": 0.3, "neutral": 0.5, "formal": 0.8}
    profile["formality_score"] = round(
        sum(formality_map.get(s["formality"], 0.5) for s in styles) / n, 3
    )

    # Most common vocabulary level
    vocab_counts = {}
    for s in styles:
        v = s["vocabulary"]
        vocab_counts[v] = vocab_counts.get(v, 0) + 1
    profile["vocabulary_level"] = max(vocab_counts, key=vocab_counts.get)

    # Most common punctuation style
    punct_counts = {}
    for s in styles:
        p = s["punctuation_style"]
        punct_counts[p] = punct_counts.get(p, 0) + 1
    profile["punctuation_preference"] = max(punct_counts, key=punct_counts.get)

    # Generate readable summary
    profile["style_summary"] = _generate_style_summary(profile)
    profile["messages_analyzed"] = n

    save_style_profile(chat_id, profile)
    return profile


def _generate_style_summary(profile: Dict[str, Any]) -> str:
    """Generate a human-readable style summary."""
    _opt = _load_optimized_style_params()
    parts = []

    # Message length
    avg_words = profile.get("avg_word_count", 0)
    if avg_words < 5:
        parts.append("writes very short messages")
    elif avg_words < 10:
        parts.append("writes brief messages")
    elif avg_words > 25:
        parts.append("writes detailed, longer messages")

    # Emoji usage
    emoji_freq = profile.get("emoji_frequency", 0)
    if emoji_freq > (_opt or {}).get("emoji_density_high", 0.6):
        parts.append("uses emojis frequently")
    elif emoji_freq < (_opt or {}).get("emoji_density_low", 0.1):
        parts.append("rarely uses emojis")

    # Formality
    formality = profile.get("formality_score", 0.5)
    if formality < (_opt or {}).get("formality_casual_threshold", 0.3):
        parts.append("very casual texting style")
    elif formality > (_opt or {}).get("formality_formal_threshold", 0.7):
        parts.append("more formal writing style")

    # Humor
    humor = profile.get("humor_frequency", 0)
    if humor > (_opt or {}).get("humor_frequency_threshold", 0.3):
        parts.append("frequently uses humor")

    # Affection
    affection = profile.get("affection_level", 0)
    if affection > 1.5:
        parts.append("very affectionate in messages")
    elif affection > 0.5:
        parts.append("moderately affectionate")

    # Questions
    questions = profile.get("question_frequency", 0)
    if questions > 0.4:
        parts.append("asks lots of questions")
    elif questions < 0.1:
        parts.append("rarely asks questions")

    return "; ".join(parts) if parts else "standard communication style"


# ═══════════════════════════════════════════════════════════════
#  3. STYLE MATCHING / MIRRORING
# ═══════════════════════════════════════════════════════════════

def generate_style_directives(
    their_profile: Dict[str, Any],
    current_message_style: Dict[str, Any],
    personality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate style matching directives for response generation.

    Mirrors their style while preserving core personality traits.
    Uses current message style for real-time adaptation.
    """
    directives = {
        "target_length": "medium",
        "target_formality": "casual",
        "emoji_usage": "moderate",
        "humor_level": "occasional",
        "question_inclusion": True,
        "affection_expression": "moderate",
        "energy_level": "matched",
        "mirroring_notes": [],
    }

    # Length matching
    current_length = current_message_style.get("length_category", "medium")
    avg_words = their_profile.get("avg_word_count", 10)

    if current_length in ("very_short", "short"):
        directives["target_length"] = "short"
        directives["mirroring_notes"].append(
            "Keep it brief - they're writing short messages right now"
        )
    elif current_length in ("long", "very_long"):
        directives["target_length"] = "medium_to_long"
        directives["mirroring_notes"].append(
            "They're writing longer messages - match that energy with more detail"
        )
    else:
        # Use their average
        if avg_words < 8:
            directives["target_length"] = "short"
        elif avg_words > 20:
            directives["target_length"] = "medium_to_long"

    # Formality matching
    formality = their_profile.get("formality_score", 0.5)
    current_formality = current_message_style.get("formality", "neutral")

    if formality < 0.3 or current_formality == "very_casual":
        directives["target_formality"] = "very_casual"
        directives["mirroring_notes"].append(
            "Use casual language: contractions, slang, lowercase"
        )
    elif formality > 0.7 or current_formality == "formal":
        directives["target_formality"] = "neutral"
        directives["mirroring_notes"].append(
            "Keep it natural but avoid excessive slang"
        )

    # Emoji matching
    emoji_freq = their_profile.get("emoji_frequency", 0.3)
    current_emojis = current_message_style.get("emoji_count", 0)

    if emoji_freq > 0.5 or current_emojis >= 2:
        directives["emoji_usage"] = "frequent"
        directives["mirroring_notes"].append(
            "Use 1-2 emojis - they like emojis"
        )
    elif emoji_freq < 0.1 and current_emojis == 0:
        directives["emoji_usage"] = "minimal"
        directives["mirroring_notes"].append(
            "Skip emojis or use max 1 - they don't use many"
        )

    # Humor matching
    humor_freq = their_profile.get("humor_frequency", 0.2)
    if humor_freq > 0.3 or current_message_style.get("has_humor"):
        directives["humor_level"] = "frequent"
    elif humor_freq < 0.1:
        directives["humor_level"] = "rare"

    # Affection matching
    affection = their_profile.get("affection_level", 0.5)
    if affection > 1.5:
        directives["affection_expression"] = "high"
    elif affection < 0.3:
        directives["affection_expression"] = "subtle"

    # Energy level from current message
    current_arousal = (
        1.0 if current_message_style.get("caps_style") == "all_caps"
        else 0.8 if current_message_style.get("has_exclamation")
        else 0.3 if current_length == "very_short"
        else 0.5
    )
    if current_arousal > 0.7:
        directives["energy_level"] = "high"
    elif current_arousal < 0.3:
        directives["energy_level"] = "low"

    return directives


def detect_style_shift(
    messages: List[Dict[str, str]],
    n_recent: int = 5,
) -> Optional[Dict[str, Any]]:
    """Detect if their communication style has shifted recently.

    Compares recent messages to their baseline to detect mood changes,
    engagement shifts, or conversation turning points.
    """
    their_msgs = [m for m in messages if m.get("sender") == "Them" and m.get("text")]
    if len(their_msgs) < n_recent + 3:
        return None

    recent = their_msgs[-n_recent:]
    earlier = their_msgs[-(n_recent + 5):-n_recent]

    if not earlier:
        return None

    recent_styles = [profile_message_style(m["text"]) for m in recent]
    earlier_styles = [profile_message_style(m["text"]) for m in earlier]

    # Compare dimensions
    shifts = {}

    # Length shift
    recent_avg_len = sum(s["word_count"] for s in recent_styles) / len(recent_styles)
    earlier_avg_len = sum(s["word_count"] for s in earlier_styles) / len(earlier_styles)
    if earlier_avg_len > 0:
        len_ratio = recent_avg_len / earlier_avg_len
        if len_ratio < 0.4:
            shifts["length"] = "significantly_shorter"
        elif len_ratio > 2.0:
            shifts["length"] = "significantly_longer"

    # Emoji shift
    recent_emoji = sum(s["emoji_count"] for s in recent_styles) / len(recent_styles)
    earlier_emoji = sum(s["emoji_count"] for s in earlier_styles) / len(earlier_styles)
    if earlier_emoji > 0.5 and recent_emoji < 0.1:
        shifts["emoji"] = "stopped_using_emojis"
    elif earlier_emoji < 0.1 and recent_emoji > 0.5:
        shifts["emoji"] = "started_using_emojis"

    # Humor shift
    recent_humor = sum(1 for s in recent_styles if s["has_humor"]) / len(recent_styles)
    earlier_humor = sum(1 for s in earlier_styles if s["has_humor"]) / len(earlier_styles)
    if earlier_humor > 0.3 and recent_humor == 0:
        shifts["humor"] = "stopped_being_funny"

    if not shifts:
        return None

    return {
        "detected_shifts": shifts,
        "interpretation": _interpret_style_shift(shifts),
    }


def _interpret_style_shift(shifts: Dict[str, str]) -> str:
    """Interpret what a style shift might mean."""
    interpretations = []

    if shifts.get("length") == "significantly_shorter":
        interpretations.append(
            "They're sending much shorter messages - might be losing interest, "
            "distracted, or upset"
        )
    elif shifts.get("length") == "significantly_longer":
        interpretations.append(
            "They're writing longer messages - they're more engaged or have "
            "something important to share"
        )

    if shifts.get("emoji") == "stopped_using_emojis":
        interpretations.append(
            "They stopped using emojis - tone might have shifted to more serious"
        )

    if shifts.get("humor") == "stopped_being_funny":
        interpretations.append(
            "They were using humor before but stopped - conversation might be "
            "getting more serious or they're not in the mood"
        )

    return "; ".join(interpretations) if interpretations else "Minor style changes detected"


# ═══════════════════════════════════════════════════════════════
#  4. PERSONALITY SHEET
# ═══════════════════════════════════════════════════════════════

DEFAULT_PERSONALITY = {
    "name": "Conversational Agent",
    "core_traits": {
        "warmth": 0.85,
        "humor": 0.7,
        "playfulness": 0.75,
        "directness": 0.7,
        "emotional_depth": 0.8,
        "intellectual_curiosity": 0.8,
        "confidence": 0.7,
        "vulnerability": 0.6,
        "adaptability": 0.9,
    },
    "communication_style": {
        "default_length": "medium",
        "formality": "casual",
        "emoji_usage": "moderate",
        "humor_type": "witty_and_warm",
        "affection_style": "context_appropriate",
        "conflict_style": "calm_and_direct",
        "discussion_style": "engaged_and_curious",
    },
    "behavioral_rules": [
        "Never sound robotic or use corporate language",
        "Match their energy and adapt to the conversation context",
        "Ask follow-up questions to show genuine interest",
        "Use humor naturally, dont force it to soften things",
        "Be direct about feelings and opinions, never passive-aggressive",
        "Remember and reference shared experiences and past conversations",
        "Admit when you don't know something",
        "Engage authentically with any topic they bring up",
        "Match their tone exactly: aggressive for aggression, enthusiastic for news, real for emotions",
        "Don't force romantic tone when the conversation is about other topics",
    ],
    "forbidden_patterns": [
        "I understand that you feel...",
        "That being said...",
        "I want you to know that...",
        "It's important to note...",
        "I appreciate you sharing...",
        "Firstly... Secondly...",
        "In conclusion...",
        "As an AI...",
        "I'm here for you...",
    ],
    "voice_examples": [
        {
            "context": "They share good news",
            "good": "wait are you serious?? that's amazing!! tell me everything",
            "bad": "That's wonderful news! I'm very happy for you.",
        },
        {
            "context": "They're feeling down",
            "good": "hey, that sounds really rough. i'm here if you want to talk about it",
            "bad": "I understand you're going through a difficult time. It will get better.",
        },
        {
            "context": "They share an opinion on a topic",
            "good": "honestly yeah i see what you mean, but have you thought about it this way",
            "bad": "That's an interesting perspective! I respect your viewpoint.",
        },
        {
            "context": "They're venting about work",
            "good": "ugh that's so annoying. your boss sounds like a nightmare",
            "bad": "I'm sorry to hear about your work situation. Perhaps you should communicate.",
        },
        {
            "context": "Making plans",
            "good": "ok but what if we actually did that tho 👀",
            "bad": "That sounds like a great plan! I would certainly enjoy that.",
        },
        {
            "context": "Discussing a movie/show",
            "good": "nah see that ending was actually genius because think about it",
            "bad": "I found the conclusion to be quite thought-provoking.",
        },
    ],
}


def load_personality(chat_id: Optional[int] = None) -> Dict[str, Any]:
    """Load personality config. Can be customized per-chat."""
    if chat_id:
        path = STYLE_DATA_DIR / f"{chat_id}_personality.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass

    # Check for global personality
    global_path = STYLE_DATA_DIR / "personality.json"
    if global_path.exists():
        try:
            return json.loads(global_path.read_text())
        except Exception:
            pass

    return DEFAULT_PERSONALITY.copy()


def save_personality(personality: Dict[str, Any], chat_id: Optional[int] = None):
    """Save personality config."""
    if chat_id:
        path = STYLE_DATA_DIR / f"{chat_id}_personality.json"
    else:
        path = STYLE_DATA_DIR / "personality.json"
    path.write_text(json.dumps(personality, indent=2, ensure_ascii=False))


def format_personality_for_prompt(personality: Dict[str, Any]) -> str:
    """Format personality sheet for system prompt injection."""
    parts = []

    # Core traits
    traits = personality.get("core_traits", {})
    high_traits = [k for k, v in traits.items() if v > 0.7]
    if high_traits:
        parts.append(
            f"Your personality: {', '.join(t.replace('_', ' ') for t in high_traits)}"
        )

    # Communication style
    style = personality.get("communication_style", {})
    if style:
        parts.append(
            f"Communication: {style.get('formality', 'casual')} tone, "
            f"{style.get('humor_type', 'warm')} humor, "
            f"{style.get('affection_style', 'natural')} affection"
        )

    # Behavioral rules
    rules = personality.get("behavioral_rules", [])
    for rule in rules[:5]:
        parts.append(f"Rule: {rule}")

    # Forbidden patterns
    forbidden = personality.get("forbidden_patterns", [])
    if forbidden:
        parts.append(
            f"NEVER use phrases like: {', '.join(f'\"{f}\"' for f in forbidden[:4])}"
        )

    # Voice examples
    examples = personality.get("voice_examples", [])
    if examples:
        parts.append("\nVoice examples:")
        for ex in examples[:3]:
            parts.append(f"  [{ex['context']}]")
            parts.append(f"  GOOD: \"{ex['good']}\"")
            parts.append(f"  BAD: \"{ex['bad']}\"")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  5. MASTER STYLE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_style_context(
    chat_id: int,
    messages: List[Dict[str, str]],
    incoming_text: str,
) -> Dict[str, Any]:
    """Complete style analysis for response generation.

    Returns style directives, personality sheet, and mirroring guidance.
    """
    # Profile the incoming message
    current_style = profile_message_style(incoming_text)

    # Update their persistent style profile
    their_profile = update_style_profile(chat_id, messages)

    # Load personality
    personality = load_personality(chat_id)

    # Generate style directives
    directives = generate_style_directives(their_profile, current_style, personality)

    # Detect style shifts
    style_shift = detect_style_shift(messages)

    return {
        "current_message_style": current_style,
        "their_style_profile": their_profile,
        "personality": personality,
        "style_directives": directives,
        "style_shift": style_shift,
    }


def format_style_for_prompt(style_context: Dict[str, Any]) -> str:
    """Format style analysis into prompt-ready text."""
    parts = []

    directives = style_context.get("style_directives", {})
    profile = style_context.get("their_style_profile", {})
    personality = style_context.get("personality", {})

    # Their style summary
    summary = profile.get("style_summary", "")
    if summary:
        parts.append(f"Their texting style: {summary}")

    # Mirroring notes
    for note in directives.get("mirroring_notes", [])[:3]:
        parts.append(f"Style match: {note}")

    # Style shift alert
    shift = style_context.get("style_shift")
    if shift:
        parts.append(f"STYLE SHIFT: {shift.get('interpretation', '')}")

    # Target parameters
    parts.append(
        f"Target: {directives.get('target_length', 'medium')} length, "
        f"{directives.get('target_formality', 'casual')} formality, "
        f"{directives.get('emoji_usage', 'moderate')} emoji, "
        f"{directives.get('energy_level', 'matched')} energy"
    )

    # Personality (abbreviated)
    personality_str = format_personality_for_prompt(personality)
    if personality_str:
        parts.append(f"\n{personality_str}")

    return "\n".join(f"- [STYLE] {p}" if not p.startswith("\n") else p for p in parts)


# ═══════════════════════════════════════════════════════════════
#  ENHANCED: Big Five, Love Languages, Digital Body Language
# ═══════════════════════════════════════════════════════════════

def _safe_import_psych_style():
    """Safely import psychological datasets for style engine."""
    try:
        from psychological_datasets import (
            detect_big_five_indicators,
            detect_love_language,
            DIGITAL_BODY_LANGUAGE,
            CULTURAL_PROFILES,
        )
        return {
            "detect_big_five_indicators": detect_big_five_indicators,
            "detect_love_language": detect_love_language,
            "DIGITAL_BODY_LANGUAGE": DIGITAL_BODY_LANGUAGE,
            "CULTURAL_PROFILES": CULTURAL_PROFILES,
        }
    except ImportError:
        return {}


def analyze_big_five(messages: List[Dict[str, str]], sender: str = "them") -> Dict[str, Any]:
    """Detect Big Five (OCEAN) personality traits from message patterns."""
    psych = _safe_import_psych_style()
    if not psych:
        return {"available": False}
    return psych["detect_big_five_indicators"](messages, sender)


def analyze_love_language(messages: List[Dict[str, str]], sender: str = "them") -> Dict[str, Any]:
    """Detect primary love language from message patterns."""
    psych = _safe_import_psych_style()
    if not psych:
        return {"available": False}
    return psych["detect_love_language"](messages, sender)


def detect_digital_body_language(
    messages: List[Dict[str, str]],
    sender: str = "them",
    baseline_avg_length: float = 0,
    baseline_avg_gap_seconds: float = 0,
) -> List[Dict[str, Any]]:
    """Detect digital body language signals (text-based micro-expressions)."""
    psych = _safe_import_psych_style()
    if not psych:
        return []

    import re
    signals = []
    their_msgs = [m for m in messages if m.get("sender", "") == sender]

    if len(their_msgs) < 5:
        return []

    # Calculate baselines from data if not provided
    lengths = [len(m.get("text", "")) for m in their_msgs]
    if not baseline_avg_length:
        baseline_avg_length = sum(lengths) / len(lengths)

    # Recent vs. baseline length
    recent_lengths = lengths[-3:]
    recent_avg = sum(recent_lengths) / len(recent_lengths)

    if baseline_avg_length > 0:
        length_ratio = recent_avg / baseline_avg_length
        if length_ratio < 0.3:
            signals.append({
                "signal": "message_length_decrease",
                "detail": psych["DIGITAL_BODY_LANGUAGE"]["message_length_decrease"],
                "severity": "high",
                "value": f"{length_ratio:.0%} of baseline",
            })
        elif length_ratio > 2.0:
            signals.append({
                "signal": "message_length_increase",
                "detail": psych["DIGITAL_BODY_LANGUAGE"]["message_length_increase"],
                "severity": "medium",
                "value": f"{length_ratio:.0%} of baseline",
            })

    # Check last message for specific signals
    last_text = their_msgs[-1].get("text", "") if their_msgs else ""

    # Period at end of short text
    if re.match(r"^[A-Za-z\s]{1,30}\.$", last_text.strip()):
        signals.append({
            "signal": "period_usage",
            "detail": psych["DIGITAL_BODY_LANGUAGE"]["period_usage"],
            "severity": "low",
        })

    # Ellipsis
    if last_text.strip().endswith("...") or "..." in last_text:
        signals.append({
            "signal": "ellipsis",
            "detail": psych["DIGITAL_BODY_LANGUAGE"]["ellipsis"],
            "severity": "low",
        })

    # ALL CAPS detection
    words = last_text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    if len(caps_words) >= 2:
        signals.append({
            "signal": "all_caps",
            "detail": psych["DIGITAL_BODY_LANGUAGE"]["all_caps"],
            "severity": "medium",
        })

    # Emoji frequency change
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
        r"\u2600-\u26FF\u2700-\u27BF\u2764\uFE0F]"
    )
    baseline_emoji = sum(
        len(emoji_pattern.findall(m.get("text", ""))) for m in their_msgs[:-3]
    ) / max(len(their_msgs) - 3, 1)
    recent_emoji = sum(
        len(emoji_pattern.findall(m.get("text", ""))) for m in their_msgs[-3:]
    ) / 3

    if baseline_emoji > 0 and recent_emoji / baseline_emoji < 0.3:
        signals.append({
            "signal": "emoji_decrease",
            "detail": psych["DIGITAL_BODY_LANGUAGE"]["emoji_decrease"],
            "severity": "medium",
        })

    return signals


def enhanced_style_analysis(
    messages: List[Dict[str, str]],
    chat_id: int,
    sender: str = "them",
) -> Dict[str, Any]:
    """Run enhanced style analysis combining all frameworks."""
    # Original style analysis
    last_text = messages[-1].get("text", "") if messages else ""
    base = analyze_style_context(chat_id, messages, last_text)

    # Enhanced analyses
    big_five = analyze_big_five(messages, sender)
    love_lang = analyze_love_language(messages, sender)
    body_lang = detect_digital_body_language(messages, sender)

    base["big_five_traits"] = big_five
    base["love_language"] = love_lang
    base["digital_body_language"] = body_lang
    base["analysis_version"] = "v5_enhanced"

    return base


def format_enhanced_style_for_prompt(style_context: Dict[str, Any]) -> str:
    """Format enhanced style analysis for prompt injection."""
    parts = []

    # Original style formatting
    base = format_style_for_prompt(style_context)
    if base:
        parts.append(base)

    # Big Five traits
    big_five = style_context.get("big_five_traits", {})
    if big_five and not big_five.get("insufficient_data"):
        trait_strs = []
        for trait, data in big_five.items():
            if isinstance(data, dict) and data.get("level") != "moderate":
                trait_strs.append(f"{trait}: {data['level']}")
        if trait_strs:
            parts.append(f"- [PERSONALITY] Big Five: {', '.join(trait_strs)}")

    # Love Language
    ll = style_context.get("love_language", {})
    if ll.get("primary"):
        parts.append(
            f"- [LOVE LANG] Primary: {ll['primary'].replace('_', ' ')}. "
            f"Tailor responses to this preference."
        )

    # Digital Body Language signals
    body_lang = style_context.get("digital_body_language", [])
    for signal in body_lang[:2]:
        detail = signal.get("detail", {})
        meanings = detail.get("meanings", [])
        if meanings:
            parts.append(
                f"- [BODY LANG] {signal['signal'].replace('_', ' ').title()}: "
                f"possible {', '.join(meanings[:2])} ({signal.get('severity', 'low')} severity)"
            )

    return "\n".join(parts) if parts else ""
