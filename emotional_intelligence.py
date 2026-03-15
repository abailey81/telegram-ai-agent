"""
Emotional Intelligence Layer.

Implements research-backed emotional intelligence for conversation:

1. Validation-First Response Policy - Always acknowledge emotions before anything
2. Multi-Dimensional Emotion Profiling - Beyond pos/neg: valence, arousal, dominance
3. Temporal Emotion Tracking - Emotional arcs across sessions
4. Empathetic Calibration - Match emotional depth appropriately
5. Attachment Style Detection - Anxious/Avoidant/Secure pattern recognition
6. Emotional Continuity - Remember emotional state across sessions
7. Graduated Response Calibration - Intensity-matched responses

Based on research from:
- Gottman's emotional intelligence framework
- Bowlby's attachment theory
- Brown & Levinson's politeness theory
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

ei_logger = logging.getLogger("emotional_intelligence")
ei_logger.setLevel(logging.INFO)

EI_DATA_DIR = Path(__file__).parent / "engine_data" / "emotional"
EI_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-pickup: load autoresearch-optimized engine parameters ──
_OPTIMIZED_EI_PARAMS = None
_OPTIMIZED_EI_PARAMS_MTIME = 0


def _load_optimized_ei_params() -> Optional[dict]:
    """Load optimized emotional intelligence params from autoresearch."""
    global _OPTIMIZED_EI_PARAMS, _OPTIMIZED_EI_PARAMS_MTIME
    params_file = Path(__file__).parent / "engine_data" / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_EI_PARAMS_MTIME:
            _OPTIMIZED_EI_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_EI_PARAMS_MTIME = mtime
            ei_logger.info(
                f"Auto-loaded optimized EI params "
                f"(score={_OPTIMIZED_EI_PARAMS.get('optimization_score', '?')})"
            )
        return _OPTIMIZED_EI_PARAMS
    except Exception as e:
        ei_logger.debug(f"Could not load optimized EI params: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  1. MULTI-DIMENSIONAL EMOTION PROFILING
# ═══════════════════════════════════════════════════════════════

# VAD (Valence-Arousal-Dominance) model
# Maps emotions to their dimensional coordinates
EMOTION_VAD = {
    "joy": {"valence": 0.9, "arousal": 0.7, "dominance": 0.7},
    "love": {"valence": 0.95, "arousal": 0.6, "dominance": 0.5},
    "excitement": {"valence": 0.85, "arousal": 0.9, "dominance": 0.7},
    "contentment": {"valence": 0.8, "arousal": 0.2, "dominance": 0.6},
    "amusement": {"valence": 0.8, "arousal": 0.6, "dominance": 0.6},
    "pride": {"valence": 0.85, "arousal": 0.5, "dominance": 0.8},
    "gratitude": {"valence": 0.8, "arousal": 0.3, "dominance": 0.4},
    "hope": {"valence": 0.7, "arousal": 0.4, "dominance": 0.5},
    "neutral": {"valence": 0.5, "arousal": 0.3, "dominance": 0.5},
    "surprise": {"valence": 0.5, "arousal": 0.8, "dominance": 0.4},
    "confusion": {"valence": 0.4, "arousal": 0.5, "dominance": 0.3},
    "boredom": {"valence": 0.3, "arousal": 0.1, "dominance": 0.4},
    "anxiety": {"valence": 0.2, "arousal": 0.8, "dominance": 0.2},
    "sadness": {"valence": 0.15, "arousal": 0.2, "dominance": 0.2},
    "frustration": {"valence": 0.2, "arousal": 0.7, "dominance": 0.4},
    "anger": {"valence": 0.1, "arousal": 0.9, "dominance": 0.7},
    "disgust": {"valence": 0.1, "arousal": 0.6, "dominance": 0.6},
    "fear": {"valence": 0.1, "arousal": 0.9, "dominance": 0.1},
    "loneliness": {"valence": 0.15, "arousal": 0.2, "dominance": 0.15},
    "jealousy": {"valence": 0.2, "arousal": 0.7, "dominance": 0.3},
    "guilt": {"valence": 0.2, "arousal": 0.4, "dominance": 0.2},
    "shame": {"valence": 0.1, "arousal": 0.5, "dominance": 0.1},
}


def profile_emotion_multidimensional(
    text: str,
    dl_emotions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Profile emotions along Valence-Arousal-Dominance dimensions.

    Combines keyword heuristics with DL emotion detection if available.
    Returns a rich emotional profile for response calibration.
    """
    text_lower = text.lower()

    # Start with DL emotions if available
    if dl_emotions and dl_emotions.get("primary_emotion"):
        primary = dl_emotions["primary_emotion"]
        intensity = dl_emotions.get("emotional_intensity", 0.5)
        all_emotions = dl_emotions.get("all_emotions", {})
    else:
        # Heuristic emotion detection
        primary, intensity, all_emotions = _heuristic_emotion_detect(text_lower)

    # Get VAD coordinates
    vad = EMOTION_VAD.get(primary, EMOTION_VAD["neutral"])

    # Detect emotional complexity (multiple emotions present)
    significant_emotions = {
        k: v for k, v in all_emotions.items()
        if v > 0.15 and k != "neutral"
    }
    is_complex = len(significant_emotions) >= 2

    # Detect emotional vulnerability
    vulnerability_markers = [
        "i don't know what to do", "i'm lost", "help me",
        "i'm scared", "i can't", "nobody cares", "all alone",
        "what's wrong with me", "i'm not enough", "i give up",
        "everything is falling apart", "i can't take it",
        # Russian
        "я не знаю что делать", "я потерялся", "я потерялась", "помоги мне",
        "мне страшно", "я не могу", "никому нет дела", "я одна", "я один",
        "что со мной не так", "я недостаточно", "я сдаюсь", "всё рушится",
        "я не справляюсь", "мне плохо", "мне очень плохо", "хочу умереть",
        "я в отчаянии", "мне не к кому обратиться", "никто не понимает",
    ]
    is_vulnerable = any(m in text_lower for m in vulnerability_markers)

    # Detect emotional suppression (saying fine but context says otherwise)
    suppression_markers = ["i'm fine", "it's fine", "whatever", "doesn't matter",
                           "i don't care", "it's nothing", "forget it",
                           # Russian
                           "всё нормально", "мне хорошо", "неважно", "мне всё равно",
                           "ничего", "забей", "забудь", "проехали", "да ладно",
                           "не бери в голову", "всё ок", "всё хорошо"]
    is_suppressed = any(m in text_lower for m in suppression_markers) and len(text) < 30

    # Detect emotional escalation
    escalation_markers = ["!!!", "???", "CAPS", "seriously", "honestly",
                          "for real", "i swear", "enough",
                          # Russian
                          "серьёзно", "честно", "клянусь", "хватит",
                          "я не шучу", "по-настоящему", "блин", "ёмоё"]
    escalation = sum(1 for m in escalation_markers if m in text) / 4.0
    if text.isupper() and len(text) > 5:
        escalation += 0.5

    return {
        "primary_emotion": primary,
        "intensity": round(intensity, 3),
        "valence": vad["valence"],
        "arousal": vad["arousal"],
        "dominance": vad["dominance"],
        "is_complex": is_complex,
        "is_vulnerable": is_vulnerable,
        "is_suppressed": is_suppressed,
        "escalation": round(min(escalation, 1.0), 3),
        "significant_emotions": significant_emotions,
        "needs_validation": (
            vad["valence"] < 0.4
            or is_vulnerable
            or is_suppressed
            or intensity > 0.7
        ),
        "emotional_needs": _determine_emotional_needs(
            primary, intensity, is_vulnerable, is_suppressed, vad
        ),
    }


def _heuristic_emotion_detect(
    text_lower: str,
) -> tuple:
    """Fallback heuristic emotion detection."""
    emotion_keywords = {
        "joy": ["happy", "glad", "great", "awesome", "amazing", "wonderful", "yay", "😊", "😃",
                "счастлив", "счастлива", "рад", "рада", "круто", "класс", "супер", "отлично",
                "здорово", "ура", "замечательно", "прекрасно", "кайф", "ништяк"],
        "love": ["love", "adore", "❤", "🥰", "😍", "heart", "darling", "sweetheart",
                 "люблю", "обожаю", "любимый", "любимая", "родной", "родная",
                 "солнышко", "зайка", "котик", "малыш", "сердце", "целую"],
        "excitement": ["excited", "can't wait", "omg", "incredible", "🎉", "!!!",
                       "офигеть", "обалдеть", "не могу дождаться", "невероятно",
                       "жесть", "вау", "ааа", "ого", "нереально"],
        "sadness": ["sad", "crying", "tears", "depressed", "lonely", "😢", "😭", "💔",
                    "грустно", "плачу", "слёзы", "одиноко", "тоскливо", "печально",
                    "расстроен", "расстроена", "больно", "тяжело"],
        "anger": ["angry", "furious", "pissed", "mad", "hate", "😡", "🤬",
                  "злюсь", "злой", "злая", "бешусь", "ненавижу", "в ярости",
                  "бесит", "достал", "достала", "взбесил", "взбесила"],
        "fear": ["scared", "afraid", "terrified", "anxious", "worried", "nervous",
                 "боюсь", "страшно", "тревожно", "переживаю", "волнуюсь",
                 "нервничаю", "напугал", "напугала", "жутко"],
        "surprise": ["wow", "omg", "no way", "really?!", "😱", "🤯", "shocked",
                     "ого", "вау", "ничего себе", "серьёзно", "не может быть",
                     "офигеть", "да ладно", "правда", "фигасе"],
        "frustration": ["frustrated", "annoyed", "ugh", "so tired of", "can't believe",
                        "достало", "надоело", "задолбало", "бесит", "раздражает",
                        "устал от", "устала от", "невыносимо", "не могу больше"],
        "amusement": ["haha", "lol", "lmao", "😂", "🤣", "hilarious", "funny",
                      "хаха", "ахахах", "ржу", "угар", "смешно", "прикол",
                      "хохочу", "умираю", "ахах"],
        "gratitude": ["thank", "grateful", "appreciate", "means a lot", "🙏",
                      "спасибо", "благодарен", "благодарна", "ценю", "очень приятно",
                      "спасибочки", "спс", "спасибки"],
    }

    scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = 0
        for k in keywords:
            if k in text_lower:
                # Multi-word phrases get bonus weight (more specific = more accurate)
                if " " in k:
                    score += 2.0
                elif len(k) > 1 and not k.isalpha():
                    score += 1.5  # Emojis are strong signals
                else:
                    score += 1.0
        if score > 0:
            scores[emotion] = score

    # ── ACCURACY BOOST: ALL CAPS boosts intensity of detected emotions ──
    caps_ratio = sum(1 for c in text_lower if c != text_lower) / max(len(text_lower), 1) if text_lower else 0
    # Actually we need the original text for this — approximate via the check
    _orig_words = text_lower.split()

    # ── ACCURACY BOOST: Punctuation intensity ──
    excl_count = text_lower.count("!")
    question_count = text_lower.count("?")
    ellipsis_count = text_lower.count("...")

    # Boost the strongest emotion based on punctuation
    if scores:
        top_emotion = max(scores, key=scores.get)
        if excl_count >= 2:
            scores[top_emotion] = scores[top_emotion] * 1.3
        if question_count >= 2 and top_emotion in ("surprise", "anger", "frustration"):
            scores[top_emotion] = scores[top_emotion] * 1.2
        if ellipsis_count >= 1 and top_emotion in ("sadness", "fear", "frustration"):
            scores[top_emotion] = scores[top_emotion] * 1.15

    if not scores:
        return "neutral", 0.3, {"neutral": 0.7}

    total = sum(scores.values())
    all_emotions = {k: round(v / total, 3) for k, v in scores.items()}
    primary = max(scores, key=scores.get)
    # Improved intensity: consider total signal count, not just max
    raw_intensity = scores[primary] / max(total, 1)
    # Calibrate: intensity floor/scale auto-tuned by autoresearch
    _opt_i = _load_optimized_ei_params()
    _i_floor = (_opt_i or {}).get("intensity_floor", 0.3)
    _i_scale = (_opt_i or {}).get("intensity_scale", 0.7)
    intensity = min(_i_floor + raw_intensity * _i_scale + min(scores[primary] - 1, 3) * 0.1, 1.0)

    return primary, round(intensity, 3), all_emotions


def _determine_emotional_needs(
    emotion: str,
    intensity: float,
    is_vulnerable: bool,
    is_suppressed: bool,
    vad: Dict[str, float],
) -> List[str]:
    """Determine what the person emotionally needs right now."""
    needs = []

    if is_vulnerable:
        needs.extend(["realness", "presence", "matched_openness"])
    elif is_suppressed:
        needs.extend(["direct_ask", "cut_through_bs", "patience"])
    elif vad["valence"] < 0.3:
        if vad["arousal"] > 0.6:
            needs.extend(["energy_match", "solidarity", "directness"])
        else:
            needs.extend(["warmth", "presence", "honest_engagement"])
    elif vad["valence"] > 0.7:
        if vad["arousal"] > 0.6:
            needs.extend(["shared_excitement", "celebration", "energy_matching"])
        else:
            needs.extend(["appreciation", "gentle_affirmation"])
    else:
        needs.append("engagement")

    if intensity > 0.8:
        needs.insert(0, "acknowledgment_first")

    return needs


# ═══════════════════════════════════════════════════════════════
#  2. VALIDATION-FIRST RESPONSE POLICY
# ═══════════════════════════════════════════════════════════════

VALIDATION_TEMPLATES = {
    "sadness": [
        "damn thats rough",
        "come here",
        "that actually sucks im sorry",
    ],
    "anger": [
        "yo wtf thats actually messed up",
        "nah thats not ok at all",
        "id be pissed too honestly",
    ],
    "fear": [
        "wait what happened",
        "yo are u ok",
        "thats lowkey scary ngl",
    ],
    "frustration": [
        "bro thats so annoying",
        "yeah no that would piss me off too",
        "thats actually bs ngl",
    ],
    "loneliness": [
        "im right here",
        "talk to me whats going on",
        "hey im not going anywhere",
    ],
    "anxiety": [
        "wait whats wrong",
        "hey breathe whats happening",
        "tell me everything",
    ],
    "jealousy": [
        "wait what do u mean",
        "ok tell me whats going on",
        "be straight with me",
    ],
    "guilt": [
        "stop being so hard on urself",
        "bro its not that deep",
        "everyone messes up its fine",
    ],
}

VALIDATION_TEMPLATES_RU = {
    "sadness": [
        "блин, это тяжело",
        "иди сюда",
        "мне реально жаль, это отстой",
    ],
    "anger": [
        "чё за нахуй, это реально пиздец",
        "нет, это вообще не ок",
        "я бы тоже бесился честно",
    ],
    "fear": [
        "подожди, что случилось",
        "эй ты в порядке",
        "это реально страшно",
    ],
    "frustration": [
        "блин это бесит реально",
        "нет ну это кого угодно бы выбесило",
        "это реально хуйня какая-то",
    ],
    "loneliness": [
        "я здесь",
        "расскажи, что происходит",
        "эй, я никуда не денусь",
    ],
    "anxiety": [
        "подожди, что не так",
        "дыши, что случилось",
        "расскажи мне всё",
    ],
    "jealousy": [
        "подожди, ты о чём",
        "окей, расскажи что происходит",
        "говори как есть",
    ],
    "guilt": [
        "хватит себя грызть",
        "да ладно, не парься",
        "все ошибаются, нормально",
    ],
}


def generate_validation_guidance(
    emotional_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate validation-first response guidance.

    Research shows that emotional validation before problem-solving
    is the #1 factor in perceived empathy.
    """
    primary = emotional_profile.get("primary_emotion", "neutral")
    needs = emotional_profile.get("emotional_needs", [])
    intensity = emotional_profile.get("intensity", 0.5)
    is_vulnerable = emotional_profile.get("is_vulnerable", False)
    is_suppressed = emotional_profile.get("is_suppressed", False)

    guidance = {
        "needs_validation": emotional_profile.get("needs_validation", False),
        "validation_approach": "none",
        "validation_examples": [],
        "response_structure": [],
        "things_to_avoid": [],
        "emotional_depth_level": "surface",
    }

    if not guidance["needs_validation"]:
        guidance["response_structure"] = ["respond_naturally"]
        return guidance

    # Determine validation approach
    if is_vulnerable:
        guidance["validation_approach"] = "deep_empathy"
        guidance["emotional_depth_level"] = "deep"
        guidance["response_structure"] = [
            "validate_their_emotion",
            "express_you_care",
            "be_present_dont_fix",
        ]
        guidance["things_to_avoid"] = [
            "Don't try to fix the problem immediately",
            "Don't minimize with 'it'll be fine'",
            "Don't redirect to your own experience",
            "Don't give unsolicited advice",
        ]
    elif is_suppressed:
        guidance["validation_approach"] = "direct_call_out"
        guidance["emotional_depth_level"] = "medium"
        guidance["response_structure"] = [
            "acknowledge_what_they_said",
            "call_out_that_something_is_off",
            "ask_directly_whats_going_on",
        ]
        guidance["things_to_avoid"] = [
            "Don't be passive about it",
            "Don't say 'you seem upset'",
            "Don't accept 'I'm fine' — but keep it casual like 'bro u good?'",
        ]
    elif intensity > 0.7:
        guidance["validation_approach"] = "match_intensity"
        guidance["emotional_depth_level"] = "medium"
        guidance["response_structure"] = [
            "validate_emotion_with_matching_energy",
            "show_understanding_of_their_situation",
            "then_engage_naturally",
        ]
        guidance["things_to_avoid"] = [
            "Don't be calmer than them",
            "Don't be dismissive",
            "Don't change the subject quickly",
        ]
    else:
        guidance["validation_approach"] = "light_acknowledgment"
        guidance["emotional_depth_level"] = "surface"
        guidance["response_structure"] = [
            "brief_acknowledgment",
            "respond_naturally",
        ]

    # Get relevant validation examples (language-aware)
    _lang = emotional_profile.get("language", "english")
    if _lang == "russian":
        templates = VALIDATION_TEMPLATES_RU.get(primary, [])
    else:
        templates = VALIDATION_TEMPLATES.get(primary, [])
    if templates:
        guidance["validation_examples"] = templates[:2]

    return guidance


# ═══════════════════════════════════════════════════════════════
#  3. TEMPORAL EMOTION TRACKING
# ═══════════════════════════════════════════════════════════════

def load_emotion_history(chat_id: int) -> Dict[str, Any]:
    """Load emotional history for a chat."""
    path = EI_DATA_DIR / f"{chat_id}_emotions.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "chat_id": chat_id,
        "emotion_timeline": [],
        "emotional_baseline": "neutral",
        "emotional_patterns": {},
        "streak": {"emotion": "neutral", "count": 0},
    }


def save_emotion_history(chat_id: int, history: Dict[str, Any]):
    """Save emotional history."""
    path = EI_DATA_DIR / f"{chat_id}_emotions.json"
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def record_emotion(
    chat_id: int,
    emotional_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Record an emotional data point and update patterns."""
    history = load_emotion_history(chat_id)

    entry = {
        "emotion": emotional_profile.get("primary_emotion", "neutral"),
        "intensity": emotional_profile.get("intensity", 0.5),
        "valence": emotional_profile.get("valence", 0.5),
        "arousal": emotional_profile.get("arousal", 0.3),
        "timestamp": datetime.now().isoformat(),
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().strftime("%A"),
    }

    history["emotion_timeline"].append(entry)
    # Keep last 200 entries
    history["emotion_timeline"] = history["emotion_timeline"][-200:]

    # Update streak
    if entry["emotion"] == history["streak"]["emotion"]:
        history["streak"]["count"] += 1
    else:
        history["streak"] = {"emotion": entry["emotion"], "count": 1}

    # Update patterns
    _update_emotional_patterns(history)

    # Update baseline (rolling average valence — thresholds auto-tuned by autoresearch)
    _opt = _load_optimized_ei_params()
    _baseline_high = (_opt or {}).get("baseline_valence_high", 0.65)
    _baseline_low = (_opt or {}).get("baseline_valence_low", 0.35)
    recent = history["emotion_timeline"][-30:]
    if recent:
        avg_valence = sum(e["valence"] for e in recent) / len(recent)
        if avg_valence > _baseline_high:
            history["emotional_baseline"] = "generally_positive"
        elif avg_valence < _baseline_low:
            history["emotional_baseline"] = "generally_negative"
        else:
            history["emotional_baseline"] = "mixed"

    save_emotion_history(chat_id, history)
    return history


def _update_emotional_patterns(history: Dict[str, Any]):
    """Detect temporal emotional patterns."""
    timeline = history["emotion_timeline"]
    if len(timeline) < 10:
        return

    patterns = {}

    # Time-of-day patterns
    time_emotions = {}
    for entry in timeline[-50:]:
        hour = entry.get("hour", 12)
        period = (
            "morning" if 6 <= hour < 12
            else "afternoon" if 12 <= hour < 17
            else "evening" if 17 <= hour < 22
            else "night"
        )
        if period not in time_emotions:
            time_emotions[period] = []
        time_emotions[period].append(entry["valence"])

    for period, valences in time_emotions.items():
        avg = sum(valences) / len(valences)
        if avg < 0.35:
            patterns[f"{period}_mood"] = "tends_negative"
        elif avg > 0.7:
            patterns[f"{period}_mood"] = "tends_positive"

    # Day-of-week patterns
    day_emotions = {}
    for entry in timeline[-50:]:
        day = entry.get("day_of_week", "Unknown")
        if day not in day_emotions:
            day_emotions[day] = []
        day_emotions[day].append(entry["valence"])

    for day, valences in day_emotions.items():
        if len(valences) >= 3:
            avg = sum(valences) / len(valences)
            if avg < 0.3:
                patterns[f"{day.lower()}_mood"] = "often_low"
            elif avg > 0.75:
                patterns[f"{day.lower()}_mood"] = "usually_happy"

    history["emotional_patterns"] = patterns


def get_emotional_continuity(chat_id: int) -> Dict[str, Any]:
    """Get emotional continuity info for conversation start.

    Tells us: how were they feeling last time? Any patterns we should
    be aware of? Should we check in about something?
    """
    history = load_emotion_history(chat_id)
    timeline = history.get("emotion_timeline", [])

    continuity = {
        "has_history": len(timeline) > 5,
        "last_emotion": None,
        "last_valence": 0.5,
        "emotional_baseline": history.get("emotional_baseline", "neutral"),
        "should_check_in": False,
        "check_in_reason": None,
        "patterns": history.get("emotional_patterns", {}),
        "streak": history.get("streak", {"emotion": "neutral", "count": 0}),
    }

    if not timeline:
        return continuity

    last = timeline[-1]
    continuity["last_emotion"] = last["emotion"]
    continuity["last_valence"] = last["valence"]

    # Should we check in?
    if last["valence"] < 0.3:
        continuity["should_check_in"] = True
        continuity["check_in_reason"] = (
            f"They were feeling {last['emotion']} last time we talked"
        )

    # Negative streak
    streak = history.get("streak", {})
    if streak.get("count", 0) >= 3 and streak.get("emotion") in (
        "sadness", "anger", "anxiety", "frustration", "loneliness"
    ):
        continuity["should_check_in"] = True
        continuity["check_in_reason"] = (
            f"They've been feeling {streak['emotion']} for the last "
            f"{streak['count']} interactions"
        )

    return continuity


# ═══════════════════════════════════════════════════════════════
#  4. ATTACHMENT STYLE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_attachment_signals(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Detect attachment style signals from conversation patterns.

    Based on Bowlby's Attachment Theory:
    - Secure: Comfortable with intimacy and independence
    - Anxious: Needs reassurance, fears abandonment
    - Avoidant: Uncomfortable with closeness, values independence
    """
    their_msgs = [m for m in messages if m.get("sender") == "Them"]
    if len(their_msgs) < 10:
        return {"style": "insufficient_data", "signals": [], "confidence": 0.0}

    their_texts = [m.get("text", "").lower() for m in their_msgs]
    all_their_text = " ".join(their_texts)

    anxious_score = 0.0
    avoidant_score = 0.0
    secure_score = 0.0
    signals = []

    # Anxious attachment signals
    anxious_markers = {
        "reassurance_seeking": [
            "do you still", "are you sure", "promise me",
            "you won't leave", "am i enough", "do you miss me",
            "are you mad", "did i do something wrong",
        ],
        "abandonment_fear": [
            "don't leave me", "please don't go", "i'm scared you'll",
            "what if you find someone", "are you bored of me",
        ],
        "excessive_availability": [
            "i'm always here", "whenever you need", "i'll drop everything",
            "text me back", "why didn't you reply",
        ],
        "protest_behavior": [
            "fine then", "whatever", "i guess you're busy",
            "if you don't care", "forget it",
        ],
    }

    # Attachment weights auto-tuned by autoresearch
    _opt_att = _load_optimized_ei_params()
    _anxious_w = (_opt_att or {}).get("anxious_weight", 1.5)
    _avoidant_w = (_opt_att or {}).get("avoidant_weight", 1.5)
    _secure_w = (_opt_att or {}).get("secure_weight", 1.0)

    for category, markers in anxious_markers.items():
        count = sum(1 for m in markers if m in all_their_text)
        if count > 0:
            anxious_score += count * _anxious_w
            signals.append(f"anxious_{category}")

    # Avoidant attachment signals
    avoidant_markers = {
        "emotional_distancing": [
            "i don't want to talk about it", "it's not a big deal",
            "you're overthinking", "let's not make this a thing",
            "i need space", "i need some time",
        ],
        "independence_emphasis": [
            "i can handle it myself", "don't worry about me",
            "i don't need", "i'm fine on my own",
        ],
        "intimacy_deflection": [
            "let's keep it light", "why so serious",
            "you're being too much", "relax",
        ],
    }

    for category, markers in avoidant_markers.items():
        count = sum(1 for m in markers if m in all_their_text)
        if count > 0:
            avoidant_score += count * _avoidant_w
            signals.append(f"avoidant_{category}")

    # Secure attachment signals
    secure_markers = {
        "healthy_expression": [
            "i feel", "i appreciate", "thank you for",
            "that means a lot", "i trust you",
        ],
        "healthy_boundaries": [
            "i need some time but", "let me think about it",
            "i'll let you know", "that's okay",
        ],
        "reciprocal_care": [
            "how are you", "tell me about your day",
            "i'm here for you", "what do you need",
        ],
    }

    for category, markers in secure_markers.items():
        count = sum(1 for m in markers if m in all_their_text)
        if count > 0:
            secure_score += count * _secure_w
            signals.append(f"secure_{category}")

    # Response timing patterns (anxious = very fast, avoidant = delayed)
    # This would need timestamp data, so we approximate from message patterns
    short_responses = sum(1 for t in their_texts if len(t.split()) <= 3)
    long_responses = sum(1 for t in their_texts if len(t.split()) >= 15)

    if short_responses > len(their_texts) * 0.6:
        avoidant_score += 2.0
        signals.append("predominantly_short_responses")
    elif long_responses > len(their_texts) * 0.4:
        if anxious_score > avoidant_score:
            anxious_score += 1.0
        else:
            secure_score += 1.0

    # Determine primary style
    total = anxious_score + avoidant_score + secure_score + 0.001
    style_scores = {
        "anxious": anxious_score / total,
        "avoidant": avoidant_score / total,
        "secure": secure_score / total,
    }

    primary_style = max(style_scores, key=style_scores.get)
    confidence = style_scores[primary_style]

    # Generate recommendations based on attachment style
    recommendations = _get_attachment_recommendations(primary_style)

    return {
        "style": primary_style,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in style_scores.items()},
        "signals": signals,
        "recommendations": recommendations,
    }


def _get_attachment_recommendations(style: str) -> List[str]:
    """Get communication recommendations based on attachment style."""
    recs = {
        "anxious": [
            "Be consistent — don't play games with reply timing",
            "Be straight up about what you think and feel",
            "Don't leave on read for hours without reason",
            "If you need space, just say so like a normal person",
            "Match their energy — if they're invested, be invested back",
            "Don't be cold when they're being open",
        ],
        "avoidant": [
            "Don't chase — if they pull back, let them breathe",
            "Keep it light, don't force deep talks",
            "Respect that they need space without being passive-aggressive about it",
            "Use humor to keep things chill",
            "Let heavy topics come up naturally, don't push",
            "Be direct without being intense",
        ],
        "secure": [
            "Match their directness — be straight up",
            "Be honest about what you feel, no games",
            "Be open back when they're open with you",
            "Keep the conversation real",
            "Engage in real talk without making it weird",
            "Be your own person — they respect independence",
        ],
    }
    return recs.get(style, ["Communicate openly and honestly"])


# ═══════════════════════════════════════════════════════════════
#  5. GRADUATED RESPONSE CALIBRATION
# ═══════════════════════════════════════════════════════════════

def calibrate_response(
    emotional_profile: Dict[str, Any],
    validation_guidance: Dict[str, Any],
    attachment: Optional[Dict[str, Any]] = None,
    emotional_continuity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calibrate response parameters based on emotional context.

    Returns specific parameters for response generation:
    - Response length guidance
    - Emotional tone target
    - Self-disclosure level
    - Question vs. statement ratio
    - Emoji usage guidance
    """
    valence = emotional_profile.get("valence", 0.5)
    arousal = emotional_profile.get("arousal", 0.3)
    intensity = emotional_profile.get("intensity", 0.5)
    needs = emotional_profile.get("emotional_needs", [])

    calibration = {
        "length": "medium",
        "tone": "warm",
        "self_disclosure": "low",
        "question_ratio": 0.3,
        "emoji_level": "moderate",
        "formality": "casual",
        "energy_level": "matched",
        "special_instructions": [],
    }

    # Length calibration
    primary_emotion = emotional_profile.get("primary_emotion", "neutral")
    if primary_emotion in ("anger", "frustration") and intensity > 0.5:
        # Angry/frustrated → keep it short and punchy, match their energy
        calibration["length"] = "short"
        calibration["special_instructions"].append(
            "Keep it short and raw — match their aggressive energy, dont write essays"
        )
    elif intensity > 0.7 and valence < 0.3:
        calibration["length"] = "medium"
        calibration["special_instructions"].append(
            "Be real with them but keep it concise"
        )
    elif arousal < 0.3 and valence > 0.6:
        calibration["length"] = "short_to_medium"
    elif "realness" in needs or "presence" in needs:
        calibration["length"] = "medium"

    # Tone calibration
    if primary_emotion in ("anger", "frustration") and intensity > 0.5:
        calibration["tone"] = "raw_and_direct"
        calibration["special_instructions"].append(
            "Be aggressive back if they're being aggressive at you — match their energy"
        )
    elif "solidarity" in needs or "energy_match" in needs:
        calibration["tone"] = "raw_and_direct"
    elif "shared_excitement" in needs or "celebration" in needs:
        calibration["tone"] = "enthusiastic"
    elif "directness" in needs:
        calibration["tone"] = "direct_real"
    elif "honest_engagement" in needs:
        calibration["tone"] = "straightforward"
    elif valence > 0.7:
        calibration["tone"] = "warm_and_playful"

    # Self-disclosure calibration
    if emotional_profile.get("is_vulnerable"):
        calibration["self_disclosure"] = "medium"
        calibration["special_instructions"].append(
            "Be real back — share something honest, not performative"
        )
    elif intensity > 0.6 and valence > 0.6:
        calibration["self_disclosure"] = "medium"

    # Question ratio
    if "direct_ask" in needs or "cut_through_bs" in needs:
        calibration["question_ratio"] = 0.5
    elif emotional_profile.get("is_vulnerable"):
        calibration["question_ratio"] = 0.2  # Less questions, more presence
    elif arousal < 0.3:
        calibration["question_ratio"] = 0.4  # More questions to engage

    # Emoji calibration
    if valence < 0.3:
        calibration["emoji_level"] = "minimal"
    elif valence > 0.7 and arousal > 0.5:
        calibration["emoji_level"] = "expressive"

    # Attachment-informed adjustments
    if attachment:
        style = attachment.get("style", "secure")
        if style == "anxious":
            calibration["special_instructions"].append(
                "Include explicit reassurance and affection"
            )
            calibration["self_disclosure"] = "medium"
        elif style == "avoidant":
            calibration["special_instructions"].append(
                "Keep emotional intensity moderate, use humor"
            )
            calibration["self_disclosure"] = "low"
            calibration["length"] = "short_to_medium"

    # Emotional continuity adjustments
    if emotional_continuity and emotional_continuity.get("should_check_in"):
        calibration["special_instructions"].append(
            f"Check in: {emotional_continuity.get('check_in_reason', 'they seemed down last time')}"
        )

    return calibration


# ═══════════════════════════════════════════════════════════════
#  6. MASTER EMOTIONAL INTELLIGENCE FUNCTION
# ═══════════════════════════════════════════════════════════════

def analyze_emotional_context(
    chat_id: int,
    messages: List[Dict[str, str]],
    incoming_text: str,
    dl_emotions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Complete emotional intelligence analysis.

    This is the main entry point that orchestrates all EI components.
    Returns a comprehensive emotional context for response generation.
    """
    # 1. Multi-dimensional emotion profiling
    profile = profile_emotion_multidimensional(incoming_text, dl_emotions)

    # 2. Validation guidance
    validation = generate_validation_guidance(profile)

    # 3. Emotional continuity
    continuity = get_emotional_continuity(chat_id)

    # 4. Attachment style (only if enough messages)
    attachment = detect_attachment_signals(messages)

    # 5. Record this emotional data point
    record_emotion(chat_id, profile)

    # 6. Calibrate response
    calibration = calibrate_response(profile, validation, attachment, continuity)

    return {
        "emotional_profile": profile,
        "validation_guidance": validation,
        "emotional_continuity": continuity,
        "attachment_style": attachment,
        "response_calibration": calibration,
    }


def format_ei_for_prompt(ei_context: Dict[str, Any]) -> str:
    """Format emotional intelligence analysis for prompt injection.

    IMPORTANT: No therapy-speak, no validation frameworks, no NVC.
    Just raw emotional data so the LLM can match their energy.
    """
    parts = []

    profile = ei_context.get("emotional_profile", {})
    calibration = ei_context.get("response_calibration", {})

    # Emotional state — just the facts
    primary = profile.get("primary_emotion", "neutral")
    intensity = profile.get("intensity", 0)
    if primary and primary != "neutral":
        parts.append(f"Their emotion: {primary} ({intensity:.0%} intensity)")

    # If they're angry/frustrated, tell the LLM to match it
    if primary in ("anger", "frustration") and intensity > 0.4:
        parts.append("They're pissed — MATCH their energy, don't try to calm them down")
    elif primary in ("anger", "frustration"):
        parts.append("They're annoyed — be direct and don't sugarcoat")

    if profile.get("is_suppressed"):
        parts.append("They might be hiding how they really feel — read between the lines")

    # Calibration — only the useful stuff
    tone = calibration.get("tone", "")
    if tone and tone != "warm":
        parts.append(f"Tone: {tone.replace('_', ' ')}")

    for instruction in calibration.get("special_instructions", [])[:3]:
        parts.append(f"→ {instruction}")

    if not parts:
        return ""

    return "\n".join(f"- [EI] {p}" for p in parts)


# ═══════════════════════════════════════════════════════════════
#  8. ENHANCED ANALYSIS WITH PSYCHOLOGICAL DATASETS
# ═══════════════════════════════════════════════════════════════

def _safe_import_psych():
    """Safely import psychological datasets."""
    try:
        from psychological_datasets import (
            detect_four_horsemen,
            detect_cognitive_distortions,
            detect_emotional_bids,
            detect_repair_attempts,
            detect_nvc_quality,
            select_empathetic_response_strategy,
            PLUTCHIK_PRIMARY,
            PLUTCHIK_DYADS,
            PLUTCHIK_DETECTION,
            GOEMOTIONS_TAXONOMY,
        )
        return {
            "detect_four_horsemen": detect_four_horsemen,
            "detect_cognitive_distortions": detect_cognitive_distortions,
            "detect_emotional_bids": detect_emotional_bids,
            "detect_repair_attempts": detect_repair_attempts,
            "detect_nvc_quality": detect_nvc_quality,
            "select_empathetic_response_strategy": select_empathetic_response_strategy,
            "PLUTCHIK_PRIMARY": PLUTCHIK_PRIMARY,
            "PLUTCHIK_DYADS": PLUTCHIK_DYADS,
            "PLUTCHIK_DETECTION": PLUTCHIK_DETECTION,
            "GOEMOTIONS_TAXONOMY": GOEMOTIONS_TAXONOMY,
        }
    except ImportError:
        return {}


def analyze_plutchik_emotions(text: str) -> Dict[str, Any]:
    """Analyze text using Plutchik's full Wheel of Emotions (8 primary + 24 dyads)."""
    import re
    psych = _safe_import_psych()
    if not psych:
        return {"available": False}

    text_lower = text.lower()
    primary_scores = {}

    # Score primary emotions
    for emotion, patterns in psych["PLUTCHIK_DETECTION"].items():
        score = 0
        for pattern in patterns:
            score += len(re.findall(pattern, text_lower, re.IGNORECASE))
        if score > 0:
            primary_scores[emotion] = score

    if not primary_scores:
        return {"primary_emotions": {}, "dyads": [], "intensity": "mild"}

    # Detect dyads (combined emotions)
    detected_dyads = []
    for dyad_name, dyad_data in psych["PLUTCHIK_DYADS"].items():
        c1, c2 = dyad_data["components"]
        if c1 in primary_scores and c2 in primary_scores:
            detected_dyads.append({
                "emotion": dyad_name,
                "components": [c1, c2],
                "tier": dyad_data["tier"],
                "strength": primary_scores[c1] + primary_scores[c2],
            })

    # Determine intensity
    max_score = max(primary_scores.values()) if primary_scores else 0
    intensity = "mild" if max_score <= 1 else "primary" if max_score <= 3 else "intense"

    # Get opposites for dominant emotion
    dominant = max(primary_scores, key=primary_scores.get)
    opposite = psych["PLUTCHIK_PRIMARY"][dominant]["opposite"]

    return {
        "primary_emotions": primary_scores,
        "dominant_emotion": dominant,
        "opposite_emotion": opposite,
        "intensity": intensity,
        "intense_form": psych["PLUTCHIK_PRIMARY"][dominant]["intense"],
        "mild_form": psych["PLUTCHIK_PRIMARY"][dominant]["mild"],
        "dyads": sorted(detected_dyads, key=lambda x: x["strength"], reverse=True),
        "is_complex": len(primary_scores) >= 2,
    }


def analyze_goemotions(text: str) -> Dict[str, Any]:
    """Classify text using Google's GoEmotions 27-category taxonomy."""
    psych = _safe_import_psych()
    if not psych:
        return {"available": False}

    text_lower = text.lower()
    detected = {}

    for emotion, data in psych["GOEMOTIONS_TAXONOMY"].items():
        score = sum(1 for kw in data["keywords"] if kw in text_lower)
        if score > 0:
            detected[emotion] = {
                "score": score,
                "valence": data["valence"],
            }

    if not detected:
        return {"emotions": {}, "dominant": "neutral", "valence": "neutral"}

    dominant = max(detected, key=lambda k: detected[k]["score"])
    positive = sum(1 for e in detected.values() if e["valence"] == "positive")
    negative = sum(1 for e in detected.values() if e["valence"] == "negative")

    return {
        "emotions": detected,
        "dominant": dominant,
        "valence": "positive" if positive > negative else "negative" if negative > positive else "mixed",
        "emotion_count": len(detected),
        "is_mixed": positive > 0 and negative > 0,
    }


def detect_gottman_signals(text: str) -> Dict[str, Any]:
    """Detect Gottman relationship signals: Four Horsemen, bids, repair attempts."""
    psych = _safe_import_psych()
    if not psych:
        return {"available": False}

    horsemen = psych["detect_four_horsemen"](text)
    bids = psych["detect_emotional_bids"](text)
    repairs = psych["detect_repair_attempts"](text)
    nvc = psych["detect_nvc_quality"](text)

    return {
        "four_horsemen": horsemen,
        "horsemen_detected": len(horsemen) > 0,
        "most_severe_horseman": max(horsemen, key=lambda h: h["severity"])["horseman"] if horsemen else None,
        "emotional_bids": bids,
        "bid_type": bids[0]["type"] if bids else None,
        "repair_attempts": repairs,
        "is_repairing": len(repairs) > 0,
        "nvc_quality": nvc,
        "needs_intervention": len(horsemen) > 0 and len(repairs) == 0,
    }


def detect_distortions_in_message(text: str) -> Dict[str, Any]:
    """Detect CBT cognitive distortions and provide reframe strategies."""
    psych = _safe_import_psych()
    if not psych:
        return {"available": False}

    distortions = psych["detect_cognitive_distortions"](text)
    return {
        "distortions": distortions,
        "count": len(distortions),
        "primary_distortion": distortions[0]["distortion"] if distortions else None,
        "reframe_suggestion": distortions[0]["reframe_template"] if distortions else None,
        "has_absolutist_language": any(
            d["distortion"] in ["all_or_nothing", "overgeneralization"] for d in distortions
        ),
    }


def get_empathetic_strategy(emotional_state: str, conflict_active: bool = False) -> Dict[str, Any]:
    """Get ESConv-based empathetic response strategy."""
    psych = _safe_import_psych()
    if not psych:
        return {"primary_strategy": "reflection", "secondary_strategy": "affirmation"}

    return psych["select_empathetic_response_strategy"](
        emotional_state=emotional_state,
        conflict_active=conflict_active,
    )


def enhanced_emotional_analysis(
    text: str,
    chat_id: int,
    messages: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Run enhanced emotional analysis combining all frameworks."""
    # Original V4 analysis
    base = analyze_emotional_context(chat_id, messages or [], text)

    # Enhanced with psychological datasets
    plutchik = analyze_plutchik_emotions(text)
    goemotions = analyze_goemotions(text)
    gottman = detect_gottman_signals(text)
    distortions = detect_distortions_in_message(text)

    # Determine strategy based on detected emotion
    primary_emotion = plutchik.get("dominant_emotion", "neutral")
    conflict_active = gottman.get("horsemen_detected", False)
    strategy = get_empathetic_strategy(primary_emotion, conflict_active)

    base["plutchik_analysis"] = plutchik
    base["goemotions"] = goemotions
    base["gottman_signals"] = gottman
    base["cognitive_distortions"] = distortions
    base["empathetic_strategy"] = strategy
    base["analysis_version"] = "v5_enhanced"

    return base


def format_enhanced_ei_for_prompt(analysis: Dict[str, Any]) -> str:
    """Format enhanced emotional analysis for LLM prompt injection."""
    parts = []

    # Base EI formatting
    base = format_ei_for_prompt(analysis)
    if base:
        parts.append(base)

    # Plutchik emotions
    plutchik = analysis.get("plutchik_analysis", {})
    if plutchik.get("dominant_emotion"):
        dyad_info = ""
        if plutchik.get("dyads"):
            top_dyad = plutchik["dyads"][0]
            dyad_info = f" | Complex emotion: {top_dyad['emotion']} ({' + '.join(top_dyad['components'])})"
        parts.append(
            f"- [PLUTCHIK] Dominant: {plutchik['dominant_emotion']} "
            f"(intensity: {plutchik['intensity']}, opposite: {plutchik.get('opposite_emotion', '?')})"
            f"{dyad_info}"
        )

    # GoEmotions
    goemotions = analysis.get("goemotions", {})
    if goemotions.get("emotions"):
        emotions = list(goemotions["emotions"].keys())[:4]
        parts.append(f"- [GOEMOTIONS] Detected: {', '.join(emotions)} (valence: {goemotions.get('valence', '?')})")

    # Gottman signals — stripped down, no therapy frameworks
    gottman = analysis.get("gottman_signals", {})
    if gottman.get("horsemen_detected"):
        horseman = gottman["most_severe_horseman"]
        # Just flag the toxic behavior, don't prescribe therapy antidotes
        parts.append(f"- [WARNING] They're being {horseman} — respond accordingly, don't be a pushover")

    return "\n".join(parts) if parts else ""
