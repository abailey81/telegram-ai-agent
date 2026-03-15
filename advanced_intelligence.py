"""
Advanced Intelligence Engine — Research-driven sophistication layer.

Implements ALL findings from deep research into SOTA conversational AI:
1. GoEmotions 28-label emotion detection (SamLowe/roberta-base-go_emotions)
2. Hidden Reasoning pre-step (chain-of-thought before generation)
3. Best-of-N response selection with multi-dimensional scoring
4. Subtext Detection (NLI-based pragmatic inference)
5. Conversation Risk Detector (escalation/withdrawal/ghosting)
6. Uncanny Valley fixes (natural texting post-processing)
7. Emoji Pattern Analysis + Temporal Signal Analysis
8. Personality Profiling (Big Five + Attachment Style from text)
9. Response Quality Scorer (multi-dimensional)
10. DSPy-style prompt self-optimization (engagement tracking)
11. Persona Consistency scoring (character break detection)

All features are designed to be called from telegram_api.py's pipeline.
Graceful degradation: if HuggingFace models aren't installed, falls back to heuristics.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import math
import random
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import Counter, deque

ai_logger = logging.getLogger("advanced_intelligence")
ai_logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════
#  1. GoEmotions 28-Label Emotion Detection
# ═══════════════════════════════════════════════════════════════

_go_emotions_pipeline = None
_go_emotions_available = None  # None = not checked yet


def _detect_device_index() -> int:
    """Detect best GPU device for HuggingFace pipeline (-1=CPU, 0=GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 0  # MPS (Apple Silicon)
    except ImportError:
        pass
    return -1


def _load_go_emotions():
    """Lazy-load the GoEmotions 28-label model. Uses GPU (CUDA/MPS) if available."""
    global _go_emotions_pipeline, _go_emotions_available
    if _go_emotions_available is not None:
        return _go_emotions_available
    try:
        from transformers import pipeline
        device_idx = _detect_device_index()
        _go_emotions_pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=device_idx,
        )
        _go_emotions_available = True
        device_name = "GPU" if device_idx >= 0 else "CPU"
        ai_logger.info(f"GoEmotions 28-label model loaded ({device_name})")
        return True
    except Exception as e:
        ai_logger.warning(f"GoEmotions model not available: {e}")
        _go_emotions_available = False
        return False


# Heuristic 28-emotion fallback when model isn't installed
_HEURISTIC_EMOTION_PATTERNS = {
    "admiration": [
        "wow", "amazing", "incredible", "brilliant", "genius", "impressive", "respect",
        # Russian
        "восхищаюсь", "потрясающе", "невероятно", "гениально", "впечатляет", "уважаю", "класс",
    ],
    "amusement": [
        "haha", "lol", "lmao", "😂", "🤣", "funny", "hilarious", "💀",
        # Russian
        "хаха", "ахах", "ахахах", "лол", "ржу", "смешно", "угар", "ору", "кек", "ппц смешно",
    ],
    "anger": [
        "angry", "mad", "furious", "pissed", "hate", "fuck", "fucking", "stfu", "gtfo",
        "bitch", "asshole", "bastard", "dickhead", "shit", "bullshit", "damn", "moron",
        "idiot", "dumbass", "loser", "screw you", "go to hell", "drop dead",
        "piss off", "wtf", "get lost", "hate you", "die", "kill",
        # Russian profanity / insults
        "бесит", "злюсь", "злой", "злая", "ненавижу",
        "блядь", "сука", "пиздец", "нахуй", "нахер", "ёбаный", "ебаный",
        "гандон", "мудак", "дебил", "долбоёб", "долбоеб", "тварь", "урод",
        "козёл", "козел", "скотина", "чмо", "придурок", "кретин", "идиот",
        "идиотка", "дура", "дурак", "лох", "отстой", "ублюдок", "выродок",
        "мразь", "подонок", "шлюха", "сволочь", "гнида", "падла", "паскуда",
        "хуй", "хуйня", "пизда", "ебать", "заебал", "заебала", "отъебись",
        "уёбок", "уебок", "уёбище", "пиздабол", "мудила", "хуесос",
        "пидор", "пидорас", "сучка", "сучара", "блять", "ёб", "еб",
        "засранец", "засранка", "говно", "говнюк", "дерьмо",
        "заткнись", "отвали", "вали", "проваливай", "пошёл", "пошел", "пошла",
        "катись", "иди", "убирайся", "свали",
    ],
    "annoyance": [
        "ugh", "annoying", "irritating", "whatever", "bruh", "smh", "ffs",
        "seriously", "come on", "for real", "give me a break", "knock it off",
        "stop it", "enough", "i cant", "jfc", "omfg",
        # Russian
        "достал", "надоел", "достала", "надоела", "задолбал", "задолбала",
        "раздражает", "заколебал", "заколебала", "запарил", "запарила",
        "бесишь", "утомил", "утомила", "напрягает", "замучил", "замучила",
        "блин", "ёлки", "фиг", "чёрт", "капец", "атас", "жесть",
    ],
    "approval": [
        "good", "nice", "great", "perfect", "exactly", "yes", "right", "👍",
        # Russian
        "молодец", "отлично", "правильно", "верно", "супер", "здорово", "так держать", "хорошо",
    ],
    "caring": [
        "take care", "be safe", "hope you're ok", "worried about",
        # Russian
        "береги себя", "как ты", "ты в порядке", "переживаю за тебя", "позаботься о себе",
        "не болей", "выздоравливай", "осторожнее",
    ],
    "confusion": [
        "what", "huh", "confused", "don't understand", "wdym",
        # Russian
        "что", "не понял", "не поняла", "не понимаю", "в смысле", "как так", "э", "чё",
    ],
    "curiosity": [
        "how", "why", "tell me", "what happened", "interesting",
        # Russian
        "как", "расскажи", "интересно", "что случилось", "а почему", "а как", "любопытно",
    ],
    "desire": [
        "want", "wish", "need", "crave", "dying to",
        # Russian
        "хочу", "мечтаю", "жажду", "так хочется", "нужно", "желаю",
    ],
    "disappointment": [
        "disappointed", "let down", "expected more",
        # Russian
        "разочарован", "разочарована", "обидно", "ожидал большего", "ожидала большего", "зря надеялся",
    ],
    "disapproval": [
        "wrong", "bad", "no", "shouldn't", "disagree", "terrible", "awful", "horrible",
        "pathetic", "ridiculous", "absurd", "unacceptable", "garbage", "trash",
        "worthless", "useless", "incompetent", "shameful", "disgraceful",
        # Russian
        "нет", "неправильно", "плохо", "ужасно", "отвратительно", "никуда не годится",
        "позор", "стыдно", "жалкий", "убогий", "бездарь", "ничтожество",
        "тупица", "неудачник", "бестолочь", "бесполезный", "бесполезная",
        "никчёмный", "никчемный", "клоун", "посмешище", "позорище",
    ],
    "disgust": [
        "gross", "disgusting", "eww", "nasty", "🤮", "vile", "repulsive",
        "revolting", "sickening", "vomit", "makes me sick", "sick of",
        "cant stand", "stomach turning",
        # Russian
        "фу", "отвратительно", "тошнит", "мерзко", "гадость", "гадко",
        "мерзость", "омерзительно", "блевать", "рвотный", "противно",
        "тьфу", "бе", "фубля",
    ],
    "embarrassment": [
        "embarrassing", "awkward", "cringe", "oops",
        # Russian
        "стыдно", "неловко", "стыд", "позорище", "краснею", "ой",
    ],
    "excitement": [
        "omg", "yay", "can't wait", "so excited", "!!!",
        # Russian
        "ура", "не могу ждать", "ааа", "вау", "класс", "круто", "офигеть", "обалдеть",
    ],
    "fear": [
        "scared", "afraid", "terrified", "worried", "anxious",
        # Russian
        "боюсь", "страшно", "жутко", "ужас", "тревожно", "пугает", "в ужасе",
    ],
    "gratitude": [
        "thank", "thanks", "appreciate", "grateful",
        # Russian
        "спасибо", "благодарю", "спс", "благодарна", "благодарен", "ценю",
    ],
    "grief": [
        "lost", "gone", "miss them", "passed away", "mourning",
        # Russian
        "потерял", "потеряла", "ушёл", "ушел", "не стало", "скорблю", "горюю",
    ],
    "joy": [
        "happy", "glad", "yay", "wonderful", "love it",
        # Russian
        "счастлив", "счастлива", "рад", "рада", "радость", "прекрасно", "замечательно", "кайф",
    ],
    "love": [
        "love you", "adore", "❤️", "😍", "🥰", "💕",
        # Russian
        "люблю", "обожаю", "любимый", "любимая", "милый", "милая", "родной", "родная",
    ],
    "nervousness": [
        "nervous", "anxious", "worried", "stressed",
        # Russian
        "нервничаю", "волнуюсь", "переживаю", "стресс", "напряжён", "дёргаюсь", "на нервах",
    ],
    "optimism": [
        "hopefully", "fingers crossed", "looking forward",
        # Russian
        "надеюсь", "всё будет", "все будет хорошо", "с оптимизмом", "верю", "будет лучше",
    ],
    "pride": [
        "proud", "nailed it", "killed it", "accomplished",
        # Russian
        "горжусь", "гордость", "добился", "добилась", "справился", "справилась", "смог", "смогла",
    ],
    "realization": [
        "oh", "ohh", "I see", "now I get it", "makes sense",
        # Russian
        "а", "понял", "поняла", "ааа", "вот оно что", "теперь ясно", "дошло", "ясно",
    ],
    "relief": [
        "phew", "thank god", "relieved", "finally",
        # Russian
        "слава богу", "наконец", "фух", "уф", "отлегло", "выдохнул", "выдохнула",
    ],
    "remorse": [
        "sorry", "apologize", "my fault", "shouldn't have",
        # Russian
        "извини", "прости", "виноват", "виновата", "мне жаль", "сожалею", "моя вина", "прощения",
    ],
    "sadness": [
        "sad", "depressed", "down", "crying", "😢", "😭",
        # Russian
        "грустно", "плачу", "тоска", "печально", "плохо", "тяжело", "хандра", "уныло",
    ],
    "surprise": [
        "wow", "no way", "really", "what", "omg",
        # Russian
        "ого", "ничего себе", "правда", "серьёзно", "не может быть", "ну и ну", "офигеть", "вот это да",
    ],
    "neutral": [],
}


def detect_emotions_28(text: str) -> Dict[str, Any]:
    """Detect emotions using 28-label GoEmotions model or heuristic fallback.

    Returns:
        {
            "emotions": [{"label": str, "score": float}, ...],  # top 5
            "primary_emotion": str,
            "primary_score": float,
            "secondary_emotion": str | None,
            "emotional_complexity": float,  # 0-1, higher = more mixed
            "valence": float,  # -1 (negative) to +1 (positive)
            "arousal": float,  # 0 (calm) to 1 (intense)
            "source": "go_emotions" | "heuristic"
        }
    """
    if _load_go_emotions() and _go_emotions_pipeline:
        try:
            # Truncate for model (max ~512 tokens)
            results = _go_emotions_pipeline(text[:512])
            if results and isinstance(results, list):
                if isinstance(results[0], list):
                    results = results[0]

                # Sort by score descending
                sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
                top5 = sorted_results[:5]

                primary = top5[0]
                secondary = top5[1] if len(top5) > 1 and top5[1]["score"] > 0.1 else None

                # Emotional complexity: high when multiple emotions are strong
                above_threshold = [r for r in sorted_results if r["score"] > 0.15]
                complexity = min(len(above_threshold) / 5.0, 1.0)

                # Valence & arousal from emotion labels
                valence = _compute_valence(sorted_results)
                arousal = _compute_arousal(sorted_results)

                return {
                    "emotions": [{"label": r["label"], "score": round(r["score"], 3)} for r in top5],
                    "primary_emotion": primary["label"],
                    "primary_score": round(primary["score"], 3),
                    "secondary_emotion": secondary["label"] if secondary else None,
                    "emotional_complexity": round(complexity, 2),
                    "valence": round(valence, 2),
                    "arousal": round(arousal, 2),
                    "source": "go_emotions",
                }
        except Exception as e:
            ai_logger.warning(f"GoEmotions inference failed: {e}")

    # Heuristic fallback
    return _heuristic_28_emotions(text)


def _heuristic_28_emotions(text: str) -> Dict[str, Any]:
    """Heuristic 28-emotion detection when model isn't available."""
    text_lower = text.lower()
    scores = {}
    for emotion, patterns in _HEURISTIC_EMOTION_PATTERNS.items():
        score = 0.0
        for p in patterns:
            if p in text_lower:
                score += 0.3
        scores[emotion] = min(score, 1.0)

    if not any(v > 0 for v in scores.values()):
        scores["neutral"] = 0.5

    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_emotions[:5]

    primary = top5[0]
    secondary = top5[1] if len(top5) > 1 and top5[1][1] > 0.1 else None
    complexity = len([s for s in sorted_emotions if s[1] > 0.15]) / 5.0

    return {
        "emotions": [{"label": e, "score": round(s, 3)} for e, s in top5],
        "primary_emotion": primary[0],
        "primary_score": round(primary[1], 3),
        "secondary_emotion": secondary[0] if secondary else None,
        "emotional_complexity": round(min(complexity, 1.0), 2),
        "valence": _compute_valence([{"label": e, "score": s} for e, s in sorted_emotions]),
        "arousal": _compute_arousal([{"label": e, "score": s} for e, s in sorted_emotions]),
        "source": "heuristic",
    }


_POSITIVE_EMOTIONS = {"admiration", "amusement", "approval", "caring", "excitement",
                       "gratitude", "joy", "love", "optimism", "pride", "relief"}
_NEGATIVE_EMOTIONS = {"anger", "annoyance", "disappointment", "disapproval", "disgust",
                       "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}
_HIGH_AROUSAL = {"anger", "annoyance", "excitement", "fear", "joy", "love",
                  "surprise", "amusement", "desire"}
_LOW_AROUSAL = {"relief", "sadness", "grief", "neutral", "realization", "caring"}


def _compute_valence(emotions: List[Dict]) -> float:
    """Compute emotional valence (-1 to +1) from emotion scores."""
    pos = sum(e["score"] for e in emotions if e["label"] in _POSITIVE_EMOTIONS)
    neg = sum(e["score"] for e in emotions if e["label"] in _NEGATIVE_EMOTIONS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _compute_arousal(emotions: List[Dict]) -> float:
    """Compute emotional arousal (0 to 1) from emotion scores."""
    high = sum(e["score"] for e in emotions if e["label"] in _HIGH_AROUSAL)
    low = sum(e["score"] for e in emotions if e["label"] in _LOW_AROUSAL)
    total = high + low
    if total == 0:
        return 0.3
    return min(high / max(total, 0.01), 1.0)


# ═══════════════════════════════════════════════════════════════
#  2. Hidden Reasoning Pre-Step (Chain-of-Thought)
# ═══════════════════════════════════════════════════════════════

def build_hidden_reasoning(
    incoming_text: str,
    emotions_28: Dict[str, Any],
    conversation_stage: str,
    topics: List[str],
    relationship_health: Dict[str, Any],
    memory_notes: List[str],
    risk_assessment: Optional[Dict] = None,
    personality_profile: Optional[Dict] = None,
    subtext: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a structured hidden reasoning chain before generating a reply.

    This is the "think before you speak" layer. It produces a structured
    analysis that gets injected into the system prompt, guiding the LLM
    to produce better responses.

    Chain: Emotion → Intent → Context → Strategy → Constraints
    """
    chain = {"steps": [], "confidence": 0.0}

    # Step 1: What are they feeling? (28-emotion analysis)
    emotion = emotions_28.get("primary_emotion", "neutral")
    secondary = emotions_28.get("secondary_emotion")
    complexity = emotions_28.get("emotional_complexity", 0)
    valence = emotions_28.get("valence", 0)
    arousal = emotions_28.get("arousal", 0.3)

    emotional_read = f"Primary: {emotion} ({emotions_28.get('primary_score', 0):.0%})"
    if secondary:
        emotional_read += f", Secondary: {secondary}"
    if complexity > 0.4:
        emotional_read += f" [Complex/mixed emotions detected]"

    chain["steps"].append({
        "step": "emotional_read",
        "analysis": emotional_read,
        "valence": valence,
        "arousal": arousal,
    })

    # Step 2: What do they actually want? (intent inference)
    intent = _infer_intent(incoming_text, emotion, conversation_stage, subtext)
    chain["steps"].append({
        "step": "intent_inference",
        "surface_intent": intent["surface"],
        "likely_real_intent": intent["real"],
        "confidence": intent["confidence"],
    })

    # Step 3: Relationship context assessment
    health_score = relationship_health.get("score", 50)
    health_grade = relationship_health.get("grade", "N/A")
    recent_conflict = relationship_health.get("signals", {}).get("recent_conflict", {}).get("penalty", 0)

    context = {
        "step": "relationship_context",
        "health": f"{health_grade} ({health_score}/100)",
        "stage": conversation_stage,
        "recent_conflict": recent_conflict > 0,
        "topics": topics[:3],
    }
    if memory_notes:
        context["relevant_memories"] = memory_notes[:3]
    chain["steps"].append(context)

    # Step 4: Risk assessment
    if risk_assessment and risk_assessment.get("risk_level") != "low":
        chain["steps"].append({
            "step": "risk_assessment",
            "risk_level": risk_assessment.get("risk_level", "low"),
            "top_signals": risk_assessment.get("top_signals", []),
            "recommended_action": risk_assessment.get("recommended_action", "proceed_normally"),
        })

    # Step 5: Response strategy decision
    strategy = _decide_response_strategy(
        emotion, intent, conversation_stage, valence, arousal,
        health_score, risk_assessment, personality_profile,
    )
    chain["steps"].append({
        "step": "response_strategy",
        **strategy,
    })

    # Overall confidence
    chain["confidence"] = min(
        intent["confidence"] * 0.4 +
        emotions_28.get("primary_score", 0.3) * 0.3 +
        (0.3 if health_score > 30 else 0.1),
        1.0,
    )
    chain["summary"] = strategy.get("one_liner", "")

    return chain


def _infer_intent(
    text: str, emotion: str, stage: str, subtext: Optional[Dict]
) -> Dict[str, Any]:
    """Infer both surface and real intent from message."""
    text_lower = text.lower().strip()
    surface = "statement"
    real = "communication"
    confidence = 0.5

    # Surface intent detection
    if "?" in text:
        surface = "question"
        confidence = 0.7
    elif any(w in text_lower for w in [
        "help", "advice", "should i", "what do i",
        "как думаешь", "что посоветуешь", "помоги", "подскажи",
    ]):
        surface = "seeking_advice"
        confidence = 0.7
    elif any(w in text_lower for w in [
        "look", "check", "see this", "sent",
        "посмотри", "глянь", "смотри что", "смотри",
    ]):
        surface = "sharing_content"
        confidence = 0.6
    elif any(w in text_lower for w in [
        "miss", "love",
        "скучаю", "люблю", "обожаю",
    ]):
        surface = "expressing_affection"
        confidence = 0.8
    elif any(w in text_lower for w in [
        "sorry", "apologize",
        "извини", "прости", "мне жаль",
    ]):
        surface = "apologizing"
        confidence = 0.8
    elif any(w in text_lower for w in [
        "ugh", "can't", "hate",
        "бесит", "достал", "задолбало", "надоело",
    ]):
        surface = "venting"
        confidence = 0.7

    # Real intent inference (what they actually want)
    if subtext and subtext.get("has_subtext"):
        real = subtext.get("likely_real_intent", real)
        confidence = max(confidence, subtext.get("confidence", 0.5))
    elif emotion in ("sadness", "fear", "grief", "nervousness"):
        real = "seeking_comfort"
    elif emotion in ("anger", "annoyance", "disappointment"):
        real = "wanting_acknowledgment"
    elif emotion == "love":
        real = "seeking_reciprocation"
    elif emotion == "curiosity":
        real = "seeking_engagement"
    elif emotion in ("excitement", "joy", "amusement"):
        real = "wanting_to_share_joy"
    elif surface == "question" and stage == "conflict":
        real = "testing_commitment"
    else:
        real = surface

    return {"surface": surface, "real": real, "confidence": round(confidence, 2)}


def _decide_response_strategy(
    emotion: str, intent: Dict, stage: str, valence: float, arousal: float,
    health_score: int, risk: Optional[Dict], personality: Optional[Dict],
) -> Dict[str, Any]:
    """Decide optimal response strategy based on all signals."""
    strategy = {
        "tone": "casual",
        "length": "match_theirs",
        "vulnerability_level": "low",
        "humor_appropriate": True,
        "should_ask_question": False,
        "emotional_response_needed": False,
        "one_liner": "",
    }

    real_intent = intent.get("real", "communication")

    # Seeking comfort → be supportive, NOT therapeutic
    if real_intent == "seeking_comfort":
        strategy["tone"] = "warm_supportive"
        strategy["vulnerability_level"] = "medium"
        strategy["humor_appropriate"] = False
        strategy["emotional_response_needed"] = True
        strategy["one_liner"] = "They need comfort. Be warm and real, not a therapist."

    # Wanting acknowledgment (upset) → validate without caving
    elif real_intent == "wanting_acknowledgment":
        strategy["tone"] = "sincere"
        strategy["vulnerability_level"] = "medium"
        strategy["humor_appropriate"] = False
        strategy["emotional_response_needed"] = True
        strategy["one_liner"] = "They want to be heard. React genuinely, then respond."

    # Testing commitment → be confident and clear
    elif real_intent == "testing_commitment":
        strategy["tone"] = "confident_sincere"
        strategy["length"] = "medium"
        strategy["vulnerability_level"] = "high"
        strategy["humor_appropriate"] = False
        strategy["one_liner"] = "They're testing you. Be specific and unwavering."

    # Sharing joy → match excitement
    elif real_intent == "wanting_to_share_joy":
        strategy["tone"] = "excited"
        strategy["humor_appropriate"] = True
        strategy["should_ask_question"] = True
        strategy["one_liner"] = "They're excited! Match their energy, ask for details."

    # Seeking reciprocation (love)
    elif real_intent == "seeking_reciprocation":
        strategy["tone"] = "romantic"
        strategy["vulnerability_level"] = "high"
        strategy["one_liner"] = "They expressed love. Reciprocate genuinely and specifically."

    # High risk situation
    if risk and risk.get("risk_level") in ("high", "critical"):
        strategy["tone"] = "careful_warm"
        strategy["humor_appropriate"] = False
        strategy["emotional_response_needed"] = True
        strategy["one_liner"] = "HIGH RISK: Be real and direct. Dont say anything stupid."

    # Low health relationship
    if health_score < 40:
        strategy["vulnerability_level"] = "high"
        strategy["should_ask_question"] = True
        if not strategy["one_liner"]:
            strategy["one_liner"] = "Relationship is rocky. Be real, put in effort, dont be fake."

    # Stage-specific overrides
    if stage == "conflict":
        strategy["humor_appropriate"] = False
        strategy["emotional_response_needed"] = True
    elif stage == "cooling_down":
        strategy["should_ask_question"] = True

    return strategy


def format_hidden_reasoning_for_prompt(chain: Dict[str, Any]) -> str:
    """Format the hidden reasoning chain into a prompt injection."""
    if not chain or not chain.get("steps"):
        return ""

    parts = ["## Hidden Reasoning (use this to calibrate your response):"]

    for step in chain["steps"]:
        step_name = step.get("step", "")

        if step_name == "emotional_read":
            parts.append(f"THEIR EMOTIONAL STATE: {step['analysis']}")
            if step.get("valence", 0) < -0.3:
                parts.append("⚠️ Negative emotional state — match their energy, dont be a therapist")

        elif step_name == "intent_inference":
            parts.append(f"WHAT THEY SAID: {step['surface_intent']}")
            if step["surface_intent"] != step["likely_real_intent"]:
                parts.append(f"WHAT THEY ACTUALLY WANT: {step['likely_real_intent']}")

        elif step_name == "relationship_context":
            parts.append(f"RELATIONSHIP: {step['health']} | Stage: {step['stage']}")
            if step.get("recent_conflict"):
                parts.append("Recent conflict — stand your ground, dont just apologize")

        elif step_name == "risk_assessment":
            parts.append(f"⚠️ RISK: {step['risk_level']} — {step.get('recommended_action', '')}")

        elif step_name == "response_strategy":
            parts.append(f"\nSTRATEGY: {step.get('one_liner', '')}")
            parts.append(f"Tone: {step.get('tone', 'casual')}")
            parts.append(f"Length: {step.get('length', 'match_theirs')}")
            if step.get("emotional_response_needed"):
                parts.append("IMPORTANT: This needs an emotional response, not a logical one")
            if not step.get("humor_appropriate", True):
                parts.append("Humor NOT appropriate right now")
            if step.get("should_ask_question"):
                parts.append("Ask them a question to keep engagement")
            vuln = step.get("vulnerability_level", "low")
            if vuln in ("medium", "high"):
                parts.append(f"Be genuine and vulnerable (level: {vuln})")

    return "\n".join(f"- {p}" if not p.startswith("#") and not p.startswith("\n") else p for p in parts)


# ═══════════════════════════════════════════════════════════════
#  3. Subtext Detection (What They Really Mean)
# ═══════════════════════════════════════════════════════════════

_SUBTEXT_PATTERNS = [
    # (pattern, subtext_meaning, confidence)
    (r"^fine\.?$", "not_fine_upset", 0.8),
    (r"^ok\.?$", "disengaging_or_upset", 0.6),
    (r"^k\.?$", "definitely_upset", 0.85),
    (r"^whatever\.?$", "frustrated_giving_up", 0.8),
    (r"^nothing\.?$", "something_is_wrong", 0.7),
    (r"^i don'?t care\.?$", "actually_cares_a_lot", 0.7),
    (r"^it'?s fine\.?$", "it_is_not_fine", 0.75),
    (r"^do what you want\.?$", "wants_opposite", 0.8),
    (r"^sure\.?$", "reluctant_agreement", 0.5),
    (r"^nice\.?$", "sarcastic_or_disengaged", 0.4),
    (r"^i guess\.?$", "not_happy_about_it", 0.6),
    (r"\bi'?m fine\b", "probably_not_fine", 0.5),
    (r"^you do you\.?$", "disapproves_but_giving_up", 0.7),
    (r"^have fun\.?$", "jealous_or_passive_aggressive", 0.5),
    (r"^it doesn'?t matter\.?$", "it_matters_a_lot", 0.7),
    # Russian
    (r"^ладно\.?$", "reluctant_upset", 0.6),
    (r"^нормально\.?$", "not_normal", 0.5),
    (r"^как хочешь\.?$", "wants_opposite", 0.75),
    (r"^мне всё равно\.?$", "cares_deeply", 0.7),
    (r"^ну и ладно\.?$", "upset_giving_up", 0.7),
]

# Contextual subtext: when surface sentiment contradicts context
_CONTRADICTION_SUBTEXTS = {
    "positive_words_period_ending": {
        "meaning": "surface_positive_actually_cold",
        "explanation": "Positive words with period = cold/dismissive",
        "confidence": 0.6,
    },
    "short_after_long": {
        "meaning": "withdrawal_or_upset",
        "explanation": "Suddenly short responses after engaging = pulling away",
        "confidence": 0.65,
    },
    "emoji_disappearance": {
        "meaning": "emotional_withdrawal",
        "explanation": "Person who normally uses emojis stopped = emotional shift",
        "confidence": 0.6,
    },
}


def detect_subtext(
    text: str,
    conversation_history: List[Dict[str, str]],
    emotions_28: Optional[Dict] = None,
    memory: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Detect subtext and implied meaning in a message.

    Returns:
        {
            "has_subtext": bool,
            "subtext_type": str,
            "likely_real_intent": str,
            "explanation": str,
            "confidence": float,
            "signals": list,
        }
    """
    text_lower = text.lower().strip()
    signals = []
    best_match = None
    best_confidence = 0.0

    # Pattern-based subtext detection
    for pattern, meaning, conf in _SUBTEXT_PATTERNS:
        if re.match(pattern, text_lower):
            if conf > best_confidence:
                best_match = meaning
                best_confidence = conf
                signals.append(f"Pattern match: {meaning}")

    # Contextual contradictions
    # 1. Period ending on short positive message
    if text_lower.endswith(".") and len(text_lower) < 30 and "..." not in text_lower:
        positive_words = [
            "good", "great", "nice", "fine", "ok", "sure", "cool",
            # Russian
            "хорошо", "прекрасно", "отлично", "ок", "хаха", "норм", "нормально", "ладно",
        ]
        if any(w in text_lower for w in positive_words):
            signals.append("Positive words with cold period ending")
            if best_confidence < 0.6:
                best_match = "surface_positive_actually_cold"
                best_confidence = 0.6

    # 2. Short response after they were being long
    if conversation_history:
        their_recent = [m for m in conversation_history[-5:] if m.get("sender") == "Them"]
        if len(their_recent) >= 2:
            prev_avg = sum(len(m.get("text", "")) for m in their_recent[:-1]) / max(len(their_recent) - 1, 1)
            if prev_avg > 40 and len(text_lower) < 15:
                signals.append("Sudden short response after longer messages")
                if best_confidence < 0.55:
                    best_match = "withdrawal_or_upset"
                    best_confidence = 0.55

    # 3. Emotion contradiction: surface positive text + negative emotion
    if emotions_28:
        primary = emotions_28.get("primary_emotion", "neutral")
        if primary in ("anger", "sadness", "disappointment", "annoyance"):
            positive_surface = any(w in text_lower for w in ["good", "great", "fine", "ok", "haha", "lol"])
            if positive_surface:
                signals.append(f"Surface positive but emotion={primary}")
                if best_confidence < 0.65:
                    best_match = "masking_negative_emotion"
                    best_confidence = 0.65

    # 4. Emoji normally used but missing
    if memory and memory.get("conversation_patterns", {}).get("they_use_emojis"):
        has_emoji = bool(re.search(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', text))
        if not has_emoji and len(text) > 10:
            signals.append("Usually uses emojis but none present")
            if best_confidence < 0.4:
                best_match = "emotional_withdrawal"
                best_confidence = 0.4

    if not best_match:
        return {
            "has_subtext": False,
            "subtext_type": None,
            "likely_real_intent": "sincere",
            "explanation": "Message appears sincere/at face value",
            "confidence": 0.0,
            "signals": [],
        }

    # Map subtext to real intent
    intent_map = {
        "not_fine_upset": "wanting_acknowledgment",
        "disengaging_or_upset": "seeking_attention",
        "definitely_upset": "wanting_acknowledgment",
        "frustrated_giving_up": "wanting_you_to_try_harder",
        "something_is_wrong": "waiting_to_be_asked",
        "actually_cares_a_lot": "wanting_you_to_care",
        "it_is_not_fine": "wanting_acknowledgment",
        "wants_opposite": "testing_whether_you_care",
        "reluctant_agreement": "not_happy_about_it",
        "sarcastic_or_disengaged": "losing_interest",
        "not_happy_about_it": "wanting_better_option",
        "surface_positive_actually_cold": "emotional_distance",
        "withdrawal_or_upset": "pulling_away",
        "emotional_withdrawal": "processing_emotions",
        "masking_negative_emotion": "hiding_true_feelings",
        "probably_not_fine": "wanting_to_be_asked_again",
        "disapproves_but_giving_up": "wanting_you_to_reconsider",
        "jealous_or_passive_aggressive": "feeling_excluded",
        "it_matters_a_lot": "wanting_acknowledgment",
        "reluctant_upset": "wanting_acknowledgment",
        "not_normal": "something_is_wrong",
        "cares_deeply": "wanting_you_to_care",
        "upset_giving_up": "wanting_you_to_try_harder",
    }

    return {
        "has_subtext": True,
        "subtext_type": best_match,
        "likely_real_intent": intent_map.get(best_match, "wanting_acknowledgment"),
        "explanation": f"They said '{text_lower}' but likely mean: {best_match.replace('_', ' ')}",
        "confidence": round(best_confidence, 2),
        "signals": signals,
    }


# ═══════════════════════════════════════════════════════════════
#  4. Conversation Risk Detector
# ═══════════════════════════════════════════════════════════════

# Per-chat behavioral baselines
_chat_baselines: Dict[int, Dict[str, Any]] = {}
_risk_history: Dict[int, List[float]] = {}


def update_behavioral_baseline(chat_id: int, message: Dict[str, Any]):
    """Update the per-user behavioral baseline with new message data."""
    baseline = _chat_baselines.get(chat_id, {
        "avg_length": 30.0,
        "avg_emoji_count": 1.0,
        "avg_response_time": 300.0,
        "message_count": 0,
        "uses_emojis": True,
        "typical_punctuation": False,
    })

    text = message.get("text", "")
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', text))
    n = baseline["message_count"]

    # Running averages
    if n > 0:
        alpha = min(0.2, 1.0 / (n + 1))  # Smoothing factor
        baseline["avg_length"] = baseline["avg_length"] * (1 - alpha) + len(text) * alpha
        baseline["avg_emoji_count"] = baseline["avg_emoji_count"] * (1 - alpha) + emoji_count * alpha
    else:
        baseline["avg_length"] = float(len(text))
        baseline["avg_emoji_count"] = float(emoji_count)

    baseline["message_count"] = n + 1
    baseline["uses_emojis"] = baseline["avg_emoji_count"] > 0.3
    _chat_baselines[chat_id] = baseline


def detect_conversation_risk(
    chat_id: int,
    incoming_text: str,
    emotions_28: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    subtext: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Detect risk of conversation derailment / escalation / withdrawal.

    Returns:
        {
            "risk_score": float (0-1),
            "risk_level": "low" | "medium" | "high" | "critical",
            "top_signals": list of (signal_name, score) tuples,
            "recommended_action": str,
        }
    """
    baseline = _chat_baselines.get(chat_id, {
        "avg_length": 30.0, "avg_emoji_count": 1.0, "uses_emojis": True, "message_count": 0,
    })
    signals = {}

    # 1. Emotion risk: negative emotions with high arousal
    valence = emotions_28.get("valence", 0)
    arousal = emotions_28.get("arousal", 0.3)
    primary = emotions_28.get("primary_emotion", "neutral")

    if primary in ("anger", "annoyance", "disgust"):
        signals["negative_emotion"] = 0.7
    elif primary in ("sadness", "disappointment", "grief"):
        signals["negative_emotion"] = 0.5
    elif primary in ("fear", "nervousness"):
        signals["negative_emotion"] = 0.4

    if valence < -0.5:
        signals["strong_negative_valence"] = 0.6

    # 2. Length anomaly
    if baseline["message_count"] > 5:
        length_ratio = len(incoming_text) / max(baseline["avg_length"], 1)
        if length_ratio < 0.3:
            signals["message_length_drop"] = 0.5
        elif length_ratio > 3.0:
            signals["message_length_spike"] = 0.3  # Could be venting

    # 3. Emoji disappearance
    if baseline.get("uses_emojis") and baseline["message_count"] > 5:
        emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', incoming_text))
        if emoji_count == 0 and len(incoming_text) > 10:
            signals["emoji_disappearance"] = 0.5

    # 4. Formality increase (periods in short casual messages)
    if incoming_text.endswith(".") and len(incoming_text) < 40 and "..." not in incoming_text:
        signals["formality_increase"] = 0.4

    # 5. Subtext detected
    if subtext and subtext.get("has_subtext"):
        signals["subtext_detected"] = subtext.get("confidence", 0.5) * 0.8

    # 6. Conversation energy decline
    if len(conversation_history) >= 5:
        their_recent = [m for m in conversation_history[-5:] if m.get("sender") == "Them"]
        if len(their_recent) >= 3:
            lengths = [len(m.get("text", "")) for m in their_recent]
            if len(lengths) >= 3 and lengths[-1] < lengths[0] * 0.4:
                signals["energy_declining"] = 0.4

    # Compute composite score
    if not signals:
        risk_score = 0.0
    else:
        risk_score = min(sum(signals.values()) / max(len(signals), 1) * 1.5, 1.0)

    # Temporal smoothing with exponential moving average
    if chat_id not in _risk_history:
        _risk_history[chat_id] = []
    _risk_history[chat_id].append(risk_score)
    if len(_risk_history[chat_id]) > 20:
        _risk_history[chat_id] = _risk_history[chat_id][-20:]

    # EMA smoothing
    alpha = 0.4
    smoothed = risk_score
    for prev in reversed(_risk_history[chat_id][:-1]):
        smoothed = alpha * smoothed + (1 - alpha) * prev

    # Categorize
    if smoothed >= 0.7:
        level = "critical"
    elif smoothed >= 0.5:
        level = "high"
    elif smoothed >= 0.3:
        level = "medium"
    else:
        level = "low"

    # Recommendation
    recommendations = {
        "critical": "Be very careful. Validate their feelings. Don't be defensive. Ask what's wrong gently.",
        "high": "Something is off. Be warm and attentive. Address any tension directly but gently.",
        "medium": "Minor risk signals. Be extra engaged and genuine.",
        "low": "proceed_normally",
    }

    top_signals = sorted(signals.items(), key=lambda x: -x[1])[:3]

    # Update baseline
    update_behavioral_baseline(chat_id, {"text": incoming_text})

    return {
        "risk_score": round(smoothed, 2),
        "risk_level": level,
        "top_signals": top_signals,
        "recommended_action": recommendations[level],
    }


# ═══════════════════════════════════════════════════════════════
#  5. Uncanny Valley Post-Processing
# ═══════════════════════════════════════════════════════════════

# Research stats: AI uses periods 85-95%, humans 15-30%
# AI proper capitalization 95-100%, humans 30-60%
# AI em-dashes ~5-10%, humans ~0.1%


def humanize_text(text: str) -> str:
    """Post-process generated text to eliminate AI tells and sound human.

    Applied to each message segment individually. Aggressive — designed to
    catch anything the prompt-level instructions missed.
    """
    if not text or len(text) < 3:
        return text

    result = text

    # 1. Remove em-dashes and semicolons (massive AI tells)
    result = result.replace("—", " ")
    result = result.replace("–", " ")
    result = result.replace(";", ",")

    # 2. Remove formal conjunctions AI overuses
    formal_starts = [
        "However, ", "Furthermore, ", "Moreover, ", "Nevertheless, ",
        "Additionally, ", "Consequently, ", "Therefore, ", "Indeed, ",
        "Certainly, ", "Absolutely, ", "Definitely, ", "Of course, ",
        "In fact, ", "To be honest, ", "Honestly speaking, ",
    ]
    for formal in formal_starts:
        if result.startswith(formal):
            result = result[len(formal):]
            result = result[0].lower() + result[1:] if result else result

    # 3. Strip therapist/AI opener phrases (flexible — catches "I completely understand", etc.)
    therapist_openers = [
        r"^I\s+(?:\w+\s+)?understand\s+how\s+you\s+feel[^.!?]*[.!?]?\s*",
        r"^That\s+must\s+be\s+(?:really\s+)?[^.!?]*[.!?]?\s*",
        r"^I\s+(?:\w+\s+)?appreciate\s+you\s+sharing[^.!?]*[.!?]?\s*",
        r"^That\s+sounds?\s+(?:really\s+|so\s+)?[^.!?]*[.!?]?\s*",
        r"^I\s+can\s+(?:only\s+)?imagine\s+how[^.!?]*[.!?]?\s*",
        r"^That'?s\s+(?:completely\s+|totally\s+)?understandable[^.!?]*[.!?]?\s*",
        r"^Your\s+feelings\s+are\s+(?:completely\s+|totally\s+)?valid[^.!?]*[.!?]?\s*",
        r"^I\s+hear\s+you[^.!?]*[.!?]?\s*",
        r"^I\s+want\s+you\s+to\s+know[^.!?]*[.!?]?\s*",
        r"^I'?m\s+(?:always\s+)?here\s+for\s+you[^.!?]*[.!?]?\s*",
        r"^That'?s\s+(?:so\s+)?sweet[^.!?]*[.!?]?\s*",
        r"^I\s+(?:\w+\s+)?appreciate[^.!?]*[.!?]?\s*",
        r"^Don'?t\s+hesitate\s+to[^.!?]*[.!?]?\s*",
        r"^Feel\s+free\s+to[^.!?]*[.!?]?\s*",
    ]
    for pattern in therapist_openers:
        candidate = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
        if len(candidate) > 10 and candidate != result:
            result = candidate[0].lower() + candidate[1:] if candidate else result
            break

    # 4. Replace AI-formal words with casual equivalents
    # Only replace words that are CLEAR AI tells — don't mangle natural language
    # Skip this step entirely for Russian text (Cyrillic)
    _has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in result)
    if not _has_cyrillic:
        casual_replacements = [
            (r"\bI apologize\b", "my bad"),
            (r"\bcertainly\b", "for sure"),
            (r"\bperhaps\b", "maybe"),
            (r"\bhowever\b", "but"),
            (r"\bfurthermore\b", "also"),
            (r"\badditionally\b", "also"),
            (r"\bnevertheless\b", "still"),
            (r"\bregarding\b", "about"),
        ]
        for pattern, replacement in casual_replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 5. Strip excessive emojis — keep max 1 per segment
    # Match individual emoji chars (not groups) to count accurately
    _single_emoji = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        "\U00002764\U0001F90D-\U0001F90F]", re.UNICODE
    )
    emoji_chars = _single_emoji.findall(result)
    if len(emoji_chars) > 1:
        # Keep only the last emoji char, remove all others
        kept = False
        new_result = []
        for ch in reversed(result):
            if _single_emoji.match(ch):
                if not kept:
                    new_result.append(ch)
                    kept = True
                # else: skip this emoji
            else:
                new_result.append(ch)
        result = "".join(reversed(new_result))

    # 6. Remove trailing periods (always for casual text, preserve "...")
    if result.endswith(".") and not result.endswith("..."):
        result = result[:-1]

    # 7. Lowercase first letter (~50% of the time, ONLY for English)
    if result and result[0].isupper() and not _has_cyrillic and random.random() < 0.50:
        # Don't lowercase standalone "I" or "I'm", "I'll" etc.
        if not (result[0] == "I" and (len(result) < 2 or not result[1].isalpha())):
            # Don't lowercase if entire word is CAPS (like LMAOOO, YOOO)
            first_word = result.split()[0] if result.split() else ""
            if not (len(first_word) > 1 and first_word.isupper()):
                result = result[0].lower() + result[1:]

    # 8. Remove double spaces
    result = re.sub(r"  +", " ", result)

    # 9. Occasionally drop "I" → "i" (25%, ONLY for English)
    if not _has_cyrillic and random.random() < 0.25:
        result = re.sub(r"\bI\b(?!')", "i", result)

    # 10. Remove quotation marks around words
    result = re.sub(r'"(\w+)"', r'\1', result)

    # 11. Strip multiple exclamation/question marks to max 2
    result = re.sub(r"!{3,}", "!!", result)
    result = re.sub(r"\?{3,}", "??", result)

    # 12. Remove "Haha, " or "Haha. " at start (AI pattern)
    result = re.sub(r"^[Hh]aha[,.]?\s+", "", result)

    return result.strip()


# ═══════════════════════════════════════════════════════════════
#  6. Emoji Pattern Analysis + Temporal Signals
# ═══════════════════════════════════════════════════════════════

_emoji_history: Dict[int, List[Dict]] = {}  # chat_id -> [{timestamp, emojis, count}]


def analyze_emoji_patterns(
    chat_id: int, text: str, conversation_history: List[Dict]
) -> Dict[str, Any]:
    """Track and analyze emoji usage patterns over time.

    Returns:
        {
            "current_emojis": list,
            "emoji_density": float,
            "trend": "increasing" | "stable" | "decreasing" | "disappeared",
            "anomaly": bool,
            "emotional_signal": str,
        }
    """
    # Extract emojis from current message
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\uFE00-\uFE0F]|[❤️💕💖😍🥰😘💗💓🤗😊💋♥💞💝🥺✨🌹🔥💯😂🤣😄😁🙌👏😢😭😡😤💔😞😔😒🙄🤮😰😩😫🤬👍👎🤔😏😉😈💦😮]')
    current_emojis = emoji_pattern.findall(text)
    emoji_count = len(current_emojis)
    word_count = max(len(text.split()), 1)
    density = emoji_count / word_count

    # Update history
    if chat_id not in _emoji_history:
        _emoji_history[chat_id] = []
    _emoji_history[chat_id].append({
        "timestamp": time.time(),
        "emojis": current_emojis,
        "count": emoji_count,
        "density": density,
    })
    # Keep last 50
    _emoji_history[chat_id] = _emoji_history[chat_id][-50:]

    # Analyze trend
    history = _emoji_history[chat_id]
    trend = "stable"
    anomaly = False

    if len(history) >= 5:
        recent_avg = sum(h["count"] for h in history[-3:]) / 3
        older_avg = sum(h["count"] for h in history[-8:-3]) / max(len(history[-8:-3]), 1)

        if older_avg > 1 and recent_avg < 0.3:
            trend = "disappeared"
            anomaly = True
        elif recent_avg > older_avg * 1.5:
            trend = "increasing"
        elif recent_avg < older_avg * 0.5:
            trend = "decreasing"
            anomaly = older_avg > 1  # Only anomaly if they normally use emojis

    # Emotional signal from emoji types
    love_emojis = {"❤️", "💕", "💖", "😍", "🥰", "😘", "💗", "💓", "💋", "♥", "💞", "💝", "❤"}
    sad_emojis = {"😢", "😭", "😞", "😔", "💔"}
    fun_emojis = {"😂", "🤣", "😄", "😁", "💀", "🔥"}
    angry_emojis = {"😡", "😤", "🤬", "😒", "🙄"}

    signal = "neutral"
    if any(e in love_emojis for e in current_emojis):
        signal = "affectionate"
    elif any(e in sad_emojis for e in current_emojis):
        signal = "sad"
    elif any(e in fun_emojis for e in current_emojis):
        signal = "playful"
    elif any(e in angry_emojis for e in current_emojis):
        signal = "frustrated"

    return {
        "current_emojis": current_emojis[:5],
        "emoji_density": round(density, 3),
        "trend": trend,
        "anomaly": anomaly,
        "emotional_signal": signal,
    }


# ═══════════════════════════════════════════════════════════════
#  7. Personality Profiling (Big Five + Attachment Style)
# ═══════════════════════════════════════════════════════════════

_personality_cache: Dict[int, Dict[str, Any]] = {}
PERSONALITY_DATA_DIR = Path(__file__).parent / ".personality_profiles"
PERSONALITY_DATA_DIR.mkdir(exist_ok=True)


def profile_personality(
    chat_id: int, conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Estimate Big Five personality traits and attachment style from text.

    Uses accumulated messages for better accuracy. Requires 20+ messages.

    Returns:
        {
            "big_five": {
                "openness": float (0-1),
                "conscientiousness": float (0-1),
                "extraversion": float (0-1),
                "agreeableness": float (0-1),
                "neuroticism": float (0-1),
            },
            "attachment_style": "secure" | "anxious" | "avoidant" | "fearful_avoidant",
            "love_language": str,
            "confidence": float,
        }
    """
    their_msgs = [m.get("text", "") for m in conversation_history if m.get("sender") == "Them"]

    if len(their_msgs) < 10:
        return {
            "big_five": None,
            "attachment_style": "unknown",
            "love_language": "unknown",
            "confidence": 0.0,
            "insufficient_data": True,
        }

    all_text = " ".join(their_msgs)
    words = all_text.lower().split()
    total_words = len(words)
    word_set = set(words)
    unique_ratio = len(word_set) / max(total_words, 1)

    # ── Big Five estimation ──

    # Openness: vocabulary diversity, question-asking, topic variety
    openness = min(unique_ratio * 2, 1.0)
    question_ratio = sum(1 for m in their_msgs if "?" in m) / max(len(their_msgs), 1)
    openness = (openness * 0.5 + min(question_ratio * 3, 1.0) * 0.5)

    # Conscientiousness: grammar correctness, message organization
    proper_caps = sum(1 for m in their_msgs if m and m[0].isupper()) / max(len(their_msgs), 1)
    period_usage = sum(1 for m in their_msgs if m.endswith(".")) / max(len(their_msgs), 1)
    conscientiousness = (proper_caps * 0.5 + period_usage * 0.5)

    # Extraversion: message length, emoji frequency, exclamation marks
    avg_length = sum(len(m) for m in their_msgs) / max(len(their_msgs), 1)
    emoji_freq = sum(len(re.findall(r'[\U00010000-\U0010ffff]', m)) for m in their_msgs) / max(len(their_msgs), 1)
    excl_freq = sum(m.count("!") for m in their_msgs) / max(len(their_msgs), 1)
    extraversion = min((avg_length / 100 * 0.3 + min(emoji_freq, 3) / 3 * 0.4 + min(excl_freq, 2) / 2 * 0.3), 1.0)

    # Agreeableness: positive words, agreement language
    agree_words = {"yes", "sure", "ok", "agree", "right", "true", "exactly", "of course",
                    "да", "конечно", "согласен", "точно", "верно"}
    agree_ratio = len(word_set & agree_words) / max(len(agree_words), 1) * 3
    agreeableness = min(agree_ratio, 1.0)

    # Neuroticism: negative words, anxiety markers, response time variability
    neuro_words = {"worried", "anxious", "scared", "nervous", "stressed", "panic",
                    "afraid", "can't", "волнуюсь", "боюсь", "нервничаю", "стресс"}
    neuro_count = sum(1 for w in words if w in neuro_words)
    neuroticism = min(neuro_count / max(total_words, 1) * 50, 1.0)

    big_five = {
        "openness": round(openness, 2),
        "conscientiousness": round(conscientiousness, 2),
        "extraversion": round(extraversion, 2),
        "agreeableness": round(agreeableness, 2),
        "neuroticism": round(neuroticism, 2),
    }

    # ── Attachment Style ──
    # Based on messaging patterns
    double_text_count = 0
    for i in range(1, len(conversation_history)):
        if (conversation_history[i].get("sender") == "Them" and
            conversation_history[i-1].get("sender") == "Them"):
            double_text_count += 1
    double_text_ratio = double_text_count / max(len(their_msgs), 1)

    reassurance_words = {"miss", "love", "need you", "where are you", "are you ok",
                          "скучаю", "ты где", "все хорошо"}
    reassurance_count = sum(1 for m in their_msgs if any(r in m.lower() for r in reassurance_words))
    reassurance_ratio = reassurance_count / max(len(their_msgs), 1)

    avg_msg_len = sum(len(m) for m in their_msgs) / max(len(their_msgs), 1)

    if double_text_ratio > 0.3 and reassurance_ratio > 0.15:
        attachment = "anxious"
    elif avg_msg_len < 15 and reassurance_ratio < 0.05:
        attachment = "avoidant"
    elif neuroticism > 0.4 and double_text_ratio > 0.2:
        attachment = "fearful_avoidant"
    else:
        attachment = "secure"

    # ── Love Language ──
    touch_words = sum(1 for m in their_msgs if any(w in m.lower() for w in [
        "hug", "kiss", "cuddle", "hold", "touch",
        # Russian
        "обнимаю", "целую", "хочу обнять", "прикоснуться", "обнять", "целовать",
        "прижаться", "обнимашки",
    ]))
    words_affirmation = sum(1 for m in their_msgs if any(w in m.lower() for w in [
        "love you", "proud of you", "you're amazing",
        # Russian
        "люблю тебя", "ты лучший", "ты лучшая", "горжусь тобой",
        "люблю", "молодец", "ты замечательный", "ты замечательная", "ты особенный", "ты особенная",
    ]))
    quality_time = sum(1 for m in their_msgs if any(w in m.lower() for w in [
        "hang out", "lets do", "come over", "together",
        # Russian
        "хочу быть с тобой", "давай вместе", "проведём время", "проведем время",
        "давай", "вместе", "побудем вдвоём", "побудем вдвоем", "давай встретимся",
    ]))
    acts_service = sum(1 for m in their_msgs if any(w in m.lower() for w in [
        "help", "can i", "let me", "for you",
        # Russian
        "я сделал для тебя", "я сделала для тебя", "я помогу", "позаботился", "позаботилась",
        "помочь", "давай я", "сделаю для тебя", "помог", "помогла",
    ]))
    gift_words = sum(1 for m in their_msgs if any(w in m.lower() for w in [
        "gift", "surprise", "bought", "got you",
        # Russian
        "я видел это и подумал о тебе", "я видела это и подумала о тебе",
        "подарок", "сюрприз", "купил", "купила", "нашёл для тебя", "нашла для тебя",
    ]))

    love_lang_scores = {
        "physical_touch": touch_words,
        "words_of_affirmation": words_affirmation,
        "quality_time": quality_time,
        "acts_of_service": acts_service,
        "receiving_gifts": gift_words,
    }
    love_language = max(love_lang_scores, key=love_lang_scores.get) if any(love_lang_scores.values()) else "unknown"

    confidence = min(len(their_msgs) / 50, 1.0)

    result = {
        "big_five": big_five,
        "attachment_style": attachment,
        "love_language": love_language,
        "confidence": round(confidence, 2),
        "insufficient_data": False,
    }

    _personality_cache[chat_id] = result
    return result


def format_personality_for_prompt(profile: Dict[str, Any]) -> str:
    """Format personality profile for prompt injection."""
    if profile.get("insufficient_data") or not profile.get("big_five"):
        return ""

    parts = []
    bf = profile["big_five"]
    attach = profile.get("attachment_style", "unknown")
    love = profile.get("love_language", "unknown")

    # Only mention notable traits
    high_traits = {k: v for k, v in bf.items() if v > 0.6}
    low_traits = {k: v for k, v in bf.items() if v < 0.3}

    if high_traits:
        parts.append(f"Their strong traits: {', '.join(high_traits.keys())}")
    if low_traits:
        parts.append(f"Their low traits: {', '.join(low_traits.keys())}")

    if attach == "anxious":
        parts.append("Attachment: anxious — needs extra reassurance, don't leave them hanging")
    elif attach == "avoidant":
        parts.append("Attachment: avoidant — don't be clingy, give them space")
    elif attach == "fearful_avoidant":
        parts.append("Attachment: fearful-avoidant — be consistent and patient")

    if love != "unknown":
        love_tips = {
            "words_of_affirmation": "They value verbal affection — use compliments and 'I love you'",
            "quality_time": "They value quality time — suggest activities and be present",
            "physical_touch": "They value physical closeness — reference touch, hugs, closeness",
            "acts_of_service": "They value acts of service — offer to help with things",
            "receiving_gifts": "They appreciate gifts and surprises — reference thoughtful gestures",
        }
        parts.append(love_tips.get(love, ""))

    return "\n".join(f"- {p}" for p in parts if p)


# ═══════════════════════════════════════════════════════════════
#  8. Response Quality Scorer (Multi-Dimensional)
# ═══════════════════════════════════════════════════════════════

def score_response_quality(
    response: str,
    incoming_text: str,
    conversation_history: List[Dict[str, str]],
    emotions_28: Dict[str, Any],
    hidden_reasoning: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a candidate response across multiple dimensions.

    Returns:
        {
            "overall_score": float (0-100),
            "dimensions": dict of dimension -> score,
            "warnings": list of str,
            "pass": bool,
        }
    """
    dimensions = {}
    warnings = []

    # 1. Length appropriateness (match their energy)
    their_len = len(incoming_text)
    our_len = len(response)
    ratio = our_len / max(their_len, 1)

    if ratio > 3.0:
        dimensions["length_match"] = 30
        warnings.append("Response much longer than their message — will seem AI-like")
    elif ratio > 2.0:
        dimensions["length_match"] = 60
    elif 0.3 <= ratio <= 2.0:
        dimensions["length_match"] = 95
    elif ratio < 0.3 and their_len > 50:
        dimensions["length_match"] = 50
        warnings.append("Response much shorter — may seem dismissive")
    else:
        dimensions["length_match"] = 80

    # 2. Emotional appropriateness
    strategy = {}
    for step in hidden_reasoning.get("steps", []):
        if step.get("step") == "response_strategy":
            strategy = step
            break

    if strategy.get("emotional_response_needed"):
        # Check if response has emotional content
        emotional_words = {"sorry", "feel", "understand", "damn", "wow", "omg", "ugh",
                           "love", "miss", "care", "worry", "блин", "жаль", "понимаю"}
        has_emotion = any(w in response.lower() for w in emotional_words)
        dimensions["emotional_match"] = 90 if has_emotion else 40
        if not has_emotion:
            warnings.append("Emotional response expected but reply seems unemotional")
    else:
        dimensions["emotional_match"] = 80

    # 3. Naturalness (uncanny valley checks)
    naturalness = 100
    # Check for AI phrases
    ai_phrases = ["I understand", "That sounds", "I appreciate", "I'm here for you",
                   "That must be", "I can imagine", "I want you to know", "That's valid",
                   "I hear you"]
    for phrase in ai_phrases:
        if phrase.lower() in response.lower():
            naturalness -= 20
            warnings.append(f"AI phrase detected: '{phrase}'")
    # Em-dashes
    if "—" in response or "–" in response:
        naturalness -= 15
        warnings.append("Em-dash detected (AI tell)")
    # Perfect grammar on short messages
    if len(response) < 50 and response[0].isupper() and response.endswith("."):
        naturalness -= 10
    dimensions["naturalness"] = max(naturalness, 0)

    # 4. Engagement quality (does it invite continued conversation?)
    has_question = "?" in response
    has_personal = any(w in response.lower() for w in ["you", "your", "u", "ur", "ты", "тебе"])
    engagement = 60
    if has_question:
        engagement += 20
    if has_personal:
        engagement += 15
    dimensions["engagement"] = min(engagement, 100)

    # 5. Persona consistency (not checking against persona — just basic checks)
    dimensions["persona_consistency"] = 85  # Default high

    # Overall weighted score
    weights = {
        "length_match": 0.20,
        "emotional_match": 0.25,
        "naturalness": 0.30,
        "engagement": 0.15,
        "persona_consistency": 0.10,
    }
    overall = sum(dimensions[k] * weights[k] for k in weights if k in dimensions)

    return {
        "overall_score": round(overall),
        "dimensions": dimensions,
        "warnings": warnings[:3],
        "pass": overall >= 60,
    }


# ═══════════════════════════════════════════════════════════════
#  9. Best-of-N Response Selection
# ═══════════════════════════════════════════════════════════════

async def generate_best_of_n(
    generate_fn,  # async callable that generates a reply
    n: int,
    incoming_text: str,
    conversation_history: List[Dict[str, str]],
    emotions_28: Dict[str, Any],
    hidden_reasoning: Dict[str, Any],
    chat_id: int,
) -> Tuple[str, Dict[str, Any]]:
    """Generate N candidate responses and return the best one.

    Args:
        generate_fn: async function that returns a reply string
        n: number of candidates (2-5)
        ...scoring parameters...

    Returns:
        (best_reply_text, scoring_details)
    """
    import asyncio

    candidates = []
    # Generate N candidates concurrently
    tasks = [generate_fn() for _ in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, str) and result:
            score = score_response_quality(
                result, incoming_text, conversation_history,
                emotions_28, hidden_reasoning,
            )
            # Blend with reward model prediction
            reward_pred = score_candidate_with_reward_model(chat_id, result)
            blended = score["overall_score"] * 0.7 + reward_pred * 100 * 0.3
            score["reward_model_score"] = reward_pred
            score["blended_score"] = round(blended)
            candidates.append({
                "text": result,
                "score": score,
            })

    if not candidates:
        return "", {"error": "no_candidates_generated"}

    # Sort by blended score (quality + reward model), pick the best
    candidates.sort(key=lambda c: c["score"].get("blended_score", c["score"]["overall_score"]), reverse=True)
    best = candidates[0]

    ai_logger.info(
        f"Best-of-{n}: selected score={best['score']['overall_score']}, "
        f"candidates={[c['score']['overall_score'] for c in candidates]}"
    )

    return best["text"], {
        "method": f"best_of_{n}",
        "selected_score": best["score"]["overall_score"],
        "all_scores": [c["score"]["overall_score"] for c in candidates],
        "warnings": best["score"].get("warnings", []),
    }


# ═══════════════════════════════════════════════════════════════
#  10. DSPy-Style Prompt Self-Optimization
# ═══════════════════════════════════════════════════════════════

_PROMPT_PERF_DIR = Path(__file__).parent / ".prompt_performance"
_PROMPT_PERF_DIR.mkdir(exist_ok=True)


def _load_prompt_performance(chat_id: int) -> Dict[str, Any]:
    path = _PROMPT_PERF_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "total_replies": 0,
        "engagement_scores": [],  # did they reply? how fast? how long?
        "successful_tones": Counter(),
        "successful_strategies": Counter(),
        "failed_patterns": [],
        "prompt_tweaks": [],  # learned adjustments
    }


def _save_prompt_performance(chat_id: int, data: Dict):
    path = _PROMPT_PERF_DIR / f"{chat_id}.json"
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        ai_logger.warning(f"Failed to save prompt performance: {e}")


def record_engagement_signal(
    chat_id: int,
    our_reply: str,
    their_response: Optional[str],
    response_delay_seconds: float,
    tone_used: str = "casual",
    strategy_used: str = "default",
):
    """Record engagement signal for prompt self-optimization.

    Called after we get a response to our reply to learn what works.
    """
    data = _load_prompt_performance(chat_id)
    data["total_replies"] = data.get("total_replies", 0) + 1

    # Compute engagement score
    # High score = they replied quickly with a long/engaged message
    # Low score = they didn't reply, or replied very slowly with little engagement
    if their_response is None:
        engagement = 0.0  # No response
    else:
        # Speed factor (faster = more engaged)
        speed_score = max(0, 1.0 - response_delay_seconds / 1800)  # 30 min = 0
        # Length factor
        length_score = min(len(their_response) / 100, 1.0)
        # Question factor (asking questions back = engaged)
        question_bonus = 0.2 if "?" in their_response else 0.0
        engagement = speed_score * 0.4 + length_score * 0.3 + question_bonus + 0.1

    engagement = round(min(engagement, 1.0), 3)

    data["engagement_scores"] = data.get("engagement_scores", [])
    data["engagement_scores"].append(engagement)
    data["engagement_scores"] = data["engagement_scores"][-100:]

    # Track successful vs unsuccessful tones
    if isinstance(data.get("successful_tones"), dict):
        data["successful_tones"] = Counter(data["successful_tones"])
    else:
        data["successful_tones"] = Counter()
    if isinstance(data.get("successful_strategies"), dict):
        data["successful_strategies"] = Counter(data["successful_strategies"])
    else:
        data["successful_strategies"] = Counter()

    if engagement > 0.5:
        data["successful_tones"][tone_used] = data["successful_tones"].get(tone_used, 0) + 1
        data["successful_strategies"][strategy_used] = data["successful_strategies"].get(strategy_used, 0) + 1
    elif engagement < 0.2 and their_response is not None:
        data["failed_patterns"] = data.get("failed_patterns", [])
        data["failed_patterns"].append({
            "our_reply_preview": our_reply[:50],
            "their_response_preview": their_response[:50] if their_response else None,
            "tone": tone_used,
        })
        data["failed_patterns"] = data["failed_patterns"][-20:]

    _save_prompt_performance(chat_id, data)


def get_prompt_optimization_hints(chat_id: int) -> str:
    """Get learned prompt optimization hints for this chat.

    Returns prompt additions based on engagement history.
    """
    data = _load_prompt_performance(chat_id)

    if data.get("total_replies", 0) < 10:
        return ""  # Not enough data

    hints = []

    # Average engagement trend
    scores = data.get("engagement_scores", [])
    if len(scores) >= 10:
        recent = sum(scores[-10:]) / 10
        overall = sum(scores) / len(scores)
        if recent < overall * 0.7:
            hints.append("ALERT: Recent engagement is dropping. Try different approaches.")
        elif recent > overall * 1.3:
            hints.append("Current approach is working well. Keep doing what you're doing.")

    # Best tones
    successful_tones = data.get("successful_tones", {})
    if successful_tones:
        best_tone = max(successful_tones, key=successful_tones.get)
        hints.append(f"Most engaging tone for this person: {best_tone}")

    # Failed patterns to avoid
    failed = data.get("failed_patterns", [])
    if len(failed) >= 3:
        # Find common failing tone
        fail_tones = Counter(f.get("tone", "") for f in failed[-10:])
        worst = fail_tones.most_common(1)
        if worst:
            hints.append(f"Avoid using '{worst[0][0]}' tone — it hasn't been working")

    return "\n".join(f"- {h}" for h in hints) if hints else ""


# ═══════════════════════════════════════════════════════════════
#  11. Persona Consistency System
# ═══════════════════════════════════════════════════════════════

_PERSONA_BREAKS = [
    # (pattern, severity, description)
    (r"\bas an ai\b", "critical", "Broke character as AI"),
    (r"\bi'?m a (language )?model\b", "critical", "Identified as language model"),
    (r"\bi cannot\b.*\breal\b", "critical", "Said something about not being real"),
    (r"\bi don'?t have (feelings|emotions|a body)\b", "critical", "Denied having feelings"),
    (r"\bI'?m not (actually|really)\b.*\b(person|human|real)\b", "critical", "Broke character"),
    (r"\bmy (training|programming)\b", "critical", "Referenced training"),
    (r"\bas a (virtual|digital)\b", "critical", "Identified as virtual"),
    (r"\bI (understand|appreciate|acknowledge)\b.*\b(your|how)\b", "high", "Therapist language"),
    (r"\bthat'?s (completely )?(valid|understandable)\b", "medium", "Therapist validation phrase"),
    (r"\bI'?m here (for you|to help|to listen)\b", "medium", "Support bot phrase"),
    (r"\bthank you for sharing\b", "high", "Customer service phrase"),
    (r"\bthat (sounds|must be) (really )?(hard|difficult|challenging)\b", "high", "Therapist empathy template"),
]


def check_persona_consistency(response: str) -> Dict[str, Any]:
    """Check if a response breaks persona / character.

    Returns:
        {
            "consistent": bool,
            "breaks": list of {"pattern": str, "severity": str, "description": str},
            "severity": "none" | "low" | "medium" | "high" | "critical",
        }
    """
    breaks = []
    for pattern, severity, desc in _PERSONA_BREAKS:
        if re.search(pattern, response, re.IGNORECASE):
            breaks.append({
                "pattern": pattern,
                "severity": severity,
                "description": desc,
            })

    if not breaks:
        return {"consistent": True, "breaks": [], "severity": "none"}

    # Overall severity = worst found
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    worst = max(breaks, key=lambda b: severity_order.get(b["severity"], 0))

    return {
        "consistent": False,
        "breaks": breaks,
        "severity": worst["severity"],
    }


# ═══════════════════════════════════════════════════════════════
#  MASTER FUNCTION: Run All Intelligence
# ═══════════════════════════════════════════════════════════════

def run_advanced_intelligence(
    chat_id: int,
    incoming_text: str,
    conversation_history: List[Dict[str, str]],
    nlp_analysis: Dict[str, Any],
    memory: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run the complete advanced intelligence pipeline.

    This is the main entry point called from telegram_api.py.

    Returns a comprehensive intelligence report with all features.
    """
    result = {}

    # 1. GoEmotions 28-label detection
    emotions_28 = detect_emotions_28(incoming_text)
    result["emotions_28"] = emotions_28

    # 2. Subtext detection
    subtext = detect_subtext(
        incoming_text, conversation_history,
        emotions_28=emotions_28, memory=memory,
    )
    result["subtext"] = subtext

    # 3. Conversation risk
    risk = detect_conversation_risk(
        chat_id, incoming_text, emotions_28,
        conversation_history, subtext=subtext,
    )
    result["risk"] = risk

    # 4. Emoji pattern analysis
    emoji_patterns = analyze_emoji_patterns(chat_id, incoming_text, conversation_history)
    result["emoji_patterns"] = emoji_patterns

    # 5. Personality profiling
    personality = profile_personality(chat_id, conversation_history)
    result["personality"] = personality

    # 6. Hidden reasoning chain
    memory_notes = []
    if memory:
        notes = memory.get("notes", [])
        memory_notes = [n["text"] if isinstance(n, dict) else n for n in notes[-5:]]

    hidden_reasoning = build_hidden_reasoning(
        incoming_text=incoming_text,
        emotions_28=emotions_28,
        conversation_stage=nlp_analysis.get("conversation_stage", "unknown"),
        topics=nlp_analysis.get("topics", ["casual"]),
        relationship_health=nlp_analysis.get("relationship_health", {"score": 50, "grade": "N/A"}),
        memory_notes=memory_notes,
        risk_assessment=risk,
        personality_profile=personality if not personality.get("insufficient_data") else None,
        subtext=subtext if subtext.get("has_subtext") else None,
    )
    result["hidden_reasoning"] = hidden_reasoning

    # 7. Prompt optimization hints (DSPy-style)
    prompt_hints = get_prompt_optimization_hints(chat_id)
    result["prompt_optimization"] = prompt_hints

    # 8. Vector memory retrieval (FAISS semantic search)
    vector_memories = format_vector_memory_for_prompt(chat_id, incoming_text, max_memories=5)
    result["vector_memory"] = vector_memories

    # 9. Auto-extract and store in vector memory
    auto_extract_and_store(chat_id, incoming_text, "Them", emotions_28)

    # 10. Reflection cycle (every N messages)
    increment_message_counter(chat_id)
    result["reflection"] = ""
    if should_reflect(chat_id):
        # Collect recent emotions for reflection
        recent_emotions = [emotions_28]  # At least current
        engagement_data = _load_prompt_performance(chat_id)
        reflection = run_reflection_cycle(
            chat_id, conversation_history, personality,
            recent_emotions, engagement_data,
        )
        if not reflection.get("skipped"):
            result["reflection"] = format_reflection_for_prompt(chat_id)
    else:
        result["reflection"] = format_reflection_for_prompt(chat_id)

    # 11. Reward model insights
    result["reward_insights"] = format_reward_insights_for_prompt(chat_id)

    return result


# ═══════════════════════════════════════════════════════════════
#  12. FAISS Vector Memory (Semantic Retrieval Layer)
# ═══════════════════════════════════════════════════════════════

_VECTOR_MEMORY_DIR = Path(__file__).parent / ".vector_memory"
_VECTOR_MEMORY_DIR.mkdir(exist_ok=True)

_faiss_available = None
_sentence_model = None


def _load_faiss_and_embeddings():
    """Lazy-load FAISS and sentence-transformers."""
    global _faiss_available, _sentence_model
    if _faiss_available is not None:
        return _faiss_available
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        _faiss_available = True
        ai_logger.info("FAISS + SentenceTransformer loaded for vector memory")
        return True
    except ImportError:
        _faiss_available = False
        ai_logger.info("FAISS not available — using fallback keyword memory")
        return False


# Per-chat vector stores (in-memory, persisted to disk)
_vector_stores: Dict[int, Any] = {}
_vector_texts: Dict[int, List[str]] = {}
_vector_metadata: Dict[int, List[Dict]] = {}

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _get_vector_store(chat_id: int):
    """Get or create FAISS index for a chat."""
    if not _faiss_available:
        return None
    import faiss
    if chat_id not in _vector_stores:
        index_path = _VECTOR_MEMORY_DIR / f"{chat_id}.faiss"
        texts_path = _VECTOR_MEMORY_DIR / f"{chat_id}_texts.json"
        meta_path = _VECTOR_MEMORY_DIR / f"{chat_id}_meta.json"

        if index_path.exists() and texts_path.exists():
            try:
                _vector_stores[chat_id] = faiss.read_index(str(index_path))
                _vector_texts[chat_id] = json.loads(texts_path.read_text())
                _vector_metadata[chat_id] = json.loads(meta_path.read_text()) if meta_path.exists() else []
                return _vector_stores[chat_id]
            except Exception as e:
                ai_logger.warning(f"Failed to load FAISS index: {e}")

        _vector_stores[chat_id] = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine after norm)
        _vector_texts[chat_id] = []
        _vector_metadata[chat_id] = []

    return _vector_stores[chat_id]


def _save_vector_store(chat_id: int):
    """Persist FAISS index to disk."""
    if not _faiss_available or chat_id not in _vector_stores:
        return
    import faiss
    try:
        faiss.write_index(_vector_stores[chat_id], str(_VECTOR_MEMORY_DIR / f"{chat_id}.faiss"))
        (_VECTOR_MEMORY_DIR / f"{chat_id}_texts.json").write_text(
            json.dumps(_vector_texts.get(chat_id, []), ensure_ascii=False)
        )
        (_VECTOR_MEMORY_DIR / f"{chat_id}_meta.json").write_text(
            json.dumps(_vector_metadata.get(chat_id, []), ensure_ascii=False, default=str)
        )
    except Exception as e:
        ai_logger.warning(f"Failed to save FAISS index: {e}")


def store_in_vector_memory(
    chat_id: int, text: str, memory_type: str = "conversation",
    emotional_tag: str = "neutral", importance: float = 0.5,
):
    """Store a piece of text in the vector memory for semantic retrieval.

    Args:
        chat_id: chat identifier
        text: text to store (conversation snippet, fact, summary)
        memory_type: "conversation" | "fact" | "summary" | "inside_joke" | "preference"
        emotional_tag: emotional context when this was stored
        importance: 0-1 weight for retrieval ranking
    """
    if not _load_faiss_and_embeddings() or not _sentence_model:
        return _store_keyword_memory(chat_id, text, memory_type, emotional_tag, importance)

    import numpy as np
    index = _get_vector_store(chat_id)
    if index is None:
        return

    # Encode and normalize for cosine similarity
    embedding = _sentence_model.encode([text], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32)

    index.add(embedding)
    _vector_texts[chat_id].append(text)
    _vector_metadata[chat_id].append({
        "type": memory_type,
        "emotion": emotional_tag,
        "importance": importance,
        "timestamp": datetime.now().isoformat(),
    })

    # Save every 10 entries
    if len(_vector_texts[chat_id]) % 10 == 0:
        _save_vector_store(chat_id)


def retrieve_from_vector_memory(
    chat_id: int, query: str, top_k: int = 5, min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """Retrieve semantically similar memories for the given query.

    Returns list of {"text": str, "score": float, "metadata": dict}
    """
    if not _faiss_available or not _sentence_model:
        return _retrieve_keyword_memory(chat_id, query, top_k)

    import numpy as np
    index = _get_vector_store(chat_id)
    if index is None or index.ntotal == 0:
        return []

    query_embedding = _sentence_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    texts = _vector_texts.get(chat_id, [])
    metadata = _vector_metadata.get(chat_id, [])

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(texts):
            continue
        if score < min_score:
            continue
        meta = metadata[idx] if idx < len(metadata) else {}
        # Boost by importance
        adjusted_score = float(score) * (0.7 + 0.3 * meta.get("importance", 0.5))
        results.append({
            "text": texts[idx],
            "score": round(adjusted_score, 3),
            "metadata": meta,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# Keyword-based fallback when FAISS isn't available
_keyword_memory: Dict[int, List[Dict]] = {}


def _store_keyword_memory(
    chat_id: int, text: str, memory_type: str,
    emotional_tag: str, importance: float,
):
    """Fallback: store text with keyword index."""
    if chat_id not in _keyword_memory:
        _keyword_memory[chat_id] = []
    _keyword_memory[chat_id].append({
        "text": text,
        "type": memory_type,
        "emotion": emotional_tag,
        "importance": importance,
        "timestamp": datetime.now().isoformat(),
        "keywords": set(text.lower().split()),
    })
    # Keep last 500 entries
    if len(_keyword_memory[chat_id]) > 500:
        _keyword_memory[chat_id] = _keyword_memory[chat_id][-500:]


def _retrieve_keyword_memory(
    chat_id: int, query: str, top_k: int,
) -> List[Dict[str, Any]]:
    """Fallback: retrieve by keyword overlap."""
    entries = _keyword_memory.get(chat_id, [])
    if not entries:
        return []
    query_words = set(query.lower().split())
    scored = []
    for entry in entries:
        overlap = len(query_words & entry.get("keywords", set()))
        if overlap > 0:
            score = overlap / max(len(query_words), 1) * entry.get("importance", 0.5)
            scored.append({
                "text": entry["text"],
                "score": round(score, 3),
                "metadata": {k: v for k, v in entry.items() if k not in ("text", "keywords")},
            })
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_k]


def auto_extract_and_store(
    chat_id: int, text: str, sender: str,
    emotions_28: Optional[Dict] = None,
):
    """Automatically extract memorable content and store in vector memory.

    Extracts: facts, preferences, inside jokes, emotional moments.
    """
    if not text or len(text) < 10:
        return

    emotional_tag = "neutral"
    importance = 0.3
    if emotions_28:
        emotional_tag = emotions_28.get("primary_emotion", "neutral")
        # High-arousal emotional moments are more memorable
        if emotions_28.get("arousal", 0) > 0.6:
            importance += 0.2
        if emotions_28.get("emotional_complexity", 0) > 0.5:
            importance += 0.1

    # Facts: "I work at...", "My name is...", "I'm from..."
    fact_patterns = [
        r"(?:i|my)\s+(?:work|job|name|live|from|study|studying|born)",
        r"(?:i'?m|i am)\s+(?:a |an |\d+|from|at|in)",
        r"(?:мен[яе]|я)\s+(?:работаю|зовут|живу|учусь|родил)",
    ]
    is_fact = any(re.search(p, text.lower()) for p in fact_patterns)
    if is_fact and sender == "Them":
        store_in_vector_memory(chat_id, f"[FACT] {text}", "fact", emotional_tag, min(importance + 0.3, 1.0))

    # Preferences: "I love...", "I hate...", "My favorite..."
    pref_patterns = [
        r"(?:i |my )(love|hate|like|prefer|favorite|favourite|can't stand)",
        r"(?:люблю|ненавижу|нравится|предпочитаю|обожаю)",
    ]
    is_pref = any(re.search(p, text.lower()) for p in pref_patterns)
    if is_pref and sender == "Them":
        store_in_vector_memory(chat_id, f"[PREFERENCE] {text}", "preference", emotional_tag, min(importance + 0.2, 1.0))

    # Emotional moments (high arousal)
    if emotions_28 and emotions_28.get("arousal", 0) > 0.7:
        store_in_vector_memory(
            chat_id,
            f"[EMOTIONAL_MOMENT] {sender}: {text} (emotion: {emotional_tag})",
            "emotional_moment",
            emotional_tag,
            min(importance + 0.2, 1.0),
        )

    # Long messages likely contain substance worth remembering
    if len(text) > 100 and sender == "Them":
        store_in_vector_memory(
            chat_id, f"[CONVERSATION] {text[:200]}",
            "conversation", emotional_tag, importance,
        )


def format_vector_memory_for_prompt(
    chat_id: int, query: str, max_memories: int = 5,
) -> str:
    """Retrieve relevant vector memories and format for prompt injection."""
    memories = retrieve_from_vector_memory(chat_id, query, top_k=max_memories)
    if not memories:
        return ""

    parts = ["## Relevant Memories (semantically retrieved):"]
    for mem in memories:
        meta = mem.get("metadata", {})
        mem_type = meta.get("type", "unknown")
        emotion = meta.get("emotion", "")
        parts.append(f"- [{mem_type}] {mem['text']} (relevance: {mem['score']:.0%})")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  13. Reflection Cycle (Persona Reflection)
# ═══════════════════════════════════════════════════════════════

_REFLECTION_DIR = Path(__file__).parent / ".reflections"
_REFLECTION_DIR.mkdir(exist_ok=True)
_message_counters: Dict[int, int] = {}
REFLECTION_INTERVAL = 30  # Reflect every N messages


def _load_reflections(chat_id: int) -> Dict[str, Any]:
    path = _REFLECTION_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "reflections": [],
        "learned_patterns": [],
        "dynamic_traits": {},
        "topics_to_revisit": [],
        "last_reflection_at": 0,
    }


def _save_reflections(chat_id: int, data: Dict):
    path = _REFLECTION_DIR / f"{chat_id}.json"
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        ai_logger.warning(f"Failed to save reflections: {e}")


def should_reflect(chat_id: int) -> bool:
    """Check if it's time for a reflection cycle."""
    count = _message_counters.get(chat_id, 0)
    return count > 0 and count % REFLECTION_INTERVAL == 0


def increment_message_counter(chat_id: int):
    """Increment the message counter for reflection tracking."""
    _message_counters[chat_id] = _message_counters.get(chat_id, 0) + 1


def run_reflection_cycle(
    chat_id: int,
    conversation_history: List[Dict[str, str]],
    personality: Dict[str, Any],
    emotions_28_recent: List[Dict[str, Any]],
    engagement_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run a reflection pass over recent conversation history.

    Reflection asks:
    1. What did I learn about them?
    2. How has the dynamic shifted?
    3. Any inconsistencies in my behavior?
    4. What topics should I bring up later?
    5. What's working and what isn't?

    Returns reflection insights that get stored and used in future prompts.
    """
    data = _load_reflections(chat_id)
    now = time.time()

    # Skip if reflected too recently (within 10 messages worth of time)
    if now - data.get("last_reflection_at", 0) < 300:  # 5 min cooldown
        return {"skipped": True, "reason": "too_recent"}

    reflection = {
        "timestamp": datetime.now().isoformat(),
        "message_count": _message_counters.get(chat_id, 0),
        "insights": [],
    }

    # 1. What did we learn?
    their_msgs = [m.get("text", "") for m in conversation_history[-30:] if m.get("sender") == "Them"]

    # Extract new facts
    new_facts = []
    for msg in their_msgs:
        for pattern in [r"(?:i|my)\s+(?:work|job|name|live)", r"(?:i'?m|i am)\s+\w+"]:
            if re.search(pattern, msg.lower()):
                new_facts.append(msg[:100])
    if new_facts:
        reflection["insights"].append(f"New facts learned: {'; '.join(new_facts[:3])}")

    # 2. Emotional trajectory
    if emotions_28_recent:
        recent_valences = [e.get("valence", 0) for e in emotions_28_recent[-10:]]
        avg_valence = sum(recent_valences) / max(len(recent_valences), 1)
        if avg_valence < -0.3:
            reflection["insights"].append("Recent emotional trajectory is NEGATIVE — relationship needs attention")
        elif avg_valence > 0.3:
            reflection["insights"].append("Recent emotional trajectory is POSITIVE — things are going well")
        else:
            reflection["insights"].append("Emotional trajectory is neutral — maintain engagement")

    # 3. Engagement trend
    if engagement_data:
        scores = engagement_data.get("engagement_scores", [])
        if len(scores) >= 5:
            recent_5 = sum(scores[-5:]) / 5
            if recent_5 < 0.3:
                reflection["insights"].append("ALERT: Engagement dropping. Need to change approach.")
                data["topics_to_revisit"].append("Ask about something they care about")
            elif recent_5 > 0.7:
                reflection["insights"].append("Engagement is high. Keep current approach.")

    # 4. Topics to bring up later
    interesting_topics = []
    for msg in their_msgs:
        if len(msg) > 50 and "?" not in msg:
            # Long statements = topics they care about
            words = msg.split()[:5]
            topic_hint = " ".join(words)
            interesting_topics.append(topic_hint)
    if interesting_topics:
        data["topics_to_revisit"] = (data.get("topics_to_revisit", []) + interesting_topics)[-10:]
        reflection["insights"].append(f"Topics to revisit later: {interesting_topics[:2]}")

    # 5. Dynamic trait updates
    if personality and not personality.get("insufficient_data"):
        data["dynamic_traits"] = {
            "current_mood_tendency": "positive" if avg_valence > 0 else "negative" if avg_valence < 0 else "neutral" if emotions_28_recent else "unknown",
            "engagement_level": "high" if engagement_data and sum(engagement_data.get("engagement_scores", [0])[-5:]) / 5 > 0.5 else "normal",
            "attachment_style": personality.get("attachment_style", "unknown"),
        }

    # Store reflection
    data["reflections"].append(reflection)
    data["reflections"] = data["reflections"][-20:]  # Keep last 20
    data["last_reflection_at"] = now
    _save_reflections(chat_id, data)

    ai_logger.info(f"Reflection cycle for {chat_id}: {len(reflection['insights'])} insights")
    return reflection


def format_reflection_for_prompt(chat_id: int) -> str:
    """Format stored reflections for prompt injection."""
    data = _load_reflections(chat_id)

    parts = []

    # Latest reflection insights
    if data.get("reflections"):
        latest = data["reflections"][-1]
        insights = latest.get("insights", [])
        if insights:
            parts.append("## Recent Reflection:")
            for ins in insights[-3:]:
                parts.append(f"- {ins}")

    # Topics to revisit
    topics = data.get("topics_to_revisit", [])
    if topics:
        parts.append(f"\nTopics you could bring up naturally: {', '.join(topics[-3:])}")

    # Dynamic traits
    traits = data.get("dynamic_traits", {})
    if traits:
        mood = traits.get("current_mood_tendency", "unknown")
        if mood != "unknown":
            parts.append(f"Their current mood tendency: {mood}")

    return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════
#  14. Enhanced Reward Model (Multi-Signal Engagement Tracking)
# ═══════════════════════════════════════════════════════════════

_REWARD_DIR = Path(__file__).parent / ".reward_model"
_REWARD_DIR.mkdir(exist_ok=True)


def _load_reward_data(chat_id: int) -> Dict[str, Any]:
    path = _REWARD_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "interactions": [],
        "reply_rate": 1.0,  # fraction of our messages they replied to
        "avg_response_speed": 300.0,  # seconds
        "avg_response_length": 30.0,
        "emoji_usage_rate": 0.5,
        "conversation_continuation_rate": 0.8,
        "total_interactions": 0,
    }


def _save_reward_data(chat_id: int, data: Dict):
    path = _REWARD_DIR / f"{chat_id}.json"
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        ai_logger.warning(f"Failed to save reward data: {e}")


def record_reward_signal(
    chat_id: int,
    our_message: str,
    their_reply: Optional[str],
    response_delay_seconds: float,
    their_reaction: Optional[str] = None,
    conversation_continued: bool = True,
    our_tone: str = "casual",
    our_strategy: str = "default",
):
    """Record a comprehensive reward signal from their response to our message.

    Tracks 5 signals from the research:
    1. Did they reply? (binary)
    2. How fast did they reply? (seconds)
    3. How long was their reply? (chars)
    4. Did they react with emoji? (warmth)
    5. Did the conversation continue after? (engagement)
    """
    data = _load_reward_data(chat_id)
    n = data.get("total_interactions", 0)

    interaction = {
        "timestamp": datetime.now().isoformat(),
        "our_message_preview": our_message[:80],
        "replied": their_reply is not None,
        "response_delay": round(response_delay_seconds, 1) if their_reply else None,
        "response_length": len(their_reply) if their_reply else 0,
        "had_reaction": their_reaction is not None,
        "reaction": their_reaction,
        "conversation_continued": conversation_continued,
        "tone": our_tone,
        "strategy": our_strategy,
    }

    # Compute reward score (0-1)
    reward = 0.0
    if their_reply:
        reward += 0.3  # They replied at all
        # Speed bonus (faster = better, max 30 min window)
        speed_score = max(0, 1.0 - response_delay_seconds / 1800)
        reward += speed_score * 0.2
        # Length bonus
        length_score = min(len(their_reply) / 100, 1.0)
        reward += length_score * 0.15
        # Emoji/reaction bonus
        if their_reaction:
            reward += 0.15
        emoji_in_reply = bool(re.search(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]', their_reply or ""))
        if emoji_in_reply:
            reward += 0.1
        # Continuation bonus
        if conversation_continued:
            reward += 0.1

    interaction["reward"] = round(min(reward, 1.0), 3)

    # Update running averages
    alpha = min(0.15, 1.0 / (n + 1))
    if their_reply:
        data["avg_response_speed"] = data["avg_response_speed"] * (1 - alpha) + response_delay_seconds * alpha
        data["avg_response_length"] = data["avg_response_length"] * (1 - alpha) + len(their_reply) * alpha
    data["reply_rate"] = data["reply_rate"] * (1 - alpha) + (1.0 if their_reply else 0.0) * alpha
    if their_reaction is not None:
        data["emoji_usage_rate"] = data["emoji_usage_rate"] * (1 - alpha) + 1.0 * alpha
    data["conversation_continuation_rate"] = (
        data["conversation_continuation_rate"] * (1 - alpha) +
        (1.0 if conversation_continued else 0.0) * alpha
    )

    data["interactions"].append(interaction)
    data["interactions"] = data["interactions"][-200:]  # Keep last 200
    data["total_interactions"] = n + 1

    _save_reward_data(chat_id, data)

    # Also record in DSPy-style prompt optimization
    record_engagement_signal(
        chat_id, our_message, their_reply,
        response_delay_seconds, our_tone, our_strategy,
    )

    return interaction


def score_candidate_with_reward_model(
    chat_id: int, candidate_response: str,
) -> float:
    """Use the reward model to predict engagement for a candidate response.

    Returns predicted engagement score (0-1).
    """
    data = _load_reward_data(chat_id)
    if data.get("total_interactions", 0) < 10:
        return 0.5  # Not enough data, neutral

    # Analyze candidate against what has worked
    score = 0.5  # Start neutral

    # Length analysis: how close is this to their typical response-generating length?
    avg_reply_len = data.get("avg_response_length", 30)
    our_len = len(candidate_response)
    # They engage more with messages similar in length to their own
    length_ratio = min(our_len, avg_reply_len * 2) / max(avg_reply_len * 2, 1)
    score += (1.0 - abs(1.0 - length_ratio)) * 0.15

    # Question engagement boost
    if "?" in candidate_response:
        score += 0.1

    # Check if we use patterns that got high rewards
    recent_good = [i for i in data.get("interactions", [])[-50:] if i.get("reward", 0) > 0.6]
    recent_bad = [i for i in data.get("interactions", [])[-50:] if i.get("reward", 0) < 0.2]

    # Reply rate factor
    reply_rate = data.get("reply_rate", 1.0)
    if reply_rate < 0.5:
        score -= 0.1  # Overall disengagement

    return round(min(max(score, 0.0), 1.0), 3)


def format_reward_insights_for_prompt(chat_id: int) -> str:
    """Format reward model insights for prompt injection."""
    data = _load_reward_data(chat_id)
    if data.get("total_interactions", 0) < 5:
        return ""

    parts = []

    reply_rate = data.get("reply_rate", 1.0)
    avg_speed = data.get("avg_response_speed", 300)
    avg_len = data.get("avg_response_length", 30)

    if reply_rate < 0.6:
        parts.append(f"⚠️ They only reply to {reply_rate:.0%} of your messages — be more engaging")
    if avg_speed > 600:
        parts.append("They typically take a while to reply — don't double-text")
    if avg_len < 20:
        parts.append("They prefer short messages — keep yours concise too")
    elif avg_len > 80:
        parts.append("They write long messages — matching their energy is good")

    # Recent reward trend
    interactions = data.get("interactions", [])
    if len(interactions) >= 10:
        recent_rewards = [i.get("reward", 0.5) for i in interactions[-10:]]
        avg_recent = sum(recent_rewards) / len(recent_rewards)
        if avg_recent < 0.3:
            parts.append("ALERT: Recent engagement is very low. Try a different approach entirely.")
        elif avg_recent > 0.7:
            parts.append("Engagement is high — current approach is working well.")

    return "\n".join(f"- {p}" for p in parts) if parts else ""


def format_advanced_intelligence_for_prompt(intelligence: Dict[str, Any]) -> str:
    """Format the full intelligence report into prompt injections."""
    parts = []

    # Hidden reasoning (most important)
    reasoning = intelligence.get("hidden_reasoning", {})
    reasoning_text = format_hidden_reasoning_for_prompt(reasoning)
    if reasoning_text:
        parts.append(reasoning_text)

    # Subtext warning
    subtext = intelligence.get("subtext", {})
    if subtext.get("has_subtext"):
        parts.append(
            f"\n⚠️ SUBTEXT DETECTED: {subtext['explanation']} "
            f"(confidence: {subtext['confidence']:.0%})\n"
            f"What they actually want: {subtext['likely_real_intent'].replace('_', ' ')}"
        )

    # Risk warning
    risk = intelligence.get("risk", {})
    if risk.get("risk_level") in ("high", "critical"):
        signals = ", ".join(f"{s[0]}" for s in risk.get("top_signals", []))
        parts.append(
            f"\n🚨 RISK LEVEL: {risk['risk_level'].upper()} "
            f"(signals: {signals})\n"
            f"Action: {risk.get('recommended_action', '')}"
        )
    elif risk.get("risk_level") == "medium":
        parts.append(f"\n⚠️ Minor risk signals detected. Be extra attentive.")

    # Emoji patterns
    emoji = intelligence.get("emoji_patterns", {})
    if emoji.get("anomaly"):
        parts.append(f"\n📊 Emoji anomaly: trend={emoji['trend']} — may indicate emotional shift")

    # Personality insights (only if sufficient data)
    personality = intelligence.get("personality", {})
    personality_text = format_personality_for_prompt(personality)
    if personality_text:
        parts.append(f"\n## Their Personality Profile:\n{personality_text}")

    # Prompt optimization (DSPy-style)
    opt = intelligence.get("prompt_optimization", "")
    if opt:
        parts.append(f"\n## Learned Preferences:\n{opt}")

    # Vector memory retrieval (FAISS)
    vec_mem = intelligence.get("vector_memory", "")
    if vec_mem:
        parts.append(f"\n{vec_mem}")

    # Reflection insights
    reflection = intelligence.get("reflection", "")
    if reflection:
        parts.append(f"\n{reflection}")

    # Reward model insights
    reward = intelligence.get("reward_insights", "")
    if reward:
        parts.append(f"\n## Engagement Model Insights:\n{reward}")

    return "\n".join(parts)


def warmup_models():
    """Eagerly preload all ML models at startup instead of lazy-loading on first message.

    Call this during server boot to eliminate the 1-2 minute delay on the first reply.
    """
    ai_logger.info("Warming up ML models (this may take a minute on first run)...")

    # GoEmotions 28-label
    loaded = _load_go_emotions()
    ai_logger.info(f"  GoEmotions 28-label: {'OK' if loaded else 'unavailable'}")

    # FAISS + SentenceTransformer
    loaded = _load_faiss_and_embeddings()
    ai_logger.info(f"  FAISS + SentenceTransformer: {'OK' if loaded else 'unavailable'}")

    ai_logger.info("ML model warmup complete.")
