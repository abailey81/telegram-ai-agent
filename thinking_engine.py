"""
Advanced Thinking Engine
=========================
Deep pre-generation reasoning, Monte Carlo response simulation,
response tree exploration, and strategic conversation planning.

This engine THINKS before generating a response:
1. Situation Assessment — what's happening in the conversation?
2. Intent Prediction — what do they want/expect from us?
3. Monte Carlo Simulation — simulate N possible responses, score outcomes
4. Strategic Planning — choose optimal conversation direction
5. Response Framing — frame the response with optimal angle/tone/length
6. Chain-of-thought reasoning — structured multi-step reasoning

All thinking happens BEFORE the LLM call, producing a reasoning
context that guides generation toward optimal outcomes.
"""

import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

thinking_logger = logging.getLogger("thinking_engine")

# ═══════════════════════════════════════════════════════════════
#  1. SITUATION ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def assess_situation(
    incoming_text: str,
    messages: List[Dict[str, Any]],
    nlp_analysis: Optional[Dict] = None,
    engagement: Optional[Dict] = None,
    conflict: Optional[Dict] = None,
    personality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Deep assessment of the current conversational situation.
    Determines what's happening, what's at stake, and what matters.
    """
    text_lower = incoming_text.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    assessment = {
        "message_type": _classify_message_type(incoming_text),
        "urgency": _assess_urgency(incoming_text, nlp_analysis),
        "emotional_temperature": _assess_emotional_temp(incoming_text, nlp_analysis),
        "stakes": _assess_stakes(incoming_text, messages, conflict),
        "conversation_phase": _detect_conversation_phase(messages),
        "their_intent": _predict_intent(incoming_text, nlp_analysis),
        "subtext": _detect_subtext(incoming_text, messages),
        "requires_action": _check_action_required(incoming_text),
    }

    return assessment


def _classify_message_type(text: str) -> str:
    """Classify the type of incoming message."""
    text_lower = text.lower().strip()

    if not text_lower:
        return "empty"
    if len(text_lower.split()) <= 2:
        # Short message classification
        if text_lower in ("hey", "hi", "hello", "yo", "sup", "heyy", "heyyy",
                          "привет", "хай", "здарова", "салют", "йо", "приветик", "хей"):
            return "greeting"
        if text_lower in ("ok", "okay", "k", "kk", "alright", "aight", "sure", "fine",
                          "ок", "окей", "ладно", "хорошо", "да", "ага", "угу", "норм"):
            return "acknowledgment"
        if text_lower in ("lol", "haha", "lmao", "😂", "🤣", "hehe",
                          "хаха", "ахах", "ахахах", "ржу", "хехе", "лол"):
            return "laugh_react"
        if text_lower in ("bye", "gn", "goodnight", "night", "ttyl", "later",
                          "пока", "спокойной ночи", "ночи", "споки", "до завтра", "давай"):
            return "farewell"
        if "?" in text_lower:
            return "short_question"
        return "brief"

    if text_lower.count("?") >= 2:
        return "multi_question"
    if "?" in text_lower:
        # What kind of question?
        if any(text_lower.startswith(w) for w in ("why", "how come", "explain",
                                                   "почему", "зачем", "отчего", "объясни")):
            return "why_question"
        if any(text_lower.startswith(w) for w in ("what", "which", "where", "when", "who",
                                                   "что", "какой", "какая", "где", "когда", "кто", "куда")):
            return "wh_question"
        if any(text_lower.startswith(w) for w in ("do you", "are you", "have you", "will you", "can you",
                                                   "ты", "а ты", "тебе", "можешь")):
            return "yes_no_question"
        return "question"

    if any(w in text_lower for w in ("tell me", "explain", "describe", "what do you think",
                                      "расскажи", "объясни", "опиши", "что ты думаешь", "как считаешь")):
        return "request_for_elaboration"
    if any(w in text_lower for w in ("im feeling", "i feel", "im so", "i just",
                                      "я чувствую", "мне", "я так", "я просто")):
        return "emotional_share"
    if any(w in text_lower for w in ("lets", "let's", "wanna", "want to", "should we", "we could",
                                      "давай", "хочешь", "может", "а если мы", "нам стоит")):
        return "proposal"
    if any(w in text_lower for w in ("guess what", "you know what", "omg", "dude", "bro listen",
                                      "слушай", "представь", "угадай", "короче", "знаешь что")):
        return "story_share"
    if re.search(r"\b(fuck|shit|hate|piss|angry|furious|бесит|ненавижу|достал|заебал|пиздец|блять)\b", text_lower):
        return "venting"
    if any(w in text_lower for w in ("miss you", "thinking about you", "love you", "❤️", "😍", "🥰",
                                      "скучаю", "думаю о тебе", "люблю тебя", "целую")):
        return "romantic"
    if any(w in text_lower for w in ("sorry", "my bad", "i apologize", "forgive",
                                      "извини", "прости", "мне жаль", "виноват", "виновата")):
        return "apology"
    if any(w in text_lower for w in ("thanks", "thank you", "appreciate", "grateful",
                                      "спасибо", "спс", "благодарю", "ценю")):
        return "gratitude"

    if len(words) > 30:
        return "long_message"

    return "statement"


def _assess_urgency(text: str, nlp: Optional[Dict]) -> str:
    """How urgently does this need a response?"""
    text_lower = text.lower()

    # High urgency
    if any(w in text_lower for w in ("help", "emergency", "urgent", "asap", "please respond", "answer me",
                                      "помоги", "срочно", "пожалуйста ответь", "ответь мне", "это важно")):
        return "high"
    if text_lower.count("?") >= 3:
        return "high"
    if re.search(r"[!?]{3,}", text):
        return "high"

    # Medium urgency
    if "?" in text:
        return "medium"
    if any(w in text_lower for w in ("what do you think", "your opinion", "thoughts")):
        return "medium"

    # Low
    if any(w in text_lower for w in ("lol", "haha", "ok", "cool", "nice")):
        return "low"

    return "normal"


def _assess_emotional_temp(text: str, nlp: Optional[Dict]) -> str:
    """Emotional temperature: frozen → cold → cool → neutral → warm → hot → boiling."""
    text_lower = text.lower()

    # Check NLP first
    if nlp:
        emotion = nlp.get("emotion") or nlp.get("primary_emotion", "")
        if isinstance(emotion, dict):
            emotion = emotion.get("primary", "")
        emotion = str(emotion).lower()

        if emotion in ("anger", "rage", "fury"):
            return "boiling"
        if emotion in ("frustration", "annoyance", "irritation"):
            return "hot"
        if emotion in ("love", "desire", "passion", "excitement"):
            return "hot"
        if emotion in ("sadness", "grief", "despair"):
            return "cold"
        if emotion in ("happiness", "joy"):
            return "warm"

    # Fallback to text analysis — check BOTH English and Russian
    _boiling_en = {
        "fuck", "fucking", "hate", "angry", "furious", "pissed", "stfu", "gtfo",
        "bitch", "asshole", "idiot", "moron", "pathetic", "disgusting",
        "bastard", "dickhead", "dumbass", "dipshit", "motherfucker",
        "drop dead", "die", "bullshit", "worthless", "screw you",
        "go to hell", "trash", "garbage", "loser", "hate you",
    }
    _boiling_ru = {
        "блядь", "блять", "сука", "сучка", "сучара", "пиздец", "пизда",
        "нахуй", "нахер", "ёбаный", "ебаный", "ебать", "заебал", "заебала",
        "отъебись", "гандон", "мудак", "мудила", "дебил", "долбоёб", "долбоеб",
        "тварь", "урод", "козёл", "козел", "скотина", "чмо", "ненавижу",
        "заткнись", "иди", "пошёл", "пошел", "пошла", "вали", "проваливай",
        "придурок", "кретин", "идиотка", "дура", "дурак", "лох", "отвали",
        "ублюдок", "выродок", "мразь", "подонок", "шлюха", "сволочь",
        "гнида", "падла", "паскуда", "хуй", "хуйня", "хуесос",
        "пидор", "пидорас", "уёбок", "уебок", "уёбище", "пиздабол",
        "засранец", "говно", "говнюк", "дерьмо", "катись", "убирайся", "свали",
        "мразота", "тупица", "бездарь",
    }
    _hot_en = {
        "annoying", "frustrated", "irritated", "mad", "upset", "piss off", "screw",
        "ridiculous", "embarrassing", "disgusted", "sick of", "tired of",
        "fed up", "cant stand", "crap", "damn", "ffs", "jfc",
    }
    _hot_ru = {
        "бесишь", "бесит", "достал", "достала", "раздражает", "надоел",
        "надоела", "задолбал", "задолбала", "злюсь", "злая", "злой",
        "заколебал", "заколебала", "запарил", "запарила", "утомил", "утомила",
        "напрягает", "замучил", "замучила", "чёрт", "капец", "жесть",
    }

    # Check NLP sentiment to disambiguate profanity direction
    _nlp_sentiment = "neutral"
    if nlp:
        _sent = nlp.get("sentiment", {})
        if isinstance(_sent, dict):
            _nlp_sentiment = _sent.get("sentiment", "neutral")
            _compound = _sent.get("compound", 0)
        else:
            _compound = 0

    _has_boiling = any(w in text_lower for w in _boiling_en) or any(w in text_lower for w in _boiling_ru)
    _has_hot = any(w in text_lower for w in _hot_en) or any(w in text_lower for w in _hot_ru)

    if _has_boiling:
        # Disambiguate: "fuck this is amazing" (positive) vs "fuck you" (negative)
        if _nlp_sentiment == "positive" or (_compound and _compound > 0.3):
            return "hot"  # Intense but positive — hot, not boiling
        return "boiling"
    if _has_hot:
        if _nlp_sentiment == "positive" or (_compound and _compound > 0.3):
            return "warm"  # Frustrated words in positive context = just warm
        return "hot"

    # CAPS = shouting = hot at minimum
    if len(text) > 5 and sum(1 for c in text if c.isupper()) / len(text) > 0.5:
        if _nlp_sentiment == "positive":
            return "warm"  # Excited typing, not shouting
        return "hot"

    if re.search(r"(love|miss|❤️|😍|amazing|cant wait|люблю|скучаю|❤)", text_lower):
        return "warm"
    if re.search(r"\b(whatever|meh|ok|sure|fine|ладно|окей|ок)\b", text_lower):
        return "cool"
    if re.search(r"\b(bye|leave|dont talk|blocked|уходи|пока|не пиши)\b", text_lower):
        return "frozen"

    # Use NLP compound for borderline cases
    if _compound:
        if _compound > 0.5:
            return "warm"
        if _compound < -0.5:
            return "cool"

    return "neutral"


def _assess_stakes(
    text: str, messages: List[Dict], conflict: Optional[Dict]
) -> str:
    """What's at stake in this interaction?"""
    text_lower = text.lower()

    # High-stakes conversational moments (use regex to avoid false positives)
    _critical_patterns = re.compile(
        r'\b(?:'
        r'where is this going|do you (?:love|still love) me|'
        r'i love you|lets? break up|we need to talk|its? over|'
        r'i cant do this anymore|are you serious about (?:us|me|this)|'
        r'что мы такое|куда мы движемся|ты меня любишь|'
        r'давай расстанемся|нам надо поговорить|всё кончено'
        r')\b',
        re.IGNORECASE,
    )
    # "are we" / "what are we" only critical when NOT followed by casual words
    _are_we_critical = re.compile(
        r'\b(?:what are we|are we)\b(?!\s+(?:eating|having|doing|watching|going to|getting|making|cooking|ordering))',
        re.IGNORECASE,
    )
    if _critical_patterns.search(text_lower) or _are_we_critical.search(text_lower):
        return "critical"

    # High conflict or direct insults = high stakes
    if conflict and conflict.get("level") == "high":
        return "high"
    _insult_markers = {
        # English
        "fuck", "fucking", "stfu", "gtfo", "bitch", "asshole", "idiot",
        "hate you", "moron", "loser", "dumbass", "dipshit", "motherfucker",
        "bastard", "dickhead", "drop dead", "bullshit", "die",
        "pathetic", "worthless", "trash", "garbage",
        # Russian
        "блядь", "блять", "сука", "сучка", "пиздец", "пизда",
        "нахуй", "нахер", "гандон", "мудак", "мудила", "дебил",
        "долбоёб", "долбоеб", "тварь", "урод", "ненавижу",
        "ёбаный", "ебаный", "ебать", "заебал", "заебала", "отъебись",
        "ублюдок", "мразь", "подонок", "шлюха", "сволочь", "гнида",
        "хуй", "хуйня", "хуесос", "пидор", "уёбок", "уебок",
        "пиздабол", "говно", "дерьмо", "скотина", "чмо",
    }
    if any(w in text_lower for w in _insult_markers):
        return "high"

    # Trust/betrayal
    if any(w in text_lower for w in ("cheating", "lied", "betrayed", "trust", "honest")):
        return "high"

    # Normal question/chat
    if "?" in text:
        return "moderate"

    return "low"


def _detect_conversation_phase(messages: List[Dict]) -> str:
    """Detect what phase the conversation is in."""
    if not messages:
        return "opening"

    count = len(messages)
    if count <= 3:
        return "opening"
    elif count <= 10:
        return "building"
    elif count <= 30:
        return "flowing"
    else:
        return "deep"


def _predict_intent(text: str, nlp: Optional[Dict]) -> str:
    """Predict what the person wants/expects from us."""
    text_lower = text.lower()

    if "?" in text:
        if any(w in text_lower for w in ("what do you think", "opinion", "how do you feel")):
            return "seeking_opinion"
        if any(w in text_lower for w in ("do you", "are you", "have you", "will you")):
            return "seeking_confirmation"
        if any(w in text_lower for w in ("why", "how come", "explain")):
            return "seeking_explanation"
        return "seeking_answer"

    if any(w in text_lower for w in ("im feeling", "im so", "i feel")):
        return "wanting_reaction"
    if any(w in text_lower for w in ("guess what", "you wont believe", "omg so")):
        return "wanting_engagement"
    if any(w in text_lower for w in ("sorry", "my bad", "i apologize")):
        return "seeking_forgiveness"
    if any(w in text_lower for w in ("miss you", "love you", "thinking of you")):
        return "expressing_affection"
    if any(w in text_lower for w in ("fuck", "hate", "pissed", "angry at")):
        return "venting_anger"
    if any(w in text_lower for w in ("ok", "sure", "fine", "whatever")):
        return "low_effort_response"
    if any(w in text_lower for w in ("haha", "lol", "lmao", "😂")):
        return "social_bonding"

    return "general_communication"


def _detect_subtext(text: str, messages: List[Dict]) -> Optional[str]:
    """Detect hidden meaning / subtext in the message.

    Uses context from recent messages to distinguish genuine responses
    from passive-aggressive, sarcastic, or emotionally loaded ones.
    """
    text_lower = text.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    # Build recent conversation context
    recent = messages[-8:] if messages else []
    recent_text = " ".join(m.get("text", "") for m in recent).lower()
    _had_tension = any(w in recent_text for w in (
        "sorry", "wrong", "fight", "argue", "mad", "angry", "fuck", "hate",
        "upset", "hurt", "pissed", "annoyed",
        "ссора", "извини", "прости", "злюсь", "бесит", "обид",
        "неправ", "виноват", "ошиб", "накосяч", "поругал",
        "достал", "надоел", "задолбал", "зол",
    ))
    _had_affection = any(w in recent_text for w in (
        "love", "miss", "babe", "baby", "beautiful",
        "люблю", "скучаю", "малыш", "красив",
    ))

    # 1) Passive-aggressive dismissals (expanded — not just exact matches)
    _dismissive = re.compile(
        r'^(fine|whatever|ok|sure|cool|okay|alright|ладно|ок|окей|хорошо|ну хорошо|как хочешь|мне все равно|пофиг|ну ок|ну ладно)[\.\!\s]*$',
        re.IGNORECASE,
    )
    if _dismissive.match(text_lower) and _had_tension:
        return "passive_aggressive_dismissal"

    # 2) "Fine" variants in longer text
    if re.search(r'\b(fine then|whatever then|okay then|suit yourself|ну и ладно|ну и пожалуйста)\b', text_lower):
        return "passive_aggressive_dismissal"

    # 3) Disappointed acceptance
    if re.search(r'oh\s+okay|oh ok|oh\.\.\.|oh right|oh\s+alright|а\.{2,}|аа?\s*ладно\.{0,3}|ну ок\.{2,}|ну ладно\.{2,}|аа?\s*окей', text_lower):
        return "disappointed_acceptance"

    # 4) Minimal effort — possibly upset
    if text_lower in ("k", "kk", "к", "кк", "ага", "угу"):
        if _had_tension or _had_affection:  # Low effort after conflict OR after affection = signal
            return "minimal_effort_possibly_upset"

    # 5) "I'm fine" when they're probably not
    if re.search(r"i'?m fine|its fine|it'?s whatever|don'?t worry about it|"
                 r"всё нормально|всё хорошо|ничего|не парься|не переживай", text_lower):
        if _had_tension:
            return "possibly_not_fine"

    # 6) Passive-aggressive "freedom" giving
    if re.search(r"do what you want|up to you|your choice|whatever you want|"
                 r"как хочешь|решай сам|решай сама|твоё дело|делай что хочешь", text_lower):
        return "passive_aggressive_freedom"

    # 7) Deflecting with humor
    if re.search(r"haha\s*(yeah|sure|ok|right|fine)|lol\s*(ok|sure|yeah|fine)|"
                 r"хаха\s*(ладно|ок|да|ну)|ахах\s*(ладно|ок|ну)", text_lower):
        return "deflecting_with_humor"

    # 8) Guilt trip
    if re.search(r"i'?ll just|then i'?ll just|guess i'?ll|"
                 r"i'?ll be alone|nobody cares|no one cares|"
                 r"ну тогда я сам|значит я один|никому не нужен|никому нет дела", text_lower):
        return "guilt_trip"

    # 9) Soft rejection / avoidance
    if re.search(r"i don'?t know\.{2,}|maybe later|we'?ll see|"
                 r"not sure\.{2,}|i guess|"
                 r"не знаю\.{2,}|потом|может быть|посмотрим|ну наверное", text_lower):
        if word_count <= 5:  # Short non-committal = avoidance signal
            return "soft_rejection"

    # 10) Testing / probing
    if re.search(r"would you (even|still|ever)|do you (even|actually|really)|"
                 r"ты вообще|ты реально|тебе вообще|а тебе не|ты хоть", text_lower):
        return "testing_loyalty"

    # 11) Emotional shutdown ("I'm tired" meaning tired of this)
    if re.search(r"i'?m tired|exhausted|can'?t anymore|over it|done|"
                 r"устал|устала|не могу больше|всё|хватит|надоело", text_lower):
        if _had_tension and word_count <= 4:
            return "emotional_shutdown"

    # 12) Sarcasm markers
    if re.search(r"oh sure|yeah right|oh wow really|oh great|sure sure|"
                 r"ну конечно|ага конечно|ну да ну да|ой как мило", text_lower):
        return "sarcasm"

    # 13) Longing / missing (not explicitly said)
    if re.search(r"remember when|i miss how|it was better when|those days|"
                 r"помнишь как|скучаю по тому|раньше было", text_lower):
        return "nostalgic_longing"

    return None


def _check_action_required(text: str) -> List[str]:
    """Check if the message requires specific actions."""
    text_lower = text.lower()
    actions = []

    if "?" in text:
        actions.append("answer_question")
    if any(w in text_lower for w in ("send me", "show me", "give me")):
        actions.append("provide_content")
    if any(w in text_lower for w in ("call me", "voice", "voice note", "voice message")):
        actions.append("voice_response")
    if any(w in text_lower for w in ("lets meet", "wanna hang", "come over", "see you")):
        actions.append("plan_meetup")
    if any(w in text_lower for w in ("promise", "swear", "guarantee")):
        actions.append("make_commitment")

    return actions


# ═══════════════════════════════════════════════════════════════
#  2. MONTE CARLO RESPONSE SIMULATION
# ═══════════════════════════════════════════════════════════════

# Response strategy archetypes
RESPONSE_STRATEGIES = {
    "match_energy": {
        "description": "Mirror their exact tone and energy level",
        "best_for": ["venting", "emotional_share", "romantic", "social_bonding"],
        "risk": "low",
    },
    "escalate_up": {
        "description": "Respond with MORE energy than they gave",
        "best_for": ["greeting", "proposal", "story_share"],
        "risk": "medium",
    },
    "cool_down": {
        "description": "Pull back slightly but stay firm — not soft, just controlled",
        "best_for": ["boiling_temp", "high_conflict"],
        "risk": "medium",
    },
    "playful_tease": {
        "description": "Light teasing, banter, challenge them",
        "best_for": ["building", "flowing", "social_bonding"],
        "risk": "medium",
    },
    "direct_honest": {
        "description": "Straight-up honest, no games",
        "best_for": ["why_question", "seeking_explanation", "critical_stakes"],
        "risk": "low",
    },
    "mysterious_pull": {
        "description": "Be slightly mysterious, make them curious",
        "best_for": ["opening", "building", "low_engagement"],
        "risk": "medium",
    },
    "ride_with_them": {
        "description": "Be on their side — raw solidarity, not therapy",
        "best_for": ["emotional_share", "venting_anger", "apology"],
        "risk": "low",
    },
    "challenge_push": {
        "description": "Push back, disagree, create friction (builds engagement)",
        "best_for": ["low_effort_response", "acknowledgment", "passive_aggressive"],
        "risk": "high",
    },
    "open_up_real": {
        "description": "Be genuinely open — not performative, just real",
        "best_for": ["deep_phase", "romantic", "critical_stakes"],
        "risk": "high",
    },
    "humor_deflect": {
        "description": "Use humor to navigate tricky situations",
        "best_for": ["high_tension", "awkward_moment", "deflecting_with_humor"],
        "risk": "low",
    },
}


def monte_carlo_simulate(
    situation: Dict[str, Any],
    engagement: Optional[Dict] = None,
    personality: Optional[Dict] = None,
    n_simulations: int = 50,
) -> Dict[str, Any]:
    """
    Monte Carlo simulation of response strategies.
    Simulates N scenarios for each strategy and picks the best.

    Returns ranked strategies with expected outcomes.
    """
    msg_type = situation.get("message_type", "statement")
    temp = situation.get("emotional_temperature", "neutral")
    stakes = situation.get("stakes", "low")
    phase = situation.get("conversation_phase", "building")
    intent = situation.get("their_intent", "general_communication")
    subtext = situation.get("subtext")
    urgency = situation.get("urgency", "normal")

    eng_score = (engagement or {}).get("engagement_score", 0.5)
    archetype = (personality or {}).get("archetype", "unknown")
    att_style = (personality or {}).get("attachment_style", {}).get("primary", "unknown")

    results = {}

    for strategy_name, strategy in RESPONSE_STRATEGIES.items():
        scores = []

        for _ in range(n_simulations):
            score = _simulate_outcome(
                strategy_name, strategy,
                msg_type, temp, stakes, phase, intent, subtext, urgency,
                eng_score, archetype, att_style,
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        # Variance as risk indicator
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        results[strategy_name] = {
            "expected_outcome": round(avg_score, 3),
            "worst_case": round(min_score, 3),
            "best_case": round(max_score, 3),
            "risk_variance": round(variance, 4),
            "description": strategy["description"],
        }

    # Rank by expected outcome
    ranked = sorted(results.items(), key=lambda x: x[1]["expected_outcome"], reverse=True)

    # Pick top strategy (but consider risk)
    best = ranked[0]
    # If best has high variance and #2 is close, prefer #2
    if (len(ranked) > 1
            and best[1]["risk_variance"] > 0.05
            and ranked[1][1]["expected_outcome"] > best[1]["expected_outcome"] - 0.05
            and ranked[1][1]["risk_variance"] < best[1]["risk_variance"] * 0.7):
        best = ranked[1]

    return {
        "recommended_strategy": best[0],
        "recommended_score": best[1]["expected_outcome"],
        "strategy_description": best[1]["description"],
        "all_strategies": {name: data for name, data in ranked},
        "simulations_run": n_simulations * len(RESPONSE_STRATEGIES),
    }


def _simulate_outcome(
    strategy_name: str, strategy: Dict,
    msg_type: str, temp: str, stakes: str, phase: str,
    intent: str, subtext: Optional[str], urgency: str,
    eng_score: float, archetype: str, att_style: str,
) -> float:
    """
    Simulate a single outcome for a strategy.
    Returns score 0.0 (terrible outcome) to 1.0 (perfect outcome).
    """
    score = 0.5  # baseline
    noise = random.gauss(0, 0.08)  # random variance

    # --- Strategy-situation fit ---
    best_for = strategy.get("best_for", [])

    # Message type match
    if msg_type in best_for:
        score += 0.15
    elif f"{temp}_temp" in best_for:
        score += 0.12
    elif f"{stakes}_stakes" in best_for:
        score += 0.12
    elif f"{phase}_phase" in best_for or phase in best_for:
        score += 0.1

    # --- Temperature-strategy interaction ---
    temp_scores = {
        ("boiling", "match_energy"): 0.1,
        ("boiling", "cool_down"): -0.05,  # they want a fight, not de-escalation
        ("boiling", "challenge_push"): 0.15,  # fire meets fire
        ("boiling", "humor_deflect"): -0.1,  # not the time
        ("hot", "match_energy"): 0.12,
        ("hot", "playful_tease"): -0.1,  # risky when hot
        ("warm", "escalate_up"): 0.1,
        ("warm", "playful_tease"): 0.15,
        ("warm", "mysterious_pull"): 0.1,
        ("cool", "escalate_up"): 0.05,
        ("cool", "challenge_push"): 0.08,
        ("cool", "match_energy"): -0.05,  # matching cool = boring
        ("frozen", "direct_honest"): 0.1,
        ("frozen", "open_up_real"): -0.1,  # wrong time
        ("neutral", "playful_tease"): 0.08,
    }
    score += temp_scores.get((temp, strategy_name), 0)

    # --- Stakes-strategy interaction ---
    if stakes == "critical":
        if strategy_name in ("humor_deflect", "playful_tease"):
            score -= 0.2  # wrong time for jokes
        if strategy_name in ("direct_honest", "open_up_real"):
            score += 0.15
    elif stakes == "low":
        if strategy_name == "open_up_real":
            score -= 0.1  # too heavy for casual

    # --- Intent-strategy interaction ---
    intent_scores = {
        ("seeking_answer", "direct_honest"): 0.15,
        ("seeking_opinion", "direct_honest"): 0.12,
        ("wanting_reaction", "match_energy"): 0.15,
        ("wanting_engagement", "escalate_up"): 0.15,
        ("expressing_affection", "match_energy"): 0.12,
        ("expressing_affection", "open_up_real"): 0.1,
        ("venting_anger", "match_energy"): 0.15,
        ("venting_anger", "challenge_push"): 0.08,
        ("low_effort_response", "challenge_push"): 0.12,
        ("low_effort_response", "mysterious_pull"): 0.1,
        ("social_bonding", "playful_tease"): 0.12,
        ("social_bonding", "humor_deflect"): 0.1,
    }
    score += intent_scores.get((intent, strategy_name), 0)

    # --- Subtext awareness ---
    if subtext:
        subtext_scores = {
            ("passive_aggressive_dismissal", "challenge_push"): 0.15,
            ("passive_aggressive_dismissal", "direct_honest"): 0.1,
            ("passive_aggressive_dismissal", "humor_deflect"): -0.1,
            ("possibly_not_fine", "direct_honest"): 0.12,
            ("possibly_not_fine", "match_energy"): -0.15,  # matching "fine" misses the point
            ("minimal_effort_possibly_upset", "challenge_push"): 0.1,
            ("deflecting_with_humor", "direct_honest"): 0.1,
        }
        score += subtext_scores.get((subtext, strategy_name), 0)

    # --- Engagement level interaction ---
    if eng_score < 0.3:
        # Low engagement: need to spark interest
        if strategy_name in ("mysterious_pull", "challenge_push"):
            score += 0.1
        if strategy_name in ("ride_with_them", "open_up_real"):
            score -= 0.1  # too intense for someone disengaged
    elif eng_score > 0.7:
        # High engagement: ride the wave
        if strategy_name in ("escalate_up", "playful_tease"):
            score += 0.1

    # --- Personality interaction ---
    archetype_scores = {
        ("challenger", "challenge_push"): 0.15,
        ("challenger", "ride_with_them"): -0.1,
        ("gentle_introvert", "challenge_push"): -0.15,
        ("gentle_introvert", "ride_with_them"): 0.12,
        ("volatile_reactor", "match_energy"): 0.12,
        ("volatile_reactor", "humor_deflect"): -0.08,
        ("social_butterfly", "escalate_up"): 0.12,
        ("social_butterfly", "playful_tease"): 0.1,
        ("power_player", "challenge_push"): 0.1,
        ("power_player", "open_up_real"): -0.15,
    }
    score += archetype_scores.get((archetype, strategy_name), 0)

    # --- Attachment style interaction ---
    att_scores = {
        ("anxious", "mysterious_pull"): -0.15,  # triggers anxiety
        ("anxious", "ride_with_them"): 0.1,
        ("avoidant", "open_up_real"): -0.15,
        ("avoidant", "mysterious_pull"): 0.1,
        ("avoidant", "playful_tease"): 0.08,
        ("disorganized", "direct_honest"): 0.12,
    }
    score += att_scores.get((att_style, strategy_name), 0)

    # --- Risk penalty ---
    risk = strategy.get("risk", "low")
    if risk == "high":
        # High risk strategies have wider variance
        noise *= 2.0

    score += noise
    return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════
#  3. RESPONSE PREDICTION (What will they say next?)
# ═══════════════════════════════════════════════════════════════

def predict_their_response(
    our_proposed_reply: str,
    situation: Dict[str, Any],
    personality: Optional[Dict] = None,
    engagement: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Predict how they'll respond to our proposed reply.
    Uses Monte Carlo simulation of likely reactions.
    """
    their_archetype = (personality or {}).get("archetype", "unknown")
    att_style = (personality or {}).get("attachment_style", {}).get("primary", "unknown")
    eng_score = (engagement or {}).get("engagement_score", 0.5)
    current_temp = situation.get("emotional_temperature", "neutral")

    our_text = our_proposed_reply.lower()
    our_length = len(our_text.split())

    # Classify our response
    our_has_question = "?" in our_proposed_reply
    our_is_short = our_length <= 5
    our_is_aggressive = bool(re.search(r"\b(wtf|stfu|lol ok|u done|whatever)\b", our_text))
    our_is_sweet = bool(re.search(r"\b(miss you|love|❤️|beautiful|thinking of you)\b", our_text))
    our_is_funny = bool(re.search(r"\b(lol|haha|lmao|💀|😂)\b", our_text))

    # Simulate N outcomes
    n_sims = 30
    outcomes = defaultdict(int)

    for _ in range(n_sims):
        outcome = _simulate_their_reaction(
            our_has_question, our_is_short, our_is_aggressive,
            our_is_sweet, our_is_funny, our_length,
            current_temp, their_archetype, att_style, eng_score,
        )
        outcomes[outcome] += 1

    # Convert to probabilities
    probs = {k: v / n_sims for k, v in outcomes.items()}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    most_likely = sorted_probs[0][0]
    second_likely = sorted_probs[1][0] if len(sorted_probs) > 1 else "unknown"

    return {
        "most_likely_reaction": most_likely,
        "probability": round(sorted_probs[0][1], 2),
        "second_likely": second_likely,
        "second_probability": round(sorted_probs[1][1], 2) if len(sorted_probs) > 1 else 0,
        "all_outcomes": dict(sorted_probs),
        "favorable_outcome_prob": round(
            sum(v for k, v in probs.items() if k in (
                "engaged_reply", "enthusiastic", "flirty_back", "laughing",
                "asks_question", "opens_up",
            )), 2
        ),
    }


def _simulate_their_reaction(
    our_has_question: bool, our_is_short: bool, our_is_aggressive: bool,
    our_is_sweet: bool, our_is_funny: bool, our_length: int,
    current_temp: str, archetype: str, att_style: str, eng_score: float,
) -> str:
    """Simulate a single reaction outcome."""
    # Possible outcomes
    outcomes = [
        "engaged_reply", "enthusiastic", "flirty_back", "laughing",
        "asks_question", "opens_up",  # positive
        "short_reply", "acknowledgment_only", "delayed_reply",  # neutral
        "ignores", "annoyed", "defensive", "escalates",  # negative
    ]

    weights = {o: 1.0 for o in outcomes}

    # --- Our message effects ---
    if our_has_question:
        weights["engaged_reply"] += 3.0
        weights["asks_question"] += 1.5
        weights["short_reply"] += 1.0
        weights["ignores"] -= 0.5

    if our_is_short:
        weights["short_reply"] += 2.0
        weights["acknowledgment_only"] += 1.5
        weights["engaged_reply"] -= 1.0

    if our_is_aggressive:
        if current_temp in ("boiling", "hot"):
            weights["escalates"] += 3.0
            weights["defensive"] += 2.0
        else:
            weights["annoyed"] += 2.0
            weights["defensive"] += 1.0

    if our_is_sweet:
        if current_temp in ("warm", "neutral"):
            weights["flirty_back"] += 3.0
            weights["enthusiastic"] += 2.0
        elif current_temp in ("cold", "frozen"):
            weights["acknowledgment_only"] += 2.0  # too late for sweet

    if our_is_funny:
        weights["laughing"] += 3.0
        weights["engaged_reply"] += 1.5
        weights["escalates"] -= 1.0

    # --- Engagement effect ---
    if eng_score > 0.7:
        weights["engaged_reply"] += 2.0
        weights["enthusiastic"] += 1.5
        weights["ignores"] -= 2.0
    elif eng_score < 0.3:
        weights["ignores"] += 2.0
        weights["delayed_reply"] += 2.0
        weights["engaged_reply"] -= 2.0

    # --- Archetype effect ---
    if archetype == "challenger":
        weights["asks_question"] += 1.0
        weights["escalates"] += 1.0
    elif archetype == "gentle_introvert":
        weights["opens_up"] += 0.5
        weights["short_reply"] += 1.0
    elif archetype == "social_butterfly":
        weights["enthusiastic"] += 2.0
        weights["engaged_reply"] += 1.5
    elif archetype == "volatile_reactor":
        weights["escalates"] += 1.5
        weights["enthusiastic"] += 1.0  # extremes

    # --- Attachment effect ---
    if att_style == "anxious":
        weights["engaged_reply"] += 1.0
        weights["asks_question"] += 1.5  # seeks reassurance
    elif att_style == "avoidant":
        weights["short_reply"] += 1.5
        weights["delayed_reply"] += 1.0
        weights["opens_up"] -= 1.0

    # Normalize and sample
    # Clamp weights to positive
    for k in weights:
        weights[k] = max(0.1, weights[k])

    total = sum(weights.values())
    r = random.random() * total
    cumulative = 0
    for outcome, w in weights.items():
        cumulative += w
        if r <= cumulative:
            return outcome

    return "engaged_reply"  # fallback


# ═══════════════════════════════════════════════════════════════
#  3B. ADVANCED MONTE CARLO — MULTI-ROUND TRAJECTORY SIMULATION
# ═══════════════════════════════════════════════════════════════

# Historical outcome tracking for Bayesian prior updates
_MC_HISTORY_FILE = Path("engine_data/mc_history.json")
_MC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

_mc_history_cache: Optional[Dict[str, Any]] = None


def _load_mc_history() -> Dict[str, Any]:
    global _mc_history_cache
    if _mc_history_cache is not None:
        return _mc_history_cache
    if _MC_HISTORY_FILE.exists():
        try:
            _mc_history_cache = json.loads(_MC_HISTORY_FILE.read_text())
            return _mc_history_cache
        except Exception:
            pass
    _mc_history_cache = {"strategy_outcomes": {}, "trajectory_actuals": [], "updated": None}
    return _mc_history_cache


def _save_mc_history(history: Dict[str, Any]):
    global _mc_history_cache
    history["updated"] = datetime.now().isoformat()
    _mc_history_cache = history
    try:
        _MC_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str))
    except Exception:
        pass


def record_mc_outcome(
    strategy_used: str,
    predicted_score: float,
    actual_outcome: str,
    chat_id: int,
):
    """Record actual outcome for Bayesian prior updating.

    Call this when we get their response to learn from our predictions.
    """
    history = _load_mc_history()
    outcomes = history.setdefault("strategy_outcomes", {})
    entries = outcomes.setdefault(strategy_used, [])

    outcome_score = {
        "engaged_reply": 0.8, "enthusiastic": 0.95, "flirty_back": 0.9,
        "laughing": 0.85, "asks_question": 0.75, "opens_up": 0.9,
        "short_reply": 0.4, "acknowledgment_only": 0.3, "delayed_reply": 0.35,
        "ignores": 0.1, "annoyed": 0.15, "defensive": 0.2, "escalates": 0.1,
    }.get(actual_outcome, 0.5)

    entries.append({
        "predicted": predicted_score,
        "actual": outcome_score,
        "error": abs(predicted_score - outcome_score),
        "chat_id": chat_id,
        "timestamp": time.time(),
    })

    # Keep last 100 per strategy
    if len(entries) > 100:
        outcomes[strategy_used] = entries[-100:]

    _save_mc_history(history)


def _get_bayesian_prior(strategy_name: str) -> float:
    """Get Bayesian-updated prior for a strategy based on historical performance."""
    history = _load_mc_history()
    entries = history.get("strategy_outcomes", {}).get(strategy_name, [])

    if not entries:
        return 0.0  # no adjustment

    # Exponentially weighted moving average of actual outcomes
    # Recent outcomes matter more
    total_weight = 0.0
    weighted_sum = 0.0
    for i, entry in enumerate(entries):
        decay = math.exp(-0.05 * (len(entries) - 1 - i))
        weighted_sum += entry["actual"] * decay
        total_weight += decay

    if total_weight == 0:
        return 0.0

    historical_avg = weighted_sum / total_weight
    # Bias: shift score toward historical performance
    # 0.5 is neutral, so (historical_avg - 0.5) is the bias
    return (historical_avg - 0.5) * 0.15  # max ±0.075 shift


def multi_round_trajectory_simulate(
    situation: Dict[str, Any],
    strategy_name: str,
    engagement: Optional[Dict] = None,
    personality: Optional[Dict] = None,
    n_simulations: int = 30,
    rounds: int = 3,
) -> Dict[str, Any]:
    """
    Advanced multi-round Monte Carlo simulation.

    Instead of simulating just ONE response, simulates entire
    conversation trajectories over multiple rounds.

    Returns trajectory forecast with:
    - Expected conversation evolution (warming/cooling/escalating)
    - Probability of conversation dying
    - Expected engagement delta
    - Risk of ghosting or conflict escalation
    """
    eng_score = (engagement or {}).get("engagement_score", 0.5)
    archetype = (personality or {}).get("archetype", "unknown")
    att_style = (personality or {}).get("attachment_style", {}).get("primary", "unknown")
    current_temp = situation.get("emotional_temperature", "neutral")

    temp_order = ["frozen", "cold", "cool", "neutral", "warm", "hot", "boiling"]

    trajectories = []

    for _ in range(n_simulations):
        traj = {
            "rounds": [],
            "final_temp": current_temp,
            "final_engagement": eng_score,
            "conversation_died": False,
            "conflict_escalated": False,
            "ghosted": False,
        }

        sim_temp = current_temp
        sim_eng = eng_score

        for round_num in range(rounds):
            # Simulate their reaction
            # Adjust parameters per round based on evolving state
            reaction = _simulate_their_reaction(
                our_has_question=(round_num == 0),  # first round mirrors our message
                our_is_short=(strategy_name in ("mysterious_pull", "cool_down")),
                our_is_aggressive=(strategy_name == "challenge_push"),
                our_is_sweet=(strategy_name == "ride_with_them"),
                our_is_funny=(strategy_name in ("humor_deflect", "playful_tease")),
                our_length=15,  # avg estimate
                current_temp=sim_temp,
                archetype=archetype,
                att_style=att_style,
                eng_score=sim_eng,
            )

            # Update simulated state based on reaction
            reaction_effects = {
                "engaged_reply": {"eng_delta": 0.08, "temp_shift": 0},
                "enthusiastic": {"eng_delta": 0.12, "temp_shift": 1},
                "flirty_back": {"eng_delta": 0.1, "temp_shift": 1},
                "laughing": {"eng_delta": 0.06, "temp_shift": 0},
                "asks_question": {"eng_delta": 0.05, "temp_shift": 0},
                "opens_up": {"eng_delta": 0.1, "temp_shift": 1},
                "short_reply": {"eng_delta": -0.05, "temp_shift": -1},
                "acknowledgment_only": {"eng_delta": -0.08, "temp_shift": -1},
                "delayed_reply": {"eng_delta": -0.06, "temp_shift": 0},
                "ignores": {"eng_delta": -0.15, "temp_shift": -2},
                "annoyed": {"eng_delta": -0.1, "temp_shift": 1},
                "defensive": {"eng_delta": -0.08, "temp_shift": 1},
                "escalates": {"eng_delta": -0.05, "temp_shift": 2},
            }

            effects = reaction_effects.get(reaction, {"eng_delta": 0, "temp_shift": 0})
            sim_eng = max(0.0, min(1.0, sim_eng + effects["eng_delta"]))

            # Temperature shift
            temp_idx = temp_order.index(sim_temp) if sim_temp in temp_order else 3
            temp_idx = max(0, min(len(temp_order) - 1, temp_idx + effects["temp_shift"]))
            sim_temp = temp_order[temp_idx]

            traj["rounds"].append({
                "reaction": reaction,
                "engagement": round(sim_eng, 3),
                "temperature": sim_temp,
            })

            # Check for conversation death
            if reaction == "ignores":
                traj["conversation_died"] = True
                traj["ghosted"] = True
                break
            if reaction == "escalates" and sim_temp == "boiling":
                traj["conflict_escalated"] = True

        traj["final_temp"] = sim_temp
        traj["final_engagement"] = round(sim_eng, 3)
        trajectories.append(traj)

    # Aggregate trajectory results
    avg_final_eng = sum(t["final_engagement"] for t in trajectories) / len(trajectories)
    death_rate = sum(1 for t in trajectories if t["conversation_died"]) / len(trajectories)
    ghost_rate = sum(1 for t in trajectories if t["ghosted"]) / len(trajectories)
    escalation_rate = sum(1 for t in trajectories if t["conflict_escalated"]) / len(trajectories)
    eng_delta = avg_final_eng - eng_score

    # Temperature distribution at end
    final_temps = [t["final_temp"] for t in trajectories]
    temp_dist = {t: final_temps.count(t) / len(final_temps) for t in set(final_temps)}

    # Determine overall trend
    if eng_delta > 0.05:
        trend = "warming_up"
    elif eng_delta < -0.05:
        trend = "cooling_down"
    elif escalation_rate > 0.3:
        trend = "escalating"
    elif death_rate > 0.3:
        trend = "dying"
    else:
        trend = "stable"

    return {
        "strategy": strategy_name,
        "rounds_simulated": rounds,
        "n_simulations": n_simulations,
        "trend": trend,
        "avg_final_engagement": round(avg_final_eng, 3),
        "engagement_delta": round(eng_delta, 3),
        "conversation_death_rate": round(death_rate, 3),
        "ghost_rate": round(ghost_rate, 3),
        "escalation_rate": round(escalation_rate, 3),
        "temperature_distribution": {k: round(v, 3) for k, v in temp_dist.items()},
    }


def risk_adjusted_strategy_score(
    strategy_name: str,
    mc_result: Dict[str, Any],
    trajectory: Dict[str, Any],
) -> float:
    """
    Sharpe-ratio-inspired risk-adjusted score.

    score = (expected_return - risk_free_rate) / volatility

    Where:
    - expected_return = MC expected outcome
    - risk_free_rate = 0.4 (baseline "just acknowledge" score)
    - volatility = sqrt(variance) + death_rate + ghost_rate
    """
    strat_data = mc_result.get("all_strategies", {}).get(strategy_name, {})
    expected = strat_data.get("expected_outcome", 0.5)
    variance = strat_data.get("risk_variance", 0.01)

    death_penalty = trajectory.get("conversation_death_rate", 0) * 0.3
    ghost_penalty = trajectory.get("ghost_rate", 0) * 0.5
    escalation_penalty = trajectory.get("escalation_rate", 0) * 0.2

    risk_free = 0.4
    volatility = math.sqrt(variance) + death_penalty + ghost_penalty + escalation_penalty

    if volatility < 0.01:
        volatility = 0.01

    sharpe = (expected - risk_free) / volatility
    return round(sharpe, 3)


def advanced_monte_carlo_analysis(
    situation: Dict[str, Any],
    engagement: Optional[Dict] = None,
    personality: Optional[Dict] = None,
    n_simulations: int = 50,
) -> Dict[str, Any]:
    """
    Full advanced Monte Carlo pipeline:
    1. Run standard MC simulation for each strategy
    2. Run multi-round trajectory for top 3 strategies
    3. Apply Bayesian priors from historical data
    4. Compute risk-adjusted Sharpe scores
    5. Factor in temporal dynamics (time of day, day of week)
    6. Return optimal strategy with confidence bounds

    This is the brain that picks the best possible response approach.
    """
    # Step 1: Standard MC
    mc_result = monte_carlo_simulate(situation, engagement, personality, n_simulations)

    # Step 2: Get top 3 strategies
    all_strats = mc_result.get("all_strategies", {})
    ranked = sorted(all_strats.items(), key=lambda x: x[1].get("expected_outcome", 0), reverse=True)
    top_3 = ranked[:3] if len(ranked) >= 3 else ranked

    # Step 3: Multi-round trajectories for top strategies
    trajectories = {}
    for strat_name, _ in top_3:
        traj = multi_round_trajectory_simulate(
            situation, strat_name, engagement, personality,
            n_simulations=30, rounds=3,
        )
        trajectories[strat_name] = traj

    # Step 4: Bayesian prior adjustments
    bayesian_adjustments = {}
    for strat_name, _ in top_3:
        prior = _get_bayesian_prior(strat_name)
        bayesian_adjustments[strat_name] = prior

    # Step 5: Risk-adjusted scoring
    sharpe_scores = {}
    for strat_name, _ in top_3:
        traj = trajectories.get(strat_name, {})
        sharpe = risk_adjusted_strategy_score(strat_name, mc_result, traj)
        sharpe_scores[strat_name] = sharpe

    # Step 6: Temporal dynamics
    hour = datetime.now().hour
    temporal_bonus = {}
    for strat_name, _ in top_3:
        bonus = 0.0
        # Late night favors personal/vulnerable strategies
        if 22 <= hour or hour < 2:
            if strat_name in ("open_up_real", "ride_with_them"):
                bonus = 0.08
            elif strat_name == "challenge_push":
                bonus = -0.05  # don't start fights at night
        # Morning favors upbeat
        elif 7 <= hour < 11:
            if strat_name in ("escalate_up", "playful_tease"):
                bonus = 0.05
        # Afternoon is neutral
        temporal_bonus[strat_name] = bonus

    # Step 7: Composite scoring — weighted combination
    composite_scores = {}
    for strat_name, strat_data in top_3:
        base_score = strat_data.get("expected_outcome", 0.5)
        bayesian = bayesian_adjustments.get(strat_name, 0)
        sharpe = sharpe_scores.get(strat_name, 0)
        temporal = temporal_bonus.get(strat_name, 0)
        traj = trajectories.get(strat_name, {})
        eng_delta = traj.get("engagement_delta", 0)

        composite = (
            base_score * 0.40           # MC expected outcome
            + sharpe * 0.10             # Risk-adjusted score
            + bayesian                  # Historical performance bias
            + temporal                  # Time-of-day bonus
            + eng_delta * 0.20          # Trajectory engagement impact
            + (1.0 - traj.get("conversation_death_rate", 0)) * 0.15  # Survival bonus
            + (1.0 - traj.get("ghost_rate", 0)) * 0.15              # Anti-ghost bonus
        )
        composite_scores[strat_name] = round(composite, 4)

    # Step 8: Pick winner
    best_strat = max(composite_scores, key=composite_scores.get)
    best_score = composite_scores[best_strat]

    # Confidence: how much better is #1 vs #2?
    sorted_composites = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_composites) >= 2:
        margin = sorted_composites[0][1] - sorted_composites[1][1]
        confidence = min(margin / 0.15, 1.0)  # 0.15 margin = 100% confident
    else:
        confidence = 1.0

    return {
        "recommended_strategy": best_strat,
        "composite_score": best_score,
        "confidence": round(confidence, 3),
        "strategy_description": RESPONSE_STRATEGIES.get(best_strat, {}).get("description", ""),
        "mc_base": mc_result,
        "trajectories": trajectories,
        "sharpe_scores": sharpe_scores,
        "bayesian_adjustments": bayesian_adjustments,
        "temporal_bonuses": temporal_bonus,
        "composite_scores": composite_scores,
        "all_strategies_ranked": sorted_composites,
        "total_simulations": n_simulations * len(RESPONSE_STRATEGIES) + 30 * 3 * len(top_3),
    }


# ═══════════════════════════════════════════════════════════════
#  4. CHAIN-OF-THOUGHT REASONING
# ═══════════════════════════════════════════════════════════════

def build_chain_of_thought(
    situation: Dict[str, Any],
    mc_result: Dict[str, Any],
    engagement: Optional[Dict] = None,
    conflict: Optional[Dict] = None,
    personality: Optional[Dict] = None,
    ghost: Optional[Dict] = None,
    trajectory: Optional[Dict] = None,
) -> str:
    """
    Build a structured chain-of-thought reasoning block
    to inject into the system prompt BEFORE generation.
    This is the brain that thinks before speaking.
    """
    lines = []

    # Core situation — concise, not verbose
    msg_type = situation.get("message_type", "unknown")
    temp = situation.get("emotional_temperature", "neutral")
    stakes = situation.get("stakes", "low")
    intent = situation.get("their_intent", "unknown")
    subtext = situation.get("subtext")

    # Only include what matters — skip "normal" defaults
    _situation_parts = []
    if msg_type not in ("statement", "unknown"):
        _situation_parts.append(msg_type.replace("_", " "))
    if temp not in ("neutral",):
        _situation_parts.append(f"{temp.replace('_', ' ')} energy")
    if stakes not in ("low",):
        _situation_parts.append(f"{stakes} stakes")
    if intent not in ("general_communication", "unknown"):
        _situation_parts.append(f"they want: {intent.replace('_', ' ')}")

    if _situation_parts:
        lines.append("Read: " + " | ".join(_situation_parts))

    # Subtext is the MOST important signal — highlighted clearly
    if subtext:
        lines.append(f"⚠ They might mean: {subtext.replace('_', ' ')} — respond to the real meaning")

    # Strategy — just the name, not a verbose description
    strategy = mc_result.get("recommended_strategy", "match_energy")
    lines.append(f"Approach: {strategy.replace('_', ' ')}")

    # Only flag warnings, not normal states
    if conflict and conflict.get("level", "none") in ("high", "critical"):
        lines.append(f"⚠ Conflict {conflict['level']} — be careful")
    if ghost and ghost.get("level", "none") in ("moderate", "high"):
        lines.append(f"⚠ Ghost risk {ghost['level']} — keep it interesting, don't bore them")

    # The one rule that matters most
    lines.append("→ Reply to what THEY said. Stay connected. No random topics.")

    return "\n".join(lines) if lines else ""


# ═══════════════════════════════════════════════════════════════
#  5. FULL THINKING PIPELINE
# ═══════════════════════════════════════════════════════════════

def think(
    incoming_text: str,
    messages: List[Dict[str, Any]],
    nlp_analysis: Optional[Dict] = None,
    engagement: Optional[Dict] = None,
    conflict: Optional[Dict] = None,
    personality: Optional[Dict] = None,
    ghost: Optional[Dict] = None,
    trajectory: Optional[Dict] = None,
    n_simulations: int = 50,
) -> Tuple[Dict[str, Any], str]:
    """
    Full thinking pipeline. Call this BEFORE generate_reply().

    Returns:
        (thinking_results_dict, chain_of_thought_prompt_string)
    """
    # Step 1: Assess situation
    situation = assess_situation(
        incoming_text, messages, nlp_analysis, engagement, conflict, personality,
    )

    # Step 2: Advanced Monte Carlo simulation (multi-round + Bayesian + risk-adjusted)
    advanced_mc = advanced_monte_carlo_analysis(
        situation, engagement, personality, n_simulations,
    )

    # Backward-compatible mc_result from advanced analysis
    mc_result = {
        "recommended_strategy": advanced_mc["recommended_strategy"],
        "recommended_score": advanced_mc["composite_score"],
        "strategy_description": advanced_mc["strategy_description"],
        "all_strategies": advanced_mc["mc_base"].get("all_strategies", {}),
        "simulations_run": advanced_mc["total_simulations"],
        # Advanced fields
        "confidence": advanced_mc["confidence"],
        "sharpe_scores": advanced_mc.get("sharpe_scores"),
        "trajectories": advanced_mc.get("trajectories"),
        "composite_scores": advanced_mc.get("composite_scores"),
    }

    # Step 3: Build chain of thought
    cot = build_chain_of_thought(
        situation, mc_result, engagement, conflict, personality, ghost, trajectory,
    )

    # Add trajectory insights to CoT
    best_traj = advanced_mc.get("trajectories", {}).get(advanced_mc["recommended_strategy"], {})
    if best_traj:
        traj_lines = [
            f"\n8. TRAJECTORY FORECAST ({best_traj.get('rounds_simulated', 3)}-round simulation):",
            f"   Trend: {best_traj.get('trend', 'unknown')}",
            f"   Engagement delta: {best_traj.get('engagement_delta', 0):+.3f}",
            f"   Death risk: {best_traj.get('conversation_death_rate', 0):.1%}",
            f"   Ghost risk: {best_traj.get('ghost_rate', 0):.1%}",
            f"   Confidence: {advanced_mc.get('confidence', 0):.0%}",
        ]
        cot += "\n" + "\n".join(traj_lines)

    results = {
        "situation": situation,
        "monte_carlo": mc_result,
        "advanced_mc": advanced_mc,
        "chain_of_thought_length": len(cot),
    }

    return results, cot
