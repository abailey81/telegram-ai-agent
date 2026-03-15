"""
Conversation Intelligence Engine.

The central orchestration layer that transforms a basic chatbot into
a sophisticated conversational agent. Handles:

1. Dynamic Context Assembly - Weighted message selection, not naive last-N
2. Few-Shot Personality Anchoring - Extracts exemplary past exchanges
3. Conversation State Machine - Explicit dialogue state tracking
4. Structured Summary Compression - Anchored iterative summarization
5. Conversation Goal Tracking - What are we trying to achieve?
6. Dialogue Act Planning - Decide WHAT to say before HOW
7. Proactive Topic Introduction - Reference past conversations naturally
8. Response Candidate Ranking - Score multiple angles before committing

All functions gracefully degrade if dependencies are missing.
"""

import json
import logging
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

engine_logger = logging.getLogger("conversation_engine")
engine_logger.setLevel(logging.INFO)

ENGINE_DATA_DIR = Path(__file__).parent / "engine_data"
ENGINE_DATA_DIR.mkdir(exist_ok=True)

SUMMARIES_DIR = ENGINE_DATA_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)

PROFILES_DIR = ENGINE_DATA_DIR / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)

GOALS_DIR = ENGINE_DATA_DIR / "goals"
GOALS_DIR.mkdir(exist_ok=True)


# ── Auto-pickup: load autoresearch-optimized engine parameters ──
_OPTIMIZED_ENGINE_PARAMS = None
_OPTIMIZED_ENGINE_PARAMS_MTIME = 0


def _load_optimized_engine_params() -> Optional[dict]:
    """Load optimized engine params from autoresearch (auto-pickup on file change)."""
    global _OPTIMIZED_ENGINE_PARAMS, _OPTIMIZED_ENGINE_PARAMS_MTIME
    params_file = ENGINE_DATA_DIR / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_ENGINE_PARAMS_MTIME:
            _OPTIMIZED_ENGINE_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_ENGINE_PARAMS_MTIME = mtime
            engine_logger.info(
                f"Auto-loaded optimized engine params "
                f"(score={_OPTIMIZED_ENGINE_PARAMS.get('optimization_score', '?')})"
            )
        return _OPTIMIZED_ENGINE_PARAMS
    except Exception as e:
        engine_logger.debug(f"Could not load optimized engine params: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  1. DYNAMIC CONTEXT ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def assemble_weighted_context(
    messages: List[Dict[str, str]],
    incoming_text: str,
    max_messages: int = 20,
) -> List[Dict[str, Any]]:
    """Select and weight messages by importance instead of naive last-N.

    Scoring dimensions:
    - Recency: exponential decay from most recent
    - Emotional significance: high-emotion messages score higher
    - Question/answer pairs: keep Q+A together
    - Length: longer messages are usually more important
    - Turning points: messages where sentiment shifts dramatically

    Returns ordered list of messages with weights and annotations.
    """
    if not messages:
        return []

    # Auto-pickup optimized params (recency_decay, recency_weight, max_messages)
    _opt = _load_optimized_engine_params()
    if _opt:
        max_messages = _opt.get("max_messages", max_messages)
    _recency_decay = (_opt or {}).get("recency_decay", 0.15)
    _recency_weight = (_opt or {}).get("recency_weight", 3.0)

    scored = []
    for i, msg in enumerate(messages):
        text = msg.get("text", "")
        score = 0.0
        annotations = []

        # Recency score (exponential decay — auto-tuned by autoresearch)
        position_from_end = len(messages) - i - 1
        recency = math.exp(-_recency_decay * position_from_end)
        score += recency * _recency_weight

        # Length bonus (longer messages have more content)
        word_count = len(text.split())
        if word_count > 20:
            score += 1.5
            annotations.append("detailed")
        elif word_count > 10:
            score += 0.5

        # Emotional significance
        emotional_markers = [
            "love", "miss", "hate", "angry", "sorry", "forgive",
            "hurt", "happy", "sad", "worried", "scared", "excited",
            "jealous", "trust", "afraid", "proud", "grateful",
            "❤", "😢", "😡", "🥰", "😭", "💔", "😍",
            # Russian
            "люблю", "скучаю", "ненавижу", "злюсь", "извини", "прости",
            "больно", "счастлив", "грустно", "волнуюсь", "боюсь",
            "горжусь", "благодарен", "ревную",
        ]
        emotion_count = sum(1 for m in emotional_markers if m in text.lower())
        if emotion_count >= 2:
            score += 2.0
            annotations.append("emotionally_significant")
        elif emotion_count == 1:
            score += 0.8

        # Question detection (keep questions for context)
        if "?" in text:
            score += 1.0
            annotations.append("question")

        # First-time disclosures (personal sharing)
        disclosure_markers = [
            "never told", "first time", "to be honest", "tbh",
            "confession", "secret", "real talk", "actually",
            "my family", "my mom", "my dad", "my ex",
            # Russian
            "никому не говорил", "никому не говорила", "впервые",
            "по-честному", "честно говоря", "признание", "секрет",
            "на самом деле", "моя семья", "моя мама", "мой папа",
            "мой бывший", "моя бывшая",
        ]
        if any(m in text.lower() for m in disclosure_markers):
            score += 2.5
            annotations.append("personal_disclosure")

        # Conflict markers (important to keep for resolution)
        conflict_markers = [
            "we need to talk", "i'm upset", "don't like",
            "why did you", "why didn't you", "not fair",
            "you never", "you always", "i can't believe",
            # English profanity
            "fuck", "fucking", "shit", "bitch", "asshole", "idiot", "stfu", "gtfo",
            "hate you", "piss off", "screw you", "go to hell", "moron", "loser",
            "dumbass", "dipshit", "motherfucker", "bastard", "dickhead", "drop dead",
            "bullshit", "pathetic", "worthless", "disgusting", "die",
            # Russian profanity
            "блядь", "блять", "сука", "сучка", "сучара", "пиздец", "пизда",
            "нахуй", "нахер", "гандон", "мудак", "мудила",
            "дебил", "долбоёб", "долбоеб", "тварь", "урод", "козёл", "козел",
            "ненавижу", "заткнись", "отвали", "пошёл", "пошел", "пошла",
            "вали", "проваливай", "идиот", "идиотка", "кретин", "дура", "дурак",
            "ублюдок", "выродок", "мразь", "подонок", "шлюха", "сволочь",
            "гнида", "падла", "паскуда", "хуй", "хуйня", "хуесос",
            "ёбаный", "ебаный", "ебать", "заебал", "заебала", "отъебись",
            "уёбок", "уебок", "пиздабол", "говно", "дерьмо", "скотина",
            "чмо", "придурок", "лох", "тупица", "бездарь", "мразота",
        ]
        if any(m in text.lower() for m in conflict_markers):
            score += 2.0
            annotations.append("conflict")

        # Plans and commitments
        plan_markers = [
            "let's", "we should", "want to", "this weekend",
            "tomorrow", "tonight", "next week", "meet",
            "date", "dinner", "plans", "come over",
            # Russian
            "давай", "нам стоит", "хочу", "на выходных", "завтра",
            "сегодня вечером", "на следующей неделе", "встретимся",
            "свидание", "ужин", "планы", "приезжай",
        ]
        if any(m in text.lower() for m in plan_markers):
            score += 1.5
            annotations.append("plans")

        scored.append({
            **msg,
            "_score": round(score, 3),
            "_annotations": annotations,
            "_position": i,
        })

    # Sort by score descending, take top max_messages
    scored.sort(key=lambda x: x["_score"], reverse=True)
    selected = scored[:max_messages]

    # Re-sort by original position for chronological order
    selected.sort(key=lambda x: x["_position"])

    return selected


def format_weighted_context(
    weighted_messages: List[Dict[str, Any]],
    show_annotations: bool = False,
) -> str:
    """Format weighted messages into a context block for the prompt.

    Optionally includes importance annotations as subtle markers.
    """
    lines = []
    for msg in weighted_messages:
        sender = msg.get("sender", "Unknown")
        text = msg.get("text", "")
        prefix = "Me" if sender == "Me" else "Them"

        if show_annotations and msg.get("_annotations"):
            annotations = ", ".join(msg["_annotations"])
            lines.append(f"[{annotations}] {prefix}: {text}")
        else:
            lines.append(f"{prefix}: {text}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  2. FEW-SHOT PERSONALITY ANCHORING
# ═══════════════════════════════════════════════════════════════

def extract_exemplary_exchanges(
    messages: List[Dict[str, str]],
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """Extract the best past exchanges to use as few-shot examples.

    Looks for exchanges where:
    - We said something and they responded positively
    - Our message matched their energy
    - The exchange felt natural and engaging

    Returns list of {our_msg, their_response, category} dicts.
    """
    examples = []
    if len(messages) < 4:
        return examples

    for i in range(len(messages) - 1):
        current = messages[i]
        next_msg = messages[i + 1]

        # Look for our message followed by their positive response
        if current.get("sender") == "Me" and next_msg.get("sender") == "Them":
            our_text = current.get("text", "")
            their_response = next_msg.get("text", "")

            if not our_text or not their_response:
                continue

            # Score the exchange quality
            quality = 0.0
            category = "general"

            # Positive response indicators
            positive_markers = [
                "haha", "lol", "😂", "🤣", "❤", "🥰", "😍",
                "aww", "that's sweet", "love", "cute", "omg",
                "yes!", "exactly", "same", "me too", "so true",
                "i like", "perfect", "amazing", "wow",
            ]
            positive_count = sum(
                1 for m in positive_markers if m in their_response.lower()
            )
            if positive_count >= 2:
                quality += 3.0
                category = "highly_positive"
            elif positive_count == 1:
                quality += 1.5
                category = "positive"

            # Length reciprocity (they matched our energy)
            len_ratio = len(their_response) / max(len(our_text), 1)
            if 0.5 <= len_ratio <= 2.0:
                quality += 1.0

            # Their response was longer (we engaged them)
            if len(their_response) > len(our_text) * 1.3:
                quality += 1.5
                category = "engaging"

            # Emotional exchange
            emotional = any(
                m in our_text.lower()
                for m in ["miss you", "love you", "thinking of you", "care about"]
            )
            if emotional and positive_count > 0:
                quality += 2.0
                category = "emotional"

            # Humor exchange
            if any(m in their_response.lower() for m in ["haha", "lol", "😂", "🤣"]):
                if len(our_text) > 10:
                    quality += 1.5
                    category = "humor"

            if quality >= 2.0:
                examples.append({
                    "our_msg": our_text,
                    "their_response": their_response[:100],
                    "quality": round(quality, 2),
                    "category": category,
                })

    # Sort by quality, take best
    examples.sort(key=lambda x: x["quality"], reverse=True)

    # Diversify categories
    seen_categories = set()
    diversified = []
    for ex in examples:
        if ex["category"] not in seen_categories or len(diversified) < max_examples:
            diversified.append(ex)
            seen_categories.add(ex["category"])
        if len(diversified) >= max_examples:
            break

    return diversified


def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    """Format exemplary exchanges into a prompt section."""
    if not examples:
        return ""

    lines = ["Examples of messages that landed well in this conversation:"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"  {i}. You said: \"{ex['our_msg'][:120]}\"")
        lines.append(f"     They responded: \"{ex['their_response'][:80]}\"")

    lines.append("Use these as reference for tone and style that works with this person.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  3. CONVERSATION STATE MACHINE
# ═══════════════════════════════════════════════════════════════

CONVERSATION_STATES = {
    "greeting": {
        "description": "Opening the conversation",
        "transitions": ["small_talk", "checking_in", "topic_intro"],
        "dialogue_acts": ["greet", "ask_about_day", "express_excitement"],
    },
    "small_talk": {
        "description": "Light casual conversation",
        "transitions": ["deepening", "flirting", "topic_discussion", "closing"],
        "dialogue_acts": ["share_update", "ask_question", "react", "joke"],
    },
    "checking_in": {
        "description": "Asking about their wellbeing",
        "transitions": ["supporting", "celebrating", "small_talk"],
        "dialogue_acts": ["ask_open_question", "follow_up", "validate"],
    },
    "deepening": {
        "description": "Getting into deeper conversation",
        "transitions": ["emotional_sharing", "planning", "flirting", "debating"],
        "dialogue_acts": ["share_perspective", "ask_deeper", "self_disclose", "reflect"],
    },
    "emotional_sharing": {
        "description": "Sharing or receiving emotional content",
        "transitions": ["supporting", "deepening", "reconnecting"],
        "dialogue_acts": ["react_naturally", "share_feeling", "active_listen"],
    },
    "supporting": {
        "description": "Providing emotional support",
        "transitions": ["emotional_sharing", "reconnecting", "small_talk"],
        "dialogue_acts": ["react_naturally", "active_listen", "express_care"],
    },
    "flirting": {
        "description": "Playful romantic interaction",
        "transitions": ["deepening", "planning", "small_talk"],
        "dialogue_acts": ["compliment", "tease", "be_playful", "escalate"],
    },
    "planning": {
        "description": "Making plans together",
        "transitions": ["small_talk", "flirting", "closing"],
        "dialogue_acts": ["suggest", "confirm", "express_excitement", "coordinate"],
    },
    "conflict": {
        "description": "Disagreement or fight",
        "transitions": ["reconnecting", "emotional_sharing"],
        "dialogue_acts": ["match_energy", "stand_ground", "push_back"],
    },
    "de_escalating": {
        "description": "Cooling off after a fight",
        "transitions": ["reconnecting", "small_talk"],
        "dialogue_acts": ["react_naturally", "redirect", "share_feeling"],
    },
    "reconnecting": {
        "description": "Getting back to normal after tension",
        "transitions": ["small_talk", "flirting", "deepening"],
        "dialogue_acts": ["share_feeling", "express_care", "playful_tease"],
    },
    "celebrating": {
        "description": "Sharing in good news or joy",
        "transitions": ["flirting", "planning", "small_talk"],
        "dialogue_acts": ["celebrate", "express_pride", "ask_details", "plan_celebration"],
    },
    "topic_discussion": {
        "description": "Actively discussing a specific topic (work, hobby, news, etc.)",
        "transitions": ["deepening", "debating", "small_talk", "advising"],
        "dialogue_acts": ["share_perspective", "ask_question", "react", "provide_info", "discuss"],
    },
    "debating": {
        "description": "Friendly debate or exchanging different viewpoints",
        "transitions": ["deepening", "topic_discussion", "small_talk"],
        "dialogue_acts": ["present_argument", "acknowledge_point", "counter_argument", "find_common_ground"],
    },
    "storytelling": {
        "description": "One person is telling a story or recounting events",
        "transitions": ["small_talk", "emotional_sharing", "topic_discussion"],
        "dialogue_acts": ["active_listen", "react_to_detail", "ask_clarification", "share_reaction"],
    },
    "advising": {
        "description": "One person is seeking or giving advice",
        "transitions": ["supporting", "topic_discussion", "small_talk"],
        "dialogue_acts": ["listen_to_situation", "offer_perspective", "ask_clarifying", "suggest_options"],
    },
    "venting": {
        "description": "They're venting frustration or stress",
        "transitions": ["supporting", "small_talk", "de_escalating"],
        "dialogue_acts": ["react_naturally", "match_energy", "active_listen", "share_opinion"],
    },
    "brainstorming": {
        "description": "Collaborative thinking or idea generation",
        "transitions": ["planning", "topic_discussion", "small_talk"],
        "dialogue_acts": ["suggest_idea", "build_on_idea", "evaluate", "encourage"],
    },
    "closing": {
        "description": "Wrapping up the conversation",
        "transitions": ["greeting"],
        "dialogue_acts": ["express_care", "say_goodbye", "plan_next_chat"],
    },
}


def detect_conversation_state(
    messages: List[Dict[str, str]],
    incoming_text: str,
    previous_state: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect current conversation state using multi-signal analysis.

    Returns state info with confidence and recommended dialogue acts.
    """
    text_lower = incoming_text.lower().strip()
    their_recent = [m for m in messages[-6:] if m.get("sender") == "Them"]
    all_recent_text = " ".join(m.get("text", "") for m in messages[-6:]).lower()

    state_scores: Dict[str, float] = {s: 0.0 for s in CONVERSATION_STATES}

    # Greeting signals
    greetings_prefix = ["hi ", "hey", "hello", "good morning", "good evening",
                        "what's up", "sup", "hola", "привет", "хай",
                        # Russian
                        "доброе утро", "добрый день", "добрый вечер",
                        "как дела", "салют", "здарова", "привет привет", "йо"]
    greetings_exact = {"hi", "yo", "sup"}
    words = set(text_lower.split())
    if any(text_lower.startswith(g) for g in greetings_prefix) or (words & greetings_exact):
        state_scores["greeting"] += 3.0
    if len(messages) < 3 and not any(f in text_lower for f in ["hot", "cute", "sexy", "miss"]):
        state_scores["greeting"] += 2.0

    # Small talk signals
    small_talk = ["how was your day", "what are you doing", "what's up",
                  "how's it going", "nothing much", "just chilling", "bored",
                  # Russian
                  "как прошёл день", "что делаешь", "как дела", "как оно",
                  "ничего особенного", "просто отдыхаю", "скучно",
                  "чем занят", "чем занята"]
    if any(s in text_lower for s in small_talk):
        state_scores["small_talk"] += 2.5

    # Checking in signals
    checking = ["how are you", "are you ok", "feeling better", "how's everything",
                "you alright", "everything okay", "how have you been",
                # Russian
                "как ты", "ты в порядке", "тебе лучше", "как всё",
                "всё хорошо", "как ты себя чувствуешь"]
    if any(c in text_lower for c in checking):
        state_scores["checking_in"] += 3.0

    # Deepening signals
    deep = ["what do you think about", "have you ever", "do you believe",
            "what's your opinion", "tell me about", "i've been thinking",
            "sometimes i wonder", "real talk",
            # Russian
            "что ты думаешь о", "ты когда-нибудь", "ты веришь",
            "какое твоё мнение", "расскажи мне о", "я думал о",
            "иногда я задумываюсь", "по-честному"]
    if any(d in text_lower for d in deep):
        state_scores["deepening"] += 3.0

    # Emotional sharing signals
    emotional = ["i feel", "i'm feeling", "im feeling",
                 "i'm sad", "im sad", "i'm happy", "im happy",
                 "i'm scared", "im scared", "i'm worried", "im worried",
                 "i'm stressed", "im stressed", "i'm anxious", "im anxious",
                 "breaks my heart", "makes me feel",
                 "i can't stop", "i cant stop",
                 "i'm so scared", "im so scared", "i'm so worried", "im so worried",
                 "i'm so sad", "im so sad", "im so stressed", "i'm so stressed",
                 "я чувствую", "мне грустно", "мне страшно", "я переживаю",
                 # Russian expanded
                 "мне больно", "мне одиноко", "мне тревожно",
                 "я расстроен", "я расстроена", "я в панике", "мне плохо"]
    if any(e in text_lower for e in emotional):
        state_scores["emotional_sharing"] += 3.5

    # Support-seeking signals
    support = ["i need", "help me", "don't know what to do", "i'm struggling",
               "it's hard", "can you", "i just need", "i'm lost",
               # Russian
               "мне нужна помощь", "помоги мне", "я не знаю что делать",
               "мне тяжело", "это сложно"]
    if any(s in text_lower for s in support):
        state_scores["supporting"] += 3.0

    # Flirting signals
    flirty = ["\U0001f60f", "\U0001f618", "\U0001f970", "\U0001f60d",
              "cute", "hot", "handsome", "beautiful",
              "miss you", "miss your", "wish you were here",
              "can't stop thinking", "cant stop thinking",
              "you're so", "youre so", "ur so",
              "\U0001f608", "come over", "kiss",
              "want you", "need you", "gorgeous", "sexy",
              "drive me crazy", "turn me on", "so fine",
              "making me blush", "your eyes", "your smile",
              "looking good", "look good", "attracted",
              # Russian
              "хочу тебя", "думаю о тебе", "ты так красив",
              "ты такая красивая", "приезжай ко мне", "хочу целоваться"]
    flirty_count = sum(1 for f in flirty if f in text_lower)
    if flirty_count > 0:
        state_scores["flirting"] += 3.0 + (flirty_count - 1) * 1.0  # bonus for multiple signals

    # Planning signals
    plans = ["let's", "lets", "we should", "want to go", "this weekend",
             "tomorrow", "what if we", "how about", "dinner", "movie",
             "trip", "meet up", "date", "restaurant", "reservation",
             # Russian
             "давай", "нам стоит", "на выходные", "завтра",
             "встретимся", "свидание", "ужин", "ресторан"]
    plan_count = sum(1 for p in plans if p in text_lower)
    if plan_count > 0:
        state_scores["planning"] += 3.0 + min(plan_count - 1, 3) * 0.5  # bonus for multiple

    # Conflict signals
    conflict = ["i'm upset", "im upset", "you never", "you always", "why did you",
                "that's not fair", "thats not fair", "i can't believe", "i cant believe",
                "you don't care", "you dont care",
                "we need to talk", "i'm angry", "im angry",
                "don't talk to me", "dont talk to me",
                "fuck you", "hate you", "leave me alone", "go away"]
    if any(c in text_lower for c in conflict):
        state_scores["conflict"] += 4.0

    # De-escalation signals
    deescalate = ["i'm sorry", "im sorry", "my fault", "i understand",
                  "let's not fight", "lets not fight",
                  "can we talk", "i didn't mean", "i didnt mean",
                  "forgive me", "i overreacted",
                  "make up", "lets make up", "let's make up",
                  "can we just", "i take it back",
                  "извини", "прости", "я был неправ", "давай помиримся",
                  # Russian expanded
                  "мне очень жаль", "я был неправ", "я была неправа",
                  "давай не ссориться", "забудем об этом"]
    if any(d in text_lower for d in deescalate):
        state_scores["de_escalating"] += 3.5

    # Reconnecting signals
    reconnect = ["i love you", "you mean a lot", "i appreciate you",
                 "let's move on", "lets move on", "fresh start",
                 "no hard feelings", "water under the bridge",
                 "clean slate", "start over", "я тебя люблю", "ты для меня"]
    reconnect_count = sum(1 for r in reconnect if r in text_lower)
    if reconnect_count > 0:
        state_scores["reconnecting"] += 3.0 + (reconnect_count - 1) * 1.0  # bonus for multiple

    # Celebrating signals
    celebrate = ["guess what", "i got", "great news", "i passed", "i won",
                 "promotion", "accepted", "so excited", "omg", "can't believe it",
                 # Russian
                 "угадай что", "я получил", "хорошие новости", "я прошёл",
                 "я выиграл", "повышение", "приняли"]
    if any(c in text_lower for c in celebrate):
        state_scores["celebrating"] += 3.0

    # Topic discussion signals — broad topic markers
    topic_discuss = ["what do you think about", "have you heard about", "did you see",
                     "i read that", "apparently", "i was watching", "this article",
                     "i saw this", "check this out", "interesting fact",
                     "speaking of", "on the topic of", "talking about that",
                     "ты знал что", "читал что", "видел что", "слышал про",
                     "кстати о", "по поводу"]
    # Topic keywords that signal discussion of a specific subject
    topic_keywords = ["premier league", "champions league", "nba", "nfl",
                      "stock", "crypto", "bitcoin", "market", "economy",
                      "iphone", "android", "tesla", "ai ", "machine learning",
                      "movie", "film", "series", "show", "album", "song",
                      "election", "politics", "climate", "technology",
                      "game", "gaming", "playstation", "xbox", "nintendo",
                      "футбол", "крипта", "фильм", "сериал", "технолог"]
    if any(t in text_lower for t in topic_discuss):
        state_scores["topic_discussion"] += 3.5
    if any(tk in text_lower for tk in topic_keywords):
        state_scores["topic_discussion"] += 2.5

    # Debating signals
    debate = ["i disagree", "actually no", "but what about", "on the other hand",
              "thats not true", "that's not true", "my point is", "to be fair",
              "hot take", "controversial", "unpopular opinion", "i think youre wrong",
              "nah thats not", "ur wrong", "counterpoint", "false equivalence",
              "by that logic", "thats a stretch", "respectfully ur wrong",
              "не согласен", "на самом деле", "но с другой стороны", "спорно"]
    if any(d in text_lower for d in debate):
        state_scores["debating"] += 3.5

    # Storytelling signals
    story = ["so basically", "let me tell you", "story time", "you wont believe",
             "so what happened was", "the thing is", "long story short",
             "get this", "guess what happened", "picture this", "there i was",
             "and then", "wait it gets", "chapter two", "the saga",
             "so anyway what happened", "buckle up", "grab popcorn",
             "короче", "представь себе", "знаешь что случилось", "слушай",
             "в общем история такая", "всё началось с"]
    if any(s in text_lower for s in story):
        state_scores["storytelling"] += 3.0

    # Advising signals
    advise = ["what should i", "should i", "what would you do", "advice",
              "recommend", "help me decide", "i cant decide", "thoughts on",
              "what do u think i should", "help me choose", "torn between",
              "heres what i think u should", "if i were u", "my honest advice",
              "как думаешь", "что посоветуешь", "стоит ли", "не могу решить",
              "что бы ты сделал", "помоги решить"]
    if any(a in text_lower for a in advise):
        state_scores["advising"] += 3.5

    # Venting signals — handle both apostrophe and no-apostrophe variants
    # Note: "ugh" uses word boundary check to avoid matching inside "thought"
    import re as _re
    vent_phrases = ["cant deal", "can't deal", "so annoyed", "i swear", "fml",
                    "need to rant", "cant believe", "can't believe", "pisses me off",
                    "hate my", "im so done", "i'm so done", "over it", "sick of this",
                    "so fed up", "so frustrated", "driving me crazy", "driving me insane",
                    "killing me", "everything sucks", "worst day", "burning out",
                    "burned out", "burnt out", "cant take it", "at my limit",
                    "breaking point", "done with everything", "so tired of",
                    "need to vent", "just venting", "rant incoming", "let me vent",
                    "бесит", "достало", "задолбало", "ненавижу", "не могу больше",
                    "устал от всего", "всё достало", "выгорел", "на пределе"]
    vent_words = ["ugh", "ughhh", "ugggh"]  # standalone words only
    vent_match = any(v in text_lower for v in vent_phrases)
    if not vent_match:
        words = set(text_lower.split())
        vent_match = bool(words & set(vent_words))
    if vent_match:
        state_scores["venting"] += 3.5

    # Brainstorming signals — differentiate from planning
    brainstorm = ["what if we tried", "hear me out", "brainstorm",
                  "think about this", "from scratch", "wild idea",
                  "no bad ideas", "building on", "flip it",
                  "lateral thinking", "blue sky", "reverse engineer",
                  "decompose the problem", "lets think", "lets figure",
                  "how would we", "what approach", "combine both",
                  "hybrid approach", "mvp", "prototype", "a b test",
                  "trade off", "feasibility", "highest impact",
                  "а что если попробовать", "мозговой штурм", "давай подумаем",
                  "без ограничений", "перевернём задачу", "есть идея"]
    if any(b in text_lower for b in brainstorm):
        state_scores["brainstorming"] += 3.5

    # Closing signals
    closing = ["goodnight", "good night", "gotta go", "talk later",
               "bye", "see you", "ttyl", "sleep", "heading out",
               "спокойной ночи", "пока",
               # Russian expanded
               "до свидания", "пока-пока", "до завтра",
               "поговорим позже", "удачи"]
    if any(c in text_lower for c in closing):
        state_scores["closing"] += 3.5

    # State continuity bonus (prefer staying in same state if signals are close)
    if previous_state and previous_state in state_scores:
        state_scores[previous_state] += 1.0

    # Get best state
    best_state = max(state_scores, key=state_scores.get)
    best_score = state_scores[best_state]
    total_score = sum(state_scores.values()) or 1.0
    confidence = best_score / total_score

    # If confidence is very low, default to small_talk (threshold auto-tuned)
    _opt_conf = (_load_optimized_engine_params() or {}).get("state_confidence_threshold", 0.15)
    if confidence < _opt_conf:
        best_state = "small_talk"
        confidence = 0.3

    state_info = CONVERSATION_STATES[best_state]

    return {
        "state": best_state,
        "confidence": round(confidence, 3),
        "description": state_info["description"],
        "recommended_acts": state_info["dialogue_acts"],
        "valid_transitions": state_info["transitions"],
        "all_scores": {k: round(v, 2) for k, v in state_scores.items() if v > 0},
    }


# ═══════════════════════════════════════════════════════════════
#  4. STRUCTURED SUMMARY COMPRESSION
# ═══════════════════════════════════════════════════════════════

def load_conversation_summary(chat_id: int) -> Dict[str, Any]:
    """Load the persistent conversation summary for a chat."""
    path = SUMMARIES_DIR / f"{chat_id}_summary.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "relationship_narrative": "",
        "key_events": [],
        "shared_memories": [],
        "recurring_topics": [],
        "their_preferences": {},
        "communication_patterns": {},
        "emotional_baseline": "neutral",
        "last_conversation_summary": "",
        "total_messages_processed": 0,
        "last_updated": None,
    }


def save_conversation_summary(chat_id: int, summary: Dict[str, Any]):
    """Save conversation summary to disk."""
    summary["last_updated"] = datetime.now().isoformat()
    path = SUMMARIES_DIR / f"{chat_id}_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def update_summary_from_conversation(
    chat_id: int,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Update the persistent summary with new conversation data.

    Uses anchored iterative summarization - only processes new messages
    since last update, merges with existing summary.
    """
    summary = load_conversation_summary(chat_id)
    processed = summary.get("total_messages_processed", 0)

    # Only process new messages
    new_messages = messages[processed:]
    if not new_messages:
        return summary

    # Extract key events from new messages
    for msg in new_messages:
        text = msg.get("text", "")
        sender = msg.get("sender", "Unknown")

        # Detect key events
        event = _detect_key_event(text, sender)
        if event:
            summary["key_events"].append(event)
            # Keep only last 30 events
            summary["key_events"] = summary["key_events"][-30:]

        # Track recurring topics
        topics = _extract_topics(text)
        for topic in topics:
            if topic not in summary["recurring_topics"]:
                summary["recurring_topics"].append(topic)
        summary["recurring_topics"] = summary["recurring_topics"][-20:]

        # Track their preferences
        if sender == "Them":
            prefs = _extract_preferences(text)
            summary["their_preferences"].update(prefs)

    # Update last conversation summary
    recent_text = " | ".join(
        f"{m.get('sender', '?')}: {m.get('text', '')[:60]}"
        for m in messages[-10:]
    )
    summary["last_conversation_summary"] = recent_text[:500]
    summary["total_messages_processed"] = len(messages)

    save_conversation_summary(chat_id, summary)
    return summary


def _detect_key_event(text: str, sender: str) -> Optional[Dict[str, Any]]:
    """Detect if a message contains a key relationship event."""
    text_lower = text.lower()
    events = {
        "first_i_love_you": ["i love you", "я тебя люблю", "люблю тебя"],
        "conflict_start": ["we need to talk", "i'm upset with you", "don't talk to me"],
        "conflict_resolution": ["i forgive you", "let's move on", "i'm sorry"],
        "future_plans": ["move in", "travel together", "meet my family", "get married"],
        "personal_milestone": ["got the job", "graduated", "promotion", "accepted",
                               "passed the exam", "got accepted", "finished my thesis"],
        "career_change": ["new job", "quit my job", "starting a business", "got fired", "laid off"],
        "vulnerability": ["never told anyone", "my biggest fear", "i'm scared of", "nobody knows"],
        "gift_or_surprise": ["got something for you", "surprise", "present for you"],
        "health_event": ["going to hospital", "doctor said", "diagnosis", "surgery", "test results"],
        "travel_event": ["going to", "booked flights", "trip to", "traveling to"],
        "learning_achievement": ["learned how to", "finally figured out", "completed the course"],
        "shared_experience": ["remember when we", "that time we", "our first"],
    }

    for event_type, markers in events.items():
        if any(m in text_lower for m in markers):
            return {
                "type": event_type,
                "sender": sender,
                "text_preview": text[:80],
                "timestamp": datetime.now().isoformat(),
            }
    return None


def _extract_topics(text: str) -> List[str]:
    """Extract conversation topics from text."""
    text_lower = text.lower()
    topic_markers = {
        "work": ["work", "job", "boss", "office", "meeting", "project", "colleague", "deadline",
                 "работа", "начальник", "офис", "проект"],
        "career": ["promotion", "interview", "salary", "fired", "quit", "career",
                   "повышение", "собеседование", "зарплата", "карьера"],
        "family": ["mom", "dad", "parent", "sister", "brother", "family", "grandma",
                   "мама", "папа", "родители", "семья", "бабушка"],
        "friends": ["friend", "bestie", "hangout", "party", "друг", "подруга", "тусовка"],
        "health": ["doctor", "sick", "gym", "workout", "sleep", "tired", "pain",
                   "врач", "болеть", "тренировка", "здоровье"],
        "mental_health": ["anxiety", "depression", "therapy", "overwhelmed", "panic",
                          "тревога", "депрессия", "терапия"],
        "food": ["dinner", "lunch", "cook", "restaurant", "eat", "hungry", "recipe",
                 "ужин", "готовить", "ресторан", "еда"],
        "travel": ["trip", "travel", "vacation", "flight", "hotel", "country",
                   "путешествие", "отпуск", "рейс", "страна"],
        "entertainment": ["movie", "show", "netflix", "series", "фильм", "сериал"],
        "music": ["song", "music", "album", "concert", "band", "песня", "музыка", "концерт"],
        "books": ["book", "read", "novel", "author", "книга", "читать"],
        "gaming": ["game", "play", "stream", "console", "level", "игра", "играть"],
        "technology": ["code", "programming", "app", "phone", "computer", "ai",
                       "код", "программирование", "приложение"],
        "sports": ["football", "soccer", "basketball", "match", "team",
                   "футбол", "баскетбол", "матч", "команда"],
        "fitness": ["gym", "workout", "exercise", "run", "yoga", "diet",
                    "тренировка", "зал", "бег", "йога"],
        "finance": ["money", "pay", "rent", "budget", "crypto", "invest",
                    "деньги", "зарплата", "бюджет", "крипто"],
        "education": ["school", "university", "exam", "study", "class",
                      "школа", "университет", "экзамен", "учиться"],
        "news": ["news", "election", "war", "government", "politics",
                 "новости", "выборы", "политика"],
        "philosophy": ["meaning", "life", "purpose", "existence", "believe",
                       "смысл", "жизнь", "цель", "вселенная"],
        "pets": ["cat", "dog", "pet", "puppy", "kitten", "кот", "собака", "питомец"],
        "weather": ["weather", "rain", "snow", "cold", "hot", "погода", "дождь"],
        "relationship": ["us", "together", "love", "future", "relationship",
                         "вместе", "любовь", "отношения"],
        "plans": ["plan", "weekend", "trip", "dinner", "meet", "завтра", "планы"],
        "conflict": ["argue", "fight", "upset", "angry", "sorry", "ссора"],
    }

    found = []
    for topic, markers in topic_markers.items():
        if any(m in text_lower for m in markers):
            found.append(topic)
    return found


def _extract_preferences(text: str) -> Dict[str, str]:
    """Extract stated preferences from their messages."""
    prefs = {}
    text_lower = text.lower()

    # "I love X" / "I hate X" patterns
    love_match = re.findall(r"i (?:love|really like|adore)\s+(.+?)(?:\.|!|,|$)", text_lower)
    for match in love_match[:2]:
        prefs[f"loves_{match.strip()[:30]}"] = match.strip()[:30]

    hate_match = re.findall(r"i (?:hate|can't stand|despise)\s+(.+?)(?:\.|!|,|$)", text_lower)
    for match in hate_match[:2]:
        prefs[f"dislikes_{match.strip()[:30]}"] = match.strip()[:30]

    # "My favorite X is Y" patterns
    fav_match = re.findall(
        r"my (?:fav(?:orite|ourite)?)\s+(\w+)\s+is\s+(.+?)(?:\.|!|,|$)", text_lower
    )
    for item_type, item_value in fav_match[:2]:
        prefs[f"favorite_{item_type}"] = item_value.strip()[:30]

    return prefs


def format_summary_for_prompt(summary: Dict[str, Any]) -> str:
    """Format the conversation summary for injection into the prompt."""
    parts = []

    if summary.get("relationship_narrative"):
        parts.append(f"Relationship context: {summary['relationship_narrative']}")

    if summary.get("key_events"):
        recent_events = summary["key_events"][-5:]
        events_str = "; ".join(
            f"{e['type'].replace('_', ' ')}: \"{e['text_preview'][:40]}\""
            for e in recent_events
        )
        parts.append(f"Key moments: {events_str}")

    if summary.get("their_preferences"):
        prefs = summary["their_preferences"]
        if prefs:
            prefs_str = ", ".join(f"{k}: {v}" for k, v in list(prefs.items())[:8])
            parts.append(f"Their preferences: {prefs_str}")

    if summary.get("recurring_topics"):
        topics = summary["recurring_topics"][-8:]
        parts.append(f"Topics they care about: {', '.join(topics)}")

    if summary.get("last_conversation_summary"):
        parts.append(f"Last conversation: {summary['last_conversation_summary'][:200]}")

    return "\n".join(f"- {p}" for p in parts) if parts else ""


# ═══════════════════════════════════════════════════════════════
#  5. CONVERSATION GOAL TRACKING
# ═══════════════════════════════════════════════════════════════

def load_conversation_goals(chat_id: int) -> Dict[str, Any]:
    """Load active conversation goals."""
    path = GOALS_DIR / f"{chat_id}_goals.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "active_goals": [],
        "completed_goals": [],
        "pending_followups": [],
    }


def save_conversation_goals(chat_id: int, goals: Dict[str, Any]):
    """Save conversation goals."""
    path = GOALS_DIR / f"{chat_id}_goals.json"
    path.write_text(json.dumps(goals, indent=2, ensure_ascii=False))


def generate_session_goals(
    chat_id: int,
    summary: Dict[str, Any],
    messages: List[Dict[str, str]],
    time_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate conversation goals for the current session.

    Based on: conversation history, pending follow-ups, time of day,
    emotional patterns, and relationship stage.
    """
    goals_data = load_conversation_goals(chat_id)
    new_goals = []

    # Check for pending follow-ups
    for followup in goals_data.get("pending_followups", []):
        new_goals.append({
            "type": "followup",
            "description": followup["topic"],
            "priority": "high",
            "reason": f"They mentioned this: {followup.get('context', '')}",
        })

    # Time-based goals
    if time_context:
        period = time_context.get("period", "")
        if period in ("morning", "early_morning"):
            new_goals.append({
                "type": "check_in",
                "description": "Morning check-in - ask about their day ahead",
                "priority": "medium",
                "reason": "Good time for a warm morning message",
            })
        elif period in ("evening", "night"):
            new_goals.append({
                "type": "reflection",
                "description": "Ask about their day / share something meaningful",
                "priority": "medium",
                "reason": "Evening is good for deeper conversation",
            })

    # Event-based goals from summary
    key_events = summary.get("key_events", [])
    for event in key_events[-3:]:
        if event["type"] == "personal_milestone":
            new_goals.append({
                "type": "celebrate",
                "description": f"Follow up on: {event['text_preview'][:50]}",
                "priority": "high",
                "reason": "Important to celebrate their achievements",
            })
        elif event["type"] == "vulnerability":
            new_goals.append({
                "type": "support",
                "description": f"Check in on: {event['text_preview'][:50]}",
                "priority": "high",
                "reason": "They opened up, follow up with care",
            })

    # Relationship maintenance goals
    recent_theirs = [m for m in messages[-20:] if m.get("sender") == "Them"]
    if len(recent_theirs) > len(messages[-20:]) - len(recent_theirs):
        new_goals.append({
            "type": "reciprocate",
            "description": "Share more about yourself - they've been doing most of the sharing",
            "priority": "medium",
            "reason": "Balance the conversation energy",
        })

    # Update active goals
    goals_data["active_goals"] = new_goals[:5]
    save_conversation_goals(chat_id, goals_data)

    return new_goals[:5]


def add_followup(chat_id: int, topic: str, context: str = ""):
    """Add a pending follow-up for next session."""
    goals_data = load_conversation_goals(chat_id)
    goals_data["pending_followups"].append({
        "topic": topic,
        "context": context,
        "added": datetime.now().isoformat(),
    })
    # Keep only recent followups
    goals_data["pending_followups"] = goals_data["pending_followups"][-10:]
    save_conversation_goals(chat_id, goals_data)


def format_goals_for_prompt(goals: List[Dict[str, Any]]) -> str:
    """Format goals into a prompt section."""
    if not goals:
        return ""

    lines = ["Conversation goals for this session:"]
    for i, goal in enumerate(goals, 1):
        lines.append(
            f"  {i}. [{goal['priority'].upper()}] {goal['description']}"
        )
        if goal.get("reason"):
            lines.append(f"     Why: {goal['reason']}")

    lines.append("Try to naturally address these goals during the conversation.")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  6. DIALOGUE ACT PLANNING
# ═══════════════════════════════════════════════════════════════

DIALOGUE_ACTS = {
    "match_energy": {
        "description": "Mirror their emotional energy exactly — aggressive gets aggressive back",
        "examples": ["bro wtf", "nah thats not ok", "dont even start"],
        "when": "They're angry, frustrated, or hostile",
    },
    "stand_ground": {
        "description": "Hold your position firmly, push back, don't back down",
        "examples": ["thats not what happened", "nah im not apologizing for that", "u know thats not true"],
        "when": "They're attacking you or being unfair in a conflict",
    },
    "react_naturally": {
        "description": "React like a real person would — raw, unfiltered, genuine",
        "examples": ["damn thats rough", "wait what", "bro no way"],
        "when": "They express emotions — react, don't validate like a therapist",
    },
    "ask_open_question": {
        "description": "Ask a question to keep the convo going",
        "examples": ["wait what happened", "and then what", "wdym"],
        "when": "Conversation needs more info or they shared something briefly",
    },
    "share_feeling": {
        "description": "Express your own feelings authentically",
        "examples": ["that makes me happy ngl", "honestly thats amazing"],
        "when": "Building connection or reciprocating energy",
    },
    "playful_tease": {
        "description": "Tease, be sarcastic, roast them a little",
        "examples": ["oh u think ur funny huh", "says the person who...", "lmao ok sure"],
        "when": "Conversation is light and they're in a good mood",
    },
    "genuine_compliment": {
        "description": "Give a specific, earned compliment",
        "examples": ["u go so hard at this stuff", "thats actually impressive"],
        "when": "They shared an achievement or showed a quality you admire",
    },
    "callback_reference": {
        "description": "Reference a shared memory or inside joke",
        "examples": ["remember when we...", "like that time u..."],
        "when": "Building connection through shared history",
    },
    "redirect": {
        "description": "Change the topic if its going nowhere",
        "examples": ["anyway", "oh that reminds me", "speaking of which"],
        "when": "Conversation is stalling or boring",
    },
    "express_care": {
        "description": "Show you care but casually, not like a therapist",
        "examples": ["go eat smth", "dont stay up too late", "take care of urself idiot"],
        "when": "Late at night or they're stressed",
    },
    "share_opinion": {
        "description": "Share your genuine perspective — disagree if you disagree",
        "examples": ["honestly i think...", "nah i disagree", "imo thats wrong"],
        "when": "They're discussing something and you have thoughts",
    },
    "active_listen": {
        "description": "Show you're paying attention to their story",
        "examples": ["wait no way", "and then what happened??", "are u serious rn"],
        "when": "They're telling a story",
    },
    "push_back": {
        "description": "Disagree, challenge them, don't just agree with everything",
        "examples": ["ok fair but...", "nah i dont agree", "thats cap"],
        "when": "Debate or you genuinely disagree",
    },
    "celebrate": {
        "description": "Hype them up for good news",
        "examples": ["LETS GOO", "no way!! thats insane", "u deserve this fr"],
        "when": "They shared good news or an accomplishment",
    },
}


def select_dialogue_acts(
    state: Dict[str, Any],
    incoming_text: str,
    emotional_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Select appropriate dialogue acts for the current context.

    Returns ordered list of dialogue act names to guide response generation.
    """
    recommended = state.get("recommended_acts", [])
    text_lower = incoming_text.lower()
    acts = []

    # Emotion-first: match energy for negative emotions
    negative_emotions = ["sadness", "fear", "disgust"]
    aggressive_emotions = ["anger", "frustration"]
    if emotional_state:
        primary = emotional_state.get("primary_emotion", "neutral")
        intensity = emotional_state.get("emotional_intensity", 0.0)
        if primary in aggressive_emotions and intensity > 0.5:
            acts.append("match_energy")  # match aggression, don't validate
        elif primary in negative_emotions and intensity > 0.5:
            acts.append("react_naturally")

    # Question detection - they asked something, answer first
    if "?" in text_lower:
        acts.append("respond_to_question")

    # State-based acts
    state_name = state.get("state", "small_talk")
    if state_name == "emotional_sharing":
        acts.append("react_naturally")
        acts.append("ask_open_question")
    elif state_name == "flirting":
        acts.append("playful_tease")
        acts.append("genuine_compliment")
    elif state_name == "celebrating":
        acts.append("celebrate")
        acts.append("share_feeling")
    elif state_name == "conflict":
        acts.append("match_energy")
        acts.append("stand_ground")
    elif state_name == "deepening":
        acts.append("share_feeling")
        acts.append("ask_open_question")
    elif state_name in ("greeting", "checking_in"):
        acts.append("ask_open_question")
    elif state_name == "closing":
        acts.append("express_care")
    elif state_name == "topic_discussion":
        acts.append("share_opinion")
        acts.append("ask_open_question")
    elif state_name == "debating":
        acts.append("acknowledge_point")
        acts.append("share_opinion")
    elif state_name == "storytelling":
        acts.append("active_listen")
        acts.append("ask_open_question")
    elif state_name == "advising":
        acts.append("offer_perspective")
        acts.append("ask_open_question")
    elif state_name == "venting":
        acts.append("empathize")
        if "validate_emotion" not in acts:
            acts.append("validate_emotion")
    elif state_name == "brainstorming":
        acts.append("build_on_idea")
        acts.append("ask_open_question")

    # Add recommended acts from state machine
    for act in recommended:
        if act not in acts:
            acts.append(act)

    # Deduplicate preserving order
    seen = set()
    unique_acts = []
    for act in acts:
        if act not in seen:
            unique_acts.append(act)
            seen.add(act)

    return unique_acts[:4]


def format_dialogue_acts_for_prompt(acts: List[str]) -> str:
    """Format dialogue act guidance for the prompt."""
    if not acts:
        return ""

    lines = ["Response strategy (do these in order):"]
    for i, act_name in enumerate(acts, 1):
        act_info = DIALOGUE_ACTS.get(act_name)
        if act_info:
            lines.append(f"  {i}. {act_info['description']}")
        else:
            lines.append(f"  {i}. {act_name.replace('_', ' ').title()}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  7. MASTER CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_sophisticated_context(
    chat_id: int,
    messages: List[Dict[str, str]],
    incoming_text: str,
    nlp_analysis: Optional[Dict[str, Any]] = None,
    time_context: Optional[Dict[str, Any]] = None,
    emotional_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the complete sophisticated context for response generation.

    This is the master function that orchestrates all intelligence engines.
    Returns a rich context dict with all sections for prompt construction.
    """
    # 1. Weighted context selection
    weighted = assemble_weighted_context(messages, incoming_text)
    context_block = format_weighted_context(weighted)

    # 2. Few-shot examples
    examples = extract_exemplary_exchanges(messages)
    examples_block = format_few_shot_examples(examples)

    # 3. Conversation state detection
    state = detect_conversation_state(messages, incoming_text)

    # 4. Load/update persistent summary
    summary = update_summary_from_conversation(chat_id, messages)
    summary_block = format_summary_for_prompt(summary)

    # 5. Session goals
    goals = generate_session_goals(chat_id, summary, messages, time_context)
    goals_block = format_goals_for_prompt(goals)

    # 6. Dialogue act planning
    dialogue_acts = select_dialogue_acts(state, incoming_text, emotional_state)
    acts_block = format_dialogue_acts_for_prompt(dialogue_acts)

    # 7. Detect pending follow-ups from their messages
    _detect_followup_opportunities(chat_id, messages)

    return {
        "context_block": context_block,
        "examples_block": examples_block,
        "state": state,
        "summary_block": summary_block,
        "goals_block": goals_block,
        "dialogue_acts_block": acts_block,
        "dialogue_acts": dialogue_acts,
        "weighted_message_count": len(weighted),
        "conversation_state": state["state"],
        "state_confidence": state["confidence"],
        "active_goals": goals,
        "summary": summary,
    }


def _detect_followup_opportunities(
    chat_id: int,
    messages: List[Dict[str, str]],
):
    """Detect things worth following up on from their messages."""
    their_recent = [m for m in messages[-10:] if m.get("sender") == "Them"]

    for msg in their_recent:
        text = msg.get("text", "").lower()

        # Detect future events they mentioned
        future_markers = [
            ("interview", "job interview"),
            ("exam", "exam"),
            ("doctor", "doctor's appointment"),
            ("meeting", "important meeting"),
            ("presentation", "presentation"),
            ("flight", "travel"),
            ("birthday", "birthday"),
        ]
        for marker, topic in future_markers:
            future_words = ["tomorrow", "next week", "on monday", "on tuesday",
                            "on wednesday", "on thursday", "on friday",
                            "this weekend", "next month"]
            if marker in text and any(fw in text for fw in future_words):
                add_followup(chat_id, f"Ask about their {topic}", text[:80])
                break


def format_full_prompt_context(context: Dict[str, Any]) -> str:
    """Format all context sections into a single prompt injection block.

    This combines all intelligence engine outputs into one coherent
    section to add to the system prompt.
    """
    sections = []

    # Conversation state
    state = context.get("state", {})
    if state:
        sections.append(
            f"[Conversation Phase: {state.get('description', 'Unknown')} "
            f"(confidence: {state.get('confidence', 0):.0%})]"
        )

    # Summary (relationship context)
    if context.get("summary_block"):
        sections.append(f"\n[Relationship Context]\n{context['summary_block']}")

    # Goals
    if context.get("goals_block"):
        sections.append(f"\n{context['goals_block']}")

    # Dialogue acts (response strategy)
    if context.get("dialogue_acts_block"):
        sections.append(f"\n{context['dialogue_acts_block']}")

    # Few-shot examples
    if context.get("examples_block"):
        sections.append(f"\n{context['examples_block']}")

    return "\n".join(sections) if sections else ""


# ═══════════════════════════════════════════════════════════════
#  ENHANCED: Psychological Dataset Integration
# ═══════════════════════════════════════════════════════════════

def _safe_import_psych_conv():
    """Safely import psychological datasets for conversation engine."""
    try:
        from psychological_datasets import (
            detect_knapp_stage,
            detect_emotional_bids,
            classify_bid_response,
            ESCONV_STAGES,
            EMPATHETIC_RESPONSE_INTENTS,
            select_empathetic_response_strategy,
            KNAPP_STAGES,
        )
        return {
            "detect_knapp_stage": detect_knapp_stage,
            "detect_emotional_bids": detect_emotional_bids,
            "classify_bid_response": classify_bid_response,
            "ESCONV_STAGES": ESCONV_STAGES,
            "EMPATHETIC_RESPONSE_INTENTS": EMPATHETIC_RESPONSE_INTENTS,
            "select_empathetic_response_strategy": select_empathetic_response_strategy,
            "KNAPP_STAGES": KNAPP_STAGES,
        }
    except ImportError:
        return {}


def detect_relationship_stage(messages: List[Dict[str, str]], duration_days: int = 0) -> Dict[str, Any]:
    """Detect Knapp's relationship development stage from conversation history."""
    psych = _safe_import_psych_conv()
    if not psych:
        return {"stage": "unknown", "available": False}
    return psych["detect_knapp_stage"](messages, duration_days)


def analyze_emotional_bid_patterns(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze patterns of emotional bids and responses across the conversation."""
    psych = _safe_import_psych_conv()
    if not psych:
        return {"available": False}

    bids_made = {"them": [], "me": []}
    bid_responses = {"turning_toward": 0, "turning_away": 0, "turning_against": 0}

    for i, msg in enumerate(messages):
        text = msg.get("text", "")
        sender = msg.get("sender", "")
        bids = psych["detect_emotional_bids"](text)

        if bids:
            bids_made[sender if sender in bids_made else "them"].extend(bids)
            # Check how the next message responds to the bid
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get("sender") != sender:
                    response_type = psych["classify_bid_response"](
                        next_msg.get("text", ""), text
                    )
                    bid_responses[response_type] += 1

    total_responses = sum(bid_responses.values())
    turning_toward_rate = bid_responses["turning_toward"] / max(total_responses, 1)

    return {
        "bids_by_them": len(bids_made.get("them", [])),
        "bids_by_me": len(bids_made.get("me", [])),
        "response_breakdown": bid_responses,
        "turning_toward_rate": round(turning_toward_rate, 2),
        "healthy_threshold": 0.86,  # Gottman's finding for stable couples
        "assessment": (
            "Excellent" if turning_toward_rate >= 0.86
            else "Good" if turning_toward_rate >= 0.70
            else "Needs attention" if turning_toward_rate >= 0.50
            else "At risk"
        ),
    }


def get_esconv_stage_guidance(
    emotional_state: str = "neutral",
    conflict_active: bool = False,
) -> Dict[str, Any]:
    """Get ESConv three-stage empathetic conversation guidance."""
    psych = _safe_import_psych_conv()
    if not psych:
        return {"stage": "exploration", "primary_intents": ["questioning"]}

    # Determine which ESConv stage we're in
    if conflict_active or emotional_state in ("anger", "sadness", "fear", "grief"):
        stage = "exploration"
    elif emotional_state in ("anxiety", "confusion", "frustration"):
        stage = "comforting"
    else:
        stage = "action"

    stage_data = psych["ESCONV_STAGES"].get(stage, psych["ESCONV_STAGES"]["exploration"])
    strategy = psych["select_empathetic_response_strategy"](emotional_state, conflict_active=conflict_active)

    return {
        "esconv_stage": stage,
        "stage_goal": stage_data["goal"],
        "primary_intents": stage_data["primary_intents"],
        "strategy": strategy,
    }


def build_enhanced_context(
    messages: List[Dict[str, str]],
    chat_id: int,
    relationship_duration_days: int = 0,
) -> Dict[str, Any]:
    """Build enhanced context combining original engine with psychological datasets."""
    # Original sophisticated context
    last_text = messages[-1].get("text", "") if messages else ""
    base_context = build_sophisticated_context(chat_id, messages, last_text)

    # Enhanced with psychological datasets
    relationship_stage = detect_relationship_stage(messages, relationship_duration_days)
    bid_patterns = analyze_emotional_bid_patterns(messages)

    # Get last message emotion for ESConv
    last_text = messages[-1].get("text", "") if messages else ""
    esconv = get_esconv_stage_guidance(
        emotional_state="neutral",
        conflict_active=base_context.get("state", {}).get("state") == "conflict",
    )

    base_context["relationship_stage"] = relationship_stage
    base_context["bid_patterns"] = bid_patterns
    base_context["esconv_guidance"] = esconv
    base_context["context_version"] = "v5_enhanced"

    return base_context


def format_enhanced_context_for_prompt(context: Dict[str, Any]) -> str:
    """Format enhanced context including psychological datasets for prompt injection."""
    parts = []

    # Original context
    base = format_full_prompt_context(context)
    if base:
        parts.append(base)

    # Relationship stage (Knapp's)
    stage = context.get("relationship_stage", {})
    if stage.get("stage") and stage.get("stage") != "unknown":
        warning = ""
        if stage.get("warning_level", 0) > 0:
            warning = f" [WARNING LEVEL: {stage['warning_level']:.0%}]"
        parts.append(
            f"\n[Relationship Stage: {stage['stage'].replace('_', ' ').title()} "
            f"({stage.get('phase', '').replace('_', ' ')}) — {stage.get('description', '')}]{warning}"
        )

    # Emotional bid patterns
    bids = context.get("bid_patterns", {})
    if bids.get("turning_toward_rate") is not None:
        parts.append(
            f"\n[Bid Response Rate: {bids['turning_toward_rate']:.0%} turning toward "
            f"({bids.get('assessment', 'Unknown')}). Target: 86%]"
        )

    # ESConv guidance removed — was injecting therapy frameworks

    return "\n".join(parts) if parts else ""
