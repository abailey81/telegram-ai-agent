"""
Advanced NLP Context Engine for Telegram Auto-Reply.

Provides:
- Sentiment analysis (positive/negative/neutral + intensity)
- Conversation stage detection (new, warming up, deep, conflict, makeup)
- Topic detection (casual, romantic, emotional, logistics, etc.)
- Language detection (English/Russian/mixed)
- Response strategy recommendation
- Conversation memory per chat (learns patterns, preferences, recurring topics)
"""

import re
import json
import os
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

nlp_logger = logging.getLogger("nlp_engine")
nlp_logger.setLevel(logging.INFO)

# ============= MEMORY STORAGE =============

MEMORY_DIR = Path(__file__).parent / ".chat_memory"
MEMORY_DIR.mkdir(exist_ok=True)


def _memory_path(chat_id: int) -> Path:
    return MEMORY_DIR / f"{chat_id}.json"


def load_memory(chat_id: int) -> Dict[str, Any]:
    """Load conversation memory for a chat."""
    path = _memory_path(chat_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "chat_id": chat_id,
        "created": datetime.now().isoformat(),
        "total_messages_seen": 0,
        "their_topics": [],  # recurring topics they bring up
        "their_interests": [],  # things they like
        "their_dislikes": [],  # things that upset them
        "their_language_preference": None,  # detected primary language
        "our_successful_tones": [],  # tones that got positive responses
        "pet_names_used": [],  # pet names we've used
        "pet_names_they_use": [],  # pet names they use for us
        "relationship_stage": "unknown",  # new, dating, committed, long_term
        "last_conflict": None,  # timestamp of last detected conflict
        "conversation_patterns": {
            "avg_their_message_length": 0,
            "avg_our_message_length": 0,
            "they_use_emojis": False,
            "they_use_caps": False,
            "common_greetings": [],
        },
        "notes": [],  # freeform notes
    }


def save_memory(chat_id: int, memory: Dict[str, Any]):
    """Save conversation memory for a chat."""
    path = _memory_path(chat_id)
    try:
        path.write_text(json.dumps(memory, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        nlp_logger.error(f"Failed to save memory for {chat_id}: {e}")


# ============= SENTIMENT ANALYSIS =============

# Keyword-based sentiment (fast, no external deps)
POSITIVE_MARKERS = {
    # English - emotions
    "love", "miss", "amazing", "beautiful", "wonderful", "perfect", "happy",
    "cute", "sweet", "best", "great", "awesome", "incredible", "adore",
    "gorgeous", "excited", "proud", "thankful", "grateful", "lucky",
    "fantastic", "brilliant", "excellent", "outstanding", "magnificent",
    "thrilled", "delighted", "blessed", "impressed", "inspired",
    "hilarious", "genius", "legendary", "epic",
    # English - general positive
    "nice", "cool", "dope", "fire", "sick", "lit", "goated", "valid",
    "based", "clutch", "W", "massive", "insane", "wild", "solid",
    "congrats", "congratulations", "well done", "good job", "nailed it",
    "❤️", "😍", "🥰", "💕", "💖", "😘", "💗", "💓", "🤗", "😊",
    "💋", "❤", "♥️", "💞", "💝", "🥺", "✨", "🌹", "🔥", "💯",
    "haha", "lol", "lmao", "😂", "🤣", "😄", "😁", "🙌", "👏",
    # Russian — emotions & compliments
    "люблю", "скучаю", "красивый", "красивая", "милый", "милая",
    "замечательно", "прекрасно", "отлично", "супер", "класс",
    "счастлив", "счастлива", "рада", "рад", "обожаю", "нравится",
    "солнышко", "зайка", "котик", "малыш", "родной", "родная",
    "круто", "огонь", "топ", "красавчик", "молодец", "шикарно",
    "обалдеть", "восхитительно", "потрясающе", "великолепно",
    # Russian — expanded
    "заботливый", "заботливая", "внимательный", "внимательная",
    "добрый", "добрая", "благодарен", "благодарна", "горжусь",
    "вдохновляет", "впечатляет", "невероятно", "гениально", "легенда",
    "бомба", "крутяк", "кайф", "ништяк", "жесть", "зачёт",
    "умница", "красотка", "красавица", "лапочка", "киса", "лапуля",
    "сладкий", "сладкая", "дорогой", "дорогая", "ангел", "крошка",
    "люблю тебя", "ты лучший", "ты лучшая", "ты классный", "ты классная",
    "повезло", "везучий", "везучая", "спасибо", "спасибочки",
    "ахахах", "хахаха", "ахах", "ржу", "угар",
    "поздравляю", "здорово", "молодчина", "ура", "офигеть",
}

NEGATIVE_MARKERS = {
    # English - emotions
    "angry", "upset", "mad", "hate", "annoyed", "frustrated", "sad",
    "disappointed", "hurt", "tired", "sick", "bored", "lonely",
    "stressed", "anxious", "worried", "scared", "awful", "terrible",
    "crying", "cried", "cry", "leave", "goodbye", "done", "over",
    "ignore", "ignored", "blocking", "whatever", "fine",
    "depressed", "miserable", "hopeless", "helpless", "worthless",
    "disgusted", "furious", "devastated", "heartbroken", "betrayed",
    "exhausted", "drained", "overwhelmed", "burned out", "suffocating",
    # English - general negative
    "trash", "garbage", "stupid", "dumb", "worst", "horrible",
    "pathetic", "lame", "cringe", "mid", "L", "bruh",
    "sucks", "ruined", "failed", "screwed", "messed up",
    "😢", "😭", "😡", "😤", "💔", "😞", "😔", "😒", "🙄",
    "🤮", "😰", "😩", "😫", "🤬",
    # Russian
    "злюсь", "злая", "злой", "обижен", "обижена", "расстроен",
    "расстроена", "грустно", "плохо", "ужасно", "устал", "устала",
    "надоел", "надоела", "бесит", "достал", "достала", "хватит",
    "уходи", "всё", "пока", "не хочу", "отвратительно", "кошмар",
    "тошнит", "ненавижу", "выгорел", "выгорела", "задолбало",
    "раздражает", "разочарован", "разочарована", "безнадёжно",
    # Russian profanity / insults (comprehensive for accurate aggression detection)
    "блядь", "блять", "блядина", "сука", "сучка", "сучара", "пиздец", "пизда",
    "нахуй", "нахер", "ёбаный", "ебаный", "ебать", "заебал", "заебала", "заебали",
    "отъебись", "гандон", "мудак", "мудила", "дебил", "долбоёб", "долбоеб",
    "пошёл", "пошел", "пошла", "иди", "катись", "убирайся", "свали",
    "урод", "козёл", "козел", "придурок", "тварь", "скотина",
    "дура", "дурак", "идиотка", "идиот", "кретин", "лох", "чмо", "отстой",
    "заткнись", "отвали", "вали", "проваливай", "отвяжись",
    "ублюдок", "выродок", "мразь", "подонок", "шлюха", "шалава", "сволочь",
    "гнида", "падла", "паскуда", "хуй", "хуйня", "хуесос",
    "пидор", "пидорас", "педик", "педераст", "педарез", "пидр",
    "уёбок", "уебок", "уёбище", "пиздабол", "пизданул", "пиздюк",
    "засранец", "засранка", "говно", "говнюк", "дерьмо",
    "жалкий", "ничтожество", "позор", "убогий", "тупица",
    "неудачник", "бездарь", "клоун", "посмешище", "бестолочь",
    "никчёмный", "никчемный", "позорище",
    "даун", "олух", "баран", "выблядок", "хуяк",
    "охуел", "охуела", "оборзел", "оборзела", "обнаглел", "обнаглела",
    "задолбал", "задолбала", "достал", "достала", "бесишь",
    "какого хуя", "какого хера", "чё за хуйня", "нихуя", "нихера",
    "хули", "ебал", "выебать", "поебать", "обосрался", "обосралась",
    "жопа", "сосать", "отсоси", "рот закрой", "закрой рот",
    # English profanity
    "fuck", "fucking", "shit", "shitty", "bitch", "asshole", "dick",
    "dickhead", "bastard", "damn", "crap", "moron", "idiot", "loser",
    "wtf", "stfu", "gtfo", "dumbass", "dipshit", "motherfucker",
    "piss off", "screw you", "go to hell", "drop dead", "die",
    "trash", "garbage", "pathetic", "worthless", "disgusting",
    "ffs", "jfc", "omfg", "bullshit",
}

QUESTION_MARKERS = {"?", "what", "why", "how", "when", "where", "who", "do you",
                     "are you", "can you", "will you", "could you", "would you",
                     "have you", "should i", "is it", "does it", "which",
                     "thoughts on", "opinion on", "what about", "how come",
                     # Russian question words
                     "что", "почему", "как", "когда", "где", "кто", "зачем",
                     "какой", "какая", "какие", "который", "чей", "сколько",
                     "куда", "откуда", "отчего", "неужели", "разве",
                     "ты", "а ты", "как думаешь", "что думаешь", "правда ли",
                     "как считаешь", "что скажешь", "а что если", "ну и как"}

FLIRTY_MARKERS = {"😏", "😉", "😈", "🔥", "💦", "wink", "naughty", "tonight",
                   "kiss", "cuddle", "hug", "bed", "come over", "miss your",
                   # Russian flirty
                   "целую", "обнимаю", "приезжай", "хочу", "хочу тебя",
                   "соскучилась", "соскучился", "думаю о тебе", "хочу тебя видеть",
                   "ты такой сексуальный", "ты такая сексуальная", "ты горячий", "ты горячая",
                   "скучаю по тебе", "хочу к тебе", "давай встретимся",
                   "мне нравишься", "ты мне нравишься", "обнимашки", "поцелуй"}

# General intent markers for non-romantic message classification
ADVICE_SEEKING_MARKERS = {"should i", "what should", "advice", "suggest", "recommend",
                           "what would you do", "help me decide", "thoughts on",
                           "как думаешь", "что посоветуешь", "стоит ли", "подскажи"}

SHARING_NEWS_MARKERS = {"guess what", "you won't believe", "just found out", "breaking",
                         "did you hear", "big news", "omg", "oh my god",
                         "представляешь", "не поверишь", "узнал", "узнала", "новость"}

DEBATE_MARKERS = {"i think", "i disagree", "actually", "but what about", "on the other hand",
                   "my point is", "to be fair", "controversial take", "hot take",
                   "я считаю", "не согласен", "на самом деле", "с другой стороны"}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of a message with improved accuracy.

    Uses multi-signal analysis:
    - Word-level matching (positive/negative markers)
    - Phrase-level matching for multi-word expressions
    - Negation detection (flips sentiment)
    - Emoji sentiment analysis
    - Intensity calibration via compound score (-1.0 to 1.0)
    - ALL CAPS boost for intensity
    """
    text_lower = text.lower()
    words = set(re.findall(r'\w+', text_lower))
    chars = set(text)

    pos_count = len(words & POSITIVE_MARKERS) + len(chars & POSITIVE_MARKERS)
    neg_count = len(words & NEGATIVE_MARKERS) + len(chars & NEGATIVE_MARKERS)

    # Check for emoji sentiment
    for marker in POSITIVE_MARKERS:
        if marker in text and len(marker) > 1 and not marker.isalpha():
            pos_count += 1
    for marker in NEGATIVE_MARKERS:
        if marker in text and len(marker) > 1 and not marker.isalpha():
            neg_count += 1

    # ── ACCURACY BOOST: Phrase-level sentiment (multi-word expressions) ──
    _pos_phrases = [
        "i love", "so happy", "that's great", "so cool", "love it", "can't wait",
        "miss you", "thinking of you", "so proud", "love you", "made my day",
        "так рад", "так рада", "люблю тебя", "скучаю по тебе", "обожаю",
        "мне нравится", "как круто", "здорово", "молодец", "умничка",
        "это прекрасно", "офигенно", "потрясающе", "замечательно",
    ]
    _neg_phrases = [
        "i hate", "so tired", "can't stand", "fed up", "sick of", "piss me off",
        "leave me alone", "don't care", "couldn't care less", "go away",
        "ненавижу", "достал", "достала", "бесишь", "мне плевать", "мне насрать",
        "отстань", "не трогай", "не лезь", "надоел", "надоела", "задолбал",
        "как бесит", "просто ужас", "кошмар", "отвратительно",
    ]
    for p in _pos_phrases:
        if p in text_lower:
            pos_count += 2  # Phrases count double
    for p in _neg_phrases:
        if p in text_lower:
            neg_count += 2

    # ── ACCURACY BOOST: Negation detection ──
    _negation_words = {"not", "don't", "dont", "didn't", "didnt", "isn't", "isnt",
                       "wasn't", "wasnt", "won't", "wont", "can't", "cant", "no",
                       "never", "neither", "nor", "nothing", "nowhere", "nobody",
                       "не", "нет", "ни", "никогда", "ничего", "нигде", "никуда",
                       "ничто", "никто", "некуда", "неоткуда"}
    word_list = re.findall(r'\w+', text_lower)
    for i, w in enumerate(word_list):
        if w in _negation_words and i + 1 < len(word_list):
            next_word = word_list[i + 1]
            if next_word in POSITIVE_MARKERS:
                # "not happy" → negative
                pos_count = max(0, pos_count - 1)
                neg_count += 1
            elif next_word in NEGATIVE_MARKERS:
                # "not bad" → positive
                neg_count = max(0, neg_count - 1)
                pos_count += 1

    # ── ACCURACY BOOST: ALL CAPS intensity ──
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    caps_intensity_boost = min(caps_words * 0.1, 0.3)

    # ── ACCURACY BOOST: Exclamation intensity ──
    excl_count = text.count("!")
    excl_intensity_boost = min(excl_count * 0.05, 0.2)

    total = pos_count + neg_count
    if total == 0:
        sentiment = "neutral"
        intensity = 0.0
        compound = 0.0
    elif pos_count > neg_count:
        sentiment = "positive"
        intensity = min(pos_count / max(total, 1), 1.0)
        compound = min((pos_count - neg_count) / max(total, 1) + caps_intensity_boost + excl_intensity_boost, 1.0)
    elif neg_count > pos_count:
        sentiment = "negative"
        intensity = min(neg_count / max(total, 1), 1.0)
        compound = max(-(neg_count - pos_count) / max(total, 1) - caps_intensity_boost - excl_intensity_boost, -1.0)
    else:
        sentiment = "mixed"
        intensity = 0.5
        compound = 0.0

    # Detect if message is a question (expanded patterns)
    _question_words = [
        "what", "why", "how", "when", "where", "who", "which", "whose",
        "когда", "что", "почему", "зачем", "как", "где", "кто", "куда",
        "откуда", "сколько", "какой", "какая", "какое", "какие", "чей",
    ]
    is_question = "?" in text or any(q in text_lower for q in _question_words)

    # Detect flirtiness
    flirty_count = len(words & FLIRTY_MARKERS)
    for marker in FLIRTY_MARKERS:
        if marker in text and not marker.isalpha():
            flirty_count += 1
    is_flirty = flirty_count > 0

    # Detect additional intents
    is_advice_seeking = any(m in text_lower for m in ADVICE_SEEKING_MARKERS)
    is_sharing_news = any(m in text_lower for m in SHARING_NEWS_MARKERS)
    is_debate = any(m in text_lower for m in DEBATE_MARKERS)
    is_vent = any(m in text_lower for m in [
        "ugh", "can't deal", "so annoyed", "fml", "i'm done", "im done",
        "бесит", "достало", "задолбало", "заебало", "надоело", "устал",
        "устала", "сил нет", "не могу больше",
    ])

    return {
        "sentiment": sentiment,
        "compound": round(compound, 3),
        "intensity": round(intensity, 2),
        "is_question": is_question,
        "is_flirty": is_flirty,
        "is_advice_seeking": is_advice_seeking,
        "is_sharing_news": is_sharing_news,
        "is_debate": is_debate,
        "is_vent": is_vent,
        "positive_signals": pos_count,
        "negative_signals": neg_count,
    }


# ============= LANGUAGE DETECTION =============

CYRILLIC_PATTERN = re.compile(r'[а-яА-ЯёЁ]')
LATIN_PATTERN = re.compile(r'[a-zA-Z]')


def detect_language(text: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Detect if text is English, Russian, or mixed.

    Uses current message + conversation history for robust detection.
    A single message might have no letters (emoji, "?"), so history helps.
    """
    cyrillic_count = len(CYRILLIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = cyrillic_count + latin_count

    # If current message has very few letters, check history for dominant language
    if total < 3 and conversation_history:
        their_texts = " ".join(
            m.get("text", "") for m in conversation_history[-10:]
            if m.get("sender") == "Them" and m.get("text")
        )
        hist_cyrillic = len(CYRILLIC_PATTERN.findall(their_texts))
        hist_latin = len(LATIN_PATTERN.findall(their_texts))
        cyrillic_count += hist_cyrillic
        latin_count += hist_latin
        total = cyrillic_count + latin_count

    if total == 0:
        return "unknown"

    cyrillic_ratio = cyrillic_count / total

    if cyrillic_ratio > 0.5:       # Lowered from 0.7 — even half Cyrillic = Russian
        return "russian"
    elif cyrillic_ratio > 0.15:     # Lowered from 0.2
        return "mixed"
    else:
        return "english"


# ============= CONVERSATION STAGE DETECTION =============

def detect_conversation_stage(messages: List[Dict[str, str]]) -> str:
    """Detect the current stage of the conversation using multi-dimensional analysis.

    messages: list of {"sender": "Me"|"Them", "text": "..."}
    Returns: "new_chat", "warming_up", "flowing", "deep", "conflict", "cooling_down", "makeup"
    """
    if not messages:
        return "new_chat"

    recent = messages[-15:]  # Extended to 15 for better context

    # ── CONFLICT DETECTION: Multi-signal weighted scoring ──
    their_recent = [m for m in recent if m["sender"] == "Them"]
    conflict_score = 0.0

    # Heavy profanity (direct insults, slurs) — instant conflict trigger
    _heavy_profanity = {
        # English — direct insults
        "fuck you", "fuck off", "stfu", "gtfo", "go to hell", "drop dead",
        "screw you", "piss off", "motherfucker", "piece of shit", "eat shit",
        "hate you", "die", "kill yourself",
        # Russian — прямые оскорбления
        "пошёл нахуй", "пошел нахуй", "иди нахуй", "иди на хуй",
        "отъебись", "отвали", "пошёл на", "пошел на",
        "заебал", "заебала", "заебали",
        "ты тупой", "ты тупая", "ты дебил", "ты дура", "ты дурак",
        "ты урод", "ты мразь", "ты тварь", "ты чмо",
    }
    # Medium profanity (swear words, not necessarily directed at us)
    _medium_profanity = {
        "fuck", "fucking", "bitch", "asshole", "shit", "bastard", "dickhead",
        "moron", "idiot", "loser", "dumbass", "dipshit", "bullshit",
        "pathetic", "worthless", "trash", "garbage", "ffs", "wtf",
        "блядь", "блять", "сука", "сучка", "сучара", "пиздец", "пизда",
        "нахуй", "нахер", "гандон", "мудак", "мудила", "дебил",
        "долбоёб", "долбоеб", "тварь", "урод", "козёл", "козел",
        "ёбаный", "ебаный", "ебать", "чмо", "скотина", "придурок", "кретин",
        "ублюдок", "выродок", "мразь", "подонок", "шлюха", "сволочь",
        "гнида", "падла", "паскуда", "хуй", "хуйня", "хуесос",
        "пидор", "пидорас", "педик", "педераст", "педарез", "пидр",
        "уёбок", "уебок", "уёбище", "пиздабол",
        "засранец", "говно", "говнюк", "дерьмо", "тупица", "бездарь",
        "хули", "нихуя", "нихера", "ебал", "выебать", "поебать",
        "жопа", "сосать", "отсоси", "идиот", "идиотка",
        # Additional commonly missed ones
        "пизданул", "выблядок", "баран", "сучий", "обоссаться",
        "обосрался", "пиздюк", "хуяк", "блядина", "шалава",
        "даун", "олух", "охуел", "охуела", "оборзел", "оборзела",
        "ахуеть", "ахуенно",  # can be positive but in conflict = aggression
        "чё за хуйня", "какого хуя", "какого хера",
    }
    # Hostile TONE markers (not profanity, but aggressive phrasing)
    _hostile_tone = {
        "you always", "you never", "sick of", "tired of", "fed up",
        "leave me alone", "don't talk to me", "stop", "enough",
        "shut up", "не пиши мне", "отстань", "хватит", "достал",
        "достала", "надоел", "надоела", "бесишь", "раздражаешь",
        "задолбал", "задолбала", "не трогай", "не лезь",
        "отвяжись", "замолчи", "закрой рот", "рот закрой",
        "заткнись", "вали отсюда", "проваливай", "катись",
        "не указывай", "не учи меня", "я тебя не спрашивал",
    }
    # ACCUSATION patterns
    _accusation_patterns = [
        r"ты\s+(всегда|никогда|вечно|постоянно|опять)",
        r"you\s+(always|never|still|again)",
        r"(почему|зачем)\s+ты",
        r"(why|how)\s+(do|did|could|would)\s+you",
        r"чё\s+за\s+хуйн",
        r"(что|чё|чо)\s+ты\s+(творишь|делаешь|несёшь|несешь)",
    ]

    for m in their_recent:
        m_lower = m["text"].lower()
        m_score = 0.0

        # Heavy profanity: +3.0 (instant conflict)
        if any(p in m_lower for p in _heavy_profanity):
            m_score += 3.0

        # Medium profanity: +1.5
        medium_hits = sum(1 for p in _medium_profanity if p in m_lower)
        m_score += min(medium_hits * 1.5, 3.0)

        # Hostile tone: +1.0
        if any(p in m_lower for p in _hostile_tone):
            m_score += 1.0

        # Accusation patterns: +0.8
        for pat in _accusation_patterns:
            if re.search(pat, m_lower, re.IGNORECASE):
                m_score += 0.8
                break

        # ALL CAPS sections (yelling): +0.5
        words = m["text"].split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        if caps_words >= 2:
            m_score += 0.5

        # Negative sentiment: +0.5
        sent = analyze_sentiment(m["text"])
        if sent["sentiment"] == "negative":
            m_score += 0.5 * sent.get("intensity", 0.5)

        # Exclamation marks (anger): +0.3 per ! beyond the first
        excl_count = m_lower.count("!")
        if excl_count >= 2:
            m_score += 0.3 * (excl_count - 1)

        # Recency weighting: most recent messages matter MORE
        idx = their_recent.index(m)
        recency_mult = 1.0 + 0.3 * (idx / max(len(their_recent) - 1, 1))
        m_score *= recency_mult

        conflict_score += m_score

    # ── ACCURACY BOOST: Sarcastic dismissal patterns ──
    _dismiss_patterns = [
        r"\bwhatever\b", r"\bwhatevs\b", r"\bw/e\b", r"\bidc\b",
        r"\bidgaf\b", r"\bi\s*don'?t\s*care\b", r"\bмне\s*(всё|все)\s*равно\b",
        r"\bпофиг\b", r"\bнасрать\b", r"\bплевать\b", r"\bнеинтересно\b",
        r"\bок\s+и\s+что\b", r"\bну\s+и\s+что\b", r"\bда\s+пофиг\b",
    ]
    for m in their_recent:
        m_lower = m["text"].lower()
        for pat in _dismiss_patterns:
            if re.search(pat, m_lower, re.IGNORECASE):
                conflict_score += 0.6
                break

    # ── ACCURACY BOOST: Short hostile responses (1-3 words) hit harder ──
    for m in their_recent:
        m_words = len(m["text"].split())
        m_lower = m["text"].lower()
        if m_words <= 3:
            # Short hostile = punchy aggression, boost it
            _short_hostile = {"нет", "отвали", "хватит", "стоп", "нахуй",
                              "no", "stop", "bye", "whatever", "leave",
                              "уйди", "вали", "свали", "пошёл", "пошел"}
            if any(w in m_lower.split() for w in _short_hostile):
                conflict_score += 0.8

    # Conflict threshold: 1.5+ = conflict (lowered for faster detection accuracy)
    if conflict_score >= 1.5:
        return "conflict"

    # If too few messages and no conflict, it's a new chat
    if len(messages) < 3:
        return "new_chat"

    # ── MAKEUP DETECTION: conflict → positive shift ──
    if len(recent) >= 4:
        older = recent[:len(recent) // 2]
        newer = recent[len(recent) // 2:]
        older_neg = sum(1 for m in older if analyze_sentiment(m["text"])["sentiment"] == "negative")
        newer_pos = sum(1 for m in newer if analyze_sentiment(m["text"])["sentiment"] == "positive")
        if older_neg >= 2 and newer_pos >= 2:
            return "makeup"

    # ── STAGE FROM ENGAGEMENT PATTERNS ──
    their_msgs = [m for m in recent if m["sender"] == "Them"]
    my_msgs = [m for m in recent if m["sender"] == "Me"]

    if their_msgs:
        avg_len = sum(len(m["text"]) for m in their_msgs) / len(their_msgs)
        # Also check engagement depth — questions, emotional words
        has_questions = any("?" in m["text"] for m in their_msgs[-3:])
        has_emotional = any(
            analyze_sentiment(m["text"])["sentiment"] in ("positive", "negative")
            for m in their_msgs[-3:]
        )

        if avg_len > 80 or (avg_len > 40 and has_emotional):
            return "deep"
        elif avg_len > 30 or has_questions:
            return "flowing"
        elif avg_len > 12:
            return "warming_up"
        else:
            # Short messages — check if cooling down or just casual
            if my_msgs and len(my_msgs) >= 2:
                their_reply_rate = len(their_msgs) / max(len(my_msgs), 1)
                if their_reply_rate < 0.5:
                    return "cooling_down"
            return "cooling_down"

    return "warming_up"


# ============= TOPIC DETECTION =============

TOPIC_KEYWORDS = {
    # === Relationship & Emotional ===
    "romantic": ["love", "miss", "heart", "forever", "together", "люблю", "скучаю", "сердце", "навсегда"],
    "intimate": ["kiss", "hug", "cuddle", "bed", "miss your", "want you", "целую", "обнимаю", "хочу"],
    "jealousy": ["who is she", "talking to", "flirt", "jealous", "ревную", "кто она", "общаешься"],
    "conflict": [
        "angry", "mad", "upset", "why did", "you always", "never", "hate",
        "fuck", "fucking", "bitch", "idiot", "stupid", "stfu", "asshole",
        "screw you", "piss off", "moron", "loser", "dumbass", "dipshit",
        "motherfucker", "bastard", "dickhead", "bullshit", "worthless",
        "pathetic", "disgusting", "trash", "garbage", "die", "drop dead",
        # Russian
        "злюсь", "бесит", "достал", "блядь", "блять", "сука", "сучка",
        "пиздец", "пизда", "нахуй", "нахер", "гандон", "мудак", "мудила",
        "дебил", "долбоёб", "долбоеб", "тварь", "урод", "козёл", "козел",
        "ненавижу", "заткнись", "отвали", "пошёл", "пошел", "вали",
        "ёбаный", "ебаный", "ебать", "заебал", "заебала", "отъебись",
        "ублюдок", "мразь", "подонок", "шлюха", "сволочь", "гнида",
        "хуй", "хуйня", "хуесос", "уёбок", "уебок", "пиздабол",
        "говно", "дерьмо", "скотина", "чмо", "придурок", "лох",
    ],
    "support": ["tired", "stressed", "hard day", "bad day", "help me", "устал", "тяжело", "плохо", "помоги"],
    "emotional": ["feel", "feeling", "sad", "happy", "cry", "hurt", "scared", "чувству", "грустно", "больно"],
    "future_together": ["marry", "kids", "move in", "свадьба", "дети", "жить вместе"],
    # === Daily Life & Casual ===
    "casual": ["how", "what", "doing", "today", "как", "что", "делаешь", "сегодня"],
    "plans": ["tomorrow", "weekend", "tonight", "meet", "see you", "завтра", "вечер", "встретимся", "давай"],
    "daily_life": ["morning", "woke up", "going to", "came home", "утро", "проснулся", "пришел домой"],
    "food": ["eat", "dinner", "lunch", "cook", "restaurant", "hungry", "recipe", "еда", "ужин", "готовить", "ресторан"],
    "weather": ["rain", "sunny", "cold", "hot", "snow", "weather", "дождь", "солнце", "холодно", "жарко", "погода"],
    "sleep": ["sleep", "tired", "insomnia", "dream", "nap", "спать", "устал", "сон", "бессонница"],
    # === Work & Career ===
    "work": ["work", "job", "boss", "office", "meeting", "deadline", "project", "client",
             "работа", "начальник", "офис", "встреча", "проект", "дедлайн", "клиент"],
    "career": ["promotion", "interview", "resume", "salary", "fired", "quit", "career",
               "повышение", "собеседование", "зарплата", "уволили", "карьера"],
    "business": ["startup", "invest", "company", "revenue", "market", "стартап", "компания", "рынок"],
    # === Education & Learning ===
    "education": ["school", "university", "class", "exam", "study", "homework", "professor", "lecture",
                  "школа", "университет", "экзамен", "учиться", "домашка", "лекция"],
    "learning": ["learn", "course", "tutorial", "skill", "practice", "учить", "курс", "навык"],
    # === Technology ===
    "technology": ["code", "programming", "app", "software", "computer", "phone", "internet", "ai",
                   "algorithm", "database", "server", "код", "программирование", "приложение", "компьютер"],
    "gaming": ["game", "play", "level", "win", "lose", "stream", "console",
               "игра", "играть", "уровень", "стрим"],
    "social_media": ["instagram", "tiktok", "twitter", "post", "followers", "viral", "feed",
                     "инстаграм", "тикток", "подписчики"],
    # === Entertainment & Culture ===
    "movies_tv": ["movie", "film", "show", "series", "watch", "netflix", "episode", "season",
                  "фильм", "сериал", "смотреть", "серия"],
    "music": ["song", "music", "album", "concert", "band", "listen", "playlist",
              "песня", "музыка", "альбом", "концерт", "слушать"],
    "books": ["book", "read", "novel", "author", "chapter", "story",
              "книга", "читать", "роман", "автор", "история"],
    "art": ["art", "draw", "paint", "museum", "gallery", "creative",
            "искусство", "рисовать", "музей", "галерея"],
    # === Sports & Fitness ===
    "sports": ["football", "soccer", "basketball", "tennis", "match", "team", "score", "champion",
               "футбол", "баскетбол", "матч", "команда", "чемпион"],
    "fitness": ["gym", "workout", "exercise", "run", "yoga", "diet", "muscle", "weight",
                "тренировка", "зал", "бег", "йога", "диета"],
    # === Travel ===
    "travel": ["trip", "travel", "vacation", "flight", "hotel", "visit", "country", "city",
               "путешествие", "отпуск", "рейс", "отель", "страна", "город"],
    # === Health ===
    "health": ["doctor", "sick", "hospital", "medicine", "pain", "headache", "fever", "health",
               "врач", "болеть", "больница", "лекарство", "боль", "температура", "здоровье"],
    "mental_health": ["anxiety", "depression", "therapy", "therapist", "panic", "overwhelmed",
                      "тревога", "депрессия", "терапия", "паника"],
    # === Finance ===
    "finance": ["money", "pay", "bill", "rent", "save", "budget", "expensive", "price", "crypto",
                "деньги", "зарплата", "счёт", "аренда", "бюджет", "цена", "крипто"],
    # === Family & Friends ===
    "family": ["mom", "dad", "parent", "sister", "brother", "family", "grandma", "grandpa",
               "мама", "папа", "родители", "сестра", "брат", "семья", "бабушка", "дедушка"],
    "friends": ["friend", "bestie", "hangout", "party", "friend group", "друг", "подруга", "тусовка", "вечеринка"],
    "pets": ["cat", "dog", "pet", "puppy", "kitten", "кот", "кошка", "собака", "питомец"],
    # === News & Current Events ===
    "news": ["news", "happened", "election", "war", "crisis", "government", "politics",
             "новости", "выборы", "война", "кризис", "правительство", "политика"],
    # === Philosophy & Deep Talk ===
    "philosophy": ["meaning", "life", "purpose", "existence", "universe", "believe", "consciousness",
                   "смысл", "жизнь", "цель", "существование", "вселенная", "сознание"],
    "religion": ["god", "pray", "faith", "church", "spiritual", "бог", "молиться", "вера", "церковь"],
    # === Humor & Playful ===
    "playful": ["haha", "lol", "😂", "joke", "funny", "tease", "dare", "хаха", "смешно", "прикол"],
    "memes": ["meme", "viral", "trend", "мем", "тренд"],
    # === Photos & Media ===
    "photos": ["photo", "pic", "selfie", "look", "outfit", "фото", "фотка", "селфи"],
    # === Advice & Help ===
    "advice": ["advice", "suggest", "recommend", "should i", "what should", "opinion",
               "совет", "подскажи", "что делать", "стоит ли", "мнение"],
    "vent": ["ugh", "can't deal", "so annoyed", "need to rant", "fml", "hate my life",
             "бесит", "достало", "не могу", "задолбало"],
}


def detect_topics(text: str) -> List[str]:
    """Detect conversation topics in a message."""
    text_lower = text.lower()
    detected = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches >= 1:
            detected.append(topic)

    return detected if detected else ["casual"]


# ============= RESPONSE STRATEGY =============

def recommend_strategy(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Based on full context analysis, recommend a response strategy."""
    sentiment = analysis["sentiment"]
    stage = analysis["conversation_stage"]
    topics = analysis["topics"]
    language = analysis["language"]
    is_question = sentiment.get("is_question", False) if isinstance(sentiment, dict) else False
    is_flirty = sentiment.get("is_flirty", False) if isinstance(sentiment, dict) else False

    strategy = {
        "recommended_tone": "casual",
        "recommended_length": "short",  # short, medium, long
        "should_ask_question": False,
        "should_use_emoji": False,
        "should_be_vulnerable": False,
        "language": language,
        "priority_action": None,
        "notes": [],
    }

    # Stage-based adjustments
    if stage == "conflict":
        strategy["recommended_tone"] = "sincere"
        strategy["recommended_length"] = "medium"
        strategy["should_be_vulnerable"] = True
        strategy["priority_action"] = "acknowledge_feelings"
        strategy["notes"].append("They seem upset. Be genuine and empathetic, don't deflect.")

    elif stage == "makeup":
        strategy["recommended_tone"] = "sweet"
        strategy["recommended_length"] = "medium"
        strategy["should_use_emoji"] = True
        strategy["priority_action"] = "reinforce_positive"
        strategy["notes"].append("Things are getting better after conflict. Be warm and reassuring.")

    elif stage == "cooling_down":
        strategy["recommended_tone"] = "playful"
        strategy["should_ask_question"] = True
        strategy["priority_action"] = "re_engage"
        strategy["notes"].append("Conversation is dying down. Ask something interesting to re-engage.")

    elif stage == "deep":
        strategy["recommended_tone"] = "sincere"
        strategy["recommended_length"] = "medium"
        strategy["notes"].append("They're being open and detailed. Match their energy.")

    elif stage == "new_chat":
        strategy["recommended_tone"] = "playful"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Fresh conversation. Be warm and engaging.")

    # Topic-based adjustments
    if "emotional" in topics or "support" in topics:
        strategy["recommended_tone"] = "supportive"
        strategy["recommended_length"] = "medium"
        strategy["should_be_vulnerable"] = True
        strategy["priority_action"] = "show_empathy"
        strategy["notes"].append("They're sharing emotions. Listen and validate first, then comfort.")

    if "mental_health" in topics:
        strategy["recommended_tone"] = "supportive"
        strategy["recommended_length"] = "medium"
        strategy["should_be_vulnerable"] = True
        strategy["priority_action"] = "show_empathy"
        strategy["notes"].append("Mental health topic. Be gentle, validating, non-judgmental. Don't try to fix.")

    if "romantic" in topics or "intimate" in topics:
        strategy["recommended_tone"] = "romantic"
        strategy["should_use_emoji"] = True
        if not is_flirty:
            strategy["notes"].append("Romantic context but not explicitly flirty. Be sweet, not over the top.")

    if "jealousy" in topics:
        strategy["recommended_tone"] = "sincere"
        strategy["priority_action"] = "reassure"
        strategy["notes"].append("Jealousy detected. Reassure without being dismissive.")

    if "future_together" in topics:
        strategy["recommended_tone"] = "sincere"
        strategy["recommended_length"] = "medium"
        strategy["notes"].append("They're talking about the future together. Be thoughtful and honest.")

    if "playful" in topics and stage != "conflict":
        strategy["recommended_tone"] = "playful"
        strategy["should_use_emoji"] = True

    if "photos" in topics:
        strategy["notes"].append("They're sharing or discussing photos. React naturally and specifically.")

    # --- General-purpose topic adjustments ---
    if "work" in topics or "career" in topics:
        strategy["recommended_tone"] = "engaged"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Work/career topic. Show genuine interest, ask about details.")

    if "business" in topics:
        strategy["recommended_tone"] = "engaged"
        strategy["notes"].append("Business topic. Be thoughtful and supportive of their ambitions.")

    if "education" in topics or "learning" in topics:
        strategy["recommended_tone"] = "encouraging"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Education topic. Be encouraging and show interest in what they're learning.")

    if "technology" in topics or "gaming" in topics:
        strategy["recommended_tone"] = "engaged"
        strategy["notes"].append("Tech/gaming topic. Match their enthusiasm, discuss specifics.")

    if "movies_tv" in topics or "music" in topics or "books" in topics or "art" in topics:
        strategy["recommended_tone"] = "enthusiastic"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Entertainment/culture topic. Share opinions, ask about theirs.")

    if "sports" in topics or "fitness" in topics:
        strategy["recommended_tone"] = "energetic"
        strategy["notes"].append("Sports/fitness topic. Match their energy, be engaged.")

    if "travel" in topics:
        strategy["recommended_tone"] = "enthusiastic"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Travel topic. Show excitement, ask about experiences or plans.")

    if "health" in topics:
        strategy["recommended_tone"] = "caring"
        strategy["notes"].append("Health topic. Show concern, be supportive without being overbearing.")

    if "finance" in topics:
        strategy["recommended_tone"] = "practical"
        strategy["notes"].append("Finance topic. Be helpful and non-judgmental.")

    if "family" in topics:
        strategy["recommended_tone"] = "warm"
        strategy["recommended_length"] = "medium"
        strategy["notes"].append("Family topic. Be respectful and warm, show you care about their family too.")

    if "friends" in topics:
        strategy["recommended_tone"] = "casual"
        strategy["should_ask_question"] = True
        strategy["notes"].append("Friends topic. Be chill, show interest in their social life.")

    if "news" in topics:
        strategy["recommended_tone"] = "thoughtful"
        strategy["recommended_length"] = "medium"
        strategy["notes"].append("News/current events. Be thoughtful, share perspectives without being preachy.")

    if "philosophy" in topics or "religion" in topics:
        strategy["recommended_tone"] = "thoughtful"
        strategy["recommended_length"] = "medium"
        strategy["notes"].append("Deep/philosophical topic. Be genuine, share authentic thoughts. Don't be dismissive.")

    if "advice" in topics:
        strategy["recommended_tone"] = "helpful"
        strategy["recommended_length"] = "medium"
        strategy["should_ask_question"] = True
        strategy["notes"].append("They're seeking advice. Listen first, then offer perspective. Don't lecture.")

    if "vent" in topics:
        strategy["recommended_tone"] = "validating"
        strategy["priority_action"] = "show_empathy"
        strategy["notes"].append("They're venting. Validate first. Don't jump to solutions unless asked.")

    if "memes" in topics:
        strategy["recommended_tone"] = "playful"
        strategy["notes"].append("Meme/humor sharing. Match the energy, be fun.")

    # Sentiment-based adjustments
    if isinstance(sentiment, dict):
        if sentiment.get("sentiment") == "negative" and sentiment.get("intensity", 0) > 0.5:
            strategy["recommended_tone"] = "supportive"
            strategy["recommended_length"] = "medium"
            strategy["notes"].append("Strong negative sentiment. Be careful and empathetic.")

        if is_flirty:
            strategy["recommended_tone"] = "flirty"
            strategy["should_use_emoji"] = True

    # Question handling
    if is_question:
        strategy["notes"].append("They asked a question. Make sure to actually answer it.")

    return strategy


# ============= MAIN ANALYSIS FUNCTION =============

def analyze_context(
    messages: List[Dict[str, str]],
    incoming_text: str,
    chat_id: int,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """Full context analysis of a conversation.

    Args:
        messages: list of {"sender": "Me"|"Them", "text": "..."}
        incoming_text: the latest incoming message
        chat_id: Telegram chat ID
        username: Telegram username if available

    Returns comprehensive analysis dict.
    """
    # Analyze the incoming message
    sentiment = analyze_sentiment(incoming_text)
    language = detect_language(incoming_text, messages)
    topics = detect_topics(incoming_text)
    # Include current message in stage detection (otherwise profanity in
    # the current message won't trigger conflict detection)
    _msgs_with_current = list(messages) + [{"sender": "Them", "text": incoming_text}]
    stage = detect_conversation_stage(_msgs_with_current)

    # Load and update memory
    memory = load_memory(chat_id)
    memory["total_messages_seen"] = memory.get("total_messages_seen", 0) + 1

    # Update language preference
    if language != "unknown":
        memory["their_language_preference"] = language

    # Track their topics
    for topic in topics:
        if topic not in memory.get("their_topics", []):
            memory["their_topics"] = memory.get("their_topics", [])
            memory["their_topics"].append(topic)
            # Keep last 20 topics
            memory["their_topics"] = memory["their_topics"][-20:]

    # Detect pet names they use
    pet_name_patterns = [
        r'\b(baby|babe|honey|sweetheart|darling|love)\b',
        r'(малыш|малышка|зайка|зай|котик|котёнок|солнышко|солнце|милый|милая|родной|родная|любимый|любимая|киса|кисуля|лапочка|лапуля|сладкий|сладкая|дорогой|дорогая|крошка|ангел|ангелочек|пупсик|мышонок|зайчик|зайчонок|золотой|золотая|звёздочка)',
    ]
    for pattern in pet_name_patterns:
        found = re.findall(pattern, incoming_text.lower())
        for name in found:
            if name not in memory.get("pet_names_they_use", []):
                memory["pet_names_they_use"] = memory.get("pet_names_they_use", [])
                memory["pet_names_they_use"].append(name)

    # Track emoji usage
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\uFE00-\uFE0F]', incoming_text))
    if emoji_count > 0:
        memory["conversation_patterns"]["they_use_emojis"] = True

    # Track message length patterns
    their_msgs = [m for m in messages if m["sender"] == "Them"]
    if their_msgs:
        avg = sum(len(m["text"]) for m in their_msgs) / len(their_msgs)
        memory["conversation_patterns"]["avg_their_message_length"] = round(avg, 1)

    # Detect conflict
    if stage == "conflict":
        memory["last_conflict"] = datetime.now().isoformat()

    # Save updated memory
    save_memory(chat_id, memory)

    # Build full analysis
    analysis = {
        "sentiment": sentiment,
        "language": language,
        "topics": topics,
        "conversation_stage": stage,
        "memory": {
            "their_language": memory.get("their_language_preference"),
            "their_interests": memory.get("their_topics", [])[-5:],
            "pet_names_they_use": memory.get("pet_names_they_use", []),
            "they_use_emojis": memory["conversation_patterns"].get("they_use_emojis", False),
            "total_messages": memory.get("total_messages_seen", 0),
        },
    }

    # Get strategy recommendation
    analysis["strategy"] = recommend_strategy(analysis)

    return analysis


def format_context_for_prompt(analysis: Dict[str, Any]) -> str:
    """Format the NLP analysis into a prompt section for Claude."""
    parts = []

    # Sentiment
    s = analysis["sentiment"]
    parts.append(f"Their mood: {s['sentiment']} (intensity: {s['intensity']})")
    if s.get("is_question"):
        parts.append("They asked a question - make sure to answer it")
    if s.get("is_flirty"):
        parts.append("They're being flirty - match that energy")
    if s.get("is_advice_seeking"):
        parts.append("They're seeking advice - listen first, then offer perspective")
    if s.get("is_sharing_news"):
        parts.append("They're sharing news - react with genuine interest/excitement")
    if s.get("is_debate"):
        parts.append("They want to discuss/debate - engage thoughtfully with your own perspective")
    if s.get("is_vent"):
        parts.append("They're venting - react genuinely, dont try to fix it")

    # Language
    lang = analysis["language"]
    if lang == "russian":
        parts.append("IMPORTANT: They wrote in Russian. Reply in Russian.")
    elif lang == "mixed":
        parts.append("They're mixing Russian and English. You can use either or both.")

    # Conversation stage
    stage = analysis["conversation_stage"]
    stage_descriptions = {
        "new_chat": "This is a new/fresh conversation",
        "warming_up": "Conversation is warming up",
        "flowing": "Conversation is flowing naturally",
        "deep": "They're having a deep/meaningful conversation",
        "conflict": "CONFLICT — match their energy, stand your ground, dont be a pushover",
        "cooling_down": "Conversation is dying down. Try to re-engage.",
        "makeup": "Making up after conflict — be real, not fake nice",
    }
    parts.append(f"Conversation stage: {stage_descriptions.get(stage, stage)}")

    # Topics
    topics = analysis["topics"]
    if topics:
        parts.append(f"Topics detected: {', '.join(topics)}")

    # Strategy
    strat = analysis["strategy"]
    parts.append(f"Recommended tone: {strat['recommended_tone']}")
    parts.append(f"Recommended length: {strat['recommended_length']}")
    if strat.get("should_ask_question"):
        parts.append("Consider asking them a question back")
    if strat.get("should_use_emoji"):
        parts.append("Use a couple emojis naturally")
    if strat.get("should_be_vulnerable"):
        parts.append("Be genuine and vulnerable, not deflective")
    if strat.get("priority_action"):
        actions = {
            "acknowledge_feelings": "Priority: React to their feelings like a real person",
            "show_empathy": "Priority: Show you get it — dont be a therapist about it",
            "reassure": "Priority: Be straight with them",
            "reinforce_positive": "Priority: Keep the good energy going",
            "re_engage": "Priority: Re-engage them with something interesting",
        }
        parts.append(actions.get(strat["priority_action"], f"Priority: {strat['priority_action']}"))
    for note in strat.get("notes", []):
        parts.append(f"Note: {note}")

    # Memory insights
    mem = analysis.get("memory", {})
    if mem.get("pet_names_they_use"):
        parts.append(f"They call you: {', '.join(mem['pet_names_they_use'][-3:])}")
    if mem.get("total_messages", 0) > 50:
        parts.append("You've been chatting with this person for a while. Be natural and familiar.")

    return "\n".join(f"- {p}" for p in parts)


# ============= MEMORY MANAGEMENT API =============

def get_all_memories() -> Dict[int, Dict[str, Any]]:
    """Get all chat memories."""
    memories = {}
    for path in MEMORY_DIR.glob("*.json"):
        try:
            chat_id = int(path.stem)
            memories[chat_id] = json.loads(path.read_text())
        except Exception:
            continue
    return memories


def get_memory_summary(chat_id: int) -> Dict[str, Any]:
    """Get a summary of memory for a specific chat."""
    memory = load_memory(chat_id)
    return {
        "chat_id": chat_id,
        "total_messages_seen": memory.get("total_messages_seen", 0),
        "their_language": memory.get("their_language_preference"),
        "their_topics": memory.get("their_topics", [])[-10:],
        "pet_names_they_use": memory.get("pet_names_they_use", []),
        "they_use_emojis": memory.get("conversation_patterns", {}).get("they_use_emojis", False),
        "avg_message_length": memory.get("conversation_patterns", {}).get("avg_their_message_length", 0),
        "last_conflict": memory.get("last_conflict"),
        "notes": memory.get("notes", []),
    }


def add_memory_note(chat_id: int, note: str):
    """Add a freeform note to a chat's memory."""
    memory = load_memory(chat_id)
    memory["notes"] = memory.get("notes", [])
    memory["notes"].append({
        "text": note,
        "added": datetime.now().isoformat(),
    })
    # Keep last 20 notes
    memory["notes"] = memory["notes"][-20:]
    save_memory(chat_id, memory)


def clear_memory(chat_id: int):
    """Clear memory for a specific chat."""
    path = _memory_path(chat_id)
    if path.exists():
        path.unlink()


# ============= ADVANCED INTELLIGENCE SYSTEMS =============
# Everything below was added to make the bot significantly more sophisticated.


# ─── Time-Aware Intelligence ─────────────────────────────────

def get_time_context() -> Dict[str, Any]:
    """Get rich time-of-day and day-of-week context for response calibration."""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    is_weekend = weekday >= 5

    if 5 <= hour < 9:
        period = "early_morning"
        energy = "waking_up"
        vibe = "gentle, soft, good morning energy"
    elif 9 <= hour < 12:
        period = "morning"
        energy = "fresh"
        vibe = "upbeat, energetic, start of day"
    elif 12 <= hour < 14:
        period = "midday"
        energy = "moderate"
        vibe = "casual, lunchtime, check-in energy"
    elif 14 <= hour < 17:
        period = "afternoon"
        energy = "moderate"
        vibe = "relaxed, afternoon energy"
    elif 17 <= hour < 20:
        period = "evening"
        energy = "winding_down"
        vibe = "cozy, relaxed, end of day"
    elif 20 <= hour < 23:
        period = "night"
        energy = "intimate"
        vibe = "warm, intimate, deep conversation time"
    else:
        period = "late_night"
        energy = "low"
        vibe = "sleepy, vulnerable, late night thoughts"

    return {
        "hour": hour,
        "period": period,
        "is_weekend": is_weekend,
        "day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday],
        "energy": energy,
        "vibe": vibe,
        "suggested_greeting": _time_greeting(period, is_weekend),
    }


def _time_greeting(period: str, is_weekend: bool, language: str = "english") -> Optional[str]:
    """Suggest a time-appropriate greeting if starting a new conversation."""
    if language == "russian":
        greetings = {
            "early_morning": "доброе утро, соня",
            "morning": "доброе утро",
            "midday": None,
            "afternoon": None,
            "evening": "привет, как день прошёл?",
            "night": "привет",
            "late_night": "не спишь?",
        }
    else:
        greetings = {
            "early_morning": "good morning sleepyhead",
            "morning": "good morning",
            "midday": None,
            "afternoon": None,
            "evening": "hey, how was your day?",
            "night": "hey you",
            "late_night": "can't sleep?",
        }
    return greetings.get(period)


# ─── Passive-Aggressive Detection ────────────────────────────

PA_PATTERNS = [
    # English patterns
    (r'\bfine\b\.?$', 0.6),              # "fine" or "fine." as full response
    (r'\bok\b\.?$', 0.5),                # just "ok" or "ok."
    (r'\bk\b\.?$', 0.7),                 # just "k"
    (r'\bwhatever\b', 0.7),
    (r'\bnothing\b\.?$', 0.5),           # "nothing" as full response
    (r'\bi don\'?t care\b', 0.6),
    (r'\bdo what you want\b', 0.7),
    (r'\bup to you\b', 0.4),
    (r'\bif you say so\b', 0.7),
    (r'\byeah sure\b', 0.5),
    (r'\bthanks a lot\b', 0.4),          # can be sarcastic
    (r'\bwow\b\.?$', 0.4),              # "wow" alone
    (r'\bnice\b\.?$', 0.3),             # "nice" alone
    (r'\bgood for you\b', 0.6),
    (r'\byou always\b', 0.5),
    (r'\byou never\b', 0.5),
    (r'\bforget it\b', 0.6),
    (r'\bnever ?mind\b', 0.5),
    (r'\bI guess\b', 0.3),
    (r'\.{3,}', 0.3),                    # excessive ellipsis
    (r'\b(lol|haha)\b.*\b(sure|ok|right)\b', 0.5),  # "lol sure", "haha ok"
    # Russian patterns
    (r'\bладно\b\.?$', 0.5),            # "fine"
    (r'\bнормально\b\.?$', 0.4),        # "normally" as dismissive
    (r'\bкак хочешь\b', 0.7),           # "as you want"
    (r'\bмне всё равно\b', 0.6),        # "I don't care"
    (r'\bдa ?ладно\b', 0.5),            # "yeah right"
    (r'\bну и ладно\b', 0.6),           # "well fine then"
    (r'\bзабей\b', 0.6),               # "forget it"
    (r'\bда нет\b', 0.4),              # "well no" (dismissive)
    (r'\bокей\b\.?$', 0.4),             # "okay" dismissive
    (r'\bвот и хорошо\b', 0.5),         # "good then"
    (r'\bпусть как хочешь\b', 0.6),     # "let it be your way"
    (r'\bмне фиолетово\b', 0.7),        # "I don't give a damn"
    (r'\bкак скажешь\b', 0.6),          # "as you say"
    (r'\bделай что хочешь\b', 0.7),     # "do what you want"
    (r'\bмне без разницы\b', 0.6),      # "makes no difference to me"
    (r'\bну ну\b', 0.4),               # "well well" (dismissive)
    (r'\bда конечно\b', 0.5),           # "yeah sure" (sarcastic)
    (r'\bспасибо большое\b', 0.3),      # can be sarcastic
]

# Short dismissive responses (high PA probability when conversation was active)
SHORT_DISMISSIVE = {"k", "ok", "fine", "whatever", "sure", "cool", "nice", "wow",
                     "ладно", "ок", "норм", "ну", "да", "угу", "ага"}


def detect_passive_aggression(text: str, prev_messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Detect passive-aggressive patterns in text."""
    text_lower = text.lower().strip()
    score = 0.0
    signals = []

    # Check PA patterns
    for pattern, weight in PA_PATTERNS:
        if re.search(pattern, text_lower):
            score += weight
            signals.append(f"Pattern: {pattern}")

    # Short response after long conversation = likely PA
    if prev_messages and len(text_lower) < 10:
        their_recent = [m for m in prev_messages[-5:] if m["sender"] == "Them"]
        if their_recent:
            prev_avg_len = sum(len(m["text"]) for m in their_recent) / len(their_recent)
            if prev_avg_len > 30 and len(text_lower) < 10:
                score += 0.3
                signals.append("Sudden short response after longer messages")

    # Period ending (using periods in casual text = colder)
    if text_lower.endswith(".") and len(text_lower) < 30 and "..." not in text_lower:
        score += 0.2
        signals.append("Period ending in short message (cold tone)")

    # All lowercase single word from SHORT_DISMISSIVE set
    if text_lower in SHORT_DISMISSIVE:
        score += 0.3
        signals.append(f"Dismissive single-word: '{text_lower}'")

    is_pa = score >= 0.5
    return {
        "is_passive_aggressive": is_pa,
        "score": round(min(score, 1.0), 2),
        "signals": signals[:3],  # top 3
    }


# ─── Sarcasm Detection ───────────────────────────────────────

SARCASM_PATTERNS = [
    (r'\boh really\b', 0.6),
    (r'\bwow\b.*\bamazing\b', 0.5),
    (r'\bsuuure\b', 0.8),
    (r'\byeah right\b', 0.7),
    (r'\btotally\b\.?$', 0.4),
    (r'\boh how nice\b', 0.6),
    (r'\bthat\'?s great\b\.?$', 0.3),  # could be genuine
    (r'\bof course\b\.?$', 0.4),
    (r'\bsure you (did|do|are|were)\b', 0.6),
    (r'\bI\'?m so happy for you\b', 0.4),
    (r'(?:ha){3,}', 0.3),  # "hahahaha" can be sarcastic in context
    # Caps with short message
    (r'^[A-Z\s!?]{5,}$', 0.3),  # ALL CAPS short
    # Russian sarcasm
    (r'\bну ?да\b.*\bконечно\b', 0.7),
    (r'\bда ?ладно\b', 0.5),
    (r'\bкруто\b\.?$', 0.3),
    (r'\bзамечательно\b\.?$', 0.4),       # "wonderful" sarcastically
    (r'\bпрекрасно\b\.?$', 0.4),           # "wonderful" sarcastically
    (r'\bвосхитительно\b\.?$', 0.5),       # "delightful" sarcastically
    (r'\bкак мило\b', 0.5),                 # "how cute"
    (r'\bкак здорово\b\.?$', 0.4),          # "how great"
    (r'\bну молодец\b', 0.5),               # "well done" sarcastically
    (r'\bумничка\b\.?$', 0.4),              # "smarty" sarcastically
    (r'\bбраво\b\.?$', 0.5),                # "bravo" sarcastically
    (r'\bспасибо тебе\b\.?$', 0.3),         # "thanks to you"
    (r'\bну конечно\b', 0.6),               # "of course" sarcastically
    (r'\bа то\b', 0.3),                     # "yeah right"
]


def detect_sarcasm(text: str) -> Dict[str, Any]:
    """Detect sarcasm heuristically."""
    text_lower = text.lower().strip()
    score = 0.0
    signals = []

    for pattern, weight in SARCASM_PATTERNS:
        if re.search(pattern, text_lower):
            score += weight
            signals.append(pattern)

    # Contradicting sentiment: positive words + negative emoji
    pos_words = sum(1 for w in POSITIVE_MARKERS if w in text_lower and len(w) > 2 and w.isalpha())
    neg_emoji = sum(1 for e in NEGATIVE_MARKERS if e in text and not e.isalpha())
    if pos_words > 0 and neg_emoji > 0:
        score += 0.4
        signals.append("Positive words with negative emoji")

    return {
        "likely_sarcastic": score >= 0.5,
        "score": round(min(score, 1.0), 2),
    }


# ─── Testing Behavior Detection ──────────────────────────────

TESTING_PATTERNS = [
    # Loyalty tests
    (r'\bwhat would you do if\b', "hypothetical_test"),
    (r'\bif (another|some|a) girl\b', "jealousy_test"),
    (r'\bdo you (still|even|really) (love|like|care)\b', "reassurance_test"),
    (r'\bwould you (still|even)\b.*\bif\b', "conditional_love_test"),
    (r'\bif I (was|were|got|became)\b', "acceptance_test"),
    (r'\bam I (pretty|beautiful|enough|fat|ugly)\b', "validation_test"),
    (r'\bdo you think (she|her|they)\b.*(pretty|hot|attractive)\b', "comparison_test"),
    (r'\bwho do you (like|love|prefer) more\b', "comparison_test"),
    (r'\byou (probably|must) (think|find|like)\b', "insecurity_probe"),
    (r'\bwhat if (we|I) (break|broke|split)\b', "commitment_test"),
    (r'\bdo you miss me\b', "reassurance_test"),
    (r'\bwhat am I to you\b', "definition_test"),
    # Russian
    (r'\bты (меня|ещё|правда) (любишь|скучаешь)\b', "reassurance_test"),
    (r'\bа если (я|другая|другой)\b', "hypothetical_test"),
    (r'\bкто тебе (больше|лучше)\b', "comparison_test"),
    (r'\bя (толстая|некрасивая|страшная|уродина|жирная)\b', "validation_test"),
    (r'\bчто я для тебя\b', "definition_test"),
    (r'\bты бы (ещё|всё равно)\b.*\bесли\b', "conditional_love_test"),
    (r'\bя (красивая|нравлюсь тебе|нужна тебе)\b\??', "validation_test"),
    (r'\bскучаешь (по мне|без меня)\b', "reassurance_test"),
    (r'\bтебе (нравится|нравилась) (она|та)\b', "comparison_test"),
    (r'\bты (думаешь|считаешь) (о ней|об? ?\w+)\b', "jealousy_test"),
    (r'\bмы (расстанемся|разойдёмся)\b', "commitment_test"),
    (r'\bкем я тебе (прихожусь|являюсь)\b', "definition_test"),
    (r'\bты (когда-нибудь|вообще) (бросишь|уйдёшь|предашь)\b', "commitment_test"),
]


def detect_testing(text: str) -> Dict[str, Any]:
    """Detect if someone is 'testing' you with their message."""
    text_lower = text.lower()
    tests_found = []

    for pattern, test_type in TESTING_PATTERNS:
        if re.search(pattern, text_lower):
            tests_found.append(test_type)

    if tests_found:
        return {
            "is_testing": True,
            "test_types": list(set(tests_found)),
            "recommended_approach": _test_response_strategy(tests_found[0]),
        }
    return {"is_testing": False, "test_types": [], "recommended_approach": None}


def _test_response_strategy(test_type: str) -> str:
    """Get strategy for responding to a specific type of test."""
    strategies = {
        "jealousy_test": "Be secure and confident. Don't get defensive. Reassure but don't over-explain.",
        "reassurance_test": "Be genuine and specific. Don't just say 'yes' — say WHY. Reference specific things about them.",
        "hypothetical_test": "Take it seriously. Don't deflect with humor. Show you've thought about it.",
        "conditional_love_test": "Reassure unconditionally. Don't add conditions back. Be absolute in your answer.",
        "acceptance_test": "Accept them completely. Be specific about what you love. Don't hesitate.",
        "validation_test": "Compliment genuinely and specifically. Reference unique things, not generic beauty. Be emphatic.",
        "comparison_test": "Make it clear there's no comparison. They're the only one. Be specific about why.",
        "insecurity_probe": "Address the insecurity directly. Don't dismiss it. Show you only have eyes for them.",
        "commitment_test": "Show commitment without panic. Be calm, sure, and clear about your feelings.",
        "definition_test": "Define the relationship warmly. Use meaningful, specific words rather than generic labels.",
    }
    return strategies.get(test_type, "Be genuine, thoughtful, and specific in your response.")


# ─── Urgency & Importance Detection ──────────────────────────

URGENCY_PATTERNS = [
    (r'\bplease\b.*\b(respond|reply|answer|call|text)\b', 0.7),
    (r'\bASAP\b', 0.8),
    (r'\burgent\b', 0.9),
    (r'\bemergency\b', 0.9),
    (r'\bneed (you|to talk|help)\b', 0.6),
    (r'\bcan we talk\b', 0.7),
    (r'\bwe need to talk\b', 0.8),
    (r'\bare you (there|ok|alive)\b', 0.5),
    (r'\bwhy (aren\'?t|arent) you (answering|replying|responding)\b', 0.8),
    (r'\bhello\b\?+', 0.5),  # "hello??" = waiting
    (r'\?{2,}', 0.4),  # multiple question marks
    (r'\!{2,}', 0.3),  # multiple exclamation marks
    (r'\b(help|помоги|пожалуйста|ответь|позвони)\b', 0.5),
    (r'\bмне (плохо|страшно|нужна помощь|очень плохо)\b', 0.8),
    (r'\bгде ты\b', 0.5),
    (r'\bпочему (не|ты не) (отвечаешь|пишешь)\b', 0.7),
    (r'\bсрочно\b', 0.8),
    (r'\bэто (важно|срочно|серьёзно)\b', 0.7),
    (r'\bмне нужн[аоы]\b', 0.5),
    (r'\bнам надо поговорить\b', 0.7),
    (r'\bты (жив[аа]?|в порядке|ок)\b\??', 0.5),
    (r'\bалло\b\??', 0.5),
]


def detect_urgency(text: str) -> Dict[str, Any]:
    """Detect urgency/importance level of a message."""
    text_lower = text.lower()
    score = 0.0

    for pattern, weight in URGENCY_PATTERNS:
        if re.search(pattern, text_lower):
            score += weight

    # All caps = more urgent
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars and len(text) > 5:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.7:
            score += 0.3

    # Multiple messages in quick succession (if available) would boost this
    # but we handle that at the caller level

    level = "normal"
    if score >= 0.8:
        level = "critical"
    elif score >= 0.5:
        level = "high"
    elif score >= 0.3:
        level = "moderate"

    return {
        "urgency_level": level,
        "score": round(min(score, 1.0), 2),
        "should_rush_reply": level in ("critical", "high"),
    }


# ─── Relationship Health Score ────────────────────────────────

def compute_relationship_health(messages: List[Dict[str, str]], memory: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a composite relationship health score (0-100) from conversation signals.

    Based on adapted Gottman's research:
    - 5:1 positive-to-negative ratio is healthy
    - Balanced initiation is good
    - Increasing message length = increasing engagement
    - Consistent emoji use = comfort
    - Questions ratio = interest
    """
    if not messages or len(messages) < 10:
        return {"score": 50, "grade": "N/A", "insufficient_data": True, "signals": {}}

    their_msgs = [m for m in messages if m["sender"] == "Them"]
    our_msgs = [m for m in messages if m["sender"] == "Me"]

    if not their_msgs or not our_msgs:
        return {"score": 50, "grade": "N/A", "insufficient_data": True, "signals": {}}

    signals = {}

    # 1. Sentiment ratio (Gottman's 5:1 adapted)
    their_pos = sum(1 for m in their_msgs if analyze_sentiment(m["text"])["sentiment"] == "positive")
    their_neg = sum(1 for m in their_msgs if analyze_sentiment(m["text"])["sentiment"] == "negative")
    their_neutral = len(their_msgs) - their_pos - their_neg

    if their_neg > 0:
        pos_neg_ratio = (their_pos + their_neutral * 0.3) / their_neg
    else:
        pos_neg_ratio = 10.0  # very positive

    # Score: 5:1 or better = 100, 3:1 = 70, 1:1 = 30, worse = 10
    if pos_neg_ratio >= 5:
        sentiment_score = 100
    elif pos_neg_ratio >= 3:
        sentiment_score = 70 + (pos_neg_ratio - 3) * 15
    elif pos_neg_ratio >= 1:
        sentiment_score = 30 + (pos_neg_ratio - 1) * 20
    else:
        sentiment_score = max(10, pos_neg_ratio * 30)
    signals["sentiment_ratio"] = {"score": round(sentiment_score), "ratio": round(pos_neg_ratio, 1)}

    # 2. Message balance (who texts more — balanced is best)
    balance_ratio = len(their_msgs) / max(len(our_msgs), 1)
    # Perfect balance = 1.0. Score drops as it deviates
    if 0.6 <= balance_ratio <= 1.5:
        balance_score = 100
    elif 0.3 <= balance_ratio <= 2.5:
        balance_score = 60
    else:
        balance_score = 30
    signals["message_balance"] = {"score": balance_score, "their_count": len(their_msgs), "our_count": len(our_msgs)}

    # 3. Engagement trend (are their messages getting longer or shorter?)
    if len(their_msgs) >= 6:
        first_half = their_msgs[:len(their_msgs)//2]
        second_half = their_msgs[len(their_msgs)//2:]
        first_avg = sum(len(m["text"]) for m in first_half) / len(first_half)
        second_avg = sum(len(m["text"]) for m in second_half) / len(second_half)

        if second_avg > first_avg * 1.2:
            engagement_score = 90  # getting more engaged
            trend = "increasing"
        elif second_avg > first_avg * 0.8:
            engagement_score = 70  # stable
            trend = "stable"
        else:
            engagement_score = 40  # declining
            trend = "declining"
    else:
        engagement_score = 60
        trend = "unknown"
    signals["engagement_trend"] = {"score": engagement_score, "trend": trend}

    # 4. Question ratio (asking questions = showing interest)
    their_questions = sum(1 for m in their_msgs if "?" in m["text"])
    question_ratio = their_questions / max(len(their_msgs), 1)
    if question_ratio > 0.3:
        question_score = 90  # they ask lots of questions
    elif question_ratio > 0.15:
        question_score = 70
    elif question_ratio > 0.05:
        question_score = 50
    else:
        question_score = 30  # they never ask questions
    signals["their_curiosity"] = {"score": question_score, "question_ratio": round(question_ratio, 2)}

    # 5. Emoji warmth
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\uFE00-\uFE0F]')
    their_emoji_msgs = sum(1 for m in their_msgs if emoji_pattern.search(m["text"]))
    emoji_ratio = their_emoji_msgs / max(len(their_msgs), 1)
    if emoji_ratio > 0.4:
        emoji_score = 90
    elif emoji_ratio > 0.2:
        emoji_score = 70
    elif emoji_ratio > 0.05:
        emoji_score = 50
    else:
        emoji_score = 40  # no emojis isn't necessarily bad
    signals["emoji_warmth"] = {"score": emoji_score, "ratio": round(emoji_ratio, 2)}

    # 6. Pet name usage (higher = more intimate/comfortable)
    pet_names = memory.get("pet_names_they_use", [])
    if len(pet_names) >= 3:
        pet_name_score = 95
    elif len(pet_names) >= 1:
        pet_name_score = 75
    else:
        pet_name_score = 50
    signals["intimacy"] = {"score": pet_name_score, "pet_names": pet_names}

    # 7. Recent conflict penalty
    last_conflict = memory.get("last_conflict")
    conflict_penalty = 0
    if last_conflict:
        try:
            conflict_dt = datetime.fromisoformat(last_conflict)
            hours_since = (datetime.now() - conflict_dt).total_seconds() / 3600
            if hours_since < 6:
                conflict_penalty = 20
            elif hours_since < 24:
                conflict_penalty = 10
            elif hours_since < 72:
                conflict_penalty = 5
        except Exception:
            pass
    signals["recent_conflict"] = {"penalty": conflict_penalty}

    # Weighted composite score
    weights = {
        "sentiment_ratio": 0.25,
        "message_balance": 0.15,
        "engagement_trend": 0.20,
        "their_curiosity": 0.15,
        "emoji_warmth": 0.10,
        "intimacy": 0.15,
    }
    composite = sum(signals[k]["score"] * w for k, w in weights.items())
    composite = max(0, min(100, composite - conflict_penalty))

    if composite >= 85:
        grade = "A"
        summary = "Relationship is thriving. Strong emotional connection, balanced engagement."
    elif composite >= 70:
        grade = "B"
        summary = "Relationship is healthy. Good communication with room for deeper connection."
    elif composite >= 55:
        grade = "C"
        summary = "Relationship is okay. Some areas could use more attention."
    elif composite >= 40:
        grade = "D"
        summary = "Relationship needs attention. Communication patterns suggest some distance."
    else:
        grade = "F"
        summary = "Relationship is struggling. Consider having an honest conversation."

    return {
        "score": round(composite),
        "grade": grade,
        "summary": summary,
        "signals": signals,
        "insufficient_data": False,
    }


# ─── Response Variation Tracking ──────────────────────────────

RESPONSE_HISTORY_DIR = Path(__file__).parent / ".response_history"
RESPONSE_HISTORY_DIR.mkdir(exist_ok=True)


def _response_history_path(chat_id: int) -> Path:
    return RESPONSE_HISTORY_DIR / f"{chat_id}.json"


def load_response_history(chat_id: int) -> List[str]:
    """Load recent responses sent to this chat."""
    path = _response_history_path(chat_id)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data.get("responses", [])
        except Exception:
            pass
    return []


def record_response(chat_id: int, response: str):
    """Record a response we sent to avoid repetition."""
    history = load_response_history(chat_id)
    history.append(response.lower().strip())
    # Keep last 50 responses
    history = history[-50:]
    path = _response_history_path(chat_id)
    try:
        path.write_text(json.dumps({"responses": history, "updated": datetime.now().isoformat()},
                                    ensure_ascii=False))
    except Exception as e:
        nlp_logger.error(f"Failed to record response for {chat_id}: {e}")


def check_response_staleness(chat_id: int, proposed: str) -> Dict[str, Any]:
    """Check if a proposed response is too similar to recent ones."""
    history = load_response_history(chat_id)
    if not history:
        return {"is_stale": False, "similarity": 0.0, "suggestion": None}

    proposed_lower = proposed.lower().strip()
    proposed_words = set(proposed_lower.split())

    max_similarity = 0.0
    most_similar = None

    # Build bigrams for proposed text
    proposed_tokens = proposed_lower.split()
    proposed_bigrams = set()
    for i in range(len(proposed_tokens) - 1):
        proposed_bigrams.add((proposed_tokens[i], proposed_tokens[i + 1]))

    for past in history:  # check ALL stored responses (up to 50)
        past_words = set(past.split())
        if not proposed_words or not past_words:
            continue
        # Jaccard word similarity
        intersection = len(proposed_words & past_words)
        union = len(proposed_words | past_words)
        word_sim = intersection / max(union, 1)

        # Bigram overlap (catches repeated phrases/structure)
        past_tokens = past.split()
        past_bigrams = set()
        for i in range(len(past_tokens) - 1):
            past_bigrams.add((past_tokens[i], past_tokens[i + 1]))
        if proposed_bigrams and past_bigrams:
            bi_inter = len(proposed_bigrams & past_bigrams)
            bi_union = len(proposed_bigrams | past_bigrams)
            bigram_sim = bi_inter / max(bi_union, 1)
        else:
            bigram_sim = 0.0

        # Combined: max of word-level and bigram-level
        similarity = max(word_sim, bigram_sim)

        # Exact substring check
        if proposed_lower in past or past in proposed_lower:
            similarity = max(similarity, 0.9)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = past

    is_stale = max_similarity > 0.55
    suggestion = None
    if is_stale:
        suggestion = "This response is too similar to something you've said recently. Try a different angle or wording."

    return {
        "is_stale": is_stale,
        "similarity": round(max_similarity, 2),
        "similar_to": most_similar[:50] if most_similar else None,
        "suggestion": suggestion,
    }


# ─── Smart Delay Calculator ──────────────────────────────────

def calculate_smart_delay(
    incoming_text: str,
    analysis: Dict[str, Any],
    base_min: int = 5,
    base_max: int = 30,
) -> Tuple[float, str]:
    """Calculate a context-aware reply delay instead of random.

    Returns (delay_seconds, reason).
    """
    # Start with base range
    delay_min = float(base_min)
    delay_max = float(base_max)
    reason = "standard delay"

    # Urgency check
    urgency = detect_urgency(incoming_text)
    if urgency["should_rush_reply"]:
        delay_min = 2
        delay_max = 8
        reason = "urgent message - fast reply"
        return (random.uniform(delay_min, delay_max), reason)

    # Question = slightly faster (shows attentiveness)
    sentiment = analysis.get("sentiment", {})
    if isinstance(sentiment, dict) and sentiment.get("is_question"):
        delay_max = min(delay_max, 20)
        reason = "question asked - faster reply"

    # Flirty = medium speed (not too eager, not too slow)
    if isinstance(sentiment, dict) and sentiment.get("is_flirty"):
        delay_min = max(delay_min, 8)
        delay_max = min(delay_max, 25)
        reason = "flirty message - medium pace"

    # Conflict = take a beat (don't react impulsively)
    stage = analysis.get("conversation_stage", "")
    if stage == "conflict":
        delay_min = max(delay_min, 15)
        delay_max = min(delay_max, 45)
        reason = "conflict - taking a moment to respond thoughtfully"

    # Long message = longer delay (shows you read it carefully)
    if len(incoming_text) > 150:
        delay_min = max(delay_min, 10)
        delay_max = min(delay_max, 35)
        reason = "long message - reading time"

    # Short casual message = faster
    if len(incoming_text) < 20 and stage not in ("conflict", "cooling_down"):
        delay_max = min(delay_max, 15)
        reason = "short casual message"

    # Hard cap — never exceed 60 seconds
    delay_min = min(delay_min, 60)
    delay_max = min(delay_max, 60)

    return (random.uniform(delay_min, delay_max), reason)


# ─── Conflict Resolution Engine ──────────────────────────────

ESCALATION_PHRASES = {
    # Phrases that escalate conflict — AVOID these
    "you always", "you never", "that's not true", "calm down",
    "you're overreacting", "it's not a big deal", "relax",
    "here we go again", "I didn't do anything", "stop being dramatic",
    "ты всегда", "ты никогда", "успокойся", "не преувеличивай",
    "опять начинается", "я ничего не делал",
}

CONFLICT_RESPONSE_STRATEGIES = {
    "conflict": [
        "Match their energy — if they're aggressive, be aggressive back",
        "Stand your ground, don't back down or apologize unless YOU actually messed up",
        "Keep it short and direct — no long explanations or essays",
        "If they're being unfair, call it out bluntly",
        "Don't be a pushover. push back if they're wrong",
        "Talk like a real person who's pissed, not a therapist",
    ],
    "silent_treatment": [
        "Don't chase or double-text — match their energy",
        "One short message max, then leave it",
        "Don't beg or plead — you have your own life",
        "If they come back, don't act like nothing happened",
    ],
    "passive_aggressive": [
        "Call it out directly — don't play along",
        "Say what you actually think, not what they want to hear",
        "Don't be gentle about it — be blunt",
        "If they're being snarky, be snarky right back",
    ],
}


def get_conflict_guidance(
    stage: str,
    sentiment: Dict[str, Any],
    pa_result: Dict[str, Any],
    incoming_text: str,
) -> Optional[Dict[str, Any]]:
    """Get specialized conflict resolution guidance."""
    # Check if we're in a conflict situation
    is_conflict = stage == "conflict"
    is_pa = pa_result.get("is_passive_aggressive", False)
    is_negative = sentiment.get("sentiment") == "negative"

    if not (is_conflict or is_pa or (is_negative and sentiment.get("intensity", 0) > 0.5)):
        return None

    guidance = {
        "in_conflict": True,
        "severity": "high" if is_conflict else "moderate" if is_pa else "low",
        "strategies": [],
        "avoid_phrases": list(ESCALATION_PHRASES)[:8],
        "opening_suggestions": [],
    }

    if is_conflict:
        guidance["strategies"] = CONFLICT_RESPONSE_STRATEGIES["conflict"]
        guidance["opening_suggestions"] = [
            "bro what is ur problem rn",
            "ok so we're doing this?? fine",
            "lol thats not what happened at all",
            "nah dont twist my words",
        ]
    elif is_pa:
        guidance["strategies"] = CONFLICT_RESPONSE_STRATEGIES["passive_aggressive"]
        guidance["opening_suggestions"] = [
            "just say what u mean",
            "u clearly have something to say so say it",
            "ok whats the problem",
        ]

    return guidance


# ─── Proactive Engagement Suggestions ────────────────────────

def get_proactive_suggestions(memory: Dict[str, Any], time_ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate proactive engagement suggestions based on memory and time context."""
    suggestions = []

    period = time_ctx["period"]
    is_weekend = time_ctx["is_weekend"]

    # Good morning suggestion
    if period == "early_morning" or period == "morning":
        lang = memory.get("their_language_preference", "english")
        if lang == "russian":
            suggestions.append({
                "type": "good_morning",
                "message": "доброе утро",
                "reason": "Morning greeting in their language",
            })
        else:
            suggestions.append({
                "type": "good_morning",
                "message": "good morning, how'd you sleep",
                "reason": "Start the day with a warm message",
            })

    # Good night suggestion
    if period == "night" or period == "late_night":
        suggestions.append({
            "type": "good_night",
            "message": "goodnight, sleep well",
            "reason": "End the day on a warm note",
        })

    # Weekend plans suggestion
    if is_weekend and period in ("morning", "midday"):
        suggestions.append({
            "type": "weekend_plans",
            "message": "what are you up to this weekend?",
            "reason": "Weekend is a good time to check in",
        })

    # Interest-based suggestions from memory
    topics = memory.get("their_topics", [])
    if "support" in topics or "emotional" in topics:
        suggestions.append({
            "type": "check_in",
            "message": "hey, just checking in on you. how are you feeling today?",
            "reason": "They've been going through something — show you care",
        })

    if "work" in topics or "career" in topics:
        suggestions.append({
            "type": "work_checkin",
            "message": "how's work going?",
            "reason": "They've talked about work recently — follow up",
        })

    if "education" in topics:
        suggestions.append({
            "type": "study_checkin",
            "message": "how's the studying going?",
            "reason": "They've been studying — check in",
        })

    # Haven't heard from them
    total_msgs = memory.get("total_messages_seen", 0)
    if total_msgs > 20:  # established conversation
        suggestions.append({
            "type": "thinking_of_you",
            "message": "just saw something that reminded me of you",
            "reason": "Re-engage naturally after a gap",
        })

    return suggestions[:3]  # max 3 suggestions


# ─── Enhanced Russian Cultural Intelligence ──────────────────

RUSSIAN_ENDEARMENTS = {
    # Tier 1: Light/casual (early dating, friends)
    "light": ["солнышко", "зайка", "котик", "малыш", "красотка"],
    # Tier 2: Warm/intimate (dating, early relationship)
    "warm": ["милая", "родная", "любимая", "сладкая", "дорогая"],
    # Tier 3: Deep/committed (serious relationship)
    "deep": ["моя", "единственная", "ненаглядная", "душа моя", "жизнь моя"],
    # Tier 4: Playful diminutives
    "playful": ["зайчонок", "котёнок", "мышонок", "пупсик", "лапочка"],
}

RUSSIAN_TEXTING_ABBREVIATIONS = {
    "спс": "спасибо",
    "пжл": "пожалуйста",
    "прив": "привет",
    "чд": "что делаешь",
    "кст": "кстати",
    "лан": "ладно",
    "нзч": "не за что",
    "оч": "очень",
    "кр": "короче",
    "скрн": "скоро",
    "ток": "только",
    "чё": "что",
    "ваще": "вообще",
}

# Russian diminutive patterns (makes words softer/cuter)
RUSSIAN_DIMINUTIVE_SUFFIXES = [
    "очк", "ечк", "оньк", "еньк", "ушк", "юшк", "ик", "чик",
]


def get_russian_context(text: str, memory: Dict[str, Any]) -> Dict[str, Any]:
    """Get Russian-specific cultural context for a message."""
    text_lower = text.lower()

    # Detect abbreviations used
    abbrevs_used = {abbr: full for abbr, full in RUSSIAN_TEXTING_ABBREVIATIONS.items()
                    if abbr in text_lower.split()}

    # Detect endearment tier
    endearment_tier = None
    endearments_found = []
    for tier, words in RUSSIAN_ENDEARMENTS.items():
        for word in words:
            if word in text_lower:
                endearment_tier = tier
                endearments_found.append(word)

    # Detect diminutives
    has_diminutives = any(suffix in text_lower for suffix in RUSSIAN_DIMINUTIVE_SUFFIXES)

    # Recommend appropriate endearment tier based on relationship
    relationship_stage = memory.get("relationship_stage", "unknown")
    if relationship_stage == "long_term":
        recommended_tier = "deep"
    elif relationship_stage in ("committed", "dating"):
        recommended_tier = "warm"
    else:
        recommended_tier = "light"

    return {
        "abbreviations_detected": abbrevs_used,
        "endearment_tier": endearment_tier,
        "endearments_found": endearments_found,
        "has_diminutives": has_diminutives,
        "recommended_endearment_tier": recommended_tier,
        "recommended_endearments": RUSSIAN_ENDEARMENTS.get(recommended_tier, []),
    }


# ─── Enhanced analyze_context with all new systems ────────────

def analyze_context_v2(
    messages: List[Dict[str, str]],
    incoming_text: str,
    chat_id: int,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """Enhanced full context analysis with all advanced systems.

    This is the V2 that adds: time awareness, PA/sarcasm/testing detection,
    urgency, relationship health, conflict resolution, and proactive suggestions.
    """
    # Run base analysis
    base = analyze_context(messages, incoming_text, chat_id, username)

    # Time context
    time_ctx = get_time_context()
    base["time_context"] = time_ctx

    # Advanced detections
    base["passive_aggression"] = detect_passive_aggression(incoming_text, messages)
    base["sarcasm"] = detect_sarcasm(incoming_text)
    base["testing"] = detect_testing(incoming_text)
    base["urgency"] = detect_urgency(incoming_text)

    # Relationship health
    memory = load_memory(chat_id)
    base["relationship_health"] = compute_relationship_health(messages, memory)

    # Conflict resolution guidance (if needed)
    conflict_guidance = get_conflict_guidance(
        base["conversation_stage"],
        base["sentiment"],
        base["passive_aggression"],
        incoming_text,
    )
    if conflict_guidance:
        base["conflict_guidance"] = conflict_guidance

    # Smart delay
    delay, delay_reason = calculate_smart_delay(incoming_text, base)
    base["smart_delay"] = {"seconds": round(delay, 1), "reason": delay_reason}

    # Response staleness check
    base["response_staleness"] = check_response_staleness(chat_id, incoming_text)

    # Russian cultural context (if Russian detected)
    if base["language"] in ("russian", "mixed"):
        base["russian_context"] = get_russian_context(incoming_text, memory)

    # Proactive suggestions
    base["proactive_suggestions"] = get_proactive_suggestions(memory, time_ctx)

    # Enhanced strategy with new signals
    base["strategy"] = _enhanced_strategy(base)

    return base


def _enhanced_strategy(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Build an enhanced strategy using all available signals."""
    # Start with base strategy
    strategy = recommend_strategy(analysis)

    # Incorporate passive-aggression
    pa = analysis.get("passive_aggression", {})
    if pa.get("is_passive_aggressive"):
        strategy["recommended_tone"] = "gentle"
        strategy["priority_action"] = "address_pa"
        strategy["notes"].append("PASSIVE-AGGRESSION DETECTED. Address it gently — ask if something is wrong. Don't match their coldness.")

    # Incorporate sarcasm
    sarcasm = analysis.get("sarcasm", {})
    if sarcasm.get("likely_sarcastic"):
        strategy["notes"].append("Likely sarcasm detected. Don't take at face value. Acknowledge the underlying feeling.")

    # Incorporate testing
    testing = analysis.get("testing", {})
    if testing.get("is_testing"):
        strategy["priority_action"] = "pass_test"
        test_types = testing.get("test_types", [])
        approach = testing.get("recommended_approach", "")
        strategy["notes"].append(f"TESTING DETECTED ({', '.join(test_types)}). {approach}")
        strategy["recommended_length"] = "medium"  # tests deserve thoughtful answers

    # Incorporate urgency
    urgency = analysis.get("urgency", {})
    if urgency.get("should_rush_reply"):
        strategy["notes"].append(f"URGENT MESSAGE (level: {urgency['urgency_level']}). Respond quickly and directly.")

    # Time-aware adjustments
    time_ctx = analysis.get("time_context", {})
    period = time_ctx.get("period", "")
    if period == "late_night":
        strategy["notes"].append("Late night conversation — be warmer, more intimate. Shorter messages feel more natural.")
        strategy["recommended_length"] = "short"
    elif period == "early_morning":
        strategy["notes"].append("Early morning — be gentle and sweet. Don't overwhelm.")
    elif period == "night" and time_ctx.get("is_weekend"):
        strategy["notes"].append("Weekend night — more relaxed and playful energy.")

    return strategy


def format_context_v2(analysis: Dict[str, Any]) -> str:
    """Format the enhanced V2 analysis into a prompt section for Claude."""
    # Start with base formatting
    parts = []

    # Time context
    time_ctx = analysis.get("time_context", {})
    if time_ctx:
        parts.append(f"Time: {time_ctx.get('period', 'unknown')} ({time_ctx.get('day_name', '')}, {time_ctx.get('vibe', '')})")

    # Base sentiment
    s = analysis.get("sentiment", {})
    if isinstance(s, dict):
        parts.append(f"Their mood: {s.get('sentiment', 'neutral')} (intensity: {s.get('intensity', 0)})")
        if s.get("is_question"):
            parts.append("They asked a question — ANSWER IT DIRECTLY")
        if s.get("is_flirty"):
            parts.append("They're being flirty — match that energy confidently")
        if s.get("is_advice_seeking"):
            parts.append("They're seeking advice — listen first, then offer genuine perspective")
        if s.get("is_sharing_news"):
            parts.append("They're sharing news — react with genuine interest and excitement")
        if s.get("is_debate"):
            parts.append("They want to discuss/debate — engage thoughtfully with your own perspective")
        if s.get("is_vent"):
            parts.append("They're venting — validate first, don't jump to solutions")

    # Passive-aggression warning
    pa = analysis.get("passive_aggression", {})
    if pa.get("is_passive_aggressive"):
        parts.append(f"⚠️ PASSIVE-AGGRESSION DETECTED (score: {pa['score']}). Don't match their coldness. Ask what's wrong gently.")

    # Sarcasm warning
    sarcasm = analysis.get("sarcasm", {})
    if sarcasm.get("likely_sarcastic"):
        parts.append("⚠️ Likely sarcastic. Address the real feeling underneath.")

    # Testing behavior
    testing = analysis.get("testing", {})
    if testing.get("is_testing"):
        types = ", ".join(testing.get("test_types", []))
        parts.append(f"⚠️ TESTING YOU ({types}). {testing.get('recommended_approach', '')}")

    # Urgency
    urgency = analysis.get("urgency", {})
    if urgency.get("should_rush_reply"):
        parts.append(f"🚨 URGENT ({urgency['urgency_level']}). Respond quickly and directly.")

    # Language
    lang = analysis.get("language", "english")
    if lang == "russian":
        parts.append("IMPORTANT: They wrote in Russian. Reply in Russian.")
        ru_ctx = analysis.get("russian_context", {})
        if ru_ctx.get("endearments_found"):
            parts.append(f"They used endearments: {', '.join(ru_ctx['endearments_found'])}. Use similar ones back.")
        recommended = ru_ctx.get("recommended_endearments", [])
        if recommended:
            parts.append(f"Suggested endearments to use: {', '.join(recommended[:3])}")
    elif lang == "mixed":
        parts.append("They're mixing Russian and English. You can use either or both.")

    # Conversation stage
    stage = analysis.get("conversation_stage", "unknown")
    stage_map = {
        "new_chat": "Fresh conversation — be chill and engaging",
        "warming_up": "Conversation is warming up — match their energy",
        "flowing": "Conversation flowing naturally — keep it going",
        "deep": "Deep meaningful conversation — match their depth",
        "conflict": "CONFLICT. Match their energy. Stand your ground. Dont be a pushover.",
        "cooling_down": "Conversation dying — ask something interesting to re-engage",
        "makeup": "Making up after conflict — be real, not fake nice",
    }
    parts.append(f"Stage: {stage_map.get(stage, stage)}")

    # Conflict guidance
    conflict = analysis.get("conflict_guidance")
    if conflict:
        parts.append(f"Conflict severity: {conflict['severity']}")
        for s_str in conflict.get("strategies", [])[:3]:
            parts.append(f"Strategy: {s_str}")
        avoid = conflict.get("avoid_phrases", [])[:4]
        if avoid:
            parts.append(f"AVOID saying: {', '.join(avoid)}")

    # Topics
    topics = analysis.get("topics", [])
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")

    # Strategy
    strat = analysis.get("strategy", {})
    parts.append(f"Recommended tone: {strat.get('recommended_tone', 'casual')}")
    parts.append(f"Recommended length: {strat.get('recommended_length', 'short')}")
    if strat.get("should_ask_question"):
        parts.append("Ask them a question back")
    if strat.get("should_use_emoji"):
        parts.append("Use 1-2 emojis naturally")
    if strat.get("should_be_vulnerable"):
        parts.append("Be genuine and vulnerable")
    if strat.get("priority_action"):
        actions = {
            "acknowledge_feelings": "PRIORITY: React to how they're feeling — be real not therapeutic",
            "show_empathy": "PRIORITY: Show you get it — react like a real person would",
            "reassure": "PRIORITY: Be straight with them, no empty promises",
            "reinforce_positive": "PRIORITY: Keep the good energy going",
            "re_engage": "PRIORITY: Re-engage with something interesting",
            "address_pa": "PRIORITY: Call out their passive-aggressive behavior directly",
            "pass_test": "PRIORITY: Respond confidently to their test",
        }
        action = strat.get("priority_action")
        parts.append(actions.get(action, f"PRIORITY: {action}"))

    for note in strat.get("notes", []):
        parts.append(f"Note: {note}")

    # Memory insights
    mem = analysis.get("memory", {})
    if mem.get("pet_names_they_use"):
        parts.append(f"They call you: {', '.join(mem['pet_names_they_use'][-3:])}")
    if mem.get("total_messages", 0) > 100:
        parts.append("Long-running conversation — be familiar and natural, reference shared history.")
    elif mem.get("total_messages", 0) > 30:
        parts.append("Established rapport — be comfortable and natural.")

    # Relationship health summary
    health = analysis.get("relationship_health", {})
    if not health.get("insufficient_data"):
        parts.append(f"Relationship health: {health.get('grade', 'N/A')} ({health.get('score', 0)}/100)")

    return "\n".join(f"- {p}" for p in parts)


# ═══════════════════════════════════════════════════════════════════
#  V3 ANALYSIS: DEEP LEARNING INTEGRATION
# ═══════════════════════════════════════════════════════════════════
# Combines heuristic (V2) + transformer models + custom neural networks
# for the most accurate, adaptive analysis possible.


def _safe_import_dl():
    """Safely import deep learning modules (may not be installed)."""
    try:
        from advanced_nlp import (
            deep_analyze,
            format_deep_analysis,
            dl_sentiment,
            dl_emotions,
            semantic_staleness_check,
            score_response_quality,
        )
        return {
            "deep_analyze": deep_analyze,
            "format_deep_analysis": format_deep_analysis,
            "dl_sentiment": dl_sentiment,
            "dl_emotions": dl_emotions,
            "semantic_staleness_check": semantic_staleness_check,
            "score_response_quality": score_response_quality,
        }
    except ImportError:
        return None


def _safe_import_neural():
    """Safely import neural network prediction modules."""
    try:
        from neural_networks import predict_with_neural
        from dl_models import get_model_manager
        return {
            "predict_with_neural": predict_with_neural,
            "get_model_manager": get_model_manager,
        }
    except ImportError:
        return None


def analyze_context_v3(
    messages: List[Dict[str, str]],
    incoming_text: str,
    chat_id: int,
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """V3 Analysis: combines heuristic + transformer + neural network signals.

    This is the most comprehensive analysis available:
    1. All V2 heuristic analysis (keywords, patterns, rules)
    2. Transformer-based sentiment & emotion (DistilBERT, DistilRoBERTa)
    3. Zero-shot intent/topic classification (NLI model)
    4. Custom neural network predictions (CNN, attention net)
    5. Semantic similarity for staleness (sentence-transformers)
    6. Conversation dynamics modeling (embeddings + LSTM signals)
    7. Adaptive confidence-weighted ensemble

    Falls back gracefully to V2 if DL models aren't available.
    """
    # Start with full V2 analysis
    base = analyze_context_v2(messages, incoming_text, chat_id, username)
    base["analysis_version"] = "v3"

    # ── Deep Learning Enhancement ──
    dl_modules = _safe_import_dl()
    neural_modules = _safe_import_neural()

    if dl_modules is None:
        base["dl_status"] = "unavailable"
        base["analysis_version"] = "v2_fallback"
        nlp_logger.info("DL modules not available, using V2 fallback")
        return base

    memory = load_memory(chat_id)

    # Run deep analysis (transformers)
    try:
        dl_analysis = dl_modules["deep_analyze"](
            messages, incoming_text, chat_id, memory
        )
        base["dl_analysis"] = dl_analysis
        base["dl_status"] = "active" if dl_analysis.get("dl_available") else "models_missing"
    except Exception as e:
        nlp_logger.warning(f"Deep analysis failed: {e}")
        base["dl_status"] = "error"
        dl_analysis = {}

    # ── Neural Network Predictions ──
    if neural_modules and dl_analysis.get("dl_available"):
        try:
            mm = neural_modules["get_model_manager"]()
            embedding = mm.embed_single(incoming_text)

            if embedding is not None:
                # Try CNN predictions
                for task in ["romantic_intent", "conversation_stage", "emotional_tone"]:
                    for model_name in ["textcnn"]:
                        pred = neural_modules["predict_with_neural"](task, embedding, model_name)
                        if pred:
                            base[f"nn_{task}_{model_name}"] = pred
        except Exception as e:
            nlp_logger.warning(f"Neural network predictions failed: {e}")

    # ── Sklearn Classifier Predictions ──
    sklearn_preds = {}
    try:
        from dl_models import get_model_manager
        mm = get_model_manager()
        for task in ["romantic_intent", "conversation_stage", "emotional_tone"]:
            pred = mm.predict_with_custom(task, incoming_text)
            if pred and pred.get("confidence", 0) > 0.3:
                sklearn_preds[task] = pred
                base[f"sklearn_{task}"] = pred
    except Exception:
        pass  # sklearn models may not be trained yet

    # ── Ensemble: Combine Heuristic + DL Signals ──
    base["ensemble"] = _build_ensemble_analysis(base, dl_analysis, sklearn_preds)

    # ── Enhanced strategy using all signals ──
    base["strategy"] = _v3_strategy(base)

    return base


def _build_ensemble_analysis(
    heuristic: Dict[str, Any],
    dl: Dict[str, Any],
    sklearn_preds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ensemble analysis combining heuristic, DL, and sklearn signals.

    Uses confidence-weighted combination:
    - Sklearn classifiers preferred when confidence is high (trained on task-specific data)
    - DL signals (zero-shot) used for broader understanding
    - Heuristic signals used as fallback or to break ties
    - Disagreements flagged for transparency
    """
    if sklearn_preds is None:
        sklearn_preds = {}
    ensemble = {"method": "ensemble"}

    # ── Sentiment Ensemble ──
    h_sent = heuristic.get("sentiment", {})
    dl_sent = dl.get("dl_sentiment", {})

    if dl_sent and dl_sent.get("confidence", 0) > 0.7:
        ensemble["sentiment"] = {
            "value": dl_sent["sentiment"],
            "confidence": dl_sent["confidence"],
            "source": "transformer",
        }
    elif dl_sent:
        h_value = h_sent.get("sentiment", "neutral") if isinstance(h_sent, dict) else "neutral"
        dl_value = dl_sent.get("sentiment", "neutral")

        if h_value == dl_value:
            ensemble["sentiment"] = {
                "value": dl_value,
                "confidence": dl_sent.get("confidence", 0.5),
                "source": "consensus",
            }
        else:
            ensemble["sentiment"] = {
                "value": dl_value,
                "confidence": dl_sent.get("confidence", 0.5) * 0.8,
                "source": "transformer_with_disagreement",
                "heuristic_says": h_value,
            }
    else:
        ensemble["sentiment"] = {
            "value": h_sent.get("sentiment", "neutral") if isinstance(h_sent, dict) else "neutral",
            "confidence": h_sent.get("intensity", 0.5) if isinstance(h_sent, dict) else 0.5,
            "source": "heuristic",
        }

    # ── Emotion Ensemble ──
    dl_emotions = dl.get("dl_emotions", {})
    if dl_emotions:
        ensemble["primary_emotion"] = {
            "value": dl_emotions.get("primary_emotion", "neutral"),
            "confidence": dl_emotions.get("primary_confidence", 0),
            "intensity": dl_emotions.get("emotional_intensity", 0),
            "source": "transformer",
        }
    else:
        h_val = h_sent.get("sentiment", "neutral") if isinstance(h_sent, dict) else "neutral"
        emotion_map = {"positive": "joy", "negative": "sadness", "neutral": "neutral"}
        ensemble["primary_emotion"] = {
            "value": emotion_map.get(h_val, "neutral"),
            "confidence": 0.3,
            "intensity": h_sent.get("intensity", 0) if isinstance(h_sent, dict) else 0,
            "source": "heuristic_mapped",
        }

    # ── Intent Ensemble ──
    sk_intent = sklearn_preds.get("romantic_intent")
    dl_intent = dl.get("dl_intent", {})

    if sk_intent and sk_intent.get("confidence", 0) > 0.6:
        # Sklearn classifier has high confidence — prefer it
        ensemble["intent"] = {
            "value": sk_intent["label"],
            "confidence": sk_intent["confidence"],
            "source": "sklearn",
        }
    elif dl_intent and dl_intent.get("confidence", 0) > 0.3:
        ensemble["intent"] = {
            "value": dl_intent["primary_intent"],
            "confidence": dl_intent["confidence"],
            "source": "zero-shot",
        }
    elif sk_intent:
        # Low-confidence sklearn still better than nothing
        ensemble["intent"] = {
            "value": sk_intent["label"],
            "confidence": sk_intent["confidence"],
            "source": "sklearn_low_conf",
        }

    # ── Conversation Stage from sklearn ──
    sk_stage = sklearn_preds.get("conversation_stage")
    if sk_stage and sk_stage.get("confidence", 0) > 0.5:
        ensemble["sklearn_stage"] = {
            "value": sk_stage["label"],
            "confidence": sk_stage["confidence"],
        }

    # ── Emotional Tone from sklearn ──
    sk_emotion = sklearn_preds.get("emotional_tone")
    if sk_emotion and sk_emotion.get("confidence", 0) > 0.5:
        ensemble["sklearn_emotion"] = {
            "value": sk_emotion["label"],
            "confidence": sk_emotion["confidence"],
        }

    # ── Topic Ensemble ──
    h_topics = heuristic.get("topics", [])
    dl_topics = dl.get("dl_topics", {})

    if dl_topics and dl_topics.get("confidence", 0) > 0.3:
        ensemble["topics"] = {
            "primary": dl_topics["primary_topic"],
            "confidence": dl_topics["confidence"],
            "heuristic_topics": h_topics,
            "source": "zero-shot",
        }
    else:
        ensemble["topics"] = {
            "primary": h_topics[0] if h_topics else "casual",
            "confidence": 0.5,
            "heuristic_topics": h_topics,
            "source": "heuristic",
        }

    # ── Dynamics ──
    dynamics = dl.get("conversation_dynamics", {})
    if dynamics:
        ensemble["dynamics"] = dynamics

    return ensemble


def _v3_strategy(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Build the most sophisticated strategy using ALL available signals."""
    strategy = _enhanced_strategy(analysis)

    ensemble = analysis.get("ensemble", {})

    # Emotion-based tone refinement
    emotion = ensemble.get("primary_emotion", {})
    emotion_val = emotion.get("value", "neutral")
    emotion_conf = emotion.get("confidence", 0)

    if emotion_conf > 0.5:
        emotion_tone_map = {
            "joy": "playful",
            "sadness": "supportive",
            "anger": "sincere",
            "fear": "supportive",
            "surprise": "playful",
            "disgust": "gentle",
            "neutral": None,
            # New emotion classes from expanded training
            "excitement": "enthusiastic",
            "frustration": "validating",
            "gratitude": "warm",
            "love": "romantic",
            "tenderness": "gentle",
            "desire": "flirty",
            "playful": "playful",
        }
        suggested_tone = emotion_tone_map.get(emotion_val)
        if suggested_tone and emotion_conf > 0.6:
            strategy["recommended_tone"] = suggested_tone
            strategy["notes"].append(
                f"[DL] Emotion detected: {emotion_val} ({emotion_conf:.0%}) -> tone: {suggested_tone}"
            )

    # Intent-based adjustments (handles both zero-shot labels AND sklearn class names)
    intent = ensemble.get("intent", {})
    intent_val = intent.get("value", "")
    intent_conf = intent.get("confidence", 0)

    if intent_conf > 0.4:
        # Map intent values to strategy adjustments
        # Supports both zero-shot labels ("flirting", "sharing emotions") and
        # sklearn class names ("flirty", "sharing", "curious", "opinion", etc.)
        if "flirt" in intent_val:  # covers "flirting" and "flirty"
            strategy["recommended_tone"] = "flirty"
            strategy["should_use_emoji"] = True
        elif "sharing emotions" in intent_val or intent_val == "sad":
            strategy["recommended_tone"] = "supportive"
            strategy["should_be_vulnerable"] = True
        elif "expressing love" in intent_val or intent_val == "romantic":
            strategy["recommended_tone"] = "romantic"
        elif "expressing anger" in intent_val or intent_val == "angry":
            strategy["recommended_tone"] = "sincere"
            strategy["priority_action"] = "acknowledge_feelings"
        elif "seeking support" in intent_val or intent_val == "supportive":
            strategy["recommended_tone"] = "supportive"
            strategy["priority_action"] = "show_empathy"
        elif "making plans" in intent_val or intent_val == "plans":
            strategy["recommended_tone"] = "casual"
            strategy["should_ask_question"] = True
        elif "testing" in intent_val:
            strategy["priority_action"] = "pass_test"
            strategy["recommended_length"] = "medium"
        elif "seeking advice" in intent_val or intent_val == "advice_seeking":
            strategy["recommended_tone"] = "helpful"
            strategy["should_ask_question"] = True
            strategy["priority_action"] = "listen_then_advise"
        elif "sharing news" in intent_val or intent_val == "sharing":
            strategy["recommended_tone"] = "enthusiastic"
            strategy["priority_action"] = "react_and_engage"
        elif "discussing" in intent_val or "debating" in intent_val:
            strategy["recommended_tone"] = "thoughtful"
            strategy["recommended_length"] = "medium"
        elif "venting" in intent_val or "complaining" in intent_val:
            strategy["recommended_tone"] = "validating"
            strategy["priority_action"] = "show_empathy"
        elif "humor" in intent_val or "joking" in intent_val or intent_val == "playful":
            strategy["recommended_tone"] = "playful"
        elif intent_val == "curious":
            strategy["recommended_tone"] = "informative"
            strategy["should_ask_question"] = False
            strategy["priority_action"] = "explain_clearly"
            strategy["recommended_length"] = "medium"
        elif intent_val == "opinion":
            strategy["recommended_tone"] = "thoughtful"
            strategy["should_ask_question"] = True
            strategy["priority_action"] = "engage_with_opinion"
        elif intent_val == "sincere":
            strategy["recommended_tone"] = "sincere"
            strategy["should_be_vulnerable"] = True
        elif intent_val == "casual":
            strategy["recommended_tone"] = "casual"
        elif intent_val in ("greeting", "goodbye"):
            strategy["recommended_tone"] = "warm"

        strategy["notes"].append(f"[DL] Intent: {intent_val} ({intent_conf:.0%})")

    # Dynamics-based adjustments
    dynamics = ensemble.get("dynamics", {})
    momentum = dynamics.get("momentum", "stable")
    trajectory = dynamics.get("emotional_trajectory", "stable")

    if momentum == "decelerating":
        strategy["should_ask_question"] = True
        strategy["notes"].append("[DL] Conversation momentum declining - re-engage actively")
    elif momentum == "accelerating":
        strategy["notes"].append("[DL] Conversation momentum building - ride the wave")

    if trajectory == "declining":
        strategy["notes"].append("[DL] Emotional trajectory declining - be more supportive")
    elif trajectory == "improving":
        strategy["notes"].append("[DL] Emotional trajectory improving - reinforce positivity")

    return strategy


def format_context_v3(analysis: Dict[str, Any]) -> str:
    """Format V3 analysis (heuristic + DL) into a prompt section for Claude."""
    v2_text = format_context_v2(analysis)

    dl_modules = _safe_import_dl()
    if dl_modules and analysis.get("dl_analysis"):
        dl_text = dl_modules["format_deep_analysis"](analysis["dl_analysis"])
        if dl_text:
            v2_text += "\n" + dl_text

    ensemble = analysis.get("ensemble", {})
    if ensemble:
        parts = []

        ens_sent = ensemble.get("sentiment", {})
        if ens_sent.get("source") not in (None, "heuristic"):
            parts.append(
                f"[Ensemble] Sentiment: {ens_sent['value']} "
                f"(confidence: {ens_sent.get('confidence', 0):.0%}, source: {ens_sent['source']})"
            )

        ens_emo = ensemble.get("primary_emotion", {})
        if ens_emo.get("source") != "heuristic_mapped":
            parts.append(
                f"[Ensemble] Emotion: {ens_emo['value']} "
                f"(intensity: {ens_emo.get('intensity', 0):.0%})"
            )

        if parts:
            v2_text += "\n" + "\n".join(f"- {p}" for p in parts)

    return v2_text


def score_response_v3(
    proposed_response: str,
    their_last_message: str,
    messages: List[Dict[str, str]],
    chat_id: int,
) -> Dict[str, Any]:
    """Score a proposed response using all available signals (V3)."""
    dl_modules = _safe_import_dl()
    if dl_modules:
        try:
            return dl_modules["score_response_quality"](
                proposed_response, their_last_message, messages
            )
        except Exception as e:
            nlp_logger.warning(f"DL response scoring failed: {e}")

    score = 100
    feedback = []
    if len(proposed_response) > 500:
        score -= 15
        feedback.append("Response is very long.")
    formal_words = ["therefore", "however", "furthermore", "nevertheless"]
    if any(w in proposed_response.lower() for w in formal_words):
        score -= 10
        feedback.append("Response sounds too formal.")
    if not feedback:
        feedback.append("Response looks natural.")
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return {"score": score, "grade": grade, "feedback": feedback, "method": "heuristic"}


def check_staleness_v3(chat_id: int, proposed: str) -> Dict[str, Any]:
    """Check response staleness using semantic similarity when available."""
    dl_modules = _safe_import_dl()
    if dl_modules:
        try:
            history = load_response_history(chat_id)
            if history:
                return dl_modules["semantic_staleness_check"](proposed, history)
        except Exception as e:
            nlp_logger.warning(f"Semantic staleness check failed: {e}")
    return check_response_staleness(chat_id, proposed)
