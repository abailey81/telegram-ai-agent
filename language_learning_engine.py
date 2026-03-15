"""
Language Learning Engine — Semantic Self-Awareness & Conversation Learning

Two core systems:

1. SEMANTIC SELF-AWARENESS — The bot understands what it's saying:
   - Vocabulary tracking: what words/phrases has it used, how often, with what effect
   - Context-relevance scoring: are the words appropriate for this conversation state
   - Register coherence: word choice matches tone/formality
   - Semantic field awareness: words used belong to the right domain
   - Response coherence audit: every part of the reply connects to context

2. CONVERSATION LEARNING — The bot learns how to speak from every interaction:
   - Vocabulary absorption: learns new words/phrases from them
   - Effective phrase library: tracks which phrases get positive reactions
   - Anti-pattern library: proactively learns what NOT to say
   - Context-vocabulary mapping: learns which words work in which contexts
   - Style effectiveness tracking: learns which style adaptations actually help
   - Speech pattern evolution: tracks how speech evolves over time
"""

import json
import re
import time
import math
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger("language_learning")

# Persistent storage
LANG_DATA_DIR = Path(__file__).parent / "engine_data" / "language_learning"
LANG_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. VOCABULARY TRACKER — Knows what words have been used
# ═══════════════════════════════════════════════════════════════

class VocabularyTracker:
    """Tracks vocabulary usage across conversations — both ours and theirs."""

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self._file = LANG_DATA_DIR / f"{chat_id}_vocabulary.json"
        self.our_words: Counter = Counter()      # word -> count
        self.their_words: Counter = Counter()     # word -> count
        self.our_phrases: Counter = Counter()     # 2-3 word phrase -> count
        self.their_phrases: Counter = Counter()
        self.word_effectiveness: Dict[str, Dict[str, float]] = {}  # word -> {positive_ct, negative_ct, neutral_ct}
        self.phrase_effectiveness: Dict[str, Dict[str, float]] = {}
        self.overused_words: Set[str] = set()     # words we use too often
        self.learned_words: Set[str] = set()       # words we picked up from them
        self.total_our_messages = 0
        self.total_their_messages = 0
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                self.our_words = Counter(data.get("our_words", {}))
                self.their_words = Counter(data.get("their_words", {}))
                self.our_phrases = Counter(data.get("our_phrases", {}))
                self.their_phrases = Counter(data.get("their_phrases", {}))
                self.word_effectiveness = data.get("word_effectiveness", {})
                self.phrase_effectiveness = data.get("phrase_effectiveness", {})
                self.overused_words = set(data.get("overused_words", []))
                self.learned_words = set(data.get("learned_words", []))
                self.total_our_messages = data.get("total_our_messages", 0)
                self.total_their_messages = data.get("total_their_messages", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        data = {
            "our_words": dict(self.our_words.most_common(500)),
            "their_words": dict(self.their_words.most_common(500)),
            "our_phrases": dict(self.our_phrases.most_common(300)),
            "their_phrases": dict(self.their_phrases.most_common(300)),
            "word_effectiveness": dict(list(self.word_effectiveness.items())[:300]),
            "phrase_effectiveness": dict(list(self.phrase_effectiveness.items())[:200]),
            "overused_words": list(self.overused_words)[:100],
            "learned_words": list(self.learned_words)[:200],
            "total_our_messages": self.total_our_messages,
            "total_their_messages": self.total_their_messages,
        }
        try:
            self._file.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        except Exception:
            pass

    def record_our_message(self, text: str):
        """Record words and phrases from a message we sent."""
        words = _extract_content_words(text)
        phrases = _extract_phrases(text)
        self.our_words.update(words)
        self.our_phrases.update(phrases)
        self.total_our_messages += 1
        self._detect_overuse()

    def record_their_message(self, text: str):
        """Record words and phrases from their message."""
        words = _extract_content_words(text)
        phrases = _extract_phrases(text)
        self.their_words.update(words)
        self.their_phrases.update(phrases)
        self.total_their_messages += 1
        self._detect_new_vocabulary(words, phrases)

    def record_effectiveness(self, our_text: str, outcome: str):
        """Record whether our words/phrases had positive, negative, or neutral effect.
        outcome: 'positive', 'negative', 'neutral'
        """
        words = _extract_content_words(our_text)
        phrases = _extract_phrases(our_text)

        for w in words:
            if w not in self.word_effectiveness:
                self.word_effectiveness[w] = {"positive": 0, "negative": 0, "neutral": 0}
            self.word_effectiveness[w][outcome] = self.word_effectiveness[w].get(outcome, 0) + 1

        for p in phrases:
            if p not in self.phrase_effectiveness:
                self.phrase_effectiveness[p] = {"positive": 0, "negative": 0, "neutral": 0}
            self.phrase_effectiveness[p][outcome] = self.phrase_effectiveness[p].get(outcome, 0) + 1

        self._save()

    def _detect_overuse(self):
        """Detect words we're using too frequently relative to our total vocabulary."""
        if self.total_our_messages < 10:
            return

        total_word_usage = sum(self.our_words.values())
        if total_word_usage == 0:
            return

        vocab_size = len(self.our_words)
        if vocab_size == 0:
            return

        # Expected frequency if uniform distribution
        expected_freq = total_word_usage / vocab_size

        self.overused_words.clear()
        for word, count in self.our_words.most_common(50):
            # A word is overused if it appears 3x+ more than expected
            # AND has been used in more than 20% of our messages
            msg_frequency = count / max(self.total_our_messages, 1)
            if count > expected_freq * 3 and msg_frequency > 0.2:
                # But don't flag common connectors
                if word not in _COMMON_CONNECTORS:
                    self.overused_words.add(word)

    def _detect_new_vocabulary(self, their_words: List[str], their_phrases: List[str]):
        """Detect words they use that we haven't — potential vocabulary to learn."""
        for w in their_words:
            if w not in self.our_words and w not in _COMMON_CONNECTORS:
                # They used a word we never have
                if self.their_words[w] >= 2:
                    # They use it regularly — worth learning
                    self.learned_words.add(w)

    def get_overused_words(self, top_n: int = 10) -> List[str]:
        """Get words we overuse — should be avoided or rotated."""
        return list(self.overused_words)[:top_n]

    def get_effective_phrases(self, min_uses: int = 3) -> List[Tuple[str, float]]:
        """Get phrases ranked by effectiveness (positive / total ratio)."""
        results = []
        for phrase, stats in self.phrase_effectiveness.items():
            total = stats.get("positive", 0) + stats.get("negative", 0) + stats.get("neutral", 0)
            if total >= min_uses:
                score = stats.get("positive", 0) / total
                results.append((phrase, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]

    def get_toxic_phrases(self, min_uses: int = 2) -> List[str]:
        """Get phrases that consistently get negative reactions."""
        toxic = []
        for phrase, stats in self.phrase_effectiveness.items():
            total = stats.get("positive", 0) + stats.get("negative", 0) + stats.get("neutral", 0)
            if total >= min_uses:
                neg_ratio = stats.get("negative", 0) / total
                if neg_ratio > 0.5:
                    toxic.append(phrase)
        return toxic[:15]

    def get_learned_vocabulary(self) -> List[str]:
        """Get words we've absorbed from them."""
        return list(self.learned_words)[:20]

    def get_vocabulary_richness(self) -> Dict[str, Any]:
        """Measure our vocabulary diversity over time."""
        if self.total_our_messages < 5:
            return {"status": "insufficient_data"}

        total_usage = sum(self.our_words.values())
        unique_words = len(self.our_words)

        ttr = unique_words / max(total_usage, 1)  # Type-Token Ratio
        hapax = sum(1 for w, c in self.our_words.items() if c == 1)
        hapax_ratio = hapax / max(unique_words, 1)

        return {
            "type_token_ratio": round(ttr, 3),
            "unique_words": unique_words,
            "total_usage": total_usage,
            "hapax_ratio": round(hapax_ratio, 3),  # words used only once
            "overused_count": len(self.overused_words),
            "richness": (
                "very_rich" if ttr > 0.7 else
                "rich" if ttr > 0.5 else
                "moderate" if ttr > 0.3 else
                "repetitive"
            ),
        }


# ═══════════════════════════════════════════════════════════════
#  2. SEMANTIC SELF-AWARENESS — Understands what it's saying
# ═══════════════════════════════════════════════════════════════

# Semantic fields: groups of words that belong together
_SEMANTIC_FIELDS = {
    "emotion_positive": {
        "en": {"happy", "glad", "excited", "amazing", "wonderful", "awesome", "great",
               "love", "adore", "fantastic", "brilliant", "incredible", "thrilled", "joyful",
               "delighted", "ecstatic", "cheerful", "blissful", "euphoric", "elated"},
        "ru": {"счастлив", "рад", "восторг", "удивительн", "прекрасн", "круто", "отличн",
               "люблю", "обожаю", "фантастич", "блестящ", "невероятн", "ликую", "радостн",
               "восхищ", "эйфори", "весёл", "блаженн"},
    },
    "emotion_negative": {
        "en": {"sad", "angry", "upset", "frustrated", "annoyed", "depressed", "miserable",
               "furious", "devastated", "heartbroken", "disappointed", "bitter", "hurt",
               "lonely", "anxious", "worried", "stressed", "exhausted", "overwhelmed"},
        "ru": {"грустн", "злой", "расстроен", "раздражён", "достал", "депресс", "несчастн",
               "взбешён", "опустошён", "разбит", "разочаров", "обижен", "одинок",
               "тревожн", "волнуюсь", "стресс", "измотан", "перегружен"},
    },
    "casual_social": {
        "en": {"hey", "sup", "yo", "lol", "lmao", "tbh", "ngl", "fr", "bruh", "dude",
               "omg", "imo", "btw", "wyd", "hmu", "smh", "lowkey", "highkey", "fam",
               "vibe", "chill", "mood", "slay", "bet", "no cap", "deadass", "sus"},
        "ru": {"привет", "здаров", "чё", "лол", "кек", "тбх", "чел", "братан", "бля",
               "омг", "кста", "кстати", "ваще", "норм", "зашиб", "топ", "огонь",
               "вайб", "чилл", "кайф", "жиза", "база", "рофл", "лан"},
    },
    "formal_academic": {
        "en": {"therefore", "furthermore", "consequently", "nevertheless", "regarding",
               "subsequently", "moreover", "accordingly", "thus", "hence", "indeed",
               "notably", "specifically", "essentially", "fundamentally", "inherently"},
        "ru": {"следовательно", "более того", "вследствие", "тем не менее", "относительно",
               "впоследствии", "кроме того", "соответственно", "таким образом",
               "действительно", "в частности", "по существу", "безусловно"},
    },
    "romantic": {
        "en": {"darling", "babe", "baby", "honey", "sweetheart", "gorgeous", "beautiful",
               "handsome", "miss you", "love you", "can't wait", "thinking of you",
               "dream about", "heart", "kiss", "hug", "cuddle", "forever"},
        "ru": {"дорогая", "дорогой", "малыш", "солнце", "красавица", "красавчик",
               "скучаю", "люблю", "не могу ждать", "думаю о тебе", "мечтаю",
               "сердце", "поцелуй", "обнимашки", "навсегда", "родной", "родная",
               "зайка", "котик", "милый", "милая"},
    },
    "conflict": {
        "en": {"wrong", "fault", "blame", "unfair", "lied", "cheated", "betrayed",
               "selfish", "toxic", "manipulate", "gaslighting", "insensitive",
               "disrespectful", "ungrateful", "hypocrite"},
        "ru": {"неправ", "виноват", "несправедлив", "соврал", "обманул", "предал",
               "эгоист", "токсичн", "манипулир", "газлайт", "бесчувствен",
               "неуважен", "неблагодарн", "лицемер"},
    },
    "support_comfort": {
        "en": {"here for you", "understand", "must be hard", "take your time",
               "not alone", "believe in you", "got this", "it's okay", "it'll pass",
               "lean on me", "no pressure", "whenever you're ready"},
        "ru": {"я рядом", "понимаю", "тяжело", "не торопись", "не одна", "не один",
               "верю в тебя", "справишься", "всё будет хорошо", "пройдёт",
               "обопрись на меня", "без давления", "когда будешь готов"},
    },
}

# Register levels (formality)
_REGISTER_LEVELS = {
    "very_casual": 0,
    "casual": 1,
    "neutral": 2,
    "formal": 3,
    "very_formal": 4,
}


def assess_semantic_coherence(
    reply_text: str,
    incoming_text: str,
    conversation_stage: str = "unknown",
    emotional_temperature: str = "neutral",
    their_formality: str = "casual",
    nlp_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Deep semantic analysis of our reply — do we actually understand what we're saying?

    Returns a comprehensive assessment:
    - semantic_field_match: are our words from the right semantic domain?
    - register_coherence: does formality match the conversation?
    - context_relevance: are we on-topic?
    - word_appropriateness: are individual words appropriate here?
    - coherence_score: overall 0.0-1.0
    - issues: list of specific problems found
    - suggestions: actionable fixes
    """
    issues = []
    suggestions = []
    scores = {}

    reply_lower = reply_text.lower()
    incoming_lower = incoming_text.lower()

    reply_words = set(_extract_content_words(reply_text))
    incoming_words = set(_extract_content_words(incoming_text))

    # Detect language
    is_russian = any('\u0400' <= c <= '\u04ff' for c in reply_text)
    lang_key = "ru" if is_russian else "en"

    # ── 1. Semantic field analysis ──
    # What semantic domains are we operating in?
    our_fields = _identify_semantic_fields(reply_words, lang_key)
    their_fields = _identify_semantic_fields(incoming_words, lang_key)

    # Check for field mismatches
    field_match_score = 1.0
    if their_fields and our_fields:
        # If they're in "conflict" and we're in "casual_social" — bad mismatch
        if "conflict" in their_fields and "casual_social" in our_fields:
            field_match_score -= 0.4
            issues.append("using_casual_language_during_conflict")
            suggestions.append("Match the seriousness of the conversation — drop the slang")

        # If they're emotional negative and we're casual — dismissive
        if "emotion_negative" in their_fields and "casual_social" in our_fields:
            field_match_score -= 0.3
            issues.append("casual_response_to_emotional_pain")
            suggestions.append("They're hurting — respond with empathy, not 'lol'")

        # If they're casual and we're formal — robotic
        if "casual_social" in their_fields and "formal_academic" in our_fields:
            field_match_score -= 0.3
            issues.append("formal_language_in_casual_chat")
            suggestions.append("Too formal — talk like a human, not a professor")

        # Positive alignment: matching fields
        overlap = our_fields & their_fields
        if overlap:
            field_match_score = min(field_match_score + 0.1 * len(overlap), 1.0)

    scores["semantic_field"] = max(field_match_score, 0.0)

    # ── 2. Register coherence ──
    our_register = _detect_register(reply_text, lang_key)
    their_register_level = _REGISTER_LEVELS.get(their_formality, 2)
    our_register_level = _REGISTER_LEVELS.get(our_register, 2)

    register_diff = abs(our_register_level - their_register_level)
    register_score = max(1.0 - register_diff * 0.25, 0.0)

    if register_diff >= 2:
        issues.append(f"register_mismatch_{our_register}_vs_{their_formality}")
        if our_register_level > their_register_level:
            suggestions.append("Way too formal — loosen up, match their casual vibe")
        else:
            suggestions.append("Too casual for the conversation tone — add some weight")

    scores["register"] = register_score

    # ── 3. Emotional temperature alignment ──
    temp_score = 1.0
    if emotional_temperature in ("boiling", "hot_negative"):
        # They're heated — we shouldn't be cheerful
        if "emotion_positive" in our_fields and "support_comfort" not in our_fields:
            temp_score -= 0.3
            issues.append("positive_words_during_heated_moment")
            suggestions.append("They're upset — don't be cheerful, acknowledge the tension")
    elif emotional_temperature == "cold":
        # They're cold — being overly warm can feel desperate
        if "romantic" in our_fields:
            temp_score -= 0.2
            issues.append("romantic_language_when_they_are_cold")
            suggestions.append("They're being distant — don't chase with affection")

    scores["temperature_alignment"] = max(temp_score, 0.0)

    # ── 4. Conversation stage appropriateness ──
    stage_score = 1.0
    if conversation_stage in ("conflict", "de_escalation"):
        # During conflict — no jokes, no deflection
        humor_words = _SEMANTIC_FIELDS["casual_social"][lang_key]
        humor_in_reply = reply_words & humor_words
        if len(humor_in_reply) > 1:
            stage_score -= 0.3
            issues.append("humor_during_conflict")
            suggestions.append("Not the time for jokes — address the issue directly")
    elif conversation_stage == "greeting":
        # Greeting — shouldn't dive into heavy topics
        heavy_words = _SEMANTIC_FIELDS["conflict"][lang_key] | _SEMANTIC_FIELDS["emotion_negative"][lang_key]
        heavy_in_reply = reply_words & heavy_words
        if len(heavy_in_reply) > 2:
            stage_score -= 0.2
            issues.append("heavy_topics_during_greeting")

    scores["stage_appropriateness"] = max(stage_score, 0.0)

    # ── 5. Response length appropriateness ──
    reply_word_count = len(reply_text.split())
    incoming_word_count = len(incoming_text.split())
    length_ratio = reply_word_count / max(incoming_word_count, 1)

    length_score = 1.0
    if incoming_word_count <= 3 and reply_word_count > 30:
        length_score -= 0.3
        issues.append("over_response_to_short_message")
        suggestions.append("They said very little — keep your reply short too")
    elif incoming_word_count > 30 and reply_word_count <= 3:
        length_score -= 0.2
        issues.append("under_response_to_long_message")
        suggestions.append("They wrote a lot — acknowledge it with more than a word")
    elif length_ratio > 4.0:
        length_score -= 0.2
        issues.append("reply_much_longer_than_input")

    scores["length_appropriateness"] = max(length_score, 0.0)

    # ── 6. Internal consistency ──
    # Check if the reply contradicts itself
    consistency_score = 1.0
    _contradiction_pairs_en = [
        (r"\byes\b.*\bno\b", r"\bno\b.*\byes\b"),
        (r"\bi agree\b.*\bbut i don't\b",),
        (r"\bi love\b.*\bi hate\b",),
    ]
    _contradiction_pairs_ru = [
        (r"\bда\b.*\bнет\b", r"\bнет\b.*\bда\b"),
        (r"\bсогласен\b.*\bне согласен\b",),
        (r"\bлюблю\b.*\bненавижу\b",),
    ]
    pairs = _contradiction_pairs_ru if is_russian else _contradiction_pairs_en
    for pair_group in pairs:
        for pat in pair_group:
            if re.search(pat, reply_lower):
                consistency_score -= 0.15
                issues.append("internal_contradiction")
                break

    scores["consistency"] = max(consistency_score, 0.0)

    # ── 7. AI-pattern detection (expanded) ──
    ai_score = 1.0
    _ai_patterns = [
        # Therapy-speak
        r"\b(?:i hear you|i validate|that must be|i can only imagine)\b",
        r"\b(?:я слышу тебя|я подтверждаю|должно быть тяжело)\b",
        # Robot phrasing
        r"\b(?:i want you to know that|i need you to understand|let me assure you)\b",
        r"\b(?:хочу чтобы ты знал|мне нужно чтобы ты понял|позволь заверить)\b",
        # Excessive hedging
        r"\b(?:i think maybe perhaps|it seems like it could be|i feel like maybe)\b",
        # Formulaic responses
        r"\b(?:that's(?:\s+(?:so|really|very))?\s+(?:interesting|great|amazing|wonderful|fantastic))\b",
        r"\b(?:это\s+(?:так|очень|действительно)\s+(?:интересно|здорово|замечательно|прекрасно))\b",
        # List-making in casual chat
        r"(?:1\)|first(?:ly)?|secondly|thirdly|in conclusion)",
        r"(?:во-первых|во-вторых|в-третьих|в заключение)",
    ]
    ai_hits = 0
    for pat in _ai_patterns:
        if re.search(pat, reply_lower, re.IGNORECASE):
            ai_hits += 1
    if ai_hits > 0:
        ai_score -= ai_hits * 0.15
        issues.append(f"ai_patterns_detected_{ai_hits}")
        suggestions.append("Sounds too robotic — rephrase in natural conversational language")

    scores["naturalness"] = max(ai_score, 0.0)

    # ── Composite score ──
    weights = {
        "semantic_field": 0.20,
        "register": 0.15,
        "temperature_alignment": 0.15,
        "stage_appropriateness": 0.15,
        "length_appropriateness": 0.10,
        "consistency": 0.10,
        "naturalness": 0.15,
    }
    coherence_score = sum(scores[k] * weights[k] for k in weights)

    return {
        "coherence_score": round(coherence_score, 3),
        "passed": coherence_score >= 0.6,
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "issues": issues,
        "suggestions": suggestions,
        "our_semantic_fields": list(our_fields) if our_fields else [],
        "their_semantic_fields": list(their_fields) if their_fields else [],
        "our_register": our_register,
    }


# ═══════════════════════════════════════════════════════════════
#  3. CONVERSATION LEARNING ENGINE — Learns how to speak
# ═══════════════════════════════════════════════════════════════

class ConversationLearner:
    """
    Learns speech patterns, effective language, and communication style
    from every interaction. Gets smarter over time.
    """

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self._file = LANG_DATA_DIR / f"{chat_id}_learning.json"
        self.vocab_tracker = VocabularyTracker(chat_id)

        # Effective language patterns (what works)
        self.winning_patterns: List[Dict[str, Any]] = []   # [{pattern, context, score, count}]
        # Anti-patterns (what to avoid)
        self.losing_patterns: List[Dict[str, Any]] = []
        # Context-vocabulary map: which words work in which contexts
        self.context_vocab: Dict[str, Dict[str, float]] = {}  # context -> {word: effectiveness}
        # Style adaptation effectiveness
        self.style_experiments: List[Dict[str, Any]] = []
        # Their speech patterns we should mirror
        self.their_patterns: Dict[str, Any] = {
            "greeting_style": [],       # how they greet
            "farewell_style": [],       # how they say bye
            "agreement_style": [],      # how they say yes
            "disagreement_style": [],   # how they say no
            "emotional_expressions": [],  # how they express emotions
            "humor_style": [],          # how they joke
            "affection_style": [],      # how they show affection
            "filler_words": [],         # their filler words
            "sentence_starters": [],    # how they start sentences
        }
        # Evolution tracking
        self.learning_snapshots: List[Dict[str, Any]] = []
        self.total_interactions = 0
        self.last_updated = 0

        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                self.winning_patterns = data.get("winning_patterns", [])
                self.losing_patterns = data.get("losing_patterns", [])
                self.context_vocab = data.get("context_vocab", {})
                self.style_experiments = data.get("style_experiments", [])
                self.their_patterns = {**self.their_patterns, **data.get("their_patterns", {})}
                self.learning_snapshots = data.get("learning_snapshots", [])
                self.total_interactions = data.get("total_interactions", 0)
                self.last_updated = data.get("last_updated", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        data = {
            "winning_patterns": self.winning_patterns[-100:],
            "losing_patterns": self.losing_patterns[-50:],
            "context_vocab": {k: dict(list(v.items())[:50]) for k, v in list(self.context_vocab.items())[:20]},
            "style_experiments": self.style_experiments[-50:],
            "their_patterns": self.their_patterns,
            "learning_snapshots": self.learning_snapshots[-50:],
            "total_interactions": self.total_interactions,
            "last_updated": time.time(),
        }
        try:
            self._file.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        except Exception:
            pass

    def learn_from_exchange(
        self,
        our_message: str,
        their_response: str,
        outcome: str,
        context: Dict[str, Any],
    ):
        """
        Core learning function — called after every exchange.

        our_message: what we sent
        their_response: what they replied (or empty if no reply)
        outcome: 'positive', 'negative', 'neutral' (from reward signals)
        context: {conversation_stage, emotional_temperature, formality, ...}
        """
        self.total_interactions += 1

        # 1. Track vocabulary
        self.vocab_tracker.record_our_message(our_message)
        if their_response:
            self.vocab_tracker.record_their_message(their_response)
        self.vocab_tracker.record_effectiveness(our_message, outcome)

        # 2. Extract and learn patterns
        our_phrases = _extract_phrases(our_message)
        context_key = self._make_context_key(context)

        if outcome == "positive":
            for phrase in our_phrases:
                self._record_winning_pattern(phrase, context_key)
                self._update_context_vocab(context_key, phrase, 1.0)
        elif outcome == "negative":
            for phrase in our_phrases:
                self._record_losing_pattern(phrase, context_key)
                self._update_context_vocab(context_key, phrase, -1.0)

        # 3. Learn their speech patterns
        if their_response:
            self._analyze_their_speech(their_response, context)

        # 4. Snapshot every 50 interactions
        if self.total_interactions % 50 == 0:
            self._take_snapshot()

        self._save()

    def _record_winning_pattern(self, phrase: str, context_key: str):
        """Record a phrase that got a positive response."""
        # Check if we already have this pattern
        for p in self.winning_patterns:
            if p["pattern"] == phrase and p["context"] == context_key:
                p["count"] = p.get("count", 0) + 1
                p["score"] = min(p.get("score", 0.5) + 0.1, 1.0)
                return

        self.winning_patterns.append({
            "pattern": phrase,
            "context": context_key,
            "score": 0.6,
            "count": 1,
            "timestamp": time.time(),
        })

        # Keep bounded
        if len(self.winning_patterns) > 100:
            self.winning_patterns.sort(key=lambda x: x["score"] * x["count"], reverse=True)
            self.winning_patterns = self.winning_patterns[:100]

    def _record_losing_pattern(self, phrase: str, context_key: str):
        """Record a phrase that got a negative response."""
        for p in self.losing_patterns:
            if p["pattern"] == phrase:
                p["count"] = p.get("count", 0) + 1
                return

        self.losing_patterns.append({
            "pattern": phrase,
            "context": context_key,
            "count": 1,
            "timestamp": time.time(),
        })

        if len(self.losing_patterns) > 50:
            self.losing_patterns.sort(key=lambda x: x["count"], reverse=True)
            self.losing_patterns = self.losing_patterns[:50]

    def _update_context_vocab(self, context_key: str, phrase: str, delta: float):
        """Update the context-vocabulary effectiveness map."""
        if context_key not in self.context_vocab:
            self.context_vocab[context_key] = {}

        words = phrase.split()
        for word in words:
            if word in _COMMON_CONNECTORS:
                continue
            current = self.context_vocab[context_key].get(word, 0.0)
            # Exponential moving average
            self.context_vocab[context_key][word] = current * 0.8 + delta * 0.2

    def _analyze_their_speech(self, text: str, context: Dict[str, Any]):
        """Learn their speech patterns from their messages."""
        text_lower = text.lower().strip()
        words = text_lower.split()

        if not words:
            return

        stage = context.get("conversation_stage", "unknown")

        # Learn greeting style
        _greetings_en = {"hey", "hi", "hello", "sup", "yo", "heyyy", "hiii", "heyy", "what's up", "wassup"}
        _greetings_ru = {"привет", "здаров", "здарова", "приветик", "хай", "хей", "даров", "здравствуй"}
        if any(g in text_lower for g in _greetings_en | _greetings_ru):
            # Extract just the greeting portion
            greeting = text_lower.split("!")[0].split(",")[0].split(".")[0][:30]
            if greeting and greeting not in self.their_patterns["greeting_style"]:
                self.their_patterns["greeting_style"].append(greeting)
                self.their_patterns["greeting_style"] = self.their_patterns["greeting_style"][-10:]

        # Learn farewell style
        _farewells_en = {"bye", "goodnight", "gn", "night", "gotta go", "ttyl", "cya", "see ya", "later"}
        _farewells_ru = {"пока", "спокойной ночи", "споки", "доброй ночи", "давай", "до связи", "бб"}
        if any(f in text_lower for f in _farewells_en | _farewells_ru):
            farewell = text_lower[:40]
            if farewell not in self.their_patterns["farewell_style"]:
                self.their_patterns["farewell_style"].append(farewell)
                self.their_patterns["farewell_style"] = self.their_patterns["farewell_style"][-10:]

        # Learn agreement/disagreement style
        _agree_en = {"yes", "yeah", "yep", "yea", "mhm", "totally", "exactly", "for sure", "definitely"}
        _agree_ru = {"да", "ага", "угу", "точно", "конечно", "однозначно", "именно", "ещё бы"}
        if words[0] in _agree_en | _agree_ru:
            agree = " ".join(words[:3])
            if agree not in self.their_patterns["agreement_style"]:
                self.their_patterns["agreement_style"].append(agree)
                self.their_patterns["agreement_style"] = self.their_patterns["agreement_style"][-10:]

        _disagree_en = {"no", "nah", "nope", "not really", "idk", "i don't think so"}
        _disagree_ru = {"нет", "неа", "не", "не думаю", "не совсем", "хз", "не знаю"}
        if words[0] in _disagree_en | _disagree_ru:
            disagree = " ".join(words[:3])
            if disagree not in self.their_patterns["disagreement_style"]:
                self.their_patterns["disagreement_style"].append(disagree)
                self.their_patterns["disagreement_style"] = self.their_patterns["disagreement_style"][-10:]

        # Learn filler words
        _fillers_en = {"like", "well", "so", "honestly", "basically", "literally", "actually", "anyway"}
        _fillers_ru = {"ну", "типа", "короче", "вообще", "блин", "слушай", "смотри", "ваще", "кста"}
        for w in words:
            if w in _fillers_en | _fillers_ru:
                if w not in self.their_patterns["filler_words"]:
                    self.their_patterns["filler_words"].append(w)
                    self.their_patterns["filler_words"] = self.their_patterns["filler_words"][-15:]

        # Learn sentence starters
        if len(words) >= 2:
            starter = " ".join(words[:2])
            if starter not in self.their_patterns["sentence_starters"]:
                self.their_patterns["sentence_starters"].append(starter)
                self.their_patterns["sentence_starters"] = self.their_patterns["sentence_starters"][-20:]

        # Learn emotional expressions
        _emo_patterns = re.findall(r'[!?]{2,}|[.]{3,}|[😂😍🥰❤️😭😡🔥💀😏🙄]+', text)
        for ep in _emo_patterns:
            if ep not in self.their_patterns["emotional_expressions"]:
                self.their_patterns["emotional_expressions"].append(ep)
                self.their_patterns["emotional_expressions"] = self.their_patterns["emotional_expressions"][-15:]

        # Learn humor style
        _humor_markers = re.findall(r'(?:haha+|lol+|lmao|хаха+|ахах+|ржу|кек|😂|🤣)', text_lower)
        for hm in _humor_markers:
            if hm not in self.their_patterns["humor_style"]:
                self.their_patterns["humor_style"].append(hm)
                self.their_patterns["humor_style"] = self.their_patterns["humor_style"][-10:]

        # Learn affection style
        _affection = re.findall(
            r'(?:love you|miss you|❤️|🥰|😘|😍|💕|люблю|скучаю|целую|обнимаю)', text_lower
        )
        for af in _affection:
            if af not in self.their_patterns["affection_style"]:
                self.their_patterns["affection_style"].append(af)
                self.their_patterns["affection_style"] = self.their_patterns["affection_style"][-10:]

    def _make_context_key(self, context: Dict[str, Any]) -> str:
        stage = context.get("conversation_stage", "unknown")
        temp = context.get("emotional_temperature", "neutral")
        formality = context.get("formality", "casual")

        _stage_map = {
            "greeting": "light", "small_talk": "light", "checking_in": "light",
            "deepening": "deep", "support": "deep", "conflict": "tense",
            "de_escalation": "tense", "flirting": "playful", "celebrating": "playful",
            "planning": "practical", "closing": "light",
        }
        bucket = _stage_map.get(stage, "general")
        return f"{bucket}_{temp}_{formality}"

    def _take_snapshot(self):
        """Record a learning snapshot for evolution tracking."""
        richness = self.vocab_tracker.get_vocabulary_richness()
        snapshot = {
            "timestamp": time.time(),
            "interactions": self.total_interactions,
            "winning_patterns_count": len(self.winning_patterns),
            "losing_patterns_count": len(self.losing_patterns),
            "vocabulary_richness": richness.get("type_token_ratio", 0),
            "overused_words": len(self.vocab_tracker.overused_words),
            "learned_words": len(self.vocab_tracker.learned_words),
            "contexts_learned": len(self.context_vocab),
        }
        self.learning_snapshots.append(snapshot)

    def get_language_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate language guidance for the current conversation context.
        This is the main output — what the bot should know about language
        before generating a response.
        """
        guidance = {
            "avoid_words": [],
            "avoid_phrases": [],
            "preferred_phrases": [],
            "mirror_patterns": {},
            "vocabulary_notes": [],
            "register_hint": "casual",
        }

        context_key = self._make_context_key(context)

        # 1. Words to avoid (overused)
        guidance["avoid_words"] = self.vocab_tracker.get_overused_words(8)

        # 2. Phrases to avoid (negative effectiveness)
        guidance["avoid_phrases"] = self.vocab_tracker.get_toxic_phrases(5)

        # 3. Phrases that work in this context
        if context_key in self.context_vocab:
            context_words = self.context_vocab[context_key]
            good_words = sorted(
                [(w, s) for w, s in context_words.items() if s > 0],
                key=lambda x: x[1], reverse=True
            )[:10]
            guidance["preferred_phrases"] = [w for w, _ in good_words]

        # 4. Mirror their patterns
        mirror = {}
        if self.their_patterns.get("filler_words"):
            mirror["fillers"] = self.their_patterns["filler_words"][:3]
        if self.their_patterns.get("sentence_starters"):
            mirror["starters"] = self.their_patterns["sentence_starters"][:3]
        if self.their_patterns.get("humor_style"):
            mirror["humor"] = self.their_patterns["humor_style"][:3]
        if self.their_patterns.get("emotional_expressions"):
            mirror["emotions"] = self.their_patterns["emotional_expressions"][:3]
        guidance["mirror_patterns"] = mirror

        # 5. Vocabulary notes
        richness = self.vocab_tracker.get_vocabulary_richness()
        if richness.get("richness") == "repetitive":
            guidance["vocabulary_notes"].append("Vocabulary is getting repetitive — use different words")
        if self.vocab_tracker.learned_words:
            learned = list(self.vocab_tracker.learned_words)[:5]
            guidance["vocabulary_notes"].append(
                f"Words they use that you could adopt: {', '.join(learned)}"
            )

        # 6. Winning patterns for this context
        context_winners = [
            p["pattern"] for p in self.winning_patterns
            if p["context"] == context_key and p["count"] >= 2
        ][:5]
        if context_winners:
            guidance["preferred_phrases"].extend(context_winners)

        return guidance

    def format_for_prompt(self, context: Dict[str, Any]) -> str:
        """Format language guidance as a prompt injection."""
        guidance = self.get_language_guidance(context)
        lines = []

        if guidance["avoid_words"]:
            lines.append(f"Overused words (use alternatives): {', '.join(guidance['avoid_words'][:5])}")

        if guidance["avoid_phrases"]:
            lines.append(f"Phrases that don't work (avoid): {', '.join(guidance['avoid_phrases'][:3])}")

        if guidance["preferred_phrases"]:
            lines.append(f"Phrases that work well here: {', '.join(guidance['preferred_phrases'][:5])}")

        mirror = guidance.get("mirror_patterns", {})
        if mirror.get("fillers"):
            lines.append(f"Their filler words (mirror occasionally): {', '.join(mirror['fillers'])}")
        if mirror.get("humor"):
            lines.append(f"How they laugh: {', '.join(mirror['humor'])}")

        for note in guidance.get("vocabulary_notes", [])[:2]:
            lines.append(note)

        if not lines:
            return ""

        return "LANGUAGE LEARNING: " + " | ".join(lines)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            "total_interactions": self.total_interactions,
            "winning_patterns": len(self.winning_patterns),
            "losing_patterns": len(self.losing_patterns),
            "contexts_learned": len(self.context_vocab),
            "their_patterns_learned": {
                k: len(v) for k, v in self.their_patterns.items() if v
            },
            "vocabulary": self.vocab_tracker.get_vocabulary_richness(),
            "overused_words": self.vocab_tracker.get_overused_words(5),
            "effective_phrases": self.vocab_tracker.get_effective_phrases(2)[:5],
            "learned_words": self.vocab_tracker.get_learned_vocabulary()[:10],
        }


# ═══════════════════════════════════════════════════════════════
#  4. REPLY AUDIT SYSTEM — Pre-send comprehensive check
# ═══════════════════════════════════════════════════════════════

def audit_reply(
    reply_text: str,
    incoming_text: str,
    context: Dict[str, Any],
    chat_id: int,
) -> Dict[str, Any]:
    """
    Comprehensive pre-send audit of a reply.
    Combines semantic coherence + vocabulary awareness + learned patterns.

    Returns:
    - passed: bool (should we send this?)
    - score: 0.0-1.0
    - issues: list of problems
    - suggestions: list of fixes
    - vocabulary_warnings: overused words found in reply
    - anti_patterns: known bad phrases found in reply
    """
    # 1. Semantic coherence check
    coherence = assess_semantic_coherence(
        reply_text, incoming_text,
        conversation_stage=context.get("conversation_stage", "unknown"),
        emotional_temperature=context.get("emotional_temperature", "neutral"),
        their_formality=context.get("formality", "casual"),
        nlp_analysis=context.get("nlp_analysis"),
    )

    # 2. Vocabulary check
    vocab_warnings = []
    try:
        tracker = VocabularyTracker(chat_id)
        overused = tracker.get_overused_words()
        reply_words = set(_extract_content_words(reply_text))
        found_overused = reply_words & set(overused)
        if found_overused:
            vocab_warnings.append(f"Overused words in reply: {', '.join(found_overused)}")
    except Exception:
        pass

    # 3. Anti-pattern check
    anti_patterns_found = []
    try:
        learner = ConversationLearner(chat_id)
        toxic = learner.vocab_tracker.get_toxic_phrases()
        reply_lower = reply_text.lower()
        for phrase in toxic:
            if phrase in reply_lower:
                anti_patterns_found.append(phrase)
    except Exception:
        pass

    # 4. Combine scores
    issues = coherence["issues"][:]
    suggestions = coherence["suggestions"][:]

    score = coherence["coherence_score"]
    if vocab_warnings:
        score -= 0.05 * len(vocab_warnings)
        issues.extend([f"overused_word" for _ in vocab_warnings])
    if anti_patterns_found:
        score -= 0.1 * len(anti_patterns_found)
        issues.extend([f"anti_pattern:{p}" for p in anti_patterns_found])
        suggestions.append(f"Avoid these phrases (they get bad reactions): {', '.join(anti_patterns_found)}")

    score = max(score, 0.0)

    return {
        "passed": score >= 0.55,
        "score": round(score, 3),
        "issues": issues,
        "suggestions": suggestions,
        "vocabulary_warnings": vocab_warnings,
        "anti_patterns": anti_patterns_found,
        "coherence_details": coherence["scores"],
    }


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

_COMMON_CONNECTORS = {
    # English
    "i", "me", "my", "you", "your", "we", "us", "our", "the", "a", "an",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "about",
    "and", "or", "but", "not", "so", "if", "then", "than", "that", "this",
    "it", "its", "he", "she", "they", "them", "their",
    "what", "when", "where", "who", "how", "why", "which",
    "just", "very", "really", "also", "too", "more", "most", "some", "any",
    "all", "much", "many", "here", "there", "now", "then",
    # Russian
    "я", "ты", "он", "она", "мы", "вы", "они", "это", "тот", "та", "то",
    "и", "в", "на", "с", "за", "из", "по", "к", "у", "о", "а", "но",
    "не", "да", "что", "как", "так", "уже", "ещё", "еще", "бы", "же",
    "ли", "ну", "вот", "тут", "там", "все", "мне", "тебе", "нас",
    "от", "до", "для", "при", "без", "над", "под", "про",
    "мой", "твой", "его", "её", "наш", "ваш", "их",
    "кто", "где", "когда", "зачем", "почему", "сколько",
}


def _extract_content_words(text: str) -> List[str]:
    """Extract meaningful content words from text, filtering out common connectors."""
    words = re.findall(r'[\w\u0400-\u04ff]+', text.lower())
    return [w for w in words if w not in _COMMON_CONNECTORS and len(w) >= 2]


def _extract_phrases(text: str, max_words: int = 3) -> List[str]:
    """Extract 2-3 word phrases from text."""
    words = re.findall(r'[\w\u0400-\u04ff]+', text.lower())
    content_words = [w for w in words if w not in _COMMON_CONNECTORS and len(w) >= 2]

    phrases = []
    for i in range(len(content_words) - 1):
        bigram = f"{content_words[i]} {content_words[i + 1]}"
        phrases.append(bigram)
        if i < len(content_words) - 2 and max_words >= 3:
            trigram = f"{content_words[i]} {content_words[i + 1]} {content_words[i + 2]}"
            phrases.append(trigram)

    return phrases


def _identify_semantic_fields(words: Set[str], lang_key: str) -> Set[str]:
    """Identify which semantic fields a set of words belongs to."""
    fields = set()
    for field_name, field_data in _SEMANTIC_FIELDS.items():
        field_words = field_data.get(lang_key, set())
        # Use prefix matching for Russian (morphology)
        if lang_key == "ru":
            for w in words:
                for fw in field_words:
                    if w.startswith(fw[:4]) and len(fw) >= 4:
                        fields.add(field_name)
                        break
        else:
            if words & field_words:
                fields.add(field_name)
    return fields


def _detect_register(text: str, lang_key: str) -> str:
    """Detect the formality register of text."""
    text_lower = text.lower()
    words = set(text_lower.split())

    formal_words = _SEMANTIC_FIELDS.get("formal_academic", {}).get(lang_key, set())
    casual_words = _SEMANTIC_FIELDS.get("casual_social", {}).get(lang_key, set())

    formal_count = len(words & formal_words)
    casual_count = len(words & casual_words)

    # Additional checks
    has_slang = bool(re.search(r'\b(?:lol|lmao|bruh|fr|ngl|omg|tbh|wyd|smh)\b', text_lower))
    has_cursing = bool(re.search(r'\b(?:fuck|shit|damn|ass|hell|блять?|хуй|пиздец|сука)\b', text_lower))
    has_abbreviations = bool(re.search(r'\b[a-z]{1,3}\b', text_lower))  # u, r, ur, etc.

    if lang_key == "ru":
        has_slang = has_slang or bool(re.search(r'\b(?:лол|кек|рофл|чел|норм|ваще|пц|хз)\b', text_lower))

    if has_cursing or (has_slang and casual_count > formal_count):
        return "very_casual"
    elif casual_count > formal_count:
        return "casual"
    elif formal_count > casual_count + 1:
        return "formal"
    elif formal_count > 0:
        return "neutral"
    else:
        return "casual"  # default for texting


# ═══════════════════════════════════════════════════════════════
#  5. PUBLIC API — Used by telegram_api.py
# ═══════════════════════════════════════════════════════════════

# Cache for learner instances
_learner_cache: Dict[int, ConversationLearner] = {}


def get_learner(chat_id: int) -> ConversationLearner:
    """Get or create a ConversationLearner for a chat."""
    if chat_id not in _learner_cache:
        _learner_cache[chat_id] = ConversationLearner(chat_id)
    return _learner_cache[chat_id]


def learn_from_interaction(
    chat_id: int,
    our_message: str,
    their_response: str,
    outcome: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Main entry point: learn from a completed interaction.
    Called after they respond to our message.
    """
    learner = get_learner(chat_id)
    learner.learn_from_exchange(our_message, their_response, outcome, context)
    return {"status": "learned", "interactions": learner.total_interactions}


def get_language_guidance(chat_id: int, context: Dict[str, Any]) -> str:
    """Get language guidance as a prompt injection string."""
    learner = get_learner(chat_id)
    return learner.format_for_prompt(context)


def audit_before_send(
    chat_id: int,
    reply_text: str,
    incoming_text: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Pre-send audit combining all checks."""
    return audit_reply(reply_text, incoming_text, context, chat_id)


def get_learning_stats(chat_id: int) -> Dict[str, Any]:
    """Get learning statistics for a chat."""
    learner = get_learner(chat_id)
    return learner.get_learning_stats()
