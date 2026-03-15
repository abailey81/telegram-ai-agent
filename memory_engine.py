"""
Advanced Three-Tier Memory Architecture.

Implements research-backed memory system inspired by Mem0 and human cognition:

1. Short-Term Memory - Current conversation window (in-prompt)
2. Mid-Term Memory - Session summaries, recent facts (structured JSON)
3. Long-Term Memory - Persistent knowledge graph, relationship history

Memory types (based on cognitive science):
- Episodic: Specific events and conversations ("She told me about her promotion on Jan 5")
- Semantic: Facts and knowledge ("Her cat is named Luna, she works in marketing")
- Procedural: Learned patterns ("When she's stressed, keep responses short and warm")

Features:
- Automatic fact extraction from conversations
- Knowledge graph for entity relationships
- Semantic search via embeddings (when available)
- Memory consolidation (compress old memories)
- Proactive recall based on context
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

mem_logger = logging.getLogger("memory_engine")
mem_logger.setLevel(logging.INFO)

MEMORY_DIR = Path(__file__).parent / "engine_data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Auto-pickup: load autoresearch-optimized engine parameters ──
_OPTIMIZED_MEM_PARAMS = None
_OPTIMIZED_MEM_PARAMS_MTIME = 0


def _load_optimized_mem_params():
    """Load optimized memory params from autoresearch (auto-pickup on file change)."""
    global _OPTIMIZED_MEM_PARAMS, _OPTIMIZED_MEM_PARAMS_MTIME
    params_file = Path(__file__).parent / "engine_data" / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_MEM_PARAMS_MTIME:
            _OPTIMIZED_MEM_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_MEM_PARAMS_MTIME = mtime
        return _OPTIMIZED_MEM_PARAMS
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  1. SEMANTIC MEMORY (Facts & Knowledge)
# ═══════════════════════════════════════════════════════════════

def load_semantic_memory(chat_id: int) -> Dict[str, Any]:
    """Load semantic memory (facts about the person)."""
    path = MEMORY_DIR / f"{chat_id}_semantic.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "person": {
            "name": None,
            "nickname": None,
            "age": None,
            "location": None,
            "occupation": None,
            "education": None,
            "languages": [],
        },
        "preferences": {
            "favorite_food": None,
            "favorite_music": None,
            "favorite_movie": None,
            "hobbies": [],
            "dislikes": [],
        },
        "relationships": {
            "family": {},
            "friends": {},
            "pets": {},
        },
        "important_dates": {},
        "facts": [],
        "last_updated": None,
    }


def save_semantic_memory(chat_id: int, memory: Dict[str, Any]):
    """Save semantic memory."""
    memory["last_updated"] = datetime.now().isoformat()
    path = MEMORY_DIR / f"{chat_id}_semantic.json"
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False))


def extract_facts_from_message(
    text: str, sender: str
) -> List[Dict[str, Any]]:
    """Extract structured facts from a message.

    Uses pattern matching to detect personal information,
    preferences, and relationships.
    """
    if sender != "Them":
        return []

    text_lower = text.lower()
    facts = []

    # Name extraction (NO IGNORECASE — names must be capitalized)
    name_patterns = [
        r"(?:[Mm]y name is|[Ii]'m|[Ii] am|[Cc]all me)\s+([A-Z][a-z]{2,})",
        r"(?:[Мм]еня зовут)\s+([А-Я][а-я]{2,})",
    ]
    # Common words that are NOT names
    _not_names = {
        "the", "this", "that", "here", "there", "just", "like", "really", "sure",
        "точно", "просто", "ладно", "конечно", "наверное", "хорошо", "нормально",
        "давай", "может", "сейчас", "потом", "здесь", "когда", "тогда", "очень",
    }
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            if name.lower() not in _not_names and len(name) >= 3:
                facts.append({
                    "type": "person.name",
                    "value": name,
                    "source": text[:80],
                })

    # Age extraction
    age_patterns = [
        r"(?:i'm|i am|im)\s+(\d{1,2})\s*(?:years? old|yo)?",
        r"(?:мне)\s+(\d{1,2})\s*(?:лет|год)",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            age = int(match.group(1))
            if 13 <= age <= 99:
                facts.append({"type": "person.age", "value": age, "source": text[:80]})

    # Location
    location_patterns = [
        r"(?:i live in|i'm from|i'm in|based in|living in)\s+([A-Z][a-zA-Z\s]+)",
        r"(?:я из|живу в)\s+([А-ЯA-Z][а-яa-zA-Z\s]+)",
    ]
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts.append({
                "type": "person.location",
                "value": match.group(1).strip()[:40],
                "source": text[:80],
            })

    # Occupation
    job_patterns = [
        r"(?:i work (?:as|at|in)|i'm a|my job is|i do)\s+(.+?)(?:\.|,|!|$)",
        r"(?:я работаю|работаю)\s+(.+?)(?:\.|,|!|$)",
    ]
    for pattern in job_patterns:
        match = re.search(pattern, text_lower)
        if match:
            facts.append({
                "type": "person.occupation",
                "value": match.group(1).strip()[:50],
                "source": text[:80],
            })

    # Favorites
    fav_patterns = [
        (r"(?:my fav(?:ou?rite)?)\s+(\w+)\s+is\s+(.+?)(?:\.|!|,|$)", "preferences"),
        (r"i (?:really )?love\s+(.+?)(?:\.|!|,|$)", "preferences.likes"),
        (r"i (?:really )?hate\s+(.+?)(?:\.|!|,|$)", "preferences.dislikes"),
        # Russian preferences
        (r"(?:мой любимый|моя любимая|любимое)\s+(.+?)(?:\.|!|,|$)", "preferences"),
        (r"(?:мне нравится|я люблю|обожаю)\s+(.+?)(?:\.|!|,|$)", "preferences.likes"),
        (r"(?:я ненавижу|терпеть не могу|не люблю)\s+(.+?)(?:\.|!|,|$)", "preferences.dislikes"),
    ]
    for pattern, fact_type in fav_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches[:2]:
            if isinstance(match, tuple):
                value = f"{match[0]}: {match[1]}"
            else:
                value = match
            facts.append({
                "type": fact_type,
                "value": value.strip()[:60],
                "source": text[:80],
            })

    # Pets
    pet_patterns = [
        r"my (?:cat|dog|pet|bird|fish|hamster|rabbit)\s+(?:is\s+)?(?:called|named)?\s*([A-Z][a-z]+)",
        r"(?:cat|dog|pet) (?:called|named)\s+([A-Z][a-z]+)",
        r"(?:мо[йяюего]+\s+)?(?:кот|кошк[аиу]|собак[аиу]|питомец|птиц[аиу]|рыбк[аиу]|хомяк|кролик)\s+(?:зовут\s+)?([А-Я][а-я]+)",
        r"(?:кот|кошк[аиу]|собак[аиу]|питомец)\s+(?:по имени|зовут)\s+([А-Я][а-я]+)",
    ]
    for pattern in pet_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts.append({
                "type": "relationships.pets",
                "value": match.group(1),
                "source": text[:80],
            })

    # Family mentions
    family_patterns = [
        (r"my (mom|mother|mum)", "mother"),
        (r"my (dad|father)", "father"),
        (r"my (sister|brother)", None),
        (r"my (boyfriend|girlfriend|partner|husband|wife)", None),
        # Russian family patterns
        (r"(?:моя мама|мама)\s+(.+?)(?:\.|,|!|$)", "mother"),
        (r"(?:мой папа|папа)\s+(.+?)(?:\.|,|!|$)", "father"),
        (r"(?:моя сестра|сестра)\s+(.+?)(?:\.|,|!|$)", "sister"),
        (r"(?:мой брат|брат)\s+(.+?)(?:\.|,|!|$)", "brother"),
        (r"(?:моя бабушка|бабушка)\s+(.+?)(?:\.|,|!|$)", "grandmother"),
        (r"(?:мой дедушка|дедушка)\s+(.+?)(?:\.|,|!|$)", "grandfather"),
        (r"(?:моя тётя|тётя)\s+(.+?)(?:\.|,|!|$)", "aunt"),
        (r"(?:мой дядя|дядя)\s+(.+?)(?:\.|,|!|$)", "uncle"),
        (r"(?:мой парень|парень)\s+(.+?)(?:\.|,|!|$)", "boyfriend"),
        (r"(?:моя девушка|девушка)\s+(.+?)(?:\.|,|!|$)", "girlfriend"),
        (r"(?:мой муж|муж)\s+(.+?)(?:\.|,|!|$)", "husband"),
        (r"(?:моя жена|жена)\s+(.+?)(?:\.|,|!|$)", "wife"),
    ]
    for pattern, role in family_patterns:
        match = re.search(pattern, text_lower)
        if match:
            actual_role = role or match.group(1)
            facts.append({
                "type": "relationships.family",
                "value": actual_role,
                "source": text[:80],
            })

    # Important dates
    date_patterns = [
        r"my birthday (?:is )?(?:on )?(.+?)(?:\.|!|,|$)",
        r"(?:our anniversary|anniversary) (?:is )?(?:on )?(.+?)(?:\.|!|,|$)",
        # Russian dates
        r"(?:мой день рождения|день рождения|др)\s+(.+?)(?:\.|!|,|$)",
        r"(?:наша годовщина|годовщина)\s+(.+?)(?:\.|!|,|$)",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            facts.append({
                "type": "important_dates",
                "value": match.group(1).strip()[:30],
                "source": text[:80],
            })

    return facts


def update_semantic_memory(
    chat_id: int,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Update semantic memory by extracting facts from messages."""
    memory = load_semantic_memory(chat_id)

    their_msgs = [m for m in messages if m.get("sender") == "Them"]
    for msg in their_msgs[-20:]:
        facts = extract_facts_from_message(msg.get("text", ""), "Them")
        for fact in facts:
            _integrate_fact(memory, fact)

    save_semantic_memory(chat_id, memory)
    return memory


def _integrate_fact(memory: Dict[str, Any], fact: Dict[str, Any]):
    """Integrate a fact into the semantic memory structure."""
    fact_type = fact["type"]
    value = fact["value"]

    parts = fact_type.split(".")
    if len(parts) == 2:
        section, key = parts
        if section in memory and key in memory[section]:
            if isinstance(memory[section][key], list):
                if value not in memory[section][key]:
                    memory[section][key].append(value)
            else:
                memory[section][key] = value
    elif fact_type.startswith("preferences"):
        if "likes" in fact_type:
            if value not in memory["preferences"].get("hobbies", []):
                memory["preferences"].setdefault("hobbies", []).append(value)
        elif "dislikes" in fact_type:
            if value not in memory["preferences"].get("dislikes", []):
                memory["preferences"].setdefault("dislikes", []).append(value)

    # Also add to general facts list
    fact_entry = {
        "fact": f"{fact_type}: {value}",
        "timestamp": datetime.now().isoformat(),
    }
    memory["facts"].append(fact_entry)
    _opt = _load_optimized_mem_params()
    _max_facts = (_opt or {}).get("max_facts", 50)
    memory["facts"] = memory["facts"][-_max_facts:]


# ═══════════════════════════════════════════════════════════════
#  2. EPISODIC MEMORY (Specific Events & Conversations)
# ═══════════════════════════════════════════════════════════════

def load_episodic_memory(chat_id: int) -> Dict[str, Any]:
    """Load episodic memory (conversation events)."""
    path = MEMORY_DIR / f"{chat_id}_episodic.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "episodes": [],
        "milestones": [],
        "shared_references": [],
    }


def save_episodic_memory(chat_id: int, memory: Dict[str, Any]):
    """Save episodic memory."""
    path = MEMORY_DIR / f"{chat_id}_episodic.json"
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False))


def record_episode(
    chat_id: int,
    messages: List[Dict[str, str]],
    episode_type: str = "conversation",
    summary: str = "",
) -> Dict[str, Any]:
    """Record a conversation episode for future reference."""
    memory = load_episodic_memory(chat_id)

    # Create episode summary
    if not summary:
        # Auto-summarize from messages
        key_messages = []
        for msg in messages[-10:]:
            text = msg.get("text", "")
            if len(text) > 15:
                key_messages.append(
                    f"{msg.get('sender', '?')}: {text[:60]}"
                )
        summary = " | ".join(key_messages[-5:])

    episode = {
        "type": episode_type,
        "summary": summary[:300],
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "key_topics": _extract_episode_topics(messages),
        "emotional_tone": _detect_episode_emotion(messages),
    }

    memory["episodes"].append(episode)
    _opt = _load_optimized_mem_params()
    _max_episodes = (_opt or {}).get("max_episodes", 100)
    memory["episodes"] = memory["episodes"][-_max_episodes:]

    # Detect milestones
    milestone = _detect_milestone(messages)
    if milestone:
        memory["milestones"].append(milestone)
        _max_milestones = (_opt or {}).get("max_milestones", 20)
        memory["milestones"] = memory["milestones"][-_max_milestones:]

    # Detect shared references (inside jokes, shared experiences)
    refs = _detect_shared_references(messages)
    for ref in refs:
        if ref not in memory["shared_references"]:
            memory["shared_references"].append(ref)
    _max_shared_refs = (_opt or {}).get("max_shared_references", 30)
    memory["shared_references"] = memory["shared_references"][-_max_shared_refs:]

    save_episodic_memory(chat_id, memory)
    return memory


def _extract_episode_topics(messages: List[Dict[str, str]]) -> List[str]:
    """Extract main topics from a conversation episode."""
    all_text = " ".join(m.get("text", "") for m in messages[-10:]).lower()
    topics = []

    topic_keywords = {
        "relationship": ["us", "together", "relationship", "love", "future",
                         "вместе", "отношения", "любовь", "будущее", "нас"],
        "work": ["work", "job", "boss", "project", "meeting",
                 "работа", "работу", "босс", "начальник", "проект", "встреча"],
        "personal": ["feel", "think", "believe", "want", "need",
                     "чувствую", "думаю", "верю", "хочу", "нужно"],
        "plans": ["plan", "weekend", "trip", "dinner", "meet",
                  "план", "выходные", "поездка", "ужин", "встретиться", "давай"],
        "conflict": ["argue", "fight", "upset", "angry", "sorry",
                     "ссора", "ссоримся", "расстроен", "злюсь", "извини", "прости"],
        "fun": ["fun", "laugh", "joke", "game", "movie", "show",
                "весело", "ржу", "шутка", "игра", "фильм", "сериал", "кино"],
    }

    for topic, keywords in topic_keywords.items():
        if any(k in all_text for k in keywords):
            topics.append(topic)

    return topics[:3]


def _detect_episode_emotion(messages: List[Dict[str, str]]) -> str:
    """Detect the dominant emotional tone of an episode."""
    all_text = " ".join(
        m.get("text", "") for m in messages[-10:] if m.get("sender") == "Them"
    ).lower()

    emotion_keywords = {
        "positive": ["happy", "love", "great", "amazing", "haha", "❤", "😊",
                     "счастлив", "рад", "круто", "класс", "кайф", "супер", "ахах"],
        "negative": ["sad", "angry", "upset", "hate", "frustrated", "😢", "😡",
                     "грустно", "злюсь", "расстроен", "ненавижу", "бесит", "плохо"],
        "romantic": ["miss you", "love you", "kiss", "🥰", "😍", "😘",
                     "скучаю", "люблю", "целую", "обнимаю", "хочу к тебе"],
        "playful": ["haha", "lol", "😂", "joking", "funny", "😏",
                    "хаха", "ахахах", "ржу", "лол", "ору", "угар", "смешно"],
        "serious": ["need to talk", "important", "listen", "seriously",
                    "нам надо поговорить", "это важно", "послушай", "серьёзно"],
    }

    scores = {k: 0 for k in emotion_keywords}
    for tone, keywords in emotion_keywords.items():
        scores[tone] = sum(1 for k in keywords if k in all_text)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def _detect_milestone(messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Detect significant milestones (relationship, personal, shared)."""
    all_text = " ".join(m.get("text", "") for m in messages[-10:]).lower()

    milestones = {
        # Relationship milestones
        "first_i_love_you": ["i love you", "я люблю тебя"],
        "first_date_plan": ["our first date", "go on a date",
                            "наше первое свидание", "пойдём на свидание"],
        "meeting_family": ["meet my parents", "meet my mom", "meet my family",
                           "познакомить с родителями", "встретить маму", "встретить семью"],
        "future_plans": ["move in together", "our future", "someday we",
                         "жить вместе", "наше будущее", "когда-нибудь мы"],
        "important_apology": ["i'm really sorry", "please forgive me",
                              "мне очень жаль", "пожалуйста прости"],
        # Personal milestones
        "deep_vulnerability": ["never told anyone", "my biggest secret", "nobody knows",
                               "никому не говорил", "мой секрет", "никто не знает"],
        "career_achievement": ["got the job", "got promoted", "got the offer", "accepted",
                               "устроился на работу", "повысили", "приняли", "получил оффер"],
        "education_milestone": ["graduated", "passed the exam", "got accepted", "finished my",
                                "закончил", "сдал экзамен", "поступил", "защитился"],
        "personal_growth": ["i realized", "i've changed", "i'm better now", "i learned",
                            "я понял", "я изменился", "мне стало лучше", "я научился"],
        "health_update": ["diagnosis", "surgery", "recovering", "test results",
                          "диагноз", "операция", "выздоравливаю", "результаты анализов"],
        # Shared experiences
        "shared_discovery": ["we should try", "found this place", "new favorite",
                             "давай попробуем", "нашёл место", "новое любимое"],
        "inside_joke_born": ["that's our thing", "only we", "our secret",
                             "это наше", "только мы", "наш секрет"],
        "travel_together": ["our trip", "when we went to", "let's go to",
                            "наша поездка", "когда мы ездили", "поехали в"],
        "first_disagreement_resolved": ["i'm glad we talked", "glad we worked it out",
                                        "хорошо что поговорили", "рад что разобрались"],
    }

    for milestone_type, markers in milestones.items():
        if any(m in all_text for m in markers):
            return {
                "type": milestone_type,
                "timestamp": datetime.now().isoformat(),
                "context": all_text[:100],
            }
    return None


def _detect_shared_references(messages: List[Dict[str, str]]) -> List[str]:
    """Detect potential inside jokes or shared references."""
    refs = []
    their_text = " ".join(
        m.get("text", "") for m in messages[-10:] if m.get("sender") == "Them"
    ).lower()

    # Detect "remember when" references
    remember_patterns = [
        r"remember when\s+(.+?)(?:\?|\.|!|$)",
        r"like that time\s+(.+?)(?:\?|\.|!|$)",
        r"помнишь как\s+(.+?)(?:\?|\.|!|$)",
    ]
    for pattern in remember_patterns:
        matches = re.findall(pattern, their_text)
        for match in matches:
            refs.append(match.strip()[:60])

    return refs


# ═══════════════════════════════════════════════════════════════
#  3. PROCEDURAL MEMORY (Learned Patterns & Rules)
# ═══════════════════════════════════════════════════════════════

def load_procedural_memory(chat_id: int) -> Dict[str, Any]:
    """Load procedural memory (learned communication rules)."""
    path = MEMORY_DIR / f"{chat_id}_procedural.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    return {
        "chat_id": chat_id,
        "response_rules": [],
        "successful_patterns": [],
        "failed_patterns": [],
        "learned_preferences": {},
    }


def save_procedural_memory(chat_id: int, memory: Dict[str, Any]):
    """Save procedural memory."""
    path = MEMORY_DIR / f"{chat_id}_procedural.json"
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False))


def learn_from_interaction(
    chat_id: int,
    our_message: str,
    their_response: str,
    response_quality: str = "unknown",
) -> Dict[str, Any]:
    """Learn from a message-response pair.

    If their response was positive, record the pattern as successful.
    If negative, record as a pattern to avoid.
    """
    memory = load_procedural_memory(chat_id)

    # Detect response quality from their reaction
    their_lower = their_response.lower()

    positive_signals = ["haha", "lol", "❤", "🥰", "😍", "aww", "love",
                        "yes", "exactly", "same", "omg", "wow", "amazing",
                        "хаха", "ахах", "ржу", "лол", "люблю", "обожаю",
                        "да", "точно", "вау", "круто", "класс", "огонь"]
    negative_signals = ["whatever", "k", "ok", "...", "fine", "bye",
                        "don't", "stop", "leave me", "not funny",
                        "пофиг", "ок", "ладно", "хватит", "отстань",
                        "не пиши", "не смешно", "пока", "забей"]

    pos_count = sum(1 for s in positive_signals if s in their_lower)
    neg_count = sum(1 for s in negative_signals if s in their_lower)

    if pos_count > neg_count and pos_count >= 2:
        quality = "positive"
    elif neg_count > pos_count and neg_count >= 1:
        quality = "negative"
    else:
        quality = response_quality

    pattern = {
        "our_message_preview": our_message[:80],
        "their_response_preview": their_response[:80],
        "quality": quality,
        "timestamp": datetime.now().isoformat(),
    }

    if quality == "positive":
        memory["successful_patterns"].append(pattern)
        memory["successful_patterns"] = memory["successful_patterns"][-30:]
    elif quality == "negative":
        memory["failed_patterns"].append(pattern)
        memory["failed_patterns"] = memory["failed_patterns"][-20:]

    save_procedural_memory(chat_id, memory)
    return memory


# ═══════════════════════════════════════════════════════════════
#  4. MEMORY RETRIEVAL & RECALL
# ═══════════════════════════════════════════════════════════════

def recall_relevant_memories(
    chat_id: int,
    current_context: str,
    max_memories: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve memories most relevant to current conversation context.

    Uses keyword matching (and semantic search when available).
    Returns ordered list of relevant memories.
    """
    results = []

    # Load all memory types
    semantic = load_semantic_memory(chat_id)
    episodic = load_episodic_memory(chat_id)
    procedural = load_procedural_memory(chat_id)

    context_lower = current_context.lower()
    context_words = set(context_lower.split())

    # Search semantic facts
    for fact in semantic.get("facts", []):
        fact_text = fact.get("fact", "").lower()
        overlap = len(context_words & set(fact_text.split()))
        if overlap >= 2:
            results.append({
                "type": "fact",
                "content": fact["fact"],
                "relevance": overlap,
                "source": "semantic_memory",
            })

    # Search episodic memories
    for episode in episodic.get("episodes", []):
        summary = episode.get("summary", "").lower()
        topics = episode.get("key_topics", [])
        overlap = len(context_words & set(summary.split()))
        topic_overlap = sum(1 for t in topics if t in context_lower)
        relevance = overlap + topic_overlap * 2
        if relevance >= 2:
            results.append({
                "type": "episode",
                "content": episode["summary"][:100],
                "relevance": relevance,
                "source": "episodic_memory",
                "timestamp": episode.get("timestamp", ""),
            })

    # Search milestones
    for milestone in episodic.get("milestones", []):
        milestone_text = f"{milestone['type']} {milestone.get('context', '')}".lower()
        overlap = len(context_words & set(milestone_text.split()))
        if overlap >= 1:
            results.append({
                "type": "milestone",
                "content": f"{milestone['type']}: {milestone.get('context', '')[:60]}",
                "relevance": overlap + 3,  # milestones always important
                "source": "episodic_memory",
            })

    # Search shared references
    for ref in episodic.get("shared_references", []):
        if any(w in ref.lower() for w in context_words if len(w) > 3):
            results.append({
                "type": "shared_reference",
                "content": ref,
                "relevance": 4,
                "source": "episodic_memory",
            })

    # Search successful patterns
    for pattern in procedural.get("successful_patterns", [])[-10:]:
        our_msg = pattern.get("our_message_preview", "").lower()
        if any(w in our_msg for w in context_words if len(w) > 3):
            results.append({
                "type": "successful_pattern",
                "content": f"This worked before: \"{pattern['our_message_preview']}\"",
                "relevance": 3,
                "source": "procedural_memory",
            })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)

    # Try semantic search if available
    try:
        from dl_models import get_model_manager
        mm = get_model_manager()
        if mm.has_embeddings and results:
            # Re-rank using semantic similarity
            query_emb = mm.embed_single(current_context)
            if query_emb is not None:
                for result in results:
                    content_emb = mm.embed_single(result["content"])
                    if content_emb is not None:
                        sim = mm.cosine_similarity(query_emb, content_emb)
                        result["semantic_relevance"] = round(sim, 4)
                        _opt = _load_optimized_mem_params()
                        _sem_boost = (_opt or {}).get("semantic_similarity_boost", 5)
                        result["relevance"] += sim * _sem_boost  # Boost by semantic sim

                results.sort(key=lambda x: x["relevance"], reverse=True)
    except Exception:
        pass

    return results[:max_memories]


# ═══════════════════════════════════════════════════════════════
#  5. MEMORY CONSOLIDATION (Compress & Organize)
# ═══════════════════════════════════════════════════════════════

def consolidate_memories(chat_id: int) -> Dict[str, Any]:
    """Run memory consolidation to organize and compress.

    Should be called periodically (e.g., at end of conversation session).
    """
    semantic = load_semantic_memory(chat_id)
    episodic = load_episodic_memory(chat_id)

    # Deduplicate semantic facts
    seen_facts = set()
    unique_facts = []
    for fact in semantic.get("facts", []):
        fact_key = fact.get("fact", "").lower().strip()
        if fact_key not in seen_facts:
            seen_facts.add(fact_key)
            unique_facts.append(fact)
    _opt = _load_optimized_mem_params()
    _max_facts = (_opt or {}).get("max_facts", 50)
    semantic["facts"] = unique_facts[-_max_facts:]

    # Compress old episodes (keep summary only)
    episodes = episodic.get("episodes", [])
    if len(episodes) > 50:
        # Keep last 50 full, compress older ones
        for i in range(len(episodes) - 50):
            episodes[i] = {
                "type": episodes[i].get("type", "conversation"),
                "summary": episodes[i].get("summary", "")[:100],
                "timestamp": episodes[i].get("timestamp", ""),
                "key_topics": episodes[i].get("key_topics", []),
                "emotional_tone": episodes[i].get("emotional_tone", "neutral"),
                "compressed": True,
            }

    save_semantic_memory(chat_id, semantic)
    save_episodic_memory(chat_id, episodic)

    return {
        "facts_count": len(semantic.get("facts", [])),
        "episodes_count": len(episodes),
        "milestones_count": len(episodic.get("milestones", [])),
    }


# ═══════════════════════════════════════════════════════════════
#  6. FORMAT FOR PROMPT
# ═══════════════════════════════════════════════════════════════

def format_memory_for_prompt(
    chat_id: int,
    current_context: str = "",
) -> str:
    """Format relevant memories for prompt injection.

    Assembles the most useful memory information for Claude.
    """
    parts = []

    # Semantic memory (key facts)
    semantic = load_semantic_memory(chat_id)
    person = semantic.get("person", {})
    non_null_person = {k: v for k, v in person.items() if v and v != []}
    if non_null_person:
        person_facts = ", ".join(f"{k}: {v}" for k, v in non_null_person.items())
        parts.append(f"About them: {person_facts}")

    prefs = semantic.get("preferences", {})
    non_null_prefs = {k: v for k, v in prefs.items() if v and v != []}
    if non_null_prefs:
        pref_str = ", ".join(
            f"{k}: {v if isinstance(v, str) else ', '.join(v[:3])}"
            for k, v in non_null_prefs.items()
        )
        parts.append(f"Their preferences: {pref_str}")

    relationships = semantic.get("relationships", {})
    for rel_type, rels in relationships.items():
        if rels:
            if isinstance(rels, dict):
                for name, role in list(rels.items())[:3]:
                    parts.append(f"Their {rel_type}: {name} ({role})")
            elif isinstance(rels, list):
                parts.append(f"Their {rel_type}: {', '.join(str(r) for r in rels[:3])}")

    dates = semantic.get("important_dates", {})
    if dates:
        for date_type, date_val in list(dates.items())[:3]:
            parts.append(f"Important date: {date_type} - {date_val}")

    # Episodic memory (milestones and shared references)
    episodic = load_episodic_memory(chat_id)
    milestones = episodic.get("milestones", [])
    if milestones:
        milestone_str = "; ".join(
            f"{m['type'].replace('_', ' ')}"
            for m in milestones[-3:]
        )
        parts.append(f"Relationship milestones: {milestone_str}")

    shared_refs = episodic.get("shared_references", [])
    if shared_refs:
        parts.append(f"Shared memories: {', '.join(shared_refs[-3:])}")

    # Contextually relevant memories
    if current_context:
        relevant = recall_relevant_memories(chat_id, current_context, max_memories=3)
        _opt = _load_optimized_mem_params()
        _rel_cutoff = (_opt or {}).get("memory_relevance_cutoff", 3)
        for mem in relevant:
            if mem["relevance"] >= _rel_cutoff:
                parts.append(f"Relevant memory: {mem['content']}")

    if not parts:
        return ""

    return "\n".join(f"- [MEM] {p}" for p in parts)


# ═══════════════════════════════════════════════════════════════
#  ENHANCED: Relationship Trajectory & Behavioral Patterns
# ═══════════════════════════════════════════════════════════════

def _safe_import_psych_memory():
    """Safely import psychological datasets for memory engine."""
    try:
        from psychological_datasets import (
            compute_gottman_ratio,
            detect_behavioral_pattern,
            detect_knapp_stage,
            comprehensive_psychological_analysis,
            format_psychological_analysis_for_prompt,
        )
        return {
            "compute_gottman_ratio": compute_gottman_ratio,
            "detect_behavioral_pattern": detect_behavioral_pattern,
            "detect_knapp_stage": detect_knapp_stage,
            "comprehensive_psychological_analysis": comprehensive_psychological_analysis,
            "format_psychological_analysis_for_prompt": format_psychological_analysis_for_prompt,
        }
    except ImportError:
        return {}


TRAJECTORY_DIR = Path(__file__).parent / "engine_data" / "trajectory"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


def record_relationship_snapshot(
    chat_id: int,
    messages: List[Dict[str, str]],
    sentiment_score: float = 0.5,
) -> Dict[str, Any]:
    """Record a point-in-time snapshot of relationship health for trajectory tracking."""
    psych = _safe_import_psych_memory()

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "sentiment_score": sentiment_score,
    }

    if psych:
        ratio = psych["compute_gottman_ratio"](messages, sender="them")
        snapshot["gottman_ratio"] = ratio.get("ratio", 0)
        snapshot["gottman_assessment"] = ratio.get("assessment", "Unknown")

        stage = psych["detect_knapp_stage"](messages)
        snapshot["knapp_stage"] = stage.get("stage", "unknown")
        snapshot["knapp_phase"] = stage.get("phase", "unknown")

    # Load existing trajectory
    traj_path = TRAJECTORY_DIR / f"{chat_id}_trajectory.json"
    trajectory = []
    if traj_path.exists():
        try:
            trajectory = json.loads(traj_path.read_text())
        except (json.JSONDecodeError, OSError):
            trajectory = []

    trajectory.append(snapshot)
    # Keep last 100 snapshots
    trajectory = trajectory[-100:]

    try:
        traj_path.write_text(json.dumps(trajectory, indent=2))
    except OSError:
        pass

    return snapshot


def get_relationship_trajectory(chat_id: int) -> Dict[str, Any]:
    """Get relationship trajectory analysis over time."""
    traj_path = TRAJECTORY_DIR / f"{chat_id}_trajectory.json"
    if not traj_path.exists():
        return {"snapshots": 0, "available": False}

    try:
        trajectory = json.loads(traj_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"snapshots": 0, "available": False}

    if len(trajectory) < 2:
        return {"snapshots": len(trajectory), "trend": "insufficient_data"}

    # Analyze trends
    sentiments = [s.get("sentiment_score", 0.5) for s in trajectory]
    gottman_ratios = [s.get("gottman_ratio", 5.0) for s in trajectory]

    # Simple trend detection
    recent_sentiment = sum(sentiments[-5:]) / min(len(sentiments), 5)
    older_sentiment = sum(sentiments[:-5]) / max(len(sentiments) - 5, 1)
    sentiment_trend = "improving" if recent_sentiment > older_sentiment + 0.05 else \
                      "declining" if recent_sentiment < older_sentiment - 0.05 else "stable"

    recent_ratio = sum(gottman_ratios[-5:]) / min(len(gottman_ratios), 5)
    healthy = recent_ratio >= 5.0

    # Detect stage transitions
    stages = [s.get("knapp_stage", "unknown") for s in trajectory if s.get("knapp_stage")]
    stage_changes = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            stage_changes.append({"from": stages[i - 1], "to": stages[i]})

    return {
        "snapshots": len(trajectory),
        "sentiment_trend": sentiment_trend,
        "recent_sentiment": round(recent_sentiment, 2),
        "gottman_ratio_current": round(recent_ratio, 2),
        "gottman_healthy": healthy,
        "current_stage": stages[-1] if stages else "unknown",
        "stage_transitions": stage_changes[-5:],
        "trajectory_health": (
            "thriving" if healthy and sentiment_trend == "improving"
            else "stable" if healthy
            else "needs_attention" if sentiment_trend == "declining"
            else "monitoring"
        ),
    }


def detect_behavioral_patterns_in_chat(
    messages: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Detect concerning behavioral patterns (ghosting, breadcrumbing, etc.)."""
    psych = _safe_import_psych_memory()
    if not psych or len(messages) < 10:
        return []

    # Extract time series data
    timestamps = []
    lengths = []
    sentiments = []
    positive_words = {"love", "happy", "great", "amazing", "miss", "❤️", "😊"}
    negative_words = {"hate", "angry", "upset", "annoyed", "frustrated", "😡"}

    for i, msg in enumerate(messages):
        timestamps.append(float(i))  # Use index as proxy for time
        lengths.append(len(msg.get("text", "")))
        text_lower = msg.get("text", "").lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        sentiments.append((pos - neg) / max(pos + neg, 1))

    return psych["detect_behavioral_pattern"](timestamps, lengths, sentiments)


def run_comprehensive_psychological_analysis(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Run full comprehensive psychological analysis on conversation."""
    psych = _safe_import_psych_memory()
    if not psych:
        return {"available": False}
    return psych["comprehensive_psychological_analysis"](messages)


def format_trajectory_for_prompt(chat_id: int) -> str:
    """Format relationship trajectory for prompt injection."""
    trajectory = get_relationship_trajectory(chat_id)
    if trajectory.get("snapshots", 0) < 2:
        return ""

    parts = []
    parts.append(
        f"[TRAJECTORY] Sentiment trend: {trajectory.get('sentiment_trend', '?')} | "
        f"Gottman ratio: {trajectory.get('gottman_ratio_current', '?')}:1 | "
        f"Stage: {trajectory.get('current_stage', '?')} | "
        f"Health: {trajectory.get('trajectory_health', '?')}"
    )

    transitions = trajectory.get("stage_transitions", [])
    if transitions:
        last = transitions[-1]
        parts.append(f"[STAGE CHANGE] {last['from']} → {last['to']}")

    return "\n".join(f"- [MEM] {p}" for p in parts)
