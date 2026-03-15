"""
Advanced Context Engine (RAG)
==============================
Hierarchical context management with persistent vector storage,
conversation summarization, topic threading, and temporal retrieval.

Replaces the naive "last N messages" approach with intelligent
context construction that combines:
1. Persistent FAISS vector store (per-chat, saved to disk)
2. Hierarchical summarization (session → day → relationship)
3. Topic-threaded retrieval (find relevant past convos by topic)
4. Temporal weighting (recent context weighted higher)
5. Emotional arc tracking (continuity across sessions)
6. Smart window construction (summaries + recent + relevant)
"""

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

context_logger = logging.getLogger("context_engine")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

CONTEXT_DATA_DIR = Path("engine_data/context")
CONTEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR = CONTEXT_DATA_DIR / "faiss_indexes"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR = CONTEXT_DATA_DIR / "summaries"
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
TOPICS_DIR = CONTEXT_DATA_DIR / "topics"
TOPICS_DIR.mkdir(parents=True, exist_ok=True)
ARCS_DIR = CONTEXT_DATA_DIR / "emotional_arcs"
ARCS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. EMBEDDING MODEL (lazy-loaded)
# ═══════════════════════════════════════════════════════════════

_embedder = None


def _get_embedder():
    """Lazy-load the sentence embedding model."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        context_logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        return _embedder
    except ImportError:
        context_logger.warning("sentence-transformers not available")
        return None


def _embed(texts: List[str]):
    """Embed a list of texts. Returns numpy array or None."""
    model = _get_embedder()
    if model is None:
        return None
    import numpy as np
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _embed_single(text: str):
    result = _embed([text])
    if result is not None:
        return result[0]
    return None


# ═══════════════════════════════════════════════════════════════
#  2. PERSISTENT FAISS VECTOR STORE (per-chat)
# ═══════════════════════════════════════════════════════════════

_faiss_indexes = {}  # chat_id → {"index": faiss.Index, "metadata": [...]}


def _get_faiss():
    """Import faiss safely."""
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def load_vector_store(chat_id: int) -> bool:
    """Load or create persistent FAISS index for a chat."""
    faiss = _get_faiss()
    if faiss is None:
        return False

    if chat_id in _faiss_indexes:
        return True

    index_path = FAISS_DIR / f"{chat_id}.index"
    meta_path = FAISS_DIR / f"{chat_id}_meta.json"

    if index_path.exists() and meta_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            metadata = json.loads(meta_path.read_text())
            _faiss_indexes[chat_id] = {"index": index, "metadata": metadata}
            context_logger.info(
                f"Loaded vector store for chat {chat_id}: "
                f"{index.ntotal} vectors"
            )
            return True
        except Exception as e:
            context_logger.warning(f"Failed to load vector store: {e}")

    # Create new index (384-dim for MiniLM, using inner product for normalized vecs)
    import numpy as np
    dim = 384
    index = faiss.IndexFlatIP(dim)
    _faiss_indexes[chat_id] = {"index": index, "metadata": []}
    context_logger.info(f"Created new vector store for chat {chat_id}")
    return True


def save_vector_store(chat_id: int):
    """Persist FAISS index to disk."""
    faiss = _get_faiss()
    if faiss is None or chat_id not in _faiss_indexes:
        return

    store = _faiss_indexes[chat_id]
    index_path = FAISS_DIR / f"{chat_id}.index"
    meta_path = FAISS_DIR / f"{chat_id}_meta.json"

    try:
        faiss.write_index(store["index"], str(index_path))
        meta_path.write_text(json.dumps(store["metadata"], ensure_ascii=False))
        context_logger.debug(f"Saved vector store for chat {chat_id}")
    except Exception as e:
        context_logger.error(f"Failed to save vector store: {e}")


def add_to_vector_store(
    chat_id: int,
    text: str,
    sender: str = "Them",
    timestamp: Optional[str] = None,
    emotion: str = "neutral",
    topic: str = "",
) -> bool:
    """Add a message to the persistent vector store."""
    if not text or len(text.strip()) < 3:
        return False

    if not load_vector_store(chat_id):
        return False

    embedding = _embed_single(text)
    if embedding is None:
        return False

    import numpy as np
    store = _faiss_indexes[chat_id]
    store["index"].add(np.array([embedding], dtype=np.float32))
    store["metadata"].append({
        "text": text[:500],
        "sender": sender,
        "timestamp": timestamp or datetime.now().isoformat(),
        "emotion": emotion,
        "topic": topic,
        "hash": hashlib.md5(text.encode()).hexdigest()[:8],
    })

    # Auto-save every 50 additions
    if len(store["metadata"]) % 50 == 0:
        save_vector_store(chat_id)

    return True


def search_vector_store(
    chat_id: int,
    query: str,
    top_k: int = 10,
    min_score: float = 0.3,
    sender_filter: Optional[str] = None,
    recency_boost: bool = True,
) -> List[Dict[str, Any]]:
    """Search the vector store with optional recency boosting.

    Returns: [{"text": str, "score": float, "sender": str, "timestamp": str, ...}]
    """
    if not load_vector_store(chat_id):
        return []

    query_embedding = _embed_single(query)
    if query_embedding is None:
        return []

    import numpy as np
    store = _faiss_indexes[chat_id]

    if store["index"].ntotal == 0:
        return []

    k = min(top_k * 3, store["index"].ntotal)  # Over-fetch for filtering
    scores, indices = store["index"].search(
        np.array([query_embedding], dtype=np.float32), k
    )

    results = []
    now = datetime.now()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(store["metadata"]):
            continue
        if score < min_score:
            continue

        meta = store["metadata"][idx]

        # Optional sender filter
        if sender_filter and meta["sender"] != sender_filter:
            continue

        # Recency boost: messages from last 24h get +20%, last week +10%
        final_score = float(score)
        if recency_boost:
            try:
                msg_time = datetime.fromisoformat(meta["timestamp"])
                age_hours = (now - msg_time).total_seconds() / 3600
                if age_hours < 24:
                    final_score *= 1.2
                elif age_hours < 168:  # 1 week
                    final_score *= 1.1
                elif age_hours > 720:  # 30 days
                    final_score *= 0.9
            except (ValueError, TypeError):
                pass

        results.append({
            **meta,
            "score": round(final_score, 3),
            "raw_score": round(float(score), 3),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════
#  3. HIERARCHICAL SUMMARIZATION
# ═══════════════════════════════════════════════════════════════

def _load_summaries(chat_id: int) -> Dict[str, Any]:
    """Load all summaries for a chat."""
    path = SUMMARIES_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "session_summaries": [],
        "daily_summaries": [],
        "relationship_summary": "",
        "key_facts": [],
        "last_updated": None,
    }


def _save_summaries(chat_id: int, data: Dict[str, Any]):
    path = SUMMARIES_DIR / f"{chat_id}.json"
    data["last_updated"] = datetime.now().isoformat()
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        context_logger.error(f"Failed to save summaries: {e}")


def create_session_summary(
    chat_id: int,
    messages: List[Dict[str, str]],
    max_messages: int = 50,
) -> str:
    """Create a compressed summary of a conversation session.

    Uses extractive summarization (no LLM call — works offline).
    Identifies key topics, emotional moments, and decisions.
    """
    if not messages:
        return ""

    msgs = messages[-max_messages:]

    # Extract key information
    topics = set()
    emotions = []
    decisions = []
    questions = []
    key_moments = []

    for msg in msgs:
        text = msg.get("text", "").lower()
        sender = msg.get("sender", "?")

        # Topic detection (simple keyword-based)
        topic_keywords = {
            "work": ["work", "job", "boss", "office", "meeting", "project"],
            "relationship": ["love", "miss", "together", "us", "date", "baby"],
            "plans": ["tomorrow", "weekend", "plan", "let's", "gonna", "should we"],
            "feelings": ["feel", "sad", "happy", "angry", "scared", "worried"],
            "daily": ["eat", "sleep", "morning", "night", "today", "dinner"],
            "conflict": ["fight", "argue", "wrong", "fault", "sorry", "hurt"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.add(topic)

        # Questions
        if "?" in text and sender == "Them":
            questions.append(text[:100])

        # Emotional intensity markers
        if any(w in text for w in ["!", "!!", "CAPS", "omg", "wtf", "lol", "😭", "❤"]):
            key_moments.append(f"{sender}: {text[:80]}")

    # Build summary
    parts = []
    if topics:
        parts.append(f"Topics: {', '.join(topics)}")
    if len(msgs) > 0:
        them_count = sum(1 for m in msgs if m.get("sender") == "Them")
        me_count = len(msgs) - them_count
        parts.append(f"Messages: {len(msgs)} ({them_count} from them, {me_count} from me)")
    if questions:
        parts.append(f"They asked: {'; '.join(questions[:3])}")
    if key_moments:
        parts.append(f"Key moments: {'; '.join(key_moments[:3])}")

    summary = " | ".join(parts)

    # Store
    data = _load_summaries(chat_id)
    data["session_summaries"].append({
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(msgs),
        "topics": list(topics),
    })
    # Keep last 50 session summaries
    data["session_summaries"] = data["session_summaries"][-50:]
    _save_summaries(chat_id, data)

    return summary


def create_daily_summary(chat_id: int) -> str:
    """Compress session summaries into a daily summary."""
    data = _load_summaries(chat_id)
    sessions = data.get("session_summaries", [])

    today = datetime.now().date().isoformat()
    today_sessions = [
        s for s in sessions
        if s.get("timestamp", "").startswith(today)
    ]

    if not today_sessions:
        return ""

    all_topics = set()
    total_msgs = 0
    for s in today_sessions:
        all_topics.update(s.get("topics", []))
        total_msgs += s.get("message_count", 0)

    daily = (
        f"[{today}] {len(today_sessions)} conversations, {total_msgs} messages. "
        f"Topics: {', '.join(all_topics) if all_topics else 'general chat'}"
    )

    data["daily_summaries"].append({
        "summary": daily,
        "date": today,
        "session_count": len(today_sessions),
        "total_messages": total_msgs,
        "topics": list(all_topics),
    })
    data["daily_summaries"] = data["daily_summaries"][-90:]  # Keep 3 months
    _save_summaries(chat_id, data)

    return daily


def update_relationship_summary(chat_id: int, key_fact: str = ""):
    """Update the persistent relationship-level summary."""
    data = _load_summaries(chat_id)

    if key_fact and key_fact not in data.get("key_facts", []):
        data.setdefault("key_facts", []).append(key_fact)
        data["key_facts"] = data["key_facts"][-100:]  # Cap at 100 facts

    # Build relationship summary from daily summaries
    daily = data.get("daily_summaries", [])
    if daily:
        all_topics = set()
        total_convos = 0
        for d in daily[-30:]:
            all_topics.update(d.get("topics", []))
            total_convos += d.get("session_count", 0)

        days_active = len(daily)
        data["relationship_summary"] = (
            f"Active for {days_active} days, {total_convos} conversations. "
            f"Common topics: {', '.join(list(all_topics)[:10])}. "
            f"Key facts: {'; '.join(data.get('key_facts', [])[-5:])}"
        )

    _save_summaries(chat_id, data)


# ═══════════════════════════════════════════════════════════════
#  4. TOPIC THREADING
# ═══════════════════════════════════════════════════════════════

def _load_topic_threads(chat_id: int) -> Dict[str, List[Dict]]:
    """Load topic threads for a chat."""
    path = TOPICS_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_topic_threads(chat_id: int, threads: Dict[str, List[Dict]]):
    path = TOPICS_DIR / f"{chat_id}.json"
    try:
        path.write_text(json.dumps(threads, ensure_ascii=False))
    except Exception as e:
        context_logger.error(f"Failed to save topic threads: {e}")


def track_topic(chat_id: int, topic: str, message: str, sender: str = "Them"):
    """Track a message under a topic thread."""
    threads = _load_topic_threads(chat_id)
    topic_key = topic.lower().strip()
    if not topic_key:
        return

    threads.setdefault(topic_key, []).append({
        "text": message[:200],
        "sender": sender,
        "timestamp": datetime.now().isoformat(),
    })
    # Keep last 30 messages per topic
    threads[topic_key] = threads[topic_key][-30:]
    _save_topic_threads(chat_id, threads)


def get_topic_history(chat_id: int, topic: str, limit: int = 10) -> List[Dict]:
    """Get past conversation about a specific topic."""
    threads = _load_topic_threads(chat_id)
    topic_key = topic.lower().strip()
    return threads.get(topic_key, [])[-limit:]


def get_all_topics(chat_id: int) -> List[Dict[str, Any]]:
    """Get all tracked topics with message counts and recency."""
    threads = _load_topic_threads(chat_id)
    result = []
    for topic, messages in threads.items():
        last_msg = messages[-1] if messages else {}
        result.append({
            "topic": topic,
            "message_count": len(messages),
            "last_discussed": last_msg.get("timestamp", ""),
        })
    result.sort(key=lambda x: x.get("last_discussed", ""), reverse=True)
    return result


# ═══════════════════════════════════════════════════════════════
#  5. EMOTIONAL ARC TRACKING
# ═══════════════════════════════════════════════════════════════

def _load_emotional_arc(chat_id: int) -> List[Dict]:
    """Load emotional arc data."""
    path = ARCS_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def _save_emotional_arc(chat_id: int, arc: List[Dict]):
    path = ARCS_DIR / f"{chat_id}.json"
    try:
        # Keep last 500 data points
        path.write_text(json.dumps(arc[-500:], ensure_ascii=False))
    except Exception as e:
        context_logger.error(f"Failed to save emotional arc: {e}")


def record_emotional_state(
    chat_id: int,
    emotion: str,
    valence: float,
    arousal: float,
    sender: str = "Them",
):
    """Record an emotional data point in the arc."""
    arc = _load_emotional_arc(chat_id)
    arc.append({
        "emotion": emotion,
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "sender": sender,
        "timestamp": datetime.now().isoformat(),
    })
    _save_emotional_arc(chat_id, arc)


def get_emotional_trajectory(
    chat_id: int,
    window_hours: int = 24,
) -> Dict[str, Any]:
    """Analyze emotional trajectory over a time window.

    Returns: trend (improving/declining/stable), average valence,
    emotion distribution, volatility score.
    """
    arc = _load_emotional_arc(chat_id)
    if not arc:
        return {"trend": "unknown", "data_points": 0}

    cutoff = datetime.now() - timedelta(hours=window_hours)
    recent = []
    for point in arc:
        try:
            ts = datetime.fromisoformat(point["timestamp"])
            if ts > cutoff:
                recent.append(point)
        except (ValueError, TypeError):
            continue

    if len(recent) < 3:
        return {"trend": "insufficient_data", "data_points": len(recent)}

    # Calculate valence trajectory
    their_points = [p for p in recent if p["sender"] == "Them"]
    if len(their_points) < 2:
        return {"trend": "insufficient_data", "data_points": len(their_points)}

    valences = [p["valence"] for p in their_points]
    avg_valence = sum(valences) / len(valences)

    # Trend: compare first half vs second half
    mid = len(valences) // 2
    first_half = sum(valences[:mid]) / max(mid, 1)
    second_half = sum(valences[mid:]) / max(len(valences) - mid, 1)
    delta = second_half - first_half

    if delta > 0.15:
        trend = "improving"
    elif delta < -0.15:
        trend = "declining"
    else:
        trend = "stable"

    # Volatility (standard deviation)
    variance = sum((v - avg_valence) ** 2 for v in valences) / len(valences)
    volatility = variance ** 0.5

    # Emotion distribution
    emotion_counts = defaultdict(int)
    for p in their_points:
        emotion_counts[p["emotion"]] += 1
    total = sum(emotion_counts.values())
    distribution = {
        k: round(v / total, 2) for k, v in sorted(
            emotion_counts.items(), key=lambda x: -x[1]
        )[:5]
    }

    return {
        "trend": trend,
        "delta": round(delta, 3),
        "average_valence": round(avg_valence, 3),
        "volatility": round(volatility, 3),
        "data_points": len(their_points),
        "emotion_distribution": distribution,
        "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
    }


# ═══════════════════════════════════════════════════════════════
#  6. SMART CONTEXT WINDOW CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def build_advanced_context(
    chat_id: int,
    recent_messages: List[Dict[str, str]],
    incoming_text: str,
    max_context_tokens: int = 2000,
) -> Dict[str, Any]:
    """Build the most information-dense context window possible.

    Combines:
    1. Recent messages (last 15, weighted)
    2. Relevant past messages (RAG retrieval from vector store)
    3. Session/daily summaries (compressed history)
    4. Topic thread context (if topic was discussed before)
    5. Emotional arc summary (trajectory + continuity)
    6. Key relationship facts

    Returns a structured context dict ready for prompt injection.
    """
    context = {
        "recent_messages": [],
        "relevant_past": [],
        "summaries": [],
        "topic_context": [],
        "emotional_arc": {},
        "key_facts": [],
        "context_quality": "basic",
    }

    # 1. Recent messages (always included)
    context["recent_messages"] = recent_messages[-15:]

    # 2. RAG retrieval — find relevant past messages
    if incoming_text and len(incoming_text) > 5:
        relevant = search_vector_store(
            chat_id, incoming_text, top_k=5, min_score=0.35,
        )
        if relevant:
            # Filter out messages already in recent
            recent_texts = {m.get("text", "")[:100] for m in recent_messages[-15:]}
            context["relevant_past"] = [
                r for r in relevant
                if r["text"][:100] not in recent_texts
            ][:5]
            context["context_quality"] = "enhanced"

    # 3. Summaries
    summaries_data = _load_summaries(chat_id)
    if summaries_data.get("relationship_summary"):
        context["summaries"].append({
            "type": "relationship",
            "text": summaries_data["relationship_summary"],
        })
    # Last 3 daily summaries
    for ds in summaries_data.get("daily_summaries", [])[-3:]:
        context["summaries"].append({
            "type": "daily",
            "text": ds["summary"],
        })

    # 4. Topic threads
    if incoming_text:
        # Try to find relevant topic threads
        topics = get_all_topics(chat_id)
        for topic_info in topics[:5]:
            topic = topic_info["topic"]
            if any(w in incoming_text.lower() for w in topic.split()):
                history = get_topic_history(chat_id, topic, limit=3)
                if history:
                    context["topic_context"].append({
                        "topic": topic,
                        "past_messages": history,
                    })
                    context["context_quality"] = "deep"

    # 5. Emotional arc
    arc = get_emotional_trajectory(chat_id, window_hours=24)
    if arc.get("trend") not in ("unknown", "insufficient_data"):
        context["emotional_arc"] = arc
        context["context_quality"] = "deep"

    # 6. Key facts
    context["key_facts"] = summaries_data.get("key_facts", [])[-10:]

    return context


def format_advanced_context_for_prompt(context: Dict[str, Any]) -> str:
    """Format the advanced context into a prompt-ready string.

    Designed to be information-dense but token-efficient.
    """
    parts = []

    # Relationship summary (most compressed, broadest context)
    for s in context.get("summaries", []):
        if s["type"] == "relationship":
            parts.append(f"[Relationship History] {s['text']}")

    # Key facts
    facts = context.get("key_facts", [])
    if facts:
        parts.append("[Known Facts] " + " | ".join(facts[-5:]))

    # Emotional arc
    arc = context.get("emotional_arc", {})
    if arc.get("trend") not in ("unknown", "insufficient_data", None):
        parts.append(
            f"[Emotional Trend] {arc['trend']} "
            f"(valence: {arc.get('average_valence', 0):.1f}, "
            f"volatility: {arc.get('volatility', 0):.1f}, "
            f"dominant: {arc.get('dominant_emotion', '?')})"
        )

    # Topic context (if we've discussed this before)
    for tc in context.get("topic_context", []):
        msgs = tc.get("past_messages", [])
        if msgs:
            topic_lines = [f"{m['sender']}: {m['text']}" for m in msgs[-3:]]
            parts.append(
                f"[Previously discussed '{tc['topic']}'] "
                + " | ".join(topic_lines)
            )

    # Relevant past messages (RAG results)
    relevant = context.get("relevant_past", [])
    if relevant:
        rel_lines = [
            f"{r['sender']}: {r['text'][:100]} (score: {r['score']:.2f})"
            for r in relevant[:3]
        ]
        parts.append("[Relevant Past Messages]\n" + "\n".join(rel_lines))

    # Daily summaries (compressed recent history)
    for s in context.get("summaries", []):
        if s["type"] == "daily":
            parts.append(f"[Recent] {s['text']}")

    if not parts:
        return ""

    return "\n".join(f"- {p}" for p in parts)


# ═══════════════════════════════════════════════════════════════
#  7. CONTEXT INGESTION (call after each message)
# ═══════════════════════════════════════════════════════════════

def ingest_message(
    chat_id: int,
    text: str,
    sender: str = "Them",
    emotion: str = "neutral",
    valence: float = 0.5,
    arousal: float = 0.3,
    topics: Optional[List[str]] = None,
):
    """Ingest a message into all context subsystems.

    Call this for every incoming AND outgoing message to build
    comprehensive context over time.
    """
    if not text or len(text.strip()) < 2:
        return

    timestamp = datetime.now().isoformat()

    # 1. Add to vector store
    add_to_vector_store(
        chat_id, text, sender, timestamp, emotion,
        topic=topics[0] if topics else "",
    )

    # 2. Track topics
    if topics:
        for topic in topics[:3]:
            track_topic(chat_id, topic, text, sender)

    # 3. Record emotional state
    record_emotional_state(chat_id, emotion, valence, arousal, sender)


def flush_context(chat_id: int):
    """Flush all pending data to disk for a chat."""
    save_vector_store(chat_id)
    context_logger.debug(f"Flushed context for chat {chat_id}")


# ═══════════════════════════════════════════════════════════════
#  8. ENGINE STATUS
# ═══════════════════════════════════════════════════════════════

def get_context_engine_status() -> Dict[str, Any]:
    """Get status of the context engine."""
    faiss = _get_faiss()
    embedder = _get_embedder()

    total_vectors = 0
    chat_count = 0
    for cid, store in _faiss_indexes.items():
        total_vectors += store["index"].ntotal
        chat_count += 1

    # Count on-disk stores
    disk_stores = len(list(FAISS_DIR.glob("*.index")))

    return {
        "available": faiss is not None and embedder is not None,
        "faiss_available": faiss is not None,
        "embedder_available": embedder is not None,
        "embedder_model": "all-MiniLM-L6-v2" if embedder else None,
        "loaded_stores": chat_count,
        "disk_stores": disk_stores,
        "total_vectors": total_vectors,
        "summaries_count": len(list(SUMMARIES_DIR.glob("*.json"))),
        "topic_threads_count": len(list(TOPICS_DIR.glob("*.json"))),
        "emotional_arcs_count": len(list(ARCS_DIR.glob("*.json"))),
    }
