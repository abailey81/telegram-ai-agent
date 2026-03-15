"""
Autonomy Engine — Full Conversational Autonomy
================================================
Gives the bot full autonomy to manage conversations independently:

1. Read Receipt Analysis — analyze when messages are read vs replied
2. Proactive Conversation Initiation — don't just wait, start convos
3. Conversation Continuation — detect when convo dies and revive it
4. Double-text Decision System — when to follow up
5. Activity Monitoring — track their online/offline patterns
6. Strategic Silence — when NOT to respond
7. Conversation Flow Management — topic transitions, depth control
8. Multi-conversation Orchestration — manage multiple chats simultaneously
"""

import json
import logging
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

autonomy_logger = logging.getLogger("autonomy_engine")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

AUTONOMY_DATA_DIR = Path("engine_data/autonomy")
AUTONOMY_DATA_DIR.mkdir(parents=True, exist_ok=True)
READ_RECEIPT_DIR = AUTONOMY_DATA_DIR / "read_receipts"
READ_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVITY_DIR = AUTONOMY_DATA_DIR / "activity"
ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. READ RECEIPT ANALYSIS
# ═══════════════════════════════════════════════════════════════

_read_receipt_cache: Dict[int, List[Dict]] = {}


def record_read_receipt(
    chat_id: int,
    message_id: int,
    sent_at: float,
    read_at: Optional[float] = None,
    replied_at: Optional[float] = None,
) -> None:
    """Record a read receipt event for analysis."""
    if chat_id not in _read_receipt_cache:
        _read_receipt_cache[chat_id] = _load_read_receipts(chat_id)

    entry = {
        "message_id": message_id,
        "sent_at": sent_at,
        "read_at": read_at,
        "replied_at": replied_at,
        "read_delay": (read_at - sent_at) if read_at else None,
        "reply_delay": (replied_at - read_at) if (replied_at and read_at) else None,
    }
    _read_receipt_cache[chat_id].append(entry)

    # Keep last 200
    if len(_read_receipt_cache[chat_id]) > 200:
        _read_receipt_cache[chat_id] = _read_receipt_cache[chat_id][-200:]


def analyze_read_patterns(chat_id: int) -> Dict[str, Any]:
    """
    Analyze read receipt patterns to understand their behavior.
    Returns insights about read-to-reply gaps, peak activity, etc.
    """
    receipts = _read_receipt_cache.get(chat_id, [])
    if not receipts:
        return {"status": "no_data"}

    read_delays = [r["read_delay"] for r in receipts if r.get("read_delay") is not None]
    reply_delays = [r["reply_delay"] for r in receipts if r.get("reply_delay") is not None]

    # Messages read but not replied to
    read_no_reply = [r for r in receipts if r.get("read_at") and not r.get("replied_at")]
    read_and_replied = [r for r in receipts if r.get("replied_at")]

    analysis = {
        "total_tracked": len(receipts),
        "read_no_reply_count": len(read_no_reply),
        "read_and_replied_count": len(read_and_replied),
        "reply_rate": len(read_and_replied) / max(len(receipts), 1),
    }

    if read_delays:
        analysis["avg_read_delay_seconds"] = round(sum(read_delays) / len(read_delays))
        analysis["fastest_read_seconds"] = round(min(read_delays))
        # Categorize read speed
        avg_rd = analysis["avg_read_delay_seconds"]
        if avg_rd < 60:
            analysis["read_speed"] = "instant"
        elif avg_rd < 300:
            analysis["read_speed"] = "quick"
        elif avg_rd < 1800:
            analysis["read_speed"] = "moderate"
        elif avg_rd < 7200:
            analysis["read_speed"] = "slow"
        else:
            analysis["read_speed"] = "very_slow"

    if reply_delays:
        analysis["avg_reply_delay_seconds"] = round(sum(reply_delays) / len(reply_delays))
        # Categorize reply speed
        avg_rpd = analysis["avg_reply_delay_seconds"]
        if avg_rpd < 30:
            analysis["reply_speed"] = "instant"
        elif avg_rpd < 120:
            analysis["reply_speed"] = "quick"
        elif avg_rpd < 600:
            analysis["reply_speed"] = "moderate"
        elif avg_rpd < 3600:
            analysis["reply_speed"] = "slow"
        else:
            analysis["reply_speed"] = "very_slow"

    # Trend: are they reading/replying faster or slower over time?
    if len(read_delays) >= 10:
        first_half = read_delays[:len(read_delays)//2]
        second_half = read_delays[len(read_delays)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        if avg_second < avg_first * 0.7:
            analysis["read_trend"] = "getting_faster"
        elif avg_second > avg_first * 1.3:
            analysis["read_trend"] = "getting_slower"
        else:
            analysis["read_trend"] = "stable"
    else:
        analysis["read_trend"] = "insufficient_data"

    # Left on read detection
    recent = receipts[-5:]
    recent_lor = [r for r in recent if r.get("read_at") and not r.get("replied_at")]
    if len(recent_lor) >= 3:
        analysis["left_on_read_pattern"] = True
        analysis["lor_severity"] = "high" if len(recent_lor) >= 4 else "moderate"
    else:
        analysis["left_on_read_pattern"] = False

    return analysis


# ═══════════════════════════════════════════════════════════════
#  2. PROACTIVE CONVERSATION INITIATION
# ═══════════════════════════════════════════════════════════════

# Topic banks for proactive messages
PROACTIVE_TOPICS = {
    "casual_opener": [
        "yo what are u up to",
        "wait i just thought of something",
        "ok random question",
        "so i was thinking",
        "aye",
        "yo u alive",
        "hey wyd",
        "bored, wbu",
    ],
    "callback": [
        "wait remember when u said {topic}",
        "yo that thing u mentioned about {topic}",
        "bro i keep thinking about what u said about {topic}",
    ],
    "late_night": [
        "u still up?",
        "cant sleep",
        "its so late why am i still up",
    ],
    "morning": [
        "morning",
        "gm",
        "yo u up yet",
    ],
    "evening": [
        "how was ur day",
        "what did u end up doing today",
        "hey how was today",
    ],
    "after_silence": [
        "yo stranger",
        "u fell off the face of the earth",
        "so we just not talking anymore or",
        "thought u died lol",
    ],
    "flirty": [
        "was just thinking about u",
        "u crossed my mind",
        "hey",
    ],
}

PROACTIVE_TOPICS_RU = {
    "casual_opener": [
        "чем занимаешься",
        "подожди, я тут подумал(а) кое о чём",
        "рандомный вопрос",
        "короче я тут думал(а)",
        "эй",
        "ты живой/живая вообще",
        "привет, чё делаешь",
        "скучно, а тебе",
    ],
    "callback": [
        "слушай, помнишь ты говорил(а) про {topic}",
        "кстати то что ты рассказывал(а) про {topic}",
        "я всё думаю про то что ты говорил(а) о {topic}",
    ],
    "late_night": [
        "ещё не спишь?",
        "не могу уснуть",
        "блин так поздно а я не сплю",
    ],
    "morning": [
        "доброе утро",
        "утречко",
        "уже проснулся/проснулась?",
    ],
    "evening": [
        "как день прошёл",
        "что в итоге делал(а) сегодня",
        "как сегодня прошло",
    ],
    "after_silence": [
        "эй незнакомец",
        "ты пропал(а) с радаров",
        "мы типа больше не общаемся или как",
        "думал ты пропал(а) хаха",
    ],
    "flirty": [
        "только что о тебе думал(а)",
        "ты мне в голову пришёл/пришла",
        "привет",
    ],
}


def decide_proactive_message(
    chat_id: int,
    time_since_last_msg: float,
    engagement_score: float,
    ghost_risk: float,
    read_analysis: Optional[Dict] = None,
    hour: Optional[int] = None,
    recent_topics: Optional[List[str]] = None,
    language: str = "english",
) -> Optional[Dict[str, Any]]:
    """
    Decide whether to proactively send a message and what to say.
    Returns None if should stay silent, or dict with message + reasoning.
    """
    if hour is None:
        hour = datetime.now().hour

    # Don't message between 1am-7am
    if 1 <= hour < 7:
        return None

    # --- Timing thresholds ---
    # Minimum time before proactive message (don't be clingy)
    min_wait_hours = 2.0
    if engagement_score > 0.7:
        min_wait_hours = 1.5
    elif engagement_score < 0.3:
        min_wait_hours = 6.0  # low engagement: give more space

    if time_since_last_msg < min_wait_hours * 3600:
        return None

    # Maximum time before mandatory check-in
    max_wait_hours = 48.0
    if engagement_score > 0.6:
        max_wait_hours = 24.0

    # --- Language-aware topic bank ---
    _topics = PROACTIVE_TOPICS_RU if language == "russian" else PROACTIVE_TOPICS

    # --- Ghost risk check ---
    if ghost_risk > 0.6:
        # High ghost risk: be careful
        if time_since_last_msg < 24 * 3600:
            return None  # wait at least 24h when ghost risk is high
        # After 24h with high ghost risk, send a casual message
        return {
            "action": "proactive_message",
            "category": "after_silence",
            "message": random.choice(_topics["after_silence"]),
            "reason": "high_ghost_risk_24h_silence",
            "urgency": "low",
        }

    # --- Left on read check ---
    if read_analysis and read_analysis.get("left_on_read_pattern"):
        severity = read_analysis.get("lor_severity", "moderate")
        if severity == "high" and time_since_last_msg < 12 * 3600:
            return None  # don't double-text when being left on read hard
        if time_since_last_msg > 6 * 3600:
            return {
                "action": "proactive_message",
                "category": "casual_opener",
                "message": random.choice(_topics["casual_opener"]),
                "reason": "lor_recovery_attempt",
                "urgency": "low",
            }

    # --- Time-appropriate proactive message ---
    if time_since_last_msg > min_wait_hours * 3600:
        # Pick category based on time of day
        if 7 <= hour < 10:
            category = "morning"
        elif 18 <= hour < 22:
            category = "evening"
        elif 22 <= hour or hour < 1:
            category = "late_night"
        else:
            # During the day: use callback if we have topics
            if recent_topics:
                topic = random.choice(recent_topics)
                template = random.choice(_topics["callback"])
                msg = template.format(topic=topic)
                return {
                    "action": "proactive_message",
                    "category": "callback",
                    "message": msg,
                    "reason": f"natural_callback_to_{topic}",
                    "urgency": "low",
                }
            # Random opener
            if random.random() < 0.4:  # 40% chance to initiate
                category = "casual_opener"
            elif engagement_score > 0.6 and random.random() < 0.3:
                category = "flirty"
            else:
                return None

        msg = random.choice(_topics[category])
        return {
            "action": "proactive_message",
            "category": category,
            "message": msg,
            "reason": f"time_appropriate_{category}",
            "urgency": "normal",
        }

    return None


# ═══════════════════════════════════════════════════════════════
#  3. CONVERSATION CONTINUATION / REVIVAL
# ═══════════════════════════════════════════════════════════════

def should_continue_conversation(
    messages: List[Dict[str, Any]],
    time_since_last: float,
    engagement_score: float,
    conflict_level: str = "none",
) -> Dict[str, Any]:
    """
    Analyze if the conversation has died and decide whether to revive it.
    """
    if not messages:
        return {"should_continue": False, "reason": "no_messages"}

    last_msg = messages[-1] if messages else {}
    last_sender = last_msg.get("sender", "")
    last_text = last_msg.get("text", "").lower().strip()

    # If we sent the last message, it's their turn
    if last_sender in ("Me", "me", "self"):
        # Check if they're leaving us hanging
        if time_since_last > 7200:  # 2 hours
            return {
                "should_continue": False,
                "reason": "waiting_for_their_reply",
                "their_turn": True,
                "can_double_text": time_since_last > 14400,  # 4h
            }
        return {"should_continue": False, "reason": "their_turn", "their_turn": True}

    # They sent the last message — should we respond?
    # Check if the conversation naturally ended
    natural_endings = [
        "bye", "gn", "goodnight", "night", "ttyl", "later", "gotta go",
        "ok bye", "see u", "talk later", "brb",
    ]
    if last_text in natural_endings:
        return {
            "should_continue": False,
            "reason": "natural_ending",
            "ended_naturally": True,
        }

    # Check if it was a dead-end message (no hooks to respond to)
    dead_ends = ["ok", "k", "kk", "sure", "cool", "nice", "yeah", "yep", "mhm", "hmm"]
    if last_text in dead_ends:
        if engagement_score > 0.5:
            return {
                "should_continue": True,
                "reason": "dead_end_but_engaged",
                "strategy": "introduce_new_topic",
            }
        return {
            "should_continue": False,
            "reason": "dead_end_low_engagement",
        }

    # Active conflict — be careful
    if conflict_level == "high":
        return {
            "should_continue": True,
            "reason": "active_conflict",
            "strategy": "address_conflict",
        }

    # Normal conversation — continue if engaged
    return {
        "should_continue": True,
        "reason": "active_conversation",
        "strategy": "natural_flow",
    }


# ═══════════════════════════════════════════════════════════════
#  4. DOUBLE-TEXT DECISION SYSTEM
# ═══════════════════════════════════════════════════════════════

def should_double_text(
    chat_id: int,
    time_since_our_last: float,
    time_since_their_last: float,
    our_last_message: str,
    engagement_score: float,
    ghost_risk: float,
    read_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Decide whether to send a follow-up (double-text).
    Returns decision + what kind of follow-up.
    """
    # Never double-text within 2 hours
    if time_since_our_last < 7200:
        return {"should_double_text": False, "reason": "too_soon"}

    # High ghost risk: don't look desperate
    if ghost_risk > 0.6:
        return {"should_double_text": False, "reason": "high_ghost_risk"}

    # Left on read recently: wait longer
    if read_analysis and read_analysis.get("left_on_read_pattern"):
        if time_since_our_last < 14400:  # 4h
            return {"should_double_text": False, "reason": "left_on_read_wait"}

    # --- When to double text ---
    our_text = our_last_message.lower()

    # If we asked a question and they didn't respond
    if "?" in our_last_message:
        if time_since_our_last > 10800:  # 3h
            return {
                "should_double_text": True,
                "type": "nudge",
                "message_hint": "Short follow-up, dont repeat the question",
                "reason": "unanswered_question",
            }

    # If we sent something boring/dry
    if len(our_text.split()) <= 3 and not any(c in our_text for c in "?!"):
        if time_since_our_last > 14400 and engagement_score > 0.4:
            return {
                "should_double_text": True,
                "type": "topic_change",
                "message_hint": "Change topic entirely, make it interesting",
                "reason": "recover_from_dry_message",
            }

    # Long silence with decent engagement history
    if time_since_our_last > 21600 and engagement_score > 0.5:  # 6h
        return {
            "should_double_text": True,
            "type": "casual_followup",
            "message_hint": "Casual follow-up, different topic",
            "reason": "natural_followup_after_6h",
        }

    return {"should_double_text": False, "reason": "not_needed"}


# ═══════════════════════════════════════════════════════════════
#  5. ACTIVITY PATTERN MONITORING
# ═══════════════════════════════════════════════════════════════

_activity_cache: Dict[int, List[Dict]] = {}


def record_online_status(
    chat_id: int, is_online: bool, timestamp: Optional[float] = None,
) -> None:
    """Record online/offline status change."""
    ts = timestamp or time.time()

    if chat_id not in _activity_cache:
        _activity_cache[chat_id] = []

    # Don't record duplicate states
    if _activity_cache[chat_id]:
        last = _activity_cache[chat_id][-1]
        if last.get("online") == is_online:
            return

    _activity_cache[chat_id].append({
        "timestamp": ts,
        "online": is_online,
        "hour": datetime.fromtimestamp(ts).hour,
    })

    # Keep last 500
    if len(_activity_cache[chat_id]) > 500:
        _activity_cache[chat_id] = _activity_cache[chat_id][-500:]


def analyze_activity_patterns(chat_id: int) -> Dict[str, Any]:
    """Analyze when they're typically online/active."""
    events = _activity_cache.get(chat_id, [])
    if len(events) < 5:
        return {"status": "insufficient_data"}

    # Count online events per hour
    hour_counts = defaultdict(int)
    for e in events:
        if e["online"]:
            hour_counts[e["hour"]] += 1

    total = sum(hour_counts.values()) or 1

    # Peak hours (top 5)
    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [h for h, _ in sorted_hours[:5]]

    # Average session duration
    sessions = []
    for i in range(len(events) - 1):
        if events[i]["online"] and not events[i+1]["online"]:
            duration = events[i+1]["timestamp"] - events[i]["timestamp"]
            if 0 < duration < 86400:  # sanity check
                sessions.append(duration)

    avg_session = sum(sessions) / len(sessions) if sessions else 0

    # Currently online?
    currently_online = events[-1]["online"] if events else False

    # Time since last online
    last_online = None
    for e in reversed(events):
        if e["online"]:
            last_online = e["timestamp"]
            break

    return {
        "status": "analyzed",
        "peak_hours": peak_hours,
        "hourly_distribution": {h: round(c/total, 3) for h, c in sorted_hours},
        "avg_session_seconds": round(avg_session),
        "total_sessions_tracked": len(sessions),
        "currently_online": currently_online,
        "time_since_last_online": round(time.time() - last_online) if last_online else None,
    }


# ═══════════════════════════════════════════════════════════════
#  6. STRATEGIC SILENCE
# ═══════════════════════════════════════════════════════════════

def should_stay_silent(
    incoming_text: str,
    messages: List[Dict[str, Any]],
    engagement_score: float,
    ghost_risk: float,
    conflict_level: str = "none",
    personality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Decide if strategic silence is better than responding.
    Sometimes the best move is to NOT reply.
    """
    text_lower = incoming_text.lower().strip()

    # Never go silent on direct questions or high stakes
    if "?" in incoming_text:
        return {"stay_silent": False, "reason": "direct_question"}

    # --- Situations where silence is strategic ---

    # They sent something passive-aggressive / testing
    test_patterns = [
        "whatever", "k", "fine", "sure", "ok then", "alright then",
        "do what you want", "i dont care", "up to you",
    ]
    if text_lower in test_patterns:
        if engagement_score > 0.5:
            # They're testing — silence can be powerful
            if random.random() < 0.3:
                return {
                    "stay_silent": True,
                    "reason": "strategic_nonresponse_to_test",
                    "duration_hint": "15-45 minutes",
                }

    # They're being contemptuous
    if re.search(r"\b(pathetic|worthless|loser|joke|embarrassing|disgusting|trash|garbage)\b", text_lower) or \
       re.search(r"(жалкий|ничтожество|позор|убогий|лох|неудачник|тупица|бездарь|клоун|посмешище|отстой|позорище|бестолочь|никчёмный)", text_lower):
        if random.random() < 0.4:
            return {
                "stay_silent": True,
                "reason": "dignified_silence_to_contempt",
                "duration_hint": "1-3 hours",
            }

    # We've been over-messaging (low msg_ratio)
    # Check from recent messages
    recent = messages[-10:] if messages else []
    our_recent = sum(1 for m in recent if m.get("sender") in ("Me", "me", "self"))
    if our_recent > 7:
        return {
            "stay_silent": True,
            "reason": "we_are_over_messaging",
            "duration_hint": "wait for them to invest more",
        }

    # React-only opportunity (sticker, meme, etc.)
    if text_lower in ("😂", "🤣", "💀", "😭") or len(text_lower) <= 2:
        if random.random() < 0.4:
            return {
                "stay_silent": True,
                "reason": "react_only_appropriate",
                "action": "send_reaction_instead",
            }

    return {"stay_silent": False}


# ═══════════════════════════════════════════════════════════════
#  7. CONVERSATION FLOW MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def manage_conversation_flow(
    messages: List[Dict[str, Any]],
    situation: Optional[Dict] = None,
    engagement_score: float = 0.5,
    personality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Manage the flow of conversation — topic transitions,
    depth control, pacing.
    """
    if not messages:
        return {"action": "initiate", "suggestion": "Open with something interesting"}

    # Analyze recent conversation
    recent = messages[-15:]
    recent_texts = [m.get("text", "") for m in recent]
    all_recent_text = " ".join(recent_texts).lower()

    # --- Topic staleness detection ---
    # Count unique topic words (rough proxy)
    words = all_recent_text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)

    if unique_ratio < 0.4 and len(recent) > 8:
        return {
            "action": "topic_change",
            "reason": "conversation_getting_stale",
            "suggestion": "Switch to a completely new topic",
            "urgency": "high",
        }

    # --- Depth management ---
    msg_lengths = [len(m.get("text", "").split()) for m in recent]
    avg_length = sum(msg_lengths) / max(len(msg_lengths), 1)

    if avg_length < 4 and len(recent) > 5:
        # Very short messages — conversation is shallow
        if engagement_score > 0.5:
            return {
                "action": "deepen",
                "reason": "conversation_too_shallow",
                "suggestion": "Ask a deeper question or share something meaningful",
            }
        else:
            return {
                "action": "spark",
                "reason": "flat_conversation",
                "suggestion": "Send something unexpected/interesting to spark engagement",
            }

    # --- Question-answer balance ---
    our_questions = sum(1 for m in recent if m.get("sender") in ("Me", "me", "self") and "?" in m.get("text", ""))
    their_questions = sum(1 for m in recent if m.get("sender") in ("Them", "them", "other") and "?" in m.get("text", ""))

    if our_questions > their_questions + 3:
        return {
            "action": "stop_questioning",
            "reason": "asking_too_many_questions",
            "suggestion": "Stop asking questions. Share statements, stories, or opinions instead.",
        }

    # --- Natural pacing ---
    phase = "opening" if len(messages) < 5 else "flowing" if len(messages) < 20 else "deep"

    return {
        "action": "continue_naturally",
        "phase": phase,
        "avg_length": round(avg_length, 1),
        "topic_freshness": round(unique_ratio, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  8. EXPANDED REACTION SYSTEM
# ═══════════════════════════════════════════════════════════════

ADVANCED_REACTION_MAP = {
    # Emotion → reaction candidates with probabilities
    "joy": [("😂", 0.3), ("❤️", 0.2), ("🔥", 0.15), ("👍", 0.1), ("🎉", 0.1)],
    "love": [("❤️", 0.4), ("😍", 0.25), ("🥰", 0.2), ("💕", 0.15)],
    "sadness": [("😢", 0.25), ("❤️", 0.3), ("🥺", 0.2), ("😔", 0.15)],
    "anger": [("👀", 0.2), ("💀", 0.15), ("😬", 0.15), ("🤷", 0.1)],
    "surprise": [("😮", 0.25), ("👀", 0.3), ("😱", 0.15), ("🤯", 0.15)],
    "fear": [("😱", 0.2), ("👀", 0.25), ("😳", 0.15)],
    "excitement": [("🔥", 0.3), ("🎉", 0.2), ("😍", 0.15), ("💯", 0.15)],
    "humor": [("😂", 0.4), ("💀", 0.25), ("🤣", 0.2)],
    "flirty": [("😏", 0.3), ("🔥", 0.25), ("😍", 0.2), ("👀", 0.15)],
    "agreement": [("👍", 0.3), ("💯", 0.25), ("🔥", 0.15), ("✅", 0.1)],
    "disgust": [("🤮", 0.2), ("😬", 0.25), ("💀", 0.2)],
    "sarcasm": [("👀", 0.3), ("💀", 0.25), ("😐", 0.2)],
    "photo_selfie": [("🔥", 0.35), ("😍", 0.3), ("❤️", 0.25), ("👀", 0.15)],
    "photo_food": [("😋", 0.3), ("🔥", 0.2), ("😍", 0.15)],
    "photo_nature": [("😍", 0.3), ("🔥", 0.2), ("❤️", 0.2)],
    "voice_message": [("❤️", 0.2), ("🔥", 0.15), ("👍", 0.15)],
    "sticker": [("😂", 0.3), ("❤️", 0.2), ("💀", 0.15)],
}

# Content-pattern → emotion mapping
CONTENT_EMOTION_MAP = [
    (r"\b(selfie|me rn|this is me|do i look|селфи|фотка|это я|как я выгляжу)\b", "photo_selfie"),
    (r"\b(lol|lmao|haha|😂|🤣|dead|💀|im dying|хаха|ахахах|ржу|угар|лол|умираю)\b", "humor"),
    (r"\b(love|miss|❤️|😍|🥰|babe|baby|cutie|люблю|скучаю|обожаю|зайка|солнышко|котик|малыш)\b", "love"),
    (r"\b(wtf|omg|no way|wait what|seriously|ого|вау|не может быть|серьёзно|офигеть|фигасе)\b", "surprise"),
    (r"\b(hate|angry|pissed|furious|mad|ненавижу|злюсь|бесит|бешусь|в ярости)\b", "anger"),
    (r"\b(sad|crying|😢|😭|depressed|hurt|грустно|плачу|больно|расстроен|расстроена|тоскливо)\b", "sadness"),
    (r"\b(excited|cant wait|so hyped|lets go|🎉|офигеть|не могу дождаться|ура|кайф|нереально)\b", "excitement"),
    (r"\b(exactly|fr|facts|so true|right|100)\b", "agreement"),
    (r"\b(ew|gross|nasty|disgusting)\b", "disgust"),
    (r"\b(sure\.|oh really|wow just wow|suuure|riiiight)\b", "sarcasm"),
    (r"\b(scared|nervous|afraid|worried|terrified)\b", "fear"),
    (r"😏|👀.*🔥|come over|wanna hang|ur cute", "flirty"),
]


def pick_advanced_reaction(
    text: str,
    nlp_analysis: Optional[Dict] = None,
    media_type: Optional[str] = None,
    personality: Optional[Dict] = None,
) -> Optional[str]:
    """
    Advanced reaction picker using emotion detection + personality awareness.
    Returns emoji or None.
    """
    # Base probability of reacting at all
    react_prob = 0.35

    # --- Detect emotion from content ---
    detected_emotion = None
    text_lower = text.lower()

    # Try NLP analysis first
    if nlp_analysis:
        emotion = nlp_analysis.get("emotion") or nlp_analysis.get("primary_emotion", "")
        if isinstance(emotion, dict):
            emotion = emotion.get("primary", "")
        emotion_str = str(emotion).lower()
        emotion_map = {
            "happiness": "joy", "joy": "joy", "love": "love", "sadness": "sadness",
            "anger": "anger", "fear": "fear", "surprise": "surprise",
            "excitement": "excitement", "disgust": "disgust",
        }
        detected_emotion = emotion_map.get(emotion_str)

    # Fallback to content patterns
    if not detected_emotion:
        for pattern, emotion in CONTENT_EMOTION_MAP:
            if re.search(pattern, text_lower, re.I):
                detected_emotion = emotion
                break

    # Media type override
    if media_type == "photo":
        if not detected_emotion:
            detected_emotion = "photo_selfie"  # assume selfie by default
        react_prob = 0.5
    elif media_type == "voice":
        if not detected_emotion:
            detected_emotion = "voice_message"
        react_prob = 0.4
    elif media_type in ("sticker", "gif"):
        detected_emotion = "sticker"
        react_prob = 0.45

    if not detected_emotion:
        return None

    # Roll the dice
    if random.random() > react_prob:
        return None

    # Pick from emotion's reaction pool
    candidates = ADVANCED_REACTION_MAP.get(detected_emotion, [])
    if not candidates:
        return None

    # Weighted random selection
    r = random.random()
    cumulative = 0
    for emoji, prob in candidates:
        cumulative += prob
        if r <= cumulative:
            return emoji

    return candidates[0][0]  # fallback to first


def should_react_only_advanced(
    text: str,
    engagement_score: float,
    media_type: Optional[str] = None,
) -> bool:
    """Decide if we should ONLY react (no text reply)."""
    text_lower = text.lower().strip()

    # Always react-only for certain content
    if media_type in ("sticker", "gif"):
        return random.random() < 0.35

    # Single emoji or very short
    if len(text_lower) <= 2 and not text_lower.isalpha():
        return random.random() < 0.4

    # Just a laugh
    if text_lower in ("lol", "haha", "lmao", "😂", "🤣", "💀",
                      "хаха", "ахах", "ахахах", "ржу", "лол"):
        return random.random() < 0.3

    # Low engagement: react-only saves us from looking desperate
    if engagement_score < 0.3:
        return random.random() < 0.2

    return False


# ═══════════════════════════════════════════════════════════════
#  9. RELEVANT REPLY IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

def identify_relevant_reply_target(
    incoming_text: str,
    recent_messages: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Identify which specific message they might be responding to
    (for smart quote-reply). Returns message_id or None.
    """
    if not recent_messages:
        return None

    text_lower = incoming_text.lower()

    # --- Reference detection ---
    # "that" / "this" / "what you said" → refers to recent our message
    if any(p in text_lower for p in ("that thing", "what you said", "what u said",
                                      "u said", "you said", "that msg", "that message")):
        # Find our most recent message
        for msg in reversed(recent_messages):
            if msg.get("sender") in ("Me", "me", "self") and msg.get("message_id"):
                return msg["message_id"]

    # Word overlap detection — find the message with highest overlap
    incoming_words = set(text_lower.split())
    if len(incoming_words) < 3:
        return None

    best_match = None
    best_overlap = 0

    for msg in recent_messages[-20:]:
        msg_text = msg.get("text", "").lower()
        msg_words = set(msg_text.split())
        overlap = len(incoming_words & msg_words) / max(len(incoming_words | msg_words), 1)
        if overlap > best_overlap and overlap > 0.3:
            best_overlap = overlap
            best_match = msg.get("message_id")

    return best_match


# ═══════════════════════════════════════════════════════════════
#  10. FORMAT FOR PROMPT INJECTION
# ═══════════════════════════════════════════════════════════════

def format_autonomy_for_prompt(
    continuation: Dict[str, Any],
    flow: Dict[str, Any],
    silence_decision: Dict[str, Any],
    read_analysis: Optional[Dict] = None,
) -> str:
    """Format autonomy analysis for prompt injection."""
    parts = []

    # Conversation flow
    action = flow.get("action", "")
    if action == "topic_change":
        parts.append("[FLOW: Conversation getting stale — SWITCH TOPICS]")
    elif action == "deepen":
        parts.append("[FLOW: Too shallow — go deeper, ask or share something meaningful]")
    elif action == "spark":
        parts.append("[FLOW: Flat convo — send something unexpected/interesting]")
    elif action == "stop_questioning":
        parts.append("[FLOW: You're asking too many questions — STOP. Share, don't ask.]")

    # Read receipt insights
    if read_analysis and read_analysis.get("left_on_read_pattern"):
        parts.append("[READ STATUS: They've been leaving you on read — play it cool]")
    if read_analysis:
        rs = read_analysis.get("reply_speed", "")
        if rs == "very_slow":
            parts.append("[They reply very slowly — don't expect fast responses]")
        elif rs == "instant":
            parts.append("[They reply instantly — keep the energy up]")

    # Continuation strategy
    strategy = continuation.get("strategy", "")
    if strategy == "introduce_new_topic":
        parts.append("[STRATEGY: Their last message was a dead end — introduce new topic]")
    elif strategy == "address_conflict":
        parts.append("[STRATEGY: Active conflict — address it head-on]")

    if not parts:
        return ""

    return "\n## CONVERSATION AUTONOMY\n" + "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  11. FULL AUTONOMY PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_autonomy_analysis(
    chat_id: int,
    incoming_text: str,
    messages: List[Dict[str, Any]],
    engagement_score: float = 0.5,
    conflict_level: str = "none",
    ghost_risk: float = 0.0,
    personality: Optional[Dict] = None,
    situation: Optional[Dict] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Run full autonomy analysis pipeline.
    Returns (analysis_dict, prompt_injection_string).
    """
    # Time calculations
    now = time.time()
    their_msgs = [m for m in messages if m.get("sender") in ("Them", "them", "other")]
    our_msgs = [m for m in messages if m.get("sender") in ("Me", "me", "self")]

    last_their_ts = max((m.get("timestamp", 0) for m in their_msgs), default=0)
    last_our_ts = max((m.get("timestamp", 0) for m in our_msgs), default=0)
    time_since_their_last = now - last_their_ts if last_their_ts else 99999
    time_since_our_last = now - last_our_ts if last_our_ts else 99999

    # Read analysis
    read_analysis = analyze_read_patterns(chat_id)

    # Silence decision
    silence = should_stay_silent(
        incoming_text, messages, engagement_score, ghost_risk, conflict_level, personality,
    )

    # Continuation analysis
    continuation = should_continue_conversation(
        messages, time_since_their_last, engagement_score, conflict_level,
    )

    # Flow management
    flow = manage_conversation_flow(
        messages, situation, engagement_score, personality,
    )

    # Reaction
    nlp_a = None  # Will be passed from caller if available
    reaction = pick_advanced_reaction(incoming_text, nlp_a, None, personality)

    # Reply target
    reply_target = identify_relevant_reply_target(incoming_text, messages)

    # Double text check (only if it's our turn)
    double_text = None
    if last_our_ts > last_their_ts:
        double_text = should_double_text(
            chat_id, time_since_our_last, time_since_their_last,
            our_msgs[-1].get("text", "") if our_msgs else "",
            engagement_score, ghost_risk, read_analysis,
        )

    # Prompt injection
    prompt_block = format_autonomy_for_prompt(
        continuation, flow, silence, read_analysis,
    )

    all_analysis = {
        "silence_decision": silence,
        "continuation": continuation,
        "flow": flow,
        "reaction": reaction,
        "reply_target": reply_target,
        "double_text": double_text,
        "read_analysis": read_analysis,
    }

    return all_analysis, prompt_block


# ═══════════════════════════════════════════════════════════════
#  12. PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _load_read_receipts(chat_id: int) -> List[Dict]:
    try:
        path = READ_RECEIPT_DIR / f"{chat_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        autonomy_logger.warning(f"Failed to load read receipts for {chat_id}: {e}")
    return []


def save_read_receipts(chat_id: int) -> None:
    try:
        data = _read_receipt_cache.get(chat_id, [])
        path = READ_RECEIPT_DIR / f"{chat_id}.json"
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        autonomy_logger.warning(f"Failed to save read receipts for {chat_id}: {e}")


def save_activity(chat_id: int) -> None:
    try:
        data = _activity_cache.get(chat_id, [])
        path = ACTIVITY_DIR / f"{chat_id}.json"
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        autonomy_logger.warning(f"Failed to save activity for {chat_id}: {e}")
