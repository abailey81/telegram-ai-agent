"""
Advanced Deep Learning NLP Engine.

Provides transformer-powered analysis that enhances the existing keyword-based
nlp_engine.py. This module adds:

1. Transformer-based sentiment analysis (replaces keyword matching)
2. Multi-label emotion detection (7 emotions with confidence)
3. Semantic similarity using sentence embeddings
4. Zero-shot intent/topic classification
5. Conversation dynamics modeling (momentum, engagement, reciprocity)
6. Neural response quality scoring
7. Semantic memory search (embedding-based)
8. Custom classifier predictions (romantic intent, conversation stage)
9. Ensemble analysis combining DL + heuristic signals

All functions gracefully fall back to None/defaults if models aren't available.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from dl_models import get_model_manager

adv_logger = logging.getLogger("advanced_nlp")
adv_logger.setLevel(logging.INFO)

# ─── Auto-pickup of optimized parameters ─────────────────────
_OPTIMIZED_NLP_PARAMS = None
_OPTIMIZED_NLP_PARAMS_MTIME = 0


def _load_optimized_nlp_params():
    global _OPTIMIZED_NLP_PARAMS, _OPTIMIZED_NLP_PARAMS_MTIME
    params_file = Path(__file__).parent / "engine_data" / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_NLP_PARAMS_MTIME:
            _OPTIMIZED_NLP_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_NLP_PARAMS_MTIME = mtime
        return _OPTIMIZED_NLP_PARAMS
    except Exception:
        return None


# ─── Transformer-Based Sentiment Analysis ───────────────────────

def dl_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Deep learning sentiment analysis using DistilBERT.

    Returns:
        {
            "sentiment": "positive"|"negative",
            "confidence": 0.0-1.0,
            "raw_label": "POSITIVE"|"NEGATIVE",
            "method": "transformer"
        }
    """
    mm = get_model_manager()
    result = mm.analyze_sentiment(text)
    if result is None:
        return None

    return {
        "sentiment": "positive" if result["label"] == "positive" else "negative",
        "confidence": result["score"],
        "raw_label": result["label"],
        "method": "transformer",
    }


# ─── Multi-Label Emotion Detection ─────────────────────────────

def dl_emotions(text: str) -> Optional[Dict[str, Any]]:
    """Detect emotions using DistilRoBERTa (7 emotions).

    Returns:
        {
            "primary_emotion": "joy",
            "primary_confidence": 0.85,
            "all_emotions": {"joy": 0.85, "neutral": 0.08, ...},
            "emotional_intensity": 0.0-1.0,
            "is_emotionally_charged": bool,
            "method": "transformer"
        }
    """
    mm = get_model_manager()
    results = mm.detect_emotions(text)
    if results is None:
        return None

    emotions_dict = {r["label"]: r["score"] for r in results}
    primary = results[0]

    # Emotional intensity = 1 - neutral score
    neutral_score = emotions_dict.get("neutral", 0.0)
    intensity = round(1.0 - neutral_score, 4)

    # Emotionally charged if primary emotion is strong and not neutral
    is_charged = primary["label"] != "neutral" and primary["score"] > 0.4

    return {
        "primary_emotion": primary["label"],
        "primary_confidence": primary["score"],
        "all_emotions": emotions_dict,
        "emotional_intensity": intensity,
        "is_emotionally_charged": is_charged,
        "method": "transformer",
    }


# ─── Semantic Similarity ───────────────────────────────────────

def semantic_similarity(text1: str, text2: str) -> Optional[float]:
    """Compute semantic similarity between two texts using sentence embeddings.

    Returns cosine similarity 0.0-1.0, or None if unavailable.
    """
    mm = get_model_manager()
    emb1 = mm.embed_single(text1)
    emb2 = mm.embed_single(text2)
    if emb1 is None or emb2 is None:
        return None
    return round(mm.cosine_similarity(emb1, emb2), 4)


def semantic_staleness_check(
    proposed: str, recent_responses: List[str], threshold: float = None
) -> Dict[str, Any]:
    """Check response staleness using semantic similarity (much better than Jaccard).

    Returns:
        {
            "is_stale": bool,
            "max_similarity": float,
            "most_similar_to": str|None,
            "method": "semantic"
        }
    """
    if threshold is None:
        threshold = (_load_optimized_nlp_params() or {}).get("staleness_threshold", 0.75)
    mm = get_model_manager()
    if not mm.has_embeddings or not recent_responses:
        return {"is_stale": False, "max_similarity": 0.0, "most_similar_to": None, "method": "fallback"}

    proposed_emb = mm.embed_single(proposed)
    if proposed_emb is None:
        return {"is_stale": False, "max_similarity": 0.0, "most_similar_to": None, "method": "fallback"}

    response_embs = mm.embed(recent_responses)
    if response_embs is None:
        return {"is_stale": False, "max_similarity": 0.0, "most_similar_to": None, "method": "fallback"}

    max_sim = 0.0
    most_similar = None
    for i, emb in enumerate(response_embs):
        sim = mm.cosine_similarity(proposed_emb, emb)
        if sim > max_sim:
            max_sim = sim
            most_similar = recent_responses[i]

    return {
        "is_stale": max_sim > threshold,
        "max_similarity": round(max_sim, 4),
        "most_similar_to": most_similar[:80] if most_similar else None,
        "method": "semantic",
    }


# ─── Zero-Shot Intent Classification ───────────────────────────

INTENT_LABELS = [
    "greeting",
    "asking a question",
    "flirting",
    "sharing emotions",
    "expressing love",
    "being playful",
    "expressing anger or conflict",
    "seeking support",
    "making plans",
    "testing loyalty or feelings",
    "being passive-aggressive",
    "saying goodbye",
    "sharing news",
    "giving a compliment",
]

TOPIC_LABELS = [
    "romance and love",
    "daily life and routine",
    "emotions and feelings",
    "future plans",
    "humor and jokes",
    "physical intimacy",
    "conflict and argument",
    "support and comfort",
    "jealousy",
    "photos and appearance",
    "work or school",
    "food and dining",
    "travel and adventure",
    "hobbies and interests",
]


def dl_classify_intent(text: str) -> Optional[Dict[str, Any]]:
    """Classify message intent using zero-shot NLI.

    Returns:
        {
            "primary_intent": str,
            "confidence": float,
            "all_intents": {label: score, ...},
            "method": "zero-shot"
        }
    """
    mm = get_model_manager()
    result = mm.zero_shot_classify(text, INTENT_LABELS, multi_label=True)
    if result is None:
        return None

    intents_dict = dict(zip(result["labels"], result["scores"]))
    return {
        "primary_intent": result["top_label"],
        "confidence": result["top_score"],
        "all_intents": intents_dict,
        "method": "zero-shot",
    }


def dl_classify_topics(text: str) -> Optional[Dict[str, Any]]:
    """Classify message topics using zero-shot NLI.

    Returns:
        {
            "primary_topic": str,
            "confidence": float,
            "all_topics": {label: score, ...},
            "method": "zero-shot"
        }
    """
    mm = get_model_manager()
    result = mm.zero_shot_classify(text, TOPIC_LABELS, multi_label=True)
    if result is None:
        return None

    topics_dict = dict(zip(result["labels"], result["scores"]))
    return {
        "primary_topic": result["top_label"],
        "confidence": result["top_score"],
        "all_topics": topics_dict,
        "method": "zero-shot",
    }


# ─── Custom Classifier Predictions ─────────────────────────────

def predict_romantic_intent(text: str) -> Optional[Dict[str, Any]]:
    """Predict romantic intent using custom-trained classifier.

    Categories: flirty, romantic, supportive, casual, playful, sincere,
                angry, sad, testing, greeting, goodbye
    """
    mm = get_model_manager()
    return mm.predict_with_custom("romantic_intent", text)


def predict_conversation_stage(text: str) -> Optional[Dict[str, Any]]:
    """Predict conversation stage using custom-trained classifier.

    Stages: opening, warming_up, flowing, deep, conflict, cooling_down, makeup
    """
    mm = get_model_manager()
    return mm.predict_with_custom("conversation_stage", text)


def predict_emotional_tone(text: str) -> Optional[Dict[str, Any]]:
    """Predict emotional tone using custom-trained classifier.

    Tones: joy, love, anger, sadness, surprise, fear, desire,
           tenderness, neutral, playful
    """
    mm = get_model_manager()
    return mm.predict_with_custom("emotional_tone", text)


# ─── Conversation Dynamics Modeling ─────────────────────────────

def analyze_conversation_dynamics(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Analyze conversation dynamics using embedding-based signals.

    Measures:
    - Momentum: are messages becoming more engaged or disengaged?
    - Reciprocity: how well do messages match in tone/length?
    - Topic coherence: how related are consecutive messages?
    - Emotional trajectory: trending positive or negative?

    Args:
        messages: list of {"sender": "Me"|"Them", "text": "..."}
    """
    mm = get_model_manager()
    dynamics = {
        "momentum": "stable",
        "momentum_score": 0.5,
        "reciprocity": 0.5,
        "topic_coherence": 0.5,
        "emotional_trajectory": "stable",
        "engagement_signals": [],
        "method": "heuristic",  # upgraded to "neural" if embeddings available
    }

    if not messages or len(messages) < 4:
        return dynamics

    their_msgs = [m for m in messages if m["sender"] == "Them"]
    our_msgs = [m for m in messages if m["sender"] == "Me"]

    if not their_msgs or not our_msgs:
        return dynamics

    # ── Length-based momentum ──
    if len(their_msgs) >= 4:
        first_half = their_msgs[: len(their_msgs) // 2]
        second_half = their_msgs[len(their_msgs) // 2:]
        avg_first = sum(len(m["text"]) for m in first_half) / len(first_half)
        avg_second = sum(len(m["text"]) for m in second_half) / len(second_half)

        if avg_second > avg_first * 1.3:
            dynamics["momentum"] = "accelerating"
            dynamics["momentum_score"] = min(1.0, 0.5 + (avg_second / max(avg_first, 1) - 1) * 0.5)
            dynamics["engagement_signals"].append("Their messages are getting longer (more engaged)")
        elif avg_second < avg_first * 0.7:
            dynamics["momentum"] = "decelerating"
            dynamics["momentum_score"] = max(0.0, 0.5 - (1 - avg_second / max(avg_first, 1)) * 0.5)
            dynamics["engagement_signals"].append("Their messages are getting shorter (losing interest)")

    # ── Length reciprocity ──
    recent_theirs = their_msgs[-5:]
    recent_ours = our_msgs[-5:]
    if recent_theirs and recent_ours:
        avg_their_len = sum(len(m["text"]) for m in recent_theirs) / len(recent_theirs)
        avg_our_len = sum(len(m["text"]) for m in recent_ours) / len(recent_ours)
        if avg_their_len > 0 and avg_our_len > 0:
            ratio = min(avg_their_len, avg_our_len) / max(avg_their_len, avg_our_len)
            dynamics["reciprocity"] = round(ratio, 3)
            if ratio < 0.3:
                dynamics["engagement_signals"].append(
                    "Message lengths are very unbalanced - match their energy"
                )

    # ── Neural analysis if embeddings available ──
    if mm.has_embeddings:
        dynamics["method"] = "neural"

        # Topic coherence: semantic similarity between consecutive messages
        recent = [m["text"] for m in messages[-8:]]
        if len(recent) >= 2:
            embeddings = mm.embed(recent)
            if embeddings is not None:
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = mm.cosine_similarity(embeddings[i], embeddings[i + 1])
                    similarities.append(sim)
                avg_coherence = sum(similarities) / len(similarities)
                dynamics["topic_coherence"] = round(avg_coherence, 3)

                if avg_coherence > 0.6:
                    dynamics["engagement_signals"].append("Conversation is highly focused and coherent")
                elif avg_coherence < 0.2:
                    dynamics["engagement_signals"].append("Topics are scattered - conversation lacks focus")

    # ── Emotional trajectory ──
    if len(their_msgs) >= 4:
        recent_their = their_msgs[-6:]
        sentiments = []
        for m in recent_their:
            s = dl_sentiment(m["text"])
            if s:
                val = s["confidence"] if s["sentiment"] == "positive" else -s["confidence"]
                sentiments.append(val)

        if len(sentiments) >= 3:
            first_half_avg = sum(sentiments[: len(sentiments) // 2]) / max(len(sentiments) // 2, 1)
            second_half_avg = sum(sentiments[len(sentiments) // 2:]) / max(
                len(sentiments) - len(sentiments) // 2, 1
            )

            if second_half_avg > first_half_avg + 0.2:
                dynamics["emotional_trajectory"] = "improving"
            elif second_half_avg < first_half_avg - 0.2:
                dynamics["emotional_trajectory"] = "declining"

    return dynamics


# ─── Neural Response Quality Scoring ───────────────────────────

def score_response_quality(
    proposed_response: str,
    their_last_message: str,
    conversation_context: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Score a proposed response using multiple neural signals.

    Scoring dimensions:
    1. Relevance: semantic similarity to their message (should be moderate)
    2. Tone match: emotional alignment
    3. Length appropriateness: compared to their messages
    4. Naturalness: formality detection
    5. Staleness: similarity to recent responses

    Returns:
        {
            "score": 0-100,
            "grade": "A"-"F",
            "dimensions": {...},
            "feedback": [...],
            "method": "neural"|"heuristic"
        }
    """
    mm = get_model_manager()
    score = 100
    feedback = []
    dimensions = {}
    method = "heuristic"

    # ── 1. Relevance (semantic similarity) ──
    relevance = semantic_similarity(proposed_response, their_last_message)
    if relevance is not None:
        method = "neural"
        dimensions["relevance"] = relevance
        # Sweet spot: 0.3-0.7 (related but not parroting)
        if relevance > 0.85:
            score -= 15
            feedback.append("Response is too similar to their message. Don't just parrot back.")
        elif relevance < 0.1:
            score -= 10
            feedback.append("Response seems disconnected from what they said.")
        elif 0.3 <= relevance <= 0.7:
            feedback.append("Good topical relevance to their message.")

    # ── 2. Tone match (emotion alignment) ──
    their_emotions = dl_emotions(their_last_message)
    our_emotions = dl_emotions(proposed_response)
    if their_emotions and our_emotions:
        method = "neural"
        their_primary = their_emotions["primary_emotion"]
        our_primary = our_emotions["primary_emotion"]
        dimensions["their_emotion"] = their_primary
        dimensions["our_emotion"] = our_primary

        # Emotional mismatches to penalize
        bad_matches = {
            ("sadness", "joy"): -15,
            ("anger", "joy"): -20,
            ("fear", "joy"): -10,
            ("sadness", "neutral"): -5,
            ("anger", "neutral"): -10,
        }
        penalty = bad_matches.get((their_primary, our_primary), 0)
        if penalty:
            score += penalty
            feedback.append(
                f"Tone mismatch: they feel {their_primary} but your response conveys {our_primary}."
            )

        # Good emotional matches
        good_matches = {
            ("joy", "joy"),
            ("anger", "sadness"),  # showing empathy
            ("sadness", "sadness"),  # empathy
            ("fear", "neutral"),  # calming
        }
        if (their_primary, our_primary) in good_matches:
            score += 5
            feedback.append("Good emotional tone match.")

    # ── 3. Length appropriateness ──
    their_recent = [m for m in conversation_context if m["sender"] == "Them"][-5:]
    if their_recent:
        avg_their_len = sum(len(m["text"]) for m in their_recent) / len(their_recent)
        our_len = len(proposed_response)
        ratio = our_len / max(avg_their_len, 1)
        dimensions["length_ratio"] = round(ratio, 2)

        if ratio > 3.0:
            score -= 15
            feedback.append("Response is much longer than their messages. Consider shortening.")
        elif ratio < 0.15 and avg_their_len > 30:
            score -= 10
            feedback.append("Response is very short compared to their messages.")

    # ── 4. Naturalness checks ──
    formal_words = [
        "therefore", "however", "furthermore", "nevertheless", "moreover",
        "indeed", "thus", "hence", "consequently", "accordingly",
        "in conclusion", "additionally", "regarding",
    ]
    formal_count = sum(1 for w in formal_words if w in proposed_response.lower())
    if formal_count > 0:
        score -= formal_count * 8
        feedback.append("Response sounds too formal for texting.")
    dimensions["formality_issues"] = formal_count

    # AI-sounding phrases
    ai_phrases = [
        "i understand that", "that being said", "it's important to note",
        "i appreciate you sharing", "i want you to know that",
        "i completely understand", "firstly", "secondly",
        "in terms of", "with that being said",
    ]
    ai_count = sum(1 for p in ai_phrases if p in proposed_response.lower())
    _ai_penalty = (_load_optimized_nlp_params() or {}).get("ai_detection_penalty", -10)
    if ai_count > 0:
        score += ai_count * _ai_penalty
        feedback.append("Response contains AI-sounding phrases. Rephrase naturally.")
    dimensions["ai_phrase_count"] = ai_count

    # ── 5. Staleness check ──
    our_recent = [m["text"] for m in conversation_context if m["sender"] == "Me"][-15:]
    if our_recent:
        staleness = semantic_staleness_check(proposed_response, our_recent)
        dimensions["staleness"] = staleness["max_similarity"]
        if staleness["is_stale"]:
            score += (_load_optimized_nlp_params() or {}).get("repetition_penalty", -15)
            feedback.append(
                f"Response is too similar to a recent one (similarity: {staleness['max_similarity']:.0%})."
            )

    # Clamp and grade
    score = max(0, min(100, score))
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    if not feedback:
        feedback.append("Response looks natural and well-matched.")

    return {
        "score": score,
        "grade": grade,
        "dimensions": dimensions,
        "feedback": feedback,
        "method": method,
    }


# ─── Semantic Memory Search ─────────────────────────────────────

def search_memory_semantically(
    query: str,
    memory_notes: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Search conversation memory notes using semantic similarity.

    Instead of keyword matching, finds notes that are semantically related
    to the current message context.

    Args:
        query: the current message or context
        memory_notes: list of {"text": str, "added": str} dicts
        top_k: number of results to return

    Returns: list of {"note": str, "similarity": float}
    """
    mm = get_model_manager()
    if not mm.has_embeddings or not memory_notes:
        return []

    note_texts = [n["text"] if isinstance(n, dict) else str(n) for n in memory_notes]
    if not note_texts:
        return []

    query_emb = mm.embed_single(query)
    if query_emb is None:
        return []

    note_embs = mm.embed(note_texts)
    if note_embs is None:
        return []

    results = []
    for i, (note, emb) in enumerate(zip(note_texts, note_embs)):
        sim = mm.cosine_similarity(query_emb, emb)
        results.append({"note": note, "similarity": round(sim, 4)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ─── Comprehensive Analysis (V3) ───────────────────────────────

def deep_analyze(
    messages: List[Dict[str, str]],
    incoming_text: str,
    chat_id: int,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Comprehensive deep learning analysis of conversation context.

    This is the primary analysis function that combines all DL capabilities.
    Designed to be called alongside (not replace) the existing analyze_context_v2.

    Returns a rich analysis dict with all neural signals.
    """
    mm = get_model_manager()
    analysis = {
        "dl_available": mm.is_available,
        "method": "neural" if mm.is_available else "unavailable",
    }

    if not mm.is_available:
        adv_logger.warning("DL models not available, returning empty analysis")
        return analysis

    # ── 1. Deep sentiment ──
    sentiment = dl_sentiment(incoming_text)
    if sentiment:
        analysis["dl_sentiment"] = sentiment

    # ── 2. Emotion detection ──
    emotions = dl_emotions(incoming_text)
    if emotions:
        analysis["dl_emotions"] = emotions

    # ── 3. Intent classification ──
    intent = dl_classify_intent(incoming_text)
    if intent:
        analysis["dl_intent"] = intent

    # ── 4. Topic classification ──
    topics = dl_classify_topics(incoming_text)
    if topics:
        analysis["dl_topics"] = topics

    # ── 5. Custom classifier predictions ──
    romantic_intent = predict_romantic_intent(incoming_text)
    if romantic_intent:
        analysis["romantic_intent"] = romantic_intent

    conv_stage = predict_conversation_stage(incoming_text)
    if conv_stage:
        analysis["predicted_stage"] = conv_stage

    emotional_tone = predict_emotional_tone(incoming_text)
    if emotional_tone:
        analysis["emotional_tone"] = emotional_tone

    # ── 6. Conversation dynamics ──
    dynamics = analyze_conversation_dynamics(messages)
    analysis["conversation_dynamics"] = dynamics

    # ── 7. Semantic memory search ──
    if memory and memory.get("notes"):
        relevant_notes = search_memory_semantically(incoming_text, memory["notes"])
        if relevant_notes:
            analysis["relevant_memories"] = relevant_notes

    # ── 8. Emotional trajectory of their messages ──
    their_msgs = [m for m in messages if m["sender"] == "Them"]
    if len(their_msgs) >= 3:
        emotion_trajectory = []
        for m in their_msgs[-5:]:
            e = dl_emotions(m["text"])
            if e:
                emotion_trajectory.append({
                    "text_preview": m["text"][:40],
                    "emotion": e["primary_emotion"],
                    "intensity": e["emotional_intensity"],
                })
        if emotion_trajectory:
            analysis["their_emotion_trajectory"] = emotion_trajectory

    # ── 9. Message embedding for future comparisons ──
    embedding = mm.embed_single(incoming_text)
    if embedding is not None:
        analysis["has_embedding"] = True
        # Don't include the actual embedding vector in the output (too large)
        # It's available via mm.embed_single() for downstream use

    return analysis


def format_deep_analysis(analysis: Dict[str, Any]) -> str:
    """Format deep analysis into a prompt section for Claude.

    Designed to complement (not duplicate) format_context_v2.
    Only includes DL-specific insights not already in the heuristic analysis.
    """
    if not analysis.get("dl_available"):
        return ""

    parts = []

    # DL Sentiment (more nuanced than keyword-based)
    dl_sent = analysis.get("dl_sentiment")
    if dl_sent:
        parts.append(
            f"Neural sentiment: {dl_sent['sentiment']} "
            f"(confidence: {dl_sent['confidence']:.0%})"
        )

    # Emotions (not available in keyword-based system)
    emotions = analysis.get("dl_emotions")
    if emotions and emotions["is_emotionally_charged"]:
        parts.append(
            f"Detected emotion: {emotions['primary_emotion']} "
            f"(intensity: {emotions['emotional_intensity']:.0%})"
        )
        # Show secondary emotions if significant
        secondary = [
            f"{k}={v:.0%}"
            for k, v in emotions["all_emotions"].items()
            if v > 0.15 and k != emotions["primary_emotion"]
        ]
        if secondary:
            parts.append(f"Other emotions: {', '.join(secondary)}")

    # Intent (more precise than topic detection)
    intent = analysis.get("dl_intent")
    if intent and intent["confidence"] > 0.3:
        parts.append(f"Detected intent: {intent['primary_intent']} ({intent['confidence']:.0%})")

    # Romantic intent from custom classifier
    ri = analysis.get("romantic_intent")
    if ri and ri["confidence"] > 0.4:
        parts.append(f"Romantic context: {ri['label']} ({ri['confidence']:.0%})")

    # Emotional tone from custom classifier
    tone = analysis.get("emotional_tone")
    if tone and tone["confidence"] > 0.4:
        parts.append(f"Emotional tone: {tone['label']} ({tone['confidence']:.0%})")

    # Conversation dynamics
    dynamics = analysis.get("conversation_dynamics", {})
    if dynamics.get("method") == "neural":
        if dynamics["momentum"] != "stable":
            parts.append(f"Conversation momentum: {dynamics['momentum']}")
        if dynamics["emotional_trajectory"] != "stable":
            parts.append(f"Emotional trajectory: {dynamics['emotional_trajectory']}")
        for signal in dynamics.get("engagement_signals", [])[:2]:
            parts.append(f"Signal: {signal}")

    # Relevant memories (semantic search)
    memories = analysis.get("relevant_memories", [])
    if memories:
        top_mem = memories[0]
        if top_mem["similarity"] > 0.4:
            parts.append(f"Relevant memory: \"{top_mem['note']}\" (match: {top_mem['similarity']:.0%})")

    # Emotion trajectory
    trajectory = analysis.get("their_emotion_trajectory", [])
    if len(trajectory) >= 2:
        recent_emotions = [t["emotion"] for t in trajectory]
        if len(set(recent_emotions)) > 1:
            parts.append(f"Their emotional arc: {' → '.join(recent_emotions)}")

    if not parts:
        return ""

    return "\n".join(f"- [DL] {p}" for p in parts)


# ─── Model Status Endpoint ─────────────────────────────────────

def get_dl_status() -> Dict[str, Any]:
    """Get status of all deep learning capabilities."""
    mm = get_model_manager()
    return mm.get_status()
