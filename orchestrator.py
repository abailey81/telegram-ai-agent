"""
Master Orchestration Engine
============================
The single brain that controls the ENTIRE conversation pipeline.
Every engine, every decision, every timing calculation, every action
flows through this orchestrator.

Architecture:
─────────────
  INCOMING MESSAGE
        │
        ▼
  ┌─────────────────┐
  │  PHASE 1: SENSE │  Parallel analysis (NLP + context + personality + energy)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  PHASE 2: THINK │  Situation assessment + Monte Carlo + predictions
  └────────┬────────┘
           │
           ▼
  ┌──────────────────┐
  │  PHASE 3: DECIDE │  Central decision: reply/silence/react/proactive/delay
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────┐
  │  PHASE 4: ORCHESTRATE│  Build prompt + parallel gen + timing
  └────────┬─────────────┘
           │
           ▼
  ┌──────────────────┐
  │  PHASE 5: EXECUTE│  Send + reactions + post-processing
  └────────┬─────────┘
           │
           ▼
  ┌─────────────────┐
  │  PHASE 6: LEARN │  Record outcomes + update models + state
  └─────────────────┘

Key Principles:
- All engines run through the orchestrator — never directly
- Engines are called in dependency order (personality before thinking)
- Independent analyses run in parallel via asyncio.gather
- Conflicts between engines are resolved centrally
- Every decision has a reasoning trace
- State persists across conversations
"""

import asyncio
import json
import logging
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

orch_logger = logging.getLogger("orchestrator")

# ═══════════════════════════════════════════════════════════════
#  PERSISTENT STATE
# ═══════════════════════════════════════════════════════════════

ORCH_DATA_DIR = Path("engine_data/orchestrator")
ORCH_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Per-chat orchestrator state
_chat_state: Dict[int, Dict[str, Any]] = {}

# Global decision history for learning
_decision_history: List[Dict] = []

# Outcome tracking for closed-loop learning
_outcome_tracker: Dict[int, Dict] = {}

# ─── Auto-pickup of optimized parameters ─────────────────────
_OPTIMIZED_ORCH_PARAMS = None
_OPTIMIZED_ORCH_PARAMS_MTIME = 0


def _load_optimized_orch_params():
    global _OPTIMIZED_ORCH_PARAMS, _OPTIMIZED_ORCH_PARAMS_MTIME
    params_file = Path(__file__).parent / "engine_data" / "optimized_engine_params.json"
    if not params_file.exists():
        return None
    try:
        mtime = params_file.stat().st_mtime
        if mtime != _OPTIMIZED_ORCH_PARAMS_MTIME:
            _OPTIMIZED_ORCH_PARAMS = json.loads(params_file.read_text())
            _OPTIMIZED_ORCH_PARAMS_MTIME = mtime
        return _OPTIMIZED_ORCH_PARAMS
    except Exception:
        return None


def _get_chat_state(chat_id: int) -> Dict[str, Any]:
    """Get or initialize per-chat orchestrator state."""
    if chat_id not in _chat_state:
        _chat_state[chat_id] = {
            "consecutive_silences": 0,
            "consecutive_replies": 0,
            "last_reply_time": 0,
            "last_reaction_time": 0,
            "total_interactions": 0,
            "avg_response_quality": 0.5,
            "current_strategy": "natural",
            "conversation_momentum": "neutral",  # cold → warming → flowing → hot → cooling
            "last_proactive_time": 0,
            "double_text_count_today": 0,
            "reactions_sent_today": 0,
            "messages_sent_today": 0,
            "last_day_reset": datetime.now().date().isoformat(),
        }
    # Daily reset
    today = datetime.now().date().isoformat()
    if _chat_state[chat_id].get("last_day_reset") != today:
        _chat_state[chat_id]["double_text_count_today"] = 0
        _chat_state[chat_id]["reactions_sent_today"] = 0
        _chat_state[chat_id]["messages_sent_today"] = 0
        _chat_state[chat_id]["last_day_reset"] = today
    return _chat_state[chat_id]


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: SENSE — Parallel Analysis
# ═══════════════════════════════════════════════════════════════

class SenseResult:
    """Container for all Phase 1 analysis results."""
    __slots__ = [
        "nlp_analysis", "personality_profile", "personality_prompt",
        "predictions", "prediction_prompt", "energy_info", "energy_constraints",
        "context_v6", "context_prompt", "momentum", "night_adj",
        "media_context", "media_type",
    ]

    def __init__(self):
        self.nlp_analysis = None
        self.personality_profile = None
        self.personality_prompt = ""
        self.predictions = None
        self.prediction_prompt = ""
        self.energy_info = None
        self.energy_constraints = None
        self.context_v6 = None
        self.context_prompt = ""
        self.momentum = None
        self.night_adj = None
        self.media_context = ""
        self.media_type = None


def run_sense_phase(
    chat_id: int,
    incoming_text: str,
    structured_messages: List[Dict],
    engines: Dict[str, Dict],
    username: Optional[str] = None,
    media_context: str = "",
    media_type: Optional[str] = None,
    nlp_analysis_fn: Optional[Callable] = None,
    energy_analysis_fn: Optional[Callable] = None,
    momentum_fn: Optional[Callable] = None,
    night_adj_fn: Optional[Callable] = None,
) -> SenseResult:
    """
    PHASE 1: Run all sensory/analysis engines.
    Non-async analyses run synchronously; results feed into Phase 2.
    """
    result = SenseResult()
    result.media_context = media_context
    result.media_type = media_type

    # NLP Analysis (V3/V2/V1 fallback — handled by caller)
    if nlp_analysis_fn:
        try:
            result.nlp_analysis = nlp_analysis_fn(structured_messages, incoming_text, chat_id, username)
        except Exception as e:
            orch_logger.debug(f"NLP analysis failed: {e}")

    # Energy Analysis
    if energy_analysis_fn:
        try:
            result.energy_info, result.energy_constraints = energy_analysis_fn(incoming_text, chat_id)
        except Exception:
            result.energy_constraints = {}

    # Momentum
    if momentum_fn:
        try:
            result.momentum = momentum_fn(chat_id)
        except Exception:
            pass

    # Night adjustments
    if night_adj_fn:
        try:
            result.night_adj = night_adj_fn(datetime.now().hour)
        except Exception:
            pass

    # Personality Engine (sync — builds profile from message history)
    if "personality" in engines:
        try:
            pe = engines["personality"]
            their_texts = [m["text"] for m in structured_messages if m.get("sender") == "Them" and m.get("text")]
            if len(their_texts) >= 3:
                result.personality_profile, result.personality_prompt = pe["analyze_personality"](chat_id, their_texts)
        except Exception as e:
            orch_logger.debug(f"Personality sense failed: {e}")

    # Prediction Engine (sync — feature extraction + scoring)
    if "prediction" in engines:
        try:
            pred = engines["prediction"]
            pred_messages = []
            for i, m in enumerate(structured_messages):
                pred_messages.append({
                    "sender": m.get("sender", ""),
                    "text": m.get("text", ""),
                    "timestamp": m.get("timestamp", time.time() - (len(structured_messages) - i) * 120),
                })
            result.predictions, result.prediction_prompt = pred["run_full_prediction"](
                chat_id, pred_messages, result.personality_profile,
            )
        except Exception as e:
            orch_logger.debug(f"Prediction sense failed: {e}")

    # Context V6 Engine (sync — ingest + RAG retrieval)
    if "context_v6" in engines:
        try:
            ce = engines["context_v6"]
            ce["ingest_message"](chat_id, incoming_text, "Them")
            result.context_v6 = ce["build_advanced_context"](chat_id, structured_messages, incoming_text)
            result.context_prompt = ce["format_advanced_context_for_prompt"](result.context_v6)
        except Exception as e:
            orch_logger.debug(f"Context V6 sense failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: THINK — Deep Reasoning
# ═══════════════════════════════════════════════════════════════

class ThinkResult:
    """Container for all Phase 2 thinking results."""
    __slots__ = [
        "situation", "monte_carlo", "chain_of_thought",
        "autonomy_analysis", "autonomy_prompt",
        "thinking_prompt", "thinking_raw",
    ]

    def __init__(self):
        self.situation = None
        self.monte_carlo = None
        self.chain_of_thought = ""
        self.autonomy_analysis = None
        self.autonomy_prompt = ""
        self.thinking_prompt = ""
        self.thinking_raw = None


def run_think_phase(
    chat_id: int,
    incoming_text: str,
    structured_messages: List[Dict],
    engines: Dict[str, Dict],
    sense: SenseResult,
) -> ThinkResult:
    """
    PHASE 2: Deep reasoning using sense results.
    Thinking + autonomy engines consume Phase 1 outputs.
    """
    result = ThinkResult()

    # Extract prediction sub-results for convenience
    engagement = (sense.predictions or {}).get("engagement")
    conflict = (sense.predictions or {}).get("conflict")
    ghost = (sense.predictions or {}).get("ghost")
    trajectory = (sense.predictions or {}).get("trajectory")

    # Thinking Engine: situation assessment + Monte Carlo + chain-of-thought
    if "thinking" in engines:
        try:
            te = engines["thinking"]
            result.thinking_raw, result.thinking_prompt = te["think"](
                incoming_text, structured_messages,
                nlp_analysis=sense.nlp_analysis,
                engagement=engagement,
                conflict=conflict,
                personality=sense.personality_profile,
                ghost=ghost,
                trajectory=trajectory,
                n_simulations=50,
            )
            result.situation = result.thinking_raw.get("situation")
            result.monte_carlo = result.thinking_raw.get("monte_carlo")
        except Exception as e:
            orch_logger.debug(f"Thinking phase failed: {e}")

    # Autonomy Engine: flow management + read patterns + strategic decisions
    if "autonomy" in engines:
        try:
            ae = engines["autonomy"]
            result.autonomy_analysis, result.autonomy_prompt = ae["run_autonomy_analysis"](
                chat_id, incoming_text, structured_messages,
                engagement_score=engagement.get("engagement_score", 0.5) if engagement else 0.5,
                conflict_level=conflict.get("level", "none") if conflict else "none",
                ghost_risk=ghost.get("ghost_risk", 0) if ghost else 0,
                personality=sense.personality_profile,
                situation=result.situation,
            )
        except Exception as e:
            orch_logger.debug(f"Autonomy phase failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
#  PHASE 3: DECIDE — Central Decision Engine
# ═══════════════════════════════════════════════════════════════

class Decision:
    """The orchestrator's central decision."""
    __slots__ = [
        "action",           # "reply", "silence", "react_only", "delay_then_reply", "proactive", "double_text"
        "delay_seconds",    # how long to wait before acting
        "max_tokens",       # token budget for generation
        "temperature",      # LLM temperature
        "model",            # which model to use
        "reaction_emoji",   # emoji to react with (or None)
        "use_reply_to",     # whether to quote-reply
        "reply_to_msg_id",  # which message to quote
        "extra_prompt",     # additional prompt injection
        "strategy",         # recommended strategy name
        "strategy_desc",    # strategy description
        "length_hint",      # message length guidance
        "reasoning_trace",  # human-readable reasoning for this decision
        "confidence",       # 0-1 confidence in this decision
        "voice_reply",      # should we send voice instead of text?
        "skip_staleness",   # skip staleness check?
    ]

    def __init__(self):
        self.action = "reply"
        self.delay_seconds = 5.0
        self.max_tokens = 300
        self.temperature = (_load_optimized_orch_params() or {}).get("base_temperature", 0.9)
        self.model = "claude-haiku-4-5-20251001"
        self.reaction_emoji = None
        self.use_reply_to = False
        self.reply_to_msg_id = None
        self.extra_prompt = ""
        self.strategy = "natural"
        self.strategy_desc = ""
        self.length_hint = ""
        self.reasoning_trace = []
        self.confidence = 0.5
        self.voice_reply = False
        self.skip_staleness = False


def run_decide_phase(
    chat_id: int,
    incoming_text: str,
    structured_messages: List[Dict],
    engines: Dict[str, Dict],
    sense: SenseResult,
    think: ThinkResult,
    event_msg_id: int,
    delay_min: float = 3.0,
    delay_max: float = 15.0,
    smart_delay_fn: Optional[Callable] = None,
) -> Decision:
    """
    PHASE 3: Make the central decision.
    Consumes all Phase 1+2 results, resolves conflicts, produces
    a single unified Decision object.
    """
    d = Decision()
    state = _get_chat_state(chat_id)
    trace = d.reasoning_trace

    # ── 3.1: Should we stay SILENT? ──
    silence_decision = (think.autonomy_analysis or {}).get("silence_decision", {})
    if silence_decision.get("stay_silent"):
        reason = silence_decision.get("reason", "unknown")
        trace.append(f"SILENCE: {reason}")
        # Check if we should react instead
        if silence_decision.get("action") == "send_reaction_instead":
            d.action = "react_only"
            d.reaction_emoji = _pick_reaction(incoming_text, sense, think, engines)
            trace.append(f"REACT_ONLY: sending {d.reaction_emoji}")
        else:
            d.action = "silence"
        d.confidence = 0.7
        state["consecutive_silences"] += 1
        return d

    state["consecutive_silences"] = 0

    # ── 3.2: Night mode check ──
    night = sense.night_adj or {}
    if night.get("active"):
        skip_prob = night.get("skip_probability", 0)
        if random.random() < skip_prob:
            d.action = "silence"
            trace.append(f"NIGHT_SKIP: skip_prob={skip_prob}")
            return d
        if night.get("max_tokens_override"):
            d.max_tokens = night["max_tokens_override"]
            trace.append(f"NIGHT_TOKENS: max_tokens={d.max_tokens}")
        if night.get("prompt_addon"):
            d.extra_prompt += f"\n\n{night['prompt_addon']}"

    # ── 3.3: Delay calculation ──
    base_delay = delay_min
    if smart_delay_fn and sense.nlp_analysis:
        try:
            base_delay, delay_reason = smart_delay_fn(
                incoming_text, sense.nlp_analysis, delay_min, delay_max
            )
            trace.append(f"DELAY_BASE: {base_delay:.1f}s ({delay_reason})")
        except Exception:
            base_delay = random.uniform(delay_min, delay_max)
    else:
        base_delay = random.uniform(delay_min, delay_max)

    # Apply engagement-based delay adjustment
    engagement = (sense.predictions or {}).get("engagement", {})
    eng_score = engagement.get("engagement_score", 0.5)
    if eng_score > 0.7:
        base_delay *= 0.7  # reply faster when they're engaged
        trace.append(f"DELAY_ENGAGEMENT: x0.7 (high engagement)")
    elif eng_score < 0.3:
        base_delay *= 1.5  # slow down when disengaged
        trace.append(f"DELAY_ENGAGEMENT: x1.5 (low engagement)")

    # Momentum factor
    if sense.momentum:
        mode = sense.momentum.get("mode", "normal") if isinstance(sense.momentum, dict) else "normal"
        if mode == "rapid":
            base_delay *= 0.5
            trace.append("DELAY_MOMENTUM: x0.5 (rapid mode)")
        elif mode == "cooling":
            base_delay *= 1.3
            trace.append("DELAY_MOMENTUM: x1.3 (cooling mode)")

    # Night multiplier
    if night.get("active"):
        base_delay *= night.get("delay_multiplier", 1.0)
        trace.append(f"DELAY_NIGHT: x{night.get('delay_multiplier', 1.0)}")

    # Clamp delay
    d.delay_seconds = max(1.0, min(base_delay, 120.0))

    # ── 3.4: Strategy selection (from Monte Carlo) ──
    mc = think.monte_carlo or {}
    if mc.get("recommended_strategy"):
        d.strategy = mc["recommended_strategy"]
        d.strategy_desc = mc.get("strategy_description", "")
        d.confidence = mc.get("recommended_score", 0.5)
        trace.append(f"STRATEGY: {d.strategy} ({d.confidence:.0%})")

    # ── 3.5: Dynamic message length (from predictions) ──
    dynamic_len = (sense.predictions or {}).get("dynamic_length", {})
    if dynamic_len:
        d.max_tokens = dynamic_len.get("max_tokens", d.max_tokens)
        d.length_hint = dynamic_len.get("length_hint", "")
        trace.append(f"LENGTH: target={dynamic_len.get('target_words', '?')}w, max_tokens={d.max_tokens}")

    # ── 3.6: Energy mirroring override ──
    if sense.energy_constraints:
        style_hint = sense.energy_constraints.get("style_hint", "")
        max_w = sense.energy_constraints.get("max_words")
        if style_hint:
            d.extra_prompt += f"\n\nIMPORTANT ENERGY MIRRORING: {style_hint}"
        if max_w and max_w < 30:
            d.extra_prompt += f" Keep your reply under {max_w} words."
        if max_w and max_w < 15:
            d.max_tokens = max(40, max_w * 8)
            trace.append(f"ENERGY_CAP: max_tokens={d.max_tokens}")

    # ── 3.7: Reaction decision ──
    d.reaction_emoji = _pick_reaction(incoming_text, sense, think, engines)
    if d.reaction_emoji:
        trace.append(f"REACTION: {d.reaction_emoji}")

    # Check react-only
    if d.reaction_emoji and "autonomy" in engines:
        try:
            ro = engines["autonomy"]["should_react_only_advanced"](
                incoming_text, eng_score, sense.media_type,
            )
            if ro:
                d.action = "react_only"
                trace.append("REACT_ONLY: autonomy engine says react-only")
                return d
        except Exception:
            pass

    # ── 3.8: Quote-reply decision ──
    if "autonomy" in engines:
        try:
            target = engines["autonomy"]["identify_relevant_reply_target"](
                incoming_text, structured_messages[-20:] if structured_messages else [],
            )
            if target:
                d.use_reply_to = True
                d.reply_to_msg_id = target
                trace.append(f"REPLY_TO: msg_id={target} (relevant target)")
        except Exception:
            pass
    # Fallback: reply to their message with some probability
    if not d.use_reply_to:
        if "?" in incoming_text and random.random() < 0.55:
            d.use_reply_to = True
            d.reply_to_msg_id = event_msg_id
            trace.append("REPLY_TO: quote-reply (question detected)")
        elif random.random() < 0.20:
            d.use_reply_to = True
            d.reply_to_msg_id = event_msg_id
            trace.append("REPLY_TO: random variation")

    # ── 3.9: Model selection ──
    # Use reasoning engine tier if available, otherwise heuristic
    conflict_level = (sense.predictions or {}).get("conflict", {}).get("level", "none")
    situation = think.situation or {}
    stakes = situation.get("stakes", "low")

    if stakes == "critical" or conflict_level == "high":
        d.model = "claude-sonnet-4-6"
        d.max_tokens = max(d.max_tokens, 200)
        trace.append(f"MODEL_UPGRADE: sonnet (stakes={stakes}, conflict={conflict_level})")
    elif eng_score > 0.7 and stakes != "low":
        d.model = "claude-sonnet-4-6"
        trace.append("MODEL_UPGRADE: sonnet (high engagement + non-trivial)")

    # ── 3.10: Temperature tuning ──
    _opt = _load_optimized_orch_params() or {}
    if conflict_level == "high":
        d.temperature = _opt.get("conflict_temperature", 0.7)  # more controlled in conflict
        trace.append(f"TEMP: {d.temperature} (conflict)")
    elif d.strategy == "playful_tease":
        d.temperature = _opt.get("creative_temperature", 1.0)  # more creative for banter
        trace.append(f"TEMP: {d.temperature} (playful)")
    elif d.strategy == "direct_honest":
        d.temperature = _opt.get("direct_temperature", 0.8)
        trace.append(f"TEMP: {d.temperature} (direct)")

    # ── 3.11: Voice reply decision ──
    if sense.media_type == "voice_message" and "voice" in engines:
        if random.random() < 0.3:
            d.voice_reply = True
            trace.append("VOICE: will send voice reply")

    # ── 3.12: Build composite extra prompt ──
    # Strategy injection
    if d.strategy and d.strategy != "natural":
        d.extra_prompt += f"\n\nSTRATEGY: Use '{d.strategy.replace('_', ' ')}' approach. {d.strategy_desc}"

    # Length hint
    if d.length_hint:
        d.extra_prompt += f"\n\n{d.length_hint}"

    # Subtext awareness
    subtext = situation.get("subtext")
    if subtext:
        d.extra_prompt += f"\n\nSUBTEXT DETECTED: They might mean '{subtext.replace('_', ' ')}'. Read between the lines."

    # Flow management
    flow = (think.autonomy_analysis or {}).get("flow", {})
    flow_action = flow.get("action", "")
    if flow_action == "topic_change":
        d.extra_prompt += "\n\nFLOW: Conversation is getting stale. SWITCH TOPICS — bring up something new."
    elif flow_action == "deepen":
        d.extra_prompt += "\n\nFLOW: Too shallow. Ask or share something meaningful."
    elif flow_action == "stop_questioning":
        d.extra_prompt += "\n\nFLOW: You're asking too many questions. Share, don't ask."
    elif flow_action == "spark":
        d.extra_prompt += "\n\nFLOW: Conversation is flat. Send something unexpected/interesting."

    # Final action
    d.action = "reply"
    state["consecutive_replies"] += 1
    state["total_interactions"] += 1

    trace.append(f"FINAL: action={d.action}, delay={d.delay_seconds:.1f}s, tokens={d.max_tokens}, model={d.model}")

    return d


def _pick_reaction(
    text: str, sense: SenseResult, think: ThinkResult, engines: Dict
) -> Optional[str]:
    """Pick a reaction emoji using the best available engine."""
    if "autonomy" in engines:
        try:
            return engines["autonomy"]["pick_advanced_reaction"](
                text, sense.nlp_analysis, sense.media_type, sense.personality_profile,
            )
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════
#  PHASE 4: ORCHESTRATE — Build Prompt + Parallel Execution
# ═══════════════════════════════════════════════════════════════

def build_orchestrated_prompt(
    base_system_prompt: str,
    sense: SenseResult,
    think: ThinkResult,
    decision: Decision,
) -> str:
    """
    Build the final system prompt by injecting all engine outputs
    in optimal order. Manages prompt budget to avoid bloat.
    """
    prompt = base_system_prompt

    # === Layer 1: Context (RAG + summaries) ===
    if sense.context_prompt:
        prompt += f"\n\n{sense.context_prompt}"

    # === Layer 2: Personality profile + adaptive directives ===
    if sense.personality_prompt:
        prompt += f"\n\n{sense.personality_prompt}"

    # === Layer 3: Predictive intelligence ===
    if sense.prediction_prompt:
        prompt += f"\n\n{sense.prediction_prompt}"

    # === Layer 4: Advanced reasoning (chain-of-thought) ===
    if think.thinking_prompt:
        prompt += f"\n\n{think.thinking_prompt}"

    # === Layer 5: Autonomy directives ===
    if think.autonomy_prompt:
        prompt += f"\n\n{think.autonomy_prompt}"

    # === Layer 6: Decision-specific extra prompt ===
    if decision.extra_prompt:
        prompt += f"\n\n## Orchestrator Directives:\n{decision.extra_prompt}"

    # === Layer 7: Orchestrator meta-reasoning ===
    if decision.reasoning_trace:
        # Compact trace for the LLM (only key decisions)
        key_decisions = [t for t in decision.reasoning_trace if any(
            k in t for k in ("STRATEGY:", "LENGTH:", "FLOW:", "SUBTEXT", "FINAL:")
        )]
        if key_decisions:
            prompt += "\n\n## Decision Summary:\n" + " | ".join(key_decisions[:5])

    return prompt


async def run_orchestrate_phase(
    decision: Decision,
    generate_reply_fn: Callable,
    delay_prep_fn: Callable,
    chat_id: int,
    incoming_text: str,
    username: Optional[str] = None,
    media_context: str = "",
) -> Tuple[Optional[str], str]:
    """
    PHASE 4: Execute parallel generation + delay.
    Returns (reply_text, delay_result).
    """
    async def _generate():
        return await generate_reply_fn(
            chat_id, incoming_text, username,
            media_context=media_context,
            max_tokens_override=decision.max_tokens,
            extra_system_prompt=decision.extra_prompt,
            temperature_override=decision.temperature,
        )

    async def _delay_and_prep():
        return await delay_prep_fn(decision.delay_seconds)

    gen_start = time.time()
    gen_result, delay_result = await asyncio.gather(_generate(), _delay_and_prep())
    gen_elapsed = time.time() - gen_start

    orch_logger.info(
        f"ORCHESTRATE: parallel complete in {gen_elapsed:.1f}s "
        f"(delay was {decision.delay_seconds:.1f}s, "
        f"gen {'finished first' if gen_elapsed < decision.delay_seconds else 'was the bottleneck'})"
    )

    return gen_result, delay_result


# ═══════════════════════════════════════════════════════════════
#  PHASE 5: EXECUTE — Send + Reactions + Post-Processing
# ═══════════════════════════════════════════════════════════════

class ExecutionPlan:
    """Plan for executing the response."""
    __slots__ = [
        "send_reaction", "reaction_emoji", "reaction_delay",
        "send_text", "text_parts", "use_reply_to", "reply_to_msg_id",
        "send_voice", "voice_path",
        "post_actions",  # list of (action_name, probability) tuples
        "typing_durations",  # per-part typing simulation
    ]

    def __init__(self):
        self.send_reaction = False
        self.reaction_emoji = None
        self.reaction_delay = 0.5
        self.send_text = True
        self.text_parts = []
        self.use_reply_to = False
        self.reply_to_msg_id = None
        self.send_voice = False
        self.voice_path = None
        self.post_actions = []
        self.typing_durations = []


def build_execution_plan(
    decision: Decision,
    reply_text: str,
    split_fn: Optional[Callable] = None,
) -> ExecutionPlan:
    """
    Build a detailed execution plan from the decision + generated reply.
    """
    plan = ExecutionPlan()

    # Reaction
    if decision.reaction_emoji:
        plan.send_reaction = True
        plan.reaction_emoji = decision.reaction_emoji
        plan.reaction_delay = random.uniform(0.3, 1.2)

    # Text splitting
    if decision.action == "react_only":
        plan.send_text = False
    elif reply_text:
        if split_fn:
            plan.text_parts = split_fn(reply_text)
        else:
            # Default splitting on ||
            parts = [p.strip() for p in reply_text.split("||") if p.strip()]
            plan.text_parts = parts if parts else [reply_text]

        # Calculate typing durations per part
        for part in plan.text_parts:
            # Natural typing speed: ~40ms per character, capped at 3s
            duration = max(0.5, min(len(part) * 0.04, 3.0))
            plan.typing_durations.append(duration)

    # Reply-to
    plan.use_reply_to = decision.use_reply_to
    plan.reply_to_msg_id = decision.reply_to_msg_id

    # Voice
    if decision.voice_reply:
        plan.send_voice = True

    # Post-actions (probabilistic)
    plan.post_actions = [
        ("edit_message", 0.03),      # small chance to edit
        ("delete_retype", 0.02),     # small chance to delete and retype
        ("follow_up", 0.08),         # chance for double-text
        ("send_gif", 0.05),          # chance to send GIF
        ("view_stories", 0.10),      # chance to view stories
        ("go_offline", 0.70),        # go offline after replying
    ]

    return plan


# ═══════════════════════════════════════════════════════════════
#  PHASE 6: LEARN — Record Outcomes + Update State
# ═══════════════════════════════════════════════════════════════

def run_learn_phase(
    chat_id: int,
    decision: Decision,
    reply_text: str,
    engines: Dict[str, Dict],
    sense: SenseResult,
) -> None:
    """
    PHASE 6: Record everything for learning.
    Updates state, records outcomes, feeds back to engines.
    """
    state = _get_chat_state(chat_id)
    state["last_reply_time"] = time.time()
    state["messages_sent_today"] += 1

    if decision.reaction_emoji:
        state["last_reaction_time"] = time.time()
        state["reactions_sent_today"] += 1

    # Update conversation momentum
    eng_score = (sense.predictions or {}).get("engagement", {}).get("engagement_score", 0.5)
    if eng_score > 0.7:
        state["conversation_momentum"] = "hot"
    elif eng_score > 0.5:
        state["conversation_momentum"] = "flowing"
    elif eng_score > 0.3:
        state["conversation_momentum"] = "warming"
    else:
        state["conversation_momentum"] = "cold"

    # Record decision for learning
    _decision_history.append({
        "timestamp": time.time(),
        "chat_id": chat_id,
        "action": decision.action,
        "strategy": decision.strategy,
        "confidence": decision.confidence,
        "delay": decision.delay_seconds,
        "max_tokens": decision.max_tokens,
        "engagement_score": eng_score,
        "reply_length": len(reply_text.split()) if reply_text else 0,
        "reasoning": decision.reasoning_trace[:5],
    })
    if len(_decision_history) > 500:
        _decision_history.pop(0)

    # Feed context engine
    if "context_v6" in engines and reply_text:
        try:
            engines["context_v6"]["ingest_message"](chat_id, reply_text, "Me")
        except Exception:
            pass

    # Feed prediction engine
    if "prediction" in engines:
        try:
            engines["prediction"]["record_response_event"](
                chat_id, True, decision.delay_seconds,
            )
            engines["prediction"]["record_activity"](chat_id)
        except Exception:
            pass

    # Store outcome tracker for when they reply (closed-loop)
    _outcome_tracker[chat_id] = {
        "our_reply": reply_text,
        "our_strategy": decision.strategy,
        "our_confidence": decision.confidence,
        "sent_at": time.time(),
        "engagement_at_send": eng_score,
    }


def record_outcome(
    chat_id: int,
    they_replied: bool,
    their_reply_text: str = "",
    reply_delay_seconds: float = 0,
) -> Optional[Dict]:
    """
    Called when THEY reply (or don't) — closes the feedback loop.
    Returns outcome analysis or None.
    """
    if chat_id not in _outcome_tracker:
        return None

    tracker = _outcome_tracker.pop(chat_id)
    outcome = {
        "chat_id": chat_id,
        "strategy_used": tracker["our_strategy"],
        "confidence": tracker["our_confidence"],
        "engagement_at_send": tracker["engagement_at_send"],
        "they_replied": they_replied,
        "reply_delay": reply_delay_seconds,
        "time_to_reply": time.time() - tracker["sent_at"],
    }

    # Score the outcome
    if they_replied:
        # Fast reply = good
        if reply_delay_seconds < 60:
            outcome["score"] = 0.9
        elif reply_delay_seconds < 300:
            outcome["score"] = 0.7
        elif reply_delay_seconds < 1800:
            outcome["score"] = 0.5
        else:
            outcome["score"] = 0.3

        # Long reply = engaged
        if len(their_reply_text.split()) > 10:
            outcome["score"] = min(1.0, outcome["score"] + 0.1)

        # Question = very engaged
        if "?" in their_reply_text:
            outcome["score"] = min(1.0, outcome["score"] + 0.1)
    else:
        outcome["score"] = 0.1  # ghosted

    orch_logger.info(
        f"OUTCOME: strategy={outcome['strategy_used']}, "
        f"score={outcome['score']:.2f}, replied={they_replied}"
    )

    return outcome


# ═══════════════════════════════════════════════════════════════
#  PROACTIVE ORCHESTRATION — When to initiate conversation
# ═══════════════════════════════════════════════════════════════

def should_initiate_proactive(
    chat_id: int,
    engines: Dict[str, Dict],
    time_since_last_msg: float,
) -> Optional[Dict[str, Any]]:
    """
    Decide whether to proactively send a message.
    Called periodically by the proactive loop.
    """
    state = _get_chat_state(chat_id)

    # Rate limit: max 3 proactive messages per day
    if state["messages_sent_today"] > 20:
        return None

    # Minimum gap between proactive messages: 4 hours
    time_since_proactive = time.time() - state.get("last_proactive_time", 0)
    if time_since_proactive < 14400:
        return None

    if "autonomy" not in engines:
        return None

    ae = engines["autonomy"]

    # Get engagement and ghost risk
    eng_score = 0.5
    ghost_risk = 0.0
    if "prediction" in engines:
        try:
            features = engines["prediction"]["extract_conversation_features"]([])
            eng = engines["prediction"]["predict_engagement"](features)
            eng_score = eng.get("engagement_score", 0.5)
            ghost = engines["prediction"]["predict_ghost_risk"](chat_id, features)
            ghost_risk = ghost.get("ghost_risk", 0)
        except Exception:
            pass

    # Get recent topics from context engine
    recent_topics = []
    if "context_v6" in engines:
        try:
            topics = engines["context_v6"]["get_all_topics"](chat_id)
            recent_topics = [t.get("topic", "") for t in (topics or [])[:5]]
        except Exception:
            pass

    try:
        result = ae["decide_proactive_message"](
            chat_id,
            time_since_last_msg=time_since_last_msg,
            engagement_score=eng_score,
            ghost_risk=ghost_risk,
            hour=datetime.now().hour,
            recent_topics=recent_topics,
        )
        if result:
            state["last_proactive_time"] = time.time()
            return result
    except Exception as e:
        orch_logger.debug(f"Proactive decision failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════
#  DOUBLE-TEXT ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

def should_orchestrate_double_text(
    chat_id: int,
    engines: Dict[str, Dict],
    our_last_message: str,
    time_since_our_last: float,
    time_since_their_last: float,
) -> Optional[Dict[str, Any]]:
    """
    Centralized double-text decision.
    """
    state = _get_chat_state(chat_id)

    # Max 2 double texts per day
    if state["double_text_count_today"] >= 2:
        return None

    if "autonomy" not in engines:
        return None

    # Get engagement + ghost risk
    eng_score = 0.5
    ghost_risk = 0.0
    if "prediction" in engines:
        try:
            features = engines["prediction"]["extract_conversation_features"]([])
            eng = engines["prediction"]["predict_engagement"](features)
            eng_score = eng.get("engagement_score", 0.5)
            ghost = engines["prediction"]["predict_ghost_risk"](chat_id, features)
            ghost_risk = ghost.get("ghost_risk", 0)
        except Exception:
            pass

    try:
        ae = engines["autonomy"]
        read_analysis = ae["analyze_read_patterns"](chat_id)
        result = ae["should_double_text"](
            chat_id, time_since_our_last, time_since_their_last,
            our_last_message, eng_score, ghost_risk, read_analysis,
        )
        if result and result.get("should_double_text"):
            state["double_text_count_today"] += 1
            return result
    except Exception as e:
        orch_logger.debug(f"Double-text decision failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════
#  CONFLICT RESOLUTION — When engines disagree
# ═══════════════════════════════════════════════════════════════

def resolve_engine_conflicts(
    sense: SenseResult,
    think: ThinkResult,
    decision: Decision,
) -> Decision:
    """
    Resolve conflicts between engine outputs.
    Higher-priority signals override lower-priority ones.

    Priority order (highest first):
    1. Energy matching (match their energy — aggressive gets aggressive back)
    2. Subtext detection (hidden meaning overrides surface)
    3. Engagement signals (adapt to their investment level)
    4. Personality compatibility (long-term strategy)
    5. Monte Carlo optimal strategy
    6. Conversation flow (keep things moving)
    """
    trace = decision.reasoning_trace

    # CONFLICT 1: Monte Carlo says "challenge_push" but engagement is very low
    mc_strategy = (think.monte_carlo or {}).get("recommended_strategy", "")
    engagement = (sense.predictions or {}).get("engagement", {})
    eng_score = engagement.get("engagement_score", 0.5)

    if mc_strategy == "challenge_push" and eng_score < 0.25:
        decision.strategy = "mysterious_pull"
        decision.strategy_desc = "Low engagement override — pull instead of push"
        trace.append("CONFLICT_RESOLVED: challenge_push→mysterious_pull (low engagement)")

    # CONFLICT 2: Autonomy says topic_change but it's high-stakes conversation
    situation = think.situation or {}
    flow = (think.autonomy_analysis or {}).get("flow", {})
    if flow.get("action") == "topic_change" and situation.get("stakes") in ("critical", "high"):
        # Don't change topic during critical moments
        trace.append("CONFLICT_RESOLVED: blocked topic_change (high stakes)")
        # Remove the flow directive from extra_prompt
        decision.extra_prompt = decision.extra_prompt.replace(
            "\n\nFLOW: Conversation is getting stale. SWITCH TOPICS — bring up something new.", ""
        )

    # CONFLICT 3: Prediction says "ghost risk high" but thinking says "warm temperature"
    ghost = (sense.predictions or {}).get("ghost", {})
    temp = situation.get("emotional_temperature", "neutral")
    if ghost.get("level") in ("moderate", "high") and temp in ("warm", "hot"):
        # They're warm but might ghost — prioritize engagement
        decision.extra_prompt += "\n\nThey're warm but inconsistent. Keep the energy going without being needy."
        trace.append("CONFLICT_RESOLVED: warm+ghost_risk → maintain energy without neediness")

    # CONFLICT 4: Night mode wants short reply but conflict is high
    conflict = (sense.predictions or {}).get("conflict", {})
    night = sense.night_adj or {}
    if night.get("active") and conflict.get("level") == "high":
        # Conflict overrides night mode brevity
        decision.max_tokens = max(decision.max_tokens, 150)
        trace.append("CONFLICT_RESOLVED: night_short overridden by active conflict")

    # CONFLICT 5: Personality says they prefer "long" but engagement is dropping
    if sense.personality_profile:
        pref_len = sense.personality_profile.get("communication_preferences", {}).get("preferred_length", "")
        trajectory = (sense.predictions or {}).get("trajectory", {})
        if pref_len == "long" and trajectory.get("trend") == "cooling_down":
            decision.max_tokens = min(decision.max_tokens, 100)
            decision.extra_prompt += "\n\nKeep it shorter than usual — they seem to be losing interest."
            trace.append("CONFLICT_RESOLVED: personality_long+cooling → capped tokens")

    return decision


# ═══════════════════════════════════════════════════════════════
#  MASTER ORCHESTRATE — Single Entry Point
# ═══════════════════════════════════════════════════════════════

def orchestrate_full_pipeline(
    chat_id: int,
    incoming_text: str,
    structured_messages: List[Dict],
    engines: Dict[str, Dict],
    event_msg_id: int,
    username: Optional[str] = None,
    media_context: str = "",
    media_type: Optional[str] = None,
    nlp_analysis_fn: Optional[Callable] = None,
    energy_analysis_fn: Optional[Callable] = None,
    momentum_fn: Optional[Callable] = None,
    night_adj_fn: Optional[Callable] = None,
    smart_delay_fn: Optional[Callable] = None,
    delay_min: float = 3.0,
    delay_max: float = 15.0,
) -> Tuple[SenseResult, ThinkResult, Decision]:
    """
    MASTER ORCHESTRATION — runs Phases 1-3 synchronously.
    Phase 4 (parallel gen) and Phase 5 (execute) are async
    and handled by the caller.

    Returns (sense, think, decision) tuple for the caller to execute.
    """
    orch_logger.info(f"═══ ORCHESTRATING for chat {chat_id} ═══")
    t0 = time.time()

    # Record incoming for outcome tracking
    if chat_id in _outcome_tracker:
        # They replied to our last message — record outcome
        tracker = _outcome_tracker[chat_id]
        record_outcome(
            chat_id, they_replied=True,
            their_reply_text=incoming_text,
            reply_delay_seconds=time.time() - tracker.get("sent_at", time.time()),
        )

    # PHASE 1: SENSE
    sense = run_sense_phase(
        chat_id, incoming_text, structured_messages, engines,
        username=username, media_context=media_context,
        media_type=media_type, nlp_analysis_fn=nlp_analysis_fn,
        energy_analysis_fn=energy_analysis_fn,
        momentum_fn=momentum_fn, night_adj_fn=night_adj_fn,
    )
    t1 = time.time()
    orch_logger.info(f"  SENSE: {t1-t0:.2f}s")

    # PHASE 2: THINK
    think = run_think_phase(
        chat_id, incoming_text, structured_messages, engines, sense,
    )
    t2 = time.time()
    orch_logger.info(f"  THINK: {t2-t1:.2f}s")

    # PHASE 3: DECIDE
    decision = run_decide_phase(
        chat_id, incoming_text, structured_messages, engines,
        sense, think, event_msg_id,
        delay_min=delay_min, delay_max=delay_max,
        smart_delay_fn=smart_delay_fn,
    )

    # Resolve conflicts between engines
    decision = resolve_engine_conflicts(sense, think, decision)
    t3 = time.time()
    orch_logger.info(f"  DECIDE: {t3-t2:.2f}s")

    orch_logger.info(
        f"  DECISION: action={decision.action}, strategy={decision.strategy}, "
        f"delay={decision.delay_seconds:.1f}s, tokens={decision.max_tokens}, "
        f"model={decision.model}, reaction={decision.reaction_emoji}, "
        f"confidence={decision.confidence:.0%}"
    )
    orch_logger.info(f"  TRACE: {' → '.join(decision.reasoning_trace[:8])}")
    orch_logger.info(f"═══ ORCHESTRATION COMPLETE ({t3-t0:.2f}s total) ═══")

    return sense, think, decision


# ═══════════════════════════════════════════════════════════════
#  STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_orchestrator_state() -> None:
    """Save all orchestrator state to disk."""
    try:
        state_path = ORCH_DATA_DIR / "chat_states.json"
        serializable = {}
        for cid, state in _chat_state.items():
            serializable[str(cid)] = state
        with open(state_path, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        orch_logger.warning(f"Failed to save orchestrator state: {e}")


def load_orchestrator_state() -> None:
    """Load orchestrator state from disk."""
    global _chat_state
    try:
        state_path = ORCH_DATA_DIR / "chat_states.json"
        if state_path.exists():
            with open(state_path) as f:
                raw = json.load(f)
            _chat_state = {int(k): v for k, v in raw.items()}
            orch_logger.info(f"Loaded orchestrator state for {len(_chat_state)} chats")
    except Exception as e:
        orch_logger.warning(f"Failed to load orchestrator state: {e}")


# Auto-load on import
load_orchestrator_state()


# ═══════════════════════════════════════════════════════════════
#  ANALYTICS / INTROSPECTION
# ═══════════════════════════════════════════════════════════════

def get_orchestrator_analytics(chat_id: Optional[int] = None) -> Dict[str, Any]:
    """Get orchestrator analytics for debugging and dashboards."""
    analytics = {
        "total_chats_tracked": len(_chat_state),
        "total_decisions_logged": len(_decision_history),
        "pending_outcomes": len(_outcome_tracker),
    }

    if chat_id and chat_id in _chat_state:
        state = _chat_state[chat_id]
        analytics["chat_state"] = state

        # Strategy distribution for this chat
        chat_decisions = [d for d in _decision_history if d.get("chat_id") == chat_id]
        if chat_decisions:
            strategies = defaultdict(int)
            for d in chat_decisions:
                strategies[d.get("strategy", "unknown")] += 1
            analytics["strategy_distribution"] = dict(strategies)
            analytics["avg_confidence"] = sum(d.get("confidence", 0) for d in chat_decisions) / len(chat_decisions)
            analytics["avg_delay"] = sum(d.get("delay", 0) for d in chat_decisions) / len(chat_decisions)
            analytics["total_decisions_for_chat"] = len(chat_decisions)

    # Global strategy distribution
    if _decision_history:
        global_strats = defaultdict(int)
        for d in _decision_history[-100:]:
            global_strats[d.get("strategy", "unknown")] += 1
        analytics["global_strategy_distribution"] = dict(global_strats)

    return analytics
