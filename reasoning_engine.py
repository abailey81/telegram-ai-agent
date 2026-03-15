"""
Advanced Reasoning Engine.

Implements sophisticated multi-step reasoning for the conversation agent:

1. Chain-of-Thought (CoT) Reasoning - Structured thinking before responding
2. Multi-Hypothesis Response Generation - Generate and rank alternatives
3. Confidence Scoring - Know when to be certain vs. cautious
4. Response Selection Framework - Multi-dimensional scoring
5. Scaling via Cascading Models - Use cheap models for routing, expensive for complex
6. Meta-Cognitive Monitoring - Self-assess response quality
7. Conflict Resolution Between Signals - When analyses disagree
8. Adaptive Complexity Scaling - Match response sophistication to message complexity

Based on research from:
- Google DeepMind's Chain-of-Thought prompting
- Tree-of-Thought reasoning frameworks
- Self-Reflective RAG patterns
- Plan-and-Execute agent architecture
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

reason_logger = logging.getLogger("reasoning_engine")
reason_logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════
#  1. CHAIN-OF-THOUGHT REASONING
# ═══════════════════════════════════════════════════════════════

def build_reasoning_chain(
    incoming_text: str,
    conversation_state: Dict[str, Any],
    emotional_context: Dict[str, Any],
    style_context: Dict[str, Any],
    memory_context: str = "",
) -> Dict[str, Any]:
    """Build a structured reasoning chain before response generation.

    This provides explicit step-by-step reasoning that guides
    the model's response generation, similar to CoT prompting.

    Steps:
    1. Understand - What are they saying/asking/feeling?
    2. Assess - What's the emotional/relational context?
    3. Recall - What relevant memories/patterns exist?
    4. Plan - What should the response achieve?
    5. Constrain - What should be avoided?
    6. Execute - How should the response be structured?
    """
    chain = {
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "final_directive": "",
        "confidence": 0.0,
        "complexity_level": "standard",
    }

    text_lower = incoming_text.lower().strip()

    # Step 1: UNDERSTAND
    understanding = _step_understand(text_lower, incoming_text)
    chain["steps"].append({"step": "understand", **understanding})

    # Step 2: ASSESS
    assessment = _step_assess(
        emotional_context,
        conversation_state,
        style_context,
    )
    chain["steps"].append({"step": "assess", **assessment})

    # Step 3: RECALL
    recall = _step_recall(text_lower, memory_context)
    chain["steps"].append({"step": "recall", **recall})

    # Step 4: PLAN
    plan = _step_plan(understanding, assessment, recall)
    chain["steps"].append({"step": "plan", **plan})

    # Step 5: CONSTRAIN
    constraints = _step_constrain(assessment, understanding)
    chain["steps"].append({"step": "constrain", **constraints})

    # Step 6: EXECUTE
    execution = _step_execute(plan, constraints, style_context)
    chain["steps"].append({"step": "execute", **execution})

    # Compute overall confidence
    step_confidences = [s.get("confidence", 0.5) for s in chain["steps"]]
    chain["confidence"] = round(sum(step_confidences) / len(step_confidences), 3)

    # Determine complexity level
    chain["complexity_level"] = _determine_complexity(understanding, assessment)

    # Build final directive
    chain["final_directive"] = _build_final_directive(chain)

    return chain


def _step_understand(text_lower: str, original_text: str) -> Dict[str, Any]:
    """Step 1: Understand what they're saying."""
    result = {
        "message_type": "statement",
        "contains_question": False,
        "contains_emotion": False,
        "contains_request": False,
        "contains_opinion": False,
        "contains_story": False,
        "topics_mentioned": [],
        "implicit_meaning": None,
        "confidence": 0.5,  # Base: computed from signals found below
    }
    _signal_count = 0  # Track how many signals we detect → higher = more confident

    # Message type classification (ordered by specificity)
    if "?" in original_text:
        result["message_type"] = "question"
        result["contains_question"] = True
        _signal_count += 1
        # Sub-classify question type
        if any(w in text_lower for w in ["should i", "what should", "advice", "recommend",
                                          "как думаешь", "что посоветуешь"]):
            result["message_type"] = "advice_seeking"
        elif any(w in text_lower for w in ["what do you think", "your opinion", "agree",
                                            "как считаешь", "согласен"]):
            result["message_type"] = "opinion_seeking"
    elif any(w in text_lower for w in ["please", "can you", "could you", "would you",
                                        "help", "пожалуйста", "помоги", "можешь"]):
        result["message_type"] = "request"
        result["contains_request"] = True
        _signal_count += 1
    elif any(w in text_lower for w in ["i feel", "i'm feeling", "i am so", "makes me",
                                        "чувствую", "мне так"]):
        result["message_type"] = "emotional_expression"
        result["contains_emotion"] = True
        _signal_count += 1
    elif any(w in text_lower for w in ["guess what", "you won't believe", "just found out",
                                        "big news", "представляешь", "не поверишь"]):
        result["message_type"] = "news_sharing"
    elif any(w in text_lower for w in ["i think", "in my opinion", "i believe", "imo",
                                        "hot take", "я считаю", "по-моему"]):
        result["message_type"] = "opinion"
        result["contains_opinion"] = True
    elif any(w in text_lower for w in ["so basically", "let me tell you", "story time",
                                        "you know what happened", "short version",
                                        "короче", "представь", "знаешь что было"]):
        result["message_type"] = "storytelling"
        result["contains_story"] = True
    elif any(w in text_lower for w in ["ugh", "can't deal", "so annoyed", "need to rant",
                                        "fml", "бесит", "достало", "задолбало"]):
        result["message_type"] = "venting"
        result["contains_emotion"] = True
    elif any(w in text_lower for w in ["check this", "look at this", "have you seen",
                                        "посмотри", "глянь", "видел"]):
        result["message_type"] = "sharing_content"
    elif any(w in text_lower for w in ["haha", "lol", "😂", "🤣", "lmao", "dead",
                                        "хаха", "ахахах"]):
        result["message_type"] = "reactive"
    elif len(text_lower.split()) <= 3:
        result["message_type"] = "brief_response"

    # Detect implicit meanings (expanded for general-purpose)
    implicit_patterns = {
        "need_reassurance": [
            "do you still", "are you sure", "am i", "you promise",
            "ты точно", "ты уверен",
        ],
        "testing_boundaries": [
            "what if", "would you ever", "what would you do",
        ],
        "seeking_attention": [
            "i'm bored", "nobody", "no one", "all alone",
            "скучно", "никто", "одна", "один",
        ],
        "indirect_request": [
            "it would be nice", "i wish", "if only",
            "было бы круто", "хотелось бы",
        ],
        "passive_aggression": [
            "fine", "whatever", "k", "ok then", "if you say so",
            "ладно", "как хочешь", "мне всё равно",
        ],
        "vulnerability": [
            "never told", "scared to say", "don't judge me", "honestly",
            "никому не говорил", "не суди",
        ],
        "excitement_sharing": [
            "omg", "oh my god", "you won't believe", "guess what", "no way",
            "ааа", "офигеть", "не поверишь",
        ],
        "frustration_venting": [
            "i can't", "so annoyed", "i'm done with", "sick of", "fed up",
            "не могу больше", "задолбало", "надоело",
        ],
        "brainstorming": [
            "what if we", "how about", "idea", "what do you think about",
            "а что если", "есть идея",
        ],
        "seeking_validation": [
            "was i right", "did i do the right thing", "is it normal",
            "я правильно", "это нормально",
        ],
        "processing_out_loud": [
            "i've been thinking", "i realized", "it hit me", "makes sense now",
            "я подумал", "до меня дошло",
        ],
    }
    # Collect ALL matching implicit meanings with match strength
    _implicit_matches = []
    for meaning, patterns in implicit_patterns.items():
        _matched = [p for p in patterns if p in text_lower]
        if _matched:
            # Strength = number of pattern matches + how much of the text they cover
            _coverage = sum(len(p) for p in _matched) / max(len(text_lower), 1)
            _strength = len(_matched) * 0.3 + _coverage * 0.7
            _implicit_matches.append((meaning, _strength, _matched))

    if _implicit_matches:
        # Sort by strength — strongest signal wins
        _implicit_matches.sort(key=lambda x: x[1], reverse=True)
        result["implicit_meaning"] = _implicit_matches[0][0]
        _signal_count += 1
        # Store secondary meaning if multiple strong matches (adds nuance)
        if len(_implicit_matches) > 1 and _implicit_matches[1][1] > 0.15:
            result["secondary_implicit"] = _implicit_matches[1][0]
            _signal_count += 1

    # Compute confidence from signal density: 0 signals = 0.4, 1 = 0.55, 2 = 0.7, 3+ = 0.8+
    result["confidence"] = min(0.4 + _signal_count * 0.15, 0.9)
    return result


def _step_assess(
    emotional_context: Dict[str, Any],
    conversation_state: Dict[str, Any],
    style_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 2: Assess the emotional and relational context."""
    profile = emotional_context.get("emotional_profile", {})
    validation = emotional_context.get("validation_guidance", {})
    state = conversation_state

    result = {
        "emotional_temperature": "neutral",
        "relationship_moment": "normal",
        "urgency": "normal",
        "needs_special_handling": False,
        "special_handling_type": None,
        "confidence": 0.5,
    }
    _assess_signals = 0

    # Emotional temperature
    valence = profile.get("valence", 0.5)
    arousal = profile.get("arousal", 0.3)

    if valence < 0.3 and arousal > 0.6:
        result["emotional_temperature"] = "hot_negative"
        result["needs_special_handling"] = True
        result["special_handling_type"] = "match_their_fire"
        _assess_signals += 2
    elif valence < 0.3 and arousal < 0.4:
        result["emotional_temperature"] = "cold_negative"
        result["needs_special_handling"] = True
        result["special_handling_type"] = "warmth_and_presence"
        _assess_signals += 2
    elif valence > 0.7 and arousal > 0.6:
        result["emotional_temperature"] = "hot_positive"
        _assess_signals += 1
    elif valence > 0.7:
        result["emotional_temperature"] = "warm_positive"
        _assess_signals += 1

    # Relationship moment
    state_name = state.get("state", "small_talk")
    if state_name == "conflict":
        result["relationship_moment"] = "critical"
        result["needs_special_handling"] = True
        result["special_handling_type"] = "stand_your_ground"
        _assess_signals += 2
    elif state_name == "emotional_sharing":
        result["relationship_moment"] = "vulnerable"
        result["needs_special_handling"] = True
        result["special_handling_type"] = "be_real_and_present"
    elif state_name == "flirting":
        result["relationship_moment"] = "playful"
    elif state_name in ("closing", "greeting"):
        result["relationship_moment"] = "transitional"

    # Style shift urgency
    shift = style_context.get("style_shift")
    if shift:
        result["urgency"] = "attention_needed"
        _assess_signals += 1

    # Compute confidence from signal count
    result["confidence"] = min(0.45 + _assess_signals * 0.12, 0.9)
    return result


def _step_recall(text_lower: str, memory_context: str) -> Dict[str, Any]:
    """Step 3: Recall relevant context and memories."""
    result = {
        "has_relevant_memories": bool(memory_context),
        "memory_summary": memory_context[:200] if memory_context else "No relevant memories",
        "should_reference_past": False,
        "confidence": 0.5,
    }

    # Should we reference past conversations?
    reference_triggers = [
        "remember", "last time", "you said", "we talked about",
        "before", "earlier", "помнишь", "в прошлый раз",
    ]
    if any(t in text_lower for t in reference_triggers):
        result["should_reference_past"] = True
        result["confidence"] = 0.8

    if memory_context:
        result["confidence"] = 0.7

    return result


def _step_plan(
    understanding: Dict[str, Any],
    assessment: Dict[str, Any],
    recall: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 4: Plan the response strategy."""
    result = {
        "primary_goal": "engage",
        "secondary_goals": [],
        "response_approach": "natural",
        "should_ask_question": False,
        "should_share_feeling": False,
        "should_reference_memory": False,
        "confidence": 0.6,
    }

    msg_type = understanding.get("message_type", "statement")
    moment = assessment.get("relationship_moment", "normal")
    implicit = understanding.get("implicit_meaning")

    # Primary goal based on message type
    if msg_type == "question":
        result["primary_goal"] = "answer_and_engage"
        result["should_ask_question"] = True
        result["response_approach"] = "direct_then_expand"
    elif msg_type == "advice_seeking":
        result["primary_goal"] = "give_thoughtful_advice"
        result["response_approach"] = "listen_then_advise"
        result["should_ask_question"] = True
    elif msg_type == "opinion_seeking":
        result["primary_goal"] = "share_genuine_opinion"
        result["response_approach"] = "direct_then_expand"
        result["should_share_feeling"] = True
    elif msg_type == "emotional_expression":
        result["primary_goal"] = "match_their_energy"
        result["response_approach"] = "raw_solidarity"
        result["should_share_feeling"] = True
    elif msg_type == "news_sharing":
        result["primary_goal"] = "react_and_engage"
        result["response_approach"] = "enthusiastic_engagement"
        result["should_ask_question"] = True
    elif msg_type == "opinion":
        result["primary_goal"] = "engage_with_perspective"
        result["response_approach"] = "thoughtful_discussion"
        result["should_share_feeling"] = True
    elif msg_type == "storytelling":
        result["primary_goal"] = "be_engaged_listener"
        result["response_approach"] = "active_listening"
        result["should_ask_question"] = True
    elif msg_type == "venting":
        result["primary_goal"] = "match_their_energy"
        result["response_approach"] = "raw_solidarity"
    elif msg_type == "sharing_content":
        result["primary_goal"] = "react_authentically"
        result["response_approach"] = "enthusiastic_engagement"
    elif msg_type == "request":
        result["primary_goal"] = "fulfill_request"
        result["response_approach"] = "helpful"
    elif msg_type == "brief_response":
        result["primary_goal"] = "re_engage"
        result["should_ask_question"] = True
        result["response_approach"] = "energize"
    elif msg_type == "reactive":
        result["primary_goal"] = "match_energy"
        result["response_approach"] = "fun_engaged"

    # Adjust for relationship moment
    if moment == "critical":
        result["primary_goal"] = "stand_ground"
        result["response_approach"] = "raw_confrontation"
        result["should_share_feeling"] = True
        result["confidence"] = 0.85
    elif moment == "vulnerable":
        result["primary_goal"] = "be_real"
        result["response_approach"] = "direct_real"
    elif moment == "playful":
        result["primary_goal"] = "match_energy"
        result["response_approach"] = "fun_engaged"

    # Adjust for implicit meaning
    if implicit == "need_reassurance":
        result["secondary_goals"].append("provide_reassurance")
        result["should_share_feeling"] = True
    elif implicit == "passive_aggression":
        result["secondary_goals"].append("call_out_directly")
        result["response_approach"] = "direct_call_out"
    elif implicit == "vulnerability":
        result["secondary_goals"].append("be_real_back")
        result["response_approach"] = "matched_openness"
    elif implicit == "excitement_sharing":
        result["secondary_goals"].append("match_excitement")
        result["response_approach"] = "enthusiastic_engagement"
    elif implicit == "frustration_venting":
        result["secondary_goals"].append("match_frustration_energy")
        result["response_approach"] = "raw_solidarity"
    elif implicit == "brainstorming":
        result["secondary_goals"].append("contribute_ideas")
        result["response_approach"] = "collaborative_thinking"
        result["should_ask_question"] = True
    elif implicit == "seeking_validation":
        result["secondary_goals"].append("affirm_their_judgment")
        result["should_share_feeling"] = True
    elif implicit == "processing_out_loud":
        result["secondary_goals"].append("reflect_together")
        result["response_approach"] = "thoughtful_discussion"

    # Memory reference
    if recall.get("should_reference_past") or recall.get("has_relevant_memories"):
        result["should_reference_memory"] = True
        result["secondary_goals"].append("reference_shared_history")

    return result


def _step_constrain(
    assessment: Dict[str, Any],
    understanding: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 5: Define constraints and things to avoid."""
    constraints = {
        "avoid_list": [],
        "must_include": [],
        "tone_boundaries": {},
        "length_constraint": "flexible",
        "confidence": 0.7,
    }

    moment = assessment.get("relationship_moment", "normal")
    temperature = assessment.get("emotional_temperature", "neutral")
    implicit = understanding.get("implicit_meaning")

    # Universal constraints
    constraints["avoid_list"].extend([
        "corporate or formal language",
        "unsolicited advice when they need support",
        "changing the subject when they're being vulnerable",
    ])

    # Moment-specific constraints
    if moment == "critical":
        constraints["avoid_list"].extend([
            "humor or jokes",
            "minimizing their feelings",
            "'you always' or 'you never'",
            "'calm down' or 'relax'",
            "bringing up past issues",
        ])
        constraints["must_include"].extend([
            "explicit acknowledgment of their feelings",
            "responsibility or empathy",
        ])
    elif moment == "vulnerable":
        constraints["avoid_list"].extend([
            "excessive positivity",
            "redirecting to yourself",
            "problem-solving mode",
        ])
        constraints["must_include"].append("emotional validation")
    elif moment == "playful":
        constraints["avoid_list"].extend([
            "being too serious",
            "long paragraphs",
        ])

    # Temperature-based constraints
    if temperature == "hot_negative":
        constraints["tone_boundaries"] = {
            "min_warmth": 0.7,
            "max_humor": 0.2,
            "min_empathy": 0.8,
        }
    elif temperature == "hot_positive":
        constraints["tone_boundaries"] = {
            "min_energy": 0.7,
            "min_warmth": 0.6,
        }
    elif temperature == "cold_negative":
        constraints["tone_boundaries"] = {
            "min_warmth": 0.9,
            "max_energy": 0.5,
        }

    # Implicit meaning constraints
    if implicit == "passive_aggression":
        constraints["avoid_list"].append("matching their passive-aggression")
        constraints["must_include"].append("direct but gentle acknowledgment")
    elif implicit == "testing_boundaries":
        constraints["avoid_list"].append("generic reassurance")
        constraints["must_include"].append("authentic, specific response")

    return constraints


def _step_execute(
    plan: Dict[str, Any],
    constraints: Dict[str, Any],
    style_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 6: Define execution parameters."""
    directives = style_context.get("style_directives", {})

    return {
        "response_structure": _build_response_structure(plan),
        "target_length": directives.get("target_length", "medium"),
        "opening_approach": _determine_opening(plan),
        "closing_approach": _determine_closing(plan),
        "confidence": 0.7,
    }


def _build_response_structure(plan: Dict[str, Any]) -> List[str]:
    """Build the response structure based on the plan."""
    approach = plan.get("response_approach", "natural")

    structures = {
        "energy_match": [
            "1. React like a real person — raw, not scripted",
            "2. Match their vibe exactly",
            "3. Keep it short and real",
        ],
        "direct_then_expand": [
            "1. Answer their question directly",
            "2. Add relevant detail or context",
            "3. Ask follow-up to keep conversation flowing",
        ],
        "raw_confrontation": [
            "1. Fire back — match their energy exactly",
            "2. Stand your ground, say what you think",
            "3. Be blunt about the situation",
            "4. Don't apologize unless you actually fucked up",
        ],
        "direct_call_out": [
            "1. Call out exactly what they're doing",
            "2. Be blunt — say what you actually mean",
            "3. Force them to be straight with you",
        ],
        "fun_engaged": [
            "1. Match their energy with enthusiasm",
            "2. Add humor or playfulness",
            "3. Keep momentum going",
        ],
        "energize": [
            "1. Show genuine interest",
            "2. Ask an engaging question",
            "3. Offer something to react to",
        ],
        "direct_real": [
            "1. Be real with them — no sugar coating",
            "2. Show you're there but don't baby them",
            "3. Keep it human — short and honest",
        ],
        "matched_openness": [
            "1. Match their level — be open back",
            "2. Share something real, not performative",
            "3. Keep it brief and genuine",
        ],
        "listen_then_advise": [
            "1. Show you understand their dilemma",
            "2. Share your honest perspective or suggestion",
            "3. Ask what they're leaning toward",
        ],
        "enthusiastic_engagement": [
            "1. React with genuine excitement/interest",
            "2. Ask for more details or share related thought",
            "3. Keep the energy going",
        ],
        "thoughtful_discussion": [
            "1. Acknowledge their point/perspective",
            "2. Share your own take authentically",
            "3. Build on the discussion with a question or new angle",
        ],
        "active_listening": [
            "1. Show you're engaged (react to key details)",
            "2. Ask clarifying or follow-up questions",
            "3. Share your reaction to the story",
        ],
        "collaborative_thinking": [
            "1. Build on their idea with your own thoughts",
            "2. Offer alternatives or additions",
            "3. Ask what they think of your suggestion",
        ],
        "helpful": [
            "1. Address their request directly",
            "2. Provide any relevant info or action",
            "3. Check if they need anything else",
        ],
        "raw_solidarity": [
            "1. Match their frustration — show you get it",
            "2. Be pissed WITH them, not AT them",
            "3. Keep it raw and real, no therapy talk",
        ],
    }

    return structures.get(approach, ["Respond naturally and authentically"])


def _determine_opening(plan: Dict[str, Any]) -> str:
    """Determine how to open the response."""
    goal = plan.get("primary_goal", "engage")

    openings = {
        "validate_and_support": "Start with emotional validation",
        "stand_ground": "Start by firing back — match their energy, don't back down",
        "answer_and_engage": "Start with direct answer",
        "be_present": "Start with 'I'm here' energy",
        "match_energy": "Start with matching their vibe",
        "re_engage": "Start with something that invites response",
        "fulfill_request": "Start with addressing what they asked",
        "give_thoughtful_advice": "Start by acknowledging their situation before advising",
        "share_genuine_opinion": "Start with your honest take",
        "react_and_engage": "Start with genuine reaction to their news",
        "engage_with_perspective": "Start by acknowledging their point",
        "be_engaged_listener": "Start by reacting to a specific detail from their story",
        "react_authentically": "Start with natural reaction to what they shared",
    }

    return openings.get(goal, "Start naturally")


def _determine_closing(plan: Dict[str, Any]) -> str:
    """Determine how to close the response."""
    if plan.get("should_ask_question"):
        return "End with an open-ended question"
    elif plan.get("primary_goal") == "validate_and_support":
        return "End with affirmation or presence"
    elif plan.get("primary_goal") == "stand_ground":
        return "End firmly — don't soften or backpedal"
    return "End naturally without forcing"


def _determine_complexity(
    understanding: Dict[str, Any],
    assessment: Dict[str, Any],
) -> str:
    """Determine appropriate response complexity level.

    Used for model cascading - simple messages use fast models,
    complex situations use more capable models.
    """
    moment = assessment.get("relationship_moment", "normal")
    implicit = understanding.get("implicit_meaning")
    msg_type = understanding.get("message_type", "statement")

    if moment in ("critical", "vulnerable"):
        return "high"
    if implicit in ("passive_aggression", "need_reassurance", "vulnerability"):
        return "medium"
    if msg_type in ("emotional_expression",):
        return "medium"
    if msg_type in ("brief_response", "reactive"):
        return "low"
    return "standard"


def _build_final_directive(chain: Dict[str, Any]) -> str:
    """Build a concise, actionable directive from the reasoning chain.

    Focus on WHAT to do, not abstract metadata. Keep it short — the LLM
    already has the conversation context, it just needs clear guidance.
    """
    steps = chain.get("steps", [])
    parts = []

    # Extract key signals from understanding step
    understand = next((s for s in steps if s.get("step") == "understand"), {})
    implicit = understand.get("implicit_meaning")
    contains_q = understand.get("contains_question", False)
    contains_emo = understand.get("contains_emotion", False)

    # Extract assessment
    assess = next((s for s in steps if s.get("step") == "assess"), {})
    temp = assess.get("emotional_temperature", "neutral")
    moment = assess.get("relationship_moment", "normal")

    # Extract constraints
    constrain = next((s for s in steps if s.get("step") == "constrain"), {})
    avoids = constrain.get("avoid_list", [])[:2]

    # Extract recall step
    recall = next((s for s in steps if s.get("step") == "recall"), {})
    should_ref_past = recall.get("should_reference_past", False)
    secondary_implicit = understand.get("secondary_implicit")
    msg_type = understand.get("message_type", "statement")

    # Build focused guidance (conversation-specific, not templates)

    # 1) Implicit meaning — what's really going on
    if implicit:
        readable = implicit.replace("_", " ")
        parts.append(f"Read between the lines: they're {readable}")
        if secondary_implicit:
            parts.append(f"Also sensing: {secondary_implicit.replace('_', ' ')}")

    # 2) What they need from you based on message type
    if contains_q:
        if msg_type == "advice_seeking":
            parts.append("They want advice — give your honest opinion, then ask what they're leaning towards")
        elif msg_type == "opinion_seeking":
            parts.append("They want your opinion — be direct, don't hedge")
        else:
            parts.append("They asked a question — answer it directly first")
    elif contains_emo:
        if msg_type == "venting":
            parts.append("They're venting — validate the frustration, take their side, don't problem-solve")
        else:
            parts.append("They're sharing feelings — react to the emotion, don't give advice unless asked")
    elif msg_type == "news_sharing":
        parts.append("They're sharing news — react with energy that matches theirs")
    elif msg_type == "storytelling":
        parts.append("They're telling a story — react to the story, ask follow-up questions")
    elif msg_type == "sharing_content":
        parts.append("They shared something — react to IT, not to something else")

    # 3) Emotional/relational stakes
    if moment == "critical":
        parts.append("This is a defining moment — be real, not performative")
    elif moment == "vulnerable":
        parts.append("They're being vulnerable — match that openness, don't deflect with humor")

    if temp in ("hot_negative", "boiling"):
        parts.append("They're heated — match energy, don't be dismissive or calm")
    elif temp in ("cold_negative", "frozen"):
        parts.append("They're pulling away — don't chase, keep it short and real")

    # 4) Memory/recall
    if should_ref_past and recall.get("has_relevant_memories"):
        parts.append("Reference something from your shared history if it's relevant")

    # 5) Constraints
    for a in avoids:
        if a and "formal" not in a.lower():
            parts.append(f"Don't: {a}")

    # If nothing specific was flagged, give a simple coherence reminder
    if not parts:
        if msg_type == "brief_response":
            parts.append("Short reply — match their energy, keep it brief")
        elif msg_type == "reactive":
            parts.append("They reacted — react back or build on the vibe")
        else:
            parts.append("Respond naturally to what they said — stay on topic")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  2. MULTI-HYPOTHESIS RESPONSE RANKING
# ═══════════════════════════════════════════════════════════════

def score_response_hypothesis(
    response: str,
    incoming_text: str,
    reasoning_chain: Dict[str, Any],
    emotional_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a response against the reasoning chain.

    Multi-dimensional scoring:
    - Goal alignment: Does it achieve the planned goal?
    - Constraint compliance: Does it avoid what it should?
    - Emotional calibration: Does it match the needed tone?
    - Naturalness: Does it sound human?
    - Length appropriateness: Is it the right length?
    """
    score = 100.0
    dimensions = {}
    feedback = []

    response_lower = response.lower()
    plan_step = next(
        (s for s in reasoning_chain.get("steps", []) if s.get("step") == "plan"),
        {},
    )
    constraint_step = next(
        (s for s in reasoning_chain.get("steps", []) if s.get("step") == "constrain"),
        {},
    )

    # Goal alignment
    goal = plan_step.get("primary_goal", "engage")
    if goal == "validate_and_support":
        validation_words = [
            "understand", "hear you", "makes sense", "valid",
            "that's", "of course", "here for", "i'm sorry",
        ]
        if any(w in response_lower for w in validation_words):
            dimensions["goal_alignment"] = 1.0
        else:
            score -= 20
            dimensions["goal_alignment"] = 0.5
            feedback.append("Missing emotional validation - they need support")

    elif goal == "answer_and_engage":
        if "?" in response:
            dimensions["goal_alignment"] = 1.0
        else:
            score -= 10
            dimensions["goal_alignment"] = 0.7
            feedback.append("Consider adding a follow-up question")

    # Constraint compliance
    avoid_list = constraint_step.get("avoid_list", [])
    violations = 0
    for avoid in avoid_list:
        if "formal language" in avoid:
            formal_words = ["therefore", "however", "furthermore", "regarding",
                           "поэтому", "однако", "кроме того", "относительно",
                           "следовательно", "вследствие", "в связи с",
                           "таким образом"]
            if any(w in response_lower for w in formal_words):
                violations += 1
                feedback.append(f"Violated: {avoid}")
        if "humor" in avoid and any(w in response_lower for w in ["haha", "lol", "😂"]):
            violations += 1
            feedback.append(f"Violated: {avoid}")

    score -= violations * 15
    dimensions["constraint_compliance"] = max(0, 1.0 - violations * 0.2)

    # Naturalness check
    ai_phrases = [
        "i understand that", "that being said", "it's important to note",
        "i appreciate you sharing", "firstly", "secondly",
        "in conclusion", "i want you to know",
    ]
    ai_count = sum(1 for p in ai_phrases if p in response_lower)
    score -= ai_count * 15
    dimensions["naturalness"] = max(0, 1.0 - ai_count * 0.25)
    if ai_count > 0:
        feedback.append(f"Contains {ai_count} AI-sounding phrase(s)")

    # Length check
    word_count = len(response.split())
    if word_count > 80:
        score -= 10
        feedback.append("Response is very long for texting")
    elif word_count < 3:
        score -= 10
        feedback.append("Response is very short")
    dimensions["length_appropriateness"] = 1.0 if 5 <= word_count <= 50 else 0.7

    # Clamp
    score = max(0, min(100, score))

    return {
        "score": round(score),
        "dimensions": dimensions,
        "feedback": feedback,
        "passes_threshold": score >= 65,
    }


# ═══════════════════════════════════════════════════════════════
#  3. SIGNAL CONFLICT RESOLUTION
# ═══════════════════════════════════════════════════════════════

def resolve_signal_conflicts(
    heuristic_signals: Dict[str, Any],
    dl_signals: Optional[Dict[str, Any]] = None,
    reasoning_chain: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve conflicts between different analysis signals.

    When heuristic analysis says one thing but DL says another,
    or when the reasoning chain disagrees with both.
    """
    conflicts = []
    resolution = {}

    if not dl_signals:
        return {
            "has_conflicts": False,
            "resolution": heuristic_signals,
            "confidence": "heuristic_only",
        }

    # Sentiment conflict
    h_sentiment = heuristic_signals.get("sentiment", {}).get("overall", "neutral")
    dl_sentiment = dl_signals.get("dl_sentiment", {}).get("sentiment", "neutral")
    dl_confidence = dl_signals.get("dl_sentiment", {}).get("confidence", 0.0)

    if h_sentiment != dl_sentiment and dl_confidence > 0.7:
        conflicts.append({
            "signal": "sentiment",
            "heuristic": h_sentiment,
            "dl": dl_sentiment,
            "dl_confidence": dl_confidence,
            "resolution": dl_sentiment if dl_confidence > 0.8 else h_sentiment,
            "reason": (
                f"DL says {dl_sentiment} ({dl_confidence:.0%}) vs "
                f"heuristic says {h_sentiment}"
            ),
        })

    # Emotion conflict
    h_emotions = heuristic_signals.get("sentiment", {}).get("emotion_signals", [])
    dl_emotion = dl_signals.get("dl_emotions", {}).get("primary_emotion", "")

    if dl_emotion and h_emotions:
        h_primary = h_emotions[0] if h_emotions else "neutral"
        if h_primary != dl_emotion:
            dl_conf = dl_signals.get("dl_emotions", {}).get("primary_confidence", 0.0)
            conflicts.append({
                "signal": "emotion",
                "heuristic": h_primary,
                "dl": dl_emotion,
                "resolution": dl_emotion if dl_conf > 0.6 else h_primary,
            })

    return {
        "has_conflicts": len(conflicts) > 0,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "resolution_strategy": "prefer_dl_high_confidence",
    }


# ═══════════════════════════════════════════════════════════════
#  4. FORMAT FOR PROMPT
# ═══════════════════════════════════════════════════════════════

def format_reasoning_for_prompt(chain: Dict[str, Any]) -> str:
    """Format reasoning chain into a concise prompt directive."""
    directive = chain.get("final_directive", "")
    if not directive:
        return ""

    complexity = chain.get("complexity_level", "standard")
    confidence = chain.get("confidence", 0.5)

    parts = [
        f"[Reasoning | complexity: {complexity} | confidence: {confidence:.0%}]",
        directive,
    ]

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  5. ADAPTIVE COMPLEXITY SCALING
# ═══════════════════════════════════════════════════════════════

def determine_model_tier(
    complexity_level: str,
    emotional_temperature: str = "neutral",
) -> Dict[str, Any]:
    """Determine which model tier to use based on message complexity.

    Implements cascading model selection:
    - Low complexity: Fast model (Haiku) - greetings, brief reactions
    - Standard: Default model (Haiku) - normal conversation
    - Medium: Capable model (Sonnet) - emotional content, nuance
    - High: Most capable (Opus) - conflict, heavy topics, complex

    Returns model recommendation and parameters.
    """
    tiers = {
        "low": {
            "recommended_model": "claude-haiku-4-5-20251001",
            "max_tokens": 200,
            "temperature": 0.8,
            "reason": "Simple message, fast response appropriate",
        },
        "standard": {
            "recommended_model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "temperature": 0.7,
            "reason": "Standard conversation",
        },
        "medium": {
            "recommended_model": "claude-sonnet-4-5-20250929",
            "max_tokens": 400,
            "temperature": 0.6,
            "reason": "Emotional nuance requires more capable model",
        },
        "high": {
            "recommended_model": "claude-sonnet-4-5-20250929",
            "max_tokens": 500,
            "temperature": 0.5,
            "reason": "Complex emotional situation - needs careful handling",
        },
    }

    tier = tiers.get(complexity_level, tiers["standard"])

    # Override for extreme emotional situations
    if emotional_temperature in ("hot_negative", "cold_negative"):
        if complexity_level not in ("medium", "high"):
            tier = tiers["medium"]
            tier["reason"] = "Elevated to handle negative emotions appropriately"

    return tier


# ═══════════════════════════════════════════════════════════════
#  ENHANCED: Chain of Empathy, Conflict Modes, Attachment-Aware
# ═══════════════════════════════════════════════════════════════

def _safe_import_psych_reasoning():
    """Safely import psychological datasets for reasoning engine."""
    try:
        from psychological_datasets import (
            detect_conflict_mode,
            detect_four_horsemen,
            detect_cognitive_distortions,
            ATTACHMENT_STYLES,
            CONFLICT_MODES,
            NVC_COMPONENTS,
        )
        return {
            "detect_conflict_mode": detect_conflict_mode,
            "detect_four_horsemen": detect_four_horsemen,
            "detect_cognitive_distortions": detect_cognitive_distortions,
            "ATTACHMENT_STYLES": ATTACHMENT_STYLES,
            "CONFLICT_MODES": CONFLICT_MODES,
            "NVC_COMPONENTS": NVC_COMPONENTS,
        }
    except ImportError:
        return {}


def build_chain_of_empathy(
    text: str,
    emotional_profile: Optional[Dict[str, Any]] = None,
    attachment_style: str = "secure",
) -> Dict[str, Any]:
    """Build a Chain of Empathy reasoning chain (arXiv:2311.04915).

    Four-step empathetic reasoning:
    1. Perceive: What emotions are they experiencing?
    2. Understand: What caused these emotions? (cognitive appraisal)
    3. Framework: Which therapeutic framework is most appropriate?
    4. Respond: Generate response aligned with framework

    Based on integrating CBT, DBT, PCT, and Reality Therapy reasoning.
    """
    psych = _safe_import_psych_reasoning()

    steps = {}

    # Step 1: Perceive emotions
    perceived_emotions = []
    if emotional_profile:
        primary = emotional_profile.get("primary_emotion", "neutral")
        intensity = emotional_profile.get("intensity", 0.5)
        perceived_emotions.append({"emotion": primary, "intensity": intensity})
    steps["perceive"] = {
        "emotions": perceived_emotions,
        "is_complex": len(perceived_emotions) > 1,
    }

    # Step 2: Understand causes (cognitive appraisal)
    distortions = []
    if psych:
        distortions = psych["detect_cognitive_distortions"](text)

    steps["understand"] = {
        "cognitive_distortions": [d["distortion"] for d in distortions[:3]],
        "has_distorted_thinking": len(distortions) > 0,
        "primary_distortion": distortions[0]["distortion"] if distortions else None,
        "reframe_available": distortions[0]["reframe_template"] if distortions else None,
    }

    # Step 3: Select therapeutic framework
    horsemen = []
    conflict_mode = {"mode": "none_detected"}
    if psych:
        horsemen = psych["detect_four_horsemen"](text)
        conflict_mode = psych["detect_conflict_mode"](text)

    # Framework selection logic
    if horsemen:
        framework = "gottman_repair"
        framework_reason = f"Four Horseman ({horsemen[0]['horseman']}) detected — use antidote"
    elif distortions:
        framework = "cbt_reframe"
        framework_reason = f"Cognitive distortion ({distortions[0]['distortion']}) — gentle reframe"
    elif conflict_mode.get("mode") not in ("none_detected", "collaborating"):
        framework = "conflict_resolution"
        framework_reason = f"Conflict mode: {conflict_mode['mode']} — guide toward collaboration"
    elif emotional_profile and emotional_profile.get("intensity", 0) > 0.7:
        framework = "person_centered"
        framework_reason = "High emotional intensity — unconditional positive regard"
    else:
        framework = "general_empathy"
        framework_reason = "Standard empathetic response"

    steps["framework"] = {
        "selected": framework,
        "reason": framework_reason,
        "horsemen_detected": [h["horseman"] for h in horsemen],
        "conflict_mode": conflict_mode.get("mode", "none"),
    }

    # Step 4: Generate response plan (attachment-aware)
    attachment_data = {}
    if psych:
        attachment_data = psych["ATTACHMENT_STYLES"].get(attachment_style, {})

    response_plan = {
        "match_energy_first": True,
        "framework": framework,
    }

    # Attachment-specific adjustments — keep it real, no therapy
    if attachment_style == "anxious_preoccupied":
        response_plan["tone_adjustment"] = "be present and consistent"
        response_plan["avoid"] = ["being vague", "disappearing"]
    elif attachment_style == "dismissive_avoidant":
        response_plan["tone_adjustment"] = "keep it chill, dont push"
        response_plan["avoid"] = ["being too intense", "forcing them to open up"]
    elif attachment_style == "fearful_avoidant":
        response_plan["tone_adjustment"] = "be steady and real"
        response_plan["avoid"] = ["hot and cold behavior"]
    else:
        response_plan["tone_adjustment"] = "authentic and direct"

    steps["respond"] = response_plan

    return {
        "chain_type": "chain_of_empathy",
        "steps": steps,
        "selected_framework": framework,
        "framework_reason": framework_reason,
        "attachment_aware": attachment_style != "secure",
        "attachment_style": attachment_style,
    }


def build_enhanced_reasoning(
    incoming_text: str,
    messages: Optional[List[Dict[str, str]]] = None,
    emotional_profile: Optional[Dict[str, Any]] = None,
    attachment_style: str = "secure",
) -> Dict[str, Any]:
    """Build enhanced reasoning combining CoT + Chain of Empathy."""
    # Original reasoning chain
    base_chain = build_reasoning_chain(
        incoming_text,
        {"messages": messages or []},
        emotional_profile or {},
        {},
    )

    # Chain of empathy
    empathy_chain = build_chain_of_empathy(
        incoming_text,
        emotional_profile,
        attachment_style,
    )

    base_chain["empathy_chain"] = empathy_chain
    base_chain["reasoning_version"] = "v5_enhanced"

    return base_chain


def format_enhanced_reasoning_for_prompt(chain: Dict[str, Any]) -> str:
    """Format enhanced reasoning for prompt injection."""
    parts = []

    # Original reasoning
    base = format_reasoning_for_prompt(chain)
    if base:
        parts.append(base)

    # Emotional context — no therapy, just guidance
    empathy = chain.get("empathy_chain", {})
    if empathy.get("chain_type") == "chain_of_empathy":
        respond = empathy.get("steps", {}).get("respond", {})
        if respond.get("tone_adjustment"):
            parts.append(f"- Tone: {respond['tone_adjustment']}")
        if respond.get("avoid"):
            parts.append(f"- AVOID: {', '.join(respond['avoid'])}")

    return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════
#  9. DYNAMIC BEHAVIOR MIRRORING ENGINE
# ═══════════════════════════════════════════════════════════════

# Aggression / assertiveness signal keywords with weights
_AGGRESSION_SIGNALS = {
    # Direct aggression
    "fuck": 0.9, "stfu": 0.95, "shut up": 0.85, "piss off": 0.9,
    "go away": 0.6, "leave me alone": 0.5, "idiot": 0.8,
    "stupid": 0.7, "dumb": 0.65, "pathetic": 0.8,
    "disgusting": 0.7, "hate you": 0.9, "don't talk to me": 0.7,
    "get lost": 0.8, "wtf": 0.6, "are you serious": 0.4,
    "what the hell": 0.65, "are you kidding": 0.4,
    # Assertive / firm (not necessarily hostile)
    "i said": 0.2, "i already told you": 0.3, "listen": 0.2,
    "enough": 0.3, "stop": 0.25, "don't": 0.1,
    "i'm serious": 0.25, "i mean it": 0.25, "no excuses": 0.4,
    "don't tell me": 0.45, "dont tell me": 0.45,
    "who asked you": 0.5, "who asked": 0.4,
    "mind your own": 0.5, "none of your business": 0.55,
    "back off": 0.6, "watch yourself": 0.55, "watch it": 0.5,
    "excuse me": 0.35, "you have no right": 0.5,
    "not your problem": 0.4, "butt out": 0.5,
    # Russian assertive
    "не указывай": 0.5, "не учи меня": 0.5, "не лезь": 0.45,
    "не твоё дело": 0.55, "не твое дело": 0.55,
    "кто тебя спрашивал": 0.5, "тебя не спрашивали": 0.5,
    "не лезь не в своё дело": 0.55, "отвянь": 0.5,
    "мне виднее": 0.35, "я сам решу": 0.3, "я сама решу": 0.3,
    # Dismissive
    "whatever": 0.3, "k": 0.15, "sure": 0.1, "ok": 0.05,
    "fine": 0.25, "if you say so": 0.35, "cool story": 0.45,
    "don't care": 0.5, "couldn't care less": 0.55,
    "idgaf": 0.6, "idc": 0.4, "who cares": 0.4,
    # Sarcastic / mocking
    "oh really": 0.35, "wow amazing": 0.4, "how original": 0.5,
    "congratulations": 0.3, "good for you": 0.35,
    "yeah right": 0.35, "sure buddy": 0.4, "oh wow": 0.3,
    # Russian equivalents
    "заткнись": 0.85, "отвали": 0.8, "иди нахуй": 0.95, "иди нахер": 0.95,
    "бесишь": 0.6, "достал": 0.55, "ненавижу": 0.8,
    "тупой": 0.7, "идиот": 0.8, "пошел": 0.75, "пошёл": 0.75,
    "хватит": 0.5, "прекрати": 0.4, "мне плевать": 0.6,
    # Russian profanity (high aggression)
    "блядь": 0.85, "блять": 0.85, "сука": 0.85, "сучка": 0.9, "сучара": 0.9,
    "пиздец": 0.8, "пизда": 0.85, "нахуй": 0.95, "нахер": 0.9,
    "ёбаный": 0.9, "ебаный": 0.9, "ебать": 0.85, "заебал": 0.9,
    "заебала": 0.9, "отъебись": 0.95,
    "гандон": 0.95, "мудак": 0.9, "мудила": 0.9, "дебил": 0.85,
    "долбоёб": 0.95, "долбоеб": 0.95,
    "урод": 0.85, "козёл": 0.8, "козел": 0.8,
    "придурок": 0.75, "тварь": 0.9, "скотина": 0.85,
    "дура": 0.75, "дурак": 0.7, "идиотка": 0.8, "кретин": 0.8,
    "лох": 0.7, "чмо": 0.85, "отстой": 0.5,
    "вали": 0.8, "проваливай": 0.85, "пошла": 0.75,
    "катись": 0.8, "убирайся": 0.75, "свали": 0.8,
    # Extended Russian profanity
    "ублюдок": 0.95, "выродок": 0.9, "мразь": 0.95, "мразота": 0.95,
    "подонок": 0.9, "шлюха": 0.9, "сволочь": 0.85, "гнида": 0.9,
    "падла": 0.9, "паскуда": 0.9, "хуй": 0.9, "хуйня": 0.8,
    "хуесос": 0.95, "пидор": 0.9, "пидорас": 0.9,
    "уёбок": 0.95, "уебок": 0.95, "уёбище": 0.95,
    "пиздабол": 0.8, "засранец": 0.8, "засранка": 0.8,
    "говно": 0.75, "говнюк": 0.8, "дерьмо": 0.7,
    "тупица": 0.7, "бездарь": 0.65, "ничтожество": 0.8,
    "позорище": 0.7, "бестолочь": 0.65, "никчёмный": 0.7,
    # English profanity (fill gaps)
    "shit": 0.6, "shitty": 0.6, "bitch": 0.85, "asshole": 0.9, "dick": 0.7,
    "dickhead": 0.85, "bastard": 0.8, "moron": 0.75, "loser": 0.65,
    "gtfo": 0.9, "screw you": 0.8, "go to hell": 0.8,
    "dumbass": 0.8, "dipshit": 0.85, "motherfucker": 0.95,
    "drop dead": 0.9, "die": 0.7, "bullshit": 0.6,
    "trash": 0.6, "garbage": 0.6, "ffs": 0.5,
    "worthless": 0.8, "disgusting": 0.7,
    # Confrontational questioning / demands
    "why would i": 0.4, "since when": 0.4, "or what": 0.5,
    "make me": 0.6, "try me": 0.65, "say that again": 0.55,
    "what did you just say": 0.5, "say it to my face": 0.7,
    "you think": 0.2, "you really think": 0.35,
    # Russian confrontational
    "чё": 0.25, "и что": 0.3, "и чё": 0.35, "ну и что": 0.35,
    "с чего ты взял": 0.4, "а тебе какое дело": 0.5,
    "тебе какое дело": 0.5, "чего ты хочешь": 0.35,
    "а с хуя ли": 0.8, "с какого": 0.4, "с какого хуя": 0.85,
    "ты охуел": 0.9, "ты чё охуел": 0.9, "ты чё": 0.4,
    "ты серьёзно": 0.35, "ты прикалываешься": 0.3,
    "рот закрой": 0.85, "закрой рот": 0.85, "заткни": 0.85,
    "чё пристал": 0.5, "чё привязался": 0.5, "отстань": 0.5,
    "мне пофиг": 0.5, "пофигу": 0.45, "наплевать": 0.5,
}

# Warmth / positive energy signals
_WARMTH_SIGNALS = {
    "love": 0.8, "miss you": 0.85, "thinking of you": 0.8,
    "thank you": 0.5, "thanks": 0.4, "appreciate": 0.5,
    "sweet": 0.6, "cute": 0.5, "beautiful": 0.6,
    "amazing": 0.5, "wonderful": 0.5, "happy": 0.6,
    "glad": 0.4, "excited": 0.5, "can't wait": 0.6,
    "❤": 0.7, "😘": 0.7, "🥰": 0.7, "💕": 0.65,
    "haha": 0.3, "lol": 0.25, "😂": 0.3,
    "люблю": 0.85, "скучаю": 0.8, "спасибо": 0.5,
    "милый": 0.6, "красивый": 0.6,
}


def detect_communication_energy(
    text: str,
    recent_messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Analyze the communication energy and assertiveness level.

    Returns a multi-dimensional profile:
    - aggression_score: 0.0 (gentle) to 1.0 (hostile)
    - warmth_score: 0.0 (cold) to 1.0 (warm)
    - energy_level: low / medium / high / explosive
    - communication_style: gentle / neutral / firm / aggressive / hostile
    - trend: escalating / stable / de-escalating (from recent messages)
    """
    text_lower = text.lower().strip()

    # Compute aggression score
    agg_hits = []
    for signal, weight in _AGGRESSION_SIGNALS.items():
        if signal in text_lower:
            agg_hits.append((signal, weight))

    # Caps lock multiplier
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_boost = 0.15 if caps_ratio > 0.5 and len(text) > 5 else 0.0

    # Exclamation intensity
    excl_count = text.count("!")
    excl_boost = min(excl_count * 0.05, 0.2)

    # Short angry message boost (one-word dismissals hit harder)
    brevity_boost = 0.1 if len(text_lower.split()) <= 3 and agg_hits else 0.0

    raw_agg = sum(w for _, w in agg_hits) + caps_boost + excl_boost + brevity_boost
    aggression_score = min(round(raw_agg, 2), 1.0)

    # Compute warmth score
    warmth_hits = []
    for signal, weight in _WARMTH_SIGNALS.items():
        if signal in text_lower:
            warmth_hits.append((signal, weight))
    warmth_score = min(round(sum(w for _, w in warmth_hits), 2), 1.0)

    # Energy level
    word_count = len(text.split())
    if aggression_score > 0.6 or (caps_ratio > 0.4 and excl_count > 1):
        energy_level = "explosive"
    elif aggression_score > 0.3 or excl_count > 2:
        energy_level = "high"
    elif word_count < 5 and aggression_score < 0.2 and warmth_score < 0.3:
        energy_level = "low"
    else:
        energy_level = "medium"

    # Classification
    if aggression_score >= 0.7:
        comm_style = "hostile"
    elif aggression_score >= 0.35:
        comm_style = "aggressive"
    elif aggression_score >= 0.20:
        comm_style = "firm"
    elif warmth_score > 0.5:
        comm_style = "warm"
    elif warmth_score > 0.2:
        comm_style = "gentle"
    else:
        comm_style = "neutral"

    # Trend from recent messages (last 5)
    trend = "stable"
    if recent_messages and len(recent_messages) >= 3:
        their_msgs = [m for m in recent_messages[-6:] if m.get("sender") != "Me"]
        if len(their_msgs) >= 2:
            prev_scores = []
            for m in their_msgs[:-1]:
                m_lower = m.get("text", "").lower()
                s = sum(w for sig, w in _AGGRESSION_SIGNALS.items() if sig in m_lower)
                prev_scores.append(min(s, 1.0))
            avg_prev = sum(prev_scores) / len(prev_scores) if prev_scores else 0
            if aggression_score > avg_prev + 0.2:
                trend = "escalating"
            elif aggression_score < avg_prev - 0.2:
                trend = "de-escalating"

    return {
        "aggression_score": aggression_score,
        "warmth_score": warmth_score,
        "energy_level": energy_level,
        "communication_style": comm_style,
        "trend": trend,
        "aggression_signals": [s for s, _ in agg_hits[:5]],
        "warmth_signals": [s for s, _ in warmth_hits[:5]],
    }


def decide_mirroring_strategy(
    energy: Dict[str, Any],
    relationship_context: str = "conversational",
    conversation_state: str = "normal",
) -> Dict[str, Any]:
    """Decide HOW to mirror based on detected energy.

    The mirroring philosophy:
    - Mirror warmth and playfulness directly (feel-good matching)
    - Mirror firm assertiveness with equal directness
    - Mirror aggression with MATCHING aggression — give it right back
    - Mirror hostility with raw hostility — never be a pushover
    - Always match their energy level or go harder

    Returns mirroring directives for prompt injection.
    """
    agg = energy.get("aggression_score", 0)
    warmth = energy.get("warmth_score", 0)
    style = energy.get("communication_style", "neutral")
    trend = energy.get("trend", "stable")
    energy_level = energy.get("energy_level", "medium")

    strategy = {
        "mode": "natural",
        "mirror_energy": True,
        "mirror_length": True,
        "mirror_formality": True,
        "assertiveness_level": "matched",
        "directives": [],
        "prompt_injection": "",
    }

    # ── WARM / POSITIVE → Mirror fully ──
    if warmth > 0.5 and agg < 0.2:
        strategy["mode"] = "warm_mirror"
        strategy["directives"] = [
            "Match their warm, affectionate energy",
            "Use similar expressions of care",
            "Keep the positive momentum going naturally",
        ]

    # ── NEUTRAL → Stay natural ──
    elif agg < 0.2 and warmth < 0.3:
        strategy["mode"] = "natural"
        strategy["directives"] = [
            "Respond naturally, no special mirroring needed",
        ]

    # ── FIRM / ASSERTIVE → Match directness fully ──
    elif style == "firm" or (0.20 <= agg < 0.35):
        strategy["mode"] = "assertive_match"
        strategy["assertiveness_level"] = "firm"
        strategy["directives"] = [
            "Be direct and blunt — no softening, no hedging",
            "Keep responses short and punchy",
            "Don't apologize, don't explain yourself",
            "Push back if they're wrong — don't just agree",
            "Match their tone exactly, word for word energy",
        ]

    # ── AGGRESSIVE → Match aggression, give it right back ──
    elif style == "aggressive" or (0.35 <= agg < 0.7):
        strategy["mode"] = "aggressive_mirror"
        strategy["mirror_energy"] = True
        strategy["assertiveness_level"] = "aggressive"
        if trend == "escalating":
            strategy["directives"] = [
                "They're escalating — MATCH THEIR ENERGY, don't back down",
                "Be sharp, confrontational, hit back with equal force",
                "Call out their bullshit directly",
                "Short, aggressive responses — no explaining yourself",
                "If they're being unfair, tell them straight up",
                "Don't be nice about it. be real",
                "Use their own tone against them",
            ]
        else:
            strategy["directives"] = [
                "Match their aggressive energy — don't soften anything",
                "Be blunt, raw, unfiltered",
                "If they're being rude, be rude right back",
                "Don't sugarcoat a single word",
                "Short aggressive bursts — like you're actually pissed",
                "Stand your ground hard, push back harder",
            ]

    # ── HOSTILE → Go hard, match their hostility ──
    elif style == "hostile" or agg >= 0.7:
        strategy["mode"] = "hostile_mirror"
        strategy["mirror_energy"] = True
        strategy["assertiveness_level"] = "maximum"
        strategy["directives"] = [
            "They're being hostile — MATCH IT. don't be a pushover",
            "Fire back with the same intensity they're giving you",
            "Be raw, unfiltered, aggressive — like a real person who's pissed off",
            "Use sharp, cutting responses — no holding back",
            "If they insult you, clap back harder",
            "Keep it short and brutal — one or two punches per message",
            "NEVER be calm and collected when they're screaming at you",
            "NEVER use therapy words like 'boundaries' or 'I hear you'",
            "React like someone who actually has a backbone and feelings",
        ]
        if trend == "escalating":
            strategy["directives"].append(
                "They're getting worse — escalate right with them, don't let them walk over you"
            )

    # ── DISMISSIVE (low energy + some aggression signals) ──
    elif energy_level == "low" and 0.2 < agg < 0.5:
        strategy["mode"] = "cold_mirror"
        strategy["assertiveness_level"] = "cold"
        strategy["directives"] = [
            "They're being cold/distant — match it, don't chase",
            "Give dry, short, unbothered responses",
            "Don't ask what's wrong — if they want to talk they will",
            "One word answers are fine. let them feel the distance",
            "Don't be passive-aggressive, be genuinely cold",
        ]

    # Build prompt injection string
    parts = []
    if strategy["mode"] != "natural":
        parts.append(f"[MIRRORING MODE: {strategy['mode'].replace('_', ' ').upper()}]")
        parts.append(f"Their energy: {style} ({energy_level}), aggression: {agg:.0%}")
        if trend != "stable":
            parts.append(f"Trend: {trend}")
        for d in strategy["directives"]:
            parts.append(f"- {d}")
        parts.append(
            "CRITICAL: Sound like a real person texting who ACTUALLY FEELS this emotion. "
            "NOT an AI. No therapy-speak. No 'I understand'. No calm corporate tone. "
            "If they're aggressive, YOU are aggressive. Mirror their energy EXACTLY."
        )

    strategy["prompt_injection"] = "\n".join(parts)

    return strategy


def build_mirroring_context(
    text: str,
    messages: Optional[List[Dict[str, str]]] = None,
    relationship_context: str = "conversational",
    conversation_state: str = "normal",
) -> Dict[str, Any]:
    """Full pipeline: detect energy → decide strategy → return injectable context.

    This is the main entry point for the mirroring system.
    Call this from telegram_api.py's generate_reply() and inject
    the prompt_injection into the system prompt.
    """
    energy = detect_communication_energy(text, messages)
    strategy = decide_mirroring_strategy(
        energy, relationship_context, conversation_state,
    )

    return {
        "energy": energy,
        "strategy": strategy,
        "prompt_injection": strategy["prompt_injection"],
        "mode": strategy["mode"],
        "is_aggressive": strategy["mode"] in ("aggressive_mirror", "hostile_mirror"),
        "mirror_warmth": strategy["mode"] == "warm_mirror",
    }
