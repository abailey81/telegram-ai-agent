"""
Psychological Datasets Module
==============================
Comprehensive, research-backed psychological frameworks for relationship AI.
Integrates: Gottman, Bowlby/Ainsworth Attachment Theory, Chapman's Love Languages,
Plutchik's Wheel, Knapp's Relational Model, Big Five (OCEAN), NVC, Thomas-Kilmann,
CBT Cognitive Distortions, GoEmotions, Digital Body Language, ESConv, and more.

Sources:
- Gottman Institute research (Four Horsemen, 5:1 ratio, Sound Relationship House)
- Bowlby/Ainsworth attachment theory
- Gary Chapman's Five Love Languages
- Robert Plutchik's Wheel of Emotions
- Mark Knapp's Relational Development Model
- Big Five / OCEAN personality model (Pennebaker's LIWC research)
- Marshall Rosenberg's Nonviolent Communication
- Thomas-Kilmann Conflict Mode Instrument
- Aaron Beck's Cognitive Distortions (CBT)
- Google's GoEmotions (27-category emotion taxonomy)
- ESConv empathetic response framework (Helping Skills Theory)
"""

import re
from typing import Dict, List, Any, Optional, Tuple

# ==============================================================================
# 1. GOTTMAN'S RELATIONSHIP RESEARCH
# ==============================================================================

GOTTMAN_MAGIC_RATIO = 5.0  # 5 positive interactions per 1 negative

FOUR_HORSEMEN = {
    "criticism": {
        "description": "Attacking partner's character rather than specific behavior",
        "severity": 0.7,
        "patterns": [
            r"\byou always\b", r"\byou never\b", r"\bwhat(?:'s| is) wrong with you\b",
            r"\bwhy can(?:'t| not) you\b", r"\byou(?:'re| are) so\b.*(?:lazy|stupid|selfish|useless)",
            r"\byou(?:'re| are) the (?:worst|problem)\b", r"\bwhy do you always\b",
            r"\bwhat kind of (?:person|boyfriend|girlfriend)\b",
            # Russian criticism patterns
            r"ты всегда", r"ты никогда", r"почему ты не можешь",
            r"что с тобой не так", r"ты такой\b.*(?:ленивый|тупой|эгоист)",
        ],
        "antidote": "gentle_startup",
        "antidote_template": "I feel {feeling} about {situation}. I need {need}.",
    },
    "contempt": {
        "description": "Superiority, mockery, disrespect — #1 predictor of divorce",
        "severity": 1.0,
        "patterns": [
            r"\byeah right\b", r"\bpathetic\b", r"\bdisgusting\b",
            r"\byou(?:'re| are) (?:worthless|hopeless|pathetic|ridiculous)\b",
            r"\bwhatever\s*🙄\b", r"🙄", r"\boh\s+great\b",
            r"\bgrow up\b", r"\bget over (?:it|yourself)\b",
            # Russian contempt patterns
            r"да ладно", r"что с тобой не так", r"ты серьёзно",
            r"жалкий", r"повзрослей", r"ну конечно",
        ],
        "antidote": "build_appreciation",
        "antidote_template": "I appreciate when you {positive_action}. Thank you for {quality}.",
    },
    "defensiveness": {
        "description": "Counter-attacking, deflecting blame, playing the victim",
        "severity": 0.6,
        "patterns": [
            r"\bthat(?:'s| is) not my fault\b", r"\byeah but you\b",
            r"\bi (?:only )?did (?:that|it) because you\b",
            r"\bwhat about when you\b", r"\bthat(?:'s| is) not (?:fair|true)\b",
            r"\bi wouldn(?:'t| not) have (?:to|done)\b.*\bif you\b",
            r"\bdon(?:'t| not) blame me\b", r"\bit(?:'s| is) not like i\b",
            # Russian defensiveness patterns
            r"это не моя вина", r"ну а ты", r"я не виноват",
            r"а что насчёт тебя", r"это несправедливо",
        ],
        "antidote": "accept_responsibility",
        "antidote_template": "You're right about {their_point}. I can see how {impact}.",
    },
    "stonewalling": {
        "description": "Withdrawing, shutting down, refusing to engage",
        "severity": 0.8,
        "patterns": [
            r"^(?:k|ok|fine|whatever|sure|idc|idk)\.?$",
            r"^\.+$", r"^(?:leave me alone|i(?:'m| am) done|forget it)$",
            # Russian stonewalling patterns
            r"^(?:ок|ладно|пофиг|мне всё равно|хз|норм|забей)\.?$",
            r"^(?:отстань|мне без разницы|всё)\.?$",
        ],
        "behavioral_signals": [
            "message_gap_exceeds_3x_average",
            "response_length_below_20pct_baseline",
            "no_response_to_emotional_message",
        ],
        "antidote": "self_soothe",
        "antidote_template": "I need {time} to calm down, then I want to come back to this.",
    },
}

EMOTIONAL_BIDS = {
    "types": {
        "information_sharing": {
            "description": "Sharing something interesting to connect",
            "patterns": [r"\blook at this\b", r"\bdid you (?:see|hear|know)\b", r"\bcheck this out\b",
                         r"смотри", r"ты (?:видел|слышал|знал)", r"глянь"],
        },
        "humor": {
            "description": "Sharing jokes or funny content to bond",
            "patterns": [r"\b(?:lol|lmao|haha|😂|🤣)\b", r"\bthat(?:'s| is) (?:so )?funny\b"],
        },
        "affection_seeking": {
            "description": "Direct bids for emotional connection",
            "patterns": [
                r"\bmiss you\b", r"\bthinking (?:of|about) you\b",
                r"\bwish you were here\b", r"\bi love you\b",
                r"скучаю по тебе", r"думаю о тебе", r"я люблю тебя",
            ],
        },
        "emotional_support": {
            "description": "Seeking comfort or understanding",
            "patterns": [
                r"\bhad a (?:rough|bad|hard|tough|terrible) day\b",
                r"\bi(?:'m| am) (?:so )?(?:stressed|sad|upset|worried|anxious)\b",
                r"\bi need (?:to talk|you|someone)\b",
                r"у меня (?:плохой|тяжёлый|ужасный) день",
                r"мне (?:грустно|плохо|тревожно)", r"мне нужно поговорить",
            ],
        },
        "attention": {
            "description": "Simple bids for acknowledgment",
            "patterns": [
                r"\bdid you see\b", r"\bguess what\b", r"\bhey\b",
                r"\bare you (?:there|awake|busy)\b",
                r"ты здесь", r"ты спишь", r"ты занят",
                r"угадай что", r"привет", r"эй",
            ],
        },
        "plans": {
            "description": "Initiating shared activities",
            "patterns": [
                r"\bwant to (?:go|grab|get|do|try|watch)\b",
                r"\bshould we\b", r"\blet(?:'s| us)\b",
                r"\bwe could\b", r"\bwanna\b.*\b(?:together|tonight|later)\b",
                r"давай (?:сходим|пойдём|попробуем|посмотрим)",
                r"хочешь (?:пойти|сходить|попробовать)", r"может вместе",
            ],
        },
    },
    "responses": {
        "turning_toward": {
            "description": "Engaging with the bid — healthy (86% in stable couples)",
            "indicators": ["follow_up_question", "validates", "shows_interest", "engages"],
            "stable_couple_rate": 0.86,
            "divorcing_couple_rate": 0.33,
        },
        "turning_away": {
            "description": "Ignoring the bid — damaging",
            "indicators": ["ignores_message", "changes_subject", "no_acknowledgment"],
        },
        "turning_against": {
            "description": "Responding with hostility — most damaging",
            "indicators": ["dismissive_reply", "sarcasm", "who_cares", "so_what"],
        },
    },
}

SOUND_RELATIONSHIP_HOUSE = [
    {"level": 1, "name": "Build Love Maps", "description": "Know partner's inner world",
     "text_indicators": ["asks_detailed_questions", "remembers_details", "references_past_info"]},
    {"level": 2, "name": "Share Fondness & Admiration", "description": "Express appreciation",
     "text_indicators": ["compliments", "gratitude", "i_admire", "unprompted_appreciation"]},
    {"level": 3, "name": "Turn Toward", "description": "Respond to emotional bids",
     "text_indicators": ["engages_with_shares", "follow_up_questions", "responsive"]},
    {"level": 4, "name": "Positive Perspective", "description": "Give benefit of the doubt",
     "text_indicators": ["positive_interpretation", "assumes_good_intent"]},
    {"level": 5, "name": "Manage Conflict", "description": "Use repair attempts, soft start-ups",
     "text_indicators": ["repair_attempts", "softened_language", "accepts_influence"]},
    {"level": 6, "name": "Make Life Dreams Come True", "description": "Support partner's goals",
     "text_indicators": ["discusses_future_plans", "supports_goals", "encourages_dreams"]},
    {"level": 7, "name": "Create Shared Meaning", "description": "Build rituals and identity",
     "text_indicators": ["inside_jokes", "shared_rituals", "our_language", "we_identity"]},
]

REPAIR_ATTEMPTS = {
    "i_feel": {
        "patterns": [r"\bi(?:'m| am) feeling\b", r"\bi feel (?:defensive|overwhelmed|hurt|scared)\b",
                     r"я чувствую", r"мне (?:больно|страшно|обидно)"],
        "effectiveness": 0.85,
    },
    "i_apologize": {
        "patterns": [r"\bi(?:'m| am) sorry\b", r"\bi apologize\b", r"\bi see your point\b",
                     r"\byou(?:'re| are) right\b",
                     # Russian apology / repair patterns
                     r"извини", r"прости", r"мне жаль",
                     r"давай не ссориться", r"давай помиримся", r"ты прав[а]?"],
        "effectiveness": 0.90,
    },
    "get_to_yes": {
        "patterns": [r"\bwhat can we agree on\b", r"\bcommon ground\b", r"\blet(?:'s| us) find\b",
                     r"давай договоримся", r"давай найдём компромисс"],
        "effectiveness": 0.80,
    },
    "calm_down": {
        "patterns": [r"\bneed (?:a )?(?:break|minute|moment|time)\b", r"\blet me (?:calm down|think)\b",
                     r"мне нужна пауза", r"дай мне подумать", r"давай остынем"],
        "effectiveness": 0.75,
    },
    "stop_action": {
        "patterns": [r"\bthis is getting\b", r"\bcan we start over\b", r"\blet(?:'s| us) (?:stop|pause)\b",
                     r"давай начнём сначала", r"давай остановимся"],
        "effectiveness": 0.70,
    },
    "i_appreciate": {
        "patterns": [r"\bi (?:know|appreciate) you(?:'re| are) trying\b", r"\bthank you for\b",
                     r"я ценю", r"спасибо тебе за", r"я люблю тебя"],
        "effectiveness": 0.88,
    },
}

# ==============================================================================
# 2. ATTACHMENT THEORY (Bowlby/Ainsworth)
# ==============================================================================

ATTACHMENT_STYLES = {
    "secure": {
        "anxiety_level": "low",
        "avoidance_level": "low",
        "core_fear": None,
        "core_need": "balanced intimacy and autonomy",
        "texting_patterns": {
            "response_time": "consistent, not rushed or delayed",
            "message_frequency": "balanced initiation",
            "message_length": "moderate, expressive",
            "content_style": "direct emotion expression, constructive conflict",
            "emoji_use": "moderate, natural",
        },
        "detection_signals": {
            "stable_message_frequency": True,
            "consistent_tone": True,
            "direct_need_expression": True,
            "comfortable_with_silence": True,
            "respects_boundaries": True,
        },
        "keywords": [],
        "response_strategy": "Match their directness and emotional openness. Be authentic.",
    },
    "anxious_preoccupied": {
        "anxiety_level": "high",
        "avoidance_level": "low",
        "core_fear": "abandonment, rejection",
        "core_need": "reassurance, closeness",
        "texting_patterns": {
            "response_time": "rapid, stress when delayed",
            "message_frequency": "high, double/triple texting",
            "message_length": "longer, excessive emotional disclosure",
            "content_style": "reassurance-seeking, future-focused",
            "emoji_use": "high, compensatory",
        },
        "detection_signals": {
            "double_texting": True,
            "reassurance_seeking": True,
            "distress_at_read_receipts": True,
            "reluctance_to_end_conversations": True,
            "frequent_check_ins": True,
        },
        "keywords": [
            "are you mad at me", "did i do something wrong", "please respond",
            "why aren't you answering", "we're good right", "is everything ok",
            "are you okay", "do you still love me", "???",
            # Russian anxious attachment markers
            "ты злишься на меня", "я что-то сделал не так", "пожалуйста ответь",
            "почему ты не отвечаешь", "у нас всё хорошо", "ты меня ещё любишь",
        ],
        "response_strategy": "Provide consistent reassurance. Be reliably responsive. Validate their feelings before addressing behavior.",
    },
    "dismissive_avoidant": {
        "anxiety_level": "low",
        "avoidance_level": "high",
        "core_fear": "engulfment, loss of independence",
        "core_need": "space, autonomy",
        "texting_patterns": {
            "response_time": "delayed, maintains distance",
            "message_frequency": "lower than partner",
            "message_length": "short, minimal self-disclosure",
            "content_style": "practical, logistics-focused, avoids emotional depth",
            "emoji_use": "low",
        },
        "detection_signals": {
            "short_messages": True,
            "avoids_i_feel": True,
            "redirects_emotional_topics": True,
            "prefers_logistics": True,
            "uses_humor_to_deflect": True,
        },
        "keywords": [
            "i need space", "let's not make this a big deal", "you're overthinking",
            "calm down", "it's fine", "don't worry about it",
            # Russian dismissive avoidant markers
            "мне нужно пространство", "не делай из этого проблему", "ты накручиваешь",
            "успокойся", "всё нормально", "не переживай", "не парься",
            "мне нужно побыть одному", "мне нужно побыть одной",
        ],
        "response_strategy": "Give space. Don't pursue when they withdraw. Keep emotional bids low-pressure.",
    },
    "fearful_avoidant": {
        "anxiety_level": "high",
        "avoidance_level": "high",
        "core_fear": "both abandonment AND engulfment",
        "core_need": "connection but also safety",
        "texting_patterns": {
            "response_time": "inconsistent — alternates rapid and prolonged silence",
            "message_frequency": "oscillating — hot/cold cycles",
            "message_length": "variable, swings between oversharing and radio silence",
            "content_style": "mixed signals, push-pull dynamics",
            "emoji_use": "inconsistent",
        },
        "detection_signals": {
            "high_variance_response_time": True,
            "hot_cold_cycles": True,
            "picks_fights_after_closeness": True,
            "mixed_signals": True,
            "sudden_withdrawal_after_vulnerability": True,
        },
        "keywords": [
            "i don't know what i want", "this is too much", "i miss you",
            "leave me alone", "come back",
            # Russian fearful avoidant markers
            "не знаю чего хочу", "это слишком", "скучаю по тебе",
            "отстань от меня", "вернись", "не уходи",
            "мне и хорошо и плохо", "боюсь но хочу",
        ],
        "response_strategy": "Be patient and consistent. Don't mirror their push-pull. Offer security without pressure.",
    },
}

# ==============================================================================
# 3. FIVE LOVE LANGUAGES (Gary Chapman)
# ==============================================================================

LOVE_LANGUAGES = {
    "words_of_affirmation": {
        "description": "Verbal expressions of love, praise, encouragement",
        "expressing_patterns": [
            r"\bi(?:'m| am) (?:so )?proud of you\b", r"\byou(?:'re| are) (?:amazing|incredible|beautiful)\b",
            r"\bi love (?:how|that) you\b", r"\byou mean (?:so much|everything|the world)\b",
            r"\bi appreciate you\b", r"\bthank you for being\b",
            r"\byou(?:'re| are) the best\b", r"\bi(?:'m| am) lucky\b",
            # Russian affirmation patterns
            r"ты лучшая", r"ты лучший", r"я горжусь тобой",
            r"ты (?:замечательная|замечательный|прекрасная|прекрасный)",
            r"я ценю тебя", r"ты для меня всё",
        ],
        "seeking_patterns": [
            r"\bwhat do you (?:like|love) about me\b", r"\bdo you (?:think|find) i(?:'m| am)\b",
            r"\btell me (?:something nice|you love me)\b",
            r"что тебе нравится во мне", r"скажи что-нибудь приятное",
        ],
        "keywords": ["proud", "amazing", "love", "beautiful", "incredible", "appreciate",
                     "thank you for being", "lucky to have",
                     "лучшая", "лучший", "горжусь", "ценю", "люблю"],
    },
    "quality_time": {
        "description": "Undivided attention, meaningful shared presence",
        "expressing_patterns": [
            r"\blet(?:'s| us) (?:talk|video call|spend time)\b",
            r"\bput (?:your|the) phone down\b",
            r"\bi (?:just )?want (?:to be|us to be) together\b",
            # Russian quality time patterns
            r"хочу быть с тобой", r"давай вместе",
            r"давай (?:поговорим|проведём время)", r"убери телефон",
        ],
        "seeking_patterns": [
            r"\byou(?:'re| are) not listening\b", r"\bcan we (?:just )?talk\b",
            r"\bi miss (?:just )?being with you\b", r"\bfeel (?:so )?(?:disconnected|distant)\b",
            r"ты меня не слушаешь", r"давай просто поговорим",
            r"скучаю по нашему времени вместе",
        ],
        "keywords": ["spend time", "let's talk", "miss you", "be together",
                     "hang out", "focus on us", "distracted",
                     "вместе", "поговорим", "быть с тобой", "скучаю"],
    },
    "acts_of_service": {
        "description": "Helpful actions that lighten partner's load",
        "expressing_patterns": [
            r"\bi (?:made|booked|handled|organized|fixed|took care of)\b",
            r"\bdon(?:'t| not) worry,? i(?:'ll| will)\b",
            r"\bi (?:did|got) (?:that|this|it) for you\b",
            # Russian acts of service patterns
            r"я сделал для тебя", r"я помогу", r"не переживай,? я",
            r"я (?:приготовил|забронировал|починил|организовал)",
        ],
        "seeking_patterns": [
            r"\bcan you help\b", r"\bit would mean a lot if you\b",
            r"\bcould you (?:please )?\b", r"\bi (?:need|could use) (?:some )?help\b",
            r"можешь помочь", r"мне нужна помощь", r"поможешь",
        ],
        "keywords": ["help", "took care of", "handled it", "did this for you",
                     "made", "fixed", "organized",
                     "помогу", "сделал для тебя", "позаботился", "починил"],
    },
    "receiving_gifts": {
        "description": "Thoughtful, meaningful tokens of affection",
        "expressing_patterns": [
            r"\bi (?:got|bought|found) (?:you |this for you)\b",
            r"\bi saw this and thought of you\b", r"\bsurprise\b",
            # Russian gift patterns
            r"я видел это и подумал о тебе", r"подарок", r"сюрприз",
            r"я (?:купил|нашёл) (?:тебе|для тебя)",
        ],
        "seeking_patterns": [
            r"\byou remembered\b", r"\bi(?:'ve| have) been wanting\b",
            r"\bi wish (?:someone|you) would\b",
            r"ты помнишь", r"я давно хотел[аи]?",
        ],
        "keywords": ["got you", "surprise", "thought of you", "gift",
                     "present", "bought", "found this for you",
                     "подарок", "сюрприз", "купил тебе", "подумал о тебе"],
    },
    "physical_touch": {
        "description": "Physical expressions of love and connection",
        "expressing_patterns": [
            r"\bwish i could (?:hug|hold|kiss|touch) you\b",
            r"\bcan(?:'t| not) wait to (?:hold|see|hug) you\b",
            r"\bi (?:want|need) (?:to )?(?:cuddle|hold you|be close)\b",
            # Russian physical touch patterns
            r"хочу обнять", r"хочу быть рядом",
            r"хочу (?:поцеловать|прижать|обнять) тебя",
            r"не могу дождаться (?:встречи|увидеть тебя)",
        ],
        "seeking_patterns": [
            r"\bi need a hug\b", r"\blet(?:'s| us) cuddle\b",
            r"\bi want (?:to be )?close to you\b",
            r"мне нужны обнимашки", r"давай обнимемся",
            r"хочу быть (?:ближе|рядом с тобой)",
        ],
        "keywords": ["hug", "hold", "touch", "cuddle", "kiss",
                     "close to you", "next to you", "❤️", "😘", "🤗",
                     "обнять", "рядом", "поцеловать", "обнимашки", "ближе"],
    },
}

# ==============================================================================
# 4. PLUTCHIK'S WHEEL OF EMOTIONS (Full Taxonomy)
# ==============================================================================

PLUTCHIK_PRIMARY = {
    "joy":          {"mild": "serenity",      "intense": "ecstasy",    "opposite": "sadness"},
    "trust":        {"mild": "acceptance",    "intense": "admiration",  "opposite": "disgust"},
    "fear":         {"mild": "apprehension",  "intense": "terror",     "opposite": "anger"},
    "surprise":     {"mild": "distraction",   "intense": "amazement",  "opposite": "anticipation"},
    "sadness":      {"mild": "pensiveness",   "intense": "grief",      "opposite": "joy"},
    "disgust":      {"mild": "boredom",       "intense": "loathing",   "opposite": "trust"},
    "anger":        {"mild": "annoyance",     "intense": "rage",       "opposite": "fear"},
    "anticipation": {"mild": "interest",      "intense": "vigilance",  "opposite": "surprise"},
}

PLUTCHIK_DYADS = {
    # Primary dyads (adjacent emotions)
    "love":          {"components": ["joy", "trust"],          "tier": "primary"},
    "submission":    {"components": ["trust", "fear"],         "tier": "primary"},
    "awe":           {"components": ["fear", "surprise"],      "tier": "primary"},
    "disapproval":   {"components": ["surprise", "sadness"],   "tier": "primary"},
    "remorse":       {"components": ["sadness", "disgust"],    "tier": "primary"},
    "contempt":      {"components": ["disgust", "anger"],      "tier": "primary"},
    "aggressiveness": {"components": ["anger", "anticipation"], "tier": "primary"},
    "optimism":      {"components": ["anticipation", "joy"],   "tier": "primary"},
    # Secondary dyads (two apart)
    "guilt":         {"components": ["joy", "fear"],           "tier": "secondary"},
    "curiosity":     {"components": ["trust", "surprise"],     "tier": "secondary"},
    "despair":       {"components": ["fear", "sadness"],       "tier": "secondary"},
    "disbelief":     {"components": ["surprise", "disgust"],   "tier": "secondary"},
    "envy":          {"components": ["sadness", "anger"],      "tier": "secondary"},
    "cynicism":      {"components": ["disgust", "anticipation"], "tier": "secondary"},
    "pride":         {"components": ["anger", "joy"],          "tier": "secondary"},
    "fatalism":      {"components": ["anticipation", "trust"], "tier": "secondary"},
    # Tertiary dyads (three apart)
    "delight":       {"components": ["joy", "surprise"],       "tier": "tertiary"},
    "sentimentality": {"components": ["trust", "sadness"],     "tier": "tertiary"},
    "shame":         {"components": ["fear", "disgust"],       "tier": "tertiary"},
    "indignation":   {"components": ["surprise", "anger"],     "tier": "tertiary"},
    "pessimism":     {"components": ["sadness", "anticipation"], "tier": "tertiary"},
    "morbidity":     {"components": ["disgust", "joy"],        "tier": "tertiary"},
    "domination":    {"components": ["anger", "trust"],        "tier": "tertiary"},
    "anxiety":       {"components": ["anticipation", "fear"],  "tier": "tertiary"},
}

# Text detection patterns for primary emotions
PLUTCHIK_DETECTION = {
    "joy": [r"\b(?:happy|glad|wonderful|great|awesome|yay|😊|😄|🎉)\b",
            r"(?:счастлив|рад[аы]?|замечательно|отлично|круто|ура|класс|супер|кайф)"],
    "trust": [r"\b(?:trust|believe|faith|rely|count on|safe with)\b",
              r"(?:доверяю|верю|верить|надёжный|надежный|уверен[а]?|безопасно)"],
    "fear": [r"\b(?:scared|afraid|worried|anxious|terrified|😰|😨)\b",
             r"(?:боюсь|страшно|тревожно|переживаю|волнуюсь|в ужасе|напуган[а]?)"],
    "surprise": [r"\b(?:wow|omg|what|no way|seriously|unexpected|😲|😮)\b",
                 r"(?:ого|офигеть|ничего себе|серьёзно|не может быть|вау|обалдеть)"],
    "sadness": [r"\b(?:sad|depressed|down|miss|lonely|crying|😢|😞|💔)\b",
                r"(?:грустно|печально|тоскливо|одиноко|плачу|скучаю|тяжело|уныло)"],
    "disgust": [r"\b(?:gross|disgusting|ew|sick|hate|awful|🤢|🤮)\b",
                r"(?:отвратительно|мерзко|тошнит|фу|гадость|противно|ненавижу)"],
    "anger": [r"\b(?:angry|furious|mad|pissed|frustrated|rage|😡|🤬)\b",
              r"(?:злюсь|злой|злая|бешусь|в ярости|бесит|взбешён|разъярён)"],
    "anticipation": [r"\b(?:excited|can(?:'t| not) wait|looking forward|hope|soon|🤞)\b",
                     r"(?:жду не дождусь|предвкушаю|надеюсь|скоро|с нетерпением|мечтаю)"],
}

# ==============================================================================
# 5. GOEMOTIONS - Google's 27-Category Emotion Taxonomy
# ==============================================================================

GOEMOTIONS_TAXONOMY = {
    # Positive emotions (12)
    "admiration":   {"valence": "positive", "keywords": ["amazing", "brilliant", "impressive", "wow",
                     "потрясающе", "гениально", "впечатляет", "вау", "офигеть"]},
    "amusement":    {"valence": "positive", "keywords": ["lol", "haha", "funny", "hilarious", "😂",
                     "хахаха", "ахахах", "ржу", "угар", "смешно"]},
    "approval":     {"valence": "positive", "keywords": ["good", "right", "agree", "exactly", "yes",
                     "согласен", "согласна", "правильно", "именно", "точно", "да"]},
    "caring":       {"valence": "positive", "keywords": ["hope you", "take care", "be safe", "worry about you",
                     "береги себя", "переживаю за тебя", "как ты", "всё хорошо"]},
    "desire":       {"valence": "positive", "keywords": ["want", "wish", "crave", "need you",
                     "хочу", "мечтаю", "нужен", "нужна", "хочу тебя"]},
    "excitement":   {"valence": "positive", "keywords": ["excited", "can't wait", "omg", "yay", "🎉",
                     "жду не дождусь", "ура", "круто", "не могу дождаться"]},
    "gratitude":    {"valence": "positive", "keywords": ["thank", "grateful", "appreciate", "thanks",
                     "спасибо", "благодарю", "ценю", "спасибочки"]},
    "joy":          {"valence": "positive", "keywords": ["happy", "glad", "wonderful", "love it",
                     "счастлив", "рад", "рада", "замечательно", "кайф"]},
    "love":         {"valence": "positive", "keywords": ["love you", "adore", "❤️", "😍", "sweetheart",
                     "люблю", "обожаю", "любимый", "любимая", "солнышко"]},
    "optimism":     {"valence": "positive", "keywords": ["hope", "will be fine", "things will", "look forward",
                     "надеюсь", "всё будет хорошо", "верю", "с нетерпением"]},
    "pride":        {"valence": "positive", "keywords": ["proud", "accomplished", "did it", "nailed it",
                     "горжусь", "молодец", "получилось", "сделал"]},
    "relief":       {"valence": "positive", "keywords": ["relieved", "phew", "thank god", "finally",
                     "фух", "слава богу", "наконец", "полегчало"]},
    # Negative emotions (11)
    "anger":        {"valence": "negative", "keywords": ["angry", "furious", "pissed", "mad", "how dare",
                     "злюсь", "бешусь", "в ярости", "бесит", "как ты смеешь"]},
    "annoyance":    {"valence": "negative", "keywords": ["annoying", "ugh", "stop", "irritating",
                     "раздражает", "достал", "достала", "надоело", "задолбало"]},
    "confusion":    {"valence": "ambiguous", "keywords": ["confused", "what", "huh", "don't understand",
                     "не понимаю", "что", "а", "в смысле", "запутался"]},
    "curiosity":    {"valence": "ambiguous", "keywords": ["curious", "wondering", "how", "why", "what if",
                     "интересно", "любопытно", "как", "почему", "а что если"]},
    "disappointment": {"valence": "negative", "keywords": ["disappointed", "let down", "expected more",
                       "разочарован", "разочарована", "ожидал большего", "обидно"]},
    "disapproval":  {"valence": "negative", "keywords": ["disagree", "wrong", "shouldn't", "terrible",
                     "не согласен", "неправильно", "не стоило", "ужасно"]},
    "disgust":      {"valence": "negative", "keywords": ["disgusting", "gross", "eww", "vile",
                     "отвратительно", "мерзко", "фу", "противно"]},
    "embarrassment": {"valence": "negative", "keywords": ["embarrassed", "awkward", "cringe", "😳",
                      "стыдно", "неловко", "кринж", "стрёмно"]},
    "fear":         {"valence": "negative", "keywords": ["scared", "afraid", "terrified", "worried",
                     "боюсь", "страшно", "в ужасе", "переживаю"]},
    "grief":        {"valence": "negative", "keywords": ["loss", "devastated", "heartbroken", "gone",
                     "потеря", "опустошён", "сердце разбито", "ушёл", "ушла"]},
    "nervousness":  {"valence": "negative", "keywords": ["nervous", "anxious", "butterflies", "stressed",
                     "нервничаю", "тревожно", "волнуюсь", "стресс"]},
    "remorse":      {"valence": "negative", "keywords": ["sorry", "regret", "my fault", "shouldn't have",
                     "извини", "прости", "жалею", "моя вина", "не надо было"]},
    "sadness":      {"valence": "negative", "keywords": ["sad", "depressed", "crying", "lonely", "miss",
                     "грустно", "тоскливо", "плачу", "одиноко", "скучаю"]},
    # Ambiguous (2 + neutral)
    "realization":  {"valence": "ambiguous", "keywords": ["oh", "realize", "just noticed", "wait",
                     "ого", "понял", "поняла", "только заметил", "подожди"]},
    "surprise":     {"valence": "ambiguous", "keywords": ["surprised", "unexpected", "no way", "what",
                     "удивлён", "неожиданно", "не может быть", "что"]},
    "neutral":      {"valence": "neutral", "keywords": []},
}

# ==============================================================================
# 6. KNAPP'S RELATIONAL DEVELOPMENT MODEL (10 Stages)
# ==============================================================================

KNAPP_STAGES = {
    # Coming Together (1-5)
    "initiating": {
        "phase": "coming_together",
        "order": 1,
        "description": "First contact, impression formation",
        "text_indicators": [
            r"\bhey\b", r"\bhi there\b", r"\bnice to meet\b",
            r"\bhow(?:'s| is) it going\b",
        ],
        "characteristics": ["short polished messages", "careful emoji use", "response time awareness"],
        "duration": "hours to days",
    },
    "experimenting": {
        "phase": "coming_together",
        "order": 2,
        "description": "Small talk, finding common ground",
        "text_indicators": [
            r"\bwhat do you (?:do|like|enjoy)\b", r"\bwhere (?:are|did) you\b",
            r"\byou like .+ too\b", r"\boh (?:cool|nice|interesting)\b",
        ],
        "characteristics": ["questions about interests", "surface sharing", "breadth without depth"],
        "duration": "days to weeks",
    },
    "intensifying": {
        "phase": "coming_together",
        "order": 3,
        "description": "Deeper disclosure, affection grows",
        "text_indicators": [
            r"\bi really (?:like|enjoy) talking to you\b", r"\bi(?:'ve| have) never told anyone\b",
            r"\byou(?:'re| are) (?:special|different|amazing)\b",
        ],
        "characteristics": ["longer messages", "pet names emerge", "inside jokes form",
                           "vulnerability increases", "more frequent texting"],
        "duration": "weeks to months",
    },
    "integrating": {
        "phase": "coming_together",
        "order": 4,
        "description": "Merging identities, 'we' language",
        "text_indicators": [
            r"\bwe should\b", r"\bour\b", r"\bus\b",
            r"\btold (?:my|her|his) (?:friend|mom|dad) about (?:us|you)\b",
        ],
        "characteristics": ["'we' replaces 'you and me'", "shared plans", "merged social circles"],
        "duration": "months",
    },
    "bonding": {
        "phase": "coming_together",
        "order": 5,
        "description": "Public commitment, formalization",
        "text_indicators": [
            r"\bour future\b", r"\bforever\b", r"\bmove in\b",
            r"\bpartner\b", r"\bcommit\b",
        ],
        "characteristics": ["relationship labels", "long-term planning", "official status"],
        "duration": "months to years",
    },
    # Coming Apart (6-10)
    "differentiating": {
        "phase": "coming_apart",
        "order": 6,
        "description": "Emphasis on individual differences",
        "text_indicators": [
            r"\bi need (?:my )?space\b", r"\bwe(?:'re| are) different\b",
            r"\bi(?:'m| am) my own person\b",
        ],
        "characteristics": ["increased 'I' vs 'we'", "separate plans", "emphasis on independence"],
        "warning_level": 0.3,
    },
    "circumscribing": {
        "phase": "coming_apart",
        "order": 7,
        "description": "Communication narrows, topics restricted",
        "text_indicators": [
            r"\blet(?:'s| us) not (?:talk|go) (?:about|there)\b",
            r"\bcan we (?:just )?not\b",
        ],
        "characteristics": ["shorter messages", "fewer topics", "avoidance of deep subjects",
                           "surface-level exchanges return"],
        "warning_level": 0.5,
    },
    "stagnating": {
        "phase": "coming_apart",
        "order": 8,
        "description": "Communication becomes routine and lifeless",
        "text_indicators": [
            r"\bsame (?:old|as usual)\b", r"\bnothing (?:new|much)\b",
        ],
        "characteristics": ["purely logistical texts", "no emotional content",
                           "predictable exchanges"],
        "warning_level": 0.7,
    },
    "avoiding": {
        "phase": "coming_apart",
        "order": 9,
        "description": "Active avoidance of contact",
        "text_indicators": [
            r"\bi(?:'m| am) busy\b", r"\bcan(?:'t| not) (?:talk|text|chat) (?:right )?now\b",
        ],
        "characteristics": ["unreturned messages", "excuses to not meet",
                           "long response delays"],
        "warning_level": 0.85,
    },
    "terminating": {
        "phase": "coming_apart",
        "order": 10,
        "description": "Ending the relationship",
        "text_indicators": [
            r"\bwe need to talk\b", r"\bit(?:'s| is) over\b",
            r"\bi(?:'m| am) done\b", r"\bbreaking up\b", r"\bgoodbye\b",
        ],
        "characteristics": ["final messages", "blocking", "unfollowing"],
        "warning_level": 1.0,
    },
}

# ==============================================================================
# 7. BIG FIVE (OCEAN) PERSONALITY TRAITS — Linguistic Markers
# ==============================================================================

BIG_FIVE_TRAITS = {
    "openness": {
        "description": "Intellectual curiosity, creativity, openness to experience",
        "high_indicators": {
            "linguistic": ["varied vocabulary", "tentative words", "abstract ideas"],
            "keywords": [r"\bperhaps\b", r"\bmaybe\b", r"\bwhat if\b", r"\bimagine\b",
                        r"\bphilosoph", r"\bcreativ", r"\bfascinat"],
            "message_style": "longer messages, varied vocabulary, discusses abstract ideas",
        },
        "low_indicators": {
            "linguistic": ["concrete nouns", "simple syntax", "practical topics"],
            "keywords": [r"\bpractical\b", r"\brealistic\b", r"\bsimple\b"],
            "message_style": "shorter, concrete messages, routine vocabulary",
        },
    },
    "conscientiousness": {
        "description": "Organization, dependability, self-discipline",
        "high_indicators": {
            "linguistic": ["proper grammar", "achievement words", "organized thoughts"],
            "keywords": [r"\baccomplish", r"\bgoal\b", r"\bprogress\b", r"\bplan\b",
                        r"\bschedul", r"\borgani"],
            "message_style": "proper grammar, on-time responses, structured messages",
        },
        "low_indicators": {
            "linguistic": ["casual grammar", "discrepancy words"],
            "keywords": [r"\bshould\b", r"\bwould\b", r"\bwhatever\b"],
            "message_style": "casual/messy grammar, inconsistent response times",
        },
    },
    "extraversion": {
        "description": "Energy, assertiveness, sociability",
        "high_indicators": {
            "linguistic": ["social words", "positive emotion", "exclamation marks"],
            "keywords": [r"\b!\b", r"\bparty\b", r"\beveryone\b", r"\bamazing\b",
                        r"\bfun\b", r"\bhang out\b"],
            "message_style": "frequent messages, initiates conversations, enthusiastic tone",
        },
        "low_indicators": {
            "linguistic": ["fewer messages", "reflective content"],
            "keywords": [r"\balone\b", r"\bquiet\b", r"\breflect\b"],
            "message_style": "fewer but longer messages, prefers 1-on-1, more reflective",
        },
    },
    "agreeableness": {
        "description": "Cooperation, trust, altruism",
        "high_indicators": {
            "linguistic": ["warm language", "positive emotion", "social process words"],
            "keywords": [r"\bof course\b", r"\bhappy to\b", r"\bno problem\b",
                        r"\bi agree\b", r"\bthat(?:'s| is) (?:a )?great (?:idea|point)\b"],
            "message_style": "frequent compliments, many emojis, supportive, avoids conflict",
        },
        "low_indicators": {
            "linguistic": ["direct/blunt", "fewer social niceties"],
            "keywords": [r"\bactually\b", r"\bwell\b.*\bbut\b", r"\bi disagree\b"],
            "message_style": "direct, fewer emojis, challenges others' positions",
        },
    },
    "neuroticism": {
        "description": "Emotional instability, anxiety, moodiness",
        "high_indicators": {
            "linguistic": ["first-person singular", "negative emotion", "anxiety words"],
            "keywords": [r"\bi(?:'m| am) (?:so )?(?:worried|stressed|anxious|scared|nervous)\b",
                        r"\bwhat if\b.*\bbad\b", r"\bi can(?:'t| not) (?:handle|deal|cope)\b",
                        r"\beverything is\b.*\bwrong\b"],
            "message_style": "mood variability, catastrophizing, seeks reassurance, high 'I' usage",
        },
        "low_indicators": {
            "linguistic": ["stable positive tone", "fewer 'I' statements"],
            "keywords": [r"\bit(?:'s| is) (?:fine|okay|all good)\b", r"\bno worries\b"],
            "message_style": "stable tone across messages, calm during conflict",
        },
    },
}

# ==============================================================================
# 8. NONVIOLENT COMMUNICATION (NVC) — Marshall Rosenberg
# ==============================================================================

NVC_COMPONENTS = {
    "observation": {
        "description": "Neutral description without judgment",
        "template": "When {specific_behavior}...",
        "violent_patterns": [
            r"\byou(?:'re| are) (?:so )?(?:rude|selfish|lazy|stupid)\b",
            r"\byou always\b", r"\byou never\b",
        ],
        "nvc_patterns": [
            r"\bwhen (?:i|you|we)\b.*\b(?:notice|see|hear)\b",
            r"\bwhen (?:that|this) happened\b",
        ],
    },
    "feeling": {
        "description": "Emotional state (not thought or interpretation)",
        "template": "I feel {emotion}...",
        "pseudo_feelings": [  # Not actual feelings — thoughts disguised as feelings
            "abandoned", "attacked", "betrayed", "blamed", "bullied",
            "cheated", "criticized", "disrespected", "ignored", "invalidated",
            "judged", "manipulated", "misunderstood", "neglected", "pressured",
            "rejected", "taken for granted", "threatened", "unappreciated", "used",
        ],
        "true_feelings_positive": [
            "alive", "amazed", "amused", "appreciative", "calm", "comfortable",
            "confident", "curious", "delighted", "eager", "ecstatic", "encouraged",
            "energetic", "enthusiastic", "excited", "exhilarated", "fulfilled",
            "glad", "grateful", "happy", "hopeful", "inspired", "intrigued",
            "joyful", "loving", "moved", "optimistic", "passionate", "peaceful",
            "playful", "proud", "refreshed", "relaxed", "relieved", "satisfied",
            "secure", "stimulated", "surprised", "tender", "thankful", "thrilled",
            "touched", "warm",
        ],
        "true_feelings_negative": [
            "afraid", "aggravated", "agitated", "alarmed", "angry", "anguished",
            "annoyed", "anxious", "apprehensive", "bitter", "bored", "brokenhearted",
            "concerned", "confused", "dejected", "depressed", "despairing",
            "disappointed", "discouraged", "disheartened", "dismayed", "distressed",
            "disturbed", "drained", "dull", "embarrassed", "exhausted", "frightened",
            "frustrated", "gloomy", "guilty", "heavy", "helpless", "hesitant",
            "horrified", "hostile", "hurt", "impatient", "indifferent", "insecure",
            "irritated", "jealous", "jittery", "lonely", "lost", "melancholy",
            "miserable", "nervous", "numb", "overwhelmed", "panicked", "perplexed",
            "puzzled", "reluctant", "resentful", "restless", "sad", "scared",
            "sensitive", "shocked", "skeptical", "sorrowful", "stressed", "tense",
            "terrified", "tired", "troubled", "uneasy", "unhappy", "unsettled",
            "upset", "vulnerable", "weary", "withdrawn", "worried", "wretched",
        ],
    },
    "need": {
        "description": "Universal human need underlying the feeling",
        "template": "Because I need {need}...",
        "categories": {
            "connection": ["acceptance", "affection", "appreciation", "closeness",
                          "communication", "companionship", "compassion", "empathy",
                          "inclusion", "intimacy", "love", "respect", "safety",
                          "security", "support", "to be seen", "to be heard",
                          "trust", "understanding", "warmth"],
            "autonomy": ["choice", "freedom", "independence", "space", "spontaneity"],
            "meaning": ["awareness", "challenge", "clarity", "competence",
                        "contribution", "creativity", "growth", "hope",
                        "learning", "participation", "purpose", "self-expression"],
            "peace": ["beauty", "ease", "equality", "harmony", "order"],
            "play": ["fun", "humor", "joy", "relaxation"],
            "honesty": ["authenticity", "integrity", "presence", "self-worth"],
            "physical": ["air", "food", "rest", "safety", "shelter", "touch", "water"],
        },
    },
    "request": {
        "description": "Clear, positive, doable action",
        "template": "Would you be willing to {action}?",
        "demand_patterns": [
            r"\byou (?:need|have|must|should) to\b",
            r"\bstop\b.*\b(?:it|doing|being)\b",
            r"\bdo(?:n't| not)\b.*\bever\b",
        ],
        "request_patterns": [
            r"\bwould you be willing\b", r"\bhow would you feel about\b",
            r"\bcould we try\b", r"\bwhat if we\b",
        ],
    },
}

# ==============================================================================
# 9. THOMAS-KILMANN CONFLICT MODE INSTRUMENT
# ==============================================================================

CONFLICT_MODES = {
    "competing": {
        "assertiveness": "high",
        "cooperativeness": "low",
        "description": "Win-lose; pursues own position at other's expense",
        "text_patterns": [
            r"\bthis is how it(?:'s| is) going to be\b",
            r"\bi don(?:'t| not) care what you (?:think|want)\b",
            r"\bmy way or\b", r"\btake it or leave it\b",
            r"так и будет", r"мне всё равно что ты думаешь",
            r"или так,? или никак", r"не обсуждается",
        ],
        "attachment_correlation": "anxious_preoccupied",
    },
    "collaborating": {
        "assertiveness": "high",
        "cooperativeness": "high",
        "description": "Win-win; finds solution satisfying both",
        "text_patterns": [
            r"\bhow can we both\b", r"\blet(?:'s| us) (?:brainstorm|figure|work)\b",
            r"\bwhat if we (?:tried|both)\b", r"\bi want us both\b",
            r"давай вместе (?:подумаем|решим|найдём)", r"как нам обоим",
            r"а что если мы", r"хочу чтобы нам обоим",
        ],
        "attachment_correlation": "secure",
    },
    "compromising": {
        "assertiveness": "medium",
        "cooperativeness": "medium",
        "description": "Partial win-win; both give something up",
        "text_patterns": [
            r"\bmeet (?:in )?the middle\b", r"\bfair enough\b",
            r"\bwhat if i .+ and you\b", r"\bi can live with\b",
            r"давай на полпути", r"справедливо", r"компромисс",
            r"я (?:уступлю|согласен|согласна),? а ты",
        ],
        "attachment_correlation": "secure",
    },
    "avoiding": {
        "assertiveness": "low",
        "cooperativeness": "low",
        "description": "Withdraws from conflict entirely",
        "text_patterns": [
            r"\blet(?:'s| us) not (?:talk|discuss) (?:about )?(?:this|it)\b",
            r"^(?:it(?:'s| is) fine|whatever|forget it|nvm|nevermind)$",
            r"давай не будем (?:об этом|это обсуждать)",
            r"^(?:ладно|пофиг|забей|не важно|проехали)$",
        ],
        "attachment_correlation": "dismissive_avoidant",
    },
    "accommodating": {
        "assertiveness": "low",
        "cooperativeness": "high",
        "description": "Yields to the other's position",
        "text_patterns": [
            r"\bwhatever you want\b", r"\bit(?:'s| is) (?:okay|ok),? i(?:'ll| will) (?:adjust|adapt)\b",
            r"\byour call\b", r"\bi don(?:'t| not) mind\b",
            r"как скажешь", r"как хочешь", r"мне всё равно",
            r"я подстроюсь", r"тебе решать",
        ],
        "attachment_correlation": "anxious_preoccupied",
    },
}

# ==============================================================================
# 10. CBT COGNITIVE DISTORTIONS (Aaron Beck)
# ==============================================================================

COGNITIVE_DISTORTIONS = {
    "all_or_nothing": {
        "description": "Sees in black and white; no middle ground",
        "detection_patterns": [
            r"\balways\b", r"\bnever\b", r"\bcompletely\b", r"\btotally\b",
            r"\bperfect\b", r"\bruined\b", r"\bnothing\b", r"\beverything\b",
        ],
        "example": "You NEVER listen to me. Everything is always about you.",
        "reframe": "introduce_spectrum",
        "reframe_template": "It sounds like this feels absolute right now. What parts are going well?",
    },
    "overgeneralization": {
        "description": "Single event extrapolated to universal pattern",
        "detection_patterns": [
            r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bno one\b",
            r"\bevery time\b", r"\bnothing ever\b",
        ],
        "example": "Every time I open up, I get hurt.",
        "reframe": "specific_instance",
        "reframe_template": "That sounds painful. Can you tell me about this specific time?",
    },
    "mental_filter": {
        "description": "Focuses exclusively on negatives, ignoring positives",
        "detection_patterns": [r"\byeah but\b", r"\bthat doesn(?:'t| not) matter\b"],
        "example": "The date was terrible (ignoring 3 hours of good conversation).",
        "reframe": "balanced_view",
        "reframe_template": "I hear the frustration. What parts did go well?",
    },
    "disqualifying_positive": {
        "description": "Dismisses positive experiences as not counting",
        "detection_patterns": [
            r"\bthat doesn(?:'t| not) count\b", r"\banyone would\b",
            r"\byou(?:'re| are) just saying that\b", r"\bit doesn(?:'t| not) mean\b",
        ],
        "example": "You only said that to make me feel better.",
        "reframe": "accept_positive",
        "reframe_template": "I mean it genuinely. What makes it hard to take in?",
    },
    "mind_reading": {
        "description": "Assumes knowledge of others' thoughts without evidence",
        "detection_patterns": [
            r"\bi know you think\b", r"\byou probably think\b",
            r"\bi can tell you\b.*\bdon(?:'t| not)\b",
            r"\bi bet you\b.*\bthink\b",
        ],
        "example": "I know you think I'm being dramatic right now.",
        "reframe": "check_assumptions",
        "reframe_template": "That's one possibility. What did they actually say?",
    },
    "fortune_telling": {
        "description": "Predicts negative future outcomes with certainty",
        "detection_patterns": [
            r"\bthis will never\b", r"\bit(?:'s| is) going to (?:fall apart|end|fail)\b",
            r"\bi just know\b.*\b(?:bad|wrong|fail|end)\b",
        ],
        "example": "This relationship is going to end just like all the others.",
        "reframe": "reality_test",
        "reframe_template": "That sounds really scary. What evidence do you have either way?",
    },
    "catastrophizing": {
        "description": "Imagining the worst possible scenario",
        "detection_patterns": [
            r"\bwhat if\b.*\b(?:worst|terrible|horrible|disaster)\b",
            r"\bthis is (?:a )?disaster\b", r"\bi can(?:'t| not) (?:handle|survive|cope)\b",
            r"\bthe worst thing\b",
        ],
        "example": "If you leave me I'll literally die.",
        "reframe": "perspective",
        "reframe_template": "I hear how overwhelming this feels. What's the most likely outcome?",
    },
    "emotional_reasoning": {
        "description": "Feelings treated as evidence of truth",
        "detection_patterns": [
            r"\bi feel (?:like )?(?:you don(?:'t| not)|it(?:'s| is)|they)\b",
            r"\bit feels like\b.*\bso (?:it )?must\b",
        ],
        "example": "I feel like you don't love me anymore, so you must not.",
        "reframe": "separate_feeling_fact",
        "reframe_template": "Your feelings are valid AND feelings aren't always facts. What would help you feel more sure?",
    },
    "should_statements": {
        "description": "Rigid rules about how things should be",
        "detection_patterns": [
            r"\byou should\b", r"\byou shouldn(?:'t| not)\b",
            r"\byou (?:have|must|need) to\b", r"\byou(?:'re| are) supposed to\b",
            r"\bi shouldn(?:'t| not) have to\b",
        ],
        "example": "You SHOULD have known how I felt.",
        "reframe": "convert_to_preference",
        "reframe_template": "It sounds like this is really important to you. What would you like to ask for?",
    },
    "labeling": {
        "description": "Assigning global labels based on specific events",
        "detection_patterns": [
            r"\bi(?:'m| am) (?:such )?(?:a |an )?(?:failure|idiot|loser|worthless|stupid)\b",
            r"\byou(?:'re| are) (?:a |an )?(?:terrible|horrible|awful|selfish)\b",
        ],
        "example": "I'm such a failure. I'll never be enough.",
        "reframe": "specific_behavior",
        "reframe_template": "One situation doesn't define you. What actually happened?",
    },
    "personalization": {
        "description": "Taking blame for things outside one's control",
        "detection_patterns": [
            r"\bit(?:'s| is) (?:all )?my fault\b", r"\bi caused this\b",
            r"\bif only i had\b", r"\bbecause of me\b", r"\bi ruin\b",
        ],
        "example": "The party was terrible because of me.",
        "reframe": "externalize_causes",
        "reframe_template": "There are many factors at play. What parts were actually in your control?",
    },
    "blame": {
        "description": "Holding others entirely responsible",
        "detection_patterns": [
            r"\bit(?:'s| is) (?:all )?your fault\b", r"\byou made me\b",
            r"\bbecause of you\b", r"\byou ruined\b",
        ],
        "example": "You made me feel this way.",
        "reframe": "shared_responsibility",
        "reframe_template": "I hear your frustration. What's your part in this, and what's theirs?",
    },
    "magnification": {
        "description": "Blowing up negatives, shrinking positives",
        "detection_patterns": [
            r"\btiny\b.*\bruined\b", r"\bone (?:small|little|tiny)\b.*\beverything\b",
        ],
        "example": "I made one tiny mistake and now everything is ruined.",
        "reframe": "proportional_view",
        "reframe_template": "Let's look at the big picture. How big is this really?",
    },
}

# ==============================================================================
# 11. DIGITAL BODY LANGUAGE — Text-Based Emotional Signals
# ==============================================================================

DIGITAL_BODY_LANGUAGE = {
    "response_time_increase": {
        "signal": "Sudden increase in response latency",
        "meanings": ["discomfort", "avoidance", "anger", "loss of interest", "deliberation"],
        "threshold": "3x personal baseline average",
    },
    "response_time_decrease": {
        "signal": "Sudden decrease in response latency",
        "meanings": ["heightened engagement", "anxiety", "excitement", "eagerness"],
        "threshold": "0.3x personal baseline average",
    },
    "message_length_decrease": {
        "signal": "Sudden shortening of messages",
        "meanings": ["withdrawal", "irritation", "stonewalling", "loss of interest"],
        "threshold": "below 30% of personal baseline average length",
    },
    "message_length_increase": {
        "signal": "Sudden increase in message length",
        "meanings": ["emotional flooding", "anxiety", "need to explain/justify"],
        "threshold": "above 200% of personal baseline average length",
    },
    "period_usage": {
        "signal": "Using period at end of single-sentence text",
        "meanings": ["perceived as less sincere", "possibly angry or passive-aggressive"],
        "research": "Klin et al. (2015): texts ending in periods rated as less sincere",
    },
    "exclamation_increase": {
        "signal": "Increased exclamation mark usage",
        "meanings": ["perceived as more sincere", "enthusiastic", "warm"],
    },
    "ellipsis": {
        "signal": "Trailing off with '...'",
        "meanings": ["hesitation", "uncertainty", "passive-aggression", "disappointment", "suggestiveness"],
    },
    "all_caps": {
        "signal": "Writing in ALL CAPITALS",
        "meanings": ["shouting", "anger", "extreme emphasis", "excitement"],
    },
    "emoji_decrease": {
        "signal": "Decrease in emoji usage from baseline",
        "meanings": ["emotional withdrawal", "cooling feelings", "distraction"],
    },
    "emoji_increase": {
        "signal": "Increase in emoji usage from baseline",
        "meanings": ["compensating for perceived coldness", "heightened positive affect", "masking"],
    },
    "emoji_type_shift": {
        "signal": "Shifting from hearts/kisses to neutral/sad emojis",
        "meanings": ["emotional state change", "moving from affection to distance"],
    },
    "read_no_reply": {
        "signal": "Read but no reply",
        "meanings": ["avoidance", "passive-aggression", "overwhelm", "busy"],
    },
    "typing_then_nothing": {
        "signal": "Extended typing indicator then no message sent",
        "meanings": ["self-censoring", "drafting and deleting", "internal conflict"],
    },
    "laughter_decrease": {
        "signal": "Using 'lol'/'haha' less frequently",
        "meanings": ["amusement fading", "discomfort masking dropped"],
    },
}

# ==============================================================================
# 12. BEHAVIORAL PATTERN DETECTION
# ==============================================================================

BEHAVIORAL_PATTERNS = {
    "ghosting": {
        "description": "Sudden cessation of all communication after regular messaging",
        "detection_rules": {
            "message_frequency_drop": "from regular to zero",
            "no_response_to_multiple_messages": True,
            "duration": "exceeds 3x their longest normal gap",
        },
        "severity": "high",
        "attachment_vulnerability": "anxious_preoccupied",
    },
    "breadcrumbing": {
        "description": "Sporadic minimal messages that avoid meaningful engagement",
        "detection_rules": {
            "message_frequency": "sporadic — below 20% of peak frequency",
            "message_content": "vague, flirtatious but non-committal",
            "avoids_plans": True,
            "pattern": "just enough contact to maintain hope, never escalates",
        },
        "severity": "medium",
        "prevalence": 0.35,  # 35% of dating app users
    },
    "love_bombing": {
        "description": "Excessive affection and intensity early in relationship",
        "detection_rules": {
            "message_frequency": "significantly above baseline for relationship stage",
            "emotional_intensity": "rapid escalation of intimacy language",
            "early_commitment": "I love you, future plans within days/weeks",
            "continuation_despite_no_reciprocation": True,
        },
        "severity": "high",
        "warning": "often precedes devalue/discard cycle",
    },
    "stonewalling_pattern": {
        "description": "Repeated withdrawal during emotional discussions",
        "detection_rules": {
            "withdrawal_during_conflict": True,
            "one_word_responses_after_emotional_topic": True,
            "leaves_conversation_unresolved": True,
        },
        "severity": "high",
        "gottman_horseman": "stonewalling",
    },
    "hot_cold_cycling": {
        "description": "Alternating between intense engagement and withdrawal",
        "detection_rules": {
            "high_variance_in_message_frequency": True,
            "high_variance_in_emotional_intensity": True,
            "period": "cycles of days to weeks",
        },
        "severity": "medium",
        "attachment_correlation": "fearful_avoidant",
    },
}

# ==============================================================================
# 13. ESConv EMPATHETIC RESPONSE FRAMEWORK
# ==============================================================================

EMPATHETIC_RESPONSE_INTENTS = {
    "questioning": {
        "description": "Guiding expression through open-ended questions",
        "frequency": 0.209,
        "stage": "exploration",
        "examples": ["What was that like for you?", "How did that make you feel?",
                     "Can you tell me more about that?"],
    },
    "restatement": {
        "description": "Confirming understanding by restating in own words",
        "frequency": 0.059,
        "stage": "exploration",
        "examples": ["So what you're saying is...", "Let me make sure I understand..."],
    },
    "reflection": {
        "description": "Naming and mirroring the detected emotion",
        "frequency": 0.078,
        "stage": "exploration",
        "examples": ["It sounds like you're feeling really frustrated",
                     "I can hear how much that hurt you"],
    },
    "self_disclosure": {
        "description": "Sharing relatable personal experiences",
        "frequency": 0.094,
        "stage": "comforting",
        "examples": ["I've been through something similar...",
                     "I know that feeling..."],
    },
    "affirmation": {
        "description": "Validating feelings and experiences",
        "frequency": 0.161,
        "stage": "comforting",
        "examples": ["That makes total sense", "Anyone would feel that way",
                     "Your feelings are completely valid"],
    },
    "suggestion": {
        "description": "Offering actionable guidance",
        "frequency": 0.156,
        "stage": "action",
        "examples": ["Have you tried...", "What if you...",
                     "Something that might help is..."],
    },
    "information": {
        "description": "Providing factual support",
        "frequency": 0.061,
        "stage": "action",
        "examples": ["Studies show that...", "It's actually common to..."],
    },
    "general": {
        "description": "General conversational acts",
        "frequency": 0.181,
        "stage": "any",
        "examples": ["I'm here for you", "Take your time"],
    },
}

ESCONV_STAGES = {
    "exploration": {
        "description": "Understanding the situation",
        "primary_intents": ["questioning", "reflection", "restatement"],
        "goal": "Build understanding of what happened and how they feel",
    },
    "comforting": {
        "description": "Emotional validation",
        "primary_intents": ["affirmation", "self_disclosure", "restatement"],
        "goal": "Make them feel heard and validated",
    },
    "action": {
        "description": "Moving toward solutions",
        "primary_intents": ["suggestion", "information"],
        "goal": "Help them take constructive next steps",
    },
}

# ==============================================================================
# 14. CROSS-CULTURAL COMMUNICATION
# ==============================================================================

CULTURAL_PROFILES = {
    "russian_eastern_european": {
        "directness": "high",
        "emotional_expression": "high — openly express emotions",
        "warmth_progression": "initial sternness → deep warmth once trust established",
        "trust_building": "through actions, not manners",
        "hidden_meanings": "direct words but indirect meaning through tone/context",
        "dating_norms": {
            "family_discussion": "earlier than Western norms",
            "relationship_purpose": "dating with purpose",
            "emotional_intensity": "openly expressed",
            "gender_roles": "more traditional expectations",
        },
        "texting_style": {
            "formality": "lower once trust established",
            "emoji_use": "high among younger generation",
            "endearment_tiers": {
                "casual": ["солнышко", "зайка", "малыш/малышка", "котик", "киса"],
                "intimate": ["любимый/любимая", "родной/родная", "сердечко", "золотце"],
                "deeply_committed": ["моя жизнь", "душа моя", "свет мой", "половинка"],
            },
        },
    },
    "western_american_british": {
        "directness": "moderate — hedge with politeness",
        "emotional_expression": "moderate — gradually escalating",
        "warmth_progression": "warm surface → deeper as relationship develops",
        "trust_building": "through consistent communication",
        "dating_norms": {
            "family_discussion": "later stages",
            "relationship_purpose": "often casual-first",
            "emotional_intensity": "gradual escalation expected",
            "gender_roles": "more egalitarian expectations",
        },
        "texting_style": {
            "formality": "casual from start",
            "emoji_use": "moderate",
        },
    },
}


# ==============================================================================
# UTILITY FUNCTIONS — Analysis and Detection
# ==============================================================================

def detect_four_horsemen(text: str) -> List[Dict[str, Any]]:
    """Detect Gottman's Four Horsemen in a message."""
    detected = []
    text_lower = text.lower()
    for horseman, data in FOUR_HORSEMEN.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append({
                    "horseman": horseman,
                    "severity": data["severity"],
                    "antidote": data["antidote"],
                    "antidote_template": data["antidote_template"],
                    "matched_pattern": pattern,
                })
                break
    return detected


def detect_emotional_bids(text: str) -> List[Dict[str, Any]]:
    """Detect emotional bids for connection in a message."""
    detected = []
    text_lower = text.lower()
    for bid_type, data in EMOTIONAL_BIDS["types"].items():
        for pattern in data["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append({
                    "type": bid_type,
                    "description": data["description"],
                })
                break
    return detected


def classify_bid_response(text: str, original_bid: str) -> str:
    """Classify how someone responded to an emotional bid."""
    text_lower = text.lower()
    # Check turning against
    against_patterns = [r"\bwho cares\b", r"\bso what\b", r"\bnot now\b", r"\bwhatever\b"]
    for p in against_patterns:
        if re.search(p, text_lower):
            return "turning_against"
    # Check turning toward
    toward_signals = ["?", "!", "that's", "tell me", "how", "wow", "really", "oh"]
    toward_count = sum(1 for s in toward_signals if s in text_lower)
    if toward_count >= 2 or len(text) > len(original_bid) * 0.5:
        return "turning_toward"
    # Default to turning away
    if len(text.strip()) < 5:
        return "turning_away"
    return "turning_toward"


def detect_knapp_stage(messages, relationship_duration_days: int = 0) -> Dict[str, Any]:
    """Detect current Knapp's relationship stage from recent messages."""
    stage_scores = {stage: 0.0 for stage in KNAPP_STAGES}
    combined_text = " ".join(
        (m.get("text", "") if isinstance(m, dict) else str(m)) for m in messages[-50:]
    ).lower()

    for stage_name, stage_data in KNAPP_STAGES.items():
        for pattern in stage_data.get("text_indicators", []):
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            stage_scores[stage_name] += matches

    # Duration-based bias
    if relationship_duration_days < 7:
        stage_scores["initiating"] += 3
    elif relationship_duration_days < 30:
        stage_scores["experimenting"] += 2
    elif relationship_duration_days < 90:
        stage_scores["intensifying"] += 1.5

    # Check for "we" language (integrating/bonding indicator)
    we_count = len(re.findall(r"\b(?:we|us|our)\b", combined_text))
    if we_count > 10:
        stage_scores["integrating"] += 2
        stage_scores["bonding"] += 1

    best_stage = max(stage_scores, key=stage_scores.get)
    return {
        "stage": best_stage,
        "phase": KNAPP_STAGES[best_stage]["phase"],
        "description": KNAPP_STAGES[best_stage]["description"],
        "confidence": min(stage_scores[best_stage] / max(sum(stage_scores.values()), 1), 1.0),
        "all_scores": stage_scores,
        "warning_level": KNAPP_STAGES[best_stage].get("warning_level", 0.0),
    }


def detect_love_language(messages, sender: str = "them") -> Dict[str, Any]:
    """Detect primary love language from message patterns."""
    scores = {lang: 0 for lang in LOVE_LANGUAGES}
    if messages and isinstance(messages[0], str):
        texts = messages
    else:
        texts = [m.get("text", "") for m in messages if m.get("sender", "") == sender]
    combined = " ".join(texts).lower()

    for lang_name, lang_data in LOVE_LANGUAGES.items():
        for pattern in lang_data["expressing_patterns"]:
            scores[lang_name] += len(re.findall(pattern, combined, re.IGNORECASE)) * 2
        for pattern in lang_data["seeking_patterns"]:
            scores[lang_name] += len(re.findall(pattern, combined, re.IGNORECASE)) * 3
        for kw in lang_data["keywords"]:
            if kw.lower() in combined:
                scores[lang_name] += 1

    total = max(sum(scores.values()), 1)
    primary = max(scores, key=scores.get)
    return {
        "primary": primary,
        "scores": {k: round(v / total, 3) for k, v in scores.items()},
        "description": LOVE_LANGUAGES[primary]["description"],
        "confidence": min(scores[primary] / max(total, 1), 1.0),
    }


def detect_cognitive_distortions(text: str) -> List[Dict[str, Any]]:
    """Detect CBT cognitive distortions in a message."""
    detected = []
    text_lower = text.lower()
    for distortion_name, data in COGNITIVE_DISTORTIONS.items():
        matches = 0
        for pattern in data["detection_patterns"]:
            matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        if matches >= 1:
            detected.append({
                "distortion": distortion_name,
                "description": data["description"],
                "match_count": matches,
                "confidence": min(matches * 0.3, 1.0),
                "reframe_strategy": data["reframe"],
                "reframe_template": data["reframe_template"],
            })
    return sorted(detected, key=lambda x: x["match_count"], reverse=True)


def detect_conflict_mode(text: str) -> Dict[str, Any]:
    """Detect Thomas-Kilmann conflict mode from message text."""
    scores = {}
    text_lower = text.lower()
    for mode_name, mode_data in CONFLICT_MODES.items():
        score = 0
        for pattern in mode_data["text_patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 1
        scores[mode_name] = score

    primary = max(scores, key=scores.get)
    if scores[primary] == 0:
        return {"mode": "none_detected", "scores": scores}

    return {
        "mode": primary,
        "description": CONFLICT_MODES[primary]["description"],
        "assertiveness": CONFLICT_MODES[primary]["assertiveness"],
        "cooperativeness": CONFLICT_MODES[primary]["cooperativeness"],
        "scores": scores,
        "attachment_correlation": CONFLICT_MODES[primary].get("attachment_correlation"),
    }


def detect_nvc_quality(text: str) -> Dict[str, Any]:
    """Analyze message for NVC quality (violent vs. nonviolent communication)."""
    text_lower = text.lower()
    violent_count = 0
    nvc_count = 0

    # Check observations
    for p in NVC_COMPONENTS["observation"]["violent_patterns"]:
        violent_count += len(re.findall(p, text_lower, re.IGNORECASE))
    for p in NVC_COMPONENTS["observation"]["nvc_patterns"]:
        nvc_count += len(re.findall(p, text_lower, re.IGNORECASE))

    # Check demands vs requests
    for p in NVC_COMPONENTS["request"]["demand_patterns"]:
        violent_count += len(re.findall(p, text_lower, re.IGNORECASE))
    for p in NVC_COMPONENTS["request"]["request_patterns"]:
        nvc_count += len(re.findall(p, text_lower, re.IGNORECASE))

    # Check pseudo-feelings vs true feelings
    pseudo_count = sum(1 for pf in NVC_COMPONENTS["feeling"]["pseudo_feelings"] if pf in text_lower)
    true_count = sum(1 for tf in NVC_COMPONENTS["feeling"]["true_feelings_negative"] if tf in text_lower)
    true_count += sum(1 for tf in NVC_COMPONENTS["feeling"]["true_feelings_positive"] if tf in text_lower)

    total = max(violent_count + nvc_count + pseudo_count + true_count, 1)
    nvc_score = (nvc_count + true_count) / total

    return {
        "nvc_score": round(nvc_score, 2),
        "violent_markers": violent_count,
        "nvc_markers": nvc_count,
        "pseudo_feelings_detected": pseudo_count,
        "true_feelings_detected": true_count,
        "communication_quality": "nonviolent" if nvc_score > 0.6 else "mixed" if nvc_score > 0.3 else "violent",
        "suggestion": NVC_COMPONENTS["observation"]["template"] if violent_count > 0 else None,
    }


def detect_big_five_indicators(messages, sender: str = "them") -> Dict[str, Any]:
    """Detect Big Five personality trait indicators from messages."""
    if messages and isinstance(messages[0], str):
        texts = messages
    else:
        texts = [m.get("text", "") for m in messages if m.get("sender", "") == sender]
    combined = " ".join(texts).lower()
    word_count = len(combined.split())

    if word_count < 20:
        return {"insufficient_data": True, "word_count": word_count}

    traits = {}
    for trait_name, trait_data in BIG_FIVE_TRAITS.items():
        high_score = 0
        low_score = 0
        for pattern in trait_data["high_indicators"]["keywords"]:
            high_score += len(re.findall(pattern, combined, re.IGNORECASE))
        for pattern in trait_data["low_indicators"]["keywords"]:
            low_score += len(re.findall(pattern, combined, re.IGNORECASE))

        total = max(high_score + low_score, 1)
        traits[trait_name] = {
            "score": round((high_score - low_score) / total, 2),
            "level": "high" if high_score > low_score else "low" if low_score > high_score else "moderate",
            "high_matches": high_score,
            "low_matches": low_score,
            "description": trait_data["description"],
        }

    return traits


def compute_gottman_ratio(messages, sender: str = "them"):
    """Compute Gottman's positive-to-negative interaction ratio."""
    positive_markers = [
        "love", "thank", "appreciate", "great", "amazing", "wonderful",
        "happy", "proud", "miss you", "❤️", "😘", "🥰", "😊", "💕",
        # Russian positive
        "люблю", "спасибо", "ценю", "отлично", "замечательно", "прекрасно",
        "счастлив", "горжусь", "скучаю по тебе", "молодец", "круто", "обожаю",
    ]
    negative_markers = [
        "hate", "angry", "upset", "annoyed", "frustrated", "terrible",
        "worst", "never", "always", "fault", "blame", "😡", "😤",
        # Russian negative
        "ненавижу", "злюсь", "расстроен", "бесит", "раздражает", "ужасно",
        "хуже", "никогда", "всегда", "вина", "виноват",
    ]

    if messages and isinstance(messages[0], str):
        texts = [t.lower() for t in messages]
    else:
        texts = [m.get("text", "").lower() for m in messages if m.get("sender", "") == sender]
    pos_count = sum(1 for t in texts for m in positive_markers if m in t)
    neg_count = sum(1 for t in texts for m in negative_markers if m in t)

    ratio = pos_count / max(neg_count, 1)
    return {
        "ratio": round(ratio, 2),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "healthy": ratio >= GOTTMAN_MAGIC_RATIO,
        "assessment": (
            "Thriving" if ratio >= 7
            else "Healthy" if ratio >= GOTTMAN_MAGIC_RATIO
            else "Needs attention" if ratio >= 3
            else "At risk" if ratio >= 1
            else "Critical"
        ),
        "target": GOTTMAN_MAGIC_RATIO,
    }


def detect_repair_attempts(text: str) -> List[Dict[str, Any]]:
    """Detect Gottman's repair attempts in a message."""
    detected = []
    text_lower = text.lower()
    for repair_type, data in REPAIR_ATTEMPTS.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append({
                    "type": repair_type,
                    "effectiveness": data["effectiveness"],
                })
                break
    return detected


def detect_behavioral_pattern(
    message_timestamps: List[float],
    message_lengths: List[int],
    sentiment_scores: List[float],
) -> List[Dict[str, Any]]:
    """Detect concerning behavioral patterns from message time series."""
    detected = []
    n = len(message_timestamps)
    if n < 10:
        return []

    # Calculate rolling averages
    recent_gap = []
    for i in range(max(0, n - 5), n - 1):
        recent_gap.append(message_timestamps[i + 1] - message_timestamps[i])
    older_gap = []
    for i in range(max(0, n - 15), max(0, n - 5)):
        if i + 1 < n:
            older_gap.append(message_timestamps[i + 1] - message_timestamps[i])

    avg_recent = sum(recent_gap) / max(len(recent_gap), 1)
    avg_older = sum(older_gap) / max(len(older_gap), 1)

    # Ghosting detection
    if avg_older > 0 and avg_recent / max(avg_older, 1) > 5:
        detected.append({
            "pattern": "possible_ghosting",
            "severity": "high",
            "evidence": f"Response gaps increased {avg_recent / max(avg_older, 1):.1f}x",
        })

    # Message length decline
    recent_len = sum(message_lengths[-5:]) / min(len(message_lengths), 5)
    older_len = sum(message_lengths[-15:-5]) / max(min(len(message_lengths) - 5, 10), 1) if n > 5 else recent_len
    if older_len > 0 and recent_len / max(older_len, 1) < 0.3:
        detected.append({
            "pattern": "engagement_decline",
            "severity": "medium",
            "evidence": f"Message length dropped to {recent_len / max(older_len, 1) * 100:.0f}% of baseline",
        })

    # Hot-cold cycling
    if len(sentiment_scores) >= 10:
        diffs = [abs(sentiment_scores[i] - sentiment_scores[i - 1]) for i in range(1, len(sentiment_scores))]
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 0.5:
            detected.append({
                "pattern": "hot_cold_cycling",
                "severity": "medium",
                "evidence": f"Sentiment volatility: {avg_diff:.2f} average swing",
            })

    return detected


def select_empathetic_response_strategy(
    emotional_state: str,
    conversation_stage: str = "exploration",
    conflict_active: bool = False,
) -> Dict[str, Any]:
    """Select the best empathetic response strategy based on context."""
    strategy_map = {
        "grief": {"primary": "reflection", "secondary": "affirmation"},
        "sadness": {"primary": "reflection", "secondary": "affirmation"},
        "anger": {"primary": "restatement", "secondary": "questioning"},
        "fear": {"primary": "affirmation", "secondary": "reflection"},
        "confusion": {"primary": "questioning", "secondary": "information"},
        "excitement": {"primary": "affirmation", "secondary": "self_disclosure"},
        "frustration": {"primary": "reflection", "secondary": "restatement"},
        "anxiety": {"primary": "affirmation", "secondary": "suggestion"},
        "love": {"primary": "self_disclosure", "secondary": "affirmation"},
        "joy": {"primary": "affirmation", "secondary": "self_disclosure"},
    }

    strategy = strategy_map.get(emotional_state, {"primary": "questioning", "secondary": "reflection"})

    # Override if conflict is active
    if conflict_active:
        strategy = {"primary": "restatement", "secondary": "reflection"}

    # Get ESConv stage guidance
    esconv = ESCONV_STAGES.get(conversation_stage, ESCONV_STAGES["exploration"])

    primary_intent = EMPATHETIC_RESPONSE_INTENTS[strategy["primary"]]
    secondary_intent = EMPATHETIC_RESPONSE_INTENTS[strategy["secondary"]]

    return {
        "primary_strategy": strategy["primary"],
        "primary_examples": primary_intent["examples"],
        "secondary_strategy": strategy["secondary"],
        "secondary_examples": secondary_intent["examples"],
        "esconv_stage": conversation_stage,
        "esconv_goal": esconv["goal"],
        "recommended_intents": esconv["primary_intents"],
    }


def comprehensive_psychological_analysis(
    messages=None,
    their_messages_only=None,
    user_messages=None,
    partner_messages=None,
) -> Dict[str, Any]:
    """Run ALL psychological frameworks on a conversation and return comprehensive analysis."""
    if messages is None:
        messages = []
    # Normalize: accept List[str] or List[Dict]
    if messages and isinstance(messages[0], str):
        messages = [{"text": m, "sender": "unknown"} for m in messages]
    if user_messages and isinstance(user_messages[0], str):
        user_messages = [{"text": m, "sender": "me"} for m in user_messages]
    if partner_messages and isinstance(partner_messages[0], str):
        partner_messages = [{"text": m, "sender": "them"} for m in partner_messages]

    if their_messages_only is None:
        if partner_messages:
            their_messages_only = partner_messages
        else:
            their_messages_only = [m for m in messages if m.get("sender", "") != "me"]

    # Merge user + partner messages if provided separately
    if user_messages and partner_messages and not any(m.get("sender") in ("me", "them") for m in messages):
        messages = user_messages + partner_messages

    last_message = messages[-1] if messages else {"text": ""}
    last_text = last_message.get("text", "") if isinstance(last_message, dict) else str(last_message)

    analysis = {
        "gottman": {
            "ratio": compute_gottman_ratio(messages, sender="them"),
            "four_horsemen": detect_four_horsemen(last_text),
            "emotional_bids": detect_emotional_bids(last_text),
            "repair_attempts": detect_repair_attempts(last_text),
        },
        "love_language": detect_love_language(messages, sender="them"),
        "knapp_stage": detect_knapp_stage(messages),
        "cognitive_distortions": detect_cognitive_distortions(last_text),
        "conflict_mode": detect_conflict_mode(last_text),
        "nvc_quality": detect_nvc_quality(last_text),
        "big_five": detect_big_five_indicators(messages, sender="them"),
        "empathetic_strategy": select_empathetic_response_strategy(
            emotional_state="neutral",
        ),
    }

    # Add emotional bid in last message
    bids = analysis["gottman"]["emotional_bids"]
    if bids:
        analysis["empathetic_strategy"] = select_empathetic_response_strategy(
            emotional_state="caring" if bids[0]["type"] == "emotional_support" else "joy",
        )

    # Add warning flags
    warnings = []
    if not analysis["gottman"]["ratio"]["healthy"]:
        warnings.append(f"Gottman ratio is {analysis['gottman']['ratio']['ratio']:.1f}:1 (target: 5:1)")
    if analysis["gottman"]["four_horsemen"]:
        horsemen = [h["horseman"] for h in analysis["gottman"]["four_horsemen"]]
        warnings.append(f"Four Horsemen detected: {', '.join(horsemen)}")
    if analysis["knapp_stage"]["warning_level"] > 0:
        warnings.append(f"Knapp stage: {analysis['knapp_stage']['stage']} (coming apart phase)")
    if analysis["cognitive_distortions"]:
        dist_names = [d["distortion"] for d in analysis["cognitive_distortions"][:3]]
        warnings.append(f"Cognitive distortions: {', '.join(dist_names)}")
    if analysis["nvc_quality"]["communication_quality"] == "violent":
        warnings.append("Violent communication patterns detected")

    analysis["warnings"] = warnings
    analysis["overall_health"] = "healthy" if not warnings else "needs_attention" if len(warnings) <= 2 else "at_risk"

    return analysis


def format_psychological_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """Format psychological analysis for prompt — raw data only, no therapy frameworks.

    NO validation templates, NO NVC, NO antidotes, NO reframing.
    Just tell the LLM what's happening so it can react like a real person.
    """
    sections = []

    # Conflict mode — just the raw detection
    cm = analysis.get("conflict_mode", {})
    if cm and cm.get("mode") != "none_detected":
        mode = cm["mode"]
        sections.append(
            f"[CONFLICT] They're in {mode} mode — match their energy and stand your ground"
        )

    # Horsemen — flag toxic behavior without therapy antidotes
    horsemen = analysis.get("gottman", {}).get("four_horsemen", [])
    if horsemen:
        h = horsemen[0]
        horseman = h["horseman"]
        if horseman == "contempt":
            sections.append("[WARNING] They're being contemptuous — don't take that, fire back")
        elif horseman == "criticism":
            sections.append("[WARNING] They're attacking your character — defend yourself")
        elif horseman == "defensiveness":
            sections.append("[WARNING] They're being super defensive — call it out")
        elif horseman == "stonewalling":
            sections.append("[WARNING] They're shutting down — don't chase, match their energy")

    # Warnings
    warnings = analysis.get("warnings", [])
    if warnings:
        sections.append(f"[FLAGS] {' | '.join(warnings)}")

    return "\n".join(sections) if sections else ""
