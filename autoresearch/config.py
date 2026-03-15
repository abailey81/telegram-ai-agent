"""Autoresearch configuration — all tunable constants in one place."""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
AUTORESEARCH_DIR = Path(__file__).parent
DATA_DIR = AUTORESEARCH_DIR / "data"
MODEL_VERSIONS_DIR = AUTORESEARCH_DIR / "model_versions"
ROLLBACK_DIR = MODEL_VERSIONS_DIR / "rollback"
RESULTS_FILE = AUTORESEARCH_DIR / "results.tsv"
PROGRAM_FILE = AUTORESEARCH_DIR / "program.md"

RL_DATA_DIR = PROJECT_ROOT / "rl_data"
ENGINE_DATA_DIR = PROJECT_ROOT / "engine_data"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
NEURAL_MODELS_DIR = TRAINED_MODELS_DIR / "neural"
VOICE_DATA_DIR = ENGINE_DATA_DIR / "voice"

# Ensure directories exist
for d in [DATA_DIR, MODEL_VERSIONS_DIR, ROLLBACK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Experiment Budget ===
BUDGET_SECONDS = 900  # 15 minutes per experiment (neural needs ~200s data prep + training)
DEFAULT_N_EXPERIMENTS = 10  # experiments per run

# === Data Harvesting ===
MIN_REWARD_THRESHOLD = 0.5   # only harvest experiences with reward >= this
MIN_TEXT_LENGTH = 5           # skip messages shorter than this
HARVESTED_WEIGHT = 0.5       # weight of harvested vs curated examples (1.0 = equal)

# === Model Versioning ===
MAX_MODEL_VERSIONS = 50      # prune oldest non-best versions beyond this
ROLLBACK_THRESHOLD = 0.10    # 10% degradation in live RL rewards triggers rollback
ROLLBACK_WINDOW = 50         # check over this many interactions

# === Composite Score Weights ===
TASK_WEIGHTS = {
    "emotional_tone": 0.40,
    "romantic_intent": 0.35,
    "conversation_stage": 0.25,
}

# === Context Key -> Romantic Intent Mapping ===
# Must match valid romantic_intent labels in training_data.py:
# advice_seeking, angry, apology, casual, curious, distant, flirty, goodbye,
# grateful, greeting, hurt, jealous, opinion, planning, plans, playful,
# romantic, sad, serious, sharing, sincere, small_talk, supportive, testing, venting
CONTEXT_TO_INTENT = {
    "light_neutral": "casual",
    "light_positive": "casual",
    "light_negative": "supportive",
    "deep_emotional": "sharing",
    "deep_positive": "romantic",
    "deep_negative": "supportive",
    "flirty": "flirty",
    "conflict": "venting",
    "reconnect": "sincere",
    "planning": "planning",
}

# === Conversation State -> Stage Mapping ===
# Must match valid conversation_stage labels in training_data.py:
# advising, brainstorming, closing, conflict, cooling_down, debating, deep,
# deep_conversation, flirting, flowing, makeup, opening, resolution, small_talk,
# storytelling, topic_discussion, venting, warming_up
STATE_TO_STAGE = {
    "flowing": "flowing",
    "warming_up": "warming_up",
    "cooling": "closing",
    "engaged": "deep",
    "escalating": "conflict",
    "de_escalating": "cooling_down",
    "initial": "opening",
    "stalled": "closing",
}

# === Voice Experiment Parameters ===
VOICE_PARAM_GRID = {
    "cfg_weight": [0.2, 0.3, 0.35, 0.4, 0.5],
    "exaggeration": [0.2, 0.3, 0.4, 0.5],
    "temperature": [0.65, 0.7, 0.75, 0.8],
    "repetition_penalty": [1.5, 1.8, 2.0, 2.2],
}
VOICE_TEST_PHRASES_EN = [
    "Hey, I was thinking about you today.",
    "That sounds really fun, let's do it this weekend!",
    "I'm not sure I agree, but I understand your point.",
    "You always know how to make me smile.",
    "Good morning! Hope you slept well.",
]
VOICE_TEST_PHRASES_RU = [
    "Привет, я сегодня думал о тебе.",
    "Звучит здорово, давай сделаем это на выходных!",
    "Я не уверен, что согласен, но я понимаю твою точку зрения.",
    "Ты всегда знаешь, как заставить меня улыбнуться.",
    "Доброе утро! Надеюсь, ты хорошо выспалась.",
]

# === Experiment Types ===
# Autoresearch rotates through these experiment types to improve the ENTIRE project
EXPERIMENT_TYPES = [
    "neural",          # TextCNN / EmotionAttentionNet hyperparameters
    "sklearn",         # sklearn classifier hyperparameter search
    "rl_params",       # RL engine reward weights, decay, exploration
    "engine_params",   # Conversation + emotional intelligence thresholds
    "voice",           # Voice engine TTS parameter optimization
]

# Rotation weights — higher = more experiments of this type
EXPERIMENT_TYPE_WEIGHTS = {
    "neural": 0.30,
    "sklearn": 0.25,
    "rl_params": 0.20,
    "engine_params": 0.15,
    "voice": 0.10,
}

# === sklearn Classifier Hyperparameter Search Space ===
SKLEARN_PARAM_SPACE = {
    "svm": {
        "C": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "class_weight": ["balanced", None],
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 5.0, 10.0, 50.0],
        "solver": ["lbfgs", "saga"],
        "max_iter": [1000, 2000, 5000],
        "class_weight": ["balanced", None],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
}
SKLEARN_TASKS = ["emotional_tone", "romantic_intent", "conversation_stage"]

# === RL Engine Tunable Parameters ===
RL_PARAM_SPACE = {
    # Reward signal weights (must sum to 1.0)
    "reward_weights": {
        "response_received": [0.15, 0.20, 0.25, 0.30],
        "response_speed": [0.05, 0.10, 0.15],
        "length_maintenance": [0.10, 0.15, 0.20],
        "emotional_valence": [0.15, 0.20, 0.25, 0.30],
        "engagement_signals": [0.10, 0.15, 0.20],
        "emoji_sentiment": [0.05, 0.10, 0.15],
        "conversation_continuation": [0.03, 0.05, 0.10],
    },
    # Thompson sampling parameters
    "decay_rate": [0.90, 0.93, 0.95, 0.97, 0.99],
    "decay_trigger": [30, 50, 75, 100],
    "match_bonus": [0.5, 1.0, 1.5, 2.0],
    # Response speed thresholds (seconds)
    "speed_thresholds": [
        [30, 180, 600, 1800],    # aggressive
        [60, 300, 900, 3600],    # current
        [120, 600, 1800, 7200],  # relaxed
    ],
}

# === Conversation Engine Tunable Parameters ===
ENGINE_PARAM_SPACE = {
    # Context assembly
    "recency_decay": [0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    "recency_weight": [2.0, 2.5, 3.0, 3.5, 4.0],
    "max_messages": [15, 20, 25, 30],
    # State detection
    "state_confidence_threshold": [0.10, 0.12, 0.15, 0.18, 0.20],
    # Emotion profiling
    "baseline_valence_low": [0.30, 0.35, 0.40],
    "baseline_valence_high": [0.60, 0.65, 0.70],
    # Attachment style weights
    "anxious_weight": [1.0, 1.25, 1.5, 1.75, 2.0],
    "avoidant_weight": [1.0, 1.25, 1.5, 1.75, 2.0],
    "secure_weight": [0.75, 1.0, 1.25],
    # Intensity calibration
    "intensity_floor": [0.2, 0.25, 0.3, 0.35],
    "intensity_scale": [0.6, 0.65, 0.7, 0.75, 0.8],
    # Style engine thresholds
    "emoji_density_high": [0.4, 0.5, 0.6, 0.7],
    "emoji_density_low": [0.05, 0.08, 0.10, 0.15],
    "formality_casual_threshold": [0.2, 0.25, 0.3, 0.35],
    "formality_formal_threshold": [0.6, 0.65, 0.7, 0.75],
    "humor_frequency_threshold": [0.2, 0.25, 0.3, 0.35, 0.4],
    # Memory engine retention
    "max_facts": [30, 40, 50, 60, 75],
    "max_episodes": [50, 75, 100, 150],
    "max_milestones": [15, 20, 25, 30],
    "memory_relevance_cutoff": [2.0, 2.5, 3.0, 3.5, 4.0],
    "semantic_similarity_boost": [3.0, 4.0, 5.0, 6.0, 7.0],
    # Response quality scoring
    "staleness_threshold": [0.65, 0.70, 0.75, 0.80, 0.85],
    "repetition_penalty": [-10, -12, -15, -18, -20],
    "ai_detection_penalty": [-5, -8, -10, -12, -15],
    # Orchestrator temperature
    "base_temperature": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "conflict_temperature": [0.6, 0.65, 0.7, 0.75],
    "creative_temperature": [0.9, 0.95, 1.0, 1.05],
}

# === RL Evaluation — uses historical reward data ===
RL_EVAL_FILE = AUTORESEARCH_DIR / "rl_param_results.json"
ENGINE_EVAL_FILE = AUTORESEARCH_DIR / "engine_param_results.json"
SKLEARN_MODELS_DIR = TRAINED_MODELS_DIR  # sklearn models live at trained_models/ root
