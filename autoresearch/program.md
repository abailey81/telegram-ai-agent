# Research Program: Full-Project Autonomous Improvement

## Primary Objectives

### 1. Neural Networks (30% of experiments)
Maximize weighted validation F1 across all three classification tasks:

    score = 0.4 * emotional_tone_f1 + 0.35 * romantic_intent_f1 + 0.25 * conversation_stage_f1

### 2. sklearn Classifiers (25% of experiments)
Improve cross-validated accuracy/F1 for SVM, Random Forest, Logistic Regression, and Gradient Boosting classifiers on the same 3 tasks. Currently 93-97% accuracy.

### 3. RL Engine Parameters (20% of experiments)
Optimize reward signal weights, Thompson sampling decay rates, match bonus, and speed thresholds by replaying 121+ historical experiences. Maximize reward mean, high-reward ratio, and reward stability.

### 4. Engine Parameters (15% of experiments)
Optimize 27 tunable parameters across 7 engines:
- **Conversation engine**: recency_decay, recency_weight, max_messages, state_confidence_threshold
- **Emotional intelligence**: baseline_valence_low/high, intensity_floor/scale, anxious/avoidant/secure_weight
- **Style engine**: emoji_density_high/low, formality_casual/formal_threshold, humor_frequency_threshold
- **Memory engine**: max_facts, max_episodes, max_milestones, memory_relevance_cutoff, semantic_similarity_boost
- **NLP scoring**: staleness_threshold, repetition_penalty, ai_detection_penalty
- **Orchestrator**: base_temperature, conflict_temperature, creative_temperature

### 5. Voice Engine (10% of experiments)
Grid search over Chatterbox TTS parameters (cfg_weight, exaggeration, temperature, repetition_penalty).

## Current Baselines
Check `trained_models/neural/*_meta.json` and `trained_models/*_meta.json` for current accuracy numbers. The sklearn classifiers (SVM, GradientBoosting) typically hit 93-97% accuracy. Neural models (TextCNN, EmotionAttentionNet) range from 61-95%.

Primary improvement targets:
- EmotionAttentionNet on romantic_intent (currently ~61% — weakest model)
- EmotionAttentionNet on emotional_tone
- TextCNN on all tasks (solid baseline, room for improvement)

## Constraints
- Training budget: 5 minutes per experiment (hard limit)
- Must use all-MiniLM-L6-v2 embeddings (384-dim) — do NOT change the embedder
- Must maintain compatibility with existing model loading
- Models must produce valid class probabilities via softmax
- Do not install new dependencies beyond what's in pyproject.toml
- RL/engine parameter changes must be backward-compatible (fallback to defaults)

## What You Can Modify

### Neural Networks
- Hyperparameters in the HPARAMS dict in train.py
- Model architecture (num_filters, kernel_sizes, num_heads, ff_dim, num_layers, dropout)
- Optimizer choice and parameters (adamw, sgd, adam; weight_decay, momentum)
- Learning rate schedule (cosine, step, plateau, warmup steps)
- Data handling (use_harvested_data, harvested_weight, label_smoothing)

### sklearn Classifiers
- Classifier type (SVM, RF, LR, GBT) and their hyperparameters
- Whether to use harvested data and at what weight

### RL Engine
- Reward signal weights (must sum to 1.0)
- Thompson sampling decay_rate, decay_trigger, match_bonus
- Response speed thresholds

### Engine Parameters
- Any of the 27 parameters defined in config.py ENGINE_PARAM_SPACE

## Ideas to Explore
1. EmotionAttentionNet: increase num_layers (2->4), different num_heads (4->8), larger ff_dim
2. Label smoothing (0.05-0.15) for better calibration
3. Learning rate warmup (5-10% of epochs) + cosine decay
4. Vary harvested_weight from 0.0 to 1.0 to find optimal mix
5. TextCNN kernel_sizes: try [3,4,5,6] or [2,3,5,7]
6. sklearn: try poly kernel SVM, tune C across wider range
7. RL: shift weight from response_received toward emotional_valence
8. RL: faster or slower decay to adapt to user behavior changes
9. Engine: wider context window (max_messages=30) for more context
10. Engine: lower state_confidence_threshold for more nuanced state detection
11. Engine: adjust attachment weights based on conversation patterns
12. Temperature: lower conflict temperature for more controlled responses

## Decision-Making Philosophy
- Prioritize simplicity alongside performance
- Small improvements with ugly complexity = probably not worth it
- Each experiment type tracks its own best score — improvements are independent
- Winning configs are auto-promoted via JSON files → live engines pick up automatically
- Always log your reasoning in the experiment notes
