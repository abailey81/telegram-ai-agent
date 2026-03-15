# Telegram AI Agent

A production-grade conversational AI platform with 13 intelligence engines, reinforcement learning, deep learning classifiers, voice synthesis, and autonomous model improvement. Deployed as a Telegram integration via MCP and FastAPI.

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Telegram Cloud                  │
                    └─────────────────┬───────────────────────────┘
                                      │ Telethon MTProto
                    ┌─────────────────▼───────────────────────────┐
                    │         telegram_api.py (FastAPI)            │
                    │         Port 8765 · REST + WebSocket         │
                    │         90+ endpoints · Rich dashboard       │
                    └──────┬──────────────────────────┬───────────┘
                           │                          │
              ┌────────────▼────────┐    ┌────────────▼────────────┐
              │   main.py (MCP)     │    │  agent/ (TypeScript)    │
              │   90+ MCP tools     │    │  Claude Sonnet CLI      │
              │   Claude Desktop    │    │  Vercel AI Gateway      │
              └────────────┬────────┘    └─────────────────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │       Orchestrator (orchestrator.py) │
         │    Multi-engine cascade · Temperature │
         │    control · Conflict routing         │
         └─────────┬────────────────┬───────────┘
                   │                │
    ┌──────────────▼──┐  ┌─────────▼────────────────────────────┐
    │  ML Pipeline     │  │  13 Intelligence Engines              │
    │                  │  │                                       │
    │  TextCNN         │  │  NLP · Emotional Intelligence ·      │
    │  EmotionAttnNet  │  │  Conversation · Style · Memory ·     │
    │  SVM / GBT / LR  │  │  Reasoning · Personality ·           │
    │  Thompson RL     │  │  Prediction · Voice · Media ·        │
    │                  │  │  Advanced Intelligence · Media AI    │
    └──────────┬───────┘  └─────────────────────────────────────┘
               │
    ┌──────────▼───────┐
    │  Autoresearch     │
    │  Autonomous       │
    │  improvement loop │
    │  (5 experiment    │
    │  types)           │
    └──────────────────┘
```

## Intelligence Engines

| # | Engine | Module | Capabilities |
|---|--------|--------|-------------|
| 1 | **NLP** | `nlp_engine.py` | Sentiment analysis, topic extraction, language detection, passive-aggression detection |
| 2 | **Advanced Intelligence** | `advanced_intelligence.py` | GoEmotions 28-label detection, subtext analysis, persona consistency |
| 3 | **Conversation** | `conversation_engine.py` | Weighted context assembly, dialogue acts, 13-state machine, goal tracking |
| 4 | **Emotional Intelligence** | `emotional_intelligence.py` | VAD profiling, Plutchik's Wheel (32 emotions), attachment style detection |
| 5 | **Style** | `style_engine.py` | Big Five personality traits, love languages, digital body language, communication mirroring |
| 6 | **Memory** | `memory_engine.py` | Three-tier memory (semantic / episodic / procedural) with FAISS vector retrieval |
| 7 | **Reasoning** | `reasoning_engine.py` | Chain-of-thought, chain-of-empathy, conflict mode detection, model cascade |
| 8 | **Personality** | `personality_engine.py` | Trait detection from linguistic markers, consistency scoring |
| 9 | **Prediction** | `prediction_engine.py` | Intent prediction, behavioral forecasting, next-action estimation |
| 10 | **Reinforcement Learning** | `rl_engine.py` | Thompson sampling contextual bandits, per-user strategy optimization |
| 11 | **Voice** | `voice_engine.py` | 4-backend TTS (Chatterbox, F5-TTS, Bark, Edge-TTS), zero-shot voice cloning |
| 12 | **Media Intelligence** | `media_intelligence.py` | Voice/text classification, multilingual embeddings (bge-m3) |
| 13 | **Media AI** | `media_ai.py` | Voice transcription (faster-whisper), image understanding, speech synthesis |

## ML / Deep Learning Pipeline

### Classification Models

Three classification tasks trained on 5,000-6,600 labeled examples each:

| Task | Classes | sklearn Accuracy | Neural F1 |
|------|---------|-----------------|-----------|
| **Emotional Tone** | 23 | 95.2% (GBT) | 0.90 (TextCNN) |
| **Romantic Intent** | 25 | 93.5% (SVM) | 0.89 (TextCNN) |
| **Conversation Stage** | 18 | 97.1% (GBT) | 0.91 (TextCNN) |

**Model architectures:**
- **TextCNN** — Multi-kernel 1D convolutions over MiniLM-L6-v2 embeddings (384-dim)
- **EmotionAttentionNet** — Multi-head self-attention with emotion-aware feature extraction
- **sklearn classifiers** — SVM, Gradient Boosting, Random Forest, Logistic Regression with cross-validation

### Reinforcement Learning

Thompson sampling contextual bandits with 8 response strategies per user:
- Beta distribution sampling for exploration/exploitation
- 7-signal reward computation (response received, speed, length, emotional valence, engagement, emoji sentiment, continuation)
- Configurable decay rate for adapting to behavioral changes

### Autoresearch (Autonomous Model Improvement)

Karpathy-inspired autonomous experimentation framework that continuously improves the entire system:

| Experiment Type | Weight | What It Optimizes |
|----------------|--------|-------------------|
| **Neural** | 30% | TextCNN / EmotionAttentionNet hyperparameters, architecture, optimizer, scheduler |
| **sklearn** | 25% | Classifier type, regularization, kernel, ensemble size across 3 tasks |
| **RL Parameters** | 20% | Reward signal weights, Thompson sampling decay, match bonus |
| **Engine Parameters** | 15% | 27 tunable thresholds across all engines (context, emotion, style, memory, NLP, temperature) |
| **Voice** | 10% | TTS parameters (cfg_weight, exaggeration, temperature, repetition_penalty) |

Winning configurations are automatically promoted to production. All 7 engine files auto-load optimized parameters via file-mtime caching — no restart required.

```bash
# Run 100 autonomous experiments (leave overnight)
nohup uv run python -m autoresearch.run_experiment --n 100 > autoresearch.log 2>&1 &

# Monitor progress
tail -f autoresearch.log | grep -E "Experiment|NEW BEST|did not beat"

# View results
uv run python -m autoresearch.run_experiment --results
```

## Psychological Frameworks

15+ research-backed frameworks integrated across the intelligence engines:

| Framework | Source | Application |
|-----------|--------|-------------|
| **Gottman's Four Horsemen** | 40+ years of couples research | Detects criticism, contempt, defensiveness, stonewalling |
| **Gottman's 5:1 Ratio** | Relationship research | Monitors positive-to-negative interaction balance |
| **Attachment Theory** | Bowlby/Ainsworth | Identifies secure, anxious, avoidant, fearful-avoidant patterns |
| **Five Love Languages** | Gary Chapman | Detects preferred communication modes |
| **Plutchik's Wheel** | Emotion research | 8 primary + 24 combined emotions |
| **GoEmotions** | Google Research | 27-category fine-grained emotion taxonomy |
| **Big Five (OCEAN)** | Pennebaker's LIWC | Personality traits from linguistic markers |
| **CBT Cognitive Distortions** | Aaron Beck | 13 distortion types with reframe strategies |
| **Nonviolent Communication** | Marshall Rosenberg | Observation, feeling, need, request scoring |
| **Thomas-Kilmann** | Conflict research | 5 conflict modes |
| **ESConv Framework** | Helping Skills Theory | 3-stage empathetic response |
| **Digital Body Language** | Communication research | Response time shifts, emoji changes, punctuation signals |
| **Knapp's Relational Model** | Communication theory | 10 relationship stages |
| **Chain of Empathy** | arXiv:2311.04915 | 4-step empathetic reasoning pipeline |

## Voice Engine

4-backend voice synthesis with automatic selection:

| Backend | Language | Features |
|---------|----------|----------|
| **Chatterbox** | EN, RU, Multilingual | Zero-shot voice cloning, emotion control |
| **F5-TTS** | EN (Apple Silicon) | MLX-optimized, fast inference |
| **Bark** | Multilingual | Emotional speech synthesis |
| **Edge-TTS** | 60+ languages | Microsoft Azure, low latency |

Additional capabilities:
- Zero-shot voice cloning from 6-second audio samples
- Per-user voice persona storage
- Real-time voice calls via tgcalls (WebRTC)
- Whisper-based speech-to-text transcription

## Quick Start

### Prerequisites

- Python 3.10-3.12 (3.13 not supported — no PyTorch wheels)
- [uv](https://docs.astral.sh/uv/) package manager
- Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)

### Installation

```bash
git clone https://github.com/abailey81/telegram-ai-agent.git
cd telegram-ai-agent

# Pin Python version and install dependencies
uv python pin 3.12
uv sync
```

### Configuration

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION_STRING=your_session_string
```

Generate a session string:
```bash
uv run python session_string_generator.py
```

### Running

**Terminal 1 — API Bridge + Dashboard:**
```bash
uv run python telegram_api.py
```

**Terminal 2 — CLI Agent:**
```bash
cd agent && bun install && bun run dev
```

**Alternative — MCP Server (for Claude Desktop):**
```bash
uv run main.py
```

### Training Models

```bash
# Train all classifiers and neural networks
uv run python train_all.py

# Check model status
uv run python train_all.py --status

# Run autonomous improvement
uv run python -m autoresearch.run_experiment --n 100
```

## Project Structure

```
telegram-ai-agent/
├── main.py                         # MCP server (90+ tools)
├── telegram_api.py                 # FastAPI HTTP bridge (port 8765)
├── orchestrator.py                 # Multi-engine orchestration
│
├── nlp_engine.py                   # NLP engine
├── advanced_intelligence.py        # Advanced emotion detection
├── conversation_engine.py          # Context assembly + state machine
├── emotional_intelligence.py       # VAD profiling + attachment
├── style_engine.py                 # Communication style analysis
├── memory_engine.py                # Three-tier memory + FAISS
├── reasoning_engine.py             # Chain-of-thought reasoning
├── personality_engine.py           # Personality trait detection
├── prediction_engine.py            # Intent prediction
├── rl_engine.py                    # Thompson sampling RL
├── voice_engine.py                 # 4-backend TTS + cloning
├── media_intelligence.py           # Media classification
├── media_ai.py                     # Whisper + image understanding
│
├── neural_networks.py              # TextCNN + EmotionAttentionNet
├── dl_models.py                    # PyTorch model utilities
├── train_all.py                    # Training orchestrator
├── training/                       # Training data (6,600+ examples)
│   ├── training_data.py
│   ├── expanded_data.py
│   └── real_conversations_data.py
│
├── autoresearch/                   # Autonomous improvement framework
│   ├── config.py                   # Search spaces + budgets
│   ├── run_experiment.py           # Experiment loop (5 types)
│   ├── train.py                    # Neural network experiments
│   ├── train_sklearn.py            # sklearn experiments
│   ├── optimize_rl.py              # RL parameter optimization
│   ├── optimize_engines.py         # Engine threshold optimization
│   ├── evaluate.py                 # Composite scoring
│   ├── harvest.py                  # Conversation data harvesting
│   └── program.md                  # Research objectives
│
├── agent/                          # TypeScript CLI agent
│   ├── src/agent.ts                # Claude Sonnet integration
│   └── src/tools/                  # Tool definitions
│
├── dashboard/                      # Reflex web dashboard
├── pyproject.toml                  # Dependencies + config
├── Dockerfile                      # Container deployment
└── docker-compose.yml              # Docker Compose config
```

## API Reference

### MCP Tools (90+)

The MCP server exposes tools across these categories:

| Category | Tools | Examples |
|----------|-------|---------|
| **Messaging** | 15+ | `sendMessage`, `editMessage`, `deleteMessage`, `forwardMessage` |
| **Chat Management** | 12+ | `getChats`, `getChatHistory`, `searchMessages`, `pinMessage` |
| **Contacts** | 8+ | `getContacts`, `searchContacts`, `addContact` |
| **Media** | 10+ | `sendVoiceMessage`, `sendPhoto`, `downloadMedia` |
| **Intelligence** | 20+ | `analyzeConversation`, `getEmotionalProfile`, `getMemory` |
| **Auto-Reply** | 10+ | `enableAutoReply`, `setPersonality`, `giveInstruction` |
| **Voice** | 8+ | `generateVoice`, `cloneVoice`, `startCall` |
| **Admin** | 7+ | `banUser`, `setPermissions`, `getChatAdmins` |

### REST API (80+ endpoints)

All MCP tools are also available via REST at `http://localhost:8765`:

```bash
# Send a message
curl -X POST http://localhost:8765/send-message \
  -H "Content-Type: application/json" \
  -d '{"chat_id": 123456789, "text": "Hello"}'

# Get conversation analysis
curl http://localhost:8765/analyze/123456789

# Get emotional profile
curl http://localhost:8765/emotional-profile/123456789
```

## Docker

```bash
docker-compose up -d
```

```yaml
# docker-compose.yml
services:
  telegram-ai-agent:
    build: .
    stdin_open: true
    env_file: .env
    ports:
      - "8765:8765"
```

## Configuration

### Engine Parameters

27 tunable parameters across all engines, automatically optimized by autoresearch:

| Engine | Parameters | Examples |
|--------|-----------|---------|
| **Conversation** | 4 | `recency_decay`, `max_messages`, `state_confidence_threshold` |
| **Emotional Intelligence** | 6 | `baseline_valence_low/high`, `intensity_floor/scale`, attachment weights |
| **Style** | 5 | `emoji_density_high/low`, `formality_thresholds`, `humor_frequency` |
| **Memory** | 5 | `max_facts`, `max_episodes`, `semantic_similarity_boost` |
| **NLP Scoring** | 3 | `staleness_threshold`, `repetition_penalty`, `ai_detection_penalty` |
| **Orchestrator** | 4 | `base_temperature`, `conflict_temperature`, `creative_temperature` |

All parameters are auto-loaded from `engine_data/optimized_engine_params.json` via file-mtime caching.

## Development

```bash
# Format code
black .

# Lint
flake8 .

# Run tests
pytest test_validation.py -v
```

## License

[Apache License 2.0](LICENSE)
