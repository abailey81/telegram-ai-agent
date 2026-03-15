<div align="center">

# Telegram AI Agent

### Production-Grade Conversational Intelligence Platform

*13 intelligence engines · 6 deep learning models · reinforcement learning · voice synthesis · autonomous improvement*

<br>

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP_Protocol-7C3AED?style=for-the-badge)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/Apache_2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

<br>

[![Lint & Format](https://github.com/abailey81/telegram-ai-agent/actions/workflows/python-lint-format.yml/badge.svg)](https://github.com/abailey81/telegram-ai-agent/actions/workflows/python-lint-format.yml)
[![Docker Build](https://github.com/abailey81/telegram-ai-agent/actions/workflows/docker-build.yml/badge.svg)](https://github.com/abailey81/telegram-ai-agent/actions/workflows/docker-build.yml)
[![GitHub stars](https://img.shields.io/github/stars/abailey81/telegram-ai-agent?style=social)](https://github.com/abailey81/telegram-ai-agent/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abailey81/telegram-ai-agent?style=social)](https://github.com/abailey81/telegram-ai-agent/network/members)

---

**A unified conversational AI system** that combines NLP, emotion detection, personality profiling,<br>
reinforcement learning, and voice synthesis into a single Telegram integration.<br>
Powered by 13 specialized intelligence engines, 6 deep learning models,<br>
and an autonomous improvement framework inspired by Karpathy's autoresearch.

<br>

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [Intelligence Engines](#-intelligence-engines) · [ML Pipeline](#-ml--deep-learning-pipeline) · [API Reference](#-api-reference) · [Contributing](CONTRIBUTING.md)

</div>

<br>

## Highlights

<table>
<tr>
<td width="50%">

**Conversational Intelligence**
- 13 specialized engines working in concert
- 15+ psychological frameworks (Gottman, Plutchik, Big Five, CBT, NVC)
- 13-state conversation machine with goal tracking
- Three-tier memory: semantic, episodic, procedural (FAISS)

</td>
<td width="50%">

**Machine Learning**
- 6 deep learning models (TextCNN + EmotionAttentionNet)
- 66 classification labels across 3 tasks
- Thompson sampling RL with per-user optimization
- Autonomous experiment framework (5 experiment types)

</td>
</tr>
<tr>
<td width="50%">

**Voice & Media**
- 4-backend TTS: Chatterbox, F5-TTS, Bark, Edge-TTS
- Zero-shot voice cloning from 6-second samples
- Whisper speech-to-text transcription
- Real-time voice calls via tgcalls (WebRTC)

</td>
<td width="50%">

**Platform & API**
- 90+ MCP tools for Claude Desktop integration
- 80+ REST endpoints via FastAPI bridge
- Rich terminal dashboard on port 8765
- Docker-ready with CI/CD pipelines

</td>
</tr>
</table>

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │              Telegram Cloud                  │
                         └──────────────────┬───────────────────────────┘
                                            │ Telethon MTProto
                         ┌──────────────────▼───────────────────────────┐
                         │          telegram_api.py (FastAPI)            │
                         │          Port 8765 · REST + WebSocket         │
                         │          90+ endpoints · Rich dashboard       │
                         └───────┬──────────────────────────┬───────────┘
                                 │                          │
                ┌────────────────▼─────────┐   ┌────────────▼────────────┐
                │    main.py (MCP Server)   │   │   agent/ (TypeScript)   │
                │    90+ MCP tools          │   │   Claude CLI agent      │
                │    Claude Desktop          │   │   Anthropic SDK         │
                └────────────────┬─────────┘   └─────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────────┐
              │         Orchestrator (orchestrator.py)    │
              │       Multi-engine cascade · Temperature  │
              │       control · Conflict routing           │
              └──────────┬─────────────────┬─────────────┘
                         │                 │
          ┌──────────────▼───┐  ┌──────────▼────────────────────────────┐
          │   ML Pipeline     │  │   13 Intelligence Engines              │
          │                   │  │                                        │
          │   TextCNN         │  │   NLP · Emotional Intelligence ·      │
          │   EmotionAttnNet  │  │   Conversation · Style · Memory ·     │
          │   SVM / GBT / LR  │  │   Reasoning · Personality ·           │
          │   Thompson RL     │  │   Prediction · Voice · Media ·        │
          │                   │  │   Advanced Intelligence · Media AI    │
          └──────────┬────────┘  └───────────────────────────────────────┘
                     │
          ┌──────────▼────────┐
          │   Autoresearch     │
          │   5 experiment      │
          │   types · auto-     │
          │   promotion ·       │
          │   hot-reload        │
          └────────────────────┘
```

---

## Intelligence Engines

| # | Engine | Module | Capabilities |
|:-:|:-------|:-------|:-------------|
| 1 | **NLP** | `nlp_engine.py` | Sentiment analysis, topic extraction, language detection, passive-aggression detection, sarcasm identification |
| 2 | **Advanced Intelligence** | `advanced_intelligence.py` | GoEmotions 28-label detection, subtext analysis, persona consistency scoring |
| 3 | **Conversation** | `conversation_engine.py` | Weighted context assembly, dialogue act classification, 13-state machine, goal tracking |
| 4 | **Emotional Intelligence** | `emotional_intelligence.py` | VAD profiling, Plutchik's Wheel (32 emotions), attachment style detection |
| 5 | **Style** | `style_engine.py` | Big Five personality traits, communication preferences, digital body language, mirroring |
| 6 | **Memory** | `memory_engine.py` | Three-tier memory (semantic / episodic / procedural) with FAISS vector retrieval |
| 7 | **Reasoning** | `reasoning_engine.py` | Chain-of-thought, chain-of-empathy, conflict mode detection, model cascade |
| 8 | **Personality** | `personality_engine.py` | Trait detection from linguistic markers, consistency scoring |
| 9 | **Prediction** | `prediction_engine.py` | Intent prediction, behavioral forecasting, next-action estimation |
| 10 | **Reinforcement Learning** | `rl_engine.py` | Thompson sampling contextual bandits, per-user strategy optimization |
| 11 | **Voice** | `voice_engine.py` | 4-backend TTS (Chatterbox, F5-TTS, Bark, Edge-TTS), zero-shot voice cloning |
| 12 | **Media Intelligence** | `media_intelligence.py` | Voice/text classification, multilingual embeddings (bge-m3) |
| 13 | **Media AI** | `media_ai.py` | Voice transcription (faster-whisper), image understanding, speech synthesis |

---

## ML / Deep Learning Pipeline

### Classification Models

Three classification tasks trained on **5,000–6,600 labeled examples** each:

| Task | Classes | sklearn Accuracy | Neural F1 |
|:-----|:-------:|:----------------:|:---------:|
| **Emotional Tone** | 23 | 95.2% (GBT) | 0.90 (TextCNN) |
| **Conversational Intent** | 25 | 93.5% (SVM) | 0.89 (TextCNN) |
| **Conversation Stage** | 18 | 97.1% (GBT) | 0.91 (TextCNN) |

<details>
<summary><b>Model Architecture Details</b></summary>

<br>

**TextCNN** — Multi-kernel 1D convolutions (kernels 2, 3, 4, 5) over MiniLM-L6-v2 embeddings (384-dim). Each kernel captures n-gram patterns at different scales, outputs concatenated and passed through dropout + fully connected layers.

**EmotionAttentionNet** — Multi-head self-attention with emotion-aware feature extraction. Learns to attend to emotionally salient tokens in the embedding sequence, producing context-weighted representations for classification.

**sklearn Ensemble** — SVM, Gradient Boosting, Random Forest, and Logistic Regression trained with 5-fold stratified cross-validation. Best model per task selected by validation accuracy.

</details>

### Reinforcement Learning

Thompson sampling contextual bandits with **8 response strategies** per user:

- Beta distribution sampling for exploration/exploitation balance
- 7-signal reward computation (response received, speed, length, emotional valence, engagement, emoji sentiment, continuation)
- Configurable decay rate for adapting to behavioral shifts

### Autoresearch — Autonomous Model Improvement

Karpathy-inspired autonomous experimentation framework that continuously improves the entire system:

| Experiment Type | Weight | What It Optimizes |
|:---------------|:------:|:------------------|
| **Neural** | 30% | TextCNN / EmotionAttentionNet hyperparameters, architecture, optimizer, scheduler |
| **sklearn** | 25% | Classifier type, regularization, kernel, ensemble size across 3 tasks |
| **RL Parameters** | 20% | Reward signal weights, Thompson sampling decay, match bonus |
| **Engine Parameters** | 15% | 27 tunable thresholds across all engines (context, emotion, style, memory, NLP, temperature) |
| **Voice** | 10% | TTS parameters (cfg_weight, exaggeration, temperature, repetition_penalty) |

Winning configurations are **automatically promoted to production**. All 7 engine files auto-load optimized parameters via file-mtime caching — **no restart required**.

```bash
# Run 100 autonomous experiments
nohup uv run python -m autoresearch.run_experiment --n 100 > autoresearch.log 2>&1 &

# Monitor progress
tail -f autoresearch.log | grep -E "Experiment|NEW BEST|did not beat"

# View results
uv run python -m autoresearch.run_experiment --results
```

---

## Psychological Frameworks

<details>
<summary><b>15+ research-backed frameworks integrated across the intelligence engines</b></summary>

<br>

| Framework | Source | Application |
|:----------|:-------|:------------|
| **Gottman's Four Horsemen** | 40+ years of research | Detects criticism, contempt, defensiveness, stonewalling |
| **Gottman's 5:1 Ratio** | Interaction research | Monitors positive-to-negative interaction balance |
| **Attachment Theory** | Bowlby/Ainsworth | Identifies secure, anxious, avoidant, fearful-avoidant patterns |
| **Five Communication Preferences** | Gary Chapman | Detects preferred communication modes |
| **Plutchik's Wheel** | Emotion research | 8 primary + 24 combined emotions |
| **GoEmotions** | Google Research | 27-category fine-grained emotion taxonomy |
| **Big Five (OCEAN)** | Pennebaker's LIWC | Personality traits from linguistic markers |
| **CBT Cognitive Distortions** | Aaron Beck | 13 distortion types with reframe strategies |
| **Nonviolent Communication** | Marshall Rosenberg | Observation, feeling, need, request scoring |
| **Thomas-Kilmann** | Conflict research | 5 conflict modes (competing, collaborating, compromising, avoiding, accommodating) |
| **ESConv Framework** | Helping Skills Theory | 3-stage empathetic response pipeline |
| **Digital Body Language** | Communication research | Response time shifts, emoji changes, punctuation signals |
| **Knapp's Relational Model** | Communication theory | 10 interaction stages |
| **Chain of Empathy** | arXiv:2311.04915 | 4-step empathetic reasoning pipeline |

</details>

---

## Voice Engine

4-backend voice synthesis with automatic selection and zero-shot cloning:

| Backend | Language | Features |
|:--------|:---------|:---------|
| **Chatterbox** | EN, RU, Multilingual | Zero-shot voice cloning, emotion control |
| **F5-TTS** | EN (Apple Silicon) | MLX-optimized, fast inference |
| **Bark** | Multilingual | Emotional speech synthesis |
| **Edge-TTS** | 60+ languages | Microsoft Azure, low latency |

<details>
<summary><b>Additional Voice Capabilities</b></summary>

<br>

- Zero-shot voice cloning from 6-second audio samples
- Per-user voice persona storage and selection
- Real-time voice calls via tgcalls (WebRTC)
- Whisper-based speech-to-text transcription (faster-whisper)
- Automatic language detection and backend selection
- Emotion-controlled synthesis parameters

</details>

---

## Getting Started

### Prerequisites

- Python 3.10–3.12 (3.13 not supported — no PyTorch wheels)
- [uv](https://docs.astral.sh/uv/) package manager
- Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)

### Quick Start

```bash
# Clone and install
git clone https://github.com/abailey81/telegram-ai-agent.git
cd telegram-ai-agent
uv python pin 3.12
uv sync

# Configure
cp .env.example .env
# Edit .env with your Telegram credentials

# Generate session string
uv run python session_string_generator.py

# Launch
uv run python telegram_api.py
```

### Running Modes

| Mode | Command | Description |
|:-----|:--------|:------------|
| **API Bridge** | `uv run python telegram_api.py` | FastAPI server + Rich dashboard on port 8765 |
| **CLI Agent** | `cd agent && bun install && bun run dev` | Interactive TypeScript CLI powered by Claude |
| **MCP Server** | `uv run main.py` | Model Context Protocol server for Claude Desktop |
| **Docker** | `docker compose up -d` | Containerized deployment |

### Training Models

```bash
# Train all classifiers and neural networks
uv run python train_all.py

# Check model status
uv run python train_all.py --status

# Run autonomous improvement (100 experiments)
uv run python -m autoresearch.run_experiment --n 100
```

---

## Project Structure

```
telegram-ai-agent/
│
├── main.py                          # MCP server (90+ tools)
├── telegram_api.py                  # FastAPI HTTP bridge (port 8765)
├── orchestrator.py                  # Multi-engine orchestration
│
│── Intelligence Engines ────────────────────────────────────────────
├── nlp_engine.py                    # NLP: sentiment, topics, language
├── advanced_intelligence.py         # GoEmotions, subtext, persona
├── conversation_engine.py           # Context assembly, state machine
├── emotional_intelligence.py        # VAD profiling, attachment styles
├── style_engine.py                  # Big Five, communication analysis
├── memory_engine.py                 # Three-tier memory + FAISS
├── reasoning_engine.py              # Chain-of-thought reasoning
├── personality_engine.py            # Trait detection, consistency
├── prediction_engine.py             # Intent & behavioral prediction
├── rl_engine.py                     # Thompson sampling RL
├── voice_engine.py                  # 4-backend TTS + voice cloning
├── media_intelligence.py            # Media classification, embeddings
├── media_ai.py                      # Whisper, image understanding
│
│── ML Pipeline ─────────────────────────────────────────────────────
├── neural_networks.py               # TextCNN + EmotionAttentionNet
├── dl_models.py                     # PyTorch model utilities
├── train_all.py                     # Training orchestrator
├── training/                        # Training data (6,600+ examples)
│   ├── training_data.py             # Core labeled dataset
│   ├── expanded_data.py             # Augmented training data
│   └── real_conversations_data.py   # Real conversation examples
│
│── Autoresearch ────────────────────────────────────────────────────
├── autoresearch/
│   ├── config.py                    # Search spaces, budgets, paths
│   ├── run_experiment.py            # Experiment loop (5 types)
│   ├── train.py                     # Neural network experiments
│   ├── train_sklearn.py             # sklearn experiments
│   ├── optimize_rl.py               # RL parameter optimization
│   ├── optimize_engines.py          # Engine threshold optimization
│   ├── evaluate.py                  # Composite scoring
│   ├── harvest.py                   # Conversation data harvesting
│   └── program.md                   # Research objectives
│
│── CLI Agent ───────────────────────────────────────────────────────
├── agent/
│   ├── src/agent.ts                 # Claude integration + system prompt
│   ├── src/index.ts                 # Interactive CLI with rich UI
│   └── src/tools/                   # Tool definitions (telegram, nia, aiify)
│
│── Infrastructure ──────────────────────────────────────────────────
├── dashboard/                       # Reflex web dashboard
├── pyproject.toml                   # Dependencies + configuration
├── Dockerfile                       # Container deployment
├── docker-compose.yml               # Docker Compose config
└── .github/workflows/               # CI: lint, format, Docker build
```

---

## API Reference

### MCP Tools (90+)

| Category | Count | Examples |
|:---------|:-----:|:--------|
| **Messaging** | 15+ | `sendMessage`, `editMessage`, `deleteMessage`, `forwardMessage` |
| **Chat Management** | 12+ | `getChats`, `getChatHistory`, `searchMessages`, `pinMessage` |
| **Contacts** | 8+ | `getContacts`, `searchContacts`, `addContact` |
| **Media** | 10+ | `sendVoiceMessage`, `sendPhoto`, `downloadMedia` |
| **Intelligence** | 20+ | `analyzeConversation`, `getEmotionalProfile`, `getMemory` |
| **Auto-Reply** | 10+ | `enableAutoReply`, `setInstructions`, `intervene` |
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

---

## Configuration

<details>
<summary><b>27 tunable engine parameters (auto-optimized by autoresearch)</b></summary>

<br>

| Engine | Parameters | Examples |
|:-------|:---------:|:--------|
| **Conversation** | 4 | `recency_decay`, `max_messages`, `state_confidence_threshold` |
| **Emotional Intelligence** | 6 | `baseline_valence_low/high`, `intensity_floor/scale`, attachment weights |
| **Style** | 5 | `emoji_density_high/low`, `formality_thresholds`, `humor_frequency` |
| **Memory** | 5 | `max_facts`, `max_episodes`, `semantic_similarity_boost` |
| **NLP Scoring** | 3 | `staleness_threshold`, `repetition_penalty`, `ai_detection_penalty` |
| **Orchestrator** | 4 | `base_temperature`, `conflict_temperature`, `creative_temperature` |

All parameters auto-loaded from `engine_data/optimized_engine_params.json` via file-mtime caching.

</details>

---

## Docker

```bash
# Build and run
docker compose up -d

# Build only
docker build -t telegram-ai-agent .
```

<details>
<summary><b>Docker Compose configuration</b></summary>

```yaml
services:
  telegram-ai-agent:
    build: .
    env_file: .env
    ports:
      - "8765:8765"
    restart: unless-stopped
```

</details>

---

## Development

```bash
# Format
black .

# Lint
flake8 .

# Test
pytest test_validation.py -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and [SECURITY.md](SECURITY.md) for security policy.

---

<div align="center">

**[Apache License 2.0](LICENSE)**

Built with PyTorch, FastAPI, Telethon, and the Model Context Protocol

</div>
