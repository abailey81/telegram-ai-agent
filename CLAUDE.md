# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telegram AI Agent — a production-grade conversational AI platform with 13 intelligence engines, ML/DL classifiers, reinforcement learning, voice synthesis, and 90+ tools. Uses Telethon for Telegram API, PyTorch + scikit-learn for ML models, and exposes tools via MCP and a FastAPI HTTP bridge. Requires Python 3.10-3.12 (not 3.13 — PyTorch incompatible).

## Commands

### Setup
```bash
uv python pin 3.12    # Required: pin Python to 3.12
uv sync               # Install all dependencies
```

### Terminal 1: Telegram API Bridge (Python)
```bash
uv run python telegram_api.py   # FastAPI on port 8765 + Rich dashboard
```

### Terminal 2: CLI Agent (TypeScript)
```bash
cd agent
bun install
bun run dev
```

### MCP Server (alternative usage)
```bash
uv run main.py        # Run as MCP server (90+ tools)
```

### Training
```bash
uv run python train_all.py          # Train all models
uv run python train_all.py --status # Check model status
```

### Autoresearch (autonomous improvement)
```bash
uv run python -m autoresearch.run_experiment --n 100   # Run 100 experiments
uv run python -m autoresearch.run_experiment --results  # View results
nohup uv run python -m autoresearch.run_experiment --n 100 > autoresearch.log 2>&1 &  # Background
```

### Other
```bash
uv run session_string_generator.py  # Generate Telegram session string
black .                              # Format code
flake8 .                             # Lint code
pytest test_validation.py -v         # Run tests
```

## Architecture

### Intelligence Engines (13)
1. `nlp_engine.py` — Sentiment, topic extraction, language detection
2. `advanced_intelligence.py` — GoEmotions, subtext analysis, persona consistency
3. `conversation_engine.py` — Context assembly, dialogue acts, 13-state machine
4. `emotional_intelligence.py` — VAD profiling, attachment style detection
5. `style_engine.py` — Big Five traits, communication style mirroring
6. `memory_engine.py` — Semantic/episodic/procedural memory with FAISS
7. `reasoning_engine.py` — Chain-of-thought, empathy chains, model cascade
8. `personality_engine.py` — Trait detection, consistency scoring
9. `prediction_engine.py` — Intent prediction, behavioral forecasting
10. `rl_engine.py` — Thompson sampling contextual bandits
11. `voice_engine.py` — TTS (4 backends), voice cloning, speaker ID
12. `media_intelligence.py` — Voice/text classification, multilingual embeddings
13. `media_ai.py` — Voice transcription (Whisper), image understanding, TTS

### MCP Server (`main.py`)
Single-file MCP server exposing Telegram functionality as tools:
- `FastMCP("telegram")` — MCP server instance
- `TelegramClient` — Telethon client (supports string and file-based sessions)
- `@mcp.tool()` decorated functions — 90+ tools
- `@validate_id()` decorator — validates chat_id/user_id parameters

### TypeScript Agent (`agent/`)
CLI agent using Claude Sonnet via Vercel AI Gateway:
- `telegram_api.py` — FastAPI HTTP bridge on port 8765
- `agent/src/agent.ts` — AI agent using @ai-sdk/gateway
- `agent/src/tools/` — Tool definitions

### Autoresearch (`autoresearch/`)
Karpathy-style autonomous experimentation framework:
- 5 experiment types: neural, sklearn, rl_params, engine_params, voice
- Auto-promotes winning configs to production
- All 7 engine files auto-load optimized params without restart

### ID Validation
All functions accept flexible ID formats:
- Integer: `123456789` or `-1001234567890`
- String: `"123456789"`
- Username: `"@username"` or `"username"`

## Code Style

- Python: Black formatter (line-length 99), Flake8 linter
- Target Python version: 3.10-3.12 (3.13 NOT supported — no PyTorch wheels)
- Use `Union[int, str]` for ID parameters
- Error codes follow pattern: `{CATEGORY}-ERR-{hash}`

## Environment Variables

Required in `.env`:
- `TELEGRAM_API_ID` — from my.telegram.org/apps
- `TELEGRAM_API_HASH` — from my.telegram.org/apps
- `TELEGRAM_SESSION_STRING` (preferred) or `TELEGRAM_SESSION_NAME` (file-based)

For TypeScript agent:
- `AI_GATEWAY_API_KEY` — Vercel AI Gateway key
- `TELEGRAM_API_URL` — HTTP bridge URL (default: http://localhost:8765)
