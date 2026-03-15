# Contributing to Telegram AI Agent

Thank you for your interest in contributing. This document provides guidelines and information for contributors.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/abailey81/telegram-ai-agent.git
cd telegram-ai-agent

# Pin Python version (required: 3.10-3.12, NOT 3.13)
uv python pin 3.12

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your Telegram API credentials
```

## Code Style

- **Formatter**: [Black](https://black.readthedocs.io/) (line length: 99)
- **Linter**: [Flake8](https://flake8.pycqa.org/)
- **Python**: 3.10–3.12 (3.13 not supported — no PyTorch wheels)

```bash
# Format code
black .

# Check linting
flake8 .
```

## Project Structure

| Directory | Purpose |
|:----------|:--------|
| Root `.py` files | Intelligence engines, API bridge, MCP server |
| `training/` | Training data and data loaders |
| `autoresearch/` | Autonomous model improvement framework |
| `agent/` | TypeScript CLI agent |
| `dashboard/` | Reflex web dashboard |
| `.github/workflows/` | CI/CD pipelines |

## Making Changes

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature`
3. **Make your changes** following the code style guidelines
4. **Test your changes**:
   ```bash
   pytest test_validation.py -v
   black --check .
   flake8 .
   ```
5. **Commit** with a clear, descriptive message
6. **Push** to your fork and open a **Pull Request**

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Fill out the PR template completely
- Ensure CI checks pass before requesting review
- Include relevant test coverage for new functionality

## Intelligence Engine Guidelines

When modifying or adding intelligence engines:

- Follow the existing engine pattern (class-based with `analyze()` / `process()` methods)
- Engines must be stateless per-request (user state stored in `engine_data/`)
- Use `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for embeddings — do not change the embedding model
- All tunable parameters should be exposed in the engine's config and registered with autoresearch

## ML Pipeline Guidelines

- Training data goes in `training/training_data.py` (or `expanded_data.py` / `real_conversations_data.py`)
- Neural architectures are defined in `neural_networks.py`
- Model metadata JSONs (`*_meta.json`) are committed; trained weights (`.joblib`, `.pt`) are not
- Always run `train_all.py --status` after training changes to verify model health

## Reporting Issues

Use the [issue templates](https://github.com/abailey81/telegram-ai-agent/issues/new/choose) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
