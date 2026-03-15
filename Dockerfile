# Use Python 3.12 (required for PyTorch compatibility)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure Python output is sent straight to terminal (useful for logs)
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY telegram_api.py .
COPY nlp_engine.py .
COPY dl_models.py .
COPY advanced_nlp.py .
COPY neural_networks.py .
COPY conversation_engine.py .
COPY emotional_intelligence.py .
COPY style_engine.py .
COPY memory_engine.py .
COPY reasoning_engine.py .
COPY psychological_datasets.py .
COPY media_intelligence.py .
COPY advanced_intelligence.py .
COPY media_ai.py .
COPY rl_engine.py .
COPY startup_dashboard.py .
COPY session_string_generator.py .
COPY training/ training/

# Create data directories
RUN mkdir -p engine_data/emotional engine_data/styles engine_data/memory \
    engine_data/summaries engine_data/profiles engine_data/goals \
    engine_data/trajectory .chat_memory trained_models/neural .model_cache

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables (provide at runtime)
ENV TELEGRAM_API_ID=""
ENV TELEGRAM_API_HASH=""
ENV TELEGRAM_SESSION_NAME="telegram_session"
ENV TELEGRAM_SESSION_STRING=""

# Expose the API bridge port
EXPOSE 8765

# Default: run the API bridge (use CMD override for MCP server)
CMD ["python", "telegram_api.py"]
