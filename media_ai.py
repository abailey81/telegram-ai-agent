"""
Media AI Engine — Voice transcription, image understanding, voice response.

Implements critical media processing capabilities identified from research:
1. Voice Transcription (faster-whisper / whisper) — transcribe incoming voice messages
2. Image Understanding (Claude Vision API) — understand photos/images sent in chat
3. Voice Response (Edge TTS) — generate voice notes for outgoing replies

All functions are designed to be called from telegram_api.py's pipeline.
Graceful degradation: falls back cleanly if dependencies aren't installed.
"""

import os
# Prevent OpenMP crash when torch + faster-whisper/onnxruntime coexist
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import re
import sys
import json
import asyncio
import logging
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

media_ai_logger = logging.getLogger("media_ai")
media_ai_logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════
#  GPU / DEVICE DETECTION — use MPS (Apple Silicon) or CUDA where possible
# ═══════════════════════════════════════════════════════════════

_device = None
_device_name = None


def _detect_device() -> str:
    """Detect best available compute device: cuda > mps > cpu."""
    global _device, _device_name
    if _device is not None:
        return _device
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            _device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
            _device_name = "Apple Silicon (MPS)"
        else:
            _device = "cpu"
            _device_name = "CPU"
    except ImportError:
        _device = "cpu"
        _device_name = "CPU (torch not installed)"
    media_ai_logger.info(f"Compute device: {_device} ({_device_name})")
    return _device


def _get_torch_device_index() -> int:
    """Get device index for HuggingFace pipeline (-1=CPU, 0=GPU)."""
    device = _detect_device()
    if device in ("cuda", "mps"):
        return 0
    return -1


# ═══════════════════════════════════════════════════════════════
#  1. VOICE TRANSCRIPTION (faster-whisper / whisper) — FREE, open-source
# ═══════════════════════════════════════════════════════════════

_whisper_model = None
_whisper_available = None  # None = not checked yet
_whisper_backend = None  # "faster-whisper" or "whisper"


def _load_whisper():
    """Lazy-load whisper model. Tries faster-whisper first, falls back to openai-whisper.
    Uses GPU (CUDA/MPS) where available for faster transcription."""
    global _whisper_model, _whisper_available, _whisper_backend
    if _whisper_available is not None:
        return _whisper_available

    # Prevent OpenMP crash when torch + onnxruntime are both loaded
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = _detect_device()

    # Try faster-whisper first (3-4x faster, lower memory)
    try:
        from faster_whisper import WhisperModel
        # faster-whisper supports cuda; on MPS/CPU use cpu with int8
        fw_device = "cuda" if device == "cuda" else "cpu"
        fw_compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel(
            "small",  # small model: much better accuracy + language detection
            device=fw_device,
            compute_type=fw_compute,
        )
        _whisper_available = True
        _whisper_backend = "faster-whisper"
        media_ai_logger.info(f"faster-whisper loaded (small, {fw_compute}, device={fw_device})")
        return True
    except ImportError:
        pass
    except Exception as e:
        media_ai_logger.warning(f"faster-whisper failed to load: {e}")

    # Fall back to openai-whisper (also FREE, uses GPU automatically)
    try:
        import whisper
        _whisper_model = whisper.load_model("small", device=device if device != "mps" else "cpu")
        _whisper_available = True
        _whisper_backend = "whisper"
        media_ai_logger.info(f"openai-whisper loaded (base, device={device})")
        return True
    except ImportError:
        pass
    except Exception as e:
        media_ai_logger.warning(f"openai-whisper failed to load: {e}")

    _whisper_available = False
    media_ai_logger.warning("No whisper backend available. Install: pip install faster-whisper")
    return False


def _transcribe_in_subprocess(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Run whisper transcription in a separate process to avoid CTranslate2 + PyTorch
    OpenMP conflict (SIGSEGV on macOS x86_64).
    """
    script = Path(__file__).parent / "_whisper_subprocess.py"
    if not script.exists():
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "unavailable",
            "error": "_whisper_subprocess.py not found",
        }

    cmd = [sys.executable, str(script), audio_path]
    if language:
        cmd.append(language)

    try:
        media_ai_logger.info("Running whisper in subprocess (torch isolation mode)")
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "4"},
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            media_ai_logger.error(f"Whisper subprocess failed (exit {proc.returncode}): {stderr}")
            return {
                "text": "",
                "language": "unknown",
                "confidence": 0.0,
                "duration_seconds": 0,
                "backend": "subprocess-error",
                "error": f"Subprocess exit {proc.returncode}: {stderr[:200]}",
            }

        result = json.loads(proc.stdout.strip())
        elapsed = time.time() - start_time
        if result.get("text"):
            media_ai_logger.info(
                f"Transcribed (subprocess/{result.get('backend', '?')}): "
                f"lang={result.get('language')}, "
                f"{result.get('duration_seconds', 0)}s audio in {elapsed:.1f}s, "
                f"text='{result['text'][:60]}...'"
            )
        return result

    except subprocess.TimeoutExpired:
        media_ai_logger.error("Whisper subprocess timed out (120s)")
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "subprocess-timeout",
            "error": "Transcription timed out after 120s",
        }
    except json.JSONDecodeError as e:
        media_ai_logger.error(f"Whisper subprocess returned invalid JSON: {e}")
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "subprocess-error",
            "error": f"Invalid JSON from subprocess: {e}",
        }
    except Exception as e:
        media_ai_logger.error(f"Whisper subprocess error: {e}")
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "subprocess-error",
            "error": str(e),
        }


def transcribe_voice(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe a voice message audio file to text.

    If torch is already loaded in this process, delegates to a subprocess
    to avoid the CTranslate2/PyTorch OpenMP SIGSEGV crash on macOS.

    Args:
        audio_path: Path to audio file (ogg, wav, mp3, etc.)
        language: Optional language hint ("en", "ru", etc.). Auto-detected if None.

    Returns:
        Dict with keys: text, language, confidence, duration_seconds, backend
    """
    # On macOS, ALWAYS use subprocess — CTranslate2 + PyTorch OpenMP conflict
    # causes SIGSEGV even if torch is loaded indirectly via _detect_device().
    # On Linux/Windows, only use subprocess if torch is already loaded.
    _is_macos = platform.system() == "Darwin"
    if _is_macos or "torch" in sys.modules:
        reason = "macOS (CTranslate2 safety)" if _is_macos else "torch in process"
        media_ai_logger.info(f"Using subprocess for whisper ({reason})")
        return _transcribe_in_subprocess(audio_path, language)

    # Non-macOS, no torch loaded — safe to run in-process
    if not _load_whisper():
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "unavailable",
            "error": "No whisper backend installed",
        }

    try:
        start_time = time.time()

        if _whisper_backend == "faster-whisper":
            segments, info = _whisper_model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=3,
                vad_filter=True,  # Voice Activity Detection — skip silence
            )
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            text = " ".join(text_parts)
            detected_lang = info.language
            confidence = info.language_probability
            duration = info.duration

        else:  # openai-whisper
            options = {}
            if language:
                options["language"] = language
            result = _whisper_model.transcribe(audio_path, **options)
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            confidence = 0.85  # whisper doesn't return confidence directly
            # Estimate duration from segments
            segs = result.get("segments", [])
            duration = segs[-1]["end"] if segs else 0

        elapsed = time.time() - start_time
        media_ai_logger.info(
            f"Transcribed ({_whisper_backend}): "
            f"lang={detected_lang}, {duration:.1f}s audio in {elapsed:.1f}s, "
            f"text='{text[:60]}...'"
        )

        return {
            "text": text,
            "language": detected_lang,
            "confidence": confidence,
            "duration_seconds": round(duration, 1),
            "backend": _whisper_backend,
            "processing_time": round(elapsed, 2),
        }

    except Exception as e:
        media_ai_logger.error(f"Transcription failed: {e}")
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": _whisper_backend,
            "error": str(e),
        }


async def transcribe_telegram_voice(
    tg_client, message, language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download and transcribe a Telegram voice message.

    Args:
        tg_client: Telethon TelegramClient
        message: Telethon Message object with voice media
        language: Optional language hint ("en", "ru", "tr", etc.)

    Returns:
        Transcription result dict
    """
    tmp_path = None
    try:
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
            await tg_client.download_media(message, file=tmp_path)

        # Run transcription in a thread to avoid blocking the event loop
        # (subprocess whisper can take 10-30s on first run)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, transcribe_voice, tmp_path, language
        )
        return result

    except Exception as e:
        media_ai_logger.error(f"Telegram voice transcription failed: {e}")
        return {
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "error",
            "error": str(e),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def format_voice_transcription_for_prompt(transcription: Dict[str, Any]) -> str:
    """Format voice transcription result as context for the system prompt."""
    text = transcription.get("text", "")
    if not text:
        duration = transcription.get("duration_seconds", 0)
        return f"[They sent a voice message ({duration}s) but transcription failed]"

    lang = transcription.get("language", "unknown")
    duration = transcription.get("duration_seconds", 0)
    confidence = transcription.get("confidence", 0)

    result = f"[Voice message ({duration}s, {lang})]"
    result += f'\nThey said: "{text}"'
    if confidence < 0.7:
        result += "\n(Note: transcription confidence is low, some words may be wrong)"

    return result


# ═══════════════════════════════════════════════════════════════
#  2. IMAGE UNDERSTANDING (Claude Vision API)
# ═══════════════════════════════════════════════════════════════

async def understand_image(
    image_path: str,
    caption: str = "",
    context: str = "",
) -> Dict[str, Any]:
    """
    Understand an image using Claude Vision API.

    Args:
        image_path: Path to image file
        caption: Optional caption sent with the image
        context: Optional conversation context

    Returns:
        Dict with: description, objects, mood, suggested_reaction, raw_response
    """
    import httpx
    import base64

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return {
            "description": "",
            "error": "ANTHROPIC_API_KEY not set",
        }

    try:
        # Read and base64 encode the image
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Detect media type
        media_type = "image/jpeg"  # default
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        b64_image = base64.b64encode(image_data).decode("utf-8")

        # Build prompt for natural understanding
        vision_prompt = (
            "You're analyzing a photo that someone sent in a personal Telegram chat "
            "to their partner/close friend. Describe what you see naturally and briefly. "
            "Focus on:\n"
            "1. What's in the photo (selfie, food, place, pet, etc.)\n"
            "2. The mood/vibe of the photo\n"
            "3. How their partner should react (casual, enthusiastic, flirty, etc.)\n\n"
            "Respond as JSON with keys: description, category (selfie/food/place/pet/"
            "meme/screenshot/outfit/other), mood (happy/chill/romantic/funny/sad/neutral), "
            "suggested_reaction (1-2 word reaction style)"
        )
        if caption:
            vision_prompt += f"\n\nCaption they sent with the photo: \"{caption}\""
        if context:
            vision_prompt += f"\n\nConversation context: {context}"

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 300,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": vision_prompt,
                            },
                        ],
                    }],
                },
            )

            if response.status_code != 200:
                media_ai_logger.error(f"Vision API error {response.status_code}: {response.text}")
                return {"description": "", "error": f"API error {response.status_code}"}

            data = response.json()
            raw_text = data["content"][0]["text"].strip()

            # Parse JSON response
            try:
                # Extract JSON from response (may be wrapped in markdown)
                json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = json.loads(raw_text)

                result = {
                    "description": parsed.get("description", raw_text),
                    "category": parsed.get("category", "other"),
                    "mood": parsed.get("mood", "neutral"),
                    "suggested_reaction": parsed.get("suggested_reaction", "casual"),
                    "raw_response": raw_text,
                }
            except (json.JSONDecodeError, ValueError):
                result = {
                    "description": raw_text,
                    "category": "other",
                    "mood": "neutral",
                    "suggested_reaction": "casual",
                    "raw_response": raw_text,
                }

            media_ai_logger.info(
                f"Image understood: category={result['category']}, "
                f"mood={result['mood']}, desc='{result['description'][:60]}...'"
            )
            return result

    except Exception as e:
        media_ai_logger.error(f"Image understanding failed: {e}")
        return {"description": "", "error": str(e)}


async def understand_telegram_image(tg_client, message, context: str = "") -> Dict[str, Any]:
    """
    Download and understand a Telegram photo message.

    Args:
        tg_client: Telethon TelegramClient
        message: Telethon Message with photo media
        context: Conversation context string

    Returns:
        Image understanding result dict
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            await tg_client.download_media(message, file=tmp_path)

        caption = message.message or ""
        result = await understand_image(tmp_path, caption=caption, context=context)
        return result

    except Exception as e:
        media_ai_logger.error(f"Telegram image understanding failed: {e}")
        return {"description": "", "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def format_image_understanding_for_prompt(understanding: Dict[str, Any]) -> str:
    """Format image understanding as context for the system prompt."""
    desc = understanding.get("description", "")
    if not desc:
        return "[They sent a photo but image analysis failed]"

    category = understanding.get("category", "other")
    mood = understanding.get("mood", "neutral")
    reaction = understanding.get("suggested_reaction", "casual")

    result = f"[Photo Analysis — {category}]\n"
    result += f"What you see: {desc}\n"
    result += f"Photo mood: {mood}\n"
    result += f"React style: {reaction}\n"
    result += "IMPORTANT: React naturally to what you see. Don't describe the photo back — "
    result += "respond like a partner who just saw this on their phone."

    return result


# ═══════════════════════════════════════════════════════════════
#  3. VOICE RESPONSE (Edge TTS)
# ═══════════════════════════════════════════════════════════════

_edge_tts_available = None


def _check_edge_tts():
    """Check if edge-tts is available."""
    global _edge_tts_available
    if _edge_tts_available is not None:
        return _edge_tts_available
    try:
        import edge_tts  # noqa: F401
        _edge_tts_available = True
        media_ai_logger.info("edge-tts available")
        return True
    except ImportError:
        _edge_tts_available = False
        media_ai_logger.warning("edge-tts not installed. Install: pip install edge-tts")
        return False


# Voice presets for different languages and genders
VOICE_PRESETS = {
    "en_male": "en-US-GuyNeural",
    "en_female": "en-US-JennyNeural",
    "ru_male": "ru-RU-DmitryNeural",
    "ru_female": "ru-RU-SvetlanaNeural",
    "tr_male": "tr-TR-AhmetNeural",
    "tr_female": "tr-TR-EmelNeural",
}


def detect_text_language(text: str) -> str:
    """Simple language detection based on character analysis."""
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
    turkish_chars = sum(1 for c in text if c in 'ğüşıöçĞÜŞİÖÇ')

    total = cyrillic_count + latin_count + turkish_chars
    if total == 0:
        return "en"

    if cyrillic_count / total > 0.3:
        return "ru"
    if turkish_chars > 0 and turkish_chars / total > 0.05:
        return "tr"
    return "en"


async def generate_voice_response(
    text: str,
    language: Optional[str] = None,
    gender: str = "male",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a voice note from text using Edge TTS.

    Args:
        text: Text to speak
        language: Language code ("en", "ru", "tr"). Auto-detected if None.
        gender: "male" or "female"
        output_path: Optional output file path. Creates temp file if None.

    Returns:
        Path to generated audio file, or None on failure
    """
    if not _check_edge_tts():
        return None

    try:
        import edge_tts

        # Auto-detect language
        if not language:
            language = detect_text_language(text)

        # Select voice
        voice_key = f"{language}_{gender}"
        voice = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["en_male"])

        # Generate output path if not specified
        if not output_path:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = tmp.name
            tmp.close()

        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

        media_ai_logger.info(
            f"Voice generated: voice={voice}, lang={language}, "
            f"text='{text[:40]}...', file={output_path}"
        )
        return output_path

    except Exception as e:
        media_ai_logger.error(f"Voice generation failed: {e}")
        return None


async def send_voice_response(
    tg_client, chat, text: str, language: Optional[str] = None, gender: str = "male"
) -> bool:
    """
    Generate and send a voice note reply.

    Args:
        tg_client: Telethon TelegramClient
        chat: Chat to send to
        text: Text to convert to speech
        language: Language code (auto-detected if None)
        gender: "male" or "female"

    Returns:
        True if sent successfully
    """
    voice_path = None
    try:
        voice_path = await generate_voice_response(text, language, gender)
        if not voice_path:
            return False

        # Send as voice note (Telegram recognizes .ogg files as voice)
        # Edge TTS produces mp3, send as audio file
        await tg_client.send_file(
            chat,
            voice_path,
            voice_note=True,
        )
        media_ai_logger.info(f"Voice response sent to chat")
        return True

    except Exception as e:
        media_ai_logger.error(f"Failed to send voice response: {e}")
        return False
    finally:
        if voice_path and os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
#  4. RUSSIAN NLP MODELS
# ═══════════════════════════════════════════════════════════════

_rubert_sentiment_pipeline = None
_rubert_sentiment_available = None


def _load_rubert_sentiment():
    """Lazy-load Russian sentiment model (FREE, HuggingFace). Uses GPU if available."""
    global _rubert_sentiment_pipeline, _rubert_sentiment_available
    if _rubert_sentiment_available is not None:
        return _rubert_sentiment_available
    try:
        from transformers import pipeline
        _rubert_sentiment_pipeline = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment-rusentiment",
            device=_get_torch_device_index(),
        )
        _rubert_sentiment_available = True
        device = _detect_device()
        media_ai_logger.info(f"rubert sentiment model loaded (device={device})")
        return True
    except Exception as e:
        media_ai_logger.warning(f"rubert sentiment not available: {e}")
        _rubert_sentiment_available = False
        return False


# Russian sentiment heuristic fallback
_RU_POSITIVE = {
    "хорошо", "отлично", "прекрасно", "замечательно", "круто", "класс",
    "супер", "здорово", "мило", "люблю", "обожаю", "рад", "рада",
    "счастлив", "счастлива", "спасибо", "ура", "ладно", "ок",
    "красиво", "молодец", "умница", "скучаю",
}
_RU_NEGATIVE = {
    "плохо", "ужасно", "грустно", "печально", "злюсь", "бесит",
    "ненавижу", "надоел", "надоела", "противно", "тупо", "тупой",
    "дурак", "дура", "идиот", "ужас", "кошмар", "отстой",
    "достал", "достала", "отвали", "уйди", "хватит",
}


def analyze_russian_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of Russian text.

    Returns:
        Dict with: sentiment, confidence, model_used
    """
    if _load_rubert_sentiment():
        try:
            result = _rubert_sentiment_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]

            # Map model labels to standard format
            label_map = {
                "positive": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "speech": "neutral",
                "skip": "neutral",
            }
            sentiment = label_map.get(label, "neutral")

            return {
                "sentiment": sentiment,
                "confidence": score,
                "model_used": "rubert",
                "raw_label": label,
            }
        except Exception as e:
            media_ai_logger.debug(f"rubert prediction failed: {e}")

    # Heuristic fallback
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    pos_count = len(words & _RU_POSITIVE)
    neg_count = len(words & _RU_NEGATIVE)

    if pos_count > neg_count:
        return {"sentiment": "positive", "confidence": 0.6, "model_used": "heuristic"}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "confidence": 0.6, "model_used": "heuristic"}
    return {"sentiment": "neutral", "confidence": 0.5, "model_used": "heuristic"}


def is_russian_text(text: str) -> bool:
    """Check if text is predominantly Russian/Cyrillic."""
    if not text:
        return False
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    return cyrillic / alpha > 0.3


# ═══════════════════════════════════════════════════════════════
#  5. BGE-M3 MULTILINGUAL EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

_bge_m3_model = None
_bge_m3_available = None


def _load_bge_m3():
    """Lazy-load BGE-M3 multilingual embedding model (FREE, uses GPU if available)."""
    global _bge_m3_model, _bge_m3_available
    if _bge_m3_available is not None:
        return _bge_m3_available
    try:
        from sentence_transformers import SentenceTransformer
        device = _detect_device()
        _bge_m3_model = SentenceTransformer("BAAI/bge-m3", device=device)
        _bge_m3_available = True
        media_ai_logger.info(f"BGE-M3 multilingual embedding model loaded (1024d, device={device})")
        return True
    except Exception as e:
        media_ai_logger.warning(f"BGE-M3 not available: {e}")
        _bge_m3_available = False
        return False


def embed_multilingual(text: str) -> Optional[Any]:
    """
    Generate multilingual embedding using BGE-M3.
    Supports 100+ languages including English, Russian, Turkish.

    Returns:
        numpy array of shape (1024,) or None
    """
    if not _load_bge_m3():
        return None
    try:
        embedding = _bge_m3_model.encode(text, normalize_embeddings=True)
        return embedding
    except Exception as e:
        media_ai_logger.debug(f"BGE-M3 embedding failed: {e}")
        return None


def embed_multilingual_batch(texts: list) -> Optional[Any]:
    """
    Generate multilingual embeddings for a batch of texts.

    Returns:
        numpy array of shape (n, 1024) or None
    """
    if not _load_bge_m3():
        return None
    try:
        embeddings = _bge_m3_model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings
    except Exception as e:
        media_ai_logger.debug(f"BGE-M3 batch embedding failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  6. UPGRADED FAISS VECTOR MEMORY (BGE-M3 powered)
# ═══════════════════════════════════════════════════════════════

_faiss_available = None
_vector_stores: Dict[int, Any] = {}  # chat_id -> {"index": faiss_index, "docs": [...]}
VECTOR_MEMORY_DIR = Path(__file__).parent / "vector_memory"


def _check_faiss():
    """Check if FAISS is available."""
    global _faiss_available
    if _faiss_available is not None:
        return _faiss_available
    try:
        import faiss  # noqa: F401
        _faiss_available = True
        return True
    except ImportError:
        _faiss_available = False
        media_ai_logger.warning("FAISS not available. Install: pip install faiss-cpu")
        return False


def _get_embedding_dim() -> int:
    """Get embedding dimension based on available model."""
    if _load_bge_m3():
        return 1024  # BGE-M3
    return 384  # all-MiniLM-L6-v2 fallback


def _embed_for_memory(text: str) -> Optional[Any]:
    """Get embedding for vector memory using best available model."""
    import numpy as np

    # Try BGE-M3 first (multilingual)
    emb = embed_multilingual(text)
    if emb is not None:
        return emb.astype(np.float32)

    # Fall back to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        _fallback = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _fallback.encode(text, normalize_embeddings=True)
        return emb.astype(np.float32)
    except Exception:
        return None


def _get_vector_store(chat_id: int) -> Optional[Dict]:
    """Get or create FAISS vector store for a chat."""
    if not _check_faiss():
        return None

    import faiss
    import numpy as np

    if chat_id in _vector_stores:
        return _vector_stores[chat_id]

    dim = _get_embedding_dim()

    # Try loading from disk
    VECTOR_MEMORY_DIR.mkdir(exist_ok=True)
    index_path = VECTOR_MEMORY_DIR / f"chat_{chat_id}.index"
    docs_path = VECTOR_MEMORY_DIR / f"chat_{chat_id}.json"

    if index_path.exists() and docs_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(docs_path, "r") as f:
                docs = json.load(f)
            store = {"index": index, "docs": docs, "dim": dim}
            _vector_stores[chat_id] = store
            media_ai_logger.info(f"Loaded vector memory for chat {chat_id}: {len(docs)} entries")
            return store
        except Exception as e:
            media_ai_logger.warning(f"Failed to load vector memory: {e}")

    # Create new
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity with normalized vectors)
    store = {"index": index, "docs": [], "dim": dim}
    _vector_stores[chat_id] = store
    return store


def _save_vector_store(chat_id: int):
    """Persist vector store to disk."""
    if chat_id not in _vector_stores or not _check_faiss():
        return

    import faiss

    VECTOR_MEMORY_DIR.mkdir(exist_ok=True)
    store = _vector_stores[chat_id]

    try:
        faiss.write_index(store["index"], str(VECTOR_MEMORY_DIR / f"chat_{chat_id}.index"))
        with open(VECTOR_MEMORY_DIR / f"chat_{chat_id}.json", "w") as f:
            json.dump(store["docs"], f, ensure_ascii=False, indent=2)
    except Exception as e:
        media_ai_logger.warning(f"Failed to save vector memory: {e}")


def store_memory_vector(
    chat_id: int,
    text: str,
    memory_type: str = "general",
    emotional_tag: str = "neutral",
    importance: float = 0.5,
    sender: str = "unknown",
) -> bool:
    """
    Store a text in the FAISS vector memory.

    Args:
        chat_id: Chat ID
        text: Text to store
        memory_type: Type (fact, emotion, topic, preference, general)
        emotional_tag: Emotional context
        importance: 0.0-1.0 importance score
        sender: Who said it

    Returns:
        True if stored successfully
    """
    import numpy as np

    store = _get_vector_store(chat_id)
    if store is None:
        return False

    embedding = _embed_for_memory(text)
    if embedding is None:
        return False

    # Ensure correct dimension
    if len(embedding) != store["dim"]:
        media_ai_logger.warning(
            f"Embedding dim mismatch: got {len(embedding)}, expected {store['dim']}"
        )
        return False

    # Add to index
    store["index"].add(np.array([embedding]))
    store["docs"].append({
        "text": text,
        "type": memory_type,
        "emotion": emotional_tag,
        "importance": importance,
        "sender": sender,
        "timestamp": time.time(),
    })

    # Auto-save every 10 entries
    if len(store["docs"]) % 10 == 0:
        _save_vector_store(chat_id)

    return True


def search_memory_vector(
    chat_id: int,
    query: str,
    top_k: int = 5,
    min_score: float = 0.3,
) -> list:
    """
    Search vector memory for relevant memories.

    Args:
        chat_id: Chat ID
        query: Search query
        top_k: Number of results
        min_score: Minimum similarity score

    Returns:
        List of matching memory dicts with scores
    """
    import numpy as np

    store = _get_vector_store(chat_id)
    if store is None or store["index"].ntotal == 0:
        return []

    query_embedding = _embed_for_memory(query)
    if query_embedding is None:
        return []

    # Search
    scores, indices = store["index"].search(np.array([query_embedding]), min(top_k, store["index"].ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(store["docs"]):
            continue
        if score < min_score:
            continue
        doc = store["docs"][idx].copy()
        doc["score"] = float(score)
        results.append(doc)

    return results


def format_vector_memory_for_prompt_v2(chat_id: int, query: str, max_memories: int = 5) -> str:
    """Format retrieved vector memories for injection into system prompt."""
    memories = search_memory_vector(chat_id, query, top_k=max_memories)
    if not memories:
        return ""

    lines = ["## Deep Memory (vector retrieval):"]
    for mem in memories:
        sender_tag = "You" if mem.get("sender") == "Me" else "They"
        lines.append(
            f"- [{mem['type']}] {sender_tag} said: \"{mem['text'][:100]}\" "
            f"(relevance: {mem['score']:.0%})"
        )

    return "\n".join(lines)


def auto_extract_and_store_v2(
    chat_id: int, text: str, sender: str, emotions_28: Optional[Dict] = None
) -> int:
    """
    Auto-extract important information from text and store in vector memory.

    Returns:
        Number of memories stored
    """
    if not text or len(text) < 5:
        return 0

    stored_count = 0
    emotional_tag = "neutral"
    if emotions_28:
        emotional_tag = emotions_28.get("primary_emotion", "neutral")

    # Extract facts (sentences with proper nouns, numbers, dates)
    fact_patterns = [
        r'\b(?:my|i|our|we)\b.*\b(?:name|birthday|job|work|live|study|school|from|born)\b',
        r'\b(?:мой|моя|моё|наш|наша)\b.*\b(?:имя|день рождения|работа|учусь|живу)\b',
        r'\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b',  # dates
    ]
    for pattern in fact_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            store_memory_vector(
                chat_id, text, memory_type="fact",
                emotional_tag=emotional_tag, importance=0.8, sender=sender
            )
            stored_count += 1
            break

    # Extract preferences
    pref_patterns = [
        r'\b(?:love|like|hate|prefer|favorite|fav|enjoy|can\'t stand)\b',
        r'\b(?:люблю|нравится|ненавижу|предпочитаю|любимый|обожаю)\b',
    ]
    for pattern in pref_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            store_memory_vector(
                chat_id, text, memory_type="preference",
                emotional_tag=emotional_tag, importance=0.7, sender=sender
            )
            stored_count += 1
            break

    # Store emotionally significant messages
    if emotions_28:
        primary_score = emotions_28.get("primary_score", 0)
        if primary_score > 0.7 and emotional_tag not in ("neutral", "approval"):
            store_memory_vector(
                chat_id, text, memory_type="emotion",
                emotional_tag=emotional_tag, importance=primary_score, sender=sender
            )
            stored_count += 1

    # Store anything long enough as general context (lower importance)
    if len(text) > 50 and stored_count == 0:
        store_memory_vector(
            chat_id, text, memory_type="general",
            emotional_tag=emotional_tag, importance=0.3, sender=sender
        )
        stored_count += 1

    return stored_count


# ═══════════════════════════════════════════════════════════════
#  MODULE STATUS / CAPABILITIES CHECK
# ═══════════════════════════════════════════════════════════════

def get_media_ai_status() -> Dict[str, Any]:
    """Get status of all media AI capabilities (checks library availability, lazy-loads on demand)."""
    # Check library availability without eagerly loading heavy models
    def _whisper_check():
        if _whisper_available is not None:
            return _whisper_available
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _rubert_check():
        if _rubert_sentiment_available is not None:
            return _rubert_sentiment_available
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _bge_check():
        if _bge_m3_available is not None:
            return _bge_m3_available
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    return {
        "voice_transcription": {
            "available": _whisper_check(),
            "backend": _whisper_backend or ("faster-whisper" if _whisper_check() else "none"),
        },
        "image_understanding": {
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "backend": "claude-vision",
        },
        "voice_response": {
            "available": _check_edge_tts() if _edge_tts_available is None else _edge_tts_available,
            "backend": "edge-tts",
        },
        "russian_nlp": {
            "available": _rubert_check(),
            "backend": "rubert" if _rubert_check() else "heuristic",
        },
        "multilingual_embeddings": {
            "available": _bge_check(),
            "backend": "bge-m3" if _bge_check() else "all-MiniLM-L6-v2",
        },
        "vector_memory": {
            "available": _check_faiss() if _faiss_available is None else _faiss_available,
            "backend": "faiss",
        },
    }


def warmup_media_models():
    """Preload whisper and other media models at startup."""
    media_ai_logger.info("Warming up media AI models...")
    loaded = _load_whisper()
    media_ai_logger.info(f"  Whisper: {'OK (' + _whisper_backend + ')' if loaded else 'unavailable'}")
    media_ai_logger.info("Media AI warmup complete.")
