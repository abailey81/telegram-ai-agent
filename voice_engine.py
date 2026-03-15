"""
Advanced Voice Synthesis Engine
================================
Multi-backend neural voice synthesis with emotion modulation, prosody control,
voice cloning, and conversation-aware speech generation.

Backends (in priority order):
1. Chatterbox Multilingual (Resemble AI) — SoTA voice cloning, 23 languages, zero-shot
2. F5-TTS (MLX) — Flow matching voice cloning, Apple Silicon optimized, zero-shot
3. Bark (Suno) — Ultra-natural speech with laughs, sighs, hesitations
4. Edge-TTS — Fast Microsoft neural TTS (fallback)

Features:
- Zero-shot voice cloning from 5-10s reference audio
- Emotion-to-prosody mapping (angry → fast + loud, sad → slow + soft)
- Voice style analysis from incoming voice messages
- Conversation-aware speech (match energy, adapt to context)
- Multi-language (English + Russian + auto-detect)
- Voice message caching and deduplication
- Apple Silicon (MPS) acceleration via MLX
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

voice_logger = logging.getLogger("voice_engine")

# ═══════════════════════════════════════════════════════════════
#  DIRECTORIES
# ═══════════════════════════════════════════════════════════════

VOICE_DATA_DIR = Path("engine_data/voice")
VOICE_DATA_DIR.mkdir(parents=True, exist_ok=True)
VOICE_CACHE_DIR = VOICE_DATA_DIR / "cache"
VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_PROFILES_DIR = VOICE_DATA_DIR / "profiles"
VOICE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
VOICE_REFS_DIR = VOICE_DATA_DIR / "references"
VOICE_REFS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. BACKEND DETECTION & LAZY LOADING
# ═══════════════════════════════════════════════════════════════

_optimal_voice_params_cache = None
_optimal_voice_params_mtime = 0


def _load_optimal_voice_params() -> dict:
    """Load autoresearch-optimized voice params from engine_data/voice/optimal_params.json.
    Returns empty dict if no params file exists (use defaults)."""
    global _optimal_voice_params_cache, _optimal_voice_params_mtime
    params_path = VOICE_DATA_DIR / "optimal_params.json"
    try:
        if params_path.exists():
            mtime = params_path.stat().st_mtime
            if mtime != _optimal_voice_params_mtime or _optimal_voice_params_cache is None:
                with open(params_path) as f:
                    data = json.load(f)
                _optimal_voice_params_cache = data.get("optimal_params", {})
                _optimal_voice_params_mtime = mtime
                voice_logger.info(f"Loaded autoresearch voice params: {_optimal_voice_params_cache}")
            return _optimal_voice_params_cache
    except Exception as e:
        voice_logger.debug(f"Could not load optimal voice params: {e}")
    return {}


_bark_available = None
_bark_models = {}


def _check_bark() -> bool:
    """Check if Bark is available and loadable."""
    global _bark_available
    if _bark_available is not None:
        return _bark_available
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models  # noqa: F401
        _bark_available = True
        voice_logger.info("Bark TTS backend available")
    except ImportError:
        _bark_available = False
        voice_logger.info("Bark not installed — using edge-tts fallback")
    return _bark_available


def _load_bark():
    """Preload Bark models (first call takes ~30s)."""
    if not _check_bark():
        return False
    if _bark_models.get("loaded"):
        return True
    try:
        from bark import preload_models
        preload_models()
        _bark_models["loaded"] = True
        voice_logger.info("Bark models preloaded successfully")
        return True
    except Exception as e:
        voice_logger.warning(f"Failed to preload Bark: {e}")
        _bark_available = False
        return False


# ═══════════════════════════════════════════════════════════════
#  1b. CHATTERBOX VOICE CLONING BACKEND (SoTA)
# ═══════════════════════════════════════════════════════════════

_chatterbox_available = None
_chatterbox_model = None
_chatterbox_lock = None  # created lazily to avoid binding to wrong event loop


def _get_chatterbox_lock():
    """Get or create the asyncio lock (lazy to avoid event loop binding issues)."""
    global _chatterbox_lock
    if _chatterbox_lock is None:
        _chatterbox_lock = asyncio.Lock()
    return _chatterbox_lock


def _check_chatterbox() -> bool:
    """Check if Chatterbox TTS (multilingual) is available."""
    global _chatterbox_available
    if _chatterbox_available is not None:
        return _chatterbox_available
    try:
        # Try multilingual first (23 languages inc. Russian)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401
        _chatterbox_available = True
        voice_logger.info("Chatterbox Multilingual TTS backend available (23 languages)")
    except ImportError:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
            _chatterbox_available = True
            voice_logger.info("Chatterbox TTS backend available (English)")
        except ImportError:
            _chatterbox_available = False
            voice_logger.info("Chatterbox not installed (pip install chatterbox-tts)")
    return _chatterbox_available


_chatterbox_is_multilingual = None


def _load_chatterbox():
    """Lazy-load Chatterbox model. Prefers multilingual for Russian support.
    Returns the model or None."""
    global _chatterbox_model, _chatterbox_is_multilingual
    if _chatterbox_model is not None:
        return _chatterbox_model
    if not _check_chatterbox():
        return None
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # MPS doesn't support FFT ops needed by Chatterbox, use CPU
            device = "cpu"
        # Try multilingual first (supports Russian, English, + 21 more)
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            voice_logger.info(f"Loading Chatterbox Multilingual on {device}...")
            _chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            _chatterbox_is_multilingual = True
            voice_logger.info("Chatterbox Multilingual loaded (23 languages)")
            return _chatterbox_model
        except (ImportError, Exception) as e:
            voice_logger.debug(f"Multilingual Chatterbox unavailable: {e}")
        # Fallback to English-only
        from chatterbox.tts import ChatterboxTTS
        voice_logger.info(f"Loading Chatterbox TTS on {device}...")
        _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        _chatterbox_is_multilingual = False
        voice_logger.info("Chatterbox TTS loaded (English)")
        return _chatterbox_model
    except Exception as e:
        voice_logger.warning(f"Failed to load Chatterbox: {e}")
        _chatterbox_available = False
        return None


def _find_best_reference(chat_id: int) -> Optional[str]:
    """Find the best voice reference audio file for a chat.
    Prefers WAV, then OGG. Returns path or None."""
    ref_dir = VOICE_REFS_DIR / str(chat_id)
    if not ref_dir.exists():
        return None
    # Prefer WAV files (better quality for cloning)
    wavs = sorted(ref_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if wavs:
        return str(wavs[0])
    # Then OGG
    oggs = sorted(ref_dir.glob("*.ogg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if oggs:
        return str(oggs[0])
    # Then MP3
    mp3s = sorted(ref_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    if mp3s:
        return str(mp3s[0])
    return None


def _get_user_reference() -> Optional[str]:
    """Get the user's own voice reference (for generating voice messages AS the user).
    Stored in engine_data/voice/my_voice/"""
    my_voice_dir = VOICE_DATA_DIR / "my_voice"
    my_voice_dir.mkdir(parents=True, exist_ok=True)
    # Check for any audio files
    for ext in ("*.wav", "*.ogg", "*.mp3", "*.m4a"):
        files = sorted(my_voice_dir.glob(ext), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return str(files[0])
    return None


def _normalize_russian_text(text: str) -> str:
    """Advanced Russian text normalization for TTS.

    Converts numbers, abbreviations, symbols and other non-speakable text
    into natural Russian words that the TTS model can pronounce correctly.
    """
    # ── Russian number words ──
    _units = {
        0: "", 1: "один", 2: "два", 3: "три", 4: "четыре",
        5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять",
        10: "десять", 11: "одиннадцать", 12: "двенадцать", 13: "тринадцать",
        14: "четырнадцать", 15: "пятнадцать", 16: "шестнадцать",
        17: "семнадцать", 18: "восемнадцать", 19: "девятнадцать",
    }
    _tens = {
        2: "двадцать", 3: "тридцать", 4: "сорок", 5: "пятьдесят",
        6: "шестьдесят", 7: "семьдесят", 8: "восемьдесят", 9: "девяносто",
    }
    _hundreds = {
        1: "сто", 2: "двести", 3: "триста", 4: "четыреста",
        5: "пятьсот", 6: "шестьсот", 7: "семьсот", 8: "восемьсот", 9: "девятьсот",
    }
    _thousands = {1: "тысяча", 2: "тысячи", 5: "тысяч"}

    def _num_to_words(n: int) -> str:
        if n == 0:
            return "ноль"
        if n < 0:
            return "минус " + _num_to_words(-n)

        parts = []
        if n >= 1000000:
            millions = n // 1000000
            parts.append(_num_to_words(millions) + " " + (
                "миллион" if millions % 10 == 1 and millions % 100 != 11
                else "миллиона" if 2 <= millions % 10 <= 4 and not (12 <= millions % 100 <= 14)
                else "миллионов"
            ))
            n %= 1000000

        if n >= 1000:
            thousands = n // 1000
            if thousands == 1:
                parts.append("тысяча")
            elif thousands == 2:
                parts.append("две тысячи")
            elif 3 <= thousands <= 4:
                parts.append(_num_to_words(thousands) + " тысячи")
            elif 5 <= thousands <= 20:
                parts.append(_num_to_words(thousands) + " тысяч")
            elif thousands % 10 == 1 and thousands != 11:
                parts.append(_num_to_words(thousands) + " тысяча")
            elif 2 <= thousands % 10 <= 4 and not (12 <= thousands % 100 <= 14):
                parts.append(_num_to_words(thousands) + " тысячи")
            else:
                parts.append(_num_to_words(thousands) + " тысяч")
            n %= 1000

        if n >= 100:
            parts.append(_hundreds[n // 100])
            n %= 100

        if n >= 20:
            parts.append(_tens[n // 10])
            n %= 10

        if n > 0:
            parts.append(_units[n])

        return " ".join(p for p in parts if p)

    # Replace standalone numbers (up to 7 digits) with Russian words
    def _replace_number(m):
        num = int(m.group())
        if num > 9999999:
            return m.group()  # Too large, keep as-is
        return _num_to_words(num)

    text = re.sub(r'\b(\d{1,7})\b', _replace_number, text)

    # ── Common Russian abbreviations for TTS ──
    abbreviations = {
        r'\bт\.д\.': "так далее",
        r'\bт\.п\.': "тому подобное",
        r'\bт\.к\.': "так как",
        r'\bт\.е\.': "то есть",
        r'\bи т\.д\.': "и так далее",
        r'\bи т\.п\.': "и тому подобное",
        r'\bнапр\.': "например",
        r'\bдр\.': "другие",
        r'\bг\.': "год",
        r'\bгг\.': "годы",
        r'\bв\.': "век",
        r'\bвв\.': "века",
        r'\bруб\.': "рублей",
        r'\bтыс\.': "тысяч",
        r'\bмлн\.': "миллионов",
        r'\bмлрд\.': "миллиардов",
        r'\bкм': "километров",
        r'\bм\.': "метров",
        r'\bсм\.': "смотри",
        r'\bстр\.': "страница",
        r'\bул\.': "улица",
        r'\bд\.': "дом",
        r'\bкв\.': "квартира",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # ── Symbol replacements ──
    text = text.replace("&", " и ")
    text = text.replace("%", " процентов")
    text = text.replace("$", " долларов")
    text = text.replace("€", " евро")
    text = text.replace("₽", " рублей")
    text = text.replace("+", " плюс ")
    text = text.replace("=", " равно ")
    text = text.replace("@", " собака ")

    # ── Clean up URLs and emails (unpronounceable) ──
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # ── Normalize whitespace ──
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text


def _split_for_tts(text: str, lang: str = "en", max_chars: int = 120) -> List[str]:
    """Split text into TTS-friendly sentences.

    Chatterbox produces much better audio for shorter sentences (under ~120 chars).
    For Russian, respects natural sentence boundaries (., !, ?, ;).
    Short text is returned as-is.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?;])\s+', text)

    # Merge very short fragments back together
    sentences = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) + 1 <= max_chars:
            current += " " + part
        else:
            if current:
                sentences.append(current)
            current = part
    if current:
        sentences.append(current)

    # If still too long, split on commas
    final = []
    for s in sentences:
        if len(s) <= max_chars:
            final.append(s)
        else:
            comma_parts = re.split(r',\s+', s)
            buf = ""
            for cp in comma_parts:
                if buf and len(buf) + len(cp) + 2 <= max_chars:
                    buf += ", " + cp
                else:
                    if buf:
                        final.append(buf)
                    buf = cp
            if buf:
                final.append(buf)

    return final if final else [text]


async def generate_chatterbox_audio(
    text: str,
    reference_audio: Optional[str] = None,
    language: str = "en",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> Optional[bytes]:
    """Generate audio using Chatterbox with voice cloning.

    Automatically uses the multilingual model for non-English (Russian, etc.)
    and the base model for English. Supports zero-shot cloning from 5-10s audio.

    Russian-specific tuning:
    - Higher cfg_weight (0.7) for better adherence to text
    - Lower temperature (0.65) for more stable/natural Russian output
    - Higher repetition_penalty (2.5) to avoid stuttering on Cyrillic
    - Preserves Russian punctuation for natural prosody

    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio for voice cloning (5-10s WAV/OGG/MP3)
        language: Language code ("en", "ru", etc.)
        exaggeration: Emotional expressiveness (0.0-1.0)
        cfg_weight: Classifier-free guidance weight (0.0-1.0)
        temperature: Generation temperature (None = use optimal/default)
        repetition_penalty: Repetition penalty (None = use optimal/default)

    Returns:
        WAV audio bytes, or None on failure
    """
    model = _load_chatterbox()
    if model is None:
        return None

    try:
        import torchaudio as ta

        processed_text = _preprocess_for_speech(text)

        lang_id = "en"
        if language in ("russian", "ru"):
            lang_id = "ru"
        elif language in ("turkish", "tr"):
            lang_id = "tr"
        elif language in ("german", "de"):
            lang_id = "de"
        elif language in ("french", "fr"):
            lang_id = "fr"
        elif language in ("spanish", "es"):
            lang_id = "es"

        voice_logger.info(
            f"Chatterbox generating: '{processed_text[:80]}...' "
            f"lang={lang_id}, ref={'yes' if reference_audio else 'no'}, "
            f"multilingual={_chatterbox_is_multilingual}"
        )

        # Convert OGG reference to WAV if needed
        if reference_audio and reference_audio.endswith(".ogg"):
            wav_ref = await convert_ogg_to_wav(reference_audio)
            if wav_ref:
                reference_audio = wav_ref

        # Build kwargs based on model type
        if _chatterbox_is_multilingual:
            # Multilingual model — language-specific tuning for best quality
            generate_kwargs = {
                "text": processed_text,
                "language_id": lang_id,
                "exaggeration": exaggeration,
            }
            if reference_audio and os.path.exists(reference_audio):
                generate_kwargs["audio_prompt_path"] = reference_audio

            # Voice-cloning-first parameter tuning
            # Priority: MAXIMUM voice similarity to reference > text adherence
            # If autoresearch has discovered optimal params, use those; otherwise use defaults.
            has_ref = reference_audio and os.path.exists(reference_audio)
            if has_ref:
                optimal = _load_optimal_voice_params()
                # Explicit params override optimal; optimal overrides defaults
                generate_kwargs.update({
                    "cfg_weight": cfg_weight if cfg_weight != 0.5 else optimal.get("cfg_weight", 0.35),
                    "temperature": temperature if temperature is not None else optimal.get("temperature", 0.75),
                    "repetition_penalty": repetition_penalty if repetition_penalty is not None else optimal.get("repetition_penalty", 1.8),
                    "exaggeration": exaggeration if exaggeration != 0.5 else optimal.get("exaggeration", 0.3),
                })
            else:
                generate_kwargs.update({
                    "cfg_weight": cfg_weight,
                    "temperature": temperature if temperature is not None else 0.8,
                    "exaggeration": exaggeration,
                })
                if repetition_penalty is not None:
                    generate_kwargs["repetition_penalty"] = repetition_penalty
        else:
            # English-only model uses exaggeration/cfg_weight
            generate_kwargs = {
                "text": processed_text,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
            if reference_audio and os.path.exists(reference_audio):
                generate_kwargs["audio_prompt_path"] = reference_audio

        # Pre-stress Russian text BEFORE asyncio.to_thread to avoid SQLite
        # thread-safety issues (russian_text_stresser uses SQLite internally).
        # This also gives Chatterbox pre-processed text for better pronunciation.
        if lang_id == "ru":
            try:
                from chatterbox.models.tokenizers.tokenizer import add_russian_stress
                processed_text = add_russian_stress(processed_text)
                generate_kwargs["text"] = processed_text
                voice_logger.info(f"Pre-stressed Russian text: '{processed_text[:80]}...'")
            except Exception as e:
                voice_logger.warning(f"Russian pre-stressing failed (will retry in tokenizer): {e}")

        # For long text (especially Russian), split into sentences for better quality.
        # Chatterbox handles short sentences much better than long paragraphs.
        import torch

        # Use higher char limit for Russian — Cyrillic chars are ~2 bytes each and
        # shorter splits produce more fragmented/inconsistent voice cloning results
        split_limit = 200 if lang_id == "ru" else 120
        sentences = _split_for_tts(processed_text, lang_id, max_chars=split_limit)

        if len(sentences) > 1:
            voice_logger.info(f"Splitting into {len(sentences)} sentences for better quality")

        # For multi-sentence: pre-extract speaker conditionals ONCE from reference,
        # then generate each sentence without re-computing (consistent voice across all parts)
        has_ref = "audio_prompt_path" in generate_kwargs
        if has_ref and len(sentences) > 1:
            # Pre-load reference audio into model.conds — only once
            ref_path = generate_kwargs.pop("audio_prompt_path")
            exag = generate_kwargs.get("exaggeration", 0.3)
            await asyncio.to_thread(
                lambda: model.prepare_conditionals(ref_path, exaggeration=exag)
            )
            voice_logger.info("Pre-cached speaker conditionals for multi-sentence generation")

        wav_parts = []
        for i, sent in enumerate(sentences):
            sent_kwargs = dict(generate_kwargs)
            sent_kwargs["text"] = sent
            voice_logger.debug(f"  Generating sentence {i+1}/{len(sentences)}: '{sent[:50]}...'")
            part = await asyncio.to_thread(lambda kw=sent_kwargs: model.generate(**kw))
            wav_parts.append(part)

        # Concatenate sentence audio with natural silence gaps
        if len(wav_parts) > 1:
            silence = torch.zeros(1, int(model.sr * 0.2))  # 200ms pause between sentences
            combined = []
            for i, part in enumerate(wav_parts):
                combined.append(part)
                if i < len(wav_parts) - 1:
                    combined.append(silence)
            wav = torch.cat(combined, dim=1)
        else:
            wav = wav_parts[0]

        # Convert to WAV bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        ta.save(tmp_path, wav, model.sr)

        # Post-processing: match generated audio's volume/dynamics to reference
        # This ensures the output "sounds like" the reference in terms of loudness,
        # energy envelope, and perceived closeness — beyond just the voice timbre.
        if reference_audio and os.path.exists(reference_audio) and _ffmpeg_available():
            try:
                matched_path = await _match_audio_to_reference(tmp_path, reference_audio)
                if matched_path:
                    tmp_path = matched_path
            except Exception as e:
                voice_logger.debug(f"Audio matching skipped: {e}")

        wav_bytes = Path(tmp_path).read_bytes()
        os.unlink(tmp_path)

        voice_logger.info(
            f"Chatterbox generated {len(wav_bytes)} bytes "
            f"(cloned={'yes' if reference_audio else 'no'}, lang={lang_id}, "
            f"sentences={len(sentences)})"
        )
        return wav_bytes

    except Exception as e:
        voice_logger.error(f"Chatterbox generation failed: {e}")
        return None


async def _match_audio_to_reference(generated_path: str, reference_path: str) -> Optional[str]:
    """Match generated audio's loudness and dynamics to the reference voice.

    Measures the reference's mean volume and applies loudnorm to match it.
    This makes the generated audio "sit" at the same perceived level as the
    original voice — critical for convincing voice cloning.
    """
    import subprocess
    try:
        # Measure reference audio volume
        ref_probe = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-i", reference_path, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        stderr = ref_probe.stderr.decode() if ref_probe.stderr else ""
        ref_mean = -16.0  # default
        ref_max = -1.5
        for line in stderr.split("\n"):
            if "mean_volume" in line:
                try:
                    ref_mean = float(line.split("mean_volume:")[1].strip().split(" ")[0])
                except Exception:
                    pass
            if "max_volume" in line:
                try:
                    ref_max = float(line.split("max_volume:")[1].strip().split(" ")[0])
                except Exception:
                    pass

        # Target loudness: match reference, clamped to safe range
        target_i = max(-24, min(-10, ref_mean))
        target_tp = max(-3, min(-0.5, ref_max + 0.5))

        matched_path = generated_path + ".matched.wav"
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y", "-i", generated_path,
                "-af", f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA=7",
                "-c:a", "pcm_s16le", matched_path,
            ],
            capture_output=True, timeout=15,
        )
        if result.returncode == 0 and os.path.exists(matched_path):
            os.unlink(generated_path)
            voice_logger.debug(f"Matched output volume to reference (I={target_i:.1f}dB)")
            return matched_path
    except Exception as e:
        voice_logger.debug(f"Volume matching failed: {e}")
    return None


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is installed."""
    import shutil
    return shutil.which("ffmpeg") is not None


async def convert_audio(src_path: str, dst_format: str = "wav",
                        sample_rate: int = 24000, channels: int = 1) -> Optional[str]:
    """Convert audio between formats. Tries ffmpeg first, falls back to pydub, then afconvert.

    Args:
        src_path: Source audio file path
        dst_format: Target format ("wav", "ogg", "mp3")
        sample_rate: Target sample rate (default 24000 for TTS models)
        channels: Number of channels (default 1 = mono)

    Returns:
        Path to converted file, or None on failure
    """
    import subprocess

    dst_path = src_path.rsplit(".", 1)[0] + f".{dst_format}"
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 100:
        return dst_path

    # 1. Try ffmpeg (best quality, most format support)
    if _ffmpeg_available():
        try:
            codec_args = []
            if dst_format == "ogg":
                codec_args = ["-acodec", "libopus", "-b:a", "48k"]
            elif dst_format == "wav":
                codec_args = ["-acodec", "pcm_s16le"]
            result = await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-i", src_path, "-ar", str(sample_rate),
                 "-ac", str(channels)] + codec_args + [dst_path],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(dst_path):
                voice_logger.info(f"ffmpeg: {src_path} -> {dst_path}")
                return dst_path
        except Exception as e:
            voice_logger.debug(f"ffmpeg conversion failed: {e}")

    # 2. Try pydub (pure Python, needs audioop but handles many formats)
    try:
        from pydub import AudioSegment
        src_fmt = src_path.rsplit(".", 1)[-1].lower()
        if src_fmt in ("ogg", "opus", "oga"):
            src_fmt = "ogg"
        audio = await asyncio.to_thread(AudioSegment.from_file, src_path, format=src_fmt)
        audio = audio.set_frame_rate(sample_rate).set_channels(channels)
        out_fmt = "ogg" if dst_format == "ogg" else dst_format
        await asyncio.to_thread(audio.export, dst_path, format=out_fmt)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 100:
            voice_logger.info(f"pydub: {src_path} -> {dst_path}")
            return dst_path
    except Exception as e:
        voice_logger.debug(f"pydub conversion failed: {e}")

    # 3. macOS afconvert (WAV output only, built-in on macOS)
    if dst_format == "wav":
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["afconvert", "-f", "WAVE", "-d", "LEI16",
                 "-c", str(channels), src_path, dst_path],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(dst_path):
                voice_logger.info(f"afconvert: {src_path} -> {dst_path}")
                return dst_path
        except Exception as e:
            voice_logger.debug(f"afconvert failed: {e}")

    voice_logger.warning(f"All audio conversion methods failed for {src_path} -> {dst_format}")
    return None


async def convert_to_ogg_opus(src_path: str) -> Optional[str]:
    """Convert any audio to OGG/Opus for Telegram voice notes."""
    return await convert_audio(src_path, dst_format="ogg", sample_rate=48000, channels=1)


async def convert_ogg_to_wav(ogg_path: str, preserve_voice: bool = True) -> Optional[str]:
    """Convert OGG/Opus to WAV for Chatterbox reference input (24kHz mono).

    When preserve_voice=True (default, for voice cloning reference):
    - MINIMAL processing only — just resample + trim silence
    - NO denoising, NO lowpass, NO compression — these destroy voice identity
    - Voice cloning needs the full spectral signature of the original voice

    When preserve_voice=False (for generated output post-processing):
    - Light normalization only
    """
    wav_path = ogg_path.rsplit(".", 1)[0] + ".wav"
    try:
        import subprocess
        if preserve_voice:
            # MINIMAL pipeline — preserve voice identity for cloning
            # Only: highpass rumble, trim silence, normalize volume (no spectral changes)
            af_chain = ",".join([
                "highpass=f=60",                # Remove sub-bass rumble only
                "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB",
                "loudnorm=I=-16:TP=-1.5:LRA=11",  # Volume normalization (no spectral change)
            ])
        else:
            # Light cleanup for non-reference audio
            af_chain = ",".join([
                "highpass=f=80",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
            ])
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y", "-i", ogg_path,
                "-ar", "24000",        # Chatterbox S3GEN_SR
                "-ac", "1",            # Mono
                "-af", af_chain,
                wav_path,
            ],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            voice_logger.info(f"Converted {ogg_path} -> {wav_path} (24kHz, preserve_voice={preserve_voice})")
            return wav_path
    except Exception as e:
        voice_logger.debug(f"Enhanced conversion failed, falling back: {e}")
    # Fallback to simple conversion
    return await convert_audio(ogg_path, dst_format="wav", sample_rate=24000, channels=1)


async def store_my_voice_reference(audio_bytes: bytes, filename: str = "my_voice") -> Optional[str]:
    """Store the user's own voice for generating voice messages that sound like them.

    Converts to 24kHz mono WAV with MINIMAL processing (preserve voice identity).
    Then trims to the densest 10s speech segment for optimal Chatterbox cloning
    (DEC_COND_LEN=10s at 24kHz, ENC_COND_LEN=6s at 16kHz — Chatterbox truncates to first N seconds,
    so we ensure those first seconds contain continuous speech, not silence).
    """
    my_voice_dir = VOICE_DATA_DIR / "my_voice"
    my_voice_dir.mkdir(parents=True, exist_ok=True)

    # Determine format from first bytes
    ext = "ogg"
    if audio_bytes[:4] == b"RIFF":
        ext = "wav"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        ext = "mp3"

    raw_path = my_voice_dir / f"{filename}_raw.{ext}"
    wav_path = my_voice_dir / f"{filename}.wav"
    try:
        raw_path.write_bytes(audio_bytes)
        voice_logger.info(f"Stored raw user voice reference: {raw_path} ({len(audio_bytes)//1024}KB)")

        # Convert to 24kHz WAV with MINIMAL processing (preserve voice identity)
        wav_result = await convert_ogg_to_wav(str(raw_path), preserve_voice=True)
        if wav_result and os.path.exists(wav_result):
            # Rename to standard name if needed
            if wav_result != str(wav_path):
                import shutil
                shutil.move(wav_result, str(wav_path))

            # Trim to optimal length for Chatterbox (max 10s, starting from first speech)
            await _trim_reference_to_speech(str(wav_path))

            voice_logger.info(f"Voice reference stored: {wav_path}")
            return str(wav_path)
        else:
            voice_logger.warning("Could not convert voice reference, using raw audio")
            return str(raw_path)
    except Exception as e:
        voice_logger.error(f"Failed to store voice reference: {e}")
        return None


async def _trim_reference_to_speech(wav_path: str, max_duration: float = 10.0):
    """Trim a voice reference WAV to the densest speech segment (max 10s).

    Chatterbox truncates reference to DEC_COND_LEN (10s at 24kHz). We ensure
    those 10s are packed with actual speech, not silence or noise.
    Uses ffmpeg silencedetect to find the speech-heavy region.
    """
    try:
        import subprocess
        # Check current duration
        probe = await asyncio.to_thread(
            subprocess.run,
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", wav_path],
            capture_output=True, timeout=10,
        )
        duration = float(probe.stdout.decode().strip())
        if duration <= max_duration:
            return  # Already short enough

        # Find silence segments to identify best speech window
        detect = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-i", wav_path, "-af",
             "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        stderr = detect.stderr.decode() if detect.stderr else ""

        # Parse silence start/end times
        silence_ends = []
        for line in stderr.split("\n"):
            if "silence_end" in line:
                try:
                    t = float(line.split("silence_end:")[1].strip().split(" ")[0])
                    silence_ends.append(t)
                except Exception:
                    pass

        # Start from the first speech onset (first silence_end) or 0
        start = silence_ends[0] - 0.1 if silence_ends else 0.0
        start = max(0, start)

        # Ensure we don't exceed audio length
        if start + max_duration > duration:
            start = max(0, duration - max_duration)

        # Trim
        trimmed_path = wav_path + ".trimmed.wav"
        result = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", wav_path,
             "-ss", str(start), "-t", str(max_duration),
             "-c:a", "pcm_s16le", trimmed_path],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0 and os.path.exists(trimmed_path):
            import shutil
            shutil.move(trimmed_path, wav_path)
            voice_logger.info(
                f"Trimmed reference to {max_duration}s (start={start:.1f}s from {duration:.1f}s total)"
            )
    except Exception as e:
        voice_logger.debug(f"Reference trimming skipped: {e}")


# ═══════════════════════════════════════════════════════════════
#  1c-fish. FISH SPEECH / OPENAUDIO S1 BACKEND (Premium multilingual)
# ═══════════════════════════════════════════════════════════════

_fish_speech_available = None


def _check_fish_speech() -> bool:
    """Check if Fish Speech / OpenAudio is available."""
    global _fish_speech_available
    if _fish_speech_available is not None:
        return _fish_speech_available
    try:
        from fish_speech.models.dac.inference import encode as _fs_encode  # noqa: F401
        _fish_speech_available = True
        voice_logger.info("Fish Speech / OpenAudio backend available")
        return True
    except ImportError:
        pass
    # Check for running Fish Speech API server
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:8080/v1/health", timeout=1)
        if req.status == 200:
            _fish_speech_available = True
            voice_logger.info("Fish Speech API server detected at localhost:8080")
            return True
    except Exception:
        pass
    _fish_speech_available = False
    return False


async def generate_fish_speech_audio(
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: str = "",
    language: str = "ru",
) -> Optional[bytes]:
    """Generate audio using Fish Speech / OpenAudio S1 with zero-shot voice cloning.

    Fish Speech supports 13 languages including Russian with excellent quality.
    Uses the local API server if running, otherwise falls back to direct inference.

    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio for voice cloning (10-30s WAV)
        reference_text: Transcript of the reference audio
        language: Language code

    Returns:
        WAV audio bytes, or None on failure
    """
    if not _check_fish_speech():
        return None

    processed_text = _preprocess_for_speech(text)

    # Try local HTTP API first (fastest, runs as server)
    try:
        import urllib.request
        import json as _json

        payload = {"text": processed_text, "language": language}

        if reference_audio and os.path.exists(reference_audio):
            import base64
            audio_bytes = Path(reference_audio).read_bytes()
            payload["reference_audio"] = base64.b64encode(audio_bytes).decode()
            if reference_text:
                payload["reference_text"] = reference_text

        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            "http://localhost:8080/v1/tts",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        response = await asyncio.to_thread(
            urllib.request.urlopen, req, timeout=120,
        )
        if response.status == 200:
            wav_bytes = response.read()
            voice_logger.info(f"Fish Speech API generated {len(wav_bytes)} bytes")
            return wav_bytes
    except Exception as e:
        voice_logger.debug(f"Fish Speech API failed: {e}")

    # Fallback: direct Python inference (requires local model)
    try:
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        cmd = [
            "python", "-m", "fish_speech.inference",
            "--text", processed_text,
            "--output", out_path,
        ]
        if reference_audio and os.path.exists(reference_audio):
            cmd.extend(["--reference-audio", reference_audio])
            if reference_text:
                cmd.extend(["--reference-text", reference_text])

        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, timeout=180,
        )
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            wav_bytes = Path(out_path).read_bytes()
            os.unlink(out_path)
            voice_logger.info(f"Fish Speech direct generated {len(wav_bytes)} bytes")
            return wav_bytes
        if os.path.exists(out_path):
            os.unlink(out_path)
    except Exception as e:
        voice_logger.debug(f"Fish Speech direct inference failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════
#  1c. F5-TTS MLX BACKEND (Apple Silicon optimized voice cloning)
# ═══════════════════════════════════════════════════════════════

_f5tts_available = None


def _check_f5tts() -> bool:
    """Check if F5-TTS (MLX or standard) is available."""
    global _f5tts_available
    if _f5tts_available is not None:
        return _f5tts_available
    # Try MLX variant first (Apple Silicon)
    try:
        import f5_tts_mlx  # noqa: F401
        _f5tts_available = True
        voice_logger.info("F5-TTS MLX backend available (Apple Silicon optimized)")
        return True
    except ImportError:
        pass
    # Try standard F5-TTS
    try:
        from f5_tts.api import F5TTS  # noqa: F401
        _f5tts_available = True
        voice_logger.info("F5-TTS backend available")
        return True
    except ImportError:
        pass
    _f5tts_available = False
    voice_logger.info("F5-TTS not installed (pip install f5-tts-mlx or pip install f5-tts)")
    return False


async def generate_f5tts_audio(
    text: str,
    reference_audio: Optional[str] = None,
    ref_text: str = "",
) -> Optional[bytes]:
    """Generate audio using F5-TTS with zero-shot voice cloning.

    F5-TTS uses flow matching for high-quality voice cloning.
    Optimized for Apple Silicon via MLX backend.

    Args:
        text: Text to synthesize
        reference_audio: Path to reference WAV (mono, 24kHz preferred, 5-15s)
        ref_text: Transcript of the reference audio (auto-detected if empty)

    Returns:
        WAV audio bytes, or None on failure
    """
    if not _check_f5tts():
        return None

    processed_text = _preprocess_for_speech(text)

    # Try MLX variant first (much faster on Apple Silicon)
    try:
        import subprocess
        cmd = [
            "python", "-m", "f5_tts_mlx.generate",
            "--text", processed_text,
        ]
        if reference_audio and os.path.exists(reference_audio):
            cmd.extend(["--ref-audio", reference_audio])
            if ref_text:
                cmd.extend(["--ref-text", ref_text])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        cmd.extend(["--output", out_path])

        voice_logger.info(f"F5-TTS MLX generating: '{processed_text[:60]}...'")
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            wav_bytes = Path(out_path).read_bytes()
            os.unlink(out_path)
            voice_logger.info(f"F5-TTS MLX generated {len(wav_bytes)} bytes")
            return wav_bytes
        if os.path.exists(out_path):
            os.unlink(out_path)
    except Exception as e:
        voice_logger.debug(f"F5-TTS MLX failed: {e}")

    # Fallback: standard F5-TTS Python API
    try:
        from f5_tts.api import F5TTS

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        voice_logger.info(f"F5-TTS generating: '{processed_text[:60]}...'")
        tts = F5TTS()
        await asyncio.to_thread(
            tts.infer,
            ref_file=reference_audio or "",
            ref_text=ref_text,
            gen_text=processed_text,
            file_wave=out_path,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            wav_bytes = Path(out_path).read_bytes()
            os.unlink(out_path)
            voice_logger.info(f"F5-TTS generated {len(wav_bytes)} bytes")
            return wav_bytes
        if os.path.exists(out_path):
            os.unlink(out_path)
    except Exception as e:
        voice_logger.debug(f"F5-TTS standard failed: {e}")

    return None


_edge_tts_available = None


def _check_edge_tts() -> bool:
    global _edge_tts_available
    if _edge_tts_available is not None:
        return _edge_tts_available
    try:
        import edge_tts  # noqa: F401
        _edge_tts_available = True
    except ImportError:
        _edge_tts_available = False
    return _edge_tts_available


# ═══════════════════════════════════════════════════════════════
#  2. EMOTION → PROSODY MAPPING
# ═══════════════════════════════════════════════════════════════

# Maps detected emotions to voice parameters
EMOTION_PROSODY = {
    # Emotion: (rate_multiplier, pitch_shift_Hz, volume_adjust, voice_quality)
    # Edge-TTS requires pitch in Hz (not %). ~2Hz per 1% of a ~200Hz voice.
    "anger": {"rate": 1.15, "pitch": "+10Hz", "volume": "+15%", "quality": "intense"},
    "frustration": {"rate": 1.1, "pitch": "+6Hz", "volume": "+10%", "quality": "tense"},
    "joy": {"rate": 1.1, "pitch": "+16Hz", "volume": "+5%", "quality": "bright"},
    "excitement": {"rate": 1.2, "pitch": "+20Hz", "volume": "+10%", "quality": "energetic"},
    "sadness": {"rate": 0.85, "pitch": "-10Hz", "volume": "-10%", "quality": "soft"},
    "love": {"rate": 0.9, "pitch": "-6Hz", "volume": "-5%", "quality": "warm"},
    "tenderness": {"rate": 0.85, "pitch": "-10Hz", "volume": "-10%", "quality": "gentle"},
    "fear": {"rate": 1.1, "pitch": "+10Hz", "volume": "-5%", "quality": "breathy"},
    "surprise": {"rate": 1.15, "pitch": "+24Hz", "volume": "+5%", "quality": "sharp"},
    "disgust": {"rate": 0.95, "pitch": "-4Hz", "volume": "+5%", "quality": "rough"},
    "contempt": {"rate": 0.9, "pitch": "-6Hz", "volume": "+5%", "quality": "flat"},
    "playful": {"rate": 1.05, "pitch": "+10Hz", "volume": "+3%", "quality": "lilting"},
    "sarcastic": {"rate": 0.95, "pitch": "+6Hz", "volume": "+5%", "quality": "dry"},
    "bored": {"rate": 0.85, "pitch": "-10Hz", "volume": "-10%", "quality": "flat"},
    "tired": {"rate": 0.8, "pitch": "-16Hz", "volume": "-15%", "quality": "heavy"},
    "confident": {"rate": 1.0, "pitch": "+4Hz", "volume": "+5%", "quality": "steady"},
    "nervous": {"rate": 1.1, "pitch": "+10Hz", "volume": "-5%", "quality": "shaky"},
    "neutral": {"rate": 1.0, "pitch": "+0Hz", "volume": "+0%", "quality": "natural"},
}

# Bark speaker presets mapped to emotional tone
BARK_SPEAKERS = {
    "en_neutral_m": "v2/en_speaker_6",
    "en_neutral_f": "v2/en_speaker_9",
    "en_warm_m": "v2/en_speaker_0",
    "en_warm_f": "v2/en_speaker_1",
    "en_assertive_m": "v2/en_speaker_3",
    "en_assertive_f": "v2/en_speaker_4",
    "ru_neutral_m": "v2/ru_speaker_0",
    "ru_neutral_f": "v2/ru_speaker_1",
    "ru_warm_m": "v2/ru_speaker_2",
    "ru_warm_f": "v2/ru_speaker_3",
}

# Edge-TTS voices with language and gender support
EDGE_VOICES = {
    "en_m_casual": "en-US-GuyNeural",
    "en_f_casual": "en-US-JennyNeural",
    "en_m_warm": "en-US-AndrewNeural",
    "en_f_warm": "en-US-AriaNeural",
    "en_m_assertive": "en-US-BrianNeural",
    "en_f_assertive": "en-US-EmmaNeural",
    "ru_m_casual": "ru-RU-DmitryNeural",
    "ru_f_casual": "ru-RU-SvetlanaNeural",
    "ru_m_warm": "ru-RU-DmitryNeural",
    "ru_f_warm": "ru-RU-SvetlanaNeural",
    "ru_m_assertive": "ru-RU-DmitryNeural",
    "ru_f_assertive": "ru-RU-SvetlanaNeural",
    "tr_m_casual": "tr-TR-AhmetNeural",
    "tr_f_casual": "tr-TR-EmelNeural",
    "tr_m_warm": "tr-TR-AhmetNeural",
    "tr_f_warm": "tr-TR-EmelNeural",
    "tr_m_assertive": "tr-TR-AhmetNeural",
    "tr_f_assertive": "tr-TR-EmelNeural",
}


# ═══════════════════════════════════════════════════════════════
#  3. VOICE STYLE PROFILES (per-chat)
# ═══════════════════════════════════════════════════════════════

def load_voice_profile(chat_id: int) -> Dict[str, Any]:
    """Load stored voice preferences for a chat."""
    path = VOICE_PROFILES_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "preferred_backend": "auto",
        "preferred_voice": "auto",
        "gender": "auto",
        "language": "auto",
        "speed_preference": 1.0,
        "formality": "casual",
        "emotion_intensity": 0.7,
        "bark_speaker": None,
        "edge_voice": None,
        "reference_audio_path": None,
        "samples_collected": 0,
    }


def save_voice_profile(chat_id: int, profile: Dict[str, Any]):
    """Save voice preferences for a chat."""
    path = VOICE_PROFILES_DIR / f"{chat_id}.json"
    try:
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    except Exception as e:
        voice_logger.error(f"Failed to save voice profile: {e}")


def store_voice_reference(chat_id: int, audio_bytes: bytes, label: str = "sample") -> Optional[str]:
    """Store a voice reference sample from their voice messages for future cloning."""
    ref_dir = VOICE_REFS_DIR / str(chat_id)
    ref_dir.mkdir(parents=True, exist_ok=True)

    count = len(list(ref_dir.glob("*.wav"))) + len(list(ref_dir.glob("*.ogg")))
    ext = "ogg"  # Telegram voice messages are typically OGG/Opus
    ref_path = ref_dir / f"{label}_{count}.{ext}"
    try:
        ref_path.write_bytes(audio_bytes)
        voice_logger.info(f"Stored voice reference #{count} for chat {chat_id}")

        # Update profile
        profile = load_voice_profile(chat_id)
        profile["samples_collected"] = count + 1
        profile["reference_audio_path"] = str(ref_dir)
        save_voice_profile(chat_id, profile)

        return str(ref_path)
    except Exception as e:
        voice_logger.error(f"Failed to store voice reference: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  4. VOICE ANALYSIS (analyze incoming voice messages)
# ═══════════════════════════════════════════════════════════════

def analyze_voice_characteristics(audio_bytes: bytes) -> Dict[str, Any]:
    """Analyze audio characteristics of an incoming voice message.

    Extracts: estimated energy, speaking rate, pitch range.
    Used to adapt our voice responses to match their style.
    """
    result = {
        "energy": "medium",
        "estimated_duration_s": 0,
        "speaking_rate": "normal",
        "loudness": "medium",
    }

    try:
        # Try to get duration from raw bytes (OGG has duration in header)
        if len(audio_bytes) > 0:
            # Rough duration estimate from byte size (Opus ~24kbps average for voice)
            result["estimated_duration_s"] = round(len(audio_bytes) / 3000, 1)

            # Energy estimate from byte density (higher bitrate = more dynamic audio)
            if result["estimated_duration_s"] > 0:
                bytes_per_sec = len(audio_bytes) / max(result["estimated_duration_s"], 0.1)
                if bytes_per_sec > 5000:
                    result["energy"] = "high"
                    result["loudness"] = "loud"
                elif bytes_per_sec < 2000:
                    result["energy"] = "low"
                    result["loudness"] = "quiet"

            # Speaking rate from duration vs expected
            if result["estimated_duration_s"] < 3:
                result["speaking_rate"] = "fast"
            elif result["estimated_duration_s"] > 15:
                result["speaking_rate"] = "slow"

    except Exception as e:
        voice_logger.debug(f"Voice analysis failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════
#  5. TEXT PREPROCESSING FOR SPEECH
# ═══════════════════════════════════════════════════════════════

def _preprocess_for_speech(text: str, emotion: str = "neutral") -> str:
    """Convert texting-style text to speech-friendly format.

    Language-aware: detects Russian and applies appropriate processing.
    For Russian text, preserves natural punctuation and doesn't apply English abbrevs.

    Handles:
    - Language detection (Cyrillic → Russian mode)
    - Abbreviations expansion (English only — u → you, ur → your, etc.)
    - Russian abbreviation expansion (спс → спасибо, пж → пожалуйста, etc.)
    - Emoji removal
    - Multiple message segments (||) to pauses
    """
    # Detect language
    is_russian = bool(re.search(r"[а-яА-ЯёЁ]", text))

    # Split multi-message format
    segments = [s.strip() for s in text.split("||") if s.strip()]
    processed = []

    # English texting abbreviations → spoken form
    en_abbrevs = {
        r"\bu\b": "you", r"\bur\b": "your", r"\brn\b": "right now",
        r"\bngl\b": "not gonna lie", r"\btbh\b": "to be honest",
        r"\bidk\b": "I don't know", r"\bimo\b": "in my opinion",
        r"\blol\b": "", r"\blmao\b": "", r"\bloml\b": "",
        r"\bsmth\b": "something", r"\bsmn\b": "someone",
        r"\bprob\b": "probably", r"\babt\b": "about",
        r"\bbc\b": "because", r"\btho\b": "though",
        r"\bw\b": "with", r"\bwdym\b": "what do you mean",
        r"\bfr\b": "for real", r"\bfs\b": "for sure",
        r"\bbtw\b": "by the way", r"\bbrb\b": "be right back",
        r"\bomg\b": "oh my god", r"\bwtf\b": "what the fuck",
        r"\bstfu\b": "shut the fuck up", r"\btf\b": "the fuck",
    }

    # Russian texting abbreviations → spoken form
    ru_abbrevs = {
        r"\bспс\b": "спасибо", r"\bпж\b": "пожалуйста",
        r"\bнзч\b": "не за что", r"\bкст\b": "кстати",
        r"\bхз\b": "не знаю", r"\bимхо\b": "по-моему",
        r"\bнорм\b": "нормально", r"\bоч\b": "очень",
        r"\bкороч\b": "короче", r"\bчел\b": "человек",
        r"\bлан\b": "ладно", r"\bтож\b": "тоже",
        r"\bмб\b": "может быть", r"\bтк\b": "так как",
        r"\bнп\b": "например", r"\bскок\b": "сколько",
        r"\bваще\b": "вообще", r"\bздрасте\b": "здравствуйте",
        r"\bприв\b": "привет", r"\bпасиб\b": "спасибо",
        r"\bсори\b": "извини", r"\bкмк\b": "как мне кажется",
        r"\bтп\b": "тупой", r"\bрн\b": "реально",
        # Remove chat-only sounds that don't pronounce well
        r"\bахах\w*\b": "", r"\bхахах\w*\b": "", r"\bржу\b": "",
        r"\bхх+\b": "", r"\bаа+\b": "", r"\bоо+\b": "",
    }

    for seg in segments:
        s = seg.strip()
        if not s:
            continue

        # Apply language-appropriate abbreviation expansion
        if is_russian:
            for pattern, expansion in ru_abbrevs.items():
                s = re.sub(pattern, expansion, s, flags=re.IGNORECASE)
        else:
            for pattern, expansion in en_abbrevs.items():
                s = re.sub(pattern, expansion, s, flags=re.IGNORECASE)

        # Remove emojis (they don't speak well)
        s = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            r"\U00002702-\U000027B0\U0001FA00-\U0001FA6F"
            r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            "", s,
        )

        # For Russian: keep natural punctuation (Chatterbox punc_norm handles it)
        # For English: convert ellipsis to pause
        if not is_russian:
            s = re.sub(r"\.{2,}", " ... ", s)
            s = s.rstrip(".")

        # Clean up multiple spaces
        s = re.sub(r"\s{2,}", " ", s).strip()

        if s:
            processed.append(s)

    # Join segments with natural pauses
    if len(processed) > 1:
        sep = ", " if is_russian else " ... "
        result = sep.join(processed)
    elif processed:
        result = processed[0]
    else:
        result = text

    # For Russian: normalize numbers, abbreviations, symbols into spoken words
    if is_russian:
        result = _normalize_russian_text(result)

    return result


# ═══════════════════════════════════════════════════════════════
#  6. BARK SYNTHESIS (Premium Backend)
# ═══════════════════════════════════════════════════════════════

def _select_bark_speaker(
    language: str, emotion: str, gender: str = "m",
) -> str:
    """Select optimal Bark speaker preset based on context."""
    lang = "ru" if language in ("russian", "ru") else "en"
    g = gender if gender in ("m", "f") else "m"

    # Emotion → speaker style mapping
    warm_emotions = {"love", "tenderness", "joy", "playful"}
    assertive_emotions = {"anger", "frustration", "confident", "contempt", "sarcastic"}

    if emotion in warm_emotions:
        key = f"{lang}_warm_{g}"
    elif emotion in assertive_emotions:
        key = f"{lang}_assertive_{g}"
    else:
        key = f"{lang}_neutral_{g}"

    return BARK_SPEAKERS.get(key, "v2/en_speaker_6")


async def generate_bark_audio(
    text: str,
    emotion: str = "neutral",
    language: str = "en",
    gender: str = "m",
) -> Optional[bytes]:
    """Generate audio using Bark (most natural-sounding).

    Bark can produce laughs, sighs, hesitations naturally.
    Returns WAV bytes or None on failure.
    """
    if not _load_bark():
        return None

    try:
        from bark import SAMPLE_RATE, generate_audio

        speaker = _select_bark_speaker(language, emotion, gender)
        processed_text = _preprocess_for_speech(text, emotion)

        # Add Bark emotion markers for extreme emotions
        prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
        if prosody["quality"] == "intense":
            processed_text = f"[intense] {processed_text}"
        elif prosody["quality"] == "soft":
            processed_text = f"[soft] {processed_text}"

        voice_logger.info(f"Bark generating: '{processed_text[:60]}...' speaker={speaker}")

        # Generate audio (blocking — run in thread)
        audio_array = await asyncio.to_thread(
            generate_audio,
            processed_text,
            history_prompt=speaker,
        )

        # Convert numpy array to WAV bytes
        import numpy as np
        audio_int16 = (audio_array * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            f.seek(0)
            wav_bytes = Path(f.name).read_bytes()
            os.unlink(f.name)

        voice_logger.info(f"Bark generated {len(wav_bytes)} bytes")
        return wav_bytes

    except Exception as e:
        voice_logger.error(f"Bark generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  7. EDGE-TTS SYNTHESIS (Fast Fallback)
# ═══════════════════════════════════════════════════════════════

def _select_edge_voice(
    language: str, emotion: str, gender: str = "m",
) -> str:
    """Select optimal edge-tts voice for context."""
    lang = "ru" if language in ("russian", "ru") else (
        "tr" if language in ("turkish", "tr") else "en"
    )
    g = gender if gender in ("m", "f") else "m"

    warm_emotions = {"love", "tenderness", "joy", "playful"}
    assertive_emotions = {"anger", "frustration", "confident", "contempt", "sarcastic"}

    if emotion in warm_emotions:
        key = f"{lang}_{g}_warm"
    elif emotion in assertive_emotions:
        key = f"{lang}_{g}_assertive"
    else:
        key = f"{lang}_{g}_casual"

    return EDGE_VOICES.get(key, EDGE_VOICES.get(f"{lang}_{g}_casual", "en-US-GuyNeural"))


async def generate_edge_audio(
    text: str,
    emotion: str = "neutral",
    language: str = "en",
    gender: str = "m",
    rate_override: Optional[str] = None,
    pitch_override: Optional[str] = None,
    volume_override: Optional[str] = None,
) -> Optional[bytes]:
    """Generate audio using Edge-TTS with emotion-aware prosody control."""
    if not _check_edge_tts():
        return None

    try:
        import edge_tts

        voice = _select_edge_voice(language, emotion, gender)
        processed_text = _preprocess_for_speech(text, emotion)

        # Apply emotion prosody
        prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
        rate = rate_override or f"{int((prosody['rate'] - 1) * 100):+d}%"
        pitch = pitch_override or prosody["pitch"]
        volume = volume_override or prosody["volume"]

        voice_logger.info(
            f"Edge-TTS: voice={voice}, emotion={emotion}, "
            f"rate={rate}, pitch={pitch}, volume={volume}"
        )

        communicate = edge_tts.Communicate(
            processed_text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

        # Generate to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        await communicate.save(tmp_path)
        audio_bytes = Path(tmp_path).read_bytes()
        os.unlink(tmp_path)

        voice_logger.info(f"Edge-TTS generated {len(audio_bytes)} bytes")
        return audio_bytes

    except Exception as e:
        voice_logger.error(f"Edge-TTS generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  7b. PER-CHAT VOICE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def _resolve_voice_reference(chat_id: int, profile: Dict[str, Any] = None) -> Optional[str]:
    """Resolve the best voice reference for a specific chat.

    Priority:
    1. Chat-specific voice (set via registerVoice for that chat)
    2. User's own voice (engine_data/voice/my_voice/)
    3. Partner's collected voice samples (from their voice messages)
    """
    # 1. Check chat-specific assigned voice
    if profile and profile.get("assigned_voice_path"):
        vp = profile["assigned_voice_path"]
        if os.path.exists(vp):
            return vp

    # 2. User's own voice (global — used for all chats unless overridden)
    user_ref = _get_user_reference()
    if user_ref:
        return user_ref

    # 3. Collected partner voice samples (for cloning their style)
    if chat_id:
        return _find_best_reference(chat_id)

    return None


def assign_voice_to_chat(chat_id: int, voice_path: str) -> bool:
    """Assign a specific voice reference to a chat (overrides global voice)."""
    if not os.path.exists(voice_path):
        return False
    profile = load_voice_profile(chat_id)
    profile["assigned_voice_path"] = voice_path
    save_voice_profile(chat_id, profile)
    voice_logger.info(f"Assigned voice {voice_path} to chat {chat_id}")
    return True


def list_available_voices() -> Dict[str, Any]:
    """List all available voice references organized by source."""
    voices = {"my_voice": [], "chat_voices": {}, "named_voices": []}

    # User's own voice files
    my_dir = VOICE_DATA_DIR / "my_voice"
    if my_dir.exists():
        for f in sorted(my_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix in (".wav", ".ogg", ".mp3", ".m4a"):
                voices["my_voice"].append({
                    "path": str(f), "name": f.stem, "format": f.suffix[1:],
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    # Chat-specific references
    if VOICE_REFS_DIR.exists():
        for chat_dir in VOICE_REFS_DIR.iterdir():
            if chat_dir.is_dir():
                files = []
                for f in sorted(chat_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if f.suffix in (".wav", ".ogg", ".mp3"):
                        files.append({"path": str(f), "name": f.stem, "format": f.suffix[1:]})
                if files:
                    voices["chat_voices"][chat_dir.name] = files

    # Named voice personas (engine_data/voice/personas/)
    personas_dir = VOICE_DATA_DIR / "personas"
    if personas_dir.exists():
        for f in sorted(personas_dir.glob("*")):
            if f.suffix in (".wav", ".ogg", ".mp3"):
                voices["named_voices"].append({
                    "path": str(f), "name": f.stem, "format": f.suffix[1:],
                })

    return voices


async def store_named_voice(audio_bytes: bytes, name: str) -> Optional[str]:
    """Store a named voice persona (e.g., 'default', 'deep_male', etc.)."""
    personas_dir = VOICE_DATA_DIR / "personas"
    personas_dir.mkdir(parents=True, exist_ok=True)

    ext = "ogg"
    if audio_bytes[:4] == b"RIFF":
        ext = "wav"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        ext = "mp3"

    ref_path = personas_dir / f"{name}.{ext}"
    try:
        ref_path.write_bytes(audio_bytes)
        voice_logger.info(f"Stored named voice persona: {name} -> {ref_path}")
        # Also convert to WAV if needed
        if ext == "ogg":
            await convert_ogg_to_wav(str(ref_path))
        return str(ref_path)
    except Exception as e:
        voice_logger.error(f"Failed to store named voice: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  8. UNIFIED SYNTHESIS API
# ═══════════════════════════════════════════════════════════════

async def synthesize_voice(
    text: str,
    chat_id: int = 0,
    emotion: str = "neutral",
    language: str = "auto",
    gender: str = "auto",
    backend: str = "auto",
    nlp_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Main voice synthesis entry point.

    Automatically selects backend, voice, and prosody based on:
    - Chat voice profile (stored preferences)
    - Detected emotion from NLP analysis
    - Language detection
    - Available backends

    Returns:
        {"audio": bytes, "format": "mp3"|"wav", "backend": str,
         "voice": str, "emotion": str, "duration_estimate_s": float}
    """
    start = time.time()

    # Load chat voice profile
    profile = load_voice_profile(chat_id) if chat_id else {}

    # Auto-detect language
    if language == "auto":
        if nlp_analysis and nlp_analysis.get("language"):
            language = nlp_analysis["language"]
        elif profile.get("language") not in (None, "auto"):
            language = profile["language"]
        else:
            # Simple heuristic: check for Cyrillic
            has_cyrillic = bool(re.search(r"[а-яА-ЯёЁ]", text))
            language = "russian" if has_cyrillic else "english"

    # Auto-detect gender
    if gender == "auto":
        gender = profile.get("gender", "m")
        if gender == "auto":
            gender = "m"  # Default

    # Auto-detect emotion from NLP
    if emotion == "neutral" and nlp_analysis:
        detected = nlp_analysis.get("sentiment", {}).get("sentiment", "neutral")
        ensemble = nlp_analysis.get("ensemble", {})
        if ensemble.get("primary_emotion", {}).get("value"):
            emotion = ensemble["primary_emotion"]["value"]
        elif detected in ("positive", "negative"):
            emotion = "joy" if detected == "positive" else "sadness"

    # Check cache — include chat_id to avoid cross-voice collisions
    # (different chats may have different voice references)
    cache_key = hashlib.md5(
        f"{text}:{emotion}:{language}:{gender}:{backend}:{chat_id}".encode()
    ).hexdigest()
    cache_path = VOICE_CACHE_DIR / f"{cache_key}.mp3"
    if cache_path.exists():
        voice_logger.info(f"Cache hit for voice: {cache_key[:8]}")
        return {
            "audio": cache_path.read_bytes(),
            "format": "mp3",
            "backend": "cache",
            "voice": "cached",
            "emotion": emotion,
            "duration_estimate_s": len(cache_path.read_bytes()) / 16000,
            "generation_time_s": 0,
        }

    # Select backend — priority:
    # For Russian: Fish Speech (if available) > Chatterbox > F5-TTS > Edge-TTS
    # For English: Chatterbox > Fish Speech > F5-TTS > Bark > Edge-TTS
    audio_bytes = None
    used_backend = "none"
    audio_format = "mp3"
    is_russian = language in ("russian", "ru")

    # Resolve voice reference for this specific chat
    ref_audio = _resolve_voice_reference(chat_id, profile)
    if ref_audio and ref_audio.endswith(".ogg"):
        wav_ref = await convert_ogg_to_wav(ref_audio)
        if wav_ref:
            ref_audio = wav_ref

    # 0. Fish Speech / OpenAudio S1 — Premium multilingual (best for Russian)
    if not audio_bytes and backend in ("fish", "fish-speech", "openaudio", "auto") and _check_fish_speech():
        audio_bytes = await generate_fish_speech_audio(
            text, reference_audio=ref_audio,
            language="ru" if is_russian else "en",
        )
        if audio_bytes:
            used_backend = "fish-speech" + (" (cloned)" if ref_audio else "")
            audio_format = "wav"

    # 1. Chatterbox Multilingual — SoTA voice cloning, 23 languages
    if not audio_bytes and backend in ("chatterbox", "clone", "auto") and _check_chatterbox():
        high_emotion = {"anger", "excitement", "joy", "surprise", "frustration"}
        low_emotion = {"sadness", "tired", "bored", "neutral"}
        exag = 0.7 if emotion in high_emotion else (0.3 if emotion in low_emotion else 0.5)
        audio_bytes = await generate_chatterbox_audio(
            text, reference_audio=ref_audio,
            language=language, exaggeration=exag,
        )
        if audio_bytes:
            used_backend = "chatterbox" + (" (cloned)" if ref_audio else "")
            audio_format = "wav"

    # 2. F5-TTS — Flow matching voice cloning (Apple Silicon optimized)
    if not audio_bytes and backend in ("f5", "f5-tts", "auto") and _check_f5tts():
        audio_bytes = await generate_f5tts_audio(
            text, reference_audio=ref_audio,
        )
        if audio_bytes:
            used_backend = "f5-tts" + (" (cloned)" if ref_audio else "")
            audio_format = "wav"

    # 3. Bark — natural but generic voice
    if not audio_bytes and backend in ("bark", "auto") and _check_bark():
        audio_bytes = await generate_bark_audio(text, emotion, language, gender)
        if audio_bytes:
            used_backend = "bark"
            audio_format = "wav"

    # 4. Edge-TTS — fast neural TTS (generic voice)
    if not audio_bytes and backend in ("edge", "auto"):
        audio_bytes = await generate_edge_audio(text, emotion, language, gender)
        if audio_bytes:
            used_backend = "edge-tts"
            audio_format = "mp3"

    if not audio_bytes:
        voice_logger.warning("All voice backends failed")
        return None

    # Cache result
    try:
        cache_path_actual = VOICE_CACHE_DIR / f"{cache_key}.{audio_format}"
        cache_path_actual.write_bytes(audio_bytes)
    except Exception:
        pass

    generation_time = time.time() - start
    voice_logger.info(
        f"Voice synthesized: backend={used_backend}, emotion={emotion}, "
        f"lang={language}, time={generation_time:.1f}s, size={len(audio_bytes)}"
    )

    return {
        "audio": audio_bytes,
        "format": audio_format,
        "backend": used_backend,
        "voice": _select_edge_voice(language, emotion, gender) if used_backend == "edge-tts" else "bark",
        "emotion": emotion,
        "duration_estimate_s": len(audio_bytes) / (16000 if audio_format == "wav" else 8000),
        "generation_time_s": round(generation_time, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  9. CONVERSATION-AWARE VOICE SELECTION
# ═══════════════════════════════════════════════════════════════

def select_voice_style(
    nlp_analysis: Dict[str, Any],
    conversation_stage: str = "flowing",
    mirroring_mode: str = "natural",
    time_of_day: str = "afternoon",
) -> Dict[str, str]:
    """Select voice style parameters based on conversation context.

    Returns dict with: emotion, rate_adjust, pitch_adjust, energy_level
    Used by synthesize_voice() to adapt prosody.
    """
    style = {
        "emotion": "neutral",
        "energy": "medium",
        "speed": "normal",
        "warmth": "medium",
    }

    # From NLP analysis
    sentiment = nlp_analysis.get("sentiment", {}).get("sentiment", "neutral")
    intensity = nlp_analysis.get("sentiment", {}).get("intensity", 0.5)
    stage = nlp_analysis.get("conversation_stage", conversation_stage)

    # Emotion mapping
    ensemble = nlp_analysis.get("ensemble", {})
    if ensemble.get("primary_emotion", {}).get("value"):
        style["emotion"] = ensemble["primary_emotion"]["value"]
    elif sentiment == "positive":
        style["emotion"] = "joy" if intensity > 0.6 else "playful"
    elif sentiment == "negative":
        style["emotion"] = "sadness" if intensity > 0.6 else "bored"

    # Stage-based adjustments
    if stage == "conflict":
        style["energy"] = "high"
        if style["emotion"] == "neutral":
            style["emotion"] = "frustration"
    elif stage == "flirting":
        style["warmth"] = "high"
        if style["emotion"] == "neutral":
            style["emotion"] = "playful"
    elif stage == "deep":
        style["speed"] = "slow"
        style["warmth"] = "high"
    elif stage == "cooling_down":
        style["energy"] = "low"
        style["speed"] = "slow"

    # Mirroring mode adjustments
    if mirroring_mode in ("aggressive_mirror", "hostile_mirror"):
        style["energy"] = "high"
        style["emotion"] = "anger" if mirroring_mode == "hostile_mirror" else "frustration"
    elif mirroring_mode == "warm_mirror":
        style["warmth"] = "high"
        style["emotion"] = "love" if style["emotion"] == "neutral" else style["emotion"]

    # Time of day
    if time_of_day in ("late_night", "night"):
        style["speed"] = "slow"
        style["energy"] = "low"
        if style["emotion"] == "neutral":
            style["emotion"] = "tired"

    return style


# ═══════════════════════════════════════════════════════════════
#  10. VOICE ENGINE STATUS
# ═══════════════════════════════════════════════════════════════

def get_voice_engine_status() -> Dict[str, Any]:
    """Get current voice engine capabilities and status."""
    my_voice = _get_user_reference()
    voices = list_available_voices()
    return {
        "available": _check_chatterbox() or _check_fish_speech() or _check_f5tts() or _check_edge_tts() or _check_bark(),
        "backends": {
            "fish_speech": {
                "available": _check_fish_speech(),
                "quality": "Premium multilingual TTS — 13 languages inc. Russian",
                "speed": "fast (RTF ~0.2, needs GPU)",
                "languages": "en, ru, zh, ja, de, fr, es, ko, ar, nl, it, pl, pt",
                "voice_cloning": True,
            },
            "chatterbox": {
                "available": _check_chatterbox(),
                "loaded": _chatterbox_model is not None,
                "multilingual": _chatterbox_is_multilingual,
                "quality": "SoTA voice cloning — zero-shot from 5-10s audio",
                "speed": "medium (3-10s per sentence, GPU/MPS accelerated)",
                "languages": "23 languages (en, ru, tr, fr, de, es, ja, zh, ...)" if _chatterbox_is_multilingual else "English",
                "voice_cloning": True,
            },
            "f5_tts": {
                "available": _check_f5tts(),
                "quality": "flow matching voice cloning — realistic zero-shot",
                "speed": "medium (5-15s, faster on Apple Silicon via MLX)",
                "voice_cloning": True,
                "apple_silicon_optimized": True,
            },
            "bark": {
                "available": _check_bark(),
                "loaded": _bark_models.get("loaded", False),
                "quality": "ultra-natural (laughs, sighs, hesitations)",
                "speed": "slow (5-15s per sentence)",
                "voice_cloning": False,
            },
            "edge_tts": {
                "available": _check_edge_tts(),
                "quality": "good neural TTS (generic voice)",
                "speed": "fast (<2s per sentence)",
                "voice_cloning": False,
            },
        },
        "voice_cloning": {
            "user_voice_registered": my_voice is not None,
            "user_voice_path": my_voice,
            "named_voices": len(voices.get("named_voices", [])),
            "chat_specific_voices": len(voices.get("chat_voices", {})),
            "how_to_register": (
                "Option 1: Send voice to Saved Messages, then /voice register saved\n"
                "Option 2: Place WAV/OGG in engine_data/voice/my_voice/\n"
                "Option 3: /voice register @chat to grab voice from a chat\n"
                "Option 4: /voice persona <name> to create a named voice persona"
            ),
        },
        "voices": voices,
        "supported_languages": ["english", "russian", "turkish"] + (
            ["+ 20 more via Chatterbox Multilingual"] if _chatterbox_is_multilingual else []
        ),
        "emotion_prosody_map": list(EMOTION_PROSODY.keys()),
        "cached_files": len(list(VOICE_CACHE_DIR.glob("*"))),
        "voice_profiles": len(list(VOICE_PROFILES_DIR.glob("*.json"))),
        "reference_samples": sum(
            len(list(d.glob("*"))) for d in VOICE_REFS_DIR.iterdir() if d.is_dir()
        ) if VOICE_REFS_DIR.exists() else 0,
    }


def warmup_voice_engine():
    """Pre-initialize voice backends for faster first synthesis."""
    voice_logger.info("Warming up voice engine...")
    _check_chatterbox()
    _check_f5tts()
    _check_edge_tts()
    if _check_bark():
        _load_bark()
    backends = get_voice_engine_status()["backends"]
    active = [k for k, v in backends.items() if v.get("available")]
    voice_logger.info(f"Voice engine ready: {', '.join(active) or 'edge-tts only'}")
