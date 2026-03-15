#!/usr/bin/env python3
"""
Subprocess whisper transcription worker.

Runs faster-whisper (CTranslate2) in a completely isolated process to avoid
the OpenMP library conflict with PyTorch on macOS x86_64.

Called by media_ai.transcribe_voice() when torch is already loaded.
Communicates via JSON on stdout. Errors go to stderr.

Usage:
    python _whisper_subprocess.py <audio_path> [language]
"""

import os
import sys
import json
import time

# Ensure clean environment — no torch contamination
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")


def transcribe(audio_path: str, language: str | None = None) -> dict:
    """Run transcription and return result dict."""
    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="cpu", compute_type="int8")
        start = time.time()

        kwargs = {"beam_size": 5, "best_of": 3, "vad_filter": True}
        if language:
            kwargs["language"] = language

        segments, info = model.transcribe(audio_path, **kwargs)
        text_parts = [seg.text.strip() for seg in segments]
        text = " ".join(text_parts)
        elapsed = time.time() - start

        return {
            "text": text,
            "language": info.language,
            "confidence": info.language_probability,
            "duration_seconds": round(info.duration, 1),
            "backend": "faster-whisper",
            "processing_time": round(elapsed, 2),
            "subprocess": True,
        }
    except ImportError:
        pass
    except Exception as e:
        print(f"faster-whisper failed: {e}", file=sys.stderr)

    # Fall back to openai-whisper
    try:
        import whisper

        model = whisper.load_model("small", device="cpu")
        start = time.time()

        options = {}
        if language:
            options["language"] = language
        result = model.transcribe(audio_path, **options)

        text = result["text"].strip()
        segs = result.get("segments", [])
        duration = segs[-1]["end"] if segs else 0
        elapsed = time.time() - start

        return {
            "text": text,
            "language": result.get("language", "unknown"),
            "confidence": 0.85,
            "duration_seconds": round(duration, 1),
            "backend": "whisper",
            "processing_time": round(elapsed, 2),
            "subprocess": True,
        }
    except ImportError:
        pass
    except Exception as e:
        print(f"openai-whisper failed: {e}", file=sys.stderr)

    return {
        "text": "",
        "language": "unknown",
        "confidence": 0.0,
        "duration_seconds": 0,
        "backend": "unavailable",
        "error": "No whisper backend available in subprocess",
        "subprocess": True,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "unavailable",
            "error": "Usage: _whisper_subprocess.py <audio_path> [language]",
            "subprocess": True,
        }))
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(audio_path):
        print(json.dumps({
            "text": "",
            "language": "unknown",
            "confidence": 0.0,
            "duration_seconds": 0,
            "backend": "unavailable",
            "error": f"File not found: {audio_path}",
            "subprocess": True,
        }))
        sys.exit(0)  # Exit 0 — valid JSON response even on error

    result = transcribe(audio_path, language)
    print(json.dumps(result))
