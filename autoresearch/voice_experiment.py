"""
Autoresearch Voice Parameter Optimization.

Runs grid search over Chatterbox TTS parameters to find the optimal
combination for voice cloning quality. Uses proxy metrics since
automated MOS (Mean Opinion Score) is not available.

Proxy metrics:
1. Whisper intelligibility — WER on generated audio vs input text
2. Duration ratio — actual WPM vs target (150 EN, 120 RU)
3. Generation success — did the model produce valid audio?

Usage:
    uv run python -m autoresearch.voice_experiment
    uv run python -m autoresearch.voice_experiment --quick  # Fewer combinations
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.config import (
    VOICE_PARAM_GRID, VOICE_TEST_PHRASES_EN, VOICE_TEST_PHRASES_RU,
    VOICE_DATA_DIR,
)

logger = logging.getLogger("autoresearch.voice")

OPTIMAL_PARAMS_PATH = VOICE_DATA_DIR / "optimal_params.json"


def _get_whisper_wer(audio_path: str, expected_text: str, language: str = "ru") -> float:
    """Compute Word Error Rate using faster-whisper.

    Returns WER (0.0 = perfect, 1.0+ = bad).
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language=language)
        transcription = " ".join(s.text.strip() for s in segments).strip()

        if not transcription:
            return 1.0

        # Simple WER: word-level edit distance / reference length
        ref_words = expected_text.lower().split()
        hyp_words = transcription.lower().split()

        if not ref_words:
            return 0.0

        # Levenshtein on word level
        n, m = len(ref_words), len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        return dp[n][m] / max(n, 1)
    except Exception as e:
        logger.debug(f"Whisper WER failed: {e}")
        return 1.0


def _get_duration_score(audio_path: str, text: str, target_wpm: float = 135) -> float:
    """Score based on how close the generated speech duration is to natural WPM.

    Returns score 0.0-1.0 (1.0 = perfect WPM match).
    """
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, timeout=5,
        )
        duration = float(result.stdout.decode().strip())
        if duration <= 0:
            return 0.0

        word_count = len(text.split())
        actual_wpm = (word_count / duration) * 60

        # Score: 1.0 at target_wpm, decreasing as we deviate
        ratio = actual_wpm / target_wpm
        score = max(0, 1.0 - abs(1.0 - ratio))
        return score
    except Exception:
        return 0.0


async def _test_params(
    cfg_weight: float,
    exaggeration: float,
    temperature: float,
    repetition_penalty: float,
    phrases: List[Tuple[str, str]],  # (text, lang)
    reference_audio: Optional[str] = None,
) -> Dict[str, float]:
    """Test a parameter combination on all phrases.

    Returns dict with avg_wer, avg_duration_score, success_rate, composite.
    """
    from voice_engine import generate_chatterbox_audio, _load_chatterbox

    model = _load_chatterbox()
    if model is None:
        return {"composite": 0.0, "error": "model_not_loaded"}

    # Temporarily remove optimal_params.json so the experiment's params are used
    # (otherwise generate_chatterbox_audio overrides with previously saved optimal)
    _backup_path = None
    if OPTIMAL_PARAMS_PATH.exists():
        _backup_path = OPTIMAL_PARAMS_PATH.with_suffix(".json.bak")
        OPTIMAL_PARAMS_PATH.rename(_backup_path)

    wers = []
    dur_scores = []
    successes = 0

    try:
        for text, lang in phrases:
            tmp_path = None
            try:
                # Generate audio with these params (including temperature and rep penalty
                # via generate_kwargs override)
                wav_bytes = await generate_chatterbox_audio(
                    text=text,
                    reference_audio=reference_audio,
                    language=lang,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

                if not wav_bytes or len(wav_bytes) < 1000:
                    continue

                successes += 1

                # Save to temp file for evaluation
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav_bytes)
                    tmp_path = f.name

                # Measure quality
                target_wpm = 120 if lang == "ru" else 150
                dur_score = _get_duration_score(tmp_path, text, target_wpm)
                dur_scores.append(dur_score)

                # Compute WER with correct language
                wer = _get_whisper_wer(tmp_path, text, language=lang)
                wers.append(wer)
            except Exception as e:
                logger.debug(f"Generation failed for '{text[:30]}': {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    finally:
        # Restore optimal_params.json backup
        if _backup_path and _backup_path.exists():
            _backup_path.rename(OPTIMAL_PARAMS_PATH)

    n = len(phrases)
    success_rate = successes / max(n, 1)
    avg_dur = sum(dur_scores) / max(len(dur_scores), 1)
    avg_wer = sum(wers) / max(len(wers), 1)
    intelligibility = max(0, 1.0 - avg_wer)  # 1.0 = perfect

    # Composite: weighted combination
    composite = success_rate * 0.3 + avg_dur * 0.3 + intelligibility * 0.4

    return {
        "success_rate": round(success_rate, 4),
        "avg_duration_score": round(avg_dur, 4),
        "avg_wer": round(avg_wer, 4),
        "intelligibility": round(intelligibility, 4),
        "composite": round(composite, 4),
        "n_tested": n,
        "n_success": successes,
    }


async def run_voice_experiments(quick: bool = False):
    """Run voice parameter optimization.

    Args:
        quick: If True, test fewer combinations (faster but less thorough).
    """
    # Get reference audio
    from voice_engine import _get_user_reference
    ref = _get_user_reference()
    if not ref:
        logger.warning("No reference audio found. Skipping voice experiments.")
        return None

    logger.info(f"Using reference audio: {ref}")

    # Build phrases list
    phrases = []
    n_en = 2 if quick else 5
    n_ru = 2 if quick else 5
    for text in VOICE_TEST_PHRASES_EN[:n_en]:
        phrases.append((text, "en"))
    for text in VOICE_TEST_PHRASES_RU[:n_ru]:
        phrases.append((text, "ru"))

    # Build parameter grid
    grid = VOICE_PARAM_GRID
    if quick:
        # Subset of grid for quick testing
        grid = {
            "cfg_weight": [0.3, 0.4],
            "exaggeration": [0.3, 0.4],
            "temperature": [0.7, 0.8],
            "repetition_penalty": [1.8, 2.0],
        }

    combinations = list(product(
        grid["cfg_weight"],
        grid["exaggeration"],
        grid["temperature"],
        grid["repetition_penalty"],
    ))

    logger.info(f"Testing {len(combinations)} parameter combinations on {len(phrases)} phrases")

    best_score = 0.0
    best_params = None
    results = []

    for i, (cfg, exag, temp, rep) in enumerate(combinations):
        logger.info(f"  [{i+1}/{len(combinations)}] cfg={cfg}, exag={exag}, temp={temp}, rep={rep}")

        metrics = await _test_params(
            cfg_weight=cfg,
            exaggeration=exag,
            temperature=temp,
            repetition_penalty=rep,
            phrases=phrases,
            reference_audio=ref,
        )

        params = {
            "cfg_weight": cfg,
            "exaggeration": exag,
            "temperature": temp,
            "repetition_penalty": rep,
        }
        result = {"params": params, "metrics": metrics}
        results.append(result)

        composite = metrics.get("composite", 0)
        if composite > best_score:
            best_score = composite
            best_params = params
            logger.info(f"    New best: {composite:.4f}")

    # Save optimal params
    if best_params:
        OPTIMAL_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "optimal_params": best_params,
            "best_score": best_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_combinations_tested": len(combinations),
            "n_phrases": len(phrases),
        }
        with open(OPTIMAL_PARAMS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved optimal voice params: {best_params} (score={best_score:.4f})")

    return {"best_params": best_params, "best_score": best_score, "n_tested": len(combinations)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    quick = "--quick" in sys.argv
    result = asyncio.run(run_voice_experiments(quick=quick))
    if result:
        print(f"\nBest params: {json.dumps(result['best_params'], indent=2)}")
        print(f"Best score: {result['best_score']:.4f}")
