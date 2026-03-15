"""
Telegram Voice Call Engine
===========================
High-level call management that talks to the call_bridge.py subprocess.

The bridge runs in a Python 3.10 venv (tgcalls only has x86_64 macOS wheels for 3.10).
Communication is via localhost HTTP (default port 8770).

Supports both:
- PRIVATE CALLS (1:1 voice calls with users)
- GROUP CALLS (voice chats in groups/channels)

Usage from telegram_api.py:
    from call_engine import make_call, accept_call, hangup, speak_in_call, ...
"""

import asyncio
import base64
import io
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

call_logger = logging.getLogger("call_engine")

# ═══════════════════════════════════════════════════════════════
#  BRIDGE CONNECTION
# ═══════════════════════════════════════════════════════════════

BRIDGE_PORT = int(os.environ.get("CALL_BRIDGE_PORT", "8770"))
BRIDGE_URL = f"http://127.0.0.1:{BRIDGE_PORT}"
_bridge_process: Optional[subprocess.Popen] = None
_bridge_healthy = False


async def _bridge_request(method: str, path: str, body: dict = None, timeout: float = 30) -> dict:
    """Make an HTTP request to the call bridge."""
    url = f"{BRIDGE_URL}{path}"
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                resp = await client.get(url, timeout=timeout)
            else:
                resp = await client.post(url, json=body or {}, timeout=timeout)
            return resp.json()
    except httpx.ConnectError:
        return {"success": False, "error": "Call bridge not running. Start it with: .venv-calls/bin/python call_bridge.py"}
    except Exception as e:
        return {"success": False, "error": f"Bridge request failed: {e}"}


async def check_bridge_status() -> Dict[str, Any]:
    """Check if the call bridge is running and healthy."""
    global _bridge_healthy
    result = await _bridge_request("GET", "/status")
    _bridge_healthy = result.get("ok", False)
    return result


async def start_bridge() -> Dict[str, Any]:
    """Start the call bridge subprocess if not already running."""
    global _bridge_process

    # Check if already running
    status = await check_bridge_status()
    if status.get("ok"):
        return {"success": True, "message": "Bridge already running", **status}

    # Find the Python 3.10 venv
    venv_python = Path(__file__).parent / ".venv-calls" / "bin" / "python"
    bridge_script = Path(__file__).parent / "call_bridge.py"

    if not venv_python.exists():
        return {
            "success": False,
            "error": f"Python 3.10 venv not found at {venv_python}. "
                     "Create it: uv venv --python 3.10 .venv-calls && "
                     "uv pip install --python .venv-calls/bin/python tgcalls==2.0.0 telethon python-dotenv",
        }

    if not bridge_script.exists():
        return {"success": False, "error": f"call_bridge.py not found at {bridge_script}"}

    try:
        _bridge_process = subprocess.Popen(
            [str(venv_python), str(bridge_script)],
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        call_logger.info(f"Started call bridge (PID {_bridge_process.pid})")

        # Wait for it to become healthy
        for _ in range(20):
            await asyncio.sleep(0.5)
            status = await check_bridge_status()
            if status.get("ok"):
                return {"success": True, "message": "Bridge started", **status}

        return {"success": False, "error": "Bridge started but not responding after 10s"}
    except Exception as e:
        return {"success": False, "error": f"Failed to start bridge: {e}"}


async def stop_bridge() -> Dict[str, Any]:
    """Stop the call bridge subprocess."""
    global _bridge_process, _bridge_healthy
    if _bridge_process:
        _bridge_process.terminate()
        try:
            _bridge_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bridge_process.kill()
        _bridge_process = None
    _bridge_healthy = False
    return {"success": True, "message": "Bridge stopped"}


# ═══════════════════════════════════════════════════════════════
#  CALL STATE (local mirror of bridge state)
# ═══════════════════════════════════════════════════════════════

class CallState:
    """Local mirror of an active call's state."""

    def __init__(self, chat_id: int, direction: str = "outgoing"):
        self.chat_id = chat_id
        self.direction = direction
        self.status = "ringing"
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.audio_chunks_sent = 0
        self.audio_chunks_received = 0
        self.transcription_buffer: List[str] = []
        self.call_type = "private"  # "private" or "group"
        # Autonomy: bot listens, thinks, speaks on its own
        self.autonomy = False
        self._autonomy_task: Optional[asyncio.Task] = None
        self.autonomy_language = "auto"
        self.is_speaking = False  # prevents overlapping TTS

    @property
    def duration_s(self) -> float:
        if not self.started_at:
            return 0
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "direction": self.direction,
            "status": self.status,
            "duration_s": self.duration_s,
            "chunks_sent": self.audio_chunks_sent,
            "chunks_received": self.audio_chunks_received,
            "transcriptions": self.transcription_buffer[-5:],
            "call_type": self.call_type,
            "autonomy": self.autonomy,
        }


_active_calls: Dict[int, CallState] = {}


def get_active_calls() -> Dict[int, Dict]:
    return {cid: cs.to_dict() for cid, cs in _active_calls.items()}


def get_call_state(chat_id: int) -> Optional[CallState]:
    return _active_calls.get(chat_id)


# ═══════════════════════════════════════════════════════════════
#  AUDIO CONVERSION HELPERS
# ═══════════════════════════════════════════════════════════════

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 48000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def wav_to_pcm(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()


def resample_pcm(pcm_data: bytes, src_rate: int, dst_rate: int = 48000) -> bytes:
    if src_rate == dst_rate:
        return pcm_data
    samples = struct.unpack(f"<{len(pcm_data)//2}h", pcm_data)
    ratio = dst_rate / src_rate
    new_len = int(len(samples) * ratio)
    resampled = []
    for i in range(new_len):
        src_idx = i / ratio
        idx = int(src_idx)
        frac = src_idx - idx
        if idx + 1 < len(samples):
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = samples[min(idx, len(samples) - 1)]
        resampled.append(int(max(-32768, min(32767, val))))
    return struct.pack(f"<{len(resampled)}h", *resampled)


# ═══════════════════════════════════════════════════════════════
#  TTS FOR CALLS
# ═══════════════════════════════════════════════════════════════

async def text_to_call_audio(
    text: str, chat_id: int = 0, emotion: str = "neutral", language: str = "auto",
) -> Optional[bytes]:
    """Generate PCM 16-bit 48kHz mono audio for a call from text."""
    try:
        from voice_engine import synthesize_voice
        result = await synthesize_voice(
            text, chat_id=chat_id, emotion=emotion, language=language,
        )
        if not result or not result.get("audio"):
            return None

        audio_bytes = result["audio"]
        fmt = result.get("format", "wav")

        if fmt == "wav":
            pcm_data, src_rate, channels = wav_to_pcm(audio_bytes)
            if channels > 1:
                samples = struct.unpack(f"<{len(pcm_data)//2}h", pcm_data)
                mono = [int((samples[i] + samples[i+1]) / 2) for i in range(0, len(samples), 2)]
                pcm_data = struct.pack(f"<{len(mono)}h", *mono)
            return resample_pcm(pcm_data, src_rate, 48000)
        elif fmt == "mp3":
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
                return audio.raw_data
            except Exception:
                pass
            from voice_engine import convert_audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name
            wav_path = await convert_audio(tmp, "wav", 48000, 1)
            if wav_path:
                pcm_data, _, _ = wav_to_pcm(Path(wav_path).read_bytes())
                os.unlink(tmp)
                os.unlink(wav_path)
                return pcm_data
            os.unlink(tmp)
    except Exception as e:
        call_logger.error(f"TTS for call failed: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
#  STT FOR CALLS
# ═══════════════════════════════════════════════════════════════

async def call_audio_to_text(pcm_data: bytes, language: str = None) -> Optional[str]:
    """Transcribe PCM 16-bit 48kHz audio from a call."""
    try:
        wav_bytes = pcm_to_wav(pcm_data, sample_rate=48000, channels=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base", compute_type="int8")
            segments, info = model.transcribe(tmp, language=language)
            text = " ".join(seg.text for seg in segments).strip()
            if text:
                return text
        except ImportError:
            call_logger.debug("faster-whisper not available for call STT")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    except Exception as e:
        call_logger.error(f"STT for call failed: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
#  PRIVATE CALL MANAGEMENT (via bridge)
# ═══════════════════════════════════════════════════════════════

async def make_call(
    tg_client,
    chat_id: int,
    initial_message: str = "",
    on_audio_received: Optional[Callable] = None,
    call_type: str = "private",
) -> Dict[str, Any]:
    """Initiate a voice call (private or group).

    Args:
        tg_client: Telethon client (unused — bridge has its own)
        chat_id: User ID (private) or chat ID (group)
        initial_message: First thing to say when connected
        on_audio_received: Callback for incoming audio
        call_type: "private" or "group"
    """
    if chat_id in _active_calls:
        return {"success": False, "error": f"Already in a call with {chat_id}"}

    # Ensure bridge is running
    status = await check_bridge_status()
    if not status.get("ok"):
        start_result = await start_bridge()
        if not start_result.get("success"):
            return start_result

    state = CallState(chat_id, "outgoing")
    state.call_type = call_type
    _active_calls[chat_id] = state

    # Route to private or group call
    if call_type == "group":
        result = await _bridge_request("POST", "/call/group/join", {
            "chat_id": chat_id,
        })
    else:
        result = await _bridge_request("POST", "/call/make", {
            "user_id": chat_id,
        })

    if result.get("success"):
        state.status = "ringing"
        state.started_at = time.time()

        # Send initial message audio
        if initial_message:
            asyncio.create_task(_send_speech(chat_id, initial_message))

        return {
            "success": True,
            "message": f"{'Group' if call_type == 'group' else 'Private'} call initiated with {chat_id}",
            "call_type": call_type,
            **{k: v for k, v in result.items() if k != "success"},
        }
    else:
        _active_calls.pop(chat_id, None)
        return result


async def accept_call(tg_client, chat_id: int) -> Dict[str, Any]:
    """Accept an incoming call."""
    result = await _bridge_request("POST", "/call/accept", {"user_id": chat_id})
    if result.get("success"):
        state = _active_calls.get(chat_id)
        if state:
            state.status = "active"
            state.started_at = time.time()
        else:
            state = CallState(chat_id, "incoming")
            state.status = "active"
            state.started_at = time.time()
            _active_calls[chat_id] = state
    return result


async def decline_call(tg_client, chat_id: int) -> Dict[str, Any]:
    """Decline an incoming call."""
    result = await _bridge_request("POST", "/call/decline", {"user_id": chat_id})
    state = _active_calls.pop(chat_id, None)
    if state:
        state.status = "ended"
        state.ended_at = time.time()
    return result


async def hang_up(tg_client, chat_id: int) -> Dict[str, Any]:
    """Hang up an active call (private or group)."""
    state = _active_calls.get(chat_id)
    call_type = state.call_type if state else "private"

    if call_type == "group":
        result = await _bridge_request("POST", "/call/group/leave", {"chat_id": chat_id})
    else:
        result = await _bridge_request("POST", "/call/hangup", {"user_id": chat_id})

    state = _active_calls.pop(chat_id, None)
    if state:
        state.status = "ended"
        state.ended_at = time.time()
        result["duration_s"] = state.duration_s

    return result


async def speak_in_call(
    chat_id: int, text: str, emotion: str = "neutral", language: str = "auto",
) -> Dict[str, Any]:
    """Generate TTS and send audio to an active call."""
    state = _active_calls.get(chat_id)
    if not state or state.status not in ("active", "ringing"):
        return {"success": False, "error": "No active call with this chat"}

    pcm = await text_to_call_audio(text, chat_id, emotion, language)
    if not pcm:
        return {"success": False, "error": "TTS generation failed"}

    # Send PCM to bridge
    pcm_b64 = base64.b64encode(pcm).decode()
    result = await _bridge_request("POST", "/call/audio", {
        "user_id": chat_id,
        "pcm_base64": pcm_b64,
    })

    if result.get("success"):
        state.audio_chunks_sent += 1

    return {
        "success": result.get("success", False),
        "message": f"Queued speech: '{text[:50]}...'",
        **{k: v for k, v in result.items() if k not in ("success",)},
    }


async def _send_speech(chat_id: int, text: str):
    """Background task to send initial speech."""
    try:
        await speak_in_call(chat_id, text)
    except Exception as e:
        call_logger.error(f"Initial speech failed: {e}")


async def listen_in_call(chat_id: int) -> Dict[str, Any]:
    """Get transcription of what the other party is saying."""
    result = await _bridge_request("GET", f"/audio/{chat_id}")
    if not result.get("success"):
        return result

    pcm_b64 = result.get("pcm_base64", "")
    if not pcm_b64:
        return {"success": True, "text": "", "message": "No audio recorded yet"}

    pcm = base64.b64decode(pcm_b64)
    if len(pcm) < 4800:  # Less than 50ms
        return {"success": True, "text": "", "message": "Audio too short"}

    text = await call_audio_to_text(pcm)
    state = _active_calls.get(chat_id)
    if text and state:
        state.transcription_buffer.append(text)
        state.audio_chunks_received += 1

    return {
        "success": True,
        "text": text or "",
        "audio_duration_s": result.get("duration_s", 0),
    }


# ═══════════════════════════════════════════════════════════════
#  GROUP CALL SUPPORT (via bridge)
# ═══════════════════════════════════════════════════════════════

async def join_group_call(
    tg_client, chat_id: int, initial_message: str = "",
) -> Dict[str, Any]:
    """Join a group voice chat."""
    return await make_call(
        tg_client, chat_id,
        initial_message=initial_message,
        call_type="group",
    )


async def leave_group_call(tg_client, chat_id: int) -> Dict[str, Any]:
    """Leave a group voice chat."""
    return await hang_up(tg_client, chat_id)


# ═══════════════════════════════════════════════════════════════
#  AUTONOMOUS CALL MODE
#  Bot listens → transcribes → AI generates response → speaks
#  Full conversational autonomy without human intervention
# ═══════════════════════════════════════════════════════════════

# Silence detection: minimum PCM bytes for ~1.5s of speech at 48kHz 16-bit mono
_MIN_SPEECH_BYTES = 48000 * 2 * 1.5  # ~144,000 bytes
# Poll interval for checking new audio (seconds)
_AUTONOMY_POLL_INTERVAL = 1.5
# Silence threshold: if no new audio for this many polls, consider turn over
_SILENCE_POLLS_FOR_TURN = 2
# Maximum conversation history kept for AI context
_MAX_CALL_HISTORY = 20


async def _autonomy_loop(chat_id: int):
    """Background loop: listen → transcribe → AI reply → speak.

    Runs continuously while autonomy is enabled for a call.
    Uses voice activity detection via audio buffer size changes
    to detect when the other person stops talking (turn-taking).
    """
    state = _active_calls.get(chat_id)
    if not state:
        return

    call_logger.info(f"Autonomy loop started for {chat_id}")
    consecutive_silence = 0
    pending_audio_size = 0
    call_conversation: List[Dict[str, str]] = []  # {"role": "them"/"me", "text": "..."}

    while state.autonomy and state.status in ("active", "ringing"):
        try:
            await asyncio.sleep(_AUTONOMY_POLL_INTERVAL)

            # Skip if we're currently speaking (don't listen to our own TTS)
            if state.is_speaking:
                consecutive_silence = 0
                continue

            # Check if call is still active
            if chat_id not in _active_calls:
                break

            # Fetch recorded audio from bridge
            result = await _bridge_request("GET", f"/audio/{chat_id}")
            if not result.get("success"):
                continue

            pcm_b64 = result.get("pcm_base64", "")
            audio_len = result.get("length_bytes", 0)

            if not pcm_b64 or audio_len < 1000:
                consecutive_silence += 1
                # If we've had silence after speech, the person is done talking
                if consecutive_silence >= _SILENCE_POLLS_FOR_TURN and pending_audio_size > 0:
                    # They stopped talking but we already processed — reset
                    pending_audio_size = 0
                continue

            # We have audio — decode it
            pcm = base64.b64decode(pcm_b64)

            if len(pcm) < _MIN_SPEECH_BYTES:
                # Too short to be meaningful speech
                consecutive_silence += 1
                continue

            # Reset silence counter — they're talking
            consecutive_silence = 0
            pending_audio_size = len(pcm)

            # Wait a bit more to let them finish their sentence
            await asyncio.sleep(1.0)

            # Grab any additional audio that came in
            result2 = await _bridge_request("GET", f"/audio/{chat_id}")
            if result2.get("success") and result2.get("pcm_base64"):
                extra = base64.b64decode(result2["pcm_base64"])
                if len(extra) > 1000:
                    pcm += extra

            # Transcribe what they said
            their_text = await call_audio_to_text(pcm, language=state.autonomy_language)
            if not their_text or len(their_text.strip()) < 2:
                continue

            call_logger.info(f"[CALL {chat_id}] Them: {their_text[:80]}")
            state.transcription_buffer.append(their_text)
            state.audio_chunks_received += 1

            # Add to conversation history
            call_conversation.append({"role": "them", "text": their_text})
            if len(call_conversation) > _MAX_CALL_HISTORY:
                call_conversation = call_conversation[-_MAX_CALL_HISTORY:]

            # Generate AI response using the same pipeline as auto-reply
            ai_reply = await _generate_call_response(chat_id, their_text, call_conversation)
            if not ai_reply:
                continue

            call_logger.info(f"[CALL {chat_id}] Me: {ai_reply[:80]}")
            call_conversation.append({"role": "me", "text": ai_reply})

            # Speak the response
            state.is_speaking = True
            try:
                await speak_in_call(chat_id, ai_reply, language=state.autonomy_language)
            finally:
                # Give time for TTS to finish playing before listening again
                # Estimate: ~150ms per word
                word_count = len(ai_reply.split())
                speak_duration = max(1.0, word_count * 0.15)
                await asyncio.sleep(speak_duration)
                state.is_speaking = False

        except asyncio.CancelledError:
            break
        except Exception as e:
            call_logger.error(f"Autonomy loop error for {chat_id}: {e}")
            await asyncio.sleep(2)

    call_logger.info(f"Autonomy loop ended for {chat_id}")


async def _generate_call_response(
    chat_id: int, their_text: str, conversation: List[Dict[str, str]],
) -> Optional[str]:
    """Generate an AI response for a live call.

    Uses the same Anthropic API as auto-reply but with a call-specific system prompt.
    Falls back to a simpler approach if the full pipeline isn't available.
    """
    # Try using the main app's generate_reply (same AI as auto-reply)
    try:
        from telegram_api import generate_reply
        reply = await generate_reply(
            chat_id,
            their_text,
            extra_system_prompt=_build_call_system_prompt(conversation),
            max_tokens_override=150,  # Keep responses concise for voice
            temperature_override=0.8,
        )
        if reply:
            return _clean_for_speech(reply)
    except Exception as e:
        call_logger.debug(f"generate_reply failed, trying direct API: {e}")

    # Fallback: call Anthropic directly with conversation context
    try:
        import httpx as _httpx
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            return None

        messages = []
        for turn in conversation[-10:]:
            role = "user" if turn["role"] == "them" else "assistant"
            messages.append({"role": role, "content": turn["text"]})

        # Ensure messages alternate correctly
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": their_text})

        async with _httpx.AsyncClient() as hc:
            resp = await hc.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 150,
                    "temperature": 0.8,
                    "system": _build_call_system_prompt(conversation),
                    "messages": messages,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("content", [{}])[0].get("text", "")
                return _clean_for_speech(text) if text else None
    except Exception as e:
        call_logger.error(f"Direct API call failed: {e}")

    return None


def _build_call_system_prompt(conversation: List[Dict[str, str]]) -> str:
    """Build a system prompt tailored for live voice calls."""
    turn_count = len(conversation)
    return (
        "You are on a live voice call. This is a REAL-TIME phone conversation.\n\n"
        "CRITICAL RULES FOR VOICE CALLS:\n"
        "- Speak naturally like a real person on a phone call\n"
        "- Keep responses SHORT (1-3 sentences max) — this is spoken, not written\n"
        "- Use casual, conversational language (contractions, filler words ok)\n"
        "- React naturally: laugh, express surprise, show emotion\n"
        "- Never use emojis, markdown, bullet points, or formatting\n"
        "- Never say 'as an AI' or break character\n"
        "- Match the energy and language of the other person\n"
        "- If they speak Russian, respond in Russian. Same for any language.\n"
        "- Use natural turn-taking — don't monologue\n"
        "- It's ok to use filler like 'hmm', 'yeah', 'oh really?'\n"
        "- Sound warm, engaged, and genuinely interested\n"
        f"- This call has been going for {turn_count} exchanges\n"
        + (
            "\n- The call just started, be warm and natural with a greeting"
            if turn_count <= 1 else ""
        )
    )


def _clean_for_speech(text: str) -> str:
    """Clean AI text for TTS — remove things that sound bad when spoken."""
    import re
    # Remove emojis
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
        "", text,
    )
    # Remove markdown formatting
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)  # **bold** / *italic*
    text = re.sub(r"_+([^_]+)_+", r"\1", text)
    text = re.sub(r"`[^`]+`", "", text)  # code blocks
    text = re.sub(r"#{1,6}\s", "", text)  # headers
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Limit length for voice (TTS gets slow with long text)
    if len(text) > 500:
        # Cut at last sentence boundary
        for sep in [". ", "! ", "? "]:
            idx = text[:500].rfind(sep)
            if idx > 100:
                text = text[:idx + 1]
                break
        else:
            text = text[:500]
    return text


async def enable_autonomy(
    chat_id: int, language: str = "auto",
) -> Dict[str, Any]:
    """Enable autonomous mode for an active call.

    The bot will listen, transcribe, generate AI responses, and speak
    automatically — full conversational autonomy.
    """
    state = _active_calls.get(chat_id)
    if not state:
        return {"success": False, "error": "No active call with this chat"}

    if state.autonomy:
        return {"success": True, "message": "Autonomy already enabled"}

    state.autonomy = True
    state.autonomy_language = language
    state._autonomy_task = asyncio.create_task(_autonomy_loop(chat_id))
    call_logger.info(f"Autonomy ENABLED for call with {chat_id}")
    return {
        "success": True,
        "message": f"Autonomy enabled for call with {chat_id}. "
                   "Bot will now listen, think, and speak on its own.",
        "language": language,
    }


async def disable_autonomy(chat_id: int) -> Dict[str, Any]:
    """Disable autonomous mode — bot stops auto-responding in the call."""
    state = _active_calls.get(chat_id)
    if not state:
        return {"success": False, "error": "No active call with this chat"}

    state.autonomy = False
    if state._autonomy_task and not state._autonomy_task.done():
        state._autonomy_task.cancel()
        state._autonomy_task = None
    call_logger.info(f"Autonomy DISABLED for call with {chat_id}")
    return {"success": True, "message": f"Autonomy disabled for call with {chat_id}"}


async def set_call_autonomy(
    chat_id: int, enabled: bool, language: str = "auto",
) -> Dict[str, Any]:
    """Toggle call autonomy on or off."""
    if enabled:
        return await enable_autonomy(chat_id, language)
    return await disable_autonomy(chat_id)


# ═══════════════════════════════════════════════════════════════
#  AUTO-ACCEPT INCOMING CALLS (optional)
# ═══════════════════════════════════════════════════════════════

_auto_accept_calls = False
_auto_accept_autonomy = False


def set_auto_accept(enabled: bool, with_autonomy: bool = False):
    """Configure auto-accept for incoming calls."""
    global _auto_accept_calls, _auto_accept_autonomy
    _auto_accept_calls = enabled
    _auto_accept_autonomy = with_autonomy
    call_logger.info(
        f"Auto-accept: {'ON' if enabled else 'OFF'}"
        f"{' (with autonomy)' if with_autonomy else ''}"
    )


def get_auto_accept_config() -> Dict[str, Any]:
    return {
        "auto_accept": _auto_accept_calls,
        "with_autonomy": _auto_accept_autonomy,
    }


# ═══════════════════════════════════════════════════════════════
#  INCOMING CALL NOTIFICATION (called by telegram_api.py)
# ═══════════════════════════════════════════════════════════════

def register_incoming_call(user_id: int):
    """Register an incoming call detected by the bridge.

    If auto-accept is enabled, automatically accepts and optionally
    enables autonomy so the bot handles the call entirely on its own.
    """
    if user_id not in _active_calls:
        state = CallState(user_id, "incoming")
        state.call_type = "private"
        _active_calls[user_id] = state
        call_logger.info(f"Incoming call registered from {user_id}")

        # Auto-accept if configured
        if _auto_accept_calls:
            asyncio.ensure_future(_auto_accept_and_go(user_id))


async def _auto_accept_and_go(user_id: int):
    """Auto-accept an incoming call and optionally enable autonomy."""
    try:
        await asyncio.sleep(1.0)  # Brief delay — feels more natural
        result = await _bridge_request("POST", "/call/accept", {"user_id": user_id})
        if result.get("success"):
            state = _active_calls.get(user_id)
            if state:
                state.status = "active"
                state.started_at = time.time()
            call_logger.info(f"Auto-accepted call from {user_id}")

            if _auto_accept_autonomy:
                await asyncio.sleep(0.5)
                await enable_autonomy(user_id)
                # Say hello
                await speak_in_call(user_id, "Hey! What's up?")
    except Exception as e:
        call_logger.error(f"Auto-accept failed for {user_id}: {e}")


# ═══════════════════════════════════════════════════════════════
#  AVAILABILITY CHECK
# ═══════════════════════════════════════════════════════════════

def check_call_support() -> Dict[str, Any]:
    """Check if call infrastructure is available."""
    venv_python = Path(__file__).parent / ".venv-calls" / "bin" / "python"
    bridge_script = Path(__file__).parent / "call_bridge.py"

    result = {
        "available": False,
        "backend": None,
        "error": None,
        "private_calls": False,
        "group_calls": False,
        "bridge_venv": venv_python.exists(),
        "bridge_script": bridge_script.exists(),
    }

    if not venv_python.exists():
        result["error"] = (
            "Python 3.10 venv not found. Create it:\n"
            "  uv venv --python 3.10 .venv-calls\n"
            "  uv pip install --python .venv-calls/bin/python tgcalls==2.0.0 telethon python-dotenv"
        )
        return result

    if not bridge_script.exists():
        result["error"] = "call_bridge.py not found"
        return result

    result["available"] = True
    result["backend"] = "tgcalls-bridge"
    result["private_calls"] = True
    result["group_calls"] = True
    return result


def get_call_engine_status() -> Dict[str, Any]:
    """Get call engine status and capabilities."""
    support = check_call_support()
    return {
        **support,
        "bridge_healthy": _bridge_healthy,
        "active_calls": len(_active_calls),
        "calls": get_active_calls(),
        "auto_accept": get_auto_accept_config(),
    }
