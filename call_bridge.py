#!/usr/bin/env python3
"""
Telegram Call Bridge — Runs in Python 3.10 venv
================================================
Standalone HTTP server that handles Telegram voice calls via tgcalls (MarshalX).
Communicates with the main Python 3.12 app via localhost HTTP.

This exists because tgcalls only has macOS x86_64 wheels for Python 3.10,
while the main app runs Python 3.12.

Usage:
    .venv-calls/bin/python call_bridge.py

Listens on port 8770 by default (configurable via CALL_BRIDGE_PORT env var).
"""

import asyncio
import hashlib
import json
import logging
import os
import struct
import sys
import tempfile
import time
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

# Ensure we can find .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CALL-BRIDGE] %(levelname)s: %(message)s",
)
log = logging.getLogger("call_bridge")

# ─── tgcalls imports ─────────────────────────────────────────
try:
    from tgcalls import (
        NativeInstance,
        RawAudioDeviceDescriptor,
        FileAudioDeviceDescriptor,
        RtcServer,
    )
    TGCALLS_OK = True
    log.info("tgcalls imported successfully")
except ImportError as e:
    TGCALLS_OK = False
    log.error(f"tgcalls import failed: {e}")

# ─── Telethon imports ────────────────────────────────────────
try:
    from telethon import TelegramClient
    from telethon.tl.functions.phone import (
        RequestCallRequest,
        AcceptCallRequest,
        ConfirmCallRequest,
        DiscardCallRequest,
        ReceivedCallRequest,
        SendSignalingDataRequest,
    )
    from telethon.tl.functions.messages import GetDhConfigRequest
    from telethon.tl.types import (
        PhoneCallProtocol,
        PhoneCallDiscardReasonHangup,
        PhoneCallDiscardReasonBusy,
        UpdatePhoneCall,
        PhoneCallRequested,
        PhoneCallAccepted,
        PhoneCall,
        PhoneCallDiscarded,
        PhoneCallWaiting,
        InputPhoneCall,
        UpdatePhoneCallSignalingData,
    )
    TELETHON_OK = True
except ImportError as e:
    TELETHON_OK = False
    log.error(f"telethon import failed: {e}")

# ─── Crypto for Diffie-Hellman key exchange ──────────────────
from hashlib import sha1, sha256

# ─── Configuration ───────────────────────────────────────────
BRIDGE_PORT = int(os.environ.get("CALL_BRIDGE_PORT", "8770"))
MAIN_APP_URL = os.environ.get("MAIN_APP_URL", "http://localhost:8765")
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_DURATION_MS = 20  # 20ms frames
FRAME_SIZE = SAMPLE_RATE * CHANNELS * 2 * FRAME_DURATION_MS // 1000  # bytes per frame

# ─── Protocol constants ─────────────────────────────────────
# Advertise a wide range of library versions so modern Telegram clients accept the call.
# The actual WebRTC implementation in tgcalls maps unknown versions to a reference impl.
SUPPORTED_VERSIONS = ["11.0.0", "10.0.0", "9.0.0", "8.0.0", "7.0.0", "6.0.0", "5.0.0", "4.0.2", "4.0.0", "3.0.0", "2.4.4"]

CALL_PROTOCOL = PhoneCallProtocol(
    min_layer=92,
    max_layer=92,
    udp_p2p=True,
    udp_reflector=True,
    library_versions=SUPPORTED_VERSIONS,
) if TELETHON_OK else None


# ═══════════════════════════════════════════════════════════════
#  CALL STATE
# ═══════════════════════════════════════════════════════════════

class ActiveCall:
    """Tracks state and audio for one active call."""

    def __init__(self, user_id: int, direction: str):
        self.user_id = user_id
        self.direction = direction  # "outgoing" or "incoming"
        self.status = "ringing"
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.phone_call = None  # Telethon PhoneCall object
        self.native: Optional[NativeInstance] = None
        self._native_proc = None  # subprocess for isolated native media
        self.audio_out_queue: asyncio.Queue = asyncio.Queue()
        self.audio_in_buffer = bytearray()
        self.chunks_sent = 0
        self.chunks_received = 0
        self.encryption_key: Optional[bytes] = None
        self.g_a_hash: Optional[bytes] = None
        self.g_a: Optional[int] = None
        self.g_a_bytes: Optional[bytes] = None  # g_a as 256-byte big-endian
        self.dh_a: Optional[int] = None  # outgoing: our DH private key
        self.dh_b: Optional[int] = None  # incoming: our DH private key
        self.g_b: Optional[int] = None

    @property
    def duration(self) -> float:
        if not self.started_at:
            return 0
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "direction": self.direction,
            "status": self.status,
            "duration_s": self.duration,
            "chunks_sent": self.chunks_sent,
            "chunks_received": self.chunks_received,
        }


_calls: Dict[int, ActiveCall] = {}  # private calls keyed by user_id
_group_calls: Dict[int, "GroupCallState"] = {}  # group calls keyed by chat_id
_tg_client: Optional[TelegramClient] = None
_loop: Optional[asyncio.AbstractEventLoop] = None


async def _resolve_id(identifier) -> int:
    """Resolve a username or string to a numeric Telegram ID."""
    if isinstance(identifier, int):
        return identifier
    s = str(identifier).strip().lstrip("@")
    if s.lstrip("-").isdigit():
        return int(s)
    if _tg_client:
        entity = await _tg_client.get_entity(identifier)
        return entity.id
    raise ValueError(f"Cannot resolve '{identifier}' — bridge Telethon client not connected")


def _resolve_id_sync(identifier, loop) -> int:
    """Synchronously resolve an identifier to a numeric ID (for HTTP handler thread)."""
    if isinstance(identifier, int):
        return identifier
    s = str(identifier).strip().lstrip("@")
    if s.lstrip("-").isdigit():
        return int(s)
    return asyncio.run_coroutine_threadsafe(
        _resolve_id(identifier), loop
    ).result(timeout=10)


class GroupCallState:
    """Tracks state of a group voice chat."""

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.status = "joining"
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.group_call_raw = None  # pytgcalls GroupCallRaw instance
        self.audio_out_queue: asyncio.Queue = asyncio.Queue()
        self.audio_in_buffer = bytearray()
        self.chunks_sent = 0
        self.chunks_received = 0

    @property
    def duration(self) -> float:
        if not self.started_at:
            return 0
        end = self.ended_at or time.time()
        return round(end - self.started_at, 1)

    def to_dict(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "status": self.status,
            "duration_s": self.duration,
            "chunks_sent": self.chunks_sent,
            "chunks_received": self.chunks_received,
            "call_type": "group",
        }


# ═══════════════════════════════════════════════════════════════
#  DH KEY EXCHANGE — PROPER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════

# Cached DH config from Telegram servers
_dh_config: Optional[dict] = None


async def _get_dh_config() -> dict:
    """Fetch DH config (p, g, random) from Telegram servers."""
    global _dh_config
    if _dh_config is not None:
        return _dh_config

    result = await _tg_client(GetDhConfigRequest(version=0, random_length=256))
    # result is messages.DhConfig with p, g, random
    _dh_config = {
        "p": int.from_bytes(result.p, "big"),
        "p_bytes": result.p,
        "g": result.g,
        "random": result.random,
        "version": result.version,
    }
    log.info(f"DH config fetched: g={result.g}, p_len={len(result.p)} bytes, version={result.version}")
    return _dh_config


def _calc_fingerprint(key: bytes) -> int:
    """Calculate key fingerprint for Telegram call encryption.

    Telegram uses the first 8 bytes of SHA1(key) interpreted as a signed int64.
    """
    h = sha1(key).digest()
    return int.from_bytes(h[:8], "little", signed=True)


def _int_to_bytes(n: int) -> bytes:
    """Convert a big integer to 256-byte big-endian."""
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    # Pad to 256 bytes
    if len(b) < 256:
        b = b"\x00" * (256 - len(b)) + b
    return b


def _check_g_a(g_a: int, p: int) -> bool:
    """Validate g_a as per Telegram's requirements."""
    # 1 < g_a < p - 1
    if g_a <= 1 or g_a >= p - 1:
        return False
    # g_a > 2^{2048-64} and g_a < p - 2^{2048-64}
    safety = 2 ** (2048 - 64)
    if g_a < safety or g_a > p - safety:
        return False
    return True


# ═══════════════════════════════════════════════════════════════
#  AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════

def _pcm_to_wav(pcm: bytes, rate: int = 48000, ch: int = 1) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_to_pcm(wav_bytes: bytes) -> Tuple[bytes, int]:
    buf = BytesIO(wav_bytes)
    with wave.open(buf, "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def _silence(length: int) -> bytes:
    return b"\x00" * length


# ═══════════════════════════════════════════════════════════════
#  NATIVE CALL SETUP (tgcalls WebRTC)
# ═══════════════════════════════════════════════════════════════

def _native_worker(rtc_servers: list, key_bytes: bytes, is_outgoing: bool, user_id: int):
    """Run tgcalls NativeInstance in an isolated subprocess.

    tgcalls 2.0.0 (2021) segfaults on startCall on macOS x86_64.
    Running in a subprocess isolates the crash from the main bridge process.
    If it succeeds, the media transport works. If it crashes, the call
    is still established at the Telegram protocol level (just no audio).
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    log_file = str(Path(tempfile.gettempdir()) / f"tgcalls_{user_id}.log")
    print(f"[NATIVE-WORKER] Creating NativeInstance, log: {log_file}")

    try:
        ni = NativeInstance(True, log_file)
    except Exception as e:
        print(f"[NATIVE-WORKER] Failed to create NativeInstance: {e}")
        return

    servers = []
    for i, s in enumerate(rtc_servers):
        try:
            ip = s.get("host", s.get("ip", ""))
            port = s.get("port", 0)
            login = s.get("username", "")
            password = s.get("password", "")
            is_turn = s.get("turn", False)
            is_stun = s.get("stun", False)
            if not ip or not port:
                continue
            srv = RtcServer(str(i), ip, int(port), login, password, is_turn, is_stun)
            ipv6 = s.get("ipv6", "")
            if ipv6:
                srv.ipv6 = ipv6
            servers.append(srv)
        except Exception:
            pass

    if not servers:
        print("[NATIVE-WORKER] No valid RTC servers")
        return

    key_list = list(key_bytes)
    print(f"[NATIVE-WORKER] startCall: {len(servers)} servers, outgoing={is_outgoing}, version=3.0.0")
    try:
        ni.startCall(servers, key_list, is_outgoing, "3.0.0")
        print("[NATIVE-WORKER] startCall succeeded")
        ni.startAudioDeviceModule()
        print("[NATIVE-WORKER] Audio device module started, keeping alive...")
        # Keep the subprocess alive while the call is active
        import time as _time
        while True:
            _time.sleep(1)
    except Exception as e:
        print(f"[NATIVE-WORKER] startCall failed: {e}")


def _setup_native(call: ActiveCall, rtc_servers: list, key: bytes, is_outgoing: bool):
    """Initialize tgcalls NativeInstance for media transport.

    Runs in a subprocess to isolate potential segfaults in the tgcalls native code.
    The call is marked active regardless of whether native media succeeds.
    """
    import multiprocessing
    if call.native is not None:
        log.info(f"Native already set up for {call.user_id}, skipping")
        return

    log.info(f"Launching native media worker subprocess for {call.user_id}")

    proc = multiprocessing.Process(
        target=_native_worker,
        args=(rtc_servers, key, is_outgoing, call.user_id),
        daemon=True,
    )
    proc.start()
    log.info(f"Native worker PID: {proc.pid}")

    # Store the process so we can kill it on hangup
    call._native_proc = proc

    # Mark call as active regardless — protocol-level call is established
    call.status = "active"
    call.started_at = time.time()

    # Check after a short delay if the subprocess survived
    import threading

    def _check_native():
        proc.join(timeout=3)
        if proc.is_alive():
            log.info(f"Native media worker {proc.pid} is running (audio active)")
        else:
            code = proc.exitcode
            if code == -11 or code == 139:
                log.warning(
                    f"Native media worker crashed (SIGSEGV). "
                    f"Call is connected but WITHOUT audio. "
                    f"tgcalls 2.0.0 is incompatible with this platform."
                )
            elif code != 0:
                log.warning(f"Native media worker exited with code {code}")
            else:
                log.info("Native media worker exited normally")

    threading.Thread(target=_check_native, daemon=True).start()
    log.info(f">>> CALL ACTIVE for user {call.user_id}, direction={call.direction} <<<")


def _extract_rtc_servers(phone_call) -> list:
    """Extract RTC server info from a PhoneCall object's connections."""
    from telethon.tl.types import PhoneConnectionWebrtc as PCWebrtc

    rtc_servers = []
    connections = getattr(phone_call, "connections", None) or []
    log.info(f"Extracting RTC servers from {len(connections)} connections")

    for conn in connections:
        entry = {
            "host": getattr(conn, "ip", ""),
            "port": getattr(conn, "port", 0),
            "ipv6": getattr(conn, "ipv6", None) or "",
        }
        if isinstance(conn, PCWebrtc):
            entry["username"] = getattr(conn, "username", "")
            entry["password"] = getattr(conn, "password", "")
            entry["turn"] = getattr(conn, "turn", False)
            entry["stun"] = getattr(conn, "stun", False)
        else:
            entry["peer_tag"] = getattr(conn, "peer_tag", b"")
            entry["turn"] = False
            entry["stun"] = False

        rtc_servers.append(entry)
        log.info(f"  Connection: {type(conn).__name__} {entry['host']}:{entry['port']}")

    return rtc_servers


def _try_setup_media(call: ActiveCall):
    """Try to set up native media transport if we have everything we need."""
    log.info(f"_try_setup_media called for {call.user_id}: "
             f"native={'YES' if call.native else 'NO'}, "
             f"phone_call={type(call.phone_call).__name__ if call.phone_call else 'None'}, "
             f"encryption_key={'YES' if call.encryption_key else 'NO'}")

    if call.native is not None:
        log.info(f"Native already set up for {call.user_id}")
        return
    if not call.phone_call:
        log.warning(f"Cannot setup media for {call.user_id}: no phone_call object")
        return
    if not call.encryption_key:
        log.warning(f"Cannot setup media for {call.user_id}: no encryption key")
        return

    pc = call.phone_call
    # Debug: dump all attributes of the phone_call object
    log.info(f"PhoneCall attrs: {[a for a in dir(pc) if not a.startswith('_')]}")
    log.info(f"PhoneCall type: {type(pc).__name__}")
    connections = getattr(pc, "connections", None)
    log.info(f"PhoneCall.connections: {connections} (type={type(connections).__name__ if connections is not None else 'None'})")
    if connections:
        log.info(f"  connections count: {len(connections)}")
        for i, c in enumerate(connections):
            log.info(f"  conn[{i}]: {type(c).__name__} attrs={[a for a in dir(c) if not a.startswith('_')]}")

    rtc_servers = _extract_rtc_servers(pc)

    if not rtc_servers:
        log.error(f"No RTC servers found for {call.user_id}")
        return

    is_outgoing = call.direction == "outgoing"
    _setup_native(call, rtc_servers, call.encryption_key, is_outgoing)


async def _forward_signaling(call: ActiveCall, data: bytes):
    """Forward WebRTC signaling data to Telegram via SendSignalingDataRequest."""
    if not _tg_client or not call.phone_call:
        log.warning(f"Cannot forward signaling: no client or phone_call for {call.user_id}")
        return
    try:
        pc = call.phone_call
        input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)
        await _tg_client(SendSignalingDataRequest(
            peer=input_call,
            data=data,
        ))
        log.debug(f"Forwarded {len(data)} bytes of signaling data for call {call.user_id}")
    except Exception as e:
        log.error(f"Failed to forward signaling data: {e}")


# ═══════════════════════════════════════════════════════════════
#  CALL MANAGEMENT
# ═══════════════════════════════════════════════════════════════

async def make_call(user_id: int) -> dict:
    """Initiate an outgoing call with proper DH key exchange."""
    if not TGCALLS_OK:
        return {"success": False, "error": "tgcalls not available"}
    if not _tg_client:
        return {"success": False, "error": "Telegram client not connected"}
    if user_id in _calls:
        return {"success": False, "error": f"Already in call with {user_id}"}

    call = ActiveCall(user_id, "outgoing")
    _calls[user_id] = call

    try:
        # Get DH config from Telegram
        dh = await _get_dh_config()
        p = dh["p"]
        g = dh["g"]

        # Generate our DH private value (a) and compute g_a = g^a mod p
        a = int.from_bytes(os.urandom(256), "big")
        g_a = pow(g, a, p)

        if not _check_g_a(g_a, p):
            _calls.pop(user_id, None)
            return {"success": False, "error": "DH g_a validation failed, try again"}

        g_a_bytes = _int_to_bytes(g_a)
        g_a_hash = sha256(g_a_bytes).digest()

        # Store DH state on the call object
        call.dh_a = a  # our private key
        call.g_a = g_a  # our public key (int)
        call.g_a_bytes = g_a_bytes  # our public key (bytes)
        call.g_a_hash = g_a_hash

        # Use Telethon to request the call
        entity = await _tg_client.get_entity(user_id)
        result = await _tg_client(RequestCallRequest(
            user_id=entity,
            random_id=int.from_bytes(os.urandom(4), "big", signed=True),
            g_a_hash=g_a_hash,
            protocol=CALL_PROTOCOL,
        ))
        call.phone_call = result.phone_call
        log.info(f"Call requested to {user_id}, call_id={result.phone_call.id}")
        return {
            "success": True,
            "message": f"Calling {user_id}...",
            "call_id": result.phone_call.id,
        }
    except Exception as e:
        _calls.pop(user_id, None)
        log.error(f"make_call failed: {e}")
        return {"success": False, "error": str(e)}


async def accept_call(user_id: int) -> dict:
    """Accept an incoming call with proper DH key exchange."""
    call = _calls.get(user_id)
    if not call or call.direction != "incoming":
        return {"success": False, "error": "No incoming call from this user"}
    if not call.phone_call:
        return {"success": False, "error": "No phone call object"}

    try:
        # Get DH config
        dh = await _get_dh_config()
        p = dh["p"]
        g = dh["g"]

        # Generate our DH private value (b) and compute g_b = g^b mod p
        b = int.from_bytes(os.urandom(256), "big")
        g_b = pow(g, b, p)

        if not _check_g_a(g_b, p):
            return {"success": False, "error": "DH g_b validation failed, try again"}

        g_b_bytes = _int_to_bytes(g_b)

        # Store DH state
        call.dh_b = b
        call.g_b = g_b

        # We need g_a from the caller to compute the shared key.
        # g_a comes in the PhoneCall update (after ConfirmCallRequest from caller).
        # For now, store b so we can compute the key when we get g_a.

        pc = call.phone_call
        input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)
        result = await _tg_client(AcceptCallRequest(
            peer=input_call,
            g_b=g_b_bytes,
            protocol=CALL_PROTOCOL,
        ))
        call.phone_call = result.phone_call
        call.status = "accepted"
        log.info(f"Call accepted from {user_id}")
        return {"success": True, "message": f"Accepted call from {user_id}"}
    except Exception as e:
        log.error(f"accept_call failed: {e}")
        return {"success": False, "error": str(e)}


async def decline_call(user_id: int) -> dict:
    """Decline/reject an incoming call."""
    call = _calls.pop(user_id, None)
    if not call:
        return {"success": False, "error": "No call from this user"}

    try:
        if call.phone_call:
            pc = call.phone_call
            input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)
            await _tg_client(DiscardCallRequest(
                peer=input_call,
                duration=0,
                reason=PhoneCallDiscardReasonBusy(),
                connection_id=0,
            ))
        call.status = "ended"
        call.ended_at = time.time()
        return {"success": True, "message": f"Declined call from {user_id}"}
    except Exception as e:
        log.error(f"decline_call failed: {e}")
        return {"success": False, "error": str(e)}


async def hangup(user_id: int) -> dict:
    """Hang up an active call."""
    call = _calls.pop(user_id, None)
    if not call:
        return {"success": False, "error": "No active call with this user"}

    try:
        # Stop native subprocess if running
        native_proc = getattr(call, "_native_proc", None)
        if native_proc and native_proc.is_alive():
            try:
                native_proc.terminate()
                native_proc.join(timeout=2)
                if native_proc.is_alive():
                    native_proc.kill()
            except Exception:
                pass

        # Stop native instance (if setup was done in-process)
        if call.native:
            try:
                call.native.stopAudioDeviceModule()
            except Exception:
                pass

        # Discard via Telegram
        if call.phone_call and _tg_client:
            try:
                pc = call.phone_call
                input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)
                await _tg_client(DiscardCallRequest(
                    peer=input_call,
                    duration=int(call.duration),
                    reason=PhoneCallDiscardReasonHangup(),
                    connection_id=0,
                ))
            except Exception as e:
                log.warning(f"Discard call API error: {e}")

        call.status = "ended"
        call.ended_at = time.time()
        return {
            "success": True,
            "message": f"Call ended with {user_id}",
            "duration_s": call.duration,
        }
    except Exception as e:
        log.error(f"hangup failed: {e}")
        return {"success": False, "error": str(e)}


async def queue_audio(user_id: int, pcm_data: bytes) -> dict:
    """Queue PCM audio to play in an active call."""
    call = _calls.get(user_id)
    if not call or call.status != "active":
        return {"success": False, "error": "No active call"}

    # Split into frames
    for i in range(0, len(pcm_data), FRAME_SIZE):
        frame = pcm_data[i:i + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            frame += _silence(FRAME_SIZE - len(frame))
        await call.audio_out_queue.put(frame)

    return {"success": True, "frames_queued": len(pcm_data) // FRAME_SIZE + 1}


async def get_recorded_audio(user_id: int, clear: bool = True) -> dict:
    """Get recorded audio from the remote party."""
    call = _calls.get(user_id)
    if not call:
        return {"success": False, "error": "No call with this user"}

    data = bytes(call.audio_in_buffer)
    if clear:
        call.audio_in_buffer.clear()

    import base64
    return {
        "success": True,
        "pcm_base64": base64.b64encode(data).decode() if data else "",
        "length_bytes": len(data),
        "duration_s": round(len(data) / (SAMPLE_RATE * 2), 2),
    }


# ═══════════════════════════════════════════════════════════════
#  GROUP CALL MANAGEMENT (voice chats)
# ═══════════════════════════════════════════════════════════════

async def join_group_call(chat_id: int) -> dict:
    """Join a group voice chat using pytgcalls GroupCallRaw."""
    if not TGCALLS_OK:
        return {"success": False, "error": "tgcalls not available"}
    if not _tg_client:
        return {"success": False, "error": "Telegram client not connected"}
    if chat_id in _group_calls:
        return {"success": False, "error": f"Already in group call for {chat_id}"}

    gc = GroupCallState(chat_id)
    _group_calls[chat_id] = gc

    try:
        from pytgcalls.implementation.group_call_raw import GroupCallRaw as GCRClass

        group_call = GCRClass(_tg_client)
        gc.group_call_raw = group_call

        # Set up raw audio callbacks
        @group_call.on_played_data
        async def on_played(length: int) -> bytes:
            try:
                data = gc.audio_out_queue.get_nowait()
                gc.chunks_sent += 1
                if len(data) >= length:
                    return data[:length]
                return data + _silence(length - len(data))
            except Exception:
                return _silence(length)

        @group_call.on_recorded_data
        async def on_recorded(data: bytes):
            gc.chunks_received += 1
            gc.audio_in_buffer.extend(data)
            max_size = SAMPLE_RATE * 2 * 30
            if len(gc.audio_in_buffer) > max_size:
                gc.audio_in_buffer = gc.audio_in_buffer[-max_size:]

        # Join the group call
        await group_call.start(chat_id)
        gc.status = "active"
        gc.started_at = time.time()
        log.info(f"Joined group call in {chat_id}")
        return {"success": True, "message": f"Joined group call in {chat_id}"}

    except Exception as e:
        _group_calls.pop(chat_id, None)
        log.error(f"join_group_call failed: {e}")
        return {"success": False, "error": str(e)}


async def leave_group_call(chat_id: int) -> dict:
    """Leave a group voice chat."""
    gc = _group_calls.pop(chat_id, None)
    if not gc:
        return {"success": False, "error": "Not in group call"}

    try:
        if gc.group_call_raw:
            await gc.group_call_raw.stop()
        gc.status = "ended"
        gc.ended_at = time.time()
        return {"success": True, "message": f"Left group call in {chat_id}", "duration_s": gc.duration}
    except Exception as e:
        log.error(f"leave_group_call failed: {e}")
        return {"success": False, "error": str(e)}


async def queue_group_audio(chat_id: int, pcm_data: bytes) -> dict:
    """Queue PCM audio for a group call."""
    gc = _group_calls.get(chat_id)
    if not gc or gc.status != "active":
        return {"success": False, "error": "Not in group call"}

    for i in range(0, len(pcm_data), FRAME_SIZE):
        frame = pcm_data[i:i + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            frame += _silence(FRAME_SIZE - len(frame))
        await gc.audio_out_queue.put(frame)

    return {"success": True, "frames_queued": len(pcm_data) // FRAME_SIZE + 1}


async def get_group_recorded_audio(chat_id: int, clear: bool = True) -> dict:
    """Get recorded audio from a group call."""
    gc = _group_calls.get(chat_id)
    if not gc:
        return {"success": False, "error": "Not in group call"}

    data = bytes(gc.audio_in_buffer)
    if clear:
        gc.audio_in_buffer.clear()

    import base64
    return {
        "success": True,
        "pcm_base64": base64.b64encode(data).decode() if data else "",
        "length_bytes": len(data),
        "duration_s": round(len(data) / (SAMPLE_RATE * 2), 2),
    }


# ═══════════════════════════════════════════════════════════════
#  INCOMING CALL HANDLER
# ═══════════════════════════════════════════════════════════════

async def _handle_update(event):
    """Handle Telegram phone call updates."""
    log.info(f">>> RAW UPDATE: {type(event).__name__}")
    if not hasattr(event, "phone_call"):
        log.info("  No phone_call attribute, ignoring")
        return

    pc = event.phone_call
    log.info(f"  phone_call type: {type(pc).__name__}, id={getattr(pc, 'id', '?')}")

    if isinstance(pc, PhoneCallRequested):
        user_id = pc.admin_id
        log.info(f"Incoming call from {user_id}")
        call = ActiveCall(user_id, "incoming")
        call.phone_call = pc
        _calls[user_id] = call

        # Notify the main app about the incoming call
        try:
            import urllib.request
            data = json.dumps({"user_id": user_id, "event": "incoming_call"}).encode()
            req = urllib.request.Request(
                f"{MAIN_APP_URL}/call/incoming",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.warning(f"Could not notify main app of incoming call: {e}")

        # Auto-acknowledge receipt
        try:
            input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)
            await _tg_client(ReceivedCallRequest(peer=input_call))
        except Exception as e:
            log.warning(f"ReceivedCallRequest failed: {e}")

    elif isinstance(pc, PhoneCallAccepted):
        # Our outgoing call was accepted — compute shared key and confirm
        for uid, c in _calls.items():
            if c.direction == "outgoing" and c.status == "ringing":
                c.phone_call = pc
                c.status = "accepted"
                log.info(f"Call to {uid} was accepted")

                try:
                    dh = await _get_dh_config()
                    p = dh["p"]

                    # Build InputPhoneCall for API requests
                    input_call = InputPhoneCall(id=pc.id, access_hash=pc.access_hash)

                    # The callee's g_b is in pc.g_b
                    g_b = int.from_bytes(pc.g_b, "big")

                    if not _check_g_a(g_b, p):
                        log.error("g_b validation failed!")
                        await _tg_client(DiscardCallRequest(
                            peer=input_call,
                            duration=0,
                            reason=PhoneCallDiscardReasonHangup(),
                            connection_id=0,
                        ))
                        _calls.pop(uid, None)
                        break

                    # Compute shared key: key = g_b^a mod p
                    auth_key = pow(g_b, c.dh_a, p)
                    auth_key_bytes = _int_to_bytes(auth_key)
                    c.encryption_key = auth_key_bytes

                    key_fingerprint = _calc_fingerprint(auth_key_bytes)

                    # Confirm the call with our g_a and the key fingerprint
                    result = await _tg_client(ConfirmCallRequest(
                        peer=input_call,
                        g_a=c.g_a_bytes,
                        key_fingerprint=key_fingerprint,
                        protocol=CALL_PROTOCOL,
                    ))
                    c.phone_call = result.phone_call
                    log.info(f"Call confirmed with {uid}, fingerprint={key_fingerprint}")
                    log.info(f"ConfirmCallRequest result type: {type(result.phone_call).__name__}")
                    log.info(f"Result has connections: {hasattr(result.phone_call, 'connections')}")
                    if hasattr(result.phone_call, 'connections'):
                        log.info(f"Connections count: {len(result.phone_call.connections or [])}")

                    # Set up native media NOW — don't wait for a PhoneCall update
                    _try_setup_media(c)

                except Exception as e:
                    log.error(f"ConfirmCallRequest failed: {e}")
                break

    elif isinstance(pc, PhoneCall):
        # Call is now fully established — start media
        for uid, c in _calls.items():
            if c.phone_call and getattr(c.phone_call, "id", None) == pc.id:
                c.phone_call = pc
                log.info(f"PhoneCall update received for {uid}, connections={len(pc.connections or [])}")

                # For incoming calls, compute the shared key now
                if c.direction == "incoming" and c.dh_b and c.encryption_key is None:
                    try:
                        dh = await _get_dh_config()
                        p = dh["p"]
                        g_a_bytes = pc.g_a_or_b
                        g_a = int.from_bytes(g_a_bytes, "big")

                        if not _check_g_a(g_a, p):
                            log.error("Incoming call g_a validation failed!")
                            break

                        auth_key = pow(g_a, c.dh_b, p)
                        auth_key_bytes = _int_to_bytes(auth_key)
                        c.encryption_key = auth_key_bytes
                        log.info(f"Incoming call key computed for {uid}")
                    except Exception as e:
                        log.error(f"Failed to compute incoming call key: {e}")

                # Try to set up media (idempotent — won't run twice)
                _try_setup_media(c)
                break

    elif isinstance(pc, PhoneCallDiscarded):
        # Call ended
        for uid in list(_calls.keys()):
            c = _calls[uid]
            if c.phone_call and getattr(c.phone_call, "id", None) == pc.id:
                c.status = "ended"
                c.ended_at = time.time()
                if c.native:
                    try:
                        c.native.stopAudioDeviceModule()
                    except Exception:
                        pass
                _calls.pop(uid, None)
                log.info(f"Call with {uid} ended (discarded)")

                # Notify main app
                try:
                    import urllib.request
                    data = json.dumps({
                        "user_id": uid,
                        "event": "call_ended",
                        "duration_s": c.duration,
                    }).encode()
                    req = urllib.request.Request(
                        f"{MAIN_APP_URL}/call/event",
                        data=data,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=5)
                except Exception:
                    pass
                break


async def _handle_signaling_data(event):
    """Handle incoming WebRTC signaling data from Telegram.

    When the other party sends signaling data, we must forward it
    to our tgcalls NativeInstance for the WebRTC connection to work.
    """
    if not isinstance(event, UpdatePhoneCallSignalingData):
        return

    call_id = event.phone_call_id
    data = event.data

    # Find which call this belongs to
    for uid, c in _calls.items():
        if c.phone_call and getattr(c.phone_call, "id", None) == call_id:
            if c.native:
                try:
                    c.native.receiveSignalingData(list(data))
                    log.debug(f"Received signaling data for call {uid}: {len(data)} bytes")
                except Exception as e:
                    log.error(f"receiveSignalingData failed for {uid}: {e}")
            else:
                log.warning(f"Got signaling data for {uid} but no native instance")
            return

    log.warning(f"Got signaling data for unknown call_id={call_id}")


# ═══════════════════════════════════════════════════════════════
#  HTTP SERVER (bridge interface)
# ═══════════════════════════════════════════════════════════════

class BridgeHandler(BaseHTTPRequestHandler):
    """HTTP handler for the call bridge."""

    def log_message(self, fmt, *args):
        log.debug(fmt % args)

    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/status":
            self._json_response({
                "ok": True,
                "tgcalls": TGCALLS_OK,
                "telethon": TELETHON_OK,
                "connected": _tg_client is not None and _tg_client.is_connected(),
                "active_calls": {str(k): v.to_dict() for k, v in _calls.items()},
                "group_calls": {str(k): v.to_dict() for k, v in _group_calls.items()},
            })
        elif path == "/calls":
            self._json_response({
                "calls": {str(k): v.to_dict() for k, v in _calls.items()},
                "group_calls": {str(k): v.to_dict() for k, v in _group_calls.items()},
            })
        elif path.startswith("/audio/"):
            # GET /audio/<user_id> — get recorded audio
            try:
                uid = _resolve_id_sync(path.split("/")[-1], _loop)
                result = asyncio.run_coroutine_threadsafe(
                    get_recorded_audio(uid), _loop
                ).result(timeout=5)
                self._json_response(result)
            except Exception as e:
                self._json_response({"success": False, "error": str(e)}, 500)
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_body()

        if path == "/call/make":
            uid = body.get("user_id")
            if not uid:
                self._json_response({"success": False, "error": "user_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(uid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve user: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                make_call(resolved), _loop
            ).result(timeout=30)
            self._json_response(result)

        elif path == "/call/accept":
            uid = body.get("user_id")
            if not uid:
                self._json_response({"success": False, "error": "user_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(uid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve user: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                accept_call(resolved), _loop
            ).result(timeout=30)
            self._json_response(result)

        elif path == "/call/decline":
            uid = body.get("user_id")
            if not uid:
                self._json_response({"success": False, "error": "user_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(uid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve user: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                decline_call(resolved), _loop
            ).result(timeout=10)
            self._json_response(result)

        elif path == "/call/hangup":
            uid = body.get("user_id")
            if not uid:
                self._json_response({"success": False, "error": "user_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(uid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve user: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                hangup(resolved), _loop
            ).result(timeout=10)
            self._json_response(result)

        elif path == "/call/audio":
            # POST /call/audio — queue PCM audio for playback
            uid = body.get("user_id")
            pcm_b64 = body.get("pcm_base64", "")
            if not uid or not pcm_b64:
                self._json_response({
                    "success": False,
                    "error": "user_id and pcm_base64 required",
                }, 400)
                return
            import base64
            try:
                resolved = _resolve_id_sync(uid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve user: {e}"}, 400)
                return
            pcm = base64.b64decode(pcm_b64)
            result = asyncio.run_coroutine_threadsafe(
                queue_audio(resolved, pcm), _loop
            ).result(timeout=10)
            self._json_response(result)

        # ── Group call routes ──
        elif path == "/call/group/join":
            cid = body.get("chat_id")
            if not cid:
                self._json_response({"success": False, "error": "chat_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(cid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve chat: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                join_group_call(resolved), _loop
            ).result(timeout=30)
            self._json_response(result)

        elif path == "/call/group/leave":
            cid = body.get("chat_id")
            if not cid:
                self._json_response({"success": False, "error": "chat_id required"}, 400)
                return
            try:
                resolved = _resolve_id_sync(cid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve chat: {e}"}, 400)
                return
            result = asyncio.run_coroutine_threadsafe(
                leave_group_call(resolved), _loop
            ).result(timeout=10)
            self._json_response(result)

        elif path == "/call/group/audio":
            cid = body.get("chat_id")
            pcm_b64 = body.get("pcm_base64", "")
            if not cid or not pcm_b64:
                self._json_response({
                    "success": False,
                    "error": "chat_id and pcm_base64 required",
                }, 400)
                return
            import base64
            try:
                resolved = _resolve_id_sync(cid, _loop)
            except Exception as e:
                self._json_response({"success": False, "error": f"Cannot resolve chat: {e}"}, 400)
                return
            pcm = base64.b64decode(pcm_b64)
            result = asyncio.run_coroutine_threadsafe(
                queue_group_audio(resolved, pcm), _loop
            ).result(timeout=10)
            self._json_response(result)

        else:
            self._json_response({"error": "not found"}, 404)


# ═══════════════════════════════════════════════════════════════
#  MAIN — Telethon client + HTTP server
# ═══════════════════════════════════════════════════════════════

async def _start_telethon():
    """Start Telethon client with the same session as the main app."""
    global _tg_client

    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    session = os.environ.get("TELEGRAM_SESSION_STRING")

    if not api_id or not api_hash:
        log.error("TELEGRAM_API_ID and TELEGRAM_API_HASH required")
        return False

    from telethon.sessions import StringSession

    _tg_client = TelegramClient(
        StringSession(session) if session else "call_session",
        int(api_id),
        api_hash,
    )
    await _tg_client.connect()

    if not await _tg_client.is_user_authorized():
        log.error("Telegram client not authorized. Check session string.")
        return False

    me = await _tg_client.get_me()
    log.info(f"Connected as {me.first_name} (ID: {me.id})")

    # Register handler for phone call updates
    from telethon import events
    _tg_client.add_event_handler(
        _handle_update,
        events.Raw(types=[UpdatePhoneCall]),
    )

    # Register handler for incoming signaling data (WebRTC)
    _tg_client.add_event_handler(
        _handle_signaling_data,
        events.Raw(types=[UpdatePhoneCallSignalingData]),
    )

    return True


def _run_http_server():
    """Run the HTTP bridge server in a separate thread."""
    server = HTTPServer(("127.0.0.1", BRIDGE_PORT), BridgeHandler)
    log.info(f"Call bridge HTTP server on port {BRIDGE_PORT}")
    server.serve_forever()


async def main():
    global _loop
    _loop = asyncio.get_event_loop()

    log.info("=" * 60)
    log.info("  TELEGRAM CALL BRIDGE (Python 3.10 + tgcalls)")
    log.info(f"  tgcalls: {'OK' if TGCALLS_OK else 'MISSING'}")
    log.info(f"  telethon: {'OK' if TELETHON_OK else 'MISSING'}")
    log.info(f"  Port: {BRIDGE_PORT}")
    log.info("=" * 60)

    if not TGCALLS_OK:
        log.error("Cannot start without tgcalls. Exiting.")
        sys.exit(1)

    # Start Telethon
    ok = await _start_telethon()
    if not ok:
        log.error("Telethon connection failed. Exiting.")
        sys.exit(1)

    # Start HTTP server in background thread
    http_thread = Thread(target=_run_http_server, daemon=True)
    http_thread.start()

    log.info("Call bridge is running. Press Ctrl+C to stop.")

    # Keep running
    try:
        await _tg_client.run_until_disconnected()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        if _tg_client:
            await _tg_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
