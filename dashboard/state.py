"""
Dashboard State — Real-time Telegram AI bot control & monitoring.

Architecture: Frontend-driven polling via rx.call_script + setTimeout.
- tick_poll runs as @rx.background — does NOT hold the state lock during IO
- msg_tick runs as @rx.background — same, so user clicks are never blocked
- Toggle/config actions use optimistic updates (yield before API call)
- All data sourced from telegram_api.py REST endpoints
"""

import asyncio
import reflex as rx
import httpx
import time
from typing import Any

API_BASE = "http://localhost:8765"
TIMEOUT = 3  # Fast timeout for polling (prevents UI freezes)
TIMEOUT_ACTION = 8  # Longer timeout for user-initiated actions

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _api(method: str, path: str, json_data: Any = None, params: dict = None,
         timeout: int = TIMEOUT) -> dict | None:
    """Synchronous API call to the Telegram bot backend."""
    try:
        r = httpx.request(
            method, f"{API_BASE}{path}",
            json=json_data, params=params, timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


FEATURE_KEYS = [
    "late_night_mode", "strategic_silence", "quote_reply", "smart_reactions",
    "message_editing", "gif_sticker_reply", "typing_awareness", "online_status_aware",
    "proactive_enabled", "proactive_morning", "proactive_night",
]

FEATURE_META = {
    "late_night_mode": ("Late Night Mode", "Softer tone & timing after midnight", "moon"),
    "strategic_silence": ("Strategic Silence", "Intentionally skip some messages", "volume-x"),
    "quote_reply": ("Quote Reply", "Reply to specific messages with quoting", "message-square"),
    "smart_reactions": ("Smart Reactions", "Send emoji reactions automatically", "smile-plus"),
    "message_editing": ("Message Editing", "Edit sent messages for natural corrections", "pencil"),
    "gif_sticker_reply": ("GIF / Sticker", "Respond with GIFs and stickers", "image"),
    "typing_awareness": ("Typing Awareness", "Wait when user is typing", "keyboard"),
    "online_status_aware": ("Online Status", "Adjust based on online presence", "wifi"),
    "proactive_enabled": ("Proactive Msgs", "Send good morning/night unprompted", "send"),
    "proactive_morning": ("Morning Msgs", "Good morning messages", "sunrise"),
    "proactive_night": ("Night Msgs", "Good night messages", "moon-star"),
}


def _flatten_dict(d: dict, prefix: str = "") -> list[dict[str, str]]:
    """Flatten a nested dict into list of {key, value} for table display."""
    items: list[dict[str, str]] = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, full))
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], dict):
                for i, item in enumerate(v[:10]):
                    items.extend(_flatten_dict(item, f"{full}[{i}]"))
            else:
                display = str(v[:8])
                if len(v) > 8:
                    display += "..."
                items.append({"key": full, "value": display})
        else:
            items.append({"key": full, "value": str(v)})
    return items


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


# ── Message processing (module-level, no state needed) ──
_MEDIA_LABELS = {
    "MessageMediaPhoto": "Photo",
    "MessageMediaDocument": "File",
    "MessageMediaWebPage": "Link",
    "MessageMediaGeo": "Location",
    "MessageMediaContact": "Contact",
    "MessageMediaPoll": "Poll",
    "MessageMediaVenue": "Venue",
    "MessageMediaGame": "Game",
    "MessageMediaInvoice": "Invoice",
    "MessageMediaDice": "Dice",
}


def _process_messages_data(data: dict) -> list[dict[str, str]]:
    """Transform raw /chats/{id}/messages response into normalized format."""
    raw = data.get("messages", [])
    normalized = []
    for m in raw:
        raw_date = _safe_str(m.get("date", ""))
        short_date = raw_date
        if "T" in raw_date:
            short_date = raw_date.split("T")[1][:5]

        raw_media = _safe_str(m.get("media_type", ""))
        media_label = _MEDIA_LABELS.get(raw_media, raw_media.replace("MessageMedia", ""))

        msg_text = _safe_str(m.get("text", ""))
        has_media = m.get("has_media", False)

        normalized.append({
            "id": str(m.get("id", "")),
            "text": msg_text if msg_text else (f"[{media_label}]" if has_media else ""),
            "out": "true" if m.get("out", False) else "false",
            "sender": _safe_str(m.get("sender_name", "")),
            "date": short_date,
            "has_media": "true" if has_media else "false",
            "media_type": media_label,
            "reply_to": str(m.get("reply_to_msg_id", "")),
        })
    return list(reversed(normalized))


# ══════════════════════════════════════════════════════════════
# REFLEX STATE
# ══════════════════════════════════════════════════════════════

class DashboardState(rx.State):
    """Reactive state for the entire dashboard."""

    # ── Connection ──
    connected: bool = False
    last_refresh: str = ""
    api_error: str = ""

    # ── Navigation ──
    active_tab: str = "overview"

    # ── Auto-Reply Core ──
    auto_reply_enabled: bool = False
    chat_count: int = 0
    chat_ids: list[str] = []
    recent_replies: int = 0
    active_features: int = 0
    total_features: int = 11
    delay_min: int = 5
    delay_max: int = 30
    context_messages: int = 15
    proactive_max_per_day: int = 3

    # ── Feature Flags (individual booleans) ──
    feat_late_night_mode: bool = False
    feat_strategic_silence: bool = False
    feat_quote_reply: bool = True
    feat_smart_reactions: bool = True
    feat_message_editing: bool = True
    feat_gif_sticker_reply: bool = True
    feat_typing_awareness: bool = False
    feat_online_status_aware: bool = True
    feat_proactive_enabled: bool = False
    feat_proactive_morning: bool = True
    feat_proactive_night: bool = True

    # ── Engines (list of normalized dicts) ──
    engines_data: list[dict[str, str]] = []
    engines_count: int = 0

    # ── Models ──
    sklearn_models: list[dict[str, str]] = []
    neural_models: list[dict[str, str]] = []

    # ── Media AI ──
    media_ai_available: bool = False
    media_ai_data: list[dict[str, str]] = []

    # ── Advanced Intel ──
    advanced_intel: bool = False

    # ── Activity (recent from /dashboard) ──
    activity_log: list[dict[str, str]] = []

    # ── Chats List (normalized) ──
    chats_list: list[dict[str, str]] = []
    chats_loading: bool = False

    # ── Analysis ──
    analysis_result: list[dict[str, str]] = []
    analysis_loading: bool = False
    analysis_chat_id: str = ""

    # ── Relationship Health ──
    health_result: list[dict[str, str]] = []
    health_loading: bool = False
    health_chat_id: str = ""

    # ── Psychological Analysis ──
    psych_result: list[dict[str, str]] = []
    psych_loading: bool = False
    psych_chat_id: str = ""

    # ── RL Insights ──
    rl_result: list[dict[str, str]] = []
    rl_loading: bool = False

    # ── Engine Details ──
    engine_details: list[dict[str, str]] = []

    # ── Per-Chat Instructions ──
    instructions_data: list[dict[str, str]] = []
    inst_chat_id: str = ""
    inst_text: str = ""

    # ── DL Status ──
    dl_status_data: list[dict[str, str]] = []

    # ── Training ──
    training_running: bool = False
    training_result: str = ""

    # ── My Account ──
    my_info: list[dict[str, str]] = []

    # ── Full Auto-Reply Log (normalized) ──
    full_log: list[dict[str, str]] = []

    # ── Prompt ──
    system_prompt: str = ""

    # ── Whitelist input ──
    wl_input: str = ""

    # ── Delay inputs ──
    delay_min_input: str = "5"
    delay_max_input: str = "30"

    # ── Proactive max input ──
    proactive_max_input: str = "3"

    # ── Messenger State ──
    active_chat_id: str = ""
    active_chat_name: str = ""
    active_chat_initial: str = ""
    active_chat_photo: str = ""
    active_chat_status: str = ""
    messages: list[dict[str, str]] = []
    messages_loading: bool = False
    msg_input: str = ""
    msg_search_query: str = ""
    msg_search_results: list[dict[str, str]] = []

    # ── Chat Analytics ──
    chat_analytics: list[dict[str, str]] = []
    analytics_loading: bool = False

    # ── Behavior Mirroring ──
    mirror_mode: str = "off"
    mirror_data: list[dict[str, str]] = []

    # ══════════════════════════════════════════════════════════
    # NAVIGATION
    # ══════════════════════════════════════════════════════════

    def set_tab(self, tab: str):
        self.active_tab = tab
        yield  # Show tab switch immediately
        if tab == "messenger":
            self.load_chats()
        elif tab == "engines":
            self.load_engine_details()
        elif tab == "instructions":
            self.load_instructions()
        elif tab == "models":
            self.load_dl_status()
        elif tab == "log":
            self.load_full_log()
        elif tab == "account":
            self.load_my_info()
        elif tab == "rl":
            self.load_rl_all()

    # ══════════════════════════════════════════════════════════
    # CORE REFRESH — helpers + background polling
    # ══════════════════════════════════════════════════════════

    # ── Tick counter for staggered secondary fetches ──
    _tick_count: int = 0

    def _apply_dashboard_data(self, data: dict):
        """Apply /dashboard response to state. Called with state lock held."""
        self.connected = True
        self.api_error = ""
        self.last_refresh = time.strftime("%H:%M:%S")

        ar = data.get("auto_reply", {})
        self.auto_reply_enabled = ar.get("enabled", False)
        self.chat_count = ar.get("chat_count", 0)
        self.chat_ids = [str(c) for c in ar.get("chat_ids", [])]
        self.recent_replies = ar.get("recent_replies", 0)

        features = data.get("features", {})
        self.feat_late_night_mode = features.get("late_night_mode", False)
        self.feat_strategic_silence = features.get("strategic_silence", False)
        self.feat_quote_reply = features.get("quote_reply", True)
        self.feat_smart_reactions = features.get("smart_reactions", True)
        self.feat_message_editing = features.get("message_editing", True)
        self.feat_gif_sticker_reply = features.get("gif_sticker_reply", True)
        self.feat_typing_awareness = features.get("typing_awareness", False)
        self.feat_online_status_aware = features.get("online_status_aware", True)
        self.feat_proactive_enabled = features.get("proactive_enabled", False)
        self.feat_proactive_morning = features.get("proactive_morning", self.feat_proactive_morning)
        self.feat_proactive_night = features.get("proactive_night", self.feat_proactive_night)

        self.active_features = sum(1 for k in FEATURE_KEYS
                                   if getattr(self, f"feat_{k}", False))

        engines = data.get("engines", {})
        self.engines_count = len(engines)
        self.engines_data = [
            {
                "name": k.replace("_", " ").title(),
                "key": k,
                "fn_count": str(v.get("functions", 0)),
                "status": v.get("status", "unknown"),
            }
            for k, v in engines.items()
        ]

        models = data.get("models", {})
        self.sklearn_models = [
            {
                "name": _safe_str(m.get("name", "?")),
                "classifier_type": _safe_str(m.get("classifier_type", "?")),
                "accuracy": str(round(m.get("accuracy", 0) * 100)) + "%" if isinstance(m.get("accuracy"), float) else str(m.get("accuracy", 0)),
                "class_count": str(m.get("class_count", 0)),
                "training_size": str(m.get("training_size", 0)),
            }
            for m in models.get("sklearn", [])
        ]
        self.neural_models = [
            {
                "name": _safe_str(m.get("name", "?")),
                "type": _safe_str(m.get("type", "?")),
                "accuracy": str(round(m.get("accuracy", 0) * 100)) + "%" if isinstance(m.get("accuracy"), float) else str(m.get("accuracy", 0)),
                "class_count": str(m.get("class_count", 0)),
            }
            for m in models.get("neural", [])
        ]

        self.media_ai_available = data.get("media_ai", False)
        media_status = data.get("media_ai_status") or {}
        self.media_ai_data = [
            {
                "key": k,
                "name": k.replace("_", " ").title(),
                "available": "yes" if v.get("available", False) else "no",
                "backend": _safe_str(v.get("backend", "")),
            }
            for k, v in media_status.items()
        ]

        self.advanced_intel = data.get("advanced_intel", False)

        raw_activity = data.get("recent_activity", [])
        self.activity_log = [
            {
                "time": _safe_str(e.get("timestamp", e.get("time", ""))),
                "message": _safe_str(e.get("message", e.get("detail",
                           e.get("action", str(e))))),
            }
            for e in raw_activity
        ]

    def _apply_status_data(self, status: dict):
        """Apply /auto-reply/status response. Called with state lock held."""
        self.delay_min = status.get("delay_min", 5)
        self.delay_max = status.get("delay_max", 30)
        self.context_messages = status.get("context_messages", 15)
        self.delay_min_input = str(self.delay_min)
        self.delay_max_input = str(self.delay_max)
        self.proactive_max_per_day = status.get("proactive_max_per_day",
                                                 self.proactive_max_per_day)
        self.proactive_max_input = str(self.proactive_max_per_day)

    def tick_refresh(self):
        """Synchronous refresh — only used for initial on_load."""
        data = _api("GET", "/dashboard")
        if not data:
            self.connected = False
            self.api_error = "Cannot connect to telegram_api.py"
            return
        self._apply_dashboard_data(data)
        self._tick_count += 1
        if self._tick_count % 3 == 0:
            status = _api("GET", "/auto-reply/status")
            if status:
                self._apply_status_data(status)

    def on_load(self):
        """Page load — fetch data then start polling loops."""
        self.tick_refresh()
        self.load_chats()
        return [
            rx.call_script(
                "new Promise(r => setTimeout(() => r('t'), 8000))",
                callback=DashboardState.tick_poll,
            ),
            rx.call_script(
                "new Promise(r => setTimeout(() => r('t'), 5000))",
                callback=DashboardState.msg_tick,
            ),
        ]

    @rx.event(background=True)
    async def tick_poll(self, result: str = ""):
        """Background polling — does NOT hold the state lock during HTTP IO.
        User clicks are never blocked by this method."""
        # IO outside state lock
        data = await asyncio.to_thread(_api, "GET", "/dashboard")

        tick = 0
        async with self:
            if data:
                self._apply_dashboard_data(data)
                self._tick_count += 1
                tick = self._tick_count
            else:
                self.connected = False
                self.api_error = "Cannot connect to telegram_api.py"

        # Staggered secondary fetch (also outside lock)
        if tick and tick % 3 == 0:
            status = await asyncio.to_thread(_api, "GET", "/auto-reply/status")
            if status:
                async with self:
                    self._apply_status_data(status)

        # Re-schedule
        yield rx.call_script(
            "new Promise(r => setTimeout(() => r('t'), 8000))",
            callback=DashboardState.tick_poll,
        )

    # ══════════════════════════════════════════════════════════
    # FEATURE TOGGLES
    # ══════════════════════════════════════════════════════════

    def toggle_auto_reply(self):
        # Optimistic: flip immediately, revert on failure
        new_state = not self.auto_reply_enabled
        self.auto_reply_enabled = new_state
        yield
        result = _api("POST", "/auto-reply/toggle", {"enabled": new_state})
        if not result:
            self.auto_reply_enabled = not new_state

    def toggle_feature(self, feature_key: str):
        # Optimistic: flip immediately
        current = getattr(self, f"feat_{feature_key}", False)
        new_val = not current
        setattr(self, f"feat_{feature_key}", new_val)
        self.active_features = sum(1 for k in FEATURE_KEYS
                                   if getattr(self, f"feat_{k}", False))
        yield
        result = _api("PUT", "/auto-reply/features", {feature_key: new_val})
        if not result:
            setattr(self, f"feat_{feature_key}", current)
            self.active_features = sum(1 for k in FEATURE_KEYS
                                       if getattr(self, f"feat_{k}", False))

    def toggle_all_on(self):
        # Optimistic: flip all immediately
        for k in FEATURE_KEYS:
            setattr(self, f"feat_{k}", True)
        self.active_features = len(FEATURE_KEYS)
        yield
        _api("PUT", "/auto-reply/features", {k: True for k in FEATURE_KEYS})

    def toggle_all_off(self):
        # Optimistic: flip all immediately
        for k in FEATURE_KEYS:
            setattr(self, f"feat_{k}", False)
        self.active_features = 0
        yield
        _api("PUT", "/auto-reply/features", {k: False for k in FEATURE_KEYS})

    # ══════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════

    def set_delay_min_input(self, val: str):
        self.delay_min_input = val

    def set_delay_max_input(self, val: str):
        self.delay_max_input = val

    def save_delay(self):
        try:
            lo = int(self.delay_min_input)
            hi = int(self.delay_max_input)
        except ValueError:
            return
        result = _api("PUT", "/auto-reply/delay", {"delay_min": lo, "delay_max": hi})
        if result:
            self.delay_min = lo
            self.delay_max = hi

    def set_proactive_max_input(self, val: str):
        self.proactive_max_input = val

    def save_proactive_max(self):
        try:
            val = int(self.proactive_max_input)
        except ValueError:
            return
        result = _api("PUT", "/auto-reply/features", {"proactive_max_per_day": val})
        if result:
            self.proactive_max_per_day = val

    def set_system_prompt(self, val: str):
        self.system_prompt = val

    def save_prompt(self):
        if self.system_prompt:
            _api("PUT", "/auto-reply/prompt", {"system_prompt": self.system_prompt})

    # ══════════════════════════════════════════════════════════
    # WHITELIST
    # ══════════════════════════════════════════════════════════

    def set_wl_input(self, val: str):
        self.wl_input = val

    def add_to_whitelist(self):
        if self.wl_input.strip():
            result = _api("POST", f"/auto-reply/whitelist/add?chat_id={self.wl_input.strip()}")
            if result:
                self.chat_ids = [str(c) for c in result.get("chat_ids", [])]
                self.chat_count = len(self.chat_ids)
                self.wl_input = ""

    def remove_from_whitelist(self, chat_id: str):
        result = _api("DELETE", f"/auto-reply/whitelist/remove?chat_id={chat_id}")
        if result:
            self.chat_ids = [str(c) for c in result.get("chat_ids", [])]
            self.chat_count = len(self.chat_ids)

    # ══════════════════════════════════════════════════════════
    # CHATS (normalized to: name, chat_id, chat_type)
    # ══════════════════════════════════════════════════════════

    def load_chats(self):
        self.chats_loading = True
        data = _api("GET", "/chats", params={"limit": 30})
        self.chats_loading = False
        if not data:
            self.chats_list = []
            return
        raw = data if isinstance(data, list) else data.get("chats", data.get("result", []))
        normalized = []
        for c in raw[:30]:
            # Build name: users have first_name/last_name, channels have title
            if c.get("title"):
                name = _safe_str(c["title"])
            elif c.get("first_name"):
                fn = _safe_str(c["first_name"])
                ln = _safe_str(c.get("last_name", ""))
                name = f"{fn} {ln}".strip() if ln else fn
            elif c.get("username"):
                name = "@" + _safe_str(c["username"])
            elif c.get("name"):
                name = _safe_str(c["name"])
            else:
                name = str(c.get("id", "Unknown"))
            cid = str(c.get("id", "?"))
            normalized.append({
                "name": name,
                "initial": name[0].upper() if name else "?",
                "chat_id": cid,
                "chat_type": _safe_str(c.get("type", "")),
                "unread": str(c.get("unread_count", 0)),
                "last_msg": _safe_str(c.get("last_message", ""))[:50],
                "photo_url": f"{API_BASE}/users/{cid}/avatar",
            })
        self.chats_list = normalized

    # ══════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════

    def set_analysis_chat_id(self, val: str):
        self.analysis_chat_id = val

    def run_analysis(self):
        if not self.analysis_chat_id.strip():
            return
        self.analysis_loading = True
        cid = self.analysis_chat_id.strip()
        result = None
        for path in ["/engine/analyze-v5", "/engine/analyze-v4",
                     "/nlp/analyze-v3", "/nlp/analyze-v2", "/nlp/analyze"]:
            result = _api("GET", path, params={"chat_id": cid}, timeout=TIMEOUT_ACTION)
            if result and "error" not in result:
                break
        self.analysis_loading = False
        self.analysis_result = _flatten_dict(result) if result else [
            {"key": "error", "value": "Analysis failed — check chat ID"}
        ]

    # ══════════════════════════════════════════════════════════
    # RELATIONSHIP HEALTH
    # ══════════════════════════════════════════════════════════

    def set_health_chat_id(self, val: str):
        self.health_chat_id = val

    def run_health(self):
        if not self.health_chat_id.strip():
            return
        self.health_loading = True
        result = _api("GET", "/relationship/health",
                       params={"chat_id": self.health_chat_id.strip()},
                       timeout=TIMEOUT_ACTION)
        self.health_loading = False
        self.health_result = _flatten_dict(result) if result else [
            {"key": "error", "value": "Health check failed"}
        ]

    # ══════════════════════════════════════════════════════════
    # PSYCHOLOGICAL ANALYSIS
    # ══════════════════════════════════════════════════════════

    def set_psych_chat_id(self, val: str):
        self.psych_chat_id = val

    def run_psych(self):
        if not self.psych_chat_id.strip():
            return
        self.psych_loading = True
        result = _api("GET", "/engine/psychological-analysis",
                       params={"chat_id": self.psych_chat_id.strip()},
                       timeout=TIMEOUT_ACTION)
        self.psych_loading = False
        self.psych_result = _flatten_dict(result) if result else [
            {"key": "error", "value": "Analysis failed"}
        ]

    # ══════════════════════════════════════════════════════════
    # RL INSIGHTS
    # ══════════════════════════════════════════════════════════

    def load_rl_all(self):
        self.rl_loading = True
        result = _api("GET", "/rl/insights/all", timeout=TIMEOUT_ACTION)
        self.rl_loading = False
        self.rl_result = _flatten_dict(result) if result else [
            {"key": "status", "value": "RL engine not loaded or no data"}
        ]

    # ══════════════════════════════════════════════════════════
    # ENGINE DETAILS (functions stored as list of strings)
    # ══════════════════════════════════════════════════════════

    def load_engine_details(self):
        data = _api("GET", "/engine/status")
        if not data:
            self.engine_details = []
            return
        engines = data.get("engines", {})
        details = []
        for k, v in engines.items():
            fns = v.get("functions", [])
            fn_list = fns if isinstance(fns, list) else []
            details.append({
                "name": k.replace("_", " ").title(),
                "key": k,
                "fn_list": ", ".join(fn_list) if fn_list else "",
                "fn_count": str(len(fn_list)),
                "status": "loaded",
            })
        self.engine_details = details

    # ══════════════════════════════════════════════════════════
    # INSTRUCTIONS
    # ══════════════════════════════════════════════════════════

    def load_instructions(self):
        data = _api("GET", "/auto-reply/instructions")
        if data:
            insts = data.get("instructions", {})
            self.instructions_data = [
                {"chat_id": str(k), "instructions": str(v)}
                for k, v in insts.items()
            ]
        else:
            self.instructions_data = []

    def set_inst_chat_id(self, val: str):
        self.inst_chat_id = val

    def set_inst_text(self, val: str):
        self.inst_text = val

    def save_instruction(self):
        if self.inst_chat_id.strip() and self.inst_text.strip():
            result = _api("PUT", "/auto-reply/instructions", {
                "chat_id": self.inst_chat_id.strip(),
                "instructions": self.inst_text.strip(),
            })
            if result:
                self.load_instructions()
                self.inst_chat_id = ""
                self.inst_text = ""

    def remove_instruction(self, chat_id: str):
        _api("DELETE", f"/auto-reply/instructions?chat_id={chat_id}")
        self.load_instructions()

    # ══════════════════════════════════════════════════════════
    # DL STATUS & TRAINING
    # ══════════════════════════════════════════════════════════

    def load_dl_status(self):
        data = _api("GET", "/dl/status")
        self.dl_status_data = _flatten_dict(data) if data else [
            {"key": "status", "value": "DL modules not available"}
        ]

    def preload_models(self):
        result = _api("POST", "/dl/preload", timeout=30)
        self.training_result = ("Models preloaded successfully"
                                if result and result.get("success")
                                else "Preload failed — check logs")

    def train_sklearn(self):
        self.training_running = True
        self.training_result = ""
        result = _api("POST", "/dl/train",
                       params={"task": "all", "include_neural": False},
                       timeout=60)
        self.training_running = False
        self.training_result = ("sklearn training complete"
                                if result and result.get("success")
                                else "Training failed")

    def train_all(self):
        self.training_running = True
        self.training_result = ""
        result = _api("POST", "/dl/train",
                       params={"task": "all", "include_neural": True},
                       timeout=120)
        self.training_running = False
        self.training_result = ("Full training complete (sklearn + neural)"
                                if result and result.get("success")
                                else "Training failed")

    # ══════════════════════════════════════════════════════════
    # FULL LOG (normalized to: time, chat, action, detail)
    # ══════════════════════════════════════════════════════════

    def load_full_log(self):
        data = _api("GET", "/auto-reply/log?limit=50")
        if not data:
            self.full_log = []
            return
        raw = data.get("log", [])
        normalized = []
        for e in raw:
            ts = _safe_str(e.get("timestamp", e.get("time", "")))
            if len(ts) > 16:
                ts = ts[11:19]
            normalized.append({
                "time": ts,
                "chat": str(e.get("chat_id", e.get("chat", "?"))),
                "action": _safe_str(e.get("action", e.get("type", "?"))),
                "detail": _safe_str(e.get("detail", e.get("message", "")))[:80],
            })
        self.full_log = normalized

    # ══════════════════════════════════════════════════════════
    # ACCOUNT
    # ══════════════════════════════════════════════════════════

    def load_my_info(self):
        data = _api("GET", "/me")
        self.my_info = _flatten_dict(data) if data else [
            {"key": "error", "value": "Could not fetch account info"}
        ]

    # ══════════════════════════════════════════════════════════
    # MESSENGER — Chat selection, messages, sending
    # ══════════════════════════════════════════════════════════

    def open_chat(self, chat_id: str):
        """Select a chat and load its messages — optimistic header."""
        self.active_chat_id = chat_id
        self.messages = []  # Clear old messages
        self.active_chat_status = ""
        # Find name from chats_list
        for c in self.chats_list:
            if c.get("chat_id") == chat_id:
                self.active_chat_name = c.get("name", "Chat")
                self.active_chat_initial = c.get("initial", "?")
                self.active_chat_photo = c.get("photo_url", "")
                break
        else:
            self.active_chat_name = chat_id
            self.active_chat_initial = chat_id[0] if chat_id else "?"
            self.active_chat_photo = f"{API_BASE}/users/{chat_id}/avatar"
        yield  # Show header immediately
        self.load_messages()
        self._load_user_status()

    def load_messages(self):
        """Load message history for active chat (synchronous, used by open_chat/send)."""
        if not self.active_chat_id:
            return
        self.messages_loading = True
        data = _api("GET", f"/chats/{self.active_chat_id}/messages",
                     params={"limit": 50})
        self.messages_loading = False
        if not data:
            self.messages = []
            return
        self.messages = _process_messages_data(data)

    def _load_user_status(self):
        """Get online status of the active chat user."""
        if not self.active_chat_id:
            return
        data = _api("GET", f"/users/{self.active_chat_id}/status")
        if data:
            self.active_chat_status = _safe_str(data.get("status", ""))
        else:
            self.active_chat_status = ""

    def set_msg_input(self, val: str):
        self.msg_input = val

    def send_message(self):
        """Send a text message to the active chat — optimistic UI."""
        if not self.active_chat_id or not self.msg_input.strip():
            return
        text = self.msg_input.strip()
        # Optimistic: append message to UI instantly
        self.messages.append({
            "id": "", "text": text, "out": "true",
            "sender": "You", "date": time.strftime("%H:%M"),
            "has_media": "false", "media_type": "", "reply_to": "",
        })
        self.msg_input = ""
        yield
        # Fire API in background, then refresh for real data
        _api("POST", f"/chats/{self.active_chat_id}/messages",
             json_data={"message": text})
        self.load_messages()

    def send_reaction(self, msg_id: str):
        """Send a heart reaction to a message — fire and forget."""
        if not self.active_chat_id:
            return
        yield  # Return control to UI immediately
        _api("POST",
             f"/chats/{self.active_chat_id}/messages/{msg_id}/reaction",
             json_data={"emoji": "\u2764\ufe0f"})

    def delete_message(self, msg_id: str):
        """Delete a message in the active chat — optimistic."""
        if not self.active_chat_id:
            return
        # Optimistic: remove from UI immediately
        self.messages = [m for m in self.messages if m.get("id") != msg_id]
        yield
        _api("DELETE", f"/chats/{self.active_chat_id}/messages/{msg_id}")

    def mark_read(self):
        """Mark active chat as read — optimistic."""
        if not self.active_chat_id:
            return
        # Optimistic: clear unread count in sidebar immediately
        for c in self.chats_list:
            if c.get("chat_id") == self.active_chat_id:
                c["unread"] = "0"
                break
        yield
        _api("POST", f"/chats/{self.active_chat_id}/read")

    def set_msg_search_query(self, val: str):
        self.msg_search_query = val

    def search_messages(self):
        """Search messages in the active chat."""
        if not self.active_chat_id or not self.msg_search_query.strip():
            return
        data = _api("GET", f"/chats/{self.active_chat_id}/search",
                     params={"query": self.msg_search_query.strip()})
        if not data:
            self.msg_search_results = []
            return
        raw = data.get("messages", [])
        self.msg_search_results = [
            {
                "id": str(m.get("id", "")),
                "text": _safe_str(m.get("text", ""))[:100],
                "date": _safe_str(m.get("date", "")),
                "sender": _safe_str(m.get("sender_name", "")),
            }
            for m in raw[:20]
        ]

    # ══════════════════════════════════════════════════════════
    # CHAT ANALYTICS
    # ══════════════════════════════════════════════════════════

    def load_chat_analytics(self):
        """Load analytics for the active chat."""
        if not self.active_chat_id:
            return
        self.analytics_loading = True
        data = _api("GET", f"/analytics/{self.active_chat_id}")
        self.analytics_loading = False
        self.chat_analytics = _flatten_dict(data) if data else [
            {"key": "error", "value": "Analytics not available"}
        ]

    # ══════════════════════════════════════════════════════════
    # MESSENGER POLLING — @rx.background (doesn't block UI!)
    # ══════════════════════════════════════════════════════════

    @rx.event(background=True)
    async def msg_tick(self, result: str = ""):
        """Background message polling — doesn't hold state lock during IO."""
        # Read needed state under lock
        async with self:
            chat_id = self.active_chat_id
            tab = self.active_tab

        if chat_id and tab == "messenger":
            # IO outside state lock — UI stays responsive
            data = await asyncio.to_thread(
                _api, "GET", f"/chats/{chat_id}/messages", None, {"limit": 50},
            )
            if data:
                messages = _process_messages_data(data)
                async with self:
                    self.messages = messages
                    self.messages_loading = False

        # Re-schedule
        yield rx.call_script(
            "new Promise(r => setTimeout(() => r('t'), 5000))",
            callback=DashboardState.msg_tick,
        )

    # ══════════════════════════════════════════════════════════
    # BEHAVIOR MIRRORING
    # ══════════════════════════════════════════════════════════

    def load_behavioral_patterns(self):
        """Load behavioral pattern analysis for active chat."""
        if not self.active_chat_id:
            return
        data = _api("GET", "/engine/behavioral-patterns",
                     params={"chat_id": self.active_chat_id},
                     timeout=TIMEOUT_ACTION)
        if data:
            patterns = data.get("patterns", [])
            self.mirror_data = [
                {
                    "pattern": _safe_str(p.get("pattern_name", "unknown")),
                    "confidence": str(round(p.get("confidence", 0) * 100)),
                    "indicators": _safe_str(
                        ", ".join(p.get("indicators", [])[:3])
                    ),
                }
                for p in patterns
            ]
        else:
            self.mirror_data = []

    def set_mirror_mode(self, mode: str):
        """Set behavior mirroring mode — optimistic."""
        self.mirror_mode = mode
        yield  # Show mode change immediately
        if self.active_chat_id and mode != "off":
            instruction_map = {
                "subtle": "Mirror their communication style subtly. Match energy and tone without being obvious.",
                "match": "Match their exact communication style, energy, and assertiveness level. If they are short, be short. If they are enthusiastic, match it.",
                "assertive": "If they are aggressive or dismissive, stand your ground firmly. Do not be a pushover. Match their energy and push back with confidence. Be direct and assertive.",
            }
            inst = instruction_map.get(mode, "")
            if inst:
                _api("PUT", "/auto-reply/instructions", {
                    "chat_id": self.active_chat_id,
                    "instructions": inst,
                }, timeout=TIMEOUT_ACTION)
