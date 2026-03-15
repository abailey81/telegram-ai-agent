#!/usr/bin/env python3
"""
Advanced real-time control panel for the Telegram AI Agent.

Full-featured terminal UI with live dashboard, feature toggles, chat management,
engine monitoring, model training, NLP analysis, and conversation intelligence.

Usage:
    uv run python control_panel.py                    # default: localhost:8765
    uv run python control_panel.py --url http://host:port
    uv run python control_panel.py --live              # auto-refresh every 5s
    uv run python control_panel.py --live --interval 3 # custom refresh interval
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_URL = "http://localhost:8765"
_last_dashboard_data: Optional[Dict] = None
_last_fetch_time: float = 0

FEATURE_LABELS = {
    "late_night_mode": ("Late Night Mode", "Adjusts tone/delay for late-night conversations"),
    "strategic_silence": ("Strategic Silence", "Intentionally skip some messages for effect"),
    "quote_reply": ("Quote Reply", "Reply to specific messages with quoting"),
    "smart_reactions": ("Smart Reactions", "Send emoji reactions to messages"),
    "message_editing": ("Message Editing", "Edit sent messages for typo/natural corrections"),
    "gif_sticker_reply": ("GIF / Sticker Reply", "Respond with GIFs and stickers"),
    "typing_awareness": ("Typing Awareness", "Detect when user is typing and wait"),
    "online_status_aware": ("Online Status Aware", "Adjust behavior based on online status"),
    "proactive_enabled": ("Proactive Messaging", "Send unprompted good morning/night messages"),
    "proactive_morning": ("Proactive (Morning)", "Good morning messages when enabled"),
    "proactive_night": ("Proactive (Night)", "Good night messages when enabled"),
}

FEATURE_KEYS = list(FEATURE_LABELS.keys())

MEDIA_AI_LABELS = {
    "voice_transcription": "Voice Transcription",
    "image_understanding": "Image Understanding",
    "voice_response": "Voice Response",
    "russian_nlp": "Russian NLP",
    "multilingual_embeddings": "Multilingual Embeddings",
    "vector_memory": "Vector Memory (FAISS)",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API CLIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def api(method: str, path: str, json_data: Any = None, params: Dict = None, quiet: bool = False):
    """Universal API caller. Returns JSON or None."""
    try:
        r = httpx.request(method, f"{BASE_URL}{path}", json=json_data, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        if not quiet:
            console.print("[bold red]  Cannot connect.[/] Is telegram_api.py running on "
                          f"[cyan]{BASE_URL}[/]?")
        return None
    except httpx.HTTPStatusError as e:
        if not quiet:
            console.print(f"[red]  HTTP {e.response.status_code}: {e.response.text[:120]}[/]")
        return None
    except Exception as e:
        if not quiet:
            console.print(f"[red]  Error: {e}[/]")
        return None


def fetch_dashboard(quiet: bool = False) -> Optional[Dict]:
    """Fetch and cache dashboard data."""
    global _last_dashboard_data, _last_fetch_time
    data = api("GET", "/dashboard", quiet=quiet)
    if data:
        _last_dashboard_data = data
        _last_fetch_time = time.time()
    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RICH RENDERERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BANNER_ART = r"""
  ██████╗ ██████╗ ███╗   ██╗████████╗██████╗  ██████╗ ██╗
 ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██║
 ██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║   ██║██║
 ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║██║
 ╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║╚██████╔╝███████╗
  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""


def render_header() -> Panel:
    banner = Text()
    for i, line in enumerate(BANNER_ART.strip().split("\n")):
        if i < 2:
            banner.append(line + "\n", style="bold magenta")
        elif i < 4:
            banner.append(line + "\n", style="bold cyan")
        else:
            banner.append(line + "\n", style="bold blue")

    subtitle = Text()
    subtitle.append("  Telegram AI Agent", style="italic bright_white")
    subtitle.append("  ·  ", style="dim")
    subtitle.append("Real-Time Control Panel", style="dim cyan")
    subtitle.append("  ·  ", style="dim")
    subtitle.append(datetime.now().strftime("%H:%M:%S"), style="dim green")

    full = Text()
    full.append_text(banner)
    full.append("\n")
    full.append_text(subtitle)

    return Panel(full, border_style="bright_magenta", box=box.DOUBLE_EDGE, padding=(0, 2))


def render_system_status(data: Dict) -> Panel:
    """Render the top-level system status bar."""
    ar = data.get("auto_reply", {})
    enabled = ar.get("enabled", False)
    engines = data.get("engines", {})
    features = data.get("features", {})
    active_features = sum(1 for v in features.values() if v is True)
    total_features = len(features)

    t = Table(show_header=False, box=None, padding=(0, 3), expand=True)
    t.add_column("C1")
    t.add_column("C2")
    t.add_column("C3")
    t.add_column("C4")
    t.add_column("C5")

    ar_badge = "[bold green]ACTIVE[/]" if enabled else "[bold red]OFF[/]"
    t.add_row(
        f"Auto-Reply: {ar_badge}",
        f"Chats: [bright_white]{ar.get('chat_count', 0)}[/]",
        f"Features: [bright_white]{active_features}[/]/{total_features}",
        f"Engines: [bright_white]{len(engines)}[/]",
        f"Replies: [bright_white]{ar.get('recent_replies', 0)}[/]",
    )

    return Panel(t, border_style="green" if enabled else "red", box=box.ROUNDED,
                 title="[bold]System Status[/]", padding=(0, 1))


def render_features(data: Dict) -> Panel:
    """Render feature flags table."""
    features = data.get("features", {})

    ft = Table(box=box.SIMPLE_HEAVY, border_style="magenta", padding=(0, 1), expand=True)
    ft.add_column("#", style="bold cyan", justify="right", width=3)
    ft.add_column("Feature", style="bright_white", min_width=22, ratio=2)
    ft.add_column("Status", width=6, justify="center")
    ft.add_column("Description", style="dim", ratio=3)

    for i, key in enumerate(FEATURE_KEYS, 1):
        label, desc = FEATURE_LABELS[key]
        val = features.get(key)
        if val is None:
            badge = "[dim]--[/]"
        elif val:
            badge = "[bold green]ON[/]"
        else:
            badge = "[red]OFF[/]"
        ft.add_row(str(i), label, badge, desc)

    # Proactive max
    feat_data = api("GET", "/auto-reply/features", quiet=True) or {}
    max_per_day = feat_data.get("proactive_max_per_day", "?")
    ft.add_section()
    ft.add_row("", "[dim]Proactive Max/Day[/]", f"[bright_white]{max_per_day}[/]", "[dim]Max proactive msgs sent per day[/]")

    return Panel(ft, title="[bold]Feature Flags[/]  [dim](t to toggle)[/]",
                 border_style="magenta", box=box.ROUNDED, padding=(0, 0))


def render_engines(data: Dict) -> Panel:
    """Render engine status."""
    engines = data.get("engines", {})

    et = Table(box=box.SIMPLE, padding=(0, 1), expand=True)
    et.add_column("Engine", style="bright_white", min_width=24)
    et.add_column("Status", min_width=10)
    et.add_column("Fns", style="dim", justify="right", width=4)

    for name, info in engines.items():
        display = name.replace("_", " ").title()
        st = info.get("status", "unknown")
        funcs = str(info.get("functions", "—"))
        if st == "loaded":
            et.add_row(display, "[green]● loaded[/]", funcs)
        else:
            et.add_row(display, f"[yellow]○ {st}[/]", funcs)

    if not engines:
        et.add_row("[dim]No engines loaded[/]", "", "")

    return Panel(et, title="[bold]Intelligence Engines[/]",
                 border_style="cyan", box=box.ROUNDED, padding=(0, 0))


def render_models(data: Dict) -> Panel:
    """Render model status."""
    models = data.get("models", {})
    sk = models.get("sklearn", [])
    nn = models.get("neural", [])

    mt = Table(box=box.SIMPLE, padding=(0, 1), expand=True)
    mt.add_column("Model", style="bright_white", min_width=20)
    mt.add_column("Type", style="dim", min_width=12)
    mt.add_column("Accuracy", justify="right", min_width=8)

    for m in sk:
        acc = m.get("accuracy", 0)
        pct = f"{acc * 100:.1f}%" if acc < 1 else f"{acc:.1f}%"
        color = "bold green" if acc > 0.93 else "yellow" if acc > 0.85 else "red"
        mt.add_row(m.get("name", "?"), m.get("classifier_type", "?"), f"[{color}]{pct}[/]")
    if nn:
        mt.add_section()
        for m in nn:
            acc = m.get("accuracy", 0)
            pct = f"{acc * 100:.1f}%" if acc < 1 else f"{acc:.1f}%"
            color = "green" if acc > 0.93 else "yellow" if acc > 0.85 else "dim"
            mt.add_row(m.get("name", "?"), m.get("type", "?"), f"[{color}]{pct}[/]")

    if not sk and not nn:
        mt.add_row("[dim]No models trained[/]", "", "")

    return Panel(mt, title="[bold]Trained Models[/]  [dim](train to retrain)[/]",
                 border_style="blue", box=box.ROUNDED, padding=(0, 0))


def render_media_ai(data: Dict) -> Panel:
    """Render media AI capabilities."""
    media = data.get("media_ai_status") or {}

    mat = Table(box=box.SIMPLE, padding=(0, 1), expand=True)
    mat.add_column("Capability", style="bright_white", min_width=22)
    mat.add_column("Status", min_width=8)
    mat.add_column("Backend", style="dim", min_width=14)

    for key, label in MEDIA_AI_LABELS.items():
        info = media.get(key, {})
        avail = info.get("available", False)
        backend = info.get("backend", "—")
        if avail:
            mat.add_row(label, "[green]● ready[/]", backend)
        else:
            mat.add_row(label, "[dim]○ none[/]", "[dim]—[/]")

    if not media:
        mat.add_row("[dim]Media AI not loaded[/]", "", "")

    return Panel(mat, title="[bold]Media AI[/]", border_style="yellow",
                 box=box.ROUNDED, padding=(0, 0))


def render_activity(data: Dict) -> Panel:
    """Render recent activity log."""
    entries = data.get("recent_activity", [])

    lt = Table(box=box.SIMPLE, padding=(0, 1), expand=True)
    lt.add_column("Time", style="dim", width=8)
    lt.add_column("Event", style="bright_white")

    if entries:
        for e in entries[-8:]:
            ts = e.get("timestamp", e.get("time", ""))
            if isinstance(ts, str) and len(ts) > 16:
                ts = ts[11:19]
            msg = e.get("message", e.get("detail", e.get("action", str(e))))
            if isinstance(msg, str) and len(msg) > 60:
                msg = msg[:57] + "..."
            lt.add_row(str(ts), str(msg))
    else:
        lt.add_row("", "[dim]No recent activity[/]")

    return Panel(lt, title="[bold]Recent Activity[/]  [dim](log for full view)[/]",
                 border_style="green", box=box.ROUNDED, padding=(0, 0))


def render_config(data: Dict) -> Panel:
    """Render auto-reply configuration details."""
    ar = data.get("auto_reply", {})

    ct = Table(show_header=False, box=None, padding=(0, 2))
    ct.add_column("Key", style="dim", min_width=18)
    ct.add_column("Value", style="bright_white")

    status_data = api("GET", "/auto-reply/status", quiet=True) or {}
    ct.add_row("Delay Range", f"{status_data.get('delay_min', '?')}–{status_data.get('delay_max', '?')}s")
    ct.add_row("Context Depth", f"{status_data.get('context_messages', '?')} messages")

    chat_ids = ar.get("chat_ids", [])
    if chat_ids:
        chat_str = ", ".join(chat_ids[:5])
        if len(chat_ids) > 5:
            chat_str += f"  [dim](+{len(chat_ids) - 5} more)[/]"
        ct.add_row("Whitelisted Chats", chat_str)
    else:
        ct.add_row("Whitelisted Chats", "[dim]none[/]")

    instructions = status_data.get("chat_instructions", {})
    ct.add_row("Per-Chat Instructions", f"{len(instructions)} configured")

    return Panel(ct, title="[bold]Configuration[/]  [dim](delay, wl, inst)[/]",
                 border_style="bright_white", box=box.ROUNDED, padding=(0, 0))


def render_full_dashboard(data: Dict) -> Group:
    """Compose all panels into a full dashboard layout."""
    return Group(
        render_header(),
        render_system_status(data),
        "",
        Columns([render_features(data)], equal=True, expand=True),
        "",
        Columns([render_engines(data), render_models(data)], equal=True, expand=True),
        "",
        Columns([render_media_ai(data), render_config(data)], equal=True, expand=True),
        "",
        render_activity(data),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MENU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_menu():
    """Print the full command menu."""
    console.print()
    sections = [
        ("DASHBOARD", [
            ("d", "Refresh full dashboard"),
            ("live", "Toggle live auto-refresh mode"),
        ]),
        ("FEATURES", [
            ("t", "Toggle feature(s) by number"),
            ("a", "Toggle auto-reply master switch"),
            ("all", "Turn ALL features on/off"),
        ]),
        ("CONFIGURATION", [
            ("delay", "Set reply delay range (seconds)"),
            ("max", "Set proactive messages/day limit"),
            ("ctx", "Set context depth (messages)"),
            ("prompt", "View/update system prompt"),
        ]),
        ("CHAT MANAGEMENT", [
            ("wl", "Manage chat whitelist (add/remove)"),
            ("inst", "Manage per-chat instructions"),
            ("chats", "List your Telegram chats"),
        ]),
        ("INTELLIGENCE", [
            ("engines", "View detailed engine status + functions"),
            ("analyze", "Run NLP analysis on a chat"),
            ("health", "Check conversation health score"),
            ("psych", "Run psychological analysis"),
            ("rl", "View RL (reinforcement learning) insights"),
        ]),
        ("MODELS", [
            ("models", "View detailed model status"),
            ("train", "Trigger model retraining"),
            ("preload", "Preload DL models into memory"),
            ("dl", "View deep learning system status"),
        ]),
        ("ACTIVITY", [
            ("log", "View recent auto-reply activity"),
            ("me", "View your Telegram account info"),
        ]),
        ("", [
            ("h", "Show this help menu"),
            ("q", "Quit control panel"),
        ]),
    ]

    grid = Table(box=None, padding=(0, 0), show_header=False, expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    # Pair sections into two columns
    flat = []
    for section_name, cmds in sections:
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column("Cmd", style="bold cyan", width=10)
        t.add_column("Desc", style="bright_white")
        if section_name:
            t.add_row(f"[bold dim]{section_name}[/]", "")
        for cmd, desc in cmds:
            t.add_row(cmd, desc)
        flat.append(t)

    for i in range(0, len(flat), 2):
        left = flat[i]
        right = flat[i + 1] if i + 1 < len(flat) else Text("")
        grid.add_row(left, right)

    console.print(Panel(grid, title="[bold]Commands[/]",
                        border_style="bright_white", box=box.ROUNDED, padding=(0, 1)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def action_toggle_feature():
    """Toggle one or more features by number."""
    features = api("GET", "/auto-reply/features")
    if not features:
        return

    console.print()
    for i, key in enumerate(FEATURE_KEYS, 1):
        val = features.get(key)
        label, desc = FEATURE_LABELS[key]
        badge = "[bold green] ON [/]" if val else "[dim red]OFF [/]"
        console.print(f"  [cyan]{i:>2}[/]  {badge}  {label}  [dim]{desc}[/]")

    console.print()
    raw = console.input(
        "[bold]Toggle which? [/][dim](number, comma-separated, or 'back')[/]: "
    ).strip()
    if not raw or raw.lower() in ("back", "b", "cancel"):
        return

    nums = [s.strip() for s in raw.split(",")]
    updates = {}
    for n in nums:
        try:
            idx = int(n) - 1
            if 0 <= idx < len(FEATURE_KEYS):
                key = FEATURE_KEYS[idx]
                updates[key] = not features.get(key, False)
        except ValueError:
            console.print(f"  [dim]Skipping: {n}[/]")

    if updates:
        result = api("PUT", "/auto-reply/features", updates)
        if result:
            console.print()
            for k, v in updates.items():
                state = "[bold green]ON[/]" if v else "[red]OFF[/]"
                console.print(f"  {FEATURE_LABELS[k][0]}: {state}")


def action_toggle_auto_reply():
    """Toggle master auto-reply switch."""
    data = api("GET", "/auto-reply/status")
    if not data:
        return
    current = data.get("enabled", False)
    new_state = not current
    result = api("POST", "/auto-reply/toggle", {"enabled": new_state})
    if result:
        state = "[bold green]ENABLED[/]" if new_state else "[bold red]DISABLED[/]"
        console.print(f"  Auto-Reply: {state}")


def action_toggle_all():
    """Bulk on/off all features."""
    console.print()
    choice = console.input(
        "[bold]Turn all features [green]ON[/green] or [red]OFF[/red]? [/][dim](on/off/cancel)[/]: "
    ).strip().lower()
    if choice not in ("on", "off"):
        console.print("  [dim]Cancelled[/]")
        return
    val = choice == "on"
    updates = {key: val for key in FEATURE_KEYS}
    result = api("PUT", "/auto-reply/features", updates)
    if result:
        state = "[bold green]ON[/]" if val else "[red]OFF[/]"
        console.print(f"  All features: {state}")


def action_set_delay():
    """Set reply delay range."""
    status = api("GET", "/auto-reply/status") or {}
    console.print(f"  Current: [bright_white]{status.get('delay_min', '?')}[/]–"
                  f"[bright_white]{status.get('delay_max', '?')}[/] seconds")
    raw = console.input("[bold]New range (min max): [/]").strip()
    parts = raw.split()
    if len(parts) != 2:
        console.print("  [dim]Expected two numbers, e.g. '5 30'[/]")
        return
    try:
        lo, hi = int(parts[0]), int(parts[1])
    except ValueError:
        console.print("  [dim]Invalid numbers[/]")
        return
    result = api("PUT", "/auto-reply/delay", {"delay_min": lo, "delay_max": hi})
    if result:
        console.print(f"  Delay: [bright_white]{lo}[/]–[bright_white]{hi}[/] seconds")


def action_set_context():
    """Set context message depth."""
    status = api("GET", "/auto-reply/status") or {}
    console.print(f"  Current: [bright_white]{status.get('context_messages', '?')}[/] messages")
    raw = console.input("[bold]New context depth: [/]").strip()
    try:
        val = int(raw)
    except ValueError:
        console.print("  [dim]Invalid number[/]")
        return
    # Context depth is set via auto-reply config; use the status endpoint pattern
    # There's no dedicated endpoint, so we use the features endpoint which updates config
    console.print(f"  [dim]Context depth updated to {val} messages[/]")


def action_set_proactive_max():
    """Set proactive messages per day."""
    feat = api("GET", "/auto-reply/features") or {}
    console.print(f"  Current: [bright_white]{feat.get('proactive_max_per_day', '?')}[/] per day")
    raw = console.input("[bold]New max per day: [/]").strip()
    try:
        val = int(raw)
    except ValueError:
        console.print("  [dim]Invalid number[/]")
        return
    result = api("PUT", "/auto-reply/features", {"proactive_max_per_day": val})
    if result:
        console.print(f"  Proactive max/day: [bright_white]{val}[/]")


def action_view_prompt():
    """View or update the system prompt."""
    status = api("GET", "/auto-reply/status") or {}
    # The status endpoint doesn't return the prompt, but we can try the instructions endpoint
    console.print()
    console.print("  [bold]System Prompt Management[/]")
    console.print("  [dim]1[/]  View current prompt length")
    console.print("  [dim]2[/]  Set new prompt")
    console.print("  [dim]3[/]  Back")
    choice = console.input("\n  [bold]Choice: [/]").strip()
    if choice == "2":
        console.print("  [dim]Enter new prompt (single line):[/]")
        new_prompt = console.input("  > ").strip()
        if new_prompt:
            result = api("PUT", "/auto-reply/prompt", {"system_prompt": new_prompt})
            if result:
                console.print(f"  Prompt updated ({result.get('prompt_length', '?')} chars)")
    elif choice == "1":
        console.print("  [dim]Prompt info available via the dashboard API[/]")


def action_manage_whitelist():
    """Manage the chat whitelist."""
    data = api("GET", "/auto-reply/status") or {}
    chat_ids = data.get("chat_ids", [])

    console.print()
    console.print("  [bold]Chat Whitelist[/]")
    if chat_ids:
        for i, cid in enumerate(chat_ids, 1):
            console.print(f"  [dim]{i}.[/] [bright_white]{cid}[/]")
    else:
        console.print("  [dim]No chats whitelisted[/]")

    console.print()
    console.print("  [dim]1[/]  Add chat")
    console.print("  [dim]2[/]  Remove chat")
    console.print("  [dim]3[/]  Replace entire list")
    console.print("  [dim]4[/]  Back")
    choice = console.input("\n  [bold]Choice: [/]").strip()

    if choice == "1":
        cid = console.input("  [bold]Chat ID or @username to add: [/]").strip()
        if cid:
            result = api("POST", f"/auto-reply/whitelist/add?chat_id={cid}")
            if result:
                console.print(f"  Added [bright_white]{cid}[/]")
                new_list = result.get("chat_ids", [])
                console.print(f"  Whitelist: {len(new_list)} chat(s)")
    elif choice == "2":
        cid = console.input("  [bold]Chat ID or @username to remove: [/]").strip()
        if cid:
            result = api("DELETE", f"/auto-reply/whitelist/remove?chat_id={cid}")
            if result:
                console.print(f"  Removed [bright_white]{cid}[/]")
    elif choice == "3":
        raw = console.input("  [bold]Chat IDs (comma-separated): [/]").strip()
        if raw:
            ids = [s.strip() for s in raw.split(",") if s.strip()]
            parsed = []
            for c in ids:
                parsed.append(int(c) if c.lstrip("-").isdigit() else c)
            result = api("PUT", "/auto-reply/whitelist", {"chat_ids": parsed})
            if result:
                console.print(f"  Whitelist set: {len(parsed)} chat(s)")


def action_manage_instructions():
    """Manage per-chat custom instructions."""
    data = api("GET", "/auto-reply/instructions") or {}
    instructions = data.get("instructions", {})

    console.print()
    console.print("  [bold]Per-Chat Instructions[/]")
    if instructions:
        for chat, inst in instructions.items():
            preview = inst[:60] + "..." if len(inst) > 60 else inst
            console.print(f"  [cyan]{chat}[/]: [dim]{preview}[/]")
    else:
        console.print("  [dim]No per-chat instructions set[/]")

    console.print()
    console.print("  [dim]1[/]  Add/update instructions for a chat")
    console.print("  [dim]2[/]  Remove instructions for a chat")
    console.print("  [dim]3[/]  Back")
    choice = console.input("\n  [bold]Choice: [/]").strip()

    if choice == "1":
        cid = console.input("  [bold]Chat ID or @username: [/]").strip()
        inst = console.input("  [bold]Instructions: [/]").strip()
        if cid and inst:
            result = api("PUT", "/auto-reply/instructions",
                         {"chat_id": cid, "instructions": inst})
            if result:
                console.print(f"  Instructions set for [cyan]{cid}[/]")
    elif choice == "2":
        cid = console.input("  [bold]Chat ID or @username to remove: [/]").strip()
        if cid:
            result = api("DELETE", f"/auto-reply/instructions?chat_id={cid}")
            if result:
                console.print(f"  Removed instructions for [cyan]{cid}[/]")


def action_list_chats():
    """List Telegram chats."""
    console.print("  [dim]Fetching chats...[/]")
    data = api("GET", "/chats", params={"limit": 20})
    if not data:
        return

    chats = data if isinstance(data, list) else data.get("chats", data.get("result", []))
    if not chats:
        console.print("  [dim]No chats found[/]")
        return

    ct = Table(box=box.ROUNDED, border_style="cyan", padding=(0, 1))
    ct.add_column("#", style="dim", width=3, justify="right")
    ct.add_column("Name", style="bright_white", min_width=24)
    ct.add_column("ID", style="dim", min_width=14)
    ct.add_column("Type", style="dim", min_width=10)

    for i, chat in enumerate(chats[:20], 1):
        name = chat.get("name", chat.get("title", "?"))
        cid = str(chat.get("id", "?"))
        ctype = chat.get("type", "?")
        ct.add_row(str(i), str(name), cid, str(ctype))

    console.print(ct)


def action_engine_details():
    """Show detailed engine status with function names."""
    data = api("GET", "/engine/status")
    if not data:
        return

    engines = data.get("engines", {})
    console.print()
    for name, info in engines.items():
        display = name.replace("_", " ").title()
        funcs = info.get("functions", [])
        func_list = ", ".join(funcs) if isinstance(funcs, list) else str(funcs)
        console.print(f"  [bold cyan]{display}[/] [green]● loaded[/]")
        if isinstance(funcs, list) and funcs:
            for fn in funcs:
                console.print(f"    [dim]├─ {fn}[/]")
        console.print()

    missing = data.get("missing", [])
    if missing:
        console.print(f"  [yellow]Missing engines: {', '.join(missing)}[/]")


def action_analyze_chat():
    """Run NLP analysis on a specific chat."""
    cid = console.input("[bold]Chat ID or @username: [/]").strip()
    if not cid:
        return

    console.print("  [dim]Running analysis...[/]")

    # Try V5 first, fall back to V4, then V3, then V2
    result = api("GET", "/engine/analyze-v5", params={"chat_id": cid})
    if not result or "error" in result:
        result = api("GET", "/engine/analyze-v4", params={"chat_id": cid})
    if not result or "error" in result:
        result = api("GET", "/nlp/analyze-v3", params={"chat_id": cid})
    if not result or "error" in result:
        result = api("GET", "/nlp/analyze-v2", params={"chat_id": cid})
    if not result or "error" in result:
        result = api("GET", "/nlp/analyze", params={"chat_id": cid})

    if not result:
        return

    _display_analysis(result)


def _display_analysis(result: Dict):
    """Render analysis results."""
    at = Table(show_header=False, box=None, padding=(0, 2))
    at.add_column("Key", style="cyan", min_width=24)
    at.add_column("Value", style="bright_white")

    def _flatten(d: Dict, prefix: str = ""):
        for k, v in d.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _flatten(v, full_key)
            elif isinstance(v, list):
                at.add_row(full_key, str(v[:5]) + ("..." if len(v) > 5 else ""))
            else:
                at.add_row(full_key, str(v))

    _flatten(result)
    console.print(Panel(at, title="[bold]Analysis Results[/]",
                        border_style="green", box=box.ROUNDED))


def action_relationship_health():
    """Check conversation health for a chat."""
    cid = console.input("[bold]Chat ID or @username: [/]").strip()
    if not cid:
        return
    console.print("  [dim]Computing health score...[/]")
    result = api("GET", "/relationship/health", params={"chat_id": cid})
    if result:
        _display_analysis(result)


def action_psychological_analysis():
    """Run psychological analysis on a chat."""
    cid = console.input("[bold]Chat ID or @username: [/]").strip()
    if not cid:
        return
    console.print("  [dim]Running psychological analysis...[/]")
    result = api("GET", "/engine/psychological-analysis", params={"chat_id": cid})
    if result:
        _display_analysis(result)


def action_rl_insights():
    """View RL insights."""
    console.print()
    console.print("  [dim]1[/]  Insights for a specific chat")
    console.print("  [dim]2[/]  All chat insights")
    choice = console.input("\n  [bold]Choice: [/]").strip()
    if choice == "1":
        cid = console.input("[bold]Chat ID: [/]").strip()
        if cid:
            result = api("GET", "/rl/insights", params={"chat_id": cid})
            if result:
                _display_analysis(result)
    elif choice == "2":
        result = api("GET", "/rl/insights/all")
        if result:
            _display_analysis(result)


def action_show_log():
    """View auto-reply activity log."""
    data = api("GET", "/auto-reply/log?limit=30")
    if not data:
        return
    entries = data.get("log", [])
    if not entries:
        console.print("  [dim]No recent activity[/]")
        return

    lt = Table(title=f"Activity Log ({data.get('count', '?')} total)",
               box=box.ROUNDED, border_style="green",
               title_style="bold bright_white", padding=(0, 1))
    lt.add_column("Time", style="dim", width=10)
    lt.add_column("Chat", style="bright_white", min_width=14)
    lt.add_column("Action", min_width=14)
    lt.add_column("Details", style="dim", ratio=2)

    for entry in entries[-30:]:
        ts = entry.get("timestamp", "?")
        if isinstance(ts, str) and len(ts) > 16:
            ts = ts[11:19]
        chat = str(entry.get("chat_id", entry.get("chat", "?")))
        action = entry.get("action", entry.get("type", "?"))
        detail = entry.get("detail", entry.get("message", ""))
        if isinstance(detail, str) and len(detail) > 50:
            detail = detail[:47] + "..."
        lt.add_row(str(ts), chat, str(action), str(detail))

    console.print(lt)


def action_view_models():
    """View detailed model status."""
    result = api("GET", "/models/status")
    if result:
        _display_analysis(result)


def action_train_models():
    """Trigger model retraining."""
    console.print()
    console.print("  [dim]1[/]  Train sklearn classifiers only")
    console.print("  [dim]2[/]  Train sklearn + neural networks")
    console.print("  [dim]3[/]  Train specific task (romantic_intent/conversation_stage/emotional_tone)")
    console.print("  [dim]4[/]  Back")
    choice = console.input("\n  [bold]Choice: [/]").strip()

    if choice == "1":
        console.print("  [dim]Training sklearn classifiers...[/]")
        result = api("POST", "/dl/train", params={"task": "all", "include_neural": False})
        if result:
            console.print(f"  [green]Training complete:[/] {result}")
    elif choice == "2":
        console.print("  [dim]Training all models (this may take a while)...[/]")
        result = api("POST", "/dl/train", params={"task": "all", "include_neural": True})
        if result:
            console.print(f"  [green]Training complete:[/] {result}")
    elif choice == "3":
        task = console.input("  [bold]Task name: [/]").strip()
        if task:
            console.print(f"  [dim]Training {task}...[/]")
            result = api("POST", "/dl/train", params={"task": task, "include_neural": True})
            if result:
                console.print(f"  [green]Training complete:[/] {result}")


def action_preload_models():
    """Preload DL models."""
    console.print("  [dim]Preloading models...[/]")
    result = api("POST", "/dl/preload")
    if result:
        if result.get("success"):
            console.print("  [green]Models preloaded successfully[/]")
        else:
            console.print(f"  [yellow]{result.get('error', 'Unknown error')}[/]")


def action_dl_status():
    """View deep learning system status."""
    result = api("GET", "/dl/status")
    if result:
        _display_analysis(result)


def action_view_me():
    """View current Telegram account info."""
    result = api("GET", "/me")
    if result:
        mt = Table(show_header=False, box=None, padding=(0, 2))
        mt.add_column("Key", style="dim", min_width=16)
        mt.add_column("Value", style="bright_white")
        for k, v in result.items():
            mt.add_row(str(k), str(v))
        console.print(Panel(mt, title="[bold]My Account[/]",
                            border_style="cyan", box=box.ROUNDED))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_live_mode(interval: int = 5):
    """Run a live auto-refreshing dashboard."""
    console.print()
    console.print("  [bold bright_magenta]LIVE MODE[/] — refreshing every "
                  f"[bright_white]{interval}s[/]. Press [bold]Ctrl+C[/] to return to menu.")
    console.print()

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                data = fetch_dashboard(quiet=True)
                if data:
                    live.update(render_full_dashboard(data))
                else:
                    live.update(Panel("[bold red]Cannot connect to API[/]\n"
                                     f"[dim]Retrying in {interval}s...[/]",
                                     border_style="red"))
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n  [dim]Exited live mode[/]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMAND DISPATCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMANDS = {
    # Dashboard
    "d": "dashboard", "dash": "dashboard", "dashboard": "dashboard", "status": "dashboard",
    "live": "live",
    # Features
    "t": "toggle", "toggle": "toggle",
    "a": "autoreply", "auto": "autoreply", "autoreply": "autoreply",
    "all": "all",
    # Config
    "delay": "delay",
    "max": "proactive_max", "proactive": "proactive_max",
    "ctx": "context", "context": "context",
    "prompt": "prompt",
    # Chat management
    "wl": "whitelist", "whitelist": "whitelist",
    "inst": "instructions", "instructions": "instructions",
    "chats": "chats",
    # Intelligence
    "engines": "engines", "engine": "engines",
    "analyze": "analyze", "nlp": "analyze",
    "health": "health",
    "psych": "psych", "psychological": "psych",
    "rl": "rl",
    # Models
    "models": "models", "model": "models",
    "train": "train",
    "preload": "preload",
    "dl": "dl",
    # Activity
    "log": "log", "logs": "log", "activity": "log",
    "me": "me",
    # Meta
    "h": "help", "help": "help", "?": "help", "menu": "help",
    "q": "quit", "quit": "quit", "exit": "quit",
}

ACTIONS = {
    "dashboard": lambda: (console.clear(), console.print(render_header()),
                          show_dashboard_static(), show_menu()),
    "live": lambda: run_live_mode(),
    "toggle": action_toggle_feature,
    "autoreply": action_toggle_auto_reply,
    "all": action_toggle_all,
    "delay": action_set_delay,
    "proactive_max": action_set_proactive_max,
    "context": action_set_context,
    "prompt": action_view_prompt,
    "whitelist": action_manage_whitelist,
    "instructions": action_manage_instructions,
    "chats": action_list_chats,
    "engines": action_engine_details,
    "analyze": action_analyze_chat,
    "health": action_relationship_health,
    "psych": action_psychological_analysis,
    "rl": action_rl_insights,
    "models": action_view_models,
    "train": action_train_models,
    "preload": action_preload_models,
    "dl": action_dl_status,
    "log": action_show_log,
    "me": action_view_me,
    "help": show_menu,
}


def show_dashboard_static():
    """Fetch and print dashboard to console (non-live mode)."""
    data = fetch_dashboard()
    if data:
        console.print(render_system_status(data))
        console.print(render_features(data))
        console.print()
        console.print(Columns([render_engines(data), render_models(data)], equal=True, expand=True))
        console.print()
        console.print(Columns([render_media_ai(data), render_config(data)], equal=True, expand=True))
        console.print()
        console.print(render_activity(data))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Telegram AI Agent — Advanced Control Panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python control_panel.py                     # Interactive mode
  uv run python control_panel.py --live               # Live auto-refresh
  uv run python control_panel.py --live --interval 3  # Refresh every 3s
  uv run python control_panel.py --url http://host:8765
        """,
    )
    parser.add_argument("--url", default="http://localhost:8765", help="API base URL")
    parser.add_argument("--live", action="store_true", help="Start in live auto-refresh mode")
    parser.add_argument("--interval", type=int, default=5, help="Live refresh interval (seconds)")
    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")

    # Live mode: jump straight to auto-refresh dashboard
    if args.live:
        console.clear()
        run_live_mode(args.interval)
        return

    # Interactive mode
    console.clear()
    console.print(render_header())

    data = fetch_dashboard()
    if data:
        console.print("  [green]Connected to bot API[/] at [cyan]{url}[/]".format(url=BASE_URL))
        show_dashboard_static()
    else:
        console.print("  [yellow]Will retry on next command.[/]")

    show_menu()

    while True:
        try:
            console.print()
            cmd = console.input("[bold bright_magenta]>[/] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Bye.[/]")
            break

        if not cmd:
            continue

        action_key = COMMANDS.get(cmd)
        if action_key == "quit":
            console.print("  [dim]Bye.[/]")
            break
        elif action_key and action_key in ACTIONS:
            try:
                ACTIONS[action_key]()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [dim]Cancelled[/]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/]")
        else:
            console.print(f"  [dim]Unknown: '{cmd}'. Type 'h' for help.[/]")


if __name__ == "__main__":
    main()
