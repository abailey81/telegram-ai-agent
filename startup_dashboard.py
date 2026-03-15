"""
Rich startup dashboard for the Telegram AI bot backend.
Provides visually stunning terminal output for system boot, monitoring, and live logging.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.rule import Rule
from rich import box

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BANNER_ART = """
 ████████╗ ███████╗ ██╗     ███████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗
 ╚══██╔══╝ ██╔════╝ ██║     ██╔════╝ ██╔════╝ ██╔══██╗██╔══██╗████╗ ████║
    ██║    █████╗   ██║     █████╗   ██║  ███╗██████╔╝███████║██╔████╔██║
    ██║    ██╔══╝   ██║     ██╔══╝   ██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║
    ██║    ███████╗ ███████╗███████╗ ╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
    ╚═╝    ╚══════╝ ╚══════╝╚══════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
"""


def print_banner():
    """Print the startup banner with gradient colors."""
    banner_text = Text()

    for i, line in enumerate(BANNER_ART.strip().split("\n")):
        if i < 2:
            banner_text.append(line + "\n", style="bold magenta")
        elif i < 4:
            banner_text.append(line + "\n", style="bold cyan")
        else:
            banner_text.append(line + "\n", style="bold blue")

    subtitle = Text()
    subtitle.append("  Telegram AI Agent", style="italic bright_white")
    subtitle.append("  ·  ", style="dim")
    subtitle.append("v2.0", style="dim cyan")

    full_text = Text()
    full_text.append_text(banner_text)
    full_text.append("\n")
    full_text.append_text(subtitle)

    panel = Panel(
        full_text,
        border_style="bright_magenta",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    )
    console.print(panel)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOOT SEQUENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_boot_sequence(
    v4_engines: Dict[str, Any],
    media_ai_available: bool,
    advanced_intel_available: bool,
):
    """Print animated boot sequence showing module loading status."""
    modules = [
        ("NLP Engine", True),  # Always available (core module)
        ("Conversation Engine", "conversation" in v4_engines),
        ("Emotional Intelligence", "emotional" in v4_engines or "emotional_v5" in v4_engines),
        ("Style Engine", "style" in v4_engines),
        ("Memory Engine", "memory" in v4_engines),
        ("Reasoning Engine", "reasoning" in v4_engines),
        ("Psychological Datasets", "psychological" in v4_engines),
        ("Media Intelligence", "media" in v4_engines),
        ("RL Engine", "rl" in v4_engines),
        ("Media AI (Voice/Image/NLP)", media_ai_available),
        ("Advanced Intelligence", advanced_intel_available),
        ("Telegram Connection", True),
    ]

    console.print()
    console.print("  [bold bright_white]SYSTEM BOOT[/]", highlight=False)
    console.print()

    with Progress(
        SpinnerColumn("dots", style="cyan"),
        TextColumn("[bold]{task.description}[/]", justify="left"),
        BarColumn(bar_width=30, complete_style="green", finished_style="green"),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for name, available in modules:
            task = progress.add_task(
                f"  {name}",
                total=100,
                status="[dim]loading...[/]",
            )
            # Small delay for visual effect
            for step in range(0, 101, 20):
                progress.update(task, completed=step)
                time.sleep(0.02)

            if available:
                progress.update(task, completed=100, status="[green]✓ loaded[/]")
            else:
                progress.update(task, completed=100, status="[yellow]○ skipped[/]")

    # Print summary table after progress
    loaded = sum(1 for _, avail in modules if avail)
    total = len(modules)

    table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Module", style="bright_white", min_width=30)
    table.add_column("Status", min_width=12)

    for name, available in modules:
        if available:
            table.add_row(f"  {name}", "[green]● loaded[/]")
        else:
            table.add_row(f"  {name}", "[dim]○ skipped[/]")

    panel = Panel(
        table,
        title=f"[bold]Modules {loaded}/{total}[/]",
        border_style="green" if loaded >= 10 else "yellow",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL STATUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_model_status():
    """Print trained model status from meta files."""
    table = Table(
        title="Trained Models",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold bright_white",
        padding=(0, 1),
    )
    table.add_column("Model", style="bright_white", min_width=22)
    table.add_column("Type", style="dim", min_width=12)
    table.add_column("Accuracy", min_width=10, justify="right")
    table.add_column("Classes", style="dim", justify="center", min_width=8)
    table.add_column("Examples", style="dim", justify="right", min_width=9)

    sklearn_models = ["romantic_intent", "conversation_stage", "emotional_tone"]

    for name in sklearn_models:
        meta_path = f"trained_models/{name}_meta.json"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            acc = meta.get("cv_accuracy", 0)
            acc_pct = f"{acc * 100:.1f}%" if acc < 1 else f"{acc:.1f}%"
            if acc > 0.93 or acc > 93:
                acc_style = f"[bold green]{acc_pct}[/]"
            elif acc > 0.85 or acc > 85:
                acc_style = f"[yellow]{acc_pct}[/]"
            else:
                acc_style = f"[red]{acc_pct}[/]"

            display_name = name.replace("_", " ").title()
            classifier = meta.get("best_classifier", "?")
            n_classes = str(meta.get("n_classes", "?"))
            n_examples = str(meta.get("n_training_examples", "?"))
            table.add_row(display_name, classifier, acc_style, n_classes, n_examples)
        except Exception:
            display_name = name.replace("_", " ").title()
            table.add_row(display_name, "[dim]—[/]", "[dim]not trained[/]", "—", "—")

    # Neural network models
    table.add_section()
    for name in sklearn_models:
        neural_meta_path = f"trained_models/neural/{name}_meta.json"
        try:
            with open(neural_meta_path) as f:
                nmeta = json.load(f)
            for arch_name, arch_info in nmeta.get("models", {}).items():
                acc = arch_info.get("val_accuracy", 0)
                acc_pct = f"{acc * 100:.1f}%" if acc < 1 else f"{acc:.1f}%"
                if acc > 0.93 or acc > 93:
                    acc_style = f"[green]{acc_pct}[/]"
                elif acc > 0.85 or acc > 85:
                    acc_style = f"[yellow]{acc_pct}[/]"
                else:
                    acc_style = f"[dim]{acc_pct}[/]"

                display_name = name.replace("_", " ").title()
                n_classes = str(nmeta.get("num_classes", "?"))
                table.add_row(
                    f"  {display_name}",
                    arch_name,
                    acc_style,
                    n_classes,
                    "—",
                )
        except Exception:
            pass

    console.print(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO-REPLY CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_auto_reply_config(config):
    """Print auto-reply configuration with feature flags."""
    # Config summary
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="dim", min_width=16)
    summary.add_column("Value", style="bright_white")

    enabled = config.enabled
    status_text = "[bold green]ENABLED[/]" if enabled else "[bold red]DISABLED[/]"
    summary.add_row("Status", status_text)
    summary.add_row("Active Chats", f"[bright_white]{len(config.chat_ids)}[/] [dim]({', '.join(str(c) for c in config.chat_ids[:3])}{'...' if len(config.chat_ids) > 3 else ''})[/]")
    summary.add_row("Delay Range", f"[bright_white]{config.delay_min}[/]–[bright_white]{config.delay_max}[/] [dim]seconds[/]")
    summary.add_row("Context Depth", f"[bright_white]{config.context_messages}[/] [dim]messages[/]")

    # Feature flags grid
    features = [
        ("Late Night", getattr(config, "late_night_mode", False)),
        ("Strategic Silence", getattr(config, "strategic_silence", False)),
        ("Quote Reply", getattr(config, "quote_reply", True)),
        ("Smart Reactions", getattr(config, "smart_reactions", True)),
        ("Message Editing", getattr(config, "message_editing", True)),
        ("GIF/Sticker", getattr(config, "gif_sticker_reply", True)),
        ("Typing Awareness", getattr(config, "typing_awareness", False)),
        ("Online Status", getattr(config, "online_status_aware", True)),
        ("Proactive Msgs", getattr(config, "proactive_enabled", False)),
    ]

    flags_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    flags_table.add_column("Feature 1", min_width=28)
    flags_table.add_column("Feature 2", min_width=28)

    for i in range(0, len(features), 2):
        name1, on1 = features[i]
        badge1 = f"[bold green] ON [/] {name1}" if on1 else f"[dim red]OFF [/] [dim]{name1}[/]"
        if i + 1 < len(features):
            name2, on2 = features[i + 1]
            badge2 = f"[bold green] ON [/] {name2}" if on2 else f"[dim red]OFF [/] [dim]{name2}[/]"
        else:
            badge2 = ""
        flags_table.add_row(badge1, badge2)

    # Combine in panel
    combined = Text()
    panel_content = Table.grid(padding=(1, 0))
    panel_content.add_row(summary)
    panel_content.add_row(Rule(style="dim"))
    panel_content.add_row(Text("  Feature Flags", style="bold bright_white"))
    panel_content.add_row(flags_table)

    panel = Panel(
        panel_content,
        title="[bold]Auto-Reply Configuration[/]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MEDIA AI STATUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_media_ai_status(get_status_fn=None):
    """Print media AI capabilities status."""
    if get_status_fn is None:
        return

    try:
        status = get_status_fn()
    except Exception:
        console.print("  [dim]Media AI: status unavailable[/]")
        return

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("Capability", style="bright_white", min_width=24)
    table.add_column("Status", min_width=10)
    table.add_column("Backend", style="dim", min_width=20)

    capability_names = {
        "voice_transcription": "Voice Transcription",
        "image_understanding": "Image Understanding",
        "voice_response": "Voice Response",
        "russian_nlp": "Russian NLP",
        "multilingual_embeddings": "Multilingual Embeddings",
        "vector_memory": "Vector Memory (FAISS)",
    }

    for key, info in status.items():
        name = capability_names.get(key, key)
        available = info.get("available", False)
        backend = info.get("backend", "—")
        if available:
            table.add_row(name, "[green]● ready[/]", backend)
        else:
            table.add_row(name, "[dim]○ none[/]", "[dim]—[/]")

    panel = Panel(
        table,
        title="[bold]Media AI Capabilities[/]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIVE LOG HANDLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RichAutoReplyHandler(logging.Handler):
    """Custom logging handler that formats auto-reply logs with Rich."""

    CATEGORY_STYLES = {
        "reply": ("[bold green]REPLY[/]", "green"),
        "react": ("[bold yellow]REACT[/]", "yellow"),
        "skip": ("[dim]SKIP [/]", "dim"),
        "error": ("[bold red]ERROR[/]", "red"),
        "voice": ("[bold blue]VOICE[/]", "blue"),
        "gif": ("[bold magenta]GIF  [/]", "magenta"),
        "edit": ("[bold cyan]EDIT [/]", "cyan"),
        "proactive": ("[bold bright_magenta]PROAC[/]", "bright_magenta"),
        "sticker": ("[bold yellow]STKR [/]", "yellow"),
        "status": ("[bold cyan]STATUS[/]", "cyan"),
    }

    def __init__(self):
        super().__init__()
        self.console = Console(stderr=False)

    def _detect_category(self, message: str) -> str:
        msg_lower = message.lower()
        if "error" in msg_lower or "failed" in msg_lower:
            return "error"
        if "react" in msg_lower or "reaction" in msg_lower:
            return "react"
        if "skip" in msg_lower or "silence" in msg_lower or "not replying" in msg_lower:
            return "skip"
        if "voice" in msg_lower:
            return "voice"
        if "gif" in msg_lower:
            return "gif"
        if "edit" in msg_lower:
            return "edit"
        if "proactive" in msg_lower:
            return "proactive"
        if "sticker" in msg_lower:
            return "sticker"
        if "status" in msg_lower or "online" in msg_lower or "offline" in msg_lower:
            return "status"
        if "sent" in msg_lower or "part" in msg_lower or "reply" in msg_lower or "stream" in msg_lower:
            return "reply"
        return "reply"

    def emit(self, record):
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            category = self._detect_category(msg)
            badge, _ = self.CATEGORY_STYLES.get(category, ("[dim]LOG  [/]", "dim"))

            ts = datetime.now().strftime("%H:%M:%S")
            self.console.print(
                f"  [dim]{ts}[/] {badge} {msg}",
                highlight=False,
            )
        except Exception:
            self.handleError(record)


def create_live_log_handler() -> logging.Handler:
    """Create a Rich-powered log handler for auto-reply activity."""
    handler = RichAutoReplyHandler()
    return handler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# READY MESSAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_ready():
    """Print the final ready message."""
    console.print()
    console.print(Rule(style="bright_green"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"  [bold bright_green]System ready.[/] "
        f"[dim]Listening for messages... ({now})[/]",
        highlight=False,
    )
    console.print(
        f"  [dim]Press Ctrl+C to stop[/]",
        highlight=False,
    )
    console.print(
        f"  [dim]Run [bold cyan]uv run python control_panel.py[/bold cyan] in another terminal for the interactive control panel[/]",
        highlight=False,
    )
    console.print(Rule(style="bright_green"))
    console.print()
