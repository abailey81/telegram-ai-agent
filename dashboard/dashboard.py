"""
Telegram AI Agent — Messenger-Style Dashboard
Built with Reflex | Telegram-Inspired Dark Theme | Real-Time Control

Layout: Left sidebar (chat list + nav) | Right panel (messages or control tabs)
"""

import reflex as rx
from .state import DashboardState, FEATURE_KEYS, FEATURE_META

# ═══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Telegram-inspired dark theme
# ═══════════════════════════════════════════════════════════════════════

BG = "#0e1621"
BG_SIDEBAR = "#17212b"
BG_CHAT = "#0e1621"
BG_MSG_OUT = "#2b5278"
BG_MSG_IN = "#182533"
BG_INPUT = "#242f3d"
BG_HOVER = "#202b36"
BG_HEADER = "#17212b"
BORDER_COLOR = "rgba(255,255,255,0.08)"

TEXT = "#f5f5f5"
TEXT2 = "#8b9bab"
TEXT_MSG = "#f5f5f5"
TEXT_TIME = "#6e8192"
TEXT_LINK = "#6ab2f2"

BLUE = "#3390ec"
GREEN = "#4fae4e"
RED = "#e53935"
AMBER = "#f5a623"
PURPLE = "#8b5cf6"
CYAN = "#53bdeb"

_EASE = "all 0.2s ease"


def _anim(name: str, dur: str = "0.35s", delay: str = "0s") -> str:
    return f"{name} {dur} cubic-bezier(0.25, 0.1, 0.25, 1.0) {delay} both"


# ═══════════════════════════════════════════════════════════════════════
# REUSABLE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

def icon_text(icon_name: str, label: str, color: str = TEXT2) -> rx.Component:
    return rx.hstack(
        rx.icon(tag=icon_name, size=14, color=color),
        rx.text(label, color=color, font_size="13px"),
        spacing="2",
        align="center",
    )


def sidebar_nav_item(icon: str, label: str, tab_key: str) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon(tag=icon, size=18, color=rx.cond(
                DashboardState.active_tab == tab_key, BLUE, TEXT2
            )),
            rx.text(label, color=rx.cond(
                DashboardState.active_tab == tab_key, TEXT, TEXT2
            ), font_size="13px", font_weight=rx.cond(
                DashboardState.active_tab == tab_key, "600", "400"
            )),
            spacing="3",
            align="center",
            padding_x="12px",
            padding_y="8px",
        ),
        background=rx.cond(
            DashboardState.active_tab == tab_key,
            "rgba(51,144,236,0.12)",
            "transparent",
        ),
        border_radius="8px",
        cursor="pointer",
        transition=_EASE,
        _hover={"background": "rgba(51,144,236,0.08)"},
        on_click=DashboardState.set_tab(tab_key),
        width="100%",
    )


def stat_pill(label: str, value, color: str = BLUE) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(value, color=color, font_size="20px", font_weight="700"),
            rx.text(label, color=TEXT2, font_size="11px"),
            spacing="1",
            align="center",
        ),
        padding="12px 16px",
        background=BG_SIDEBAR,
        border=f"1px solid {BORDER_COLOR}",
        border_radius="12px",
        min_width="90px",
    )


def section_card(*children, **props) -> rx.Component:
    return rx.box(
        *children,
        background=BG_SIDEBAR,
        border=f"1px solid {BORDER_COLOR}",
        border_radius="12px",
        padding="16px",
        width="100%",
        **props,
    )


def section_title(title: str, icon: str = "") -> rx.Component:
    items = []
    if icon:
        items.append(rx.icon(tag=icon, size=16, color=BLUE))
    items.append(rx.text(title, color=TEXT, font_size="14px", font_weight="600"))
    return rx.hstack(*items, spacing="2", margin_bottom="12px")


def styled_input(placeholder: str, value, on_change, **kwargs) -> rx.Component:
    return rx.el.input(
        placeholder=placeholder,
        value=value,
        on_change=on_change,
        style={
            "background": BG_INPUT,
            "border": f"1px solid {BORDER_COLOR}",
            "color": TEXT,
            "font_size": "13px",
            "padding": "8px 12px",
            "border_radius": "8px",
            "width": "100%",
            "outline": "none",
            "transition": _EASE,
            "::placeholder": {"color": TEXT2},
            ":focus": {"border_color": BLUE},
        },
        **kwargs,
    )


def action_btn(label: str, on_click, color: str = BLUE, icon: str = "") -> rx.Component:
    children = []
    if icon:
        children.append(rx.icon(tag=icon, size=14))
    children.append(rx.text(label, font_size="12px", font_weight="600"))
    return rx.box(
        rx.hstack(*children, spacing="2", align="center", justify="center"),
        background=color,
        color="white",
        padding="6px 14px",
        border_radius="8px",
        cursor="pointer",
        transition=_EASE,
        _hover={"opacity": "0.85", "transform": "scale(0.98)"},
        on_click=on_click,
    )


# ═══════════════════════════════════════════════════════════════════════
# LEFT SIDEBAR — Chat list + Navigation
# ═══════════════════════════════════════════════════════════════════════

def _avatar(photo_url, initial, size: str = "42px", font: str = "16px") -> rx.Component:
    """Profile photo with fallback to initial circle. Image overlays the initial."""
    return rx.box(
        # Fallback: initial letter (always rendered behind)
        rx.text(initial, color="white", font_size=font, font_weight="700",
                position="absolute", z_index="1"),
        # Real photo overlaid — if it loads, covers the initial
        rx.image(
            src=photo_url,
            width="100%", height="100%",
            border_radius="50%",
            object_fit="cover",
            position="absolute",
            top="0", left="0",
            z_index="2",
            loading="lazy",
        ),
        width=size,
        height=size,
        border_radius="50%",
        background=BLUE,
        display="flex",
        align_items="center",
        justify_content="center",
        flex_shrink="0",
        overflow="hidden",
        position="relative",
    )


def _render_chat_item(chat: dict) -> rx.Component:
    """Single chat in the sidebar list."""
    return rx.box(
        rx.hstack(
            # Avatar with real photo
            _avatar(chat["photo_url"], chat["initial"]),
            # Name + last message
            rx.vstack(
                rx.hstack(
                    rx.text(
                        chat["name"],
                        color=TEXT,
                        font_size="14px",
                        font_weight="500",
                        overflow="hidden",
                        text_overflow="ellipsis",
                        white_space="nowrap",
                        max_width="140px",
                    ),
                    rx.spacer(),
                    rx.cond(
                        chat["unread"] != "0",
                        rx.box(
                            rx.text(chat["unread"], color="white",
                                    font_size="10px", font_weight="700"),
                            background=BLUE,
                            padding="1px 6px",
                            border_radius="10px",
                            min_width="20px",
                            text_align="center",
                        ),
                        rx.fragment(),
                    ),
                    width="100%",
                    align="center",
                ),
                rx.text(
                    chat["last_msg"],
                    color=TEXT2,
                    font_size="12px",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                    max_width="180px",
                ),
                spacing="0",
                align_items="flex-start",
                flex="1",
            ),
            spacing="3",
            align="center",
            width="100%",
        ),
        padding="8px 12px",
        cursor="pointer",
        transition=_EASE,
        border_radius="8px",
        background=rx.cond(
            DashboardState.active_chat_id == chat["chat_id"],
            "rgba(51,144,236,0.15)",
            "transparent",
        ),
        _hover={"background": BG_HOVER},
        on_click=DashboardState.open_chat(chat["chat_id"]),
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header
            rx.hstack(
                rx.icon(tag="menu", size=20, color=TEXT2, cursor="pointer"),
                rx.text("Telegram AI", color=TEXT,
                         font_size="16px", font_weight="700"),
                rx.spacer(),
                # Connection indicator
                rx.box(
                    width="8px", height="8px", border_radius="50%",
                    background=rx.cond(DashboardState.connected, GREEN, RED),
                ),
                spacing="3",
                align="center",
                padding="12px 16px",
                width="100%",
            ),

            # Search
            rx.box(
                styled_input("Search", DashboardState.wl_input,
                             DashboardState.set_wl_input),
                padding_x="12px",
                width="100%",
            ),

            # Chat list
            rx.box(
                rx.cond(
                    DashboardState.chats_loading,
                    rx.center(
                        rx.spinner(size="3"),
                        padding="20px",
                    ),
                    rx.vstack(
                        rx.foreach(DashboardState.chats_list, _render_chat_item),
                        spacing="1",
                        width="100%",
                    ),
                ),
                flex="1",
                overflow_y="auto",
                padding="4px 8px",
                width="100%",
            ),

            # Nav section divider
            rx.box(
                height="1px",
                background=BORDER_COLOR,
                width="100%",
                margin_y="4px",
            ),

            # Navigation tabs
            rx.vstack(
                sidebar_nav_item("message-circle", "Messenger", "messenger"),
                sidebar_nav_item("layout-dashboard", "Overview", "overview"),
                sidebar_nav_item("toggle-right", "Features", "features"),
                sidebar_nav_item("settings", "Config", "config"),
                sidebar_nav_item("brain", "Intelligence", "intelligence"),
                sidebar_nav_item("cpu", "Engines", "engines"),
                sidebar_nav_item("database", "Models", "models"),
                sidebar_nav_item("scroll-text", "Log", "log"),
                sidebar_nav_item("user", "Account", "account"),
                spacing="1",
                padding="4px 8px",
                width="100%",
            ),

            spacing="2",
            height="100%",
        ),
        width="280px",
        min_width="280px",
        height="100vh",
        background=BG_SIDEBAR,
        border_right=f"1px solid {BORDER_COLOR}",
        overflow_y="auto",
    )


# ═══════════════════════════════════════════════════════════════════════
# MESSENGER VIEW — Chat messages like Telegram
# ═══════════════════════════════════════════════════════════════════════

def _render_message(msg: dict) -> rx.Component:
    """Single message bubble — out (right, blue) vs in (left, dark)."""
    is_out = msg["out"] == "true"
    return rx.box(
        rx.vstack(
            # Reply indicator
            rx.cond(
                msg["reply_to"] != "",
                rx.box(
                    rx.text("Reply", color=CYAN, font_size="11px"),
                    border_left=f"2px solid {CYAN}",
                    padding_left="8px",
                    margin_bottom="4px",
                ),
                rx.fragment(),
            ),
            # Sender name (for incoming)
            rx.cond(
                is_out,
                rx.fragment(),
                rx.text(
                    msg["sender"],
                    color=CYAN,
                    font_size="12px",
                    font_weight="600",
                    margin_bottom="2px",
                ),
            ),
            # Media indicator
            rx.cond(
                msg["has_media"] == "true",
                rx.hstack(
                    rx.icon(tag="paperclip", size=12, color=TEXT2),
                    rx.text(msg["media_type"], color=TEXT2,
                            font_size="11px", font_style="italic"),
                    spacing="1",
                    margin_bottom="4px",
                ),
                rx.fragment(),
            ),
            # Message text
            rx.cond(
                msg["text"] != "",
                rx.text(
                    msg["text"],
                    color=TEXT_MSG,
                    font_size="14px",
                    line_height="1.45",
                    white_space="pre-wrap",
                    word_break="break-word",
                ),
                rx.fragment(),
            ),
            # Timestamp
            rx.text(
                msg["date"],
                color=TEXT_TIME,
                font_size="11px",
                text_align="right",
                width="100%",
            ),
            spacing="0",
        ),
        background=rx.cond(is_out, BG_MSG_OUT, BG_MSG_IN),
        padding="8px 12px",
        border_radius=rx.cond(
            is_out,
            "12px 12px 4px 12px",
            "12px 12px 12px 4px",
        ),
        max_width="65%",
        min_width="120px",
        margin_left=rx.cond(is_out, "auto", "0"),
        margin_right=rx.cond(is_out, "0", "auto"),
        margin_y="2px",
    )


def messenger_header() -> rx.Component:
    """Top bar showing chat name, status, and actions."""
    return rx.hstack(
        # Back button for narrow screens
        rx.box(
            rx.icon(tag="arrow-left", size=20, color=TEXT2),
            cursor="pointer",
            on_click=DashboardState.set_tab("messenger"),
            display=rx.cond(DashboardState.active_chat_id != "", "block", "none"),
            padding="4px",
        ),
        # Avatar with real photo
        _avatar(DashboardState.active_chat_photo,
                DashboardState.active_chat_initial,
                size="36px", font="14px"),
        # Name + status
        rx.vstack(
            rx.text(DashboardState.active_chat_name, color=TEXT,
                     font_size="15px", font_weight="600"),
            rx.text(
                rx.cond(
                    DashboardState.active_chat_status == "online",
                    "online",
                    rx.cond(
                        DashboardState.active_chat_status == "recently",
                        "last seen recently",
                        DashboardState.active_chat_status,
                    ),
                ),
                color=rx.cond(
                    DashboardState.active_chat_status == "online",
                    CYAN,
                    TEXT2,
                ),
                font_size="12px",
            ),
            spacing="0",
        ),
        rx.spacer(),
        # Action buttons
        rx.hstack(
            rx.icon(tag="search", size=18, color=TEXT2, cursor="pointer"),
            rx.icon(tag="bar-chart-2", size=18, color=TEXT2, cursor="pointer",
                    on_click=DashboardState.load_chat_analytics),
            rx.icon(tag="brain", size=18, color=TEXT2, cursor="pointer",
                    on_click=DashboardState.load_behavioral_patterns),
            rx.icon(tag="check-check", size=18, color=TEXT2, cursor="pointer",
                    on_click=DashboardState.mark_read),
            spacing="4",
        ),
        spacing="3",
        align="center",
        padding="10px 16px",
        background=BG_HEADER,
        border_bottom=f"1px solid {BORDER_COLOR}",
        width="100%",
        min_height="56px",
    )


def messenger_input_bar() -> rx.Component:
    """Message input bar at the bottom."""
    return rx.hstack(
        rx.el.input(
            placeholder="Write a message...",
            value=DashboardState.msg_input,
            on_change=DashboardState.set_msg_input,
            style={
                "background": BG_INPUT,
                "border": "none",
                "color": TEXT,
                "font_size": "14px",
                "padding": "10px 14px",
                "border_radius": "20px",
                "width": "100%",
                "outline": "none",
                "::placeholder": {"color": TEXT2},
            },
        ),
        rx.box(
            rx.icon(tag="send", size=20, color="white"),
            background=BLUE,
            padding="8px",
            border_radius="50%",
            cursor="pointer",
            _hover={"opacity": "0.85"},
            on_click=DashboardState.send_message,
            display="flex",
            align_items="center",
            justify_content="center",
        ),
        spacing="3",
        align="center",
        padding="8px 16px",
        background=BG_HEADER,
        border_top=f"1px solid {BORDER_COLOR}",
        width="100%",
    )


def messenger_info_panel() -> rx.Component:
    """Slide-in panel showing analytics, behavioral patterns, mirroring."""
    return rx.vstack(
        # Behavior Mirroring
        section_card(
            section_title("Behavior Mirroring", "copy"),
            rx.text(
                "Adapt response style based on their communication patterns",
                color=TEXT2, font_size="12px", margin_bottom="8px",
            ),
            rx.hstack(
                action_btn("Off", DashboardState.set_mirror_mode("off"),
                           color=rx.cond(DashboardState.mirror_mode == "off",
                                         BLUE, BG_INPUT)),
                action_btn("Subtle", DashboardState.set_mirror_mode("subtle"),
                           color=rx.cond(DashboardState.mirror_mode == "subtle",
                                         GREEN, BG_INPUT)),
                action_btn("Match", DashboardState.set_mirror_mode("match"),
                           color=rx.cond(DashboardState.mirror_mode == "match",
                                         AMBER, BG_INPUT)),
                action_btn("Assertive", DashboardState.set_mirror_mode("assertive"),
                           color=rx.cond(DashboardState.mirror_mode == "assertive",
                                         RED, BG_INPUT)),
                spacing="2",
                flex_wrap="wrap",
            ),
        ),

        # Detected patterns
        rx.cond(
            DashboardState.mirror_data.length() > 0,
            section_card(
                section_title("Detected Patterns", "scan-eye"),
                rx.foreach(
                    DashboardState.mirror_data,
                    lambda p: rx.hstack(
                        rx.text(p["pattern"], color=TEXT, font_size="13px",
                                font_weight="500"),
                        rx.spacer(),
                        rx.text(p["confidence"] + "%", color=AMBER,
                                font_size="12px"),
                        width="100%",
                        padding_y="4px",
                    ),
                ),
            ),
            rx.fragment(),
        ),

        # Analytics
        rx.cond(
            DashboardState.chat_analytics.length() > 0,
            section_card(
                section_title("Chat Analytics", "bar-chart-2"),
                rx.foreach(
                    DashboardState.chat_analytics,
                    lambda row: rx.hstack(
                        rx.text(row["key"], color=TEXT2, font_size="12px",
                                max_width="120px", overflow="hidden",
                                text_overflow="ellipsis"),
                        rx.spacer(),
                        rx.text(row["value"], color=TEXT, font_size="12px"),
                        width="100%",
                        padding_y="2px",
                    ),
                ),
                max_height="300px",
                overflow_y="auto",
            ),
            rx.fragment(),
        ),

        spacing="3",
        padding="12px",
        width="100%",
    )


def messenger_view() -> rx.Component:
    """Full messenger view — header, messages, input."""
    return rx.cond(
        DashboardState.active_chat_id != "",
        # Chat open
        rx.vstack(
            messenger_header(),
            # Message area
            rx.box(
                rx.vstack(
                    rx.cond(
                        DashboardState.messages_loading,
                        rx.center(rx.spinner(size="3"), padding="40px"),
                        rx.vstack(
                            rx.foreach(DashboardState.messages, _render_message),
                            spacing="1",
                            width="100%",
                            padding="12px 16px",
                        ),
                    ),
                    width="100%",
                ),
                flex="1",
                overflow_y="auto",
                background=BG_CHAT,
                width="100%",
            ),
            # Info panel (visible when analytics loaded)
            messenger_info_panel(),
            messenger_input_bar(),
            spacing="0",
            height="100vh",
            width="100%",
        ),
        # No chat selected
        rx.center(
            rx.vstack(
                rx.icon(tag="message-circle", size=48, color=TEXT2),
                rx.text("Select a chat to start messaging",
                         color=TEXT2, font_size="16px"),
                rx.text("Your conversations will appear here",
                         color=TEXT_TIME, font_size="13px"),
                spacing="3",
                align="center",
            ),
            height="100vh",
            width="100%",
            background=BG_CHAT,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════

def overview_tab() -> rx.Component:
    return rx.vstack(
        # Status header
        rx.hstack(
            rx.text("Dashboard", color=TEXT, font_size="20px", font_weight="700"),
            rx.spacer(),
            rx.hstack(
                rx.box(width="6px", height="6px", border_radius="50%",
                       background=rx.cond(DashboardState.connected, GREEN, RED)),
                rx.text(
                    rx.cond(DashboardState.connected,
                            "Connected", "Disconnected"),
                    color=rx.cond(DashboardState.connected, GREEN, RED),
                    font_size="12px",
                ),
                spacing="2", align="center",
            ),
            width="100%",
            padding="16px",
        ),

        # Stats row
        rx.hstack(
            stat_pill("Auto-Reply",
                      rx.cond(DashboardState.auto_reply_enabled, "ON", "OFF"),
                      rx.cond(DashboardState.auto_reply_enabled, GREEN, RED)),
            stat_pill("Chats", DashboardState.chat_count, BLUE),
            stat_pill("Replies", DashboardState.recent_replies, CYAN),
            stat_pill("Features",
                      DashboardState.active_features.to(str) + "/" +
                      DashboardState.total_features.to(str), PURPLE),
            stat_pill("Engines", DashboardState.engines_count, AMBER),
            spacing="3",
            padding="0 16px",
            flex_wrap="wrap",
        ),

        # Quick toggles
        section_card(
            section_title("Quick Controls", "zap"),
            rx.hstack(
                action_btn(
                    rx.cond(DashboardState.auto_reply_enabled,
                            "Disable Auto-Reply", "Enable Auto-Reply"),
                    DashboardState.toggle_auto_reply,
                    rx.cond(DashboardState.auto_reply_enabled, RED, GREEN),
                    icon="power",
                ),
                action_btn("All Features ON", DashboardState.toggle_all_on,
                           GREEN, icon="check-check"),
                action_btn("All Features OFF", DashboardState.toggle_all_off,
                           RED, icon="x"),
                spacing="2",
                flex_wrap="wrap",
            ),
            margin_x="16px",
        ),

        # Activity feed
        section_card(
            section_title("Recent Activity", "activity"),
            rx.cond(
                DashboardState.activity_log.length() > 0,
                rx.vstack(
                    rx.foreach(
                        DashboardState.activity_log,
                        lambda e: rx.hstack(
                            rx.text(e["time"], color=TEXT_TIME, font_size="11px",
                                    min_width="60px"),
                            rx.text(e["message"], color=TEXT, font_size="12px"),
                            width="100%",
                            padding_y="3px",
                        ),
                    ),
                    spacing="0",
                    max_height="200px",
                    overflow_y="auto",
                ),
                rx.text("No recent activity", color=TEXT2, font_size="13px"),
            ),
            margin_x="16px",
        ),

        # Engine summary
        section_card(
            section_title("Intelligence Engines", "brain"),
            rx.cond(
                DashboardState.engines_data.length() > 0,
                rx.vstack(
                    rx.foreach(
                        DashboardState.engines_data,
                        lambda eng: rx.hstack(
                            rx.box(width="6px", height="6px", border_radius="50%",
                                   background=rx.cond(
                                       eng["status"] == "loaded", GREEN, AMBER)),
                            rx.text(eng["name"], color=TEXT, font_size="13px"),
                            rx.spacer(),
                            rx.text(eng["fn_count"] + " fns", color=TEXT2,
                                    font_size="11px"),
                            width="100%",
                            padding_y="3px",
                        ),
                    ),
                    spacing="0",
                ),
                rx.text("No engines loaded", color=TEXT2, font_size="13px"),
            ),
            margin_x="16px",
        ),

        spacing="4",
        padding_bottom="20px",
        width="100%",
        overflow_y="auto",
        height="100vh",
    )


# ═══════════════════════════════════════════════════════════════════════
# FEATURES TAB
# ═══════════════════════════════════════════════════════════════════════

def _render_feature_toggle(fkey: str) -> rx.Component:
    meta = FEATURE_META.get(fkey, (fkey, "", "settings"))
    label, desc, icon = meta[0], meta[1], meta[2]
    is_on = getattr(DashboardState, f"feat_{fkey}")
    return rx.hstack(
        rx.icon(tag=icon, size=16, color=rx.cond(is_on, BLUE, TEXT2)),
        rx.vstack(
            rx.text(label, color=TEXT, font_size="13px", font_weight="500"),
            rx.text(desc, color=TEXT2, font_size="11px"),
            spacing="0",
        ),
        rx.spacer(),
        rx.box(
            rx.box(
                width="16px", height="16px", border_radius="50%",
                background="white",
                position="absolute",
                top="2px",
                left=rx.cond(is_on, "22px", "2px"),
                transition=_EASE,
            ),
            width="42px", height="20px", border_radius="12px",
            background=rx.cond(is_on, BLUE, "rgba(255,255,255,0.15)"),
            position="relative",
            cursor="pointer",
            transition=_EASE,
            on_click=DashboardState.toggle_feature(fkey),
            flex_shrink="0",
        ),
        spacing="3",
        align="center",
        width="100%",
        padding="10px 16px",
        background=BG_SIDEBAR,
        border=f"1px solid {BORDER_COLOR}",
        border_radius="10px",
    )


def features_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("Feature Toggles", color=TEXT, font_size="20px",
                     font_weight="700"),
            rx.spacer(),
            rx.text(
                DashboardState.active_features.to(str) + " active",
                color=BLUE, font_size="13px",
            ),
            width="100%",
            padding="16px",
        ),
        rx.hstack(
            action_btn("Enable All", DashboardState.toggle_all_on, GREEN),
            action_btn("Disable All", DashboardState.toggle_all_off, RED),
            spacing="2",
            padding_x="16px",
        ),
        rx.vstack(
            *[_render_feature_toggle(k) for k in FEATURE_KEYS],
            spacing="2",
            padding="8px 16px",
        ),
        spacing="3",
        width="100%",
        overflow_y="auto",
        height="100vh",
        padding_bottom="20px",
    )


# ═══════════════════════════════════════════════════════════════════════
# CONFIG TAB
# ═══════════════════════════════════════════════════════════════════════

def config_tab() -> rx.Component:
    return rx.vstack(
        rx.text("Configuration", color=TEXT, font_size="20px",
                 font_weight="700", padding="16px"),

        # Delay config
        section_card(
            section_title("Reply Delay (seconds)", "clock"),
            rx.hstack(
                rx.vstack(
                    rx.text("Min", color=TEXT2, font_size="11px"),
                    styled_input("5", DashboardState.delay_min_input,
                                 DashboardState.set_delay_min_input),
                    spacing="1", flex="1",
                ),
                rx.vstack(
                    rx.text("Max", color=TEXT2, font_size="11px"),
                    styled_input("30", DashboardState.delay_max_input,
                                 DashboardState.set_delay_max_input),
                    spacing="1", flex="1",
                ),
                action_btn("Save", DashboardState.save_delay, BLUE, icon="check"),
                spacing="3", align="end", width="100%",
            ),
            margin_x="16px",
        ),

        # Proactive max
        section_card(
            section_title("Proactive Messages", "send"),
            rx.hstack(
                rx.vstack(
                    rx.text("Max per day", color=TEXT2, font_size="11px"),
                    styled_input("3", DashboardState.proactive_max_input,
                                 DashboardState.set_proactive_max_input),
                    spacing="1", flex="1",
                ),
                action_btn("Save", DashboardState.save_proactive_max,
                           BLUE, icon="check"),
                spacing="3", align="end", width="100%",
            ),
            margin_x="16px",
        ),

        # System prompt
        section_card(
            section_title("System Prompt", "file-text"),
            rx.el.textarea(
                value=DashboardState.system_prompt,
                on_change=DashboardState.set_system_prompt,
                placeholder="Custom system prompt for AI responses...",
                style={
                    "background": BG_INPUT,
                    "border": f"1px solid {BORDER_COLOR}",
                    "color": TEXT,
                    "font_size": "13px",
                    "padding": "10px",
                    "border_radius": "8px",
                    "width": "100%",
                    "min_height": "100px",
                    "resize": "vertical",
                    "outline": "none",
                    "font_family": "inherit",
                    "::placeholder": {"color": TEXT2},
                },
            ),
            action_btn("Save Prompt", DashboardState.save_prompt,
                       BLUE, icon="save"),
            margin_x="16px",
        ),

        # Whitelist
        section_card(
            section_title("Chat Whitelist", "shield"),
            rx.hstack(
                styled_input("Chat ID or @username",
                             DashboardState.wl_input,
                             DashboardState.set_wl_input),
                action_btn("Add", DashboardState.add_to_whitelist, GREEN),
                spacing="2", width="100%",
            ),
            rx.cond(
                DashboardState.chat_ids.length() > 0,
                rx.vstack(
                    rx.foreach(
                        DashboardState.chat_ids,
                        lambda cid: rx.hstack(
                            rx.text(cid, color=TEXT, font_size="13px"),
                            rx.spacer(),
                            rx.icon(tag="x", size=14, color=RED,
                                    cursor="pointer",
                                    on_click=DashboardState.remove_from_whitelist(cid)),
                            width="100%",
                            padding_y="4px",
                        ),
                    ),
                    spacing="0",
                    margin_top="8px",
                ),
                rx.text("No chats whitelisted", color=TEXT2,
                         font_size="12px", margin_top="8px"),
            ),
            margin_x="16px",
        ),

        spacing="4",
        width="100%",
        overflow_y="auto",
        height="100vh",
        padding_bottom="20px",
    )


# ═══════════════════════════════════════════════════════════════════════
# INTELLIGENCE TAB
# ═══════════════════════════════════════════════════════════════════════

def intelligence_tab() -> rx.Component:
    return rx.vstack(
        rx.text("Intelligence & Analysis", color=TEXT, font_size="20px",
                 font_weight="700", padding="16px"),

        # NLP Analysis
        section_card(
            section_title("Deep Analysis (V5)", "brain"),
            rx.hstack(
                styled_input("Chat ID", DashboardState.analysis_chat_id,
                             DashboardState.set_analysis_chat_id),
                action_btn("Analyze", DashboardState.run_analysis, BLUE,
                           icon="scan"),
                spacing="2", width="100%",
            ),
            rx.cond(
                DashboardState.analysis_loading,
                rx.center(rx.spinner(size="2"), padding="12px"),
                rx.cond(
                    DashboardState.analysis_result.length() > 0,
                    rx.box(
                        rx.foreach(
                            DashboardState.analysis_result,
                            lambda r: rx.hstack(
                                rx.text(r["key"], color=TEXT2, font_size="11px",
                                        min_width="140px"),
                                rx.text(r["value"], color=TEXT, font_size="12px"),
                                width="100%", padding_y="2px",
                            ),
                        ),
                        max_height="250px",
                        overflow_y="auto",
                        margin_top="8px",
                    ),
                    rx.fragment(),
                ),
            ),
            margin_x="16px",
        ),

        # Conversation Health
        section_card(
            section_title("Conversation Health", "heart"),
            rx.hstack(
                styled_input("Chat ID", DashboardState.health_chat_id,
                             DashboardState.set_health_chat_id),
                action_btn("Check", DashboardState.run_health, GREEN,
                           icon="heart-pulse"),
                spacing="2", width="100%",
            ),
            rx.cond(
                DashboardState.health_loading,
                rx.center(rx.spinner(size="2"), padding="12px"),
                rx.cond(
                    DashboardState.health_result.length() > 0,
                    rx.box(
                        rx.foreach(
                            DashboardState.health_result,
                            lambda r: rx.hstack(
                                rx.text(r["key"], color=TEXT2, font_size="11px",
                                        min_width="140px"),
                                rx.text(r["value"], color=TEXT, font_size="12px"),
                                width="100%", padding_y="2px",
                            ),
                        ),
                        max_height="200px",
                        overflow_y="auto",
                        margin_top="8px",
                    ),
                    rx.fragment(),
                ),
            ),
            margin_x="16px",
        ),

        # Psychological Analysis
        section_card(
            section_title("Psychological Analysis", "scan-eye"),
            rx.hstack(
                styled_input("Chat ID", DashboardState.psych_chat_id,
                             DashboardState.set_psych_chat_id),
                action_btn("Analyze", DashboardState.run_psych, PURPLE,
                           icon="brain"),
                spacing="2", width="100%",
            ),
            rx.cond(
                DashboardState.psych_loading,
                rx.center(rx.spinner(size="2"), padding="12px"),
                rx.cond(
                    DashboardState.psych_result.length() > 0,
                    rx.box(
                        rx.foreach(
                            DashboardState.psych_result,
                            lambda r: rx.hstack(
                                rx.text(r["key"], color=TEXT2, font_size="11px",
                                        min_width="140px"),
                                rx.text(r["value"], color=TEXT, font_size="12px"),
                                width="100%", padding_y="2px",
                            ),
                        ),
                        max_height="200px",
                        overflow_y="auto",
                        margin_top="8px",
                    ),
                    rx.fragment(),
                ),
            ),
            margin_x="16px",
        ),

        spacing="4",
        width="100%",
        overflow_y="auto",
        height="100vh",
        padding_bottom="20px",
    )


# ═══════════════════════════════════════════════════════════════════════
# ENGINES TAB
# ═══════════════════════════════════════════════════════════════════════

def engines_tab() -> rx.Component:
    return rx.vstack(
        rx.text("Intelligence Engines", color=TEXT, font_size="20px",
                 font_weight="700", padding="16px"),

        rx.cond(
            DashboardState.engine_details.length() > 0,
            rx.vstack(
                rx.foreach(
                    DashboardState.engine_details,
                    lambda eng: section_card(
                        rx.hstack(
                            rx.box(width="8px", height="8px", border_radius="50%",
                                   background=GREEN),
                            rx.text(eng["name"], color=TEXT, font_size="14px",
                                    font_weight="600"),
                            rx.spacer(),
                            rx.text(eng["fn_count"] + " functions",
                                    color=BLUE, font_size="12px"),
                            width="100%",
                        ),
                        margin_x="16px",
                    ),
                ),
                spacing="2",
            ),
            rx.text("Loading engines...", color=TEXT2, padding="16px"),
        ),

        spacing="3",
        width="100%",
        overflow_y="auto",
        height="100vh",
        padding_bottom="20px",
    )


# ═══════════════════════════════════════════════════════════════════════
# MODELS TAB
# ═══════════════════════════════════════════════════════════════════════

def models_tab() -> rx.Component:
    return rx.vstack(
        rx.text("ML Models", color=TEXT, font_size="20px",
                 font_weight="700", padding="16px"),

        # Training actions
        rx.hstack(
            action_btn("Preload Models", DashboardState.preload_models,
                       BLUE, icon="download"),
            action_btn("Train sklearn", DashboardState.train_sklearn,
                       GREEN, icon="dumbbell"),
            action_btn("Train All", DashboardState.train_all,
                       PURPLE, icon="brain"),
            spacing="2",
            padding_x="16px",
            flex_wrap="wrap",
        ),

        rx.cond(
            DashboardState.training_result != "",
            rx.box(
                rx.text(DashboardState.training_result, color=GREEN,
                         font_size="13px"),
                padding="8px 16px",
            ),
            rx.fragment(),
        ),

        # sklearn models
        rx.cond(
            DashboardState.sklearn_models.length() > 0,
            section_card(
                section_title("sklearn Classifiers", "cpu"),
                rx.foreach(
                    DashboardState.sklearn_models,
                    lambda m: rx.hstack(
                        rx.text(m["name"], color=TEXT, font_size="13px",
                                font_weight="500", min_width="120px"),
                        rx.text(m["classifier_type"], color=TEXT2,
                                font_size="11px"),
                        rx.spacer(),
                        rx.text("Classes: " + m["class_count"],
                                color=BLUE, font_size="11px"),
                        width="100%",
                        padding_y="4px",
                    ),
                ),
                margin_x="16px",
            ),
            rx.fragment(),
        ),

        # Neural models
        rx.cond(
            DashboardState.neural_models.length() > 0,
            section_card(
                section_title("Neural Networks", "brain"),
                rx.foreach(
                    DashboardState.neural_models,
                    lambda m: rx.hstack(
                        rx.text(m["name"], color=TEXT, font_size="13px",
                                font_weight="500", min_width="120px"),
                        rx.text(m["type"], color=TEXT2, font_size="11px"),
                        rx.spacer(),
                        rx.text("Classes: " + m["class_count"],
                                color=PURPLE, font_size="11px"),
                        width="100%",
                        padding_y="4px",
                    ),
                ),
                margin_x="16px",
            ),
            rx.fragment(),
        ),

        # DL Status
        rx.cond(
            DashboardState.dl_status_data.length() > 0,
            section_card(
                section_title("Deep Learning Status", "database"),
                rx.foreach(
                    DashboardState.dl_status_data,
                    lambda r: rx.hstack(
                        rx.text(r["key"], color=TEXT2, font_size="11px",
                                min_width="120px"),
                        rx.text(r["value"], color=TEXT, font_size="12px"),
                        width="100%", padding_y="2px",
                    ),
                ),
                max_height="250px",
                overflow_y="auto",
                margin_x="16px",
            ),
            rx.fragment(),
        ),

        spacing="4",
        width="100%",
        overflow_y="auto",
        height="100vh",
        padding_bottom="20px",
    )


# ═══════════════════════════════════════════════════════════════════════
# LOG TAB
# ═══════════════════════════════════════════════════════════════════════

def log_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("Auto-Reply Log", color=TEXT, font_size="20px",
                     font_weight="700"),
            rx.spacer(),
            action_btn("Refresh", DashboardState.load_full_log, BLUE,
                       icon="refresh-cw"),
            padding="16px",
            width="100%",
        ),
        rx.cond(
            DashboardState.full_log.length() > 0,
            rx.box(
                rx.foreach(
                    DashboardState.full_log,
                    lambda entry: rx.hstack(
                        rx.text(entry["time"], color=TEXT_TIME,
                                font_size="11px", min_width="55px",
                                font_family="monospace"),
                        rx.text(entry["chat"], color=BLUE,
                                font_size="11px", min_width="80px"),
                        rx.text(entry["action"], color=AMBER,
                                font_size="11px", min_width="60px"),
                        rx.text(entry["detail"], color=TEXT,
                                font_size="12px", overflow="hidden",
                                text_overflow="ellipsis",
                                white_space="nowrap"),
                        width="100%",
                        padding="6px 16px",
                        border_bottom=f"1px solid {BORDER_COLOR}",
                        _hover={"background": BG_HOVER},
                    ),
                ),
                width="100%",
                overflow_y="auto",
                max_height="calc(100vh - 80px)",
            ),
            rx.center(
                rx.text("No log entries", color=TEXT2, font_size="14px"),
                padding="40px",
            ),
        ),
        spacing="0",
        width="100%",
        height="100vh",
    )


# ═══════════════════════════════════════════════════════════════════════
# INSTRUCTIONS TAB (merged into config conceptually)
# ═══════════════════════════════════════════════════════════════════════

def instructions_section() -> rx.Component:
    """Per-chat instructions — embedded in config or intelligence tab."""
    return section_card(
        section_title("Per-Chat Instructions", "scroll-text"),
        rx.hstack(
            styled_input("Chat ID", DashboardState.inst_chat_id,
                         DashboardState.set_inst_chat_id),
            spacing="2", width="100%",
        ),
        rx.el.textarea(
            value=DashboardState.inst_text,
            on_change=DashboardState.set_inst_text,
            placeholder="Custom instructions for this chat...",
            style={
                "background": BG_INPUT,
                "border": f"1px solid {BORDER_COLOR}",
                "color": TEXT,
                "font_size": "13px",
                "padding": "8px",
                "border_radius": "8px",
                "width": "100%",
                "min_height": "60px",
                "resize": "vertical",
                "outline": "none",
                "margin_top": "8px",
                "::placeholder": {"color": TEXT2},
            },
        ),
        rx.box(
            action_btn("Save", DashboardState.save_instruction, GREEN,
                       icon="save"),
            margin_top="8px",
        ),
        rx.cond(
            DashboardState.instructions_data.length() > 0,
            rx.vstack(
                rx.foreach(
                    DashboardState.instructions_data,
                    lambda inst: rx.hstack(
                        rx.vstack(
                            rx.text(inst["chat_id"], color=BLUE,
                                    font_size="12px", font_weight="500"),
                            rx.text(inst["instructions"], color=TEXT,
                                    font_size="12px"),
                            spacing="0",
                        ),
                        rx.spacer(),
                        rx.icon(tag="trash-2", size=14, color=RED,
                                cursor="pointer",
                                on_click=DashboardState.remove_instruction(
                                    inst["chat_id"])),
                        width="100%",
                        padding_y="6px",
                        border_bottom=f"1px solid {BORDER_COLOR}",
                    ),
                ),
                spacing="0",
                margin_top="12px",
            ),
            rx.fragment(),
        ),
        margin_x="16px",
    )


# ═══════════════════════════════════════════════════════════════════════
# ACCOUNT TAB
# ═══════════════════════════════════════════════════════════════════════

def account_tab() -> rx.Component:
    return rx.vstack(
        rx.text("My Account", color=TEXT, font_size="20px",
                 font_weight="700", padding="16px"),
        rx.cond(
            DashboardState.my_info.length() > 0,
            section_card(
                rx.foreach(
                    DashboardState.my_info,
                    lambda r: rx.hstack(
                        rx.text(r["key"], color=TEXT2, font_size="13px",
                                min_width="120px", font_weight="500"),
                        rx.text(r["value"], color=TEXT, font_size="13px"),
                        width="100%",
                        padding_y="4px",
                    ),
                ),
                margin_x="16px",
            ),
            rx.center(
                rx.spinner(size="3"),
                padding="40px",
            ),
        ),
        spacing="3",
        width="100%",
        overflow_y="auto",
        height="100vh",
    )


# ═══════════════════════════════════════════════════════════════════════
# RL TAB
# ═══════════════════════════════════════════════════════════════════════

def rl_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("Reinforcement Learning", color=TEXT, font_size="20px",
                     font_weight="700"),
            rx.spacer(),
            action_btn("Refresh", DashboardState.load_rl_all, PURPLE,
                       icon="refresh-cw"),
            padding="16px",
            width="100%",
        ),
        rx.cond(
            DashboardState.rl_loading,
            rx.center(rx.spinner(size="3"), padding="40px"),
            rx.cond(
                DashboardState.rl_result.length() > 0,
                section_card(
                    rx.foreach(
                        DashboardState.rl_result,
                        lambda r: rx.hstack(
                            rx.text(r["key"], color=TEXT2, font_size="12px",
                                    min_width="140px"),
                            rx.text(r["value"], color=TEXT, font_size="12px"),
                            width="100%", padding_y="2px",
                        ),
                    ),
                    max_height="calc(100vh - 100px)",
                    overflow_y="auto",
                    margin_x="16px",
                ),
                rx.center(
                    rx.text("No RL data available", color=TEXT2),
                    padding="40px",
                ),
            ),
        ),
        spacing="0",
        width="100%",
        height="100vh",
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN CONTENT ROUTER
# ═══════════════════════════════════════════════════════════════════════

def content_panel() -> rx.Component:
    """Right panel — routes to the active tab content."""
    return rx.box(
        rx.match(
            DashboardState.active_tab,
            ("messenger", messenger_view()),
            ("overview", overview_tab()),
            ("features", features_tab()),
            ("config", rx.vstack(
                config_tab(),
                instructions_section(),
                spacing="0",
                width="100%",
                overflow_y="auto",
                height="100vh",
            )),
            ("intelligence", intelligence_tab()),
            ("engines", engines_tab()),
            ("models", models_tab()),
            ("log", log_tab()),
            ("account", account_tab()),
            ("rl", rl_tab()),
            overview_tab(),
        ),
        flex="1",
        height="100vh",
        overflow="hidden",
    )


# ═══════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════

@rx.page(route="/", on_load=DashboardState.on_load)
def index() -> rx.Component:
    return rx.hstack(
        sidebar(),
        content_panel(),
        spacing="0",
        width="100vw",
        height="100vh",
        background=BG,
        overflow="hidden",
    )


# ═══════════════════════════════════════════════════════════════════════
# APP + GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════════

style = {
    "@keyframes fadeIn": {
        "from": {"opacity": "0"},
        "to": {"opacity": "1"},
    },
    "@keyframes slideUp": {
        "from": {"opacity": "0", "transform": "translateY(12px)"},
        "to": {"opacity": "1", "transform": "translateY(0)"},
    },
    "body": {
        "background": BG,
        "color": TEXT,
        "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "margin": "0",
        "padding": "0",
    },
    "::-webkit-scrollbar": {
        "width": "6px",
    },
    "::-webkit-scrollbar-track": {
        "background": "transparent",
    },
    "::-webkit-scrollbar-thumb": {
        "background": "rgba(255,255,255,0.12)",
        "border_radius": "3px",
    },
    "button, [role='button']": {
        "font_family": "inherit !important",
    },
    "input, textarea": {
        "font_family": "inherit !important",
    },
}

app = rx.App(
    theme=rx.theme(appearance="dark", accent_color="blue", radius="medium"),
    style=style,
)
