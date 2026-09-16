"""Central design tokens + stylesheet builder for the HeartWatch host app.

Per the design handoff: color encodes meaning, not decoration. No gradients or
drop shadows (except a subtle focus ring). Two font weights only: 400 and 500.
No hardcoded hex values live in the widget modules -- everything routes through
the token dicts below so light/dark stay in sync.
"""

from __future__ import annotations

# --- Semantic color tokens -------------------------------------------------

LIGHT = {
    "bg": "#f4f5f7",          # window background
    "surface_1": "#ffffff",   # cards, AI bar
    "surface_2": "#eceef1",   # inset tracks, hover
    "text": "#1c1e21",
    "text_muted": "#6b7280",
    "border": "#e3e5e9",
    "accent": "#2f6fed",      # blue  -> heart rate accent / sitting
    "success": "#1f9d57",     # green -> normal activity / walking / peak HR
    "warning": "#d9902b",     # amber -> gyroscope / standing / moderate
    "danger": "#e0484d",      # red   -> heart rate line / running / danger
    "accent_soft": "#e6efff",
    "success_soft": "#e2f4ea",
    "warning_soft": "#f7ecd9",
    "danger_soft": "#f8e5e6",
}

DARK = {
    "bg": "#141619",
    "surface_1": "#1e2125",
    "surface_2": "#262a2f",
    "text": "#e7e8ea",
    "text_muted": "#9aa0a6",
    "border": "#2f333a",
    "accent": "#4f8bff",
    "success": "#37c082",
    "warning": "#efab4c",
    "danger": "#ff5c60",
    "accent_soft": "#1c2b47",
    "success_soft": "#173a2c",
    "warning_soft": "#3a2e1a",
    "danger_soft": "#3d1f22",
}

FONT_FAMILY = (
    '"SF Pro Text", "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
)

RADIUS_CARD = 12
RADIUS_CONTROL = 8
PAD_CONTENT = 20


class Theme:
    """Holds the active token set and hands out colors + a global stylesheet."""

    def __init__(self, mode: str = "dark"):
        self.mode = mode
        self.tokens = DARK if mode == "dark" else LIGHT

    def toggle(self) -> None:
        self.mode = "light" if self.mode == "dark" else "dark"
        self.tokens = DARK if self.mode == "dark" else LIGHT

    def c(self, name: str) -> str:
        """Return a token hex string. Widget code calls this, never a literal."""
        return self.tokens[name]

    # -- activity palette (shared by dashboard + live monitor) --------------
    def activity_color(self, activity: str) -> str:
        return {
            "Sitting": self.c("accent"),
            "Walking": self.c("success"),
            "Standing": self.c("warning"),
            "Running": self.c("danger"),
        }.get(activity, self.c("text_muted"))

    # -- global QSS -------------------------------------------------------
    def stylesheet(self) -> str:
        t = self.tokens
        return f"""
        * {{
            font-family: {FONT_FAMILY};
            font-weight: 400;
            color: {t['text']};
        }}
        QMainWindow, QWidget#RootWidget {{
            background: {t['bg']};
        }}
        QScrollArea {{ border: none; background: transparent; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}

        /* --- Sidebar --- */
        QWidget#Sidebar {{
            background: {t['surface_1']};
            border-right: 1px solid {t['border']};
        }}
        QToolButton#NavButton {{
            border: none;
            border-radius: {RADIUS_CONTROL}px;
            padding: 8px;
            color: {t['text_muted']};
            background: transparent;
        }}
        QToolButton#NavButton:hover {{
            background: {t['surface_2']};
        }}
        QToolButton#NavButton:checked {{
            background: {t['accent_soft']};
            color: {t['accent']};
        }}

        /* --- Topbar --- */
        QWidget#Topbar {{
            background: {t['bg']};
            border-bottom: 1px solid {t['border']};
        }}
        QLabel#TopbarTitle {{ font-size: 16px; font-weight: 500; }}
        QLabel#TopbarSubtitle {{ font-size: 12px; color: {t['text_muted']}; }}
        QLabel#ConnPill {{
            font-size: 11px;
            color: {t['text_muted']};
            background: {t['surface_2']};
            border-radius: {RADIUS_CONTROL}px;
            padding: 4px 10px;
        }}

        /* --- Cards --- */
        QFrame#Card {{
            background: {t['surface_1']};
            border-radius: {RADIUS_CARD}px;
        }}
        QLabel#CardLabel {{ font-size: 11px; color: {t['text_muted']}; }}
        QLabel#CardValue {{ font-size: 26px; font-weight: 500; }}
        QLabel#CardUnit  {{ font-size: 12px; font-weight: 500; }}
        QLabel#SectionHeading {{ font-size: 13px; font-weight: 500; }}
        QLabel#Muted {{ color: {t['text_muted']}; font-size: 12px; }}
        QLabel#Mono {{
            font-family: "SF Mono", "JetBrains Mono", "Menlo", monospace;
            font-size: 12px;
        }}

        /* --- AI bar --- */
        QWidget#AiBar {{
            background: {t['surface_1']};
            border-top: 1px solid {t['border']};
        }}
        QLineEdit#AiInput {{
            background: {t['bg']};
            border: 1px solid {t['border']};
            border-radius: {RADIUS_CONTROL}px;
            padding: 8px 12px;
            font-size: 13px;
        }}
        QLineEdit#AiInput:focus {{ border: 1px solid {t['accent']}; }}
        QPushButton#AccentButton {{
            background: {t['accent']};
            color: #ffffff;
            border: none;
            border-radius: {RADIUS_CONTROL}px;
            padding: 8px 18px;
            font-weight: 500;
        }}
        QPushButton#AccentButton:hover {{ background: {t['accent']}; }}
        QPushButton#AccentButton:disabled {{
            background: {t['surface_2']};
            color: {t['text_muted']};
        }}
        QFrame#AiResult {{
            background: {t['bg']};
            border: 1px solid {t['border']};
            border-radius: {RADIUS_CONTROL}px;
        }}

        /* --- generic controls in stub views --- */
        QRadioButton, QCheckBox {{ font-size: 13px; }}
        QPushButton#GhostButton {{
            background: {t['surface_2']};
            border: none;
            border-radius: {RADIUS_CONTROL}px;
            padding: 8px 14px;
        }}
        QFrame#Divider {{ background: {t['border']}; max-height: 1px; min-height: 1px; }}

        /* --- Sessions table + detail dialog --- */
        QTableWidget {{
            background: transparent;
            border: none;
            gridline-color: {t['border']};
            selection-background-color: {t['accent_soft']};
            selection-color: {t['accent']};
        }}
        QTableWidget::item {{ padding: 6px 4px; }}
        QHeaderView::section {{
            background: {t['surface_2']};
            color: {t['text_muted']};
            border: none;
            padding: 6px 4px;
            font-size: 11px;
        }}
        QDialog {{ background: {t['bg']}; }}
        QComboBox {{
            background: {t['surface_2']};
            border: 1px solid {t['border']};
            border-radius: {RADIUS_CONTROL}px;
            padding: 4px 8px;
        }}
        """
