from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    bg_gradient: str
    surface: str
    surface_border: str
    surface_hover_border: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent_blue: str
    accent_cyan: str
    accent_purple: str
    accent_gradient: str
    success: str
    warning: str
    error: str
    shadow: str
    glow: str


DARK = Theme(
    name="dark",
    bg_gradient="radial-gradient(circle at 15% 0%, #16213a 0%, #0a0e1a 45%, #05070d 100%)",
    surface="rgba(255, 255, 255, 0.045)",
    surface_border="rgba(255, 255, 255, 0.09)",
    surface_hover_border="rgba(56, 189, 248, 0.45)",
    text_primary="#f1f5f9",
    text_secondary="#b8c2d4",
    text_muted="#7a869e",
    accent_blue="#3b82f6",
    accent_cyan="#22d3ee",
    accent_purple="#a78bfa",
    accent_gradient="linear-gradient(135deg, #3b82f6 0%, #22d3ee 55%, #a78bfa 100%)",
    success="#34d399",
    warning="#fbbf24",
    error="#f87171",
    shadow="0 20px 60px -15px rgba(0, 0, 0, 0.65)",
    glow="0 0 0 1px rgba(56, 189, 248, 0.25), 0 12px 40px -10px rgba(56, 189, 248, 0.25)",
)

LIGHT = Theme(
    name="light",
    bg_gradient="radial-gradient(circle at 15% 0%, #eef2ff 0%, #f6f8fc 45%, #ffffff 100%)",
    surface="rgba(255, 255, 255, 0.7)",
    surface_border="rgba(15, 23, 42, 0.08)",
    surface_hover_border="rgba(37, 99, 235, 0.35)",
    text_primary="#0f172a",
    text_secondary="#3f4a5e",
    text_muted="#697489",
    accent_blue="#2563eb",
    accent_cyan="#0891b2",
    accent_purple="#7c3aed",
    accent_gradient="linear-gradient(135deg, #2563eb 0%, #0891b2 55%, #7c3aed 100%)",
    success="#059669",
    warning="#d97706",
    error="#dc2626",
    shadow="0 20px 50px -20px rgba(15, 23, 42, 0.18)",
    glow="0 0 0 1px rgba(37, 99, 235, 0.18), 0 12px 32px -12px rgba(37, 99, 235, 0.22)",
)

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Plus Jakarta Sans', sans-serif"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"
)


def get_theme(name: str) -> Theme:
    return LIGHT if name == "light" else DARK


def confidence_color(theme: Theme, confidence: float) -> str:
    if confidence >= 70:
        return theme.success
    if confidence >= 40:
        return theme.warning
    return theme.error


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
