import streamlit as st

from ui.theme import FONT_BODY, FONT_DISPLAY, GOOGLE_FONTS_URL, Theme


def inject_global_css(theme: Theme) -> None:
    st.markdown(_build_css(theme), unsafe_allow_html=True)


def _build_css(t: Theme) -> str:
    return f"""
    <style>
    @import url('{GOOGLE_FONTS_URL}');

    /* ---------- chrome cleanup ---------- */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* ---------- base app ---------- */
    .stApp {{
        background: {t.bg_gradient};
        color: {t.text_primary};
    }}
    html, body, [class*="css"] {{
        font-family: {FONT_BODY};
    }}
    [data-testid="stMainBlockContainer"] {{
        max-width: 880px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}
    h1, h2, h3 {{
        font-family: {FONT_DISPLAY};
        color: {t.text_primary};
    }}
    p, span, label, div {{
        color: {t.text_secondary};
    }}

    /* ---------- entrance animation ---------- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .anim {{
        animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}

    /* ---------- hero header ---------- */
    .hero-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }}
    .hero-brand {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .hero-logo {{
        font-size: 1.7rem;
        filter: drop-shadow(0 0 12px {t.accent_cyan}66);
    }}
    .hero-title {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        font-size: 1.35rem;
        letter-spacing: 0.01em;
        background: {t.accent_gradient};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 500;
        color: {t.text_muted};
        background: {t.surface};
        border: 1px solid {t.surface_border};
        border-radius: 999px;
        padding: 0.32rem 0.85rem;
    }}
    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {t.success};
        box-shadow: 0 0 8px {t.success};
    }}
    .hero-subtitle {{
        color: {t.text_muted};
        font-size: 0.92rem;
        margin-top: 0.35rem;
        margin-bottom: 1.6rem;
        line-height: 1.5;
    }}
    .section-heading {{
        font-family: {FONT_DISPLAY};
        font-weight: 600;
        font-size: 1.05rem;
        color: {t.text_primary};
        margin: 0 0 0.9rem 0;
    }}

    /* ---------- glass cards (st.container(key="card-...")) ---------- */
    div[class*="st-key-card-"] {{
        background: {t.surface} !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid {t.surface_border} !important;
        border-radius: 22px !important;
        padding: 1.6rem 1.7rem !important;
        box-shadow: {t.shadow};
        margin-bottom: 1.3rem;
        transition: border-color 0.35s ease, box-shadow 0.35s ease, transform 0.35s ease;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    div[class*="st-key-card-"]:hover {{
        border-color: {t.surface_hover_border} !important;
        box-shadow: {t.glow};
        transform: translateY(-2px);
    }}
    .st-key-card-actions {{ animation-delay: 0.1s; }}
    .st-key-card-gauge {{ animation-delay: 0.16s; }}
    .st-key-card-explain {{ animation-delay: 0.22s; }}
    .st-key-card-slider-Moisture {{ animation-delay: 0.02s; }}
    .st-key-card-slider-Volatile_matter {{ animation-delay: 0.05s; }}
    .st-key-card-slider-Fixed_Carbon {{ animation-delay: 0.08s; }}
    .st-key-card-slider-Std\.Ash {{ animation-delay: 0.11s; }}
    div[class*="st-key-card-slider-"] {{
        padding: 1.15rem 1.3rem !important;
    }}

    /* ---------- feature slider cards ---------- */
    .slider-head {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }}
    .slider-icon {{ font-size: 1.05rem; }}
    .slider-label {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {t.text_muted};
    }}
    .slider-value-row {{
        display: flex;
        align-items: baseline;
        gap: 0.3rem;
        margin-bottom: 0.4rem;
        animation: fadeInUp 0.35s ease both;
    }}
    .slider-value {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        font-size: 2rem;
        line-height: 1;
        background: {t.accent_gradient};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .slider-unit {{
        font-size: 1rem;
        font-weight: 500;
        color: {t.text_muted};
    }}
    .slider-minmax {{
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: {t.text_muted};
        margin-top: -0.3rem;
        padding: 0 0.2rem;
    }}

    /* circular +/- buttons */
    div[class*="st-key-btn-dec-"] [data-testid="stBaseButton-secondary"],
    div[class*="st-key-btn-inc-"] [data-testid="stBaseButton-secondary"] {{
        background: {t.accent_gradient} !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 2.4rem !important;
        height: 2.4rem !important;
        min-width: 2.4rem !important;
        padding: 0 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px -6px {t.accent_blue}aa;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }}
    div[class*="st-key-btn-dec-"] [data-testid="stBaseButton-secondary"]:hover,
    div[class*="st-key-btn-inc-"] [data-testid="stBaseButton-secondary"]:hover {{
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 10px 22px -6px {t.accent_blue}cc;
    }}
    div[class*="st-key-btn-dec-"] [data-testid="stBaseButton-secondary"]:active,
    div[class*="st-key-btn-inc-"] [data-testid="stBaseButton-secondary"]:active {{
        transform: scale(0.88);
        box-shadow: 0 0 0 8px {t.accent_cyan}33;
    }}

    /* ---------- native slider restyle ---------- */
    [data-testid="stSlider"] {{ padding-top: 0.2rem; }}
    [data-testid="stSlider"] label {{ display: none; }}
    [data-testid="stSlider"] [data-testid="stSliderTickBar"] {{
        display: none;
    }}
    [data-testid="stSlider"] div[style*="height: 0.25rem"] {{
        background: {t.accent_gradient} !important;
    }}
    [data-testid="stSlider"] div[role="slider"] {{
        background: white !important;
        box-shadow: 0 0 0 4px {t.accent_cyan}55, 0 2px 8px rgba(0,0,0,0.4) !important;
        transition: box-shadow 0.2s ease, transform 0.15s ease;
    }}
    [data-testid="stSlider"] div[role="slider"]:hover {{
        transform: scale(1.15);
    }}
    [data-testid="stSliderThumbValue"] {{
        color: {t.text_primary} !important;
        font-weight: 600;
    }}

    /* ---------- number inputs ---------- */
    [data-testid="stNumberInput"] label p {{
        font-size: 0.82rem;
        font-weight: 500;
        color: {t.text_muted};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stNumberInput"] [data-baseweb="base-input"],
    [data-testid="stNumberInput"] [data-baseweb="input"] {{
        background: rgba(127, 127, 127, 0.1) !important;
        border: 1px solid {t.surface_border} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stNumberInput"] input {{
        background: transparent !important;
        color: {t.text_primary} !important;
        font-family: {FONT_DISPLAY};
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: {t.text_primary} !important;
    }}
    [data-testid="stNumberInput"] [data-baseweb="base-input"]:focus-within {{
        border-color: {t.accent_cyan} !important;
        box-shadow: 0 0 0 3px {t.accent_cyan}26 !important;
    }}
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{
        background: rgba(127, 127, 127, 0.1) !important;
        color: {t.text_secondary} !important;
    }}

    /* ---------- buttons ---------- */
    [data-testid="stBaseButton-primary"] {{
        background: {t.accent_gradient} !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 24px -8px {t.accent_blue}88;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stBaseButton-primary"]:hover {{
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 12px 30px -8px {t.accent_blue}aa;
    }}
    [data-testid="stBaseButton-primary"]:active {{
        transform: translateY(0) scale(0.98);
    }}
    [data-testid="stBaseButton-primary"]:focus:not(:active),
    [data-testid="stBaseButton-secondary"]:focus:not(:active) {{
        outline: none !important;
        box-shadow: 0 0 0 3px {t.accent_cyan}40 !important;
    }}
    [data-testid="stBaseButton-secondary"] {{
        background: transparent !important;
        border: 1px solid {t.surface_border} !important;
        border-radius: 12px !important;
        color: {t.text_secondary} !important;
        font-weight: 500 !important;
        transition: border-color 0.2s ease, transform 0.2s ease, color 0.2s ease;
    }}
    [data-testid="stBaseButton-secondary"]:hover {{
        border-color: {t.accent_cyan}99 !important;
        color: {t.text_primary} !important;
        transform: translateY(-1px);
    }}

    /* ---------- KPI cards (custom HTML, one per st.columns slot) ---------- */
    .kpi-card {{
        background: {t.surface};
        backdrop-filter: blur(18px);
        border: 1px solid {t.surface_border};
        border-radius: 20px;
        padding: 1.25rem 1.1rem;
        box-shadow: {t.shadow};
        transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    .kpi-card:hover {{
        border-color: {t.surface_hover_border};
        box-shadow: {t.glow};
        transform: translateY(-3px);
    }}
    .kpi-icon {{ font-size: 1.3rem; margin-bottom: 0.5rem; opacity: 0.9; }}
    .kpi-label {{
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {t.text_muted};
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-family: {FONT_DISPLAY};
        font-weight: 700;
        font-size: 1.55rem;
        color: {t.text_primary};
        line-height: 1.15;
    }}
    .kpi-caption {{ font-size: 0.72rem; color: {t.text_muted}; margin-top: 0.25rem; }}

    /* ---------- explanation sentence ---------- */
    .explain-sentence {{
        font-size: 0.96rem;
        line-height: 1.6;
        color: {t.text_secondary};
        border-left: 3px solid {t.accent_cyan};
        padding-left: 0.9rem;
        margin: 0.4rem 0 1.1rem 0;
    }}

    /* ---------- sum-validation pill ---------- */
    .sum-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.82rem;
        font-weight: 500;
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
        margin-top: 0.6rem;
    }}

    /* ---------- footer ---------- */
    .app-footer {{
        text-align: center;
        font-size: 0.78rem;
        color: {t.text_muted};
        margin-top: 2.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid {t.surface_border};
    }}
    .app-footer a {{
        color: {t.accent_cyan};
        text-decoration: none;
    }}

    /* ---------- responsive ---------- */
    div[class*="st-key-card-slider-"] [data-testid="stHorizontalBlock"] {{
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.5rem;
    }}
    div[class*="st-key-card-slider-"] [data-testid="stColumn"]:first-child,
    div[class*="st-key-card-slider-"] [data-testid="stColumn"]:last-child {{
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }}
    div[class*="st-key-card-slider-"] [data-testid="stColumn"]:nth-child(2) {{
        flex: 1 1 0% !important;
        width: auto !important;
        min-width: 0 !important;
    }}

    @media (max-width: 640px) {{
        .hero-title {{ font-size: 1.1rem; }}
        div[class*="st-key-card-"] {{ padding: 1.2rem 1.1rem !important; }}
    }}
    </style>
    """
