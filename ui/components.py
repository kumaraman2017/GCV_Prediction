import streamlit as st

from ui.theme import Theme


def render_header(theme: Theme) -> None:
    col_brand, col_status, col_toggle = st.columns([5, 2, 1])
    with col_brand:
        st.markdown(
            """
            <div class="hero-row anim">
                <div class="hero-brand">
                    <span class="hero-logo">⛏️</span>
                    <h1 class="hero-title">COAL GCV PREDICTOR</h1>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-end; padding-top:6px;">
                <span class="status-pill anim"><span class="status-dot"></span>Model Online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_toggle:
        is_dark = theme.name == "dark"
        st.toggle("🌙", value=is_dark, key="dark_mode_toggle", label_visibility="collapsed")

    st.markdown(
        '<p class="hero-subtitle anim">Predicts Gross Calorific Value from proximate analysis — '
        "Moisture, Volatile Matter, Fixed Carbon, and Ash — with a per-prediction confidence "
        "score and a SHAP-based explanation.</p>",
        unsafe_allow_html=True,
    )


def render_sum_pill(total: float, theme: Theme) -> None:
    in_range = abs(total - 100.0) <= 0.5
    color = theme.success if in_range else theme.warning
    icon = "✓" if in_range else "⚠"
    note = "" if in_range else " — proximate values normally sum to ~100%"
    st.markdown(
        f"""
        <div class="sum-pill" style="background:{color}1a; color:{color}; border:1px solid {color}40;">
            {icon} Sum of inputs: {total:.1f}%{note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(icon: str, label: str, value: str, caption: str, value_color: str | None = None) -> None:
    color_style = f"color:{value_color};" if value_color else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="{color_style}">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step_value(key: str, step: float, min_value: float, max_value: float) -> None:
    st.session_state[key] = round(min(max_value, max(min_value, st.session_state[key] + step)), 1)


def render_feature_input(
    icon: str,
    label: str,
    key: str,
    min_value: float = 0.0,
    max_value: float = 100.0,
    step: float = 0.1,
) -> None:
    with st.container(key=f"card-slider-{key}"):
        st.markdown(
            f"""
            <div class="slider-head">
                <span class="slider-icon">{icon}</span>
                <span class="slider-label">{label}</span>
            </div>
            <div class="slider-value-row">
                <span class="slider-value">{st.session_state[key]:.1f}</span>
                <span class="slider-unit">%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_minus, col_slider, col_plus = st.columns([1.1, 7, 1.1], vertical_alignment="center")
        with col_minus:
            with st.container(key=f"btn-dec-{key}"):
                st.button(
                    "−",
                    key=f"{key}_dec",
                    use_container_width=True,
                    on_click=_step_value,
                    args=(key, -step, min_value, max_value),
                )
        with col_slider:
            st.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                step=step,
                key=key,
                label_visibility="collapsed",
            )
        with col_plus:
            with st.container(key=f"btn-inc-{key}"):
                st.button(
                    "+",
                    key=f"{key}_inc",
                    use_container_width=True,
                    on_click=_step_value,
                    args=(key, step, min_value, max_value),
                )
        st.markdown(
            f'<div class="slider-minmax"><span>{min_value:.0f}%</span><span>{max_value:.0f}%</span></div>',
            unsafe_allow_html=True,
        )


def render_footer(repo_url: str) -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            Built with Streamlit · scikit-learn · SHAP &nbsp;·&nbsp; v1.0
            &nbsp;·&nbsp; <a href="{repo_url}" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
