import plotly.graph_objects as go

from ui.theme import FONT_BODY, Theme, confidence_color, hex_to_rgba


def _transparent_layout(fig: go.Figure, theme: Theme, height: int) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=theme.text_secondary, size=13),
        margin=dict(t=20, b=20, l=20, r=20),
        height=height,
    )
    return fig


def confidence_gauge(confidence: float, theme: Theme) -> go.Figure:
    color = confidence_color(theme, confidence)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 34, "color": theme.text_primary}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme.text_muted, "tickfont": {"size": 10}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": hex_to_rgba(theme.error, 0.12)},
                    {"range": [40, 70], "color": hex_to_rgba(theme.warning, 0.12)},
                    {"range": [70, 100], "color": hex_to_rgba(theme.success, 0.12)},
                ],
            },
        )
    )
    return _transparent_layout(fig, theme, height=200)


def shap_waterfall(contributions: list[dict], base_value: float, prediction: float, unit: str, theme: Theme) -> go.Figure:
    ordered = sorted(contributions, key=lambda c: abs(c["shap_value"]), reverse=True)
    labels = [c["feature"] for c in ordered]
    values = [c["shap_value"] for c in ordered]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=labels + ["Prediction"],
            y=values + [0],
            base=base_value,
            increasing={"marker": {"color": theme.success}},
            decreasing={"marker": {"color": theme.error}},
            totals={"marker": {"color": theme.accent_cyan}},
            connector={"line": {"color": theme.surface_border, "width": 1}},
            text=[f"{v:+,.0f}" for v in values] + [f"{prediction:,.0f}"],
            textposition="outside",
            textfont={"color": theme.text_secondary, "size": 11},
        )
    )
    fig.update_xaxes(showgrid=False, color=theme.text_muted)
    fig.update_yaxes(showgrid=True, gridcolor=theme.surface_border, color=theme.text_muted, title=unit)
    return _transparent_layout(fig, theme, height=320)
