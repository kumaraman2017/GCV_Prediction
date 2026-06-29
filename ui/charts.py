import plotly.graph_objects as go

from ui.theme import FONT_BODY, FONT_DISPLAY, Theme


def _transparent_layout(fig: go.Figure, theme: Theme, height: int) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=theme.text_secondary, size=13),
        margin=dict(t=20, b=20, l=20, r=20),
        height=height,
    )
    return fig


def confidence_interval_chart(low: float, high: float, prediction: float, unit: str, theme: Theme) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[0, 0],
            mode="lines",
            line={"color": theme.accent_blue, "width": 10},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[prediction],
            y=[0],
            mode="markers",
            marker={"size": 18, "color": theme.accent_cyan, "line": {"width": 2, "color": "white"}},
            hoverinfo="skip",
        )
    )
    for x_value in (low, high):
        fig.add_annotation(
            x=x_value, y=0, yshift=26, showarrow=False,
            text=f"{x_value:,.0f}", font={"size": 12, "color": theme.text_secondary},
        )
    fig.add_annotation(
        x=prediction, y=0, yshift=-26, showarrow=False,
        text=f"{prediction:,.0f} {unit}",
        font={"size": 13, "color": theme.text_primary, "family": FONT_DISPLAY},
    )
    fig.update_yaxes(visible=False, range=[-1, 1], fixedrange=True)
    fig.update_xaxes(visible=False, fixedrange=True, range=[low - (high - low) * 0.15, high + (high - low) * 0.15])
    fig.update_layout(showlegend=False)
    return _transparent_layout(fig, theme, height=140)


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
