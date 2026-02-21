"""Dynamo Dark theme for Plotly charts -- NVIDIA Brand.

Loads design tokens from design_tokens.yaml and produces a reusable
plotly.graph_objects.layout.Template aligned with NVIDIA visual identity.

Typography:
    - Titles, headings, body: NVIDIA Sans (Arial fallback)
    - Tick labels, code-like text: Roboto Mono (monospace fallback)

Usage:
    from docs.diagrams.theme.plotly_dynamo import dynamo_template, load_tokens

    fig = go.Figure(data=[...], layout=go.Layout(template=dynamo_template))
    fig.write_image("chart.svg")
    fig.write_image("chart.png", scale=2)

Or apply globally:
    import plotly.io as pio
    pio.templates.default = dynamo_template
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import yaml


def load_tokens(path: str | Path | None = None) -> dict[str, Any]:
    """Load design tokens from YAML file.

    Args:
        path: Path to design_tokens.yaml. Defaults to the file
              next to this module.

    Returns:
        Parsed token dictionary.
    """
    if path is None:
        path = Path(__file__).parent / "design_tokens.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_template(tokens: dict[str, Any] | None = None) -> go.layout.Template:
    """Build a Plotly Template from design tokens.

    Args:
        tokens: Parsed design_tokens dict. Loaded from default path if None.

    Returns:
        A fully configured plotly.graph_objects.layout.Template.
    """
    if tokens is None:
        tokens = load_tokens()

    colors = tokens["colors"]
    typo = tokens["typography"]

    bg_primary = colors["background"]["primary"]
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    border_subtle = colors["border"]["subtle"]
    series = colors["chart_series"]

    # Dual font families per NVIDIA Brand Guide
    font_sans = typo["font_family"]  # NVIDIA Sans for titles/body
    font_mono = typo["font_family_mono"]  # Roboto Mono for ticks/code

    layout = go.Layout(
        # -- Background --
        paper_bgcolor=bg_primary,
        plot_bgcolor=bg_primary,
        # -- Typography (base: NVIDIA Sans) --
        font=dict(
            family=font_sans,
            color=text_primary,
            size=typo["label"]["size"],
        ),
        title=dict(
            font=dict(
                family=font_sans,
                size=typo["title"]["size"],
                color=text_primary,
            ),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        # -- Color palette --
        colorway=series,
        # -- X axis (monospace tick labels) --
        xaxis=dict(
            gridcolor=border_subtle,
            gridwidth=0.5,
            zeroline=False,
            linecolor=border_subtle,
            linewidth=0.5,
            tickfont=dict(
                family=font_mono,
                size=typo["annotation"]["size"],
                color=text_secondary,
            ),
            title=dict(
                font=dict(
                    family=font_sans,
                    size=typo["label"]["size"],
                    color=text_secondary,
                ),
            ),
        ),
        # -- Y axis (monospace tick labels for alignment) --
        yaxis=dict(
            gridcolor=border_subtle,
            gridwidth=0.5,
            zeroline=False,
            linecolor=border_subtle,
            linewidth=0.5,
            tickfont=dict(
                family=font_mono,
                size=typo["label"]["size"],
                color=text_primary,
            ),
            title=dict(
                font=dict(
                    family=font_sans,
                    size=typo["label"]["size"],
                    color=text_secondary,
                ),
            ),
            automargin=True,
        ),
        # -- Legend --
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                family=font_sans,
                size=typo["annotation"]["size"],
                color=text_secondary,
            ),
            bordercolor=border_subtle,
            borderwidth=0,
        ),
        # -- Margins --
        margin=dict(l=20, r=20, t=60, b=20),
        # -- Subtle outer frame (muted green, thin) --
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=-0.01,
                y0=-0.02,
                x1=1.01,
                y1=1.04,
                line=dict(
                    color=border_subtle,
                    width=0.5,
                ),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            ),
        ],
    )

    # -- Data defaults --
    template = go.layout.Template(layout=layout)

    # Bar chart defaults
    template.data.bar = [
        go.Bar(
            marker=dict(
                line=dict(width=0),
                opacity=0.92,
            ),
            textfont=dict(
                family=font_mono,
                size=typo["annotation"]["size"],
                color=text_primary,
            ),
            textposition="inside",
            insidetextanchor="start",
            error_x=dict(
                color=text_primary,
                thickness=1.5,
                width=4,
            ),
            error_y=dict(
                color=text_primary,
                thickness=1.5,
                width=4,
            ),
        ),
    ]

    # Scatter / line chart defaults
    template.data.scatter = [
        go.Scatter(
            marker=dict(size=8, line=dict(width=1, color=bg_primary)),
            line=dict(width=2),
        ),
    ]

    return template


# Module-level singleton -- import this directly
dynamo_template = build_template()


def apply_outer_frame(fig: go.Figure) -> go.Figure:
    """Add the NVIDIA Green outer frame to an existing figure.

    Useful when creating a figure without the template, or when
    the frame was lost during updates.
    """
    tokens = load_tokens()
    border_frame = tokens["colors"]["border"]["frame"]
    frame_width = tokens["borders"]["frame_width"]

    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=-0.01,
        y0=-0.02,
        x1=1.01,
        y1=1.04,
        line=dict(color=border_frame, width=frame_width),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    return fig


def section_annotation(
    fig: go.Figure,
    text: str,
    y: float,
    x: float = 0.0,
) -> go.Figure:
    """Add a section header annotation (e.g., 'DISPATCH (us)').

    Uses NVIDIA Sans for section headers.

    Args:
        fig: The Plotly figure.
        text: Section header text (pre-formatted; not auto-uppercased).
        y: Y position in paper coordinates (0-1).
        x: X position in paper coordinates (0-1).
    """
    tokens = load_tokens()
    typo = tokens["typography"]
    text_color = tokens["colors"]["text"]["secondary"]

    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        showarrow=False,
        font=dict(
            family=typo["font_family"],
            size=typo["heading"]["size"],
            color=text_color,
        ),
        xanchor="left",
        yanchor="bottom",
    )
    return fig
