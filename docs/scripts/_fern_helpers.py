# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for API doc generators and fernify transforms.

Generators produce GitHub-friendly Markdown (committed to main).
Fernify scripts transform that Markdown into Fern MDX in CI.

The three fernify_* functions follow the same pattern as
convert_admonitions() in fern/convert_callouts.py: pure text -> text
transforms that can be composed in a pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def escape_jsx_attr(text: str) -> str:
    """Escape text for use inside a double-quoted HTML/JSX attribute."""
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_CODE_SPAN_RE = re.compile(r"`[^`]+`")
_BRACE_EXPR_RE = re.compile(r"\{[^{}]*\}")
_PLACEHOLDER_RE = re.compile(r"\x01(\d+)\x01")


def escape_mdx_prose(text: str) -> str:
    """Escape MDX-sensitive characters in prose text.

    Protects backtick code spans, wraps brace patterns in inline code,
    and escapes remaining braces and angle brackets.
    """
    spans: list[str] = []

    def _protect(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x01{len(spans) - 1}\x01"

    text = _CODE_SPAN_RE.sub(_protect, text)
    text = _BRACE_EXPR_RE.sub(lambda m: f"`{m.group(0)}`", text)
    text = _CODE_SPAN_RE.sub(_protect, text)
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("<", "&lt;")
    text = _PLACEHOLDER_RE.sub(lambda m: spans[int(m.group(1))], text)
    return text


def slugify(text: str) -> str:
    """Convert text to a URL-safe anchor slug."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


def render_card_group(cards: list[dict[str, str]], cols: int) -> str:
    """Render a Fern <CardGroup> with <Card> elements.

    Each card dict must have 'title', 'icon', 'description'.
    Optional 'href' adds a link. Values are escaped for JSX attributes.
    Blank lines between cards ensure GitHub renders separate paragraphs.
    """
    parts = [f"<CardGroup cols={{{cols}}}>\n"]
    for i, card in enumerate(cards):
        title = escape_jsx_attr(card["title"])
        icon = escape_jsx_attr(card["icon"])
        href = f' href="{escape_jsx_attr(card["href"])}"' if card.get("href") else ""
        if i > 0:
            parts.append("\n")
        parts.append(
            f'  <Card title="{title}" icon="{icon}"{href}>\n'
            f'    {card["description"]}\n'
            f"  </Card>\n"
        )
    parts.append("</CardGroup>\n")
    return "".join(parts)


def render_markdown_table(cards: list[dict[str, str]]) -> str:
    """Render card data as a standard Markdown table (GitHub-friendly)."""
    rows = ["| Name | Description |", "| --- | --- |"]
    for card in cards:
        href = card.get("href", "")
        title = card["title"]
        link = f"[`{title}`]({href})" if href else f"`{title}`"
        rows.append(f"| {link} | {card['description']} |")
    return "\n".join(rows) + "\n"


def details_title(name: str, description: str) -> str:
    """Build a title for a <details> summary or <Accordion>.

    Returns unescaped text — the renderer (render_details / render_accordion)
    is responsible for escaping.
    """
    if description:
        return f"`{name}` \u2014 {description}"
    return f"`{name}`"


def render_details(summary: str, body: str) -> str:
    """Render a <details>/<summary> block (GitHub + Fern compatible).

    Blank line after <summary> is required for GitHub to render markdown.
    """
    escaped = escape_jsx_attr(summary)
    return (
        "<details>\n"
        f"<summary><strong>{escaped}</strong></summary>\n"
        f"\n{body}\n\n"
        "</details>\n"
    )


def render_accordion(title: str, body: str) -> str:
    """Render a Fern <Accordion> component.

    Use this for Fern-only output (not committed to main).
    """
    escaped_title = escape_jsx_attr(title)
    return f'<Accordion title="{escaped_title}">\n\n{body}\n\n</Accordion>\n'


# ---------------------------------------------------------------------------
# Composable fernify transforms (text -> text)
# ---------------------------------------------------------------------------

_DETAILS_RE = re.compile(
    r"<details>\s*\n"
    r"<summary><strong>(.*?)</strong></summary>\s*\n"
    r"(.*?)"
    r"</details>\s*\n",
    re.DOTALL,
)


def fernify_details_to_accordion(text: str) -> str:
    """Convert all <details>/<summary> blocks to Fern <Accordion> components."""

    def _replace(m: re.Match) -> str:
        title = m.group(1)
        body = m.group(2).strip()
        return render_accordion(title, body)

    return _DETAILS_RE.sub(_replace, text)


_TABLE_ROW_RE = re.compile(r"^\| \[`([^`]+)`\]\(([^)]+)\) \| (.+?) \|$", re.MULTILINE)
_TABLE_BLOCK_RE = re.compile(
    r"^\| Name \| Description \|\n\| --- \| --- \|\n((?:\| .+ \| .+ \|\n)+)",
    re.MULTILINE,
)


def fernify_table_to_cards(
    text: str,
    icon_map: dict[str, str],
    cols: int = 3,
) -> str:
    """Convert Markdown tables (Name | Description) to Fern <CardGroup> components.

    Matches tables produced by render_markdown_table(). Each row must have
    a [`title`](href) link in the Name column. icon_map provides the Font
    Awesome icon for each title; unknown titles get "regular box".
    """

    def _replace_table(m: re.Match) -> str:
        rows_text = m.group(1)
        cards: list[dict[str, str]] = []
        for row in _TABLE_ROW_RE.finditer(rows_text):
            title = row.group(1)
            href = row.group(2)
            desc = row.group(3).strip()
            cards.append(
                {
                    "title": title,
                    "icon": icon_map.get(title, "regular box"),
                    "href": href,
                    "description": desc,
                }
            )
        if not cards:
            return m.group(0)
        return render_card_group(cards, min(len(cards), cols))

    return _TABLE_BLOCK_RE.sub(_replace_table, text)


def fernify_headings_to_accordion(
    text: str,
    heading_level: int = 4,
) -> str:
    """Convert Markdown headings at a given level to Fern <Accordion> components.

    Captures everything from a heading (e.g. ####) up to the next heading
    at the same or higher level (or EOF) and wraps it in an <Accordion>.
    """
    prefix = "#" * heading_level
    pattern = re.compile(
        rf"^{prefix} (.+?)$\n(.*?)(?=^#{{1,{heading_level}}} |\Z)",
        re.MULTILINE | re.DOTALL,
    )

    def _replace(m: re.Match) -> str:
        title = m.group(1).strip()
        body = m.group(2).strip()
        return render_accordion(title, body) + "\n"

    return pattern.sub(_replace, text)
