# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared safe rendering helpers for generated MDX Markdown."""

from __future__ import annotations

import re

# Capturing group so re.split keeps code spans as odd-indexed segments.
_CODE_SPAN_RE = re.compile(r"(`[^`]*`)")


def escape_mdx_prose(text: str) -> str:
    """Escape JSX-significant characters in generated Markdown prose.

    Source comments and docstrings carry ``<`` and ``{`` (generics, template
    placeholders) that MDX would otherwise parse as JSX. Inline code spans are
    left alone: MDX does not parse JSX inside them, and HTML entities are not
    decoded there, so escaping would surface a literal ``&lt;`` to the reader.
    """
    parts = _CODE_SPAN_RE.split(text)
    return "".join(
        part if index % 2 else _escape_jsx(part) for index, part in enumerate(parts)
    ).strip()


def _escape_jsx(text: str) -> str:
    """Escape ``&``, angle brackets, and braces outside inline code."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )


def mdx_attribute(value: str) -> str:
    """Escape a value for use inside a double-quoted MDX attribute."""
    return " ".join(value.split()).replace("&", "&amp;").replace('"', "&quot;")


def escape_mdx_table_cell(text: str, *, empty: str = "-") -> str:
    """Escape source-derived text for an MDX Markdown table cell."""
    if not text:
        return empty
    normalized = text.replace("<br />", " ")
    return (
        normalized.replace("&", "&amp;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", " ")
    )
