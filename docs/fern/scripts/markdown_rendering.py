# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared safe rendering helpers for generated MDX Markdown."""

from __future__ import annotations

import re

# Capturing group so re.split keeps code spans as odd-indexed segments. The
# backtick runs are quantified because reST-style docstrings delimit literals
# with a double backtick, and matching only single ones would treat the empty
# span between them as the code and escape the literal text itself.
_CODE_SPAN_RE = re.compile(r"(`+[^`]*`+)")

# Dynamo's Python docstrings mix Google style with Sphinx cross-reference
# roles (``:class:`Foo```). Markdown has no such construct, so the role prefix
# would reach the reader as literal text in front of an otherwise correct code
# span. Dropping the prefix keeps the referenced name as inline code, which is
# how the same names already read everywhere else on the page. Only roles that
# actually appear in the curated modules are listed; an unknown ``:word:`` is
# left alone rather than guessed at.
_REST_ROLE_RE = re.compile(r":(?:class|meth|func|attr|mod|data|obj|exc):(?=`)")


def strip_rest_roles(text: str) -> str:
    """Drop Sphinx cross-reference role prefixes, keeping the code span."""
    return _REST_ROLE_RE.sub("", text)


def escape_mdx_prose(text: str) -> str:
    """Escape JSX-significant characters in generated Markdown prose.

    Source comments and docstrings carry ``<`` and ``{`` (generics, template
    placeholders) that MDX would otherwise parse as JSX. Inline code spans are
    left alone: MDX does not parse JSX inside them, and HTML entities are not
    decoded there, so escaping would surface a literal ``&lt;`` to the reader.
    Both single-backtick Markdown spans and double-backtick reST literals
    count as code.
    """
    parts = _CODE_SPAN_RE.split(strip_rest_roles(text))
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
