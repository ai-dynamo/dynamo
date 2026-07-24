# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared safe rendering helpers for generated MDX Markdown."""

from __future__ import annotations


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
