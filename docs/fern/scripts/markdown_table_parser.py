# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict Markdown table-row parsing for generated API references."""

from __future__ import annotations


def split_markdown_table_row(row: str, expected: int) -> list[str]:
    """Split a row while preserving escaped and inline-code pipe characters."""
    stripped = row.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError(f"malformed table row: {row!r}")
    cells = _split_table_cells(stripped[1:-1])
    if len(cells) != expected:
        raise ValueError(f"expected {expected} cells, got {len(cells)}: {row!r}")
    return cells


def _split_table_cells(body: str) -> list[str]:
    """Split unescaped pipes outside inline-code spans."""
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    i = 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            current.append("|")
            i += 2
            continue
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code:
            cells.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current))
    return cells
