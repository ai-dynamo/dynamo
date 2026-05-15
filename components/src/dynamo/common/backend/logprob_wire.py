# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo logprob wire-shape builder.

Per-engine adapters extract their native logprob format into the
normalized intermediates below; this module emits the Dynamo wire
dicts (matching the Rust ``TopLogprob`` shape).

This keeps the per-engine code focused on engine-specific parsing
and removes the triplicate dict-construction loops.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class TopLogprob(NamedTuple):
    """One top-k entry at a single output position."""

    rank: int
    token_id: int
    token: str | None
    logprob: float
    # Optional UTF-8 byte representation of `token`; vLLM populates this,
    # others leave it None. Use ``None`` to omit the ``bytes`` key.
    bytes_: list[int] | None = None


def build_chunk(
    selected: list[float],
    top_per_position: list[list[TopLogprob]] | None,
) -> tuple[list[float] | None, list[list[dict[str, Any]]] | None]:
    """Render the Dynamo wire shape from per-position primitives.

    Returns ``(log_probs, top_logprobs)`` ready for assignment onto a
    ``GenerateChunk``. Each top entry serializes to
    ``{rank, token_id, token, logprob[, bytes]}``.
    """
    log_probs = selected or None
    if not top_per_position:
        return log_probs, None
    top: list[list[dict[str, Any]]] = []
    for position in top_per_position:
        rendered: list[dict[str, Any]] = []
        for entry in position:
            d: dict[str, Any] = {
                "rank": entry.rank,
                "token_id": entry.token_id,
                "token": entry.token,
                "logprob": entry.logprob,
            }
            if entry.bytes_ is not None:
                d["bytes"] = entry.bytes_
            rendered.append(d)
        top.append(rendered)
    return log_probs, top
