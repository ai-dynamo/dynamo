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

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Latch the length-mismatch warning per shape so a sustained upstream
# desync (e.g. an engine version regression that drifts top/selected by
# a fixed amount) doesn't flood logs at warn level on every chunk.
_LENGTH_MISMATCH_LOGGED: set[tuple[int, int]] = set()


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

    When ``selected`` is empty the wire shape collapses to ``(None, None)``
    regardless of ``top_per_position`` — emitting ``top_logprobs`` without
    a matching ``log_probs`` violates the OpenAI contract. Callers that
    have only the chosen-token stream (no top-k) pass ``top_per_position=None``.
    """
    if not selected:
        return None, None
    if not top_per_position:
        return list(selected), None
    if len(top_per_position) != len(selected):
        # Defensive: misaligned per-position arrays from an upstream
        # extractor would surface garbage pairings downstream. Drop top-k
        # rather than silently mis-correlate, but emit a one-shot
        # breadcrumb per shape so operators can distinguish extractor
        # regression from missing-top-k without log flooding.
        shape = (len(top_per_position), len(selected))
        if shape not in _LENGTH_MISMATCH_LOGGED:
            _LENGTH_MISMATCH_LOGGED.add(shape)
            logger.warning(
                "logprob top_per_position length %d != selected length %d; "
                "dropping top_logprobs (further occurrences with this shape suppressed)",
                shape[0],
                shape[1],
            )
        return list(selected), None
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
    return list(selected), top
