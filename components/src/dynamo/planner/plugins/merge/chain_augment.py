# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-augment merge for PREDICT stage.

Sequential layered prediction: each plugin sees the running ``prediction``
on its ``PipelineContext`` and may override, augment, or pass through.
Partial-merge is field-level on ``Optional[float]`` prediction fields
(``None`` preserves the previous value; a concrete float — including
``0.0`` — overrides).

Ordering and semantics
----------------------
- Chain is sorted by ``priority`` **descending** (largest priority number
  first). Lowest priority number (highest precedence) therefore runs
  **last** — its partial-merge overrides earlier plugins' fields on
  conflict.
- ``predictions=None`` in a response ≈ ACCEPT (no opinion; chain
  continues, running prediction unchanged). The wire contract in
  ``proto/v1/plugin.proto`` does **not** expose a REJECT mechanism for
  PREDICT, so ``ChainAugmentOutcome.degraded`` is not populated;
  a future proto revision may add an explicit reject field, at which
  point this function would populate ``degraded`` similarly to how
  ``type_aware_merge`` tracks ``set_dropped``.
- ``final=True`` breaks the chain immediately. Subsequent plugins are
  **not** called — ``PredictPluginCallable.call`` is never awaited for
  them.
- **Strong contract**: ``final=True`` MUST come from the
  lowest-priority-number (highest-precedence) plugin. Otherwise the chain
  breaks BEFORE higher-precedence plugins ever run. ``chain_augment``
  detects this at runtime, logs a WARNING, and records a message in
  ``ChainAugmentOutcome.misuse_warnings``. The orchestrator emits
  Prometheus ``predict_chain_final_at_non_lowest_priority_total{plugin_id}``.

This function is async only because it awaits plugin RPCs; the algorithmic
logic is synchronous + deterministic given plugin responses.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from dynamo.planner.plugins.types import (
    PipelineContext,
    PredictionData,
)

from dynamo.planner.plugins.merge.types import (
    ChainAugmentOutcome,
    PredictPluginCallable,
)

log = logging.getLogger(__name__)

_PREDICTION_FIELDS = ("predicted_num_req", "predicted_isl", "predicted_osl")


def _partial_merge(
    prev: Optional[PredictionData], new: PredictionData
) -> PredictionData:
    """Field-level merge: a concrete value on ``new`` overrides the same
    field on ``prev``; ``None`` on ``new`` preserves ``prev``'s value.

    ``source`` takes ``new.source`` when non-empty, else ``prev.source``.
    If ``prev is None``, ``new`` is returned verbatim.
    """
    if prev is None:
        return new
    merged: dict[str, object] = {}
    for name in _PREDICTION_FIELDS:
        nv = getattr(new, name)
        merged[name] = nv if nv is not None else getattr(prev, name)
    merged["source"] = new.source or prev.source
    return PredictionData(**merged)


async def chain_augment(
    plugin_chain: Sequence[PredictPluginCallable],
    initial_context: PipelineContext,
) -> ChainAugmentOutcome:
    """Run a PREDICT chain, returning the partial-merged prediction.

    Args:
        plugin_chain: PREDICT plugins to run. Sorted by priority descending
            internally; caller may pass any order. Empty → empty outcome.
        initial_context: Base PipelineContext shared across plugins. The
            ``predictions`` field is replaced per-iteration with the
            running merged prediction; other fields are preserved.

    Returns:
        ``ChainAugmentOutcome`` — ``prediction`` is the partial-merged
        ``PredictionData`` (``None`` if no plugin produced content);
        ``final_from`` is the plugin that broke the chain (empty on full
        traversal); ``misuse_warnings`` is non-empty when a non-lowest-
        priority plugin returned ``final=True``.
    """
    chain = sorted(plugin_chain, key=lambda p: -p.priority)
    lowest_priority = min((p.priority for p in chain), default=None)
    prediction: Optional[PredictionData] = None
    final_from = ""
    misuse_warnings: list[str] = []

    for p in chain:
        ctx = initial_context.model_copy(update={"predictions": prediction})
        resp = await p.call("Predict", ctx)
        if resp.predictions is not None:
            prediction = _partial_merge(prediction, resp.predictions)
        if resp.final:
            final_from = p.plugin_id
            if lowest_priority is not None and p.priority != lowest_priority:
                msg = (
                    f"chain_augment_final_misuse: plugin_id={p.plugin_id} "
                    f"priority={p.priority} returned final=true but is NOT "
                    f"the lowest priority in the chain "
                    f"(lowest_priority={lowest_priority}). "
                    "Chain broke BEFORE the higher-precedence plugin could "
                    "run. See merge/README.md 'chain-augment final 使用规范'."
                )
                log.warning(msg)
                misuse_warnings.append(msg)
            break

    return ChainAugmentOutcome(
        prediction=prediction,
        final_from=final_from,
        degraded=[],
        misuse_warnings=misuse_warnings,
    )


__all__ = ["chain_augment"]
