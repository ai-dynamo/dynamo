# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal data types for the merge algorithms.

Three concerns live here:

- ``PluginResult``: a single plugin's stage output, paired with its
  registered priority and ``final`` flag. Consumed by ``type_aware_merge``.
- ``ComponentKey`` / ``MergeOutcome`` / ``ChainAugmentOutcome``: structured
  return values for the two merge algorithms. The orchestrator reads
  ``short_circuited`` / ``used_final_from`` / ``set_dropped`` /
  ``misuse_warnings`` to emit audit events and Prometheus metrics.
- ``PredictPluginCallable``: structural protocol for objects the
  orchestrator hands to ``chain_augment`` — a transport-backed plugin
  handle exposing ``plugin_id``, ``priority``, and an async
  ``call("Predict", context)``.

These are **pure data containers** — no behaviour, no I/O. Algorithms
live alongside in ``type_aware.py`` and ``chain_augment.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Union, runtime_checkable

from dynamo.planner.plugins.types import (
    AcceptResult,
    OverrideResult,
    PipelineContext,
    PredictionData,
    PredictStageResponse,
    RejectResult,
    ScalingProposal,
)

# ----------------------------------------------------------------------------
# Input to type_aware_merge
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginResult:
    """A single plugin's output for one stage, paired with its priority.

    Orchestrator constructs a list of these after awaiting all plugins in
    a PROPOSE / RECONCILE / CONSTRAIN stage, then hands the list to
    ``type_aware_merge``. ``final`` mirrors the on-wire flag from the
    stage response (``ProposeStageResponse.final`` /
    ``ReconcileStageResponse.final``; silently ignored for CONSTRAIN).
    """

    plugin_id: str
    priority: int
    result: Union[AcceptResult, OverrideResult, RejectResult]
    final: bool = False


# ----------------------------------------------------------------------------
# Bucket key for type-aware merge
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentKey:
    """Group key used to bucket per-plugin ``ComponentTarget`` entries in
    ``type_aware_merge``.

    Two targets belong in the same bucket iff they name the same
    ``(sub_component_type, component_name)`` pair; ``component_name=None``
    denotes the default (single-pool) instance of its type. ``frozen=True``
    makes instances hashable for use as ``dict`` / ``set`` keys.
    """

    sub_component_type: str
    component_name: Optional[str] = None


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------


@dataclass
class MergeOutcome:
    """Structured result of ``type_aware_merge``.

    Consumed by the orchestrator:

    - ``short_circuited=True`` → skip downstream stages + EXECUTE
    - ``used_final_from`` non-empty → emit audit "final override applied"
    - ``set_dropped`` non-empty → emit Prometheus counter
      ``plugin_constrain_set_dropped_total{plugin_id}`` (CONSTRAIN only)
    - ``clamped`` non-empty → emit clamp counters
      (``reconcile_clamped_total`` on RECONCILE,
      ``constrain_capped_total`` on CONSTRAIN). The tuple records the
      per-key reason (``"floor"`` when AT_LEAST raised the value,
      ``"ceiling"`` when AT_MOST lowered it) and the plugin_id that
      contributed the winning bound.

    Mutable on purpose: fields are populated step-by-step in ``type_aware_merge``
    and ``set_dropped`` / ``clamped`` are appended to as buckets are processed.
    """

    proposal: Optional[ScalingProposal]
    short_circuited: bool
    short_circuit_reason: str = ""
    used_final_from: str = ""
    set_dropped: list[ComponentKey] = field(default_factory=list)
    clamped: list[tuple[ComponentKey, str, str]] = field(default_factory=list)
    """(key, direction, source_plugin_id) — direction ∈ {"floor", "ceiling"}."""


@dataclass
class ChainAugmentOutcome:
    """Structured result of ``chain_augment`` (PREDICT stage).

    - ``prediction``: partial-merged ``PredictionData`` produced by the
      chain, or ``None`` when every plugin returned ``AcceptResult`` /
      ``RejectResult`` (no prediction content).
    - ``final_from``: plugin_id of the plugin whose ``final=True`` broke
      the chain (empty if the chain ran to completion).
    - ``degraded``: plugin_ids that returned ``RejectResult`` (the chain
      continues past a REJECT in PREDICT; contrast with type-aware merge
      where REJECT short-circuits).
    - ``misuse_warnings``: runtime detection — one WARNING message per
      plugin that returned ``final=True`` while **not** being the
      lowest-priority (numerically smallest) plugin in the chain. That
      combination risks breaking the chain before a higher-priority
      plugin ever ran. The orchestrator emits Prometheus counter
      ``predict_chain_final_at_non_lowest_priority_total{plugin_id}``.
    """

    prediction: Optional[PredictionData]
    final_from: str = ""
    degraded: list[str] = field(default_factory=list)
    misuse_warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Structural protocol for chain_augment's plugin_chain input
# ----------------------------------------------------------------------------


@runtime_checkable
class PredictPluginCallable(Protocol):
    """Structural type: what ``chain_augment`` expects per plugin handle.

    The orchestrator wraps each registered PREDICT plugin in an object
    that satisfies this protocol — exposing the registry-visible
    ``plugin_id`` / ``priority`` attributes alongside a transport-backed
    ``call`` coroutine. Using a ``Protocol`` here keeps ``merge`` decoupled
    from the concrete registry / transport types.
    """

    plugin_id: str
    priority: int

    async def call(
        self, method: str, context: PipelineContext
    ) -> PredictStageResponse: ...


__all__ = [
    "PluginResult",
    "ComponentKey",
    "MergeOutcome",
    "ChainAugmentOutcome",
    "PredictPluginCallable",
]
