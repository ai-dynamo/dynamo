# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal data types for the merge algorithms.

Three concerns live here:

- ``PluginResult``: a single plugin's stage output, paired with its
  registered priority and ``final`` flag. Consumed by ``type_aware_merge``.
- ``ComponentKey`` / ``MergeOutcome`` / ``ChainAugmentOutcome``: structured
  return values for the two merge algorithms. The orchestrator reads
  ``short_circuited`` / ``used_final_from`` / ``set_dropped`` /
  ``chain_break_warnings`` to emit audit events and Prometheus metrics.
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

    Single-pool by construction in this PR: one bucket per
    ``sub_component_type``.  Kept as a dataclass (rather than collapsing
    to a bare ``str``) so the hierarchical-planner PR can re-add the
    per-pool key axis without touching every call site.  ``frozen=True``
    makes instances hashable for use as ``dict`` / ``set`` keys.
    """

    sub_component_type: str


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------


@dataclass
class MergeOutcome:
    """Structured result of ``type_aware_merge``.

    Consumed by the orchestrator (acted on):

    - ``short_circuited=True`` → caller skips downstream stages + EXECUTE
    - ``clamped`` non-empty → emit clamp counters
      (``reconcile_clamped_total`` on RECONCILE,
      ``constrain_capped_total`` on CONSTRAIN). The tuple records the
      per-key reason (``"floor"`` when AT_LEAST raised the value,
      ``"ceiling"`` when AT_MOST lowered it) and the plugin_id that
      contributed the winning bound.

    Surfaced on ``PipelineOutcome.constrain_outcome`` for downstream
    inspection but NOT emitted as Prometheus / audit signals in PR #1:

    - ``used_final_from`` (which plugin's ``final=True`` won the stage)
    - ``set_dropped`` (component keys whose SET entries were rejected
      by ``set_allowed=False``)

    Counters / audit emit for these two are deferred to a follow-up
    observability PR.

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
    - ``chain_break_warnings``: informational events — one message per
      plugin that returned ``final=True`` while **not** being the
      lowest-priority (numerically smallest) plugin in the chain. The
      chain still breaks at that plugin, so larger-priority-number
      plugins after the final-setter are skipped (they lose the chance
      to populate fields earlier plugins left as ``None``). This may
      be intentional — e.g. a policy plugin saying "skip the expensive
      fallback for this scenario" — or a configuration mistake;
      ``chain_augment`` cannot tell which. The orchestrator surfaces
      these messages via ``PipelineOutcome.audit_events`` so operators
      can audit them; a Prometheus counter for this signal is deferred
      to a follow-up observability PR.
    """

    prediction: Optional[PredictionData]
    final_from: str = ""
    degraded: list[str] = field(default_factory=list)
    chain_break_warnings: list[str] = field(default_factory=list)


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

    ``plugin_id`` and ``priority`` are read-only (declared as ``@property``)
    so adapter implementations like ``_PredictAdapter`` that forward
    these from a wrapped ``RegisteredPlugin`` via ``@property`` satisfy
    the structural check.
    """

    @property
    def plugin_id(self) -> str:
        """Stable plugin identifier as registered with the registry."""

    @property
    def priority(self) -> int:
        """Chain-augment ordering: lower priority runs first."""

    async def call(self, method: str, context: PipelineContext) -> PredictStageResponse:
        """Dispatch a PREDICT call; transport layer handles serialisation."""


__all__ = [
    "PluginResult",
    "ComponentKey",
    "MergeOutcome",
    "ChainAugmentOutcome",
    "PredictPluginCallable",
]
