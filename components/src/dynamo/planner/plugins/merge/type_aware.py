# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type-aware merge.

Pure function used by the orchestrator in PROPOSE / RECONCILE /
CONSTRAIN stages.

Algorithm outline
-----------------
1. **REJECT short-circuit** — any ``RejectResult`` returns
   ``short_circuited=True``; out-ranks ``final``.
2. **final priority** — if any ``OverrideResult`` carries ``final=True``,
   the priority-smallest final's targets become the proposal outright.
3. **Bucket by ``(sub_component_type, component_name)``**; inside each
   bucket:
   - ``floor = max(AT_LEAST replicas)`` (defaults to ``0``)
   - ``ceiling = min(AT_MOST replicas)`` (defaults to ``+inf``)
   - ``recommendation = priority-smallest SET replicas`` else baseline
   - ``result = max(floor, min(ceiling, recommendation))`` — clamp order
     ensures ``floor`` wins when ``floor > ceiling``
4. **Cast to int** and build ``ScalingProposal``.

``set_allowed=False`` (CONSTRAIN mode) drops SET targets from both the
final path and the bucket-merge path and records dropped keys in
``MergeOutcome.set_dropped`` for orchestrator audit (emits the
``plugin_constrain_set_dropped_total{plugin_id}`` Prometheus counter).

The function is **sync** and **deterministic** — no I/O, no Clock. Output
target order preserves insertion order of ``plugin_results`` first, then
any baseline-only keys.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from dynamo.planner.plugins.types import (
    ComponentTarget,
    OverrideResult,
    OverrideType,
    RejectResult,
    ScalingProposal,
)

from dynamo.planner.plugins.merge.types import (
    ComponentKey,
    MergeOutcome,
    PluginResult,
)


def type_aware_merge(
    plugin_results: Sequence[PluginResult],
    baseline: Mapping[ComponentKey, int],
    set_allowed: bool = True,
) -> MergeOutcome:
    """Merge per-plugin OverrideResults into a single ScalingProposal.

    Args:
        plugin_results: Per-plugin stage outputs. ``AcceptResult`` entries
            are silently ignored; ``RejectResult`` short-circuits;
            ``OverrideResult`` entries are merged. ``priority``
            disambiguates conflicting SETs (smallest wins) and picks the
            final winner when multiple OverrideResults carry ``final=True``.
        baseline: Current / upstream replicas per ``ComponentKey``. Used as
            the recommendation when no plugin emits a SET for the key;
            keys present only in ``baseline`` still appear in the output
            (passthrough) so downstream stages see a complete proposal.
        set_allowed: ``True`` (PROPOSE / RECONCILE default) keeps SET
            targets. ``False`` (CONSTRAIN) drops them and records dropped
            keys in ``MergeOutcome.set_dropped``.

    Returns:
        ``MergeOutcome`` — either ``short_circuited=True`` with
        ``proposal=None`` on REJECT, or a populated ``ScalingProposal``.
    """
    # Step 1: REJECT short-circuit (REJECT > final)
    for r in plugin_results:
        if isinstance(r.result, RejectResult):
            return MergeOutcome(
                proposal=None,
                short_circuited=True,
                short_circuit_reason=f"{r.plugin_id}: {r.result.reason}",
            )

    overrides = [r for r in plugin_results if isinstance(r.result, OverrideResult)]

    # Step 2: final priority (priority-smallest final wins, verbatim targets)
    finals = [r for r in overrides if r.final]
    if finals:
        winner = min(finals, key=lambda r: r.priority)
        assert isinstance(winner.result, OverrideResult)
        targets = list(winner.result.targets)
        set_dropped_final: list[ComponentKey] = []
        if not set_allowed:
            kept: list[ComponentTarget] = []
            for t in targets:
                if t.type == OverrideType.SET:
                    set_dropped_final.append(
                        ComponentKey(
                            sub_component_type=t.sub_component_type,
                            component_name=t.component_name,
                        )
                    )
                else:
                    kept.append(t)
            targets = kept
        return MergeOutcome(
            proposal=ScalingProposal(targets=targets, source=winner.plugin_id),
            short_circuited=False,
            used_final_from=winner.plugin_id,
            set_dropped=set_dropped_final,
        )

    # Steps 3-4: bucket by ComponentKey, merge per type, clamp.
    set_dropped: list[ComponentKey] = []
    by_key: dict[ComponentKey, list[tuple[ComponentTarget, int]]] = {}
    for r in overrides:
        assert isinstance(r.result, OverrideResult)
        for t in r.result.targets:
            if t.replicas is None:  # v9 line 1078: unset = no opinion
                continue
            key = ComponentKey(
                sub_component_type=t.sub_component_type,
                component_name=t.component_name,
            )
            if t.type == OverrideType.SET and not set_allowed:
                set_dropped.append(key)
                continue
            by_key.setdefault(key, []).append((t, r.priority))

    # Deterministic output order: plugin-touched keys first (insertion),
    # then any baseline-only keys.
    ordered_keys: list[ComponentKey] = list(by_key.keys())
    for k in baseline:
        if k not in by_key:
            ordered_keys.append(k)

    final_targets: list[ComponentTarget] = []
    clamped: list[tuple[ComponentKey, str, str]] = []
    for key in ordered_keys:
        entries = by_key.get(key, [])
        at_least_entries: list[tuple[int, str]] = [
            (t.replicas, _target_source(t, plugin_results, prio))
            for t, prio in entries
            if t.type == OverrideType.AT_LEAST and t.replicas is not None
        ]
        at_most_entries: list[tuple[int, str]] = [
            (t.replicas, _target_source(t, plugin_results, prio))
            for t, prio in entries
            if t.type == OverrideType.AT_MOST and t.replicas is not None
        ]
        set_vals: list[tuple[int, int]] = [
            (t.replicas, prio)
            for t, prio in entries
            if t.type == OverrideType.SET and t.replicas is not None
        ]
        at_least_vals = [v for v, _ in at_least_entries]
        at_most_vals = [v for v, _ in at_most_entries]
        floor: float = max(at_least_vals) if at_least_vals else 0
        ceiling: float = min(at_most_vals) if at_most_vals else math.inf
        if set_vals:
            recommendation: float = min(set_vals, key=lambda x: x[1])[0]
        else:
            recommendation = baseline.get(key, 0)
        # Clamp order: floor wins when floor > ceiling.
        result_replicas = max(floor, min(ceiling, recommendation))

        # Record per-key clamps so the orchestrator emits the right
        # RECONCILE/CONSTRAIN clamp counter.  Only report clamps that
        # actually changed the value — if recommendation was already
        # within [floor, ceiling] this key was un-clamped and the
        # bounds just confirmed it.
        if at_most_vals and result_replicas < recommendation:
            winning_at_most = min(at_most_entries, key=lambda x: x[0])
            clamped.append((key, "ceiling", winning_at_most[1]))
        if at_least_vals and result_replicas > recommendation:
            winning_at_least = max(at_least_entries, key=lambda x: x[0])
            clamped.append((key, "floor", winning_at_least[1]))

        final_targets.append(
            ComponentTarget(
                sub_component_type=key.sub_component_type,
                component_name=key.component_name,
                replicas=int(result_replicas),
            )
        )

    return MergeOutcome(
        proposal=ScalingProposal(targets=final_targets, source="merged"),
        short_circuited=False,
        set_dropped=set_dropped,
        clamped=clamped,
    )


def _target_source(
    target: ComponentTarget,
    plugin_results: list,
    priority: int,
) -> str:
    """Find which plugin emitted this specific ``ComponentTarget``.

    Used to populate the ``source`` label on clamp counters so a
    dashboard can show *which* plugin (budget-constrain, user's
    custom, etc) kept dragging a component off the recommendation.

    Match by object identity first (the PluginResult holds the same
    ComponentTarget instance we're looking at); fall back to priority
    + (type, replicas) equality for the rare case where the merge
    reconstructs targets."""
    from dynamo.planner.plugins.merge.types import PluginResult  # local import to avoid cycle
    from dynamo.planner.plugins.types import OverrideResult as _OverrideResult

    for pr in plugin_results:
        if not isinstance(pr, PluginResult):
            continue
        if pr.priority != priority:
            continue
        if not isinstance(pr.result, _OverrideResult):
            continue
        for t in pr.result.targets:
            if t is target:
                return pr.plugin_id
            if (
                t.sub_component_type == target.sub_component_type
                and t.component_name == target.component_name
                and t.type == target.type
                and t.replicas == target.replicas
            ):
                return pr.plugin_id
    return "unknown"


__all__ = ["type_aware_merge"]
