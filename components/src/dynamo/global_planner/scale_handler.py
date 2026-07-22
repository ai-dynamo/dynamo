# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler for scale_request endpoint in GlobalPlanner."""

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

from dynamo.planner import KubernetesConnector, SubComponentType, TargetReplica
from dynamo.planner.connectors.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.protocol import ScaleRequest, ScaleResponse, ScaleStatus
from dynamo.planner.core import budget
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)


@dataclass
class PoolSpec:
    """Snapshot of one pool's state read from the DGD spec."""

    sub_type: str
    current_replicas: int
    gpu_per_replica: int


@dataclass
class PoolIntent:
    """Most recently observed desired replica count for a pool."""

    last_desired: int
    last_seen_at: float


class ScaleRequestHandler:
    """Handles incoming scale requests in GlobalPlanner.

    This handler:
    1. Receives scale requests from Planners
    2. Validates caller authorization (optional)
    3. Caches KubernetesConnector per DGD for efficiency
    4. Executes scaling via Kubernetes API
    5. Returns current replica counts

    Management modes:
    - **Explicit** (``--managed-namespaces`` set): Only DGDs whose Dynamo
      namespaces are listed are managed. Authorization rejects requests from
      unlisted namespaces, and GPU budget only counts these DGDs.
    - **Implicit** (no ``--managed-namespaces``): All DGDs in the Kubernetes
      namespace are managed. Any caller is accepted, and GPU budget counts
      every DGD discovered in the namespace.

    Budget enforcement:
    - ``max_total_gpus`` is a ceiling; scale-ups that would exceed it are
      rejected unless a cached opposite-direction intent can be paired.
    - ``min_total_gpus`` is a floor; scale-downs that would drop below it
      are denied unless a cached opposite-direction intent from another pool
      can be paired with them (intra-DGD or cross-DGD).
    - Paired transfers may land up to ``tolerance`` GPUs **below** ``min``,
      where tolerance = max per-replica GPU across the pools actually being
      paired. This exists to handle asymmetric per-replica GPU counts where
      a single-worker step cannot exactly cancel across pools. ``max`` is
      a hard cluster-capacity bound and is never relaxed — overshooting it
      would risk pending pods / over-admission. Asymmetric pairs whose total
      would land above ``max`` are denied.

    Intent cache semantics:
    - Every scale request seeds the per-pool intent cache *before* the
      budget decision, so a request that gets denied this tick still
      leaves its desired count behind. A subsequent opposite-direction
      request from another pool can pair against that cached intent and
      execute the transfer.
    - Bounded by TTL (``intent_cache_ttl_seconds``) and by the
      satisfied-vs-pending check (``last_desired != current_replicas``):
      stale or already-satisfied intents are not eligible partners.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        managed_namespaces: list,
        k8s_namespace: str,
        no_operation: bool = False,
        max_total_gpus: int = -1,
        min_total_gpus: int = -1,
        intent_cache_ttl_seconds: float = 360.0,
    ):
        """Initialize the scale request handler.

        Args:
            runtime: Dynamo runtime instance
            managed_namespaces: List of authorized namespaces (None = accept all)
            k8s_namespace: Kubernetes namespace where GlobalPlanner is running
            no_operation: If True, log scale requests without executing K8s scaling
            max_total_gpus: Maximum total GPUs across all managed pools (-1 = unlimited)
            min_total_gpus: Minimum total GPUs across all managed pools (-1 = no floor)
            intent_cache_ttl_seconds: How long a cached scale intent from a pool
                is considered fresh for pairing
        """
        self.runtime = runtime
        # If managed_namespaces is None, accept all namespaces
        self.managed_namespaces = (
            set(managed_namespaces) if managed_namespaces else None
        )
        self.k8s_namespace = k8s_namespace
        self.no_operation = no_operation
        self.max_total_gpus = max_total_gpus
        self.min_total_gpus = min_total_gpus
        self.intent_cache_ttl_seconds = intent_cache_ttl_seconds
        self.connectors: dict[str, KubernetesConnector] = {}  # Cache per DGD
        # Per-pool cached desired replicas from recent ScaleRequests, keyed by
        # f"{k8s_ns}/{dgd_name}/{sub_type}". Used to pair opposite-direction
        # intents across requests when one request alone would breach bounds.
        self._intent_cache: dict[str, PoolIntent] = {}
        # Serializes budget-check + scale-execution so concurrent requests from
        # different pools cannot both pass against the same pre-scale state.
        self._scale_lock = asyncio.Lock()

        if self.managed_namespaces:
            logger.info(
                f"ScaleRequestHandler initialized for namespaces: {managed_namespaces}"
            )
        else:
            logger.info("ScaleRequestHandler initialized (accepting all namespaces)")

        if self.no_operation:
            logger.info(
                "ScaleRequestHandler running in NO-OPERATION mode: "
                "scale requests will be logged but not executed"
            )

        if self.max_total_gpus >= 0:
            logger.info(
                f"GPU budget ceiling ENABLED: max {self.max_total_gpus} total GPUs"
            )
        else:
            logger.info("GPU budget ceiling DISABLED (unlimited)")

        if self.min_total_gpus >= 0:
            logger.info(
                f"GPU budget floor ENABLED: min {self.min_total_gpus} total GPUs, "
                f"intent cache TTL {self.intent_cache_ttl_seconds}s"
            )
        else:
            logger.info("GPU budget floor DISABLED")

        if self.max_total_gpus >= 0 or self.min_total_gpus >= 0:
            self._populate_k8s_connectors()
            if self.min_total_gpus >= 0:
                self._warn_if_below_floor()

    def _managed_dgd_names(self) -> set[str] | None:
        """Derive the DGD names that this GlobalPlanner manages.

        Returns:
            A set of DGD names when in explicit mode, or None in implicit mode.

        The Dynamo operator convention is:
            DYN_NAMESPACE = "{k8s_namespace}-{dgd_name}"
        so the DGD name is the Dynamo namespace with the k8s prefix stripped.
        """
        if self.managed_namespaces is None:
            return None

        prefix = f"{self.k8s_namespace}-"
        names = set()
        for ns in self.managed_namespaces:
            if ns.startswith(prefix):
                names.add(ns[len(prefix) :])
            else:
                logger.warning(
                    f"Managed namespace '{ns}' does not start with "
                    f"expected prefix '{prefix}'; cannot derive DGD name"
                )
        return names

    def _populate_k8s_connectors(self):
        """Pre-populate connectors for DGDs managed by this GlobalPlanner.

        This ensures the GPU budget calculation accounts for DGDs that already
        exist at startup, even if they haven't sent a scale request yet.

        In explicit mode (--managed-namespaces set), only DGDs whose names
        match the managed Dynamo namespaces are discovered.
        In implicit mode, all DGDs in the k8s namespace are discovered.
        """
        try:
            kube_api = KubernetesAPI(self.k8s_namespace)
            managed_names = self._managed_dgd_names()
            dgds = kube_api.list_graph_deployments()
            discovered = []
            for dgd in dgds:
                name = dgd.get("metadata", {}).get("name", "")
                if not name:
                    continue
                # In explicit mode, skip DGDs not in the managed set
                if managed_names is not None and name not in managed_names:
                    continue
                connector_key = f"{self.k8s_namespace}/{name}"
                if connector_key not in self.connectors:
                    connector = KubernetesConnector(
                        dynamo_namespace="discovered",
                        k8s_namespace=self.k8s_namespace,
                        parent_dgd_name=name,
                    )
                    self.connectors[connector_key] = connector
                discovered.append(name)
            logger.info(f"Discovered {len(discovered)} existing DGDs: {discovered}")
        except Exception as e:
            logger.warning(f"Failed to discover existing DGDs: {e}")

    def _warn_if_below_floor(self):
        """Log a warning if the discovered initial state is below min_total_gpus.

        Soft floor: we do not proactively scale up. The floor prevents
        scale-downs below it, but initial below-floor state is allowed and
        will drift toward the floor as load arrives.
        """
        try:
            total = self._total_gpus_with_overrides({})
        except Exception as e:
            logger.warning(f"Could not compute initial total GPUs: {e}")
            return
        if total < self.min_total_gpus:
            logger.warning(
                f"Current total GPUs ({total}) is below min_total_gpus "
                f"({self.min_total_gpus}); scale-up from load scaler will "
                f"drift toward the floor. No proactive fill is issued."
            )
        else:
            logger.info(
                f"Initial total GPUs ({total}) meets floor ({self.min_total_gpus})"
            )

    def _read_dgd_pools(self, connector: KubernetesConnector) -> dict[str, PoolSpec]:
        """Read the current pool state for one DGD.

        Returns a map from sub_component_type to PoolSpec. Pools with 0
        gpu_per_replica are included for completeness but contribute 0 to
        budget math.
        """
        deployment = connector.kube_api.get_graph_deployment(connector.parent_dgd_name)
        pools: dict[str, PoolSpec] = {}
        services = deployment.get("spec", {}).get("services", {})
        for svc_spec in services.values():
            sub_type = svc_spec.get("subComponentType", "")
            if not sub_type:
                continue
            gpu_per_replica = int(
                svc_spec.get("resources", {}).get("limits", {}).get("gpu", 0)
            )
            replicas = svc_spec.get("replicas", 0)
            pools[sub_type] = PoolSpec(
                sub_type=sub_type,
                current_replicas=replicas,
                gpu_per_replica=gpu_per_replica,
            )
        return pools

    def _read_all_pools(self) -> dict[str, dict[str, PoolSpec]]:
        """Read current pool state for every known DGD.

        Returns a map of dgd_key -> (sub_type -> PoolSpec). Each arbitration
        call reads fresh state once; cross-DGD partner search and budget math
        both consume this snapshot to avoid re-hitting the K8s API per lookup.

        Snapshots ``self.connectors`` up-front via ``list(...)``: this method
        runs on a worker thread (see ``asyncio.to_thread`` in scale_request),
        and a concurrent first-time request for another DGD can insert into
        the dict before it blocks on ``_scale_lock``. Without the snapshot,
        that insertion races our iteration and either raises
        ``RuntimeError: dictionary changed size during iteration`` or yields
        a snapshot missing the new DGD.
        """
        all_pools: dict[str, dict[str, PoolSpec]] = {}
        for key, connector in list(self.connectors.items()):
            try:
                all_pools[key] = self._read_dgd_pools(connector)
            except Exception as e:
                logger.warning(f"Failed to read DGD for {key}: {e}")
                all_pools[key] = {}
        return all_pools

    def _total_gpus_from_snapshot(
        self,
        all_pools: dict[str, dict[str, PoolSpec]],
        overrides: dict[tuple[str, str], int],
    ) -> int:
        """Compute total GPUs across all known DGDs from a pre-read snapshot.

        Args:
            all_pools: pool snapshot as returned by ``_read_all_pools``.
            overrides: Map from (dgd_key, sub_component_type) to the replica
                count to use in place of the current K8s replica count. Any
                entry not in ``overrides`` uses the current K8s replica count.
        """
        total_gpus = 0
        for key, pools in all_pools.items():
            for sub_type, spec in pools.items():
                if spec.gpu_per_replica == 0:
                    continue
                replicas = overrides.get((key, sub_type), spec.current_replicas)
                total_gpus += replicas * spec.gpu_per_replica
        return total_gpus

    def _total_gpus_with_overrides(self, overrides: dict[tuple[str, str], int]) -> int:
        """Compute total GPUs across all known DGDs (re-reads K8s).

        Kept for backward compatibility (startup warnings, legacy callers).
        In the hot arbitration path, prefer ``_total_gpus_from_snapshot`` with
        a pre-read ``_read_all_pools`` result.

        NOTE: GPU count is read from spec.services[].resources.limits.gpu only.
        GPUs specified via resources.requests.gpu or extraPodSpec resource
        overrides are not counted.
        """
        return self._total_gpus_from_snapshot(self._read_all_pools(), overrides)

    # ------------------------------------------------------------------ #
    # Intent cache helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pool_cache_key(dgd_key: str, sub_type: str) -> str:
        return f"{dgd_key}/{sub_type}"

    @staticmethod
    def _direction(desired: int, current: int) -> str:
        if desired > current:
            return "up"
        if desired < current:
            return "down"
        return "stable"

    def _update_intent_cache(
        self, dgd_key: str, request: ScaleRequest, dgd_pools: dict[str, PoolSpec]
    ):
        """Record the desired replicas for each pool in this request."""
        now = time.time()
        for target in request.target_replicas:
            sub_type = target.sub_component_type.value
            if sub_type not in dgd_pools:
                # Unknown pool (not yet in DGD spec); skip — without
                # gpu_per_replica we can't compute deltas or pair against it.
                continue
            key = self._pool_cache_key(dgd_key, sub_type)
            self._intent_cache[key] = PoolIntent(
                last_desired=target.desired_replicas,
                last_seen_at=now,
            )

    def _pair_tolerance(
        self,
        request_pools: list[PoolSpec],
        partner_spec: PoolSpec,
    ) -> int:
        """Tolerance for a specific paired transfer.

        Equal to max per-replica GPU across just the pools actually being
        changed (request's non-stable pools + partner). Covers step-size
        asymmetry where a single worker on one side can't exactly cancel
        a single worker on the other side.
        """
        return budget.compute_tolerance(
            [p.gpu_per_replica for p in request_pools] + [partner_spec.gpu_per_replica]
        )

    def _internal_pair_tolerance(
        self,
        changing_pools: list[PoolSpec],
    ) -> int:
        """Tolerance for an internally-paired request (no external partner)."""
        return budget.compute_tolerance(p.gpu_per_replica for p in changing_pools)

    def _iter_pair_partners(
        self,
        request_dgd_key: str,
        request_pool_keys: set[tuple[str, str]],
        all_pools: dict[str, dict[str, PoolSpec]],
        request_net_delta_gpu: int,
    ) -> Iterator[tuple[str, str, int, PoolSpec]]:
        """Yield qualifying pair-partner candidates, same-DGD first.

        A candidate qualifies when it (a) is not in the requesting pool set,
        (b) has a fresh cached intent (within TTL) whose desired differs
        from current replicas, and (c) the partner delta points opposite
        to the request's net delta.

        Yields in two passes: same-DGD candidates first (atomic-patch
        preference), then cross-DGD candidates. The caller picks the first
        one whose pair total actually lands in the budget band — checking
        feasibility per-candidate is the caller's job because tolerance and
        total depend on which partner is chosen, not just which exist.

        Args:
            request_dgd_key: "k8s_ns/dgd_name" of the requesting DGD.
            request_pool_keys: (dgd_key, sub_type) tuples already in the
                incoming request — excluded from partner search.
            all_pools: snapshot of all DGDs' pool state.
            request_net_delta_gpu: Net GPU delta of the request standalone.
                Zero short-circuits (no pairing needed).
        """
        if request_net_delta_gpu == 0:
            return
        now = time.time()
        same_dgd: list[tuple[str, str, int, PoolSpec]] = []
        cross_dgd: list[tuple[str, str, int, PoolSpec]] = []
        for dgd_key, pools in all_pools.items():
            for sub_type, spec in pools.items():
                if (dgd_key, sub_type) in request_pool_keys:
                    continue
                if spec.gpu_per_replica == 0:
                    continue
                cache_key = self._pool_cache_key(dgd_key, sub_type)
                intent = self._intent_cache.get(cache_key)
                if intent is None:
                    continue
                if now - intent.last_seen_at > self.intent_cache_ttl_seconds:
                    continue
                if intent.last_desired == spec.current_replicas:
                    continue  # Satisfied — nothing to apply.
                partner_delta_gpu = (
                    intent.last_desired - spec.current_replicas
                ) * spec.gpu_per_replica
                # Must be opposite direction of the request's net delta.
                if (request_net_delta_gpu > 0 and partner_delta_gpu >= 0) or (
                    request_net_delta_gpu < 0 and partner_delta_gpu <= 0
                ):
                    continue
                candidate = (dgd_key, sub_type, intent.last_desired, spec)
                if dgd_key == request_dgd_key:
                    same_dgd.append(candidate)
                else:
                    cross_dgd.append(candidate)
        yield from same_dgd
        yield from cross_dgd

    def _partial_partner(
        self,
        candidate: tuple[str, str, int, PoolSpec],
        all_pools: dict[str, dict[str, PoolSpec]],
        current_overrides: dict[tuple[str, str], int],
        tolerance: int,
    ) -> Optional[int]:
        """Compute a partial ``applied_desired`` for a candidate whose full
        consumption would push the combined transfer out of the band.

        Returns an integer ``K`` strictly between ``current_replicas`` and
        ``last_desired`` (direction-consistent) such that applying ``K``
        instead of ``last_desired`` lands the combined transfer at the
        appropriate band edge — the strict ceiling on the upper side
        (``max``, no tolerance) or ``min - tolerance`` on the lower side.
        ``None`` if no feasible partial exists (e.g., even one worker's
        contribution overshoots).
        """
        _, _, last_desired, spec = candidate
        current = spec.current_replicas
        gpu = spec.gpu_per_replica
        if gpu <= 0 or last_desired == current:
            return None

        # Combined total assuming this candidate stays at its current count
        # (i.e., contributes 0). The candidate is NOT in current_overrides
        # yet, so the snapshot uses its current_replicas naturally.
        baseline_total = self._total_gpus_from_snapshot(all_pools, current_overrides)

        if last_desired > current:
            # Scale-up candidate: pick K in [current+1, last_desired] that
            # keeps total <= max (strict ceiling — max is a hard hardware
            # bound, see budget.bounds_for_total).
            if self.max_total_gpus < 0:
                return last_desired
            # K <= current + (max - baseline_total) // gpu
            headroom = self.max_total_gpus - baseline_total
            if headroom <= 0:
                return None
            max_k = current + headroom // gpu
            k = min(last_desired, max_k)
            return k if k > current else None
        else:
            # Scale-down candidate: pick K in [last_desired, current-1] that
            # keeps total >= min - tolerance.
            if self.min_total_gpus < 0:
                return last_desired
            lower = self.min_total_gpus - tolerance
            # K >= current + ceil((lower - baseline_total) / gpu)
            diff = lower - baseline_total
            if diff > 0:
                # Already below floor; further scale-down impossible.
                return None
            min_k = current + math.ceil(diff / gpu)
            k = max(last_desired, min_k)
            return k if k < current else None

    def _find_pair_partner_set(
        self,
        request_dgd_key: str,
        request_pool_keys: set[tuple[str, str]],
        all_pools: dict[str, dict[str, PoolSpec]],
        request_net_delta_gpu: int,
        standalone_overrides: dict[tuple[str, str], int],
        changing_request_pools: list[PoolSpec],
    ) -> tuple[list[tuple[str, str, int, PoolSpec]], int, int]:
        """Pack as many opposite-direction cached intents as fit alongside
        the request, partially consuming one over-sized candidate if needed.

        Algorithm: greedy admission, ascending ``abs(delta_gpu)`` order. For
        each candidate, fully admit if it keeps the combined transfer in
        ``[min - tolerance, max]``; if full admission would overshoot the
        strict ceiling, try partial consumption that lands at the band edge
        and stop (the band is now tight, larger candidates won't fit either).

        Tolerance is computed **once** over the request's changing pools
        plus all candidates considered for inclusion, not iteratively
        widened — this avoids early candidates being admitted under a
        tighter tolerance that wouldn't admit them after the band widened.

        Returns (selected_partners, total_after, tolerance).
        ``selected_partners`` is empty when no feasible packing exists.
        """
        if request_net_delta_gpu == 0:
            return [], 0, 0

        all_candidates = list(
            self._iter_pair_partners(
                request_dgd_key,
                request_pool_keys,
                all_pools,
                request_net_delta_gpu,
            )
        )
        if not all_candidates:
            return [], 0, 0

        # Tolerance computed once over the universe of changing pools.
        candidate_specs = [c[3] for c in all_candidates]
        tolerance = budget.compute_tolerance(
            [s.gpu_per_replica for s in changing_request_pools]
            + [s.gpu_per_replica for s in candidate_specs]
        )

        # Sort ascending by |delta_gpu| — smaller pieces overshoot less.
        def cand_delta(c: tuple[str, str, int, PoolSpec]) -> int:
            _, _, desired, spec = c
            return (desired - spec.current_replicas) * spec.gpu_per_replica

        all_candidates.sort(key=lambda c: abs(cand_delta(c)))

        selected: list[tuple[str, str, int, PoolSpec]] = []
        overrides = dict(standalone_overrides)

        for cand in all_candidates:
            cand_dgd, cand_sub, cand_desired, cand_spec = cand

            # Try full inclusion.
            full_overrides = dict(overrides)
            full_overrides[(cand_dgd, cand_sub)] = cand_desired
            full_total = self._total_gpus_from_snapshot(all_pools, full_overrides)
            in_band, _ = budget.bounds_for_total(
                full_total,
                self.min_total_gpus,
                self.max_total_gpus,
                tolerance,
            )
            if in_band:
                # Full inclusion lands in band — accept and continue. The
                # user's framing is "as long as we stay within the threshold,
                # do the larger groups of scaling decisions" — so we keep
                # admitting more candidates while feasible. Subsequent
                # iterations either continue extending (next full also in
                # band) or trigger partial-then-break (next full crosses).
                selected.append(cand)
                overrides = full_overrides
                continue

            # Out of band. Did this candidate cross the band, or are we
            # still on the wrong side and need more help?
            #
            # Request delta sign indicates which side we started on:
            #   request_net_delta > 0 → request alone overshoots ceiling
            #     (approaching from above; candidates pull down).
            #   request_net_delta < 0 → request alone undershoots floor
            #     (approaching from below; candidates push up).
            above_ceiling = (
                self.max_total_gpus >= 0 and full_total > self.max_total_gpus
            )
            below_floor = (
                self.min_total_gpus >= 0
                and full_total < self.min_total_gpus - tolerance
            )
            still_approaching = (request_net_delta_gpu > 0 and above_ceiling) or (
                request_net_delta_gpu < 0 and below_floor
            )
            if still_approaching:
                # Candidate moved us toward the band but didn't reach it.
                # Accept full inclusion and try the next candidate.
                selected.append(cand)
                overrides = full_overrides
                continue

            # Full inclusion crossed the band. Try partial consumption that
            # lands at the appropriate band edge, then stop.
            partial_k = self._partial_partner(cand, all_pools, overrides, tolerance)
            if partial_k is not None and partial_k != cand_spec.current_replicas:
                partial_cand = (cand_dgd, cand_sub, partial_k, cand_spec)
                selected.append(partial_cand)
                overrides[(cand_dgd, cand_sub)] = partial_k
            break

        # Loop ended with the running total possibly still on the wrong side
        # (no candidate fully reached the band, none of them crossed). Try
        # partial of the last selected candidate to land in band.
        if selected:
            running_total = self._total_gpus_from_snapshot(all_pools, overrides)
            running_in_band, _ = budget.bounds_for_total(
                running_total,
                self.min_total_gpus,
                self.max_total_gpus,
                tolerance,
            )
            if not running_in_band:
                last = selected[-1]
                last_dgd, last_sub, _, last_spec = last
                # Roll back the last full inclusion so partial uses the
                # pre-last-candidate baseline.
                rollback_overrides = dict(overrides)
                if (last_dgd, last_sub) in standalone_overrides:
                    rollback_overrides[(last_dgd, last_sub)] = standalone_overrides[
                        (last_dgd, last_sub)
                    ]
                else:
                    rollback_overrides.pop((last_dgd, last_sub), None)
                partial_k = self._partial_partner(
                    last, all_pools, rollback_overrides, tolerance
                )
                if partial_k is not None and partial_k != last_spec.current_replicas:
                    selected[-1] = (last_dgd, last_sub, partial_k, last_spec)
                    overrides = dict(rollback_overrides)
                    overrides[(last_dgd, last_sub)] = partial_k

        if not selected:
            return [], 0, 0

        final_total = self._total_gpus_from_snapshot(all_pools, overrides)
        ok, _ = budget.bounds_for_total(
            final_total,
            self.min_total_gpus,
            self.max_total_gpus,
            tolerance,
        )
        if not ok:
            return [], 0, 0

        return selected, final_total, tolerance

    # ------------------------------------------------------------------ #
    # Request handling                                                   #
    # ------------------------------------------------------------------ #

    def _request_net_delta_gpu(
        self,
        request: ScaleRequest,
        dgd_pools: dict[str, PoolSpec],
    ) -> int:
        """Sum of (desired - current) * gpu_per_replica across all pools in the request."""
        net = 0
        for target in request.target_replicas:
            sub_type = target.sub_component_type.value
            spec = dgd_pools.get(sub_type)
            if spec is None or spec.gpu_per_replica == 0:
                continue
            net += (
                target.desired_replicas - spec.current_replicas
            ) * spec.gpu_per_replica
        return net

    def _request_is_internally_paired(
        self,
        request: ScaleRequest,
        dgd_pools: dict[str, PoolSpec],
    ) -> bool:
        """True if the request contains both up and down directions across pools."""
        has_up = False
        has_down = False
        for target in request.target_replicas:
            sub_type = target.sub_component_type.value
            spec = dgd_pools.get(sub_type)
            if spec is None:
                continue
            direction = self._direction(target.desired_replicas, spec.current_replicas)
            if direction == "up":
                has_up = True
            elif direction == "down":
                has_down = True
        return has_up and has_down

    def _budget_enforcement_enabled(self) -> bool:
        return self.max_total_gpus >= 0 or self.min_total_gpus >= 0

    def _bounds_for_total(
        self,
        total: int,
        paired: bool,
        tolerance: int,
    ) -> tuple[bool, str]:
        """Check whether ``total`` is within the active budget bounds.

        Returns ``(is_in_bounds, reason_if_out_of_bounds)``.

        Standalone (non-paired) requests use strict bounds; paired transfers
        get the tolerance band on the **lower** edge only — ``max_total_gpus``
        is a hard cluster-capacity bound (enforced by
        ``budget.bounds_for_total``).
        """
        return budget.bounds_for_total(
            total,
            self.min_total_gpus,
            self.max_total_gpus,
            tolerance if paired else 0,
        )

    @dynamo_endpoint(ScaleRequest, ScaleResponse)
    async def scale_request(self, request: ScaleRequest):
        """Process scaling request from a Planner.

        Args:
            request: ScaleRequest with target replicas and DGD info

        Yields:
            ScaleResponse with status and current replica counts
        """
        try:
            # Validate caller namespace (if authorization is enabled)
            if (
                self.managed_namespaces is not None
                and request.caller_namespace not in self.managed_namespaces
            ):
                yield {
                    "status": ScaleStatus.ERROR.value,
                    "message": f"Namespace {request.caller_namespace} not authorized",
                    "current_replicas": {},
                }
                return

            # No-operation mode: log and return success without touching K8s
            if self.no_operation:
                replicas_summary = {
                    r.sub_component_type.value: r.desired_replicas
                    for r in request.target_replicas
                }
                logger.info(
                    f"[NO-OP] Scale request from {request.caller_namespace} "
                    f"for DGD {request.graph_deployment_name} "
                    f"in K8s namespace {request.k8s_namespace}: {replicas_summary}"
                )
                yield {
                    "status": ScaleStatus.SUCCESS.value,
                    "message": "[no-operation] Scale request received and logged (not executed)",
                    "current_replicas": {},
                }
                return

            logger.info(
                f"Processing scale request from {request.caller_namespace} "
                f"for DGD {request.graph_deployment_name} "
                f"in K8s namespace {request.k8s_namespace}"
            )

            # Get or create connector for this DGD
            connector_key = f"{request.k8s_namespace}/{request.graph_deployment_name}"
            if connector_key not in self.connectors:
                connector = KubernetesConnector(
                    dynamo_namespace=request.caller_namespace,
                    k8s_namespace=request.k8s_namespace,
                    parent_dgd_name=request.graph_deployment_name,
                )
                self.connectors[connector_key] = connector
                logger.debug(f"Created new connector for {connector_key}")
            else:
                connector = self.connectors[connector_key]
                logger.debug(f"Reusing cached connector for {connector_key}")

            # Lock ensures the budget check and scale execution are atomic
            # so concurrent requests from different pools cannot both pass
            # against the same pre-scale replica counts.
            async with self._scale_lock:
                # Read ALL known DGDs' current state once. Cross-DGD partner
                # search needs to see every pool's current replicas and
                # gpu_per_replica; cross-DGD budget math also consumes this.
                # Run the synchronous K8s GETs off-thread so the event loop
                # (health checks, other endpoints) isn't blocked for the
                # N round-trips it takes across managed DGDs.
                all_pools = await asyncio.to_thread(self._read_all_pools)
                dgd_pools = all_pools.get(connector_key, {})

                # Always update the intent cache with this request's targets,
                # regardless of decision. A later request from a complementary
                # pool may need to pair with this intent.
                self._update_intent_cache(connector_key, request, dgd_pools)

                # Build standalone overrides (request targets only).
                request_key = connector_key
                request_pool_keys = {
                    (request_key, t.sub_component_type.value)
                    for t in request.target_replicas
                }
                standalone_overrides = {
                    (request_key, t.sub_component_type.value): t.desired_replicas
                    for t in request.target_replicas
                }

                # Selected pair-partners (possibly multiple). Empty list
                # means "no partners involved" — the standalone or
                # no-budget path.
                selected_partners: list[
                    tuple[str, str, int, PoolSpec]
                ] = []  # (dgd_key, sub_type, applied_desired, spec)

                if self._budget_enforcement_enabled():
                    net_delta = self._request_net_delta_gpu(request, dgd_pools)
                    internally_paired = self._request_is_internally_paired(
                        request, dgd_pools
                    )

                    total_standalone = self._total_gpus_from_snapshot(
                        all_pools, standalone_overrides
                    )

                    # Tolerance depends on which pools are actually changing.
                    changing_request_pools = [
                        dgd_pools[t.sub_component_type.value]
                        for t in request.target_replicas
                        if t.sub_component_type.value in dgd_pools
                        and t.desired_replicas
                        != dgd_pools[t.sub_component_type.value].current_replicas
                    ]
                    standalone_tolerance = self._internal_pair_tolerance(
                        changing_request_pools
                    )

                    # Internally-paired requests get tolerance even without an
                    # external partner.
                    standalone_ok, standalone_reason = self._bounds_for_total(
                        total_standalone, internally_paired, standalone_tolerance
                    )

                    # Multi-partner packing: pack as many opposite-direction
                    # cached intents as fit within the band, partially
                    # consuming one over-sized candidate if needed.
                    (
                        selected_partners,
                        total_paired,
                        paired_tolerance,
                    ) = self._find_pair_partner_set(
                        request_key,
                        request_pool_keys,
                        all_pools,
                        net_delta,
                        standalone_overrides,
                        changing_request_pools,
                    )

                    # Decide:
                    # 1. Non-empty pair set → apply request + all partners.
                    # 2. Else if standalone in bounds → apply standalone.
                    # 3. Else deny.
                    if selected_partners:
                        scope = (
                            "intra-DGD"
                            if all(p[0] == request_key for p in selected_partners)
                            else "cross-DGD"
                        )
                        partners_desc = ", ".join(
                            f"{p[0]}/{p[1]}={p[2]}" for p in selected_partners
                        )
                        logger.info(
                            f"Paired transfer ({scope}, "
                            f"{len(selected_partners)} partner(s)) for DGD "
                            f"{request.graph_deployment_name}: "
                            f"request {sorted(request_pool_keys)} + "
                            f"[{partners_desc}]; total {total_paired} GPUs "
                            f"(bounds "
                            f"[{self.min_total_gpus if self.min_total_gpus >= 0 else '-inf'} - {paired_tolerance}, "
                            f"{self.max_total_gpus if self.max_total_gpus >= 0 else '+inf'}])"
                        )
                    elif standalone_ok:
                        logger.info(
                            f"Standalone scale request for DGD {request.graph_deployment_name}: "
                            f"total {total_standalone} GPUs "
                            f"(internally_paired={internally_paired})"
                        )
                    else:
                        # Budget breach: standalone out-of-bounds and no
                        # feasible partner set found.
                        logger.warning(
                            f"Rejecting scale request from {request.caller_namespace}: "
                            f"{standalone_reason}; no feasible pair packing"
                        )
                        # Soft denial: budget breach is an expected operational
                        # outcome in fixed-total mode, not a fault. Local
                        # planners should treat this as a no-op for this tick.
                        yield {
                            "status": ScaleStatus.REJECTED.value,
                            "message": (
                                f"GPU budget breach: {standalone_reason}; "
                                f"no feasible pair packing"
                            ),
                            "current_replicas": {},
                        }
                        return

                # Apply: request + selected partners (may be empty), grouped
                # by DGD with at most one set_component_replicas call per DGD.
                # Direction-aware order: scale-down DGDs first (most negative
                # net delta), so that GPUs are freed before scale-up DGDs
                # submit new pods. Within each DGD, the request's targets and
                # any same-DGD partners are combined into a single atomic
                # patch. Cross-DGD partners get separate per-DGD patches.
                dgd_targets: dict[str, list[TargetReplica]] = defaultdict(list)
                dgd_targets[request_key].extend(request.target_replicas)
                for p_dgd, p_sub, p_desired, _ in selected_partners:
                    dgd_targets[p_dgd].append(
                        TargetReplica(
                            sub_component_type=SubComponentType(p_sub),
                            desired_replicas=p_desired,
                        )
                    )

                # Compute net GPU delta per DGD for ordering.
                dgd_net_deltas: dict[str, int] = {}
                for dgd_key_iter, targets in dgd_targets.items():
                    pools = all_pools.get(dgd_key_iter, {})
                    net = 0
                    for t in targets:
                        spec = pools.get(t.sub_component_type.value)
                        if spec is not None and spec.gpu_per_replica > 0:
                            net += (
                                t.desired_replicas - spec.current_replicas
                            ) * spec.gpu_per_replica
                    dgd_net_deltas[dgd_key_iter] = net

                # Sort: most negative (scale-down) first, most positive
                # (scale-up) last.
                ordered_dgds = sorted(
                    dgd_targets.keys(), key=lambda k: dgd_net_deltas[k]
                )

                applied_dgds: list[str] = []
                for i, dgd_key_iter in enumerate(ordered_dgds):
                    targets = dgd_targets[dgd_key_iter]
                    target_conn = (
                        connector
                        if dgd_key_iter == request_key
                        else self.connectors.get(dgd_key_iter)
                    )
                    if target_conn is None:
                        if i == 0:
                            # First patch: missing connector is unrecoverable
                            # since nothing has been applied yet.
                            logger.error(
                                f"Multi-partner transfer aborted: missing "
                                f"connector for first DGD ({dgd_key_iter})"
                            )
                            yield {
                                "status": ScaleStatus.ERROR.value,
                                "message": (
                                    f"Multi-partner transfer: missing "
                                    f"connector for {dgd_key_iter}"
                                ),
                                "current_replicas": {},
                            }
                            return
                        logger.error(
                            f"Multi-partner transfer: missing connector for "
                            f"{dgd_key_iter} after applying {applied_dgds}; "
                            f"will self-correct on next tick"
                        )
                        continue
                    try:
                        await target_conn.set_component_replicas(
                            targets, blocking=request.blocking
                        )
                        applied_dgds.append(dgd_key_iter)
                    except Exception as patch_err:
                        if i == 0:
                            # First patch failure: nothing applied, propagate
                            # to the outer try so the caller sees ERROR.
                            raise
                        logger.error(
                            f"Multi-partner transfer: patch on {dgd_key_iter} "
                            f"failed after applying {applied_dgds}: "
                            f"{patch_err}; will self-correct on next tick"
                        )

            # Get current replica counts
            current_replicas = {}
            deployment = connector.kube_api.get_graph_deployment(
                connector.parent_dgd_name
            )
            for service_name, service_spec in deployment["spec"]["services"].items():
                sub_type = service_spec.get("subComponentType", "")
                if sub_type:
                    current_replicas[sub_type] = service_spec.get("replicas", 0)

            logger.info(
                f"Successfully scaled {request.graph_deployment_name}: {current_replicas}"
            )
            yield {
                "status": ScaleStatus.SUCCESS.value,
                "message": f"Scaled {request.graph_deployment_name} successfully",
                "current_replicas": current_replicas,
            }

        except Exception as e:
            logger.exception(f"Error processing scale request: {e}")
            yield {
                "status": ScaleStatus.ERROR.value,
                "message": str(e),
                "current_replicas": {},
            }
