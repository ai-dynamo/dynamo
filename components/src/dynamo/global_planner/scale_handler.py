# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler for scale_request endpoint in GlobalPlanner."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from dynamo.planner import KubernetesConnector, SubComponentType, TargetReplica
from dynamo.planner.connectors.kubernetes_api import KubernetesAPI
from dynamo.planner.connectors.protocol import ScaleRequest, ScaleResponse, ScaleStatus
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
    - Paired transfers may land up to ``tolerance`` GPUs outside
      [min, max] where tolerance = max per-replica GPU across the two pools
      actually being paired. This exists to handle asymmetric per-replica
      GPU counts where a single-worker step cannot exactly cancel across pools.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        managed_namespaces: list,
        k8s_namespace: str,
        no_operation: bool = False,
        max_total_gpus: int = -1,
        min_total_gpus: int = -1,
        intent_cache_ttl_seconds: float = 120.0,
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
        """
        all_pools: dict[str, dict[str, PoolSpec]] = {}
        for key, connector in self.connectors.items():
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

    def _calculate_total_gpus_after_request(self, request: ScaleRequest) -> int:
        """Calculate total GPUs across all managed DGDs if this request is granted.

        For the requesting DGD, uses the desired replica counts from the request.
        For all other known DGDs, uses their current replica counts.
        """
        request_key = f"{request.k8s_namespace}/{request.graph_deployment_name}"
        overrides = {
            (request_key, target.sub_component_type.value): target.desired_replicas
            for target in request.target_replicas
        }
        return self._total_gpus_with_overrides(overrides)

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
                # Unknown pool (not yet in DGD spec); still cache, other math
                # will ignore it because gpu_per_replica is unknown.
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
        gpus = [p.gpu_per_replica for p in request_pools if p.gpu_per_replica > 0]
        if partner_spec.gpu_per_replica > 0:
            gpus.append(partner_spec.gpu_per_replica)
        return max(gpus, default=0)

    def _internal_pair_tolerance(
        self,
        changing_pools: list[PoolSpec],
    ) -> int:
        """Tolerance for an internally-paired request (no external partner)."""
        gpus = [p.gpu_per_replica for p in changing_pools if p.gpu_per_replica > 0]
        return max(gpus, default=0)

    def _find_pair_partner(
        self,
        request_dgd_key: str,
        request_pool_keys: set[tuple[str, str]],
        all_pools: dict[str, dict[str, PoolSpec]],
        request_net_delta_gpu: int,
    ) -> Optional[tuple[str, str, int, PoolSpec]]:
        """Find any pool (same or different DGD) with a fresh opposite-direction pending intent.

        Same-DGD candidates are preferred over cross-DGD candidates so that
        the resulting transfer can be applied as a single atomic K8s patch.

        Args:
            request_dgd_key: "k8s_ns/dgd_name" of the requesting DGD.
            request_pool_keys: (dgd_key, sub_type) tuples already in the
                incoming request — these are excluded from partner search.
            all_pools: snapshot of all DGDs' pool state.
            request_net_delta_gpu: Net GPU delta this request would apply if
                executed standalone. Partner must push net back toward
                [min, max], i.e., opposite sign.

        Returns:
            (partner_dgd_key, partner_sub_type, partner_desired_replicas,
             partner_pool_spec) if a suitable pair is found; None otherwise.
        """
        if request_net_delta_gpu == 0:
            return None
        now = time.time()
        same_dgd_match: Optional[tuple[str, str, int, PoolSpec]] = None
        cross_dgd_match: Optional[tuple[str, str, int, PoolSpec]] = None
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
                    same_dgd_match = candidate
                    # Same-DGD is the strongest preference; no need to keep
                    # scanning this DGD for more options.
                    break
                elif cross_dgd_match is None:
                    cross_dgd_match = candidate
            if same_dgd_match is not None:
                break
        return same_dgd_match if same_dgd_match is not None else cross_dgd_match

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

        Returns (is_in_bounds, reason_if_out_of_bounds).
        """
        if self.max_total_gpus >= 0:
            hi = self.max_total_gpus + (tolerance if paired else 0)
            if total > hi:
                return (
                    False,
                    f"total {total} exceeds ceiling "
                    f"({self.max_total_gpus}{' + tol ' + str(tolerance) if paired else ''})",
                )
        if self.min_total_gpus >= 0:
            lo = self.min_total_gpus - (tolerance if paired else 0)
            if total < lo:
                return (
                    False,
                    f"total {total} below floor "
                    f"({self.min_total_gpus}{' - tol ' + str(tolerance) if paired else ''})",
                )
        return (True, "")

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

                # Track whether a paired partner is being applied and, if so,
                # which DGD + pool it's in. Needed to decide atomic vs two-step
                # execution.
                partner_info: Optional[
                    tuple[str, str, int, PoolSpec]
                ] = None  # (dgd_key, sub_type, desired, spec)

                if self._budget_enforcement_enabled():
                    net_delta = self._request_net_delta_gpu(request, dgd_pools)
                    internally_paired = self._request_is_internally_paired(
                        request, dgd_pools
                    )

                    # Look for a pair partner across ALL DGDs (same-DGD
                    # preferred over cross-DGD).
                    partner = self._find_pair_partner(
                        request_key,
                        request_pool_keys,
                        all_pools,
                        net_delta,
                    )

                    paired_overrides = dict(standalone_overrides)
                    if partner is not None:
                        partner_dgd, partner_sub, partner_desired, _ = partner
                        paired_overrides[(partner_dgd, partner_sub)] = partner_desired

                    total_standalone = self._total_gpus_from_snapshot(
                        all_pools, standalone_overrides
                    )
                    total_paired = (
                        self._total_gpus_from_snapshot(all_pools, paired_overrides)
                        if partner is not None
                        else total_standalone
                    )

                    # Tolerance depends on context:
                    # - Standalone + internally_paired: max gpu_per_replica
                    #   across the request's changing pools.
                    # - Paired: max across request's changing pools + partner.
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
                    paired_tolerance = (
                        self._pair_tolerance(changing_request_pools, partner[3])
                        if partner is not None
                        else 0
                    )

                    # Internally-paired requests get tolerance even without an
                    # external partner.
                    standalone_is_paired = internally_paired
                    standalone_ok, standalone_reason = self._bounds_for_total(
                        total_standalone, standalone_is_paired, standalone_tolerance
                    )

                    if partner is not None:
                        paired_ok, paired_reason = self._bounds_for_total(
                            total_paired, True, paired_tolerance
                        )
                    else:
                        paired_ok, paired_reason = False, "no partner"

                    # Decide:
                    # 1. If pair exists and is in bounds → apply pair.
                    # 2. Else if standalone is in bounds → apply standalone.
                    # 3. Else deny.
                    if paired_ok:
                        partner_info = partner  # type: ignore[assignment]
                        (
                            partner_dgd,
                            partner_sub,
                            partner_desired,
                            _,
                        ) = partner  # type: ignore[misc]
                        pair_scope = (
                            "intra-DGD" if partner_dgd == request_key else "cross-DGD"
                        )
                        logger.info(
                            f"Paired transfer ({pair_scope}) for DGD "
                            f"{request.graph_deployment_name}: "
                            f"request {sorted(request_pool_keys)} + partner "
                            f"{partner_dgd}/{partner_sub}={partner_desired}; "
                            f"total {total_paired} GPUs (bounds "
                            f"[{self.min_total_gpus if self.min_total_gpus >= 0 else '-inf'} - {paired_tolerance}, "
                            f"{self.max_total_gpus if self.max_total_gpus >= 0 else '+inf'} + {paired_tolerance}])"
                        )
                    elif standalone_ok:
                        logger.info(
                            f"Standalone scale request for DGD {request.graph_deployment_name}: "
                            f"total {total_standalone} GPUs "
                            f"(internally_paired={internally_paired})"
                        )
                    else:
                        # Budget breach and no feasible pair. (Reachable only
                        # when standalone is out-of-bounds and either no partner
                        # was found or pairing would also be out-of-bounds.)
                        deny_reason = standalone_reason
                        if partner is not None:
                            deny_reason = (
                                f"{standalone_reason}; paired with "
                                f"{partner[0]}/{partner[1]} would be: {paired_reason}"
                            )
                        logger.warning(
                            f"Rejecting scale request from {request.caller_namespace}: "
                            f"{deny_reason}"
                        )
                        yield {
                            "status": ScaleStatus.ERROR.value,
                            "message": f"GPU budget breach: {deny_reason}",
                            "current_replicas": {},
                        }
                        return

                # Execute the request's own targets on its connector.
                # If partner is in the same DGD, combine into one call.
                # If partner is in a different DGD, issue two calls (not atomic).
                if partner_info is not None and partner_info[0] == request_key:
                    # Intra-DGD: single atomic call.
                    combined_targets: list[TargetReplica] = list(
                        request.target_replicas
                    )
                    _, partner_sub, partner_desired, _ = partner_info
                    combined_targets.append(
                        TargetReplica(
                            sub_component_type=SubComponentType(partner_sub),
                            desired_replicas=partner_desired,
                        )
                    )
                    await connector.set_component_replicas(
                        combined_targets, blocking=request.blocking
                    )
                elif partner_info is None:
                    # No pair: just apply the request.
                    await connector.set_component_replicas(
                        list(request.target_replicas), blocking=request.blocking
                    )
                else:
                    # Cross-DGD pair: apply the scale-DOWN side first so its
                    # GPUs are freed before the scale-UP side submits new pods.
                    # Without this ordering, under a tight ceiling new pods
                    # can sit Pending waiting for the eventual down-patch.
                    partner_dgd, partner_sub, partner_desired, _ = partner_info
                    partner_connector = self.connectors.get(partner_dgd)
                    partner_targets = [
                        TargetReplica(
                            sub_component_type=SubComponentType(partner_sub),
                            desired_replicas=partner_desired,
                        )
                    ]
                    request_targets = list(request.target_replicas)

                    # Partner's direction is opposite of request's net delta by
                    # construction of _find_pair_partner. If request is net
                    # scale-down (net_delta < 0), request is the down side;
                    # otherwise the partner is.
                    first_connector: Optional[KubernetesConnector]
                    second_connector: Optional[KubernetesConnector]
                    if net_delta < 0:
                        first_label = f"request side ({request_key})"
                        second_label = f"partner side ({partner_dgd}/{partner_sub})"
                        first_connector, first_targets = connector, request_targets
                        second_connector, second_targets = (
                            partner_connector,
                            partner_targets,
                        )
                    else:
                        first_label = f"partner side ({partner_dgd}/{partner_sub})"
                        second_label = f"request side ({request_key})"
                        first_connector, first_targets = (
                            partner_connector,
                            partner_targets,
                        )
                        second_connector, second_targets = connector, request_targets

                    if first_connector is None:
                        logger.error(
                            f"Cross-DGD pair failed before first patch: "
                            f"no connector for {first_label}; nothing applied."
                        )
                        # Skip both — safer to self-correct from unchanged state.
                    else:
                        # First patch: let exceptions propagate (outer handler
                        # reports the error; nothing has been applied yet).
                        await first_connector.set_component_replicas(
                            first_targets, blocking=request.blocking
                        )
                        # Second patch: scale-up side may fail independently.
                        # We've already freed (or claimed) GPUs on the first
                        # side; log loudly so operators can spot the drift.
                        if second_connector is None:
                            logger.error(
                                f"Cross-DGD pair second-patch failed: no "
                                f"connector for {second_label}; first-patch "
                                f"({first_label}) already applied. System "
                                f"will self-correct from new state."
                            )
                        else:
                            try:
                                await second_connector.set_component_replicas(
                                    second_targets, blocking=request.blocking
                                )
                            except Exception as pair_err:
                                logger.error(
                                    f"Cross-DGD pair second-patch failed "
                                    f"({second_label}): {pair_err}; first-patch "
                                    f"({first_label}) already applied. System "
                                    f"will self-correct from new state."
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
