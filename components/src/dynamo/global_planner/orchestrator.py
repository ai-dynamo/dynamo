# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator for the GlobalPlanner capacity control loop.

:class:`Orchestrator` is the application core and a deliberate **no-Kubernetes
zone**. It owns the participant registry and authorization, the GPU-budget
arbitration (ceiling/floor bounds, multi-partner pairing, intent cache), and the
observe -> decide -> scale loop, delegating the infrastructure reads/writes to
a :class:`~dynamo.global_planner.capacity_manager.CapacityManager`. Its vocabulary
is neutral — ``participant_id`` / ``caller_name`` / ``deployment_name`` /
``namespace`` — and it imports no Kubernetes SDK and no transport types. The
API layer maps wire fields onto these neutral names; a concrete backend maps them
back onto its infrastructure.

Time is injected (``now``) so offline replay can drive the intent-cache TTL on
sim-time. The serialization lock is toggleable (``use_lock``, default ``True``):
under the async runtime endpoint it must stay on; a single-threaded replay
coordinator can disable it, keeping the core free of concurrency assumptions.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterator, Optional, TypeVar
from weakref import WeakValueDictionary

from dynamo.common.async_utils import run_async_to_completion, run_sync_in_thread
from dynamo.global_planner.capacity_manager import (
    CapacityManager,
    PoolSnapshot,
    PoolSpec,
)
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import ReplicaUpdateResult
from dynamo.planner.connectors.protocol import ScaleStatus
from dynamo.planner.core import budget

# Planner-level "not ready" signal a backend may raise from ``scale``; the
# orchestrator treats it as a soft rejection. Not a Kubernetes SDK type.
from dynamo.planner.errors import (
    DynamoGraphDeploymentNotReadyError,
    EmptyTargetReplicasError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PoolIntent:
    """Most recently observed desired replica count for a pool."""

    last_desired: int
    last_seen_at: float


@dataclass
class PartnerTransfer:
    """One pool selected to change alongside a paired transfer.

    Apply ``applied_desired`` replicas to the ``sub_type`` pool of
    ``participant_id``; ``spec`` is that pool's observed state.
    """

    participant_id: str
    sub_type: str
    applied_desired: int
    spec: PoolSpec


@dataclass
class MediationResult:
    """Outcome of a budget-arbitration decision.

    ``approved`` requests apply the requesting pool's targets plus any
    ``selected_partners`` (empty for a standalone apply). A rejected result
    carries the soft-denial ``reject_message`` the caller returns unchanged.
    """

    approved: bool
    # Partners to apply alongside the request. Empty means "standalone /
    # no-budget path".
    selected_partners: list[PartnerTransfer]
    reject_message: Optional[str] = None


@dataclass
class ScaleOutcome:
    """Result of an orchestrated scale request (infra-agnostic).

    The API layer serializes this into the wire ``ScaleResponse``.
    """

    status: ScaleStatus
    message: str
    current_replicas: dict = field(default_factory=dict)


class _ParticipantMutationBusy(Exception):
    """Signal that a prior response still owns one participant readback."""

    def __init__(self, lock: asyncio.Lock):
        self.lock = lock

    async def wait(self) -> None:
        await self.lock.acquire()
        self.lock.release()


class Orchestrator:
    """Coordinates GPU-budget arbitration and execution across participants.

    Budget enforcement:
    - ``max_total_gpus`` is a ceiling; scale-ups that would exceed it are
      rejected unless a cached opposite-direction intent can be paired.
    - ``min_total_gpus`` is a floor; scale-downs that would drop below it are
      denied unless a cached opposite-direction intent from another participant
      can be paired with them (intra- or cross-participant).
    - Paired transfers may land up to ``tolerance`` GPUs **below** ``min``, where
      tolerance = max per-replica GPU across the pools actually being paired.
      ``max`` is a hard cluster-capacity bound and is never relaxed.

    Intent cache semantics:
    - Every request seeds the per-pool intent cache *before* the budget decision,
      so a denied request still leaves its desired count behind for a later
      opposite-direction request to pair against.
    - Bounded by TTL (``intent_cache_ttl_seconds``) and by the
      satisfied-vs-pending check (``last_desired != current_replicas``).
    """

    def __init__(
        self,
        capacity_manager: CapacityManager,
        managed_deployments: Optional[list],
        max_total_gpus: int = -1,
        min_total_gpus: int = -1,
        intent_cache_ttl_seconds: float = 360.0,
        now: Callable[[], float] = time.time,
        use_lock: bool = True,
    ):
        self.capacity_manager = capacity_manager

        # TODO(global-planner): Validate configuration and requests at the
        # boundary: reject min > max and invalid TTLs, preserve an explicit empty
        # allowlist instead of treating it as None, and reject empty, duplicate,
        # unknown, or negative targets before caching, mediation, or execution.
        # Budget bounds + intent cache.
        self.max_total_gpus = max_total_gpus
        self.min_total_gpus = min_total_gpus
        self.intent_cache_ttl_seconds = intent_cache_ttl_seconds
        # Injected clock — production uses wall-clock; offline replay passes a
        # sim-time source so intent-cache TTL is meaningful there too.
        self._now = now
        # TODO(global-planner): Replace process-local discovery,
        # participant/intent state, and locking with fail-closed cluster-wide
        # coordination, and register participants inside the same serialized
        # boundary as observe/decide/apply so the hard GPU ceiling always uses a
        # complete snapshot.
        # Per-pool cached desired replicas from recent requests, keyed by
        # f"{participant_id}/{sub_type}". Used to pair opposite-direction intents
        # across requests when one request alone would breach bounds.
        self._intent_cache: dict[str, PoolIntent] = {}

        # Authorization: the set of managed deployment identities allowed to send
        # scale requests. None/empty accepts all callers.
        self.managed_deployments = (
            set(managed_deployments) if managed_deployments else None
        )

        # Serializes budget-check + scale-execution so concurrent requests from
        # different pools cannot both pass against the same pre-scale state.
        # Toggleable: a single-threaded replay coordinator can disable it.
        self._scale_lock: Optional[asyncio.Lock] = asyncio.Lock() if use_lock else None
        self._participant_mutation_locks: WeakValueDictionary[
            str, asyncio.Lock
        ] = WeakValueDictionary()

    # ------------------------------------------------------------------ #
    # Participant registry / authorization                               #
    # ------------------------------------------------------------------ #

    def is_authorized(self, caller_name: str) -> bool:
        """Whether ``caller_name`` may send scale requests.

        In implicit mode (no managed callers) every caller is authorized.
        """
        if (
            self.managed_deployments is not None
            and caller_name not in self.managed_deployments
        ):
            return False
        return True

    def register(
        self,
        participant_id: str,
        caller_name: str,
        namespace: str,
        deployment_name: str,
    ) -> None:
        """Ensure the participant exists in the capacity backend (idempotent).

        ``caller_name`` / ``namespace`` / ``deployment_name`` are forwarded
        opaquely to the backend, which uses them to construct its participant.
        """
        self.capacity_manager.ensure_participant(
            participant_id,
            caller_name=caller_name,
            namespace=namespace,
            deployment_name=deployment_name,
        )

    # ------------------------------------------------------------------ #
    # Startup                                                            #
    # ------------------------------------------------------------------ #

    def startup(self) -> None:
        """Discover existing participants and warn if the initial total GPUs are
        below the floor. Only runs when budget enforcement is enabled."""
        if not self.budget_enforcement_enabled():
            return
        # TODO(global-planner): Add ongoing lifecycle reconciliation after
        # startup: discover new participants, evict deleted participants,
        # connectors, and role hints, and prune expired or satisfied intents
        # instead of only skipping them.
        self.capacity_manager.discover(self.managed_deployments)
        if self.min_total_gpus >= 0:
            self._warn_if_below_floor()

    def _warn_if_below_floor(self) -> None:
        """Log a warning if the discovered initial state is below min_total_gpus.

        Soft floor: we do not proactively scale up. The floor prevents
        scale-downs below it, but initial below-floor state is allowed and will
        drift toward the floor as load arrives.
        """
        try:
            total = self.total_gpus(
                self.capacity_manager.observe(
                    require_complete=self.budget_enforcement_enabled()
                )
            )
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

    # ------------------------------------------------------------------ #
    # observe -> decide -> scale                                       #
    # ------------------------------------------------------------------ #

    async def _run_with_conflict_retry(
        self, operation: Callable[[], Awaitable[T]]
    ) -> T:
        retry = getattr(self.capacity_manager, "run_with_conflict_retry", None)
        if retry is None:
            return await operation()
        return await retry(operation)

    def _capture_current_replicas(
        self, participant_id: str
    ) -> Callable[[], dict[str, int]]:
        capture = getattr(self.capacity_manager, "capture_current_replicas", None)
        if capture is not None:
            return capture(participant_id)
        return lambda: self.capacity_manager.current_replicas(participant_id)

    def _supports_split_readiness(self) -> bool:
        supports = getattr(self.capacity_manager, "supports_split_readiness", None)
        return bool(supports()) if supports is not None else False

    def _capture_request_intents(
        self, participant_id: str, targets: list[TargetReplica]
    ) -> dict[str, PoolIntent | None]:
        """Snapshot request-owned cache entries until the plan is admitted."""
        return {
            self._pool_cache_key(participant_id, target.sub_component_type.value): (
                self._intent_cache.get(
                    self._pool_cache_key(
                        participant_id, target.sub_component_type.value
                    )
                )
            )
            for target in targets
        }

    def _restore_request_intents(self, previous: dict[str, PoolIntent | None]) -> None:
        for key, intent in previous.items():
            if intent is None:
                self._intent_cache.pop(key, None)
            else:
                self._intent_cache[key] = intent

    @staticmethod
    def _target_pool(
        pools: dict[str, PoolSpec], target: TargetReplica
    ) -> PoolSpec | None:
        role_pool = pools.get(target.sub_component_type.value)
        if role_pool is not None:
            if (
                target.component_name
                and role_pool.component_name
                and target.component_name != role_pool.component_name
            ):
                raise ValueError(
                    f"component {target.component_name!r} does not match observed "
                    f"{target.sub_component_type.value} pool "
                    f"{role_pool.component_name!r}"
                )
            return role_pool
        if target.component_name:
            for pool in pools.values():
                if pool.component_name == target.component_name:
                    if pool.sub_type != target.sub_component_type.value:
                        raise ValueError(
                            f"component {target.component_name!r} belongs to observed "
                            f"{pool.sub_type} pool, not requested "
                            f"{target.sub_component_type.value} pool"
                        )
                    return pool
        return None

    def _budget_target_stages(
        self,
        grouped_targets: dict[str, list[TargetReplica]],
        all_pools: PoolSnapshot,
    ) -> list[dict[str, list[TargetReplica]]]:
        """Split a hard-budget plan into settled reductions, checks, and increases."""
        reductions: dict[str, list[TargetReplica]] = defaultdict(list)
        verifications: dict[str, list[TargetReplica]] = defaultdict(list)
        increases: dict[str, list[TargetReplica]] = defaultdict(list)
        for participant_id, targets in grouped_targets.items():
            pools = all_pools.get(participant_id, {})
            for target in targets:
                pool = self._target_pool(pools, target)
                if pool is None:
                    raise RuntimeError(
                        f"accepted target for {participant_id} has no observed pool"
                    )
                if target.desired_replicas < pool.current_replicas:
                    reductions[participant_id].append(target)
                elif target.desired_replicas > pool.current_replicas:
                    increases[participant_id].append(target)
                else:
                    verifications[participant_id].append(target)
        return [stage for stage in (reductions, verifications, increases) if stage]

    def _validate_budget_targets(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        all_pools: PoolSnapshot,
    ) -> None:
        """Reject targets that resolve to the same observed pool."""
        pools = all_pools.get(participant_id, {})
        seen: set[str] = set()
        for target in targets:
            pool = self._target_pool(pools, target)
            if pool is None:
                continue
            pool_id = pool.component_name or target.sub_component_type.value
            if pool_id in seen:
                raise ValueError(f"duplicate replica target for pool {pool_id!r}")
            seen.add(pool_id)

    async def _scale_with_result(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
    ) -> ReplicaUpdateResult:
        scale_with_result = getattr(self.capacity_manager, "scale_with_result", None)
        if scale_with_result is not None:
            result = await scale_with_result(participant_id, targets, blocking)
        else:
            result = await self.capacity_manager.scale(
                participant_id, targets, blocking
            )
        if isinstance(result, ReplicaUpdateResult):
            return result
        if result is None:
            return ReplicaUpdateResult(accepted=True, applied=True)
        accepted = bool(result)
        return ReplicaUpdateResult(accepted=accepted, applied=accepted)

    def _is_conflict(self, error: Exception) -> bool:
        is_conflict = getattr(self.capacity_manager, "is_conflict", None)
        return bool(is_conflict(error)) if is_conflict is not None else False

    async def submit(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
        deployment_name: str,
        caller_name: str,
    ) -> ScaleOutcome:
        """Arbitrate and execute one scale request.

        The observe -> decide -> scale span is serialized under the scale lock
        (when enabled) so concurrent requests cannot both pass against the same
        pre-scale state. Backend conflict retries repeat this complete span so
        every write is mediated against its fresh observation. Other patch
        failures propagate to the caller (the API layer maps them to REJECTED /
        ERROR).
        """

        if not targets:
            raise EmptyTargetReplicasError()

        admitted = asyncio.Event()
        committed_participants: list[str] = []
        held_participant_locks: dict[str, asyncio.Lock] = {}
        transaction_attempts = 0

        def complete_after_cancel() -> bool:
            return admitted.is_set()

        async def operation() -> Optional[ScaleOutcome]:
            nonlocal transaction_attempts
            transaction_attempts += 1
            return await self._observe_decide_apply(
                participant_id,
                targets,
                blocking,
                deployment_name,
                caller_name,
                admitted,
                committed_participants,
                held_participant_locks,
            )

        async def transaction() -> Optional[ScaleOutcome]:
            nonlocal transaction_attempts
            transaction_attempts = 0
            try:
                return await self._run_with_conflict_retry(operation)
            except Exception as error:
                if committed_participants:
                    return self._partial_plan_error(
                        participant_id,
                        committed_participants,
                        f"transaction retry failed: {error}",
                    )[0]
                raise

        async def execute_transaction() -> tuple[
            Optional[ScaleOutcome], Callable[[], dict[str, int]] | None
        ]:
            terminal = await transaction()
            if terminal is not None:
                return terminal, None

            # Freeze backend identity under the transaction lock, but do not
            # hold the global lock across the potentially unbounded read GET.
            return None, self._capture_current_replicas(participant_id)

        async def execute_serialized_transaction() -> tuple[
            Optional[ScaleOutcome], Callable[[], dict[str, int]] | None
        ]:
            while True:
                try:
                    if self._scale_lock is not None:
                        async with self._scale_lock:
                            return await execute_transaction()
                    return await execute_transaction()
                except _ParticipantMutationBusy as busy:
                    if transaction_attempts > 1:
                        raise RuntimeError(
                            "participant set changed during conflict retry; "
                            "refusing to restart without the retained identity fence"
                        ) from busy
                    await busy.wait()

        try:
            terminal, readback = await run_async_to_completion(
                execute_serialized_transaction(),
                complete_on_cancel=complete_after_cancel,
            )
            if terminal is not None:
                return terminal
            assert readback is not None

            # The captured readback depends only on the requester.
            for locked_participant_id in tuple(held_participant_locks):
                if locked_participant_id == participant_id:
                    continue
                held_participant_locks.pop(locked_participant_id).release()

            # Preserve the response contract: current_replicas is an observed
            # post-transaction snapshot, not the requested desired state.
            current_replicas = await run_sync_in_thread(readback)
            logger.info(f"Successfully scaled {deployment_name}: {current_replicas}")
            return ScaleOutcome(
                status=ScaleStatus.SUCCESS,
                message=f"Scaled {deployment_name} successfully",
                current_replicas=current_replicas,
            )
        finally:
            for participant_lock in reversed(held_participant_locks.values()):
                participant_lock.release()

    async def _reserve_participant_mutations(
        self,
        participant_ids: list[str],
        held_participant_locks: dict[str, asyncio.Lock],
    ) -> None:
        """Reserve participants through settlement.

        The requester remains reserved through its response readback.
        """
        pending: list[tuple[str, asyncio.Lock]] = []
        for participant_id in sorted(set(participant_ids)):
            if participant_id in held_participant_locks:
                continue
            participant_lock = self._participant_mutation_locks.setdefault(
                participant_id, asyncio.Lock()
            )
            if participant_lock.locked():
                raise _ParticipantMutationBusy(participant_lock)
            pending.append((participant_id, participant_lock))

        for participant_id, participant_lock in pending:
            await participant_lock.acquire()
            held_participant_locks[participant_id] = participant_lock

    async def _observe_decide_apply(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        blocking: bool,
        deployment_name: str,
        caller_name: str,
        admitted: asyncio.Event,
        committed_participants: list[str],
        held_participant_locks: dict[str, asyncio.Lock],
    ) -> Optional[ScaleOutcome]:
        """Observe all pools, decide the budget outcome, and apply it.

        Returns the result from the serialized observation and accepted absolute
        targets. May raise a patch error from the first applied participant
        (propagated to the API layer).
        """
        # Let the backend record any request context it needs to read pool state
        # correctly (e.g. component-name role hints for generic workers).
        self.capacity_manager.remember_roles(participant_id, targets)

        # Cross-participant mediation needs a complete snapshot only when a
        # budget is active. The unbounded mode approves without pool state and
        # performs a targeted post-write readback for the response.
        if self.budget_enforcement_enabled():
            reconcile_pending = getattr(
                self.capacity_manager, "reconcile_pending_settlements", None
            )
            if reconcile_pending is not None:
                await reconcile_pending()
            all_pools = await run_sync_in_thread(self.capacity_manager.observe, True)
        else:
            refresh_participant = getattr(
                self.capacity_manager, "refresh_participant", None
            )
            if refresh_participant is not None:
                await run_sync_in_thread(refresh_participant, participant_id)
            reconcile_pending = getattr(
                self.capacity_manager, "reconcile_pending_settlements", None
            )
            if reconcile_pending is not None:
                await reconcile_pending(participant_id)
            all_pools = {}

        previous_intents = self._capture_request_intents(participant_id, targets)
        if self.budget_enforcement_enabled():
            self._validate_budget_targets(participant_id, targets, all_pools)
        result = self.mediate(
            participant_id,
            targets,
            all_pools,
            log_deployment_name=deployment_name,
            log_caller_name=caller_name,
        )
        if not result.approved:
            if committed_participants:
                return self._partial_plan_error(
                    participant_id,
                    committed_participants,
                    result.reject_message
                    or "fresh budget mediation rejected the retry",
                )[0]
            # Soft denial: budget breach is an expected operational outcome in
            # fixed-total mode, not a fault.
            return ScaleOutcome(
                status=ScaleStatus.REJECTED,
                message=result.reject_message or "",
                current_replicas={},
            )
        # Apply: request + selected partners (may be empty), grouped by
        # participant with at most one scale call per participant.
        # Budgeted plans are split below so every reduction settles before any
        # increase and the final stage settles before serialization is released.
        grouped_targets: dict[str, list[TargetReplica]] = defaultdict(list)
        grouped_targets[participant_id].extend(targets)
        for partner in result.selected_partners:
            grouped_targets[partner.participant_id].append(
                TargetReplica(
                    sub_component_type=SubComponentType(partner.sub_type),
                    component_name=partner.spec.component_name or None,
                    desired_replicas=partner.applied_desired,
                )
            )

        # Compute net GPU delta per participant for ordering.
        net_deltas: dict[str, int] = {}
        for pid, tgts in grouped_targets.items():
            pools = all_pools.get(pid, {})
            net = 0
            for t in tgts:
                spec = self._target_pool(pools, t)
                if spec is not None and spec.gpu_per_replica > 0:
                    net += (
                        t.desired_replicas - spec.current_replicas
                    ) * spec.gpu_per_replica
            net_deltas[pid] = net

        # Sort: most negative (scale-down) first, most positive (scale-up) last.
        ordered_participants = sorted(
            grouped_targets.keys(), key=lambda k: net_deltas[k]
        )

        # Cancellation remains safe until every participant selected by this
        # observation is reserved. Once the locks are held, completion must be
        # shielded because the first Kubernetes write can commit immediately.
        try:
            await self._reserve_participant_mutations(
                ordered_participants, held_participant_locks
            )
        except BaseException:
            self._restore_request_intents(previous_intents)
            raise
        admitted.set()

        stages = (
            self._budget_target_stages(grouped_targets, all_pools)
            if self.budget_enforcement_enabled()
            else [grouped_targets]
        )
        for stage in stages:
            stage_order = [
                participant
                for participant in ordered_participants
                if participant in stage
            ]
            terminal, _ = await self._apply_approved_targets(
                stage,
                stage_order,
                blocking=blocking,
                settle=self.budget_enforcement_enabled(),
                committed_participants=committed_participants,
                held_participant_locks=held_participant_locks,
            )
            if terminal is not None:
                return terminal
        return None

    async def _apply_approved_targets(
        self,
        grouped_targets: dict[str, list[TargetReplica]],
        ordered_participants: list[str],
        *,
        blocking: bool,
        settle: bool,
        committed_participants: list[str],
        held_participant_locks: dict[str, asyncio.Lock],
    ) -> tuple[Optional[ScaleOutcome], list[str]]:
        """Apply one admitted stage and settle it when capacity depends on it."""

        await self._reserve_participant_mutations(
            ordered_participants, held_participant_locks
        )

        # TODO(global-planner): Persist enough state to repair or roll back a
        # plan that still becomes partial because a later participant fails.
        completed: list[str] = []
        for pid in ordered_participants:
            tgts = grouped_targets[pid]
            if not self.capacity_manager.participant_exists(pid):
                if not committed_participants:
                    logger.error(
                        f"Multi-partner transfer aborted: unknown participant ({pid})"
                    )
                    return (
                        ScaleOutcome(
                            status=ScaleStatus.ERROR,
                            message=f"Multi-partner transfer: unknown participant {pid}",
                            current_replicas={},
                        ),
                        completed,
                    )
                logger.error(
                    f"Multi-partner transfer: unknown participant {pid} "
                    f"after applying {committed_participants}; "
                    "stopping plan until the next tick"
                )
                return self._partial_plan_error(
                    pid, committed_participants, "participant is unknown"
                )
            try:
                split_readiness = self._supports_split_readiness()
                result = await self._scale_with_result(
                    pid,
                    tgts,
                    blocking=(blocking or settle) and not split_readiness,
                )
            except DynamoGraphDeploymentNotReadyError as patch_err:
                if not committed_participants:
                    raise
                logger.warning(
                    "Multi-partner transfer: patch on %s was skipped "
                    "after applying %s because the participant is not ready: %s; "
                    "stopping plan until the next tick",
                    pid,
                    committed_participants,
                    patch_err,
                )
                return self._partial_plan_error(
                    pid,
                    committed_participants,
                    f"participant is not ready: {patch_err}",
                )
            except Exception as patch_err:
                if self._is_conflict(patch_err):
                    raise
                if not committed_participants:
                    # No write has completed, so preserve the original failure
                    # for the API-layer handler.
                    raise
                logger.error(
                    f"Multi-partner transfer: patch on {pid} "
                    f"failed after applying {committed_participants}: "
                    f"{patch_err}; stopping plan until the next tick"
                )
                return self._partial_plan_error(
                    pid, committed_participants, str(patch_err)
                )
            if not result.accepted:
                logger.warning(
                    "Multi-partner transfer: participant %s did not accept its "
                    "target after applying %s; stopping plan until the next tick",
                    pid,
                    committed_participants,
                )
                if committed_participants:
                    return self._partial_plan_error(
                        pid,
                        committed_participants,
                        "participant did not accept its scale target",
                    )
                return (
                    ScaleOutcome(
                        status=ScaleStatus.REJECTED,
                        message=f"Participant {pid} did not accept its scale target",
                        current_replicas={},
                    ),
                    completed,
                )
            completed.append(pid)
            if result.applied and pid not in committed_participants:
                committed_participants.append(pid)
            if (blocking or settle or result.requires_settlement) and split_readiness:
                try:
                    await self.capacity_manager.wait_for_ready(pid)
                except Exception as readiness_error:
                    if not committed_participants:
                        raise
                    logger.error(
                        "Multi-partner transfer: readiness wait on %s failed "
                        "after applying %s: %s; stopping plan until the next tick",
                        pid,
                        committed_participants,
                        readiness_error,
                    )
                    return self._partial_plan_error(
                        pid,
                        committed_participants,
                        f"readiness wait failed: {readiness_error}",
                    )

        return None, completed

    @staticmethod
    def _partial_plan_error(
        participant_id: str, applied: list[str], reason: str
    ) -> tuple[ScaleOutcome, list[str]]:
        """Report a stopped plan after at least one participant was changed."""
        return (
            ScaleOutcome(
                status=ScaleStatus.ERROR,
                message=(
                    f"Scale plan stopped at {participant_id} after applying "
                    f"{applied}: {reason}"
                ),
                current_replicas={},
            ),
            applied,
        )

    # ------------------------------------------------------------------ #
    # Budget arbitration (decision)                                      #
    # ------------------------------------------------------------------ #

    def budget_enforcement_enabled(self) -> bool:
        return self.max_total_gpus >= 0 or self.min_total_gpus >= 0

    def total_gpus(
        self,
        all_pools: PoolSnapshot,
        overrides: Optional[dict[tuple[str, str], int]] = None,
    ) -> int:
        """Total GPUs across all known participants from a snapshot."""
        return self._total_gpus_from_snapshot(all_pools, overrides or {})

    def mediate(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        all_pools: PoolSnapshot,
        log_deployment_name: str = "",
        log_caller_name: str = "",
    ) -> MediationResult:
        """Decide whether ``targets`` for ``participant_id`` fit the budget.

        Always updates the intent cache first (so a denied request can still be
        paired against later). Returns an approved result (standalone or with
        selected partners) or a soft rejection with a budget-breach message. The
        ``log_*`` values are used only for parity-identical log lines.
        """
        pools = all_pools.get(participant_id, {})

        # Always update the intent cache with this request's targets, regardless
        # of decision. A later request from a complementary pool may need to pair
        # with this intent.
        self.update_intent_cache(participant_id, targets, pools)

        request_pool_keys = {
            (participant_id, t.sub_component_type.value) for t in targets
        }
        standalone_overrides = {
            (participant_id, t.sub_component_type.value): t.desired_replicas
            for t in targets
        }

        if not self.budget_enforcement_enabled():
            return MediationResult(approved=True, selected_partners=[])

        net_delta = self.net_delta_gpu(targets, pools)
        internally_paired = self.is_internally_paired(targets, pools)

        total_standalone = self._total_gpus_from_snapshot(
            all_pools, standalone_overrides
        )

        # Tolerance depends on which pools are actually changing.
        changing_request_pools = [
            pools[t.sub_component_type.value]
            for t in targets
            if t.sub_component_type.value in pools
            and t.desired_replicas != pools[t.sub_component_type.value].current_replicas
        ]
        standalone_tolerance = self._internal_pair_tolerance(changing_request_pools)

        # Internally-paired requests get tolerance even without an external
        # partner.
        standalone_ok, standalone_reason = self._bounds_for_total(
            total_standalone, internally_paired, standalone_tolerance
        )

        # TODO(global-planner): Rework boundary and partner selection to allow
        # monotonic recovery from an already-out-of-bounds state, compute
        # tolerance only from selected pools, continue after an unpartializable
        # candidate, prefer an already-valid standalone request, and preserve
        # same-participant priority during sorting.
        # Multi-partner packing: pack as many opposite-direction cached intents
        # as fit within the band, partially consuming one over-sized candidate if
        # needed.
        (
            selected_partners,
            total_paired,
            paired_tolerance,
        ) = self._find_pair_partner_set(
            participant_id,
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
                "intra-participant"
                if all(p.participant_id == participant_id for p in selected_partners)
                else "cross-participant"
            )
            partners_desc = ", ".join(
                f"{p.participant_id}/{p.sub_type}={p.applied_desired}"
                for p in selected_partners
            )
            logger.info(
                f"Paired transfer ({scope}, "
                f"{len(selected_partners)} partner(s)) for "
                f"{log_deployment_name}: "
                f"request {sorted(request_pool_keys)} + "
                f"[{partners_desc}]; total {total_paired} GPUs "
                f"(bounds "
                f"[{self.min_total_gpus if self.min_total_gpus >= 0 else '-inf'} - {paired_tolerance}, "
                f"{self.max_total_gpus if self.max_total_gpus >= 0 else '+inf'}])"
            )
            return MediationResult(approved=True, selected_partners=selected_partners)
        elif standalone_ok:
            logger.info(
                f"Standalone scale request for {log_deployment_name}: "
                f"total {total_standalone} GPUs "
                f"(internally_paired={internally_paired})"
            )
            return MediationResult(approved=True, selected_partners=[])
        else:
            # Budget breach: standalone out-of-bounds and no feasible partner set
            # found.
            logger.warning(
                f"Rejecting scale request from {log_caller_name}: "
                f"{standalone_reason}; no feasible pair packing"
            )
            # Soft denial: budget breach is an expected operational outcome in
            # fixed-total mode, not a fault. Local planners should treat this as a
            # no-op for this tick.
            return MediationResult(
                approved=False,
                selected_partners=[],
                reject_message=(
                    f"GPU budget breach: {standalone_reason}; no feasible pair packing"
                ),
            )

    def _total_gpus_from_snapshot(
        self,
        all_pools: PoolSnapshot,
        overrides: dict[tuple[str, str], int],
    ) -> int:
        """Compute total GPUs across all known participants from a snapshot.

        ``overrides`` maps ``(participant_id, sub_type)`` to the replica count to
        use in place of the current count. Entries not in ``overrides`` use the
        current replica count.
        """
        total_gpus = 0
        for key, pools in all_pools.items():
            for sub_type, spec in pools.items():
                if spec.gpu_per_replica == 0:
                    continue
                replicas = overrides.get((key, sub_type), spec.current_replicas)
                total_gpus += replicas * spec.gpu_per_replica
        return total_gpus

    # ------------------------------------------------------------------ #
    # Intent cache helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pool_cache_key(participant_id: str, sub_type: str) -> str:
        return f"{participant_id}/{sub_type}"

    @staticmethod
    def _direction(desired: int, current: int) -> str:
        if desired > current:
            return "up"
        if desired < current:
            return "down"
        return "stable"

    def update_intent_cache(
        self,
        participant_id: str,
        targets: list[TargetReplica],
        pools: dict[str, PoolSpec],
    ):
        """Record the desired replicas for each pool in this request."""
        now = self._now()
        for target in targets:
            sub_type = target.sub_component_type.value
            if sub_type not in pools:
                # Unknown pool (not yet observed); skip — without gpu_per_replica
                # we can't compute deltas or pair against it.
                continue
            key = self._pool_cache_key(participant_id, sub_type)
            self._intent_cache[key] = PoolIntent(
                last_desired=target.desired_replicas,
                last_seen_at=now,
            )

    # ------------------------------------------------------------------ #
    # Tolerance                                                          #
    # ------------------------------------------------------------------ #

    def _pair_tolerance(
        self,
        request_pools: list[PoolSpec],
        partner_spec: PoolSpec,
    ) -> int:
        """Tolerance for a specific paired transfer.

        Equal to max per-replica GPU across just the pools actually being changed
        (request's non-stable pools + partner). Covers step-size asymmetry where a
        single worker on one side can't exactly cancel a single worker on the
        other side.
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

    # ------------------------------------------------------------------ #
    # Pair-partner search                                                #
    # ------------------------------------------------------------------ #

    def _iter_pair_partners(
        self,
        request_participant_id: str,
        request_pool_keys: set[tuple[str, str]],
        all_pools: PoolSnapshot,
        request_net_delta_gpu: int,
    ) -> Iterator[PartnerTransfer]:
        """Yield qualifying pair-partner candidates, same-participant first.

        A candidate qualifies when it (a) is not in the requesting pool set, (b)
        has a fresh cached intent (within TTL) whose desired differs from current
        replicas, and (c) the partner delta points opposite to the request's net
        delta.

        Yields in two passes: same-participant candidates first (atomic-patch
        preference), then cross-participant candidates. The caller picks the first
        one whose pair total actually lands in the budget band.
        """
        if request_net_delta_gpu == 0:
            return
        now = self._now()
        same: list[PartnerTransfer] = []
        cross: list[PartnerTransfer] = []
        for pid, pools in all_pools.items():
            for sub_type, spec in pools.items():
                if (pid, sub_type) in request_pool_keys:
                    continue
                if spec.gpu_per_replica == 0:
                    continue
                cache_key = self._pool_cache_key(pid, sub_type)
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
                candidate = PartnerTransfer(pid, sub_type, intent.last_desired, spec)
                if pid == request_participant_id:
                    same.append(candidate)
                else:
                    cross.append(candidate)
        yield from same
        yield from cross

    def _partial_partner(
        self,
        candidate: PartnerTransfer,
        all_pools: PoolSnapshot,
        current_overrides: dict[tuple[str, str], int],
        tolerance: int,
    ) -> Optional[int]:
        """Compute a partial ``applied_desired`` for a candidate whose full
        consumption would push the combined transfer out of the band.

        Returns an integer ``K`` strictly between ``current_replicas`` and
        ``last_desired`` (direction-consistent) such that applying ``K`` instead
        of ``last_desired`` lands the combined transfer at the appropriate band
        edge — the strict ceiling on the upper side (``max``, no tolerance) or
        ``min - tolerance`` on the lower side. ``None`` if no feasible partial
        exists.
        """
        last_desired = candidate.applied_desired
        spec = candidate.spec
        current = spec.current_replicas
        gpu = spec.gpu_per_replica
        if gpu <= 0 or last_desired == current:
            return None

        # Combined total assuming this candidate stays at its current count (i.e.,
        # contributes 0). The candidate is NOT in current_overrides yet, so the
        # snapshot uses its current_replicas naturally.
        baseline_total = self._total_gpus_from_snapshot(all_pools, current_overrides)

        if last_desired > current:
            # Scale-up candidate: pick K in [current+1, last_desired] that keeps
            # total <= max (strict ceiling — max is a hard hardware bound, see
            # budget.bounds_for_total).
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
            # Scale-down candidate: pick K in [last_desired, current-1] that keeps
            # total >= min - tolerance.
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
        request_participant_id: str,
        request_pool_keys: set[tuple[str, str]],
        all_pools: PoolSnapshot,
        request_net_delta_gpu: int,
        standalone_overrides: dict[tuple[str, str], int],
        changing_request_pools: list[PoolSpec],
    ) -> tuple[list[PartnerTransfer], int, int]:
        """Pack as many opposite-direction cached intents as fit alongside the
        request, partially consuming one over-sized candidate if needed.

        Algorithm: greedy admission, ascending ``abs(delta_gpu)`` order. For each
        candidate, fully admit if it keeps the combined transfer in
        ``[min - tolerance, max]``; if full admission would overshoot the strict
        ceiling, try partial consumption that lands at the band edge and stop.

        Tolerance is computed **once** over the request's changing pools plus all
        candidates considered for inclusion, not iteratively widened.

        Returns ``(selected_partners, total_after, tolerance)``.
        ``selected_partners`` is empty when no feasible packing exists.
        """
        if request_net_delta_gpu == 0:
            return [], 0, 0

        all_candidates = list(
            self._iter_pair_partners(
                request_participant_id,
                request_pool_keys,
                all_pools,
                request_net_delta_gpu,
            )
        )
        if not all_candidates:
            return [], 0, 0

        # Tolerance computed once over the universe of changing pools.
        candidate_specs = [c.spec for c in all_candidates]
        tolerance = budget.compute_tolerance(
            [s.gpu_per_replica for s in changing_request_pools]
            + [s.gpu_per_replica for s in candidate_specs]
        )

        # Sort ascending by |delta_gpu| — smaller pieces overshoot less.
        def cand_delta(c: PartnerTransfer) -> int:
            return (
                c.applied_desired - c.spec.current_replicas
            ) * c.spec.gpu_per_replica

        all_candidates.sort(key=lambda c: abs(cand_delta(c)))

        selected: list[PartnerTransfer] = []
        overrides = dict(standalone_overrides)

        for cand in all_candidates:
            cand_pid, cand_sub, cand_desired, cand_spec = (
                cand.participant_id,
                cand.sub_type,
                cand.applied_desired,
                cand.spec,
            )

            # Try full inclusion.
            full_overrides = dict(overrides)
            full_overrides[(cand_pid, cand_sub)] = cand_desired
            full_total = self._total_gpus_from_snapshot(all_pools, full_overrides)
            in_band, _ = budget.bounds_for_total(
                full_total,
                self.min_total_gpus,
                self.max_total_gpus,
                tolerance,
            )
            if in_band:
                # Full inclusion lands in band — accept and continue. Keep
                # admitting more candidates while feasible.
                selected.append(cand)
                overrides = full_overrides
                continue

            # Out of band. Did this candidate cross the band, or are we still on
            # the wrong side and need more help?
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
                # Candidate moved us toward the band but didn't reach it. Accept
                # full inclusion and try the next candidate.
                selected.append(cand)
                overrides = full_overrides
                continue

            # Full inclusion crossed the band. Try partial consumption that lands
            # at the appropriate band edge, then stop.
            partial_k = self._partial_partner(cand, all_pools, overrides, tolerance)
            if partial_k is not None and partial_k != cand_spec.current_replicas:
                partial_cand = PartnerTransfer(cand_pid, cand_sub, partial_k, cand_spec)
                selected.append(partial_cand)
                overrides[(cand_pid, cand_sub)] = partial_k
            break

        # Loop ended with the running total possibly still on the wrong side (no
        # candidate fully reached the band, none of them crossed). Try partial of
        # the last selected candidate to land in band.
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
                last_pid, last_sub, last_spec = (
                    last.participant_id,
                    last.sub_type,
                    last.spec,
                )
                # Roll back the last full inclusion so partial uses the
                # pre-last-candidate baseline.
                rollback_overrides = dict(overrides)
                if (last_pid, last_sub) in standalone_overrides:
                    rollback_overrides[(last_pid, last_sub)] = standalone_overrides[
                        (last_pid, last_sub)
                    ]
                else:
                    rollback_overrides.pop((last_pid, last_sub), None)
                partial_k = self._partial_partner(
                    last, all_pools, rollback_overrides, tolerance
                )
                if partial_k is not None and partial_k != last_spec.current_replicas:
                    selected[-1] = PartnerTransfer(
                        last_pid, last_sub, partial_k, last_spec
                    )
                    overrides = dict(rollback_overrides)
                    overrides[(last_pid, last_sub)] = partial_k

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
    # Request shape helpers                                              #
    # ------------------------------------------------------------------ #

    def net_delta_gpu(
        self,
        targets: list[TargetReplica],
        pools: dict[str, PoolSpec],
    ) -> int:
        """Sum of (desired - current) * gpu_per_replica across request pools."""
        net = 0
        for target in targets:
            sub_type = target.sub_component_type.value
            spec = pools.get(sub_type)
            if spec is None or spec.gpu_per_replica == 0:
                continue
            net += (
                target.desired_replicas - spec.current_replicas
            ) * spec.gpu_per_replica
        return net

    def is_internally_paired(
        self,
        targets: list[TargetReplica],
        pools: dict[str, PoolSpec],
    ) -> bool:
        """True if the request contains both up and down directions across pools."""
        has_up = False
        has_down = False
        for target in targets:
            sub_type = target.sub_component_type.value
            spec = pools.get(sub_type)
            if spec is None:
                continue
            direction = self._direction(target.desired_replicas, spec.current_replicas)
            if direction == "up":
                has_up = True
            elif direction == "down":
                has_down = True
        return has_up and has_down

    def _bounds_for_total(
        self,
        total: int,
        paired: bool,
        tolerance: int,
    ) -> tuple[bool, str]:
        """Check whether ``total`` is within the active budget bounds.

        Standalone (non-paired) requests use strict bounds; paired transfers get
        the tolerance band on the **lower** edge only — ``max_total_gpus`` is a
        hard cluster-capacity bound (enforced by ``budget.bounds_for_total``).
        """
        return budget.bounds_for_total(
            total,
            self.min_total_gpus,
            self.max_total_gpus,
            tolerance if paired else 0,
        )
