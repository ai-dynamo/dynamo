# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.core.budget import minimum_power_footprint_fits
from dynamo.planner.core.types import FpmObservations, TrafficObservation
from dynamo.planner.environment.interface import (
    PlannerEnvironment,
    RuntimeNamespaceSource,
)
from dynamo.planner.environment.metrics_provider.interface import (
    FpmMetricsProvider,
    TrafficMetricsProvider,
)
from dynamo.planner.environment.state import ComponentState, DeploymentState
from dynamo.planner.errors import DeploymentValidationError
from dynamo.planner.monitoring.dgd_services import ComponentPowerConfig
from dynamo.planner.monitoring.traffic_metrics import Metrics

logger = logging.getLogger(__name__)

_MDC_REFRESH_FIELDS = (
    "total_kv_blocks",
    "kv_cache_block_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "context_length",
    "speculative_nextn",
)


class NoopTrafficMetricsProvider:
    async def collect_traffic(self) -> Optional[TrafficObservation]:
        return None

    def collect_accept_length(self, interval_str: str) -> Optional[float]:
        del interval_str
        return None

    async def collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        del duration_s
        return None


class NoopFpmMetricsProvider:
    async def async_init(self, namespace: Optional[str] = None) -> None:
        del namespace

    async def refresh(self, state: DeploymentState) -> None:
        del state

    def collect_fpm(self) -> FpmObservations:
        return FpmObservations(prefill=None, decode=None)

    async def shutdown(self) -> None:
        return None


class PlannerEnvironmentImpl(PlannerEnvironment):
    """Default environment facade consumed by planner core."""

    def __init__(
        self,
        *,
        config: PlannerConfig,
        controller: PlannerConnector,
        require_prefill: bool,
        require_decode: bool,
        traffic_provider: Optional[TrafficMetricsProvider] = None,
        fpm_provider: Optional[FpmMetricsProvider] = None,
        runtime_namespace_source: Optional[RuntimeNamespaceSource] = None,
    ) -> None:
        self.config = config
        self.require_prefill = require_prefill
        self.require_decode = require_decode
        self.controller = controller
        self.traffic_provider = traffic_provider or NoopTrafficMetricsProvider()
        self.fpm_provider = fpm_provider or NoopFpmMetricsProvider()
        self.runtime_namespace_source = runtime_namespace_source
        self._state = DeploymentState()
        self._metrics_state = Metrics()

    async def initialize(self) -> None:
        await self.controller.async_init()
        defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)
        await self.controller.validate_deployment(
            prefill_component_name=(
                defaults.prefill_worker_k8s_name
                if self.require_prefill and defaults
                else None
            ),
            decode_component_name=(
                defaults.decode_worker_k8s_name
                if self.require_decode and defaults
                else None
            ),
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
        )
        # wait_for_deployment_ready(include_planner=False) blocks until the
        # worker rollout is stable, so a planner that (re)starts after a DGD
        # template cap change reads the settled desired cap here — this is what
        # makes restart adoption of a changed static cap safe. The first
        # deployment-state read below therefore runs under the strict
        # initialization policy (any power-resolve failure is fatal).
        await self.controller.wait_for_deployment_ready(include_planner=False)
        if self.runtime_namespace_source is not None:
            await self.runtime_namespace_source.refresh_runtime_namespace()
        await self._refresh_deployment_state(is_initialization=True)
        await self.fpm_provider.async_init(self._runtime_namespace_or_none())
        await self._refresh_deployment_state(is_initialization=True)

    async def refresh(self) -> DeploymentState:
        namespace_changed = False
        if self.runtime_namespace_source is not None:
            namespace_changed = (
                await self.runtime_namespace_source.refresh_runtime_namespace()
            )

        await self._refresh_deployment_state(is_initialization=False)
        if namespace_changed:
            await self.fpm_provider.async_init(self._runtime_namespace_or_none())
            await self._refresh_deployment_state(is_initialization=False)
        await self.fpm_provider.refresh(self._state)
        return self._state

    def deployment_state(self) -> DeploymentState:
        return self._state

    def runtime_namespace(self) -> str:
        return self._runtime_namespace_or_none() or self.config.namespace

    def metrics_state(self) -> Metrics:
        return self._metrics_state

    async def collect_traffic(self) -> Optional[TrafficObservation]:
        return await self.traffic_provider.collect_traffic()

    def collect_accept_length(self, interval_str: str) -> Optional[float]:
        return self.traffic_provider.collect_accept_length(interval_str)

    async def collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        return await self.traffic_provider.collect_kv_hit_rate_observation(duration_s)

    def collect_fpm(self) -> FpmObservations:
        return self.fpm_provider.collect_fpm()

    async def apply_scaling(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        await self.controller.set_component_replicas(targets, blocking=blocking)

    async def shutdown(self) -> None:
        await self.fpm_provider.shutdown()

    async def _refresh_deployment_state(self, *, is_initialization: bool) -> None:
        self._refresh_worker_info()
        self._refresh_gpu_counts()
        self._refresh_power_configs(is_initialization=is_initialization)
        try:
            await self._refresh_replica_counts()
        except Exception as exc:
            # Replica counts are a separate DGD GET. A transport / apiserver
            # failure here must not undo the power conservative path above or
            # terminate the tick loop — keep last-good counts and, when power
            # awareness is on, latch scale-up suppression. Init still fails
            # closed: startup cannot proceed without known inventory.
            if is_initialization:
                raise
            logger.warning(
                "Replica count refresh failed (%s); keeping last-good counts",
                exc,
            )
            if self.config.enable_power_awareness:
                self._mark_power_scale_up_blocked(
                    f"replica count refresh failed ({exc}); keeping last-good "
                    "inventory and suppressing scale-up"
                )
        self._refresh_model_name()

    def _refresh_worker_info(self) -> None:
        get_worker_info = getattr(self.controller, "get_worker_info", None)
        if not callable(get_worker_info):
            return
        if self.require_prefill:
            self._refresh_component_worker_info(
                SubComponentType.PREFILL,
                self._state.prefill,
                get_worker_info,
            )
        if self.require_decode:
            self._refresh_component_worker_info(
                SubComponentType.DECODE,
                self._state.decode,
                get_worker_info,
            )

    def _refresh_component_worker_info(
        self, sub_type: SubComponentType, component_state, get_worker_info
    ) -> None:
        if (
            component_state.info is not None
            and component_state.info.max_num_batched_tokens is not None
        ):
            return

        try:
            fresh = get_worker_info(sub_type, self.config.backend)
        except Exception as exc:
            logger.debug(
                "get_worker_info refresh for %s failed: %s", sub_type.value, exc
            )
            return
        if fresh is None:
            return

        if component_state.info is None:
            component_state.info = fresh
            return

        for field_name in _MDC_REFRESH_FIELDS:
            fresh_val = getattr(fresh, field_name)
            if (
                fresh_val is not None
                and getattr(component_state.info, field_name) != fresh_val
            ):
                setattr(component_state.info, field_name, fresh_val)

    def _refresh_gpu_counts(self) -> None:
        state = self.deployment_state()
        try:
            prefill_gpus, decode_gpus = self.controller.get_gpu_counts(
                require_prefill=self.require_prefill,
                require_decode=self.require_decode,
            )
        except Exception as exc:
            # Transient apiserver / transport errors must not terminate the
            # whole refresh — that would skip the power-config conservative
            # path below. Catch broadly (not only ApiException): urllib3
            # timeouts and connection errors often escape the typed wrapper.
            # Fall back to last observed / configured GPU counts; a genuinely
            # missing value still fails closed via the errors check. Stale
            # GPU topology can under-enforce max_gpu_budget when caps change
            # while total watts stay flat, so latch scale-up suppression when
            # power awareness is on.
            logger.warning(
                "Could not read GPU counts from deployment (%s: %s), "
                "falling back to last observed or configured values",
                type(exc).__name__,
                exc,
            )
            if self.config.enable_power_awareness:
                self._mark_power_scale_up_blocked(
                    f"GPU count refresh failed ({exc}); keeping last-good "
                    "topology and suppressing scale-up"
                )
            prefill_gpus = state.prefill.num_gpus
            decode_gpus = state.decode.num_gpus

        if prefill_gpus is None:
            prefill_gpus = state.prefill.num_gpus
        if prefill_gpus is None:
            prefill_gpus = self.config.prefill_engine_num_gpu
        if decode_gpus is None:
            decode_gpus = state.decode.num_gpus
        if decode_gpus is None:
            decode_gpus = self.config.decode_engine_num_gpu

        errors = []
        if self.require_prefill and prefill_gpus is None:
            errors.append("Missing prefill_engine_num_gpu in config")
        if self.require_decode and decode_gpus is None:
            errors.append("Missing decode_engine_num_gpu in config")
        if errors:
            raise DeploymentValidationError(errors)

        if self.require_prefill:
            state.prefill.num_gpus = prefill_gpus
        if self.require_decode:
            state.decode.num_gpus = decode_gpus

    def _refresh_power_configs(self, *, is_initialization: bool) -> None:
        """Resolve DGD-owned per-GPU caps into deployment state.

        No-op when power awareness is off. Policy differs by phase:

        * Initialization (strict / fail closed): any resolve failure raises
          ``DeploymentValidationError`` and aborts planner startup — power math
          must not run on guessed caps. ``initialize()`` waits for worker
          readiness first, so this reads the settled DGD-desired cap.
        * Runtime (conservative): a resolve failure or a changed valid cap
          keeps the last-good per-role watts (taking the per-role maximum of
          old/new — over-estimating projected watts is safe under a power
          ceiling), sets the deployment-scoped scale-up-blocked flag, and
          leaves adoption of the new cap to a post-rollout planner restart.
          It never mutates Pods.
        """
        if not self.config.enable_power_awareness:
            return

        state = self.deployment_state()
        prefill_name, decode_name = self._power_component_names()
        try:
            prefill_cfg, decode_cfg = self.controller.get_component_power_configs(
                require_prefill=self.require_prefill,
                require_decode=self.require_decode,
                prefill_component_name=prefill_name,
                decode_component_name=decode_name,
            )
        except Exception as exc:
            # Transient apiserver / transport errors and malformed/missing caps
            # both land here. Startup fails closed; runtime keeps the last-good
            # caps and blocks scale-up rather than terminating refresh. Catch
            # broadly so urllib3 timeouts that are not wrapped as ApiException
            # still take the conservative path.
            if is_initialization:
                raise DeploymentValidationError(
                    [f"Failed to resolve DGD-owned power caps at startup: {exc}"]
                ) from exc
            self._mark_power_scale_up_blocked(
                f"power cap refresh failed ({exc}); keeping last-good caps and "
                "suppressing scale-up"
            )
            return

        reasons: list[str] = []
        if self.require_prefill and prefill_cfg is not None:
            reason = self._apply_power_config_for_role(
                state.prefill, prefill_cfg, is_initialization=is_initialization
            )
            if reason:
                reasons.append(reason)
        if self.require_decode and decode_cfg is not None:
            reason = self._apply_power_config_for_role(
                state.decode, decode_cfg, is_initialization=is_initialization
            )
            if reason:
                reasons.append(reason)
        if reasons:
            self._mark_power_scale_up_blocked("; ".join(reasons))

        if is_initialization:
            self._validate_minimum_power_footprint(prefill_cfg, decode_cfg)

    def _power_component_names(self) -> tuple[Optional[str], Optional[str]]:
        """Backend-default component names used as explicit-name fallbacks.

        Role resolution matches by ``type`` first (disagg) and by the unique
        generic ``type: worker`` component (agg), so these names only matter
        when the DGD renames a component; mirrors ``initialize()``.
        """
        defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)
        prefill_name = (
            defaults.prefill_worker_k8s_name
            if self.require_prefill and defaults
            else None
        )
        decode_name = (
            defaults.decode_worker_k8s_name
            if self.require_decode and defaults
            else None
        )
        return prefill_name, decode_name

    def _apply_power_config_for_role(
        self,
        component_state: ComponentState,
        cfg: ComponentPowerConfig,
        *,
        is_initialization: bool,
    ) -> Optional[str]:
        """Update one component's power fields; return a block reason or None.

        Returns a non-empty reason string only when a *changed* valid cap is
        observed at runtime (the caller then suppresses scale-up). Init and
        first-observation always adopt the DGD-desired value with no block.
        """
        new_watts = cfg.watts_per_replica
        old_watts = component_state.power_watts_per_replica

        if is_initialization or old_watts is None:
            component_state.power_gpu_limit_watts = cfg.gpu_power_limit_watts
            component_state.power_watts_per_replica = new_watts
            return None

        if new_watts == old_watts:
            # Unchanged desired cap: keep the (identical) per-GPU value fresh.
            component_state.power_gpu_limit_watts = cfg.gpu_power_limit_watts
            return None

        # Changed valid cap mid-run. A DGD template change produces a mixed
        # old/new Pod population while the DGD exposes only the new desired
        # value, so the planner cannot prove the effective per-replica power.
        # Hold the conservative per-role maximum and block scale-up; only raise
        # the stored value (never lower it) so projected watts over-estimate.
        if new_watts > old_watts:
            component_state.power_gpu_limit_watts = cfg.gpu_power_limit_watts
            component_state.power_watts_per_replica = new_watts
        return (
            f"{cfg.role} per-replica power changed {old_watts}W -> {new_watts}W; "
            f"holding {max(old_watts, new_watts)}W and suppressing scale-up until "
            "the worker rollout completes and the planner restarts"
        )

    def _mark_power_scale_up_blocked(self, reason: str) -> None:
        """Latch the deployment-scoped scale-up block (sticky until restart)."""
        state = self.deployment_state()
        if not state.power_scale_up_blocked:
            logger.warning("Power scale-up blocked: %s", reason)
        state.power_scale_up_blocked = True
        state.power_scale_up_blocked_reason = reason

    def _validate_minimum_power_footprint(
        self,
        prefill_cfg: Optional[ComponentPowerConfig],
        decode_cfg: Optional[ComponentPowerConfig],
    ) -> None:
        """Fail closed at startup if the minimum footprint can't fit the budget.

        ``min_endpoint`` replicas of every required role must fit
        ``total_gpu_power_limit``; otherwise the ceiling is unsatisfiable and
        the planner must not start rather than clamp to an impossible target.
        """
        budget = self.config.total_gpu_power_limit
        if budget is None:
            return
        p_watts = prefill_cfg.watts_per_replica if prefill_cfg else None
        d_watts = decode_cfg.watts_per_replica if decode_cfg else None
        if not minimum_power_footprint_fits(
            budget, self.config.min_endpoint, p_watts, d_watts
        ):
            raise DeploymentValidationError(
                [
                    "Infeasible power budget: minimum footprint "
                    f"(min_endpoint={self.config.min_endpoint} of "
                    f"prefill={p_watts}W, decode={d_watts}W per replica) exceeds "
                    f"total_gpu_power_limit={budget}W. Raise the budget or lower "
                    "the per-GPU caps on the worker podTemplate annotations."
                ]
            )

    async def _refresh_replica_counts(self) -> None:
        prefill_name = (
            self._state.prefill.info.k8s_name
            if self.require_prefill and self._state.prefill.info is not None
            else None
        )
        decode_name = (
            self._state.decode.info.k8s_name
            if self.require_decode and self._state.decode.info is not None
            else None
        )
        (
            prefill_count,
            decode_count,
            stable,
        ) = await self.controller.get_actual_worker_counts(
            prefill_component_name=prefill_name,
            decode_component_name=decode_name,
        )
        if self.require_prefill:
            self._state.prefill.replicas.active = prefill_count
            self._state.prefill.replicas.expected = prefill_count if stable else None
            self._state.prefill.replicas.scaling = not stable
        if self.require_decode:
            self._state.decode.replicas.active = decode_count
            self._state.decode.replicas.expected = decode_count if stable else None
            self._state.decode.replicas.scaling = not stable

    def _refresh_model_name(self) -> None:
        try:
            self._state.model_name = self.controller.get_model_name(
                require_prefill=self.require_prefill,
                require_decode=self.require_decode,
            )
        except Exception as exc:
            logger.warning("Failed to refresh planner model name: %s", exc)

    def _runtime_namespace_or_none(self) -> Optional[str]:
        if self.runtime_namespace_source is None:
            return None
        return self.runtime_namespace_source.runtime_namespace()
