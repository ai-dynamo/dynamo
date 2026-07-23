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
        # Per-role ``gpus_per_replica`` (nodeCount × per-pod GPUs) the startup
        # power caps were computed against. ``power_watts_per_replica`` is
        # cached once at startup; if this topology quantity changes under a
        # running planner the cached wattage goes stale, so the runtime guard
        # fails closed until the planner restarts and re-reads the caps.
        self._power_startup_gpus_per_replica: Optional[
            tuple[Optional[int], Optional[int]]
        ] = None

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
        # template cap change reads the settled desired cap once at startup.
        await self.controller.wait_for_deployment_ready(include_planner=False)
        if self.runtime_namespace_source is not None:
            await self.runtime_namespace_source.refresh_runtime_namespace()
        await self._refresh_deployment_state()
        self._load_static_power_caps_at_startup()
        await self.fpm_provider.async_init(self._runtime_namespace_or_none())
        await self._refresh_deployment_state()

    async def refresh(self) -> DeploymentState:
        namespace_changed = False
        if self.runtime_namespace_source is not None:
            namespace_changed = (
                await self.runtime_namespace_source.refresh_runtime_namespace()
            )

        await self._refresh_deployment_state()
        if namespace_changed:
            await self.fpm_provider.async_init(self._runtime_namespace_or_none())
            await self._refresh_deployment_state()
        self._assert_power_topology_static()
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

    async def _refresh_deployment_state(self) -> None:
        self._refresh_worker_info()
        self._refresh_gpu_counts()
        await self._refresh_replica_counts()
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
        except DeploymentValidationError as exc:
            logger.warning(
                "Could not read GPU counts from deployment (%s), "
                "falling back to last observed or configured values",
                exc,
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

    def _load_static_power_caps_at_startup(self) -> None:
        """Read DGD-owned per-GPU caps once at startup; fail closed if bad.

        The annotation is treated as static for the planner lifetime. Cap
        changes take effect after the worker rollout completes and the planner
        restarts (re-running this hook).
        """
        if not self.config.enable_power_awareness:
            return

        try:
            prefill_cfg, decode_cfg = self._resolve_power_configs()
        except Exception as exc:
            raise DeploymentValidationError(
                [f"Failed to resolve DGD-owned power caps at startup: {exc}"]
            ) from exc

        state = self.deployment_state()
        if self.require_prefill and prefill_cfg is not None:
            self._adopt_power_config(state.prefill, prefill_cfg)
        if self.require_decode and decode_cfg is not None:
            self._adopt_power_config(state.decode, decode_cfg)
        self._validate_minimum_power_footprint(prefill_cfg, decode_cfg)
        # Snapshot the replica-wide GPU total the cached wattage was derived
        # from so the runtime guard can detect a topology change (see
        # ``_assert_power_topology_static``). Uses the power resolver's
        # ``gpus_per_replica`` (not ``get_gpu_counts`` / ``num_gpus``), so
        # aggregate ``type: worker`` and named generic workers are visible.
        self._power_startup_gpus_per_replica = (
            prefill_cfg.gpus_per_replica
            if self.require_prefill and prefill_cfg is not None
            else None,
            decode_cfg.gpus_per_replica
            if self.require_decode and decode_cfg is not None
            else None,
        )

    def _resolve_power_configs(
        self,
    ) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
        """Resolve DGD power configs with the same name/generic semantics as startup.

        Uses ``get_component_power_configs`` (explicit backend names + unique
        generic ``type: worker`` fallback). Does not write environment state —
        callers decide whether to adopt caps or only compare topology.
        """
        get_configs = getattr(self.controller, "get_component_power_configs", None)
        if not callable(get_configs):
            raise DeploymentValidationError(
                [
                    "Power awareness requires a connector that can resolve "
                    "DGD-owned per-GPU caps; this connector does not implement "
                    "get_component_power_configs."
                ]
            )

        prefill_name, decode_name = self._power_component_names()
        return get_configs(
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
            prefill_component_name=prefill_name,
            decode_component_name=decode_name,
        )

    def _assert_power_topology_static(self) -> None:
        """Fail closed if GPUs-per-replica changed since startup.

        ``power_watts_per_replica`` is read once at startup and treated as
        static for the planner lifetime. A worker rollout that changes
        per-pod GPUs or ``multinode.nodeCount`` leaves that cached wattage
        stale, so the power budget would be projected against the old
        per-replica watts and could admit a scale-up that exceeds the real
        requested-cap total.

        Re-resolve topology through the same power-config path used at
        startup (explicit component names + aggregate generic-worker
        fallback). Do **not** trust ``get_gpu_counts`` / ``num_gpus`` here —
        that shared path cannot see generic ``type: worker`` components and
        would silently retain the previous count on ``DeploymentValidationError``,
        leaving this guard blind. Cap watts stay static; only topology is
        re-checked. On change or resolution failure, refuse to act and require
        a restart.
        """
        if not self.config.enable_power_awareness:
            return
        baseline = self._power_startup_gpus_per_replica
        if baseline is None:
            return

        try:
            prefill_cfg, decode_cfg = self._resolve_power_configs()
        except Exception as exc:
            raise DeploymentValidationError(
                [
                    "Failed to re-verify power-relevant GPU topology at runtime "
                    f"({exc}). power_watts_per_replica is cached at startup, so "
                    "the planner cannot safely project the budget. Restart the "
                    "Planner to re-read the DGD power annotation."
                ]
            ) from exc

        startup_p, startup_d = baseline
        changed = []
        if self.require_prefill and startup_p is not None:
            current_p = (
                prefill_cfg.gpus_per_replica if prefill_cfg is not None else None
            )
            if current_p != startup_p:
                changed.append(f"prefill {startup_p} -> {current_p}")
        if self.require_decode and startup_d is not None:
            current_d = decode_cfg.gpus_per_replica if decode_cfg is not None else None
            if current_d != startup_d:
                changed.append(f"decode {startup_d} -> {current_d}")
        if changed:
            raise DeploymentValidationError(
                [
                    "Power-relevant GPU topology changed at runtime ("
                    + ", ".join(changed)
                    + " GPUs/replica). power_watts_per_replica is cached at "
                    "startup, so the power budget would be projected against "
                    "stale per-replica watts. Restart the Planner to re-read "
                    "the DGD power annotation against the new topology."
                ]
            )

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

    @staticmethod
    def _adopt_power_config(
        component_state: ComponentState, cfg: ComponentPowerConfig
    ) -> None:
        component_state.power_gpu_limit_watts = cfg.gpu_power_limit_watts
        component_state.power_watts_per_replica = cfg.watts_per_replica

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
