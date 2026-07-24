# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
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
        # template cap change reads the settled desired cap once at startup.
        await self.controller.wait_for_deployment_ready(include_planner=False)
        if self.runtime_namespace_source is not None:
            await self.runtime_namespace_source.refresh_runtime_namespace()
        # Share one DGD GET across GPU-count refresh and power-cap load when
        # the connector exposes get_graph_deployment (Kubernetes only).
        deployment = self._shared_dgd_deployment()
        await self._refresh_deployment_state(deployment=deployment)
        self._load_static_power_caps_at_startup(deployment=deployment)
        # FPM init can change the effective runtime namespace / discovery view;
        # re-refresh replica/GPU/model state afterward so the first tick sees
        # post-init truth. This second call intentionally omits the shared
        # deployment snapshot (power caps stay startup-static) and may issue
        # a separate DGD GET for GPU counts — that is expected, not a merge
        # of the shared-GET path above.
        await self.fpm_provider.async_init(self._runtime_namespace_or_none())
        await self._refresh_deployment_state()

    async def refresh(self) -> DeploymentState:
        namespace_changed = False
        if self.runtime_namespace_source is not None:
            namespace_changed = (
                await self.runtime_namespace_source.refresh_runtime_namespace()
            )

        # Power caps are static for the planner lifetime: read once at startup
        # (see ``_load_static_power_caps_at_startup``) and never re-read here.
        # A cap change requires a worker rollout plus a Planner restart, so
        # refresh() does not re-resolve or drift-check the DGD annotation.
        await self._refresh_deployment_state()
        if namespace_changed:
            await self.fpm_provider.async_init(self._runtime_namespace_or_none())
            await self._refresh_deployment_state()
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

    async def _refresh_deployment_state(
        self, deployment: Optional[dict] = None
    ) -> None:
        self._refresh_worker_info()
        self._refresh_gpu_counts(deployment=deployment)
        await self._refresh_replica_counts()
        self._refresh_model_name()

    def _shared_dgd_deployment(self) -> Optional[dict]:
        """One DGD GET for GPU + power reads when power awareness is on.

        Uses the Kubernetes-only ``get_graph_deployment`` duck-type; other
        connectors keep separate (or no-op) paths. Not added to
        ``PlannerConnector``.
        """
        if not self.config.enable_power_awareness:
            return None
        fetch = getattr(self.controller, "get_graph_deployment", None)
        if not callable(fetch):
            return None
        return fetch()

    @staticmethod
    def _call_with_optional_deployment(method, *, deployment=None, **kwargs):
        """Invoke a connector method, forwarding ``deployment`` when accepted.

        Inspects the callable signature so a real ``TypeError`` raised inside
        the connector (wrong arg types, ``None`` arithmetic, etc.) is not
        swallowed by a retry without ``deployment``.
        """
        if deployment is not None:
            try:
                params = inspect.signature(method).parameters
            except (TypeError, ValueError):
                # Builtins / C extensions without an inspectable signature —
                # fall through to the no-deployment call rather than guess.
                return method(**kwargs)
            if "deployment" in params:
                return method(**kwargs, deployment=deployment)
        return method(**kwargs)

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

    def _refresh_gpu_counts(self, deployment: Optional[dict] = None) -> None:
        state = self.deployment_state()
        try:
            prefill_gpus, decode_gpus = self._call_with_optional_deployment(
                self.controller.get_gpu_counts,
                deployment=deployment,
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

    def _load_static_power_caps_at_startup(
        self, deployment: Optional[dict] = None
    ) -> None:
        """Read DGD-owned per-GPU caps once at startup; fail closed if bad.

        The annotation is treated as static for the planner lifetime. Cap
        changes take effect after the worker rollout completes and the planner
        restarts (re-running this hook).
        """
        if not self.config.enable_power_awareness:
            return

        try:
            prefill_cfg, decode_cfg = self._resolve_power_configs(deployment=deployment)
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

    def _resolve_power_configs(
        self,
        deployment: Optional[dict] = None,
    ) -> tuple[Optional[ComponentPowerConfig], Optional[ComponentPowerConfig]]:
        """Resolve DGD power configs with the same name/generic semantics as startup.

        Uses ``get_component_power_configs`` (explicit backend names + unique
        generic ``type: worker`` fallback). Read once at startup to adopt the
        static caps; not called on the per-tick refresh path.
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
        return self._call_with_optional_deployment(
            get_configs,
            deployment=deployment,
            require_prefill=self.require_prefill,
            require_decode=self.require_decode,
            prefill_component_name=prefill_name,
            decode_component_name=decode_name,
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
