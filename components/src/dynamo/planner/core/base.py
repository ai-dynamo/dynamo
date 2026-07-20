# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed planner runtime plumbing.

This is the candidate replacement for ``core/base.py``.  Planner decision logic
still lives in the engine/plugins; this class owns engine lifecycle,
diagnostics, tick orchestration, and delegates deployment/metrics I/O to a
``PlannerEnvironment``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import aiohttp.web
from kubernetes.client import ApiException
from prometheus_client import start_http_server

from dynamo.planner.config.backend_components import get_planner_k8s_component_names
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.core import util
from dynamo.planner.core.budget import POWER_ANNOTATION_KEY
from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    PlannerEffects,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.environment.interface import PlannerEnvironment
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.errors import DynamoGraphDeploymentNotFoundError, PlannerError
from dynamo.planner.monitoring.diagnostics_recorder import DiagnosticsRecorder
from dynamo.planner.monitoring.live_dashboard import start_live_dashboard
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.planner.offline.trace_data import extract_metrics_from_trace
from dynamo.runtime import DistributedRuntime

if TYPE_CHECKING:
    from dynamo.planner.monitoring.worker_info import WorkerInfo

logger = logging.getLogger(__name__)


def _engine_caps(worker_info, num_gpu: Optional[int]) -> Optional[EngineCapabilities]:
    if worker_info is None and num_gpu is None:
        return None
    return EngineCapabilities(
        num_gpu=num_gpu,
        max_num_batched_tokens=(
            worker_info.max_num_batched_tokens if worker_info else None
        ),
        max_num_seqs=worker_info.max_num_seqs if worker_info else None,
        context_length=worker_info.context_length if worker_info else None,
        max_kv_tokens=worker_info.max_kv_tokens if worker_info else None,
        kv_cache_block_size=worker_info.kv_cache_block_size if worker_info else None,
        speculative_nextn=worker_info.speculative_nextn if worker_info else None,
    )


def build_worker_capabilities(state: DeploymentState) -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=_engine_caps(state.prefill.info, state.prefill.num_gpus),
        decode=_engine_caps(state.decode.info, state.decode.num_gpus),
    )


class NativePlannerBase:
    """Base adapter shared by planner modes."""

    require_prefill: bool = False
    require_decode: bool = False

    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        config: PlannerConfig,
        environment: PlannerEnvironment,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.environment = environment
        self.namespace = config.namespace

        self.prometheus_port = config.metric_reporting_prometheus_port
        self.prometheus_metrics = PlannerPrometheusMetrics()
        if self.prometheus_port != 0:
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    "Started Prometheus metrics server on port %s",
                    self.prometheus_port,
                )
            except Exception as exc:
                logger.error("Failed to start Prometheus metrics server: %s", exc)
            self.prometheus_metrics.sla_target_ttft_ms.set(config.ttft_ms)
            self.prometheus_metrics.sla_target_itl_ms.set(config.itl_ms)

        self._cumulative_gpu_hours: float = 0.0
        self._last_gpu_hours_update_ts: Optional[float] = None

        # Power-annotation sweep throttle (Phase 1). The sweep reconciles pod
        # power-limit annotations against K8s; running it every tick is needless
        # apiserver load, so run() throttles it to
        # config.power_annotation_interval_seconds and forces a per-tick sweep
        # for one interval after each scale-up (when new pods appear).
        self._last_power_annotation_sweep_s: float = 0.0
        self._force_power_annotations_until_s: float = 0.0
        # When power awareness is disabled, latch off removal sweeps after the
        # first pass that finds no planner-owned annotations (greenfield) or
        # after stale caps are cleared, so default-off planners do zero apiserver
        # LIST work in steady state.
        self._disabled_power_removal_latched: bool = False
        self._power_projected_zero_warned: bool = False
        # Per-replica GPU totals (per-pod × multinode nodeCount) for the power
        # budget projection, resolved once off the tick loop during _async_init.
        # _publish_power_budget_metrics reads this cache so it never blocks the
        # asyncio loop on an apiserver GET, and the projection unit stays fixed
        # (replica-total, or a logged per-pod fallback) for the whole run.
        self._power_projection_gpu_counts: Optional[tuple[int, int]] = None

        self._recorder = DiagnosticsRecorder(config=config)
        self._dashboard_runner: Optional[aiohttp.web.AppRunner] = None
        self._engine: Optional[EngineProtocol] = None
        self._last_worker_counts: Optional[WorkerCounts] = None

    async def _async_init(self) -> None:
        await self.environment.initialize()

        if self.config.enable_power_awareness:
            await asyncio.to_thread(self._resolve_power_projection_gpu_counts)

        await self._bootstrap_regression()
        await self._bootstrap_engine_plugins_if_needed()

        if self.config.advisory:
            logger.info(
                "[ADVISORY] Planner started in advisory mode; "
                "scaling decisions will be logged but NOT executed."
            )

        if self.config.live_dashboard_port:
            try:
                self._dashboard_runner = await start_live_dashboard(
                    self._recorder, self.config.live_dashboard_port
                )
            except Exception as exc:
                logger.error("Failed to start live dashboard: %s", exc)

    def _build_worker_capabilities(self) -> WorkerCapabilities:
        return build_worker_capabilities(self.environment.deployment_state())

    def _runtime_namespace(self) -> str:
        return self.environment.runtime_namespace()

    def _required_worker_info(self, component: SubComponentType) -> "WorkerInfo":
        state = self.environment.deployment_state()
        info = (
            state.prefill.info
            if component == SubComponentType.PREFILL
            else state.decode.info
        )
        if info is None:
            raise RuntimeError(f"Missing worker info for {component.value}")
        return info

    def _ensure_engine(self) -> EngineProtocol:
        if self._engine is not None:
            return self._engine
        # Deliberately lazy: importing planner core should not load the plugin stack.
        from dynamo.planner.plugins.builtins.observe import EnvironmentObservePlugin
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        self._engine = OrchestratorEngineAdapter(
            self.config,
            self._build_worker_capabilities(),
            observe_plugin=EnvironmentObservePlugin(
                self.environment,
                require_prefill=self.require_prefill,
                require_decode=self.require_decode,
            ),
        )
        return self._engine

    async def _install_benchmark_fpms(
        self,
        *,
        prefill_fpms=None,
        decode_fpms=None,
        agg_fpms=None,
    ) -> None:
        # Keep the orchestrator dependency aligned with lazy engine construction.
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        engine = self._ensure_engine()
        assert isinstance(engine, OrchestratorEngineAdapter)
        await engine.bootstrap_from_fpms(
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
            agg_fpms=agg_fpms,
            historical_traffic=self._load_predictor_warmup_observations(),
        )

    def _load_predictor_warmup_observations(
        self,
    ) -> Optional[list[TrafficObservation]]:
        if self.config.load_predictor_warmup_trace is None:
            return None
        metrics = extract_metrics_from_trace(
            self.config.load_predictor_warmup_trace,
            self.config.throughput_adjustment_interval_seconds,
        )
        return [
            TrafficObservation(
                duration_s=self.config.throughput_adjustment_interval_seconds,
                num_req=float(m["request_count"]),
                isl=float(m["avg_isl"]),
                osl=float(m["avg_osl"]),
            )
            for m in metrics
        ]

    async def _bootstrap_engine_plugins_if_needed(self) -> None:
        # Keep the orchestrator dependency aligned with lazy engine construction.
        from dynamo.planner.plugins.orchestrator.engine_adapter import (
            OrchestratorEngineAdapter,
        )

        engine = self._ensure_engine()
        if isinstance(engine, OrchestratorEngineAdapter):
            if engine.plugins_bootstrapped:
                return
            await engine.bootstrap_plugins(
                historical_traffic=self._load_predictor_warmup_observations()
            )

    async def _bootstrap_regression(self) -> None:
        pass

    async def _refresh_and_update_capabilities(self) -> None:
        old_state = self.environment.deployment_state().clone()
        await self.environment.refresh()
        new_state = self.environment.deployment_state().clone()
        if not util.deployment_state_changed(
            old_state,
            new_state,
            self.require_prefill,
            self.require_decode,
        ):
            return
        update_engine_capabilities = getattr(self._engine, "update_capabilities", None)
        if callable(update_engine_capabilities):
            update_engine_capabilities(self._build_worker_capabilities())

    async def _collect_traffic(self) -> Optional[TrafficObservation]:
        return await self.environment.collect_traffic()

    async def _collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        return await self.environment.collect_kv_hit_rate_observation(duration_s)

    def _collect_fpm(self) -> FpmObservations:
        return self.environment.collect_fpm()

    def _emit_observed_traffic_metrics(self) -> None:
        if self.prometheus_port == 0:
            return
        m = self.environment.metrics_state()
        if not m.is_valid():
            return
        # ``is_valid`` enforces these invariants, but mypy cannot narrow
        # dataclass fields through a separate predicate method.
        assert m.ttft is not None
        assert m.itl is not None
        assert m.num_req is not None
        assert m.request_duration is not None
        assert m.isl is not None
        assert m.osl is not None
        self.prometheus_metrics.observed_ttft_ms.set(m.ttft)
        self.prometheus_metrics.observed_itl_ms.set(m.itl)
        self.prometheus_metrics.observed_requests_per_second.set(
            m.num_req / self.config.throughput_adjustment_interval_seconds
        )
        self.prometheus_metrics.observed_request_duration_seconds.set(
            m.request_duration
        )
        self.prometheus_metrics.observed_input_sequence_tokens.set(m.isl)
        self.prometheus_metrics.observed_output_sequence_tokens.set(m.osl)

    def _emit_per_engine_fpm(
        self,
        prefill_stats: Optional[dict] = None,
        decode_stats: Optional[dict] = None,
    ) -> None:
        pm = self.prometheus_metrics
        pm.engine_queued_prefill_tokens.clear()
        pm.engine_queued_decode_kv_tokens.clear()
        pm.engine_inflight_decode_kv_tokens.clear()

        if prefill_stats:
            for (wid, dp), fpm in prefill_stats.items():
                labels = dict(worker_id=wid, dp_rank=str(dp))
                pm.engine_queued_prefill_tokens.labels(**labels).set(
                    fpm.queued_requests.sum_prefill_tokens
                )

        if decode_stats:
            for (wid, dp), fpm in decode_stats.items():
                labels = dict(worker_id=wid, dp_rank=str(dp))
                pm.engine_queued_decode_kv_tokens.labels(**labels).set(
                    fpm.queued_requests.sum_decode_kv_tokens
                )
                pm.engine_inflight_decode_kv_tokens.labels(**labels).set(
                    fpm.scheduled_requests.sum_decode_kv_tokens
                )

    async def _collect_worker_counts(self) -> WorkerCounts:
        state = self.environment.deployment_state()
        return WorkerCounts(
            ready_num_prefill=(
                state.prefill.replicas.active if self.require_prefill else None
            ),
            ready_num_decode=(
                state.decode.replicas.active if self.require_decode else None
            ),
            expected_num_prefill=(
                state.prefill.replicas.expected if self.require_prefill else None
            ),
            expected_num_decode=(
                state.decode.replicas.expected if self.require_decode else None
            ),
            prefill_scaling_in_progress=(
                self.require_prefill and state.prefill.replicas.scaling
            ),
            decode_scaling_in_progress=(
                self.require_decode and state.decode.replicas.scaling
            ),
        )

    async def _gather_tick_input(self, tick: ScheduledTick) -> TickInput:
        now = time.time()
        traffic = None
        worker_counts = None
        fpm_obs = None

        if tick.need_traffic_metrics:
            if tick.use_full_traffic_metrics:
                traffic = await self._collect_traffic()
            else:
                traffic = await self._collect_kv_hit_rate_observation(
                    tick.traffic_metrics_duration_s
                )
        if tick.need_worker_states:
            worker_counts = await self._collect_worker_counts()
        if tick.need_worker_fpm:
            fpm_obs = self._collect_fpm()
        return TickInput(
            now_s=now,
            traffic=traffic,
            worker_counts=worker_counts,
            fpm_observations=fpm_obs,
        )

    async def _observe_tick(
        self, engine: EngineProtocol, tick: ScheduledTick
    ) -> TickInput:
        observe = getattr(engine, "observe", None)
        if callable(observe):
            return await observe(tick, time.time())
        return await self._gather_tick_input(tick)

    def _publish_observation_metrics(self, tick_input: TickInput) -> None:
        if tick_input.traffic is not None:
            self._emit_observed_traffic_metrics()
        if tick_input.fpm_observations is not None and self.prometheus_port != 0:
            self._emit_per_engine_fpm(
                tick_input.fpm_observations.prefill,
                tick_input.fpm_observations.decode,
            )

    async def _apply_effects(self, effects: PlannerEffects) -> None:
        pass

    async def _apply_power_annotations(self) -> None:
        """Annotate worker pods with per-GPU power limits (Phase 1).

        Reads the actual annotation from each Pod object returned by
        get_component_pods(). Only PATCHes when annotation is missing or wrong.
        K8s is the source of truth — no local cache. The DGD is read once per
        sweep and shared across the prefill/decode pod lookups.

        Performs one full reconcile sweep. The caller
        (_should_sweep_power_annotations in run()) throttles this to
        config.power_annotation_interval_seconds in steady state and forces a
        per-tick sweep for one interval after a scale-up, so newly-created pods
        pick up the cap promptly while steady-state apiserver load stays
        bounded. AIC-driven cap changes (Phase 3+) propagate on the next sweep.

        Skipped in advisory mode — annotations are a cluster-visible side
        effect on customer-owned Pod objects, so advisory runs must observe
        the contract that ``_apply_scaling_targets`` already follows: log
        intent, mutate nothing.

        The kubernetes client is synchronous, so the sweep itself runs in a
        worker thread via ``asyncio.to_thread``. The run loop still waits for
        the reconcile to finish before the next tick, but a slow apiserver or
        a large DGD does not block unrelated asyncio tasks such as metrics,
        diagnostics, and shutdown handling.
        """
        if self.config.advisory:
            return
        connector = getattr(self.environment, "controller", None)
        if not isinstance(connector, KubernetesConnector):
            return
        if not (self.require_prefill or self.require_decode):
            return
        if not self.config.enable_power_awareness:
            await asyncio.to_thread(self._run_power_annotation_removal, connector)
            return

        self._disabled_power_removal_latched = False
        await asyncio.to_thread(self._run_power_annotation_sweep, connector)

    def _run_power_annotation_removal(self, connector: KubernetesConnector) -> None:
        """Remove planner-owned power annotations from surviving worker pods.

        Called when power awareness is disabled so that pods previously
        annotated by the planner are cleaned up.  The Power Agent observes the
        missing key and releases the GPU cap through its normal release path.
        """
        try:
            deployment = connector.kube_api.get_graph_deployment(
                connector.graph_deployment_name
            )
            prefill_name, decode_name = get_planner_k8s_component_names(
                self.config.backend
            )

            # strict=True: a resolution failure (renamed component, schema
            # drift) must NOT masquerade as an empty pod list. Otherwise the
            # "no annotations found" latch below would fire on a failed lookup
            # and permanently disable cleanup, leaving stale caps in place.
            pods_to_clean: list = []
            if self.require_prefill:
                pods_to_clean.extend(
                    connector.get_component_pods(
                        SubComponentType.PREFILL,
                        deployment=deployment,
                        component_name=prefill_name,
                        strict=True,
                    )
                )
            if self.require_decode:
                pods_to_clean.extend(
                    connector.get_component_pods(
                        SubComponentType.DECODE,
                        deployment=deployment,
                        component_name=decode_name,
                        strict=True,
                    )
                )
        except (
            ApiException,
            DynamoGraphDeploymentNotFoundError,
            PlannerError,
            ValueError,
        ) as e:
            # Do not latch: every required component must resolve and list
            # successfully before a clean sweep can turn cleanup off. Retry on
            # the next sweep.
            logger.warning(
                "Power-annotation removal skipped: failed to read DGD, resolve a "
                "component, or list worker pods (%s); retrying on the next sweep.",
                e,
            )
            return

        any_annotation_found = False
        for pod in pods_to_clean:
            if (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY) is None:
                continue
            any_annotation_found = True
            try:
                connector.kube_api.remove_pod_annotation(
                    pod.metadata.name, POWER_ANNOTATION_KEY
                )
                logger.info(
                    "Removed power annotation from pod %s",
                    pod.metadata.name,
                )
            except ApiException as e:
                if e.status == 409:
                    logger.debug(
                        "409 Conflict removing power annotation from pod %s; "
                        "will retry on next sweep.",
                        pod.metadata.name,
                    )
                elif e.status == 429:
                    logger.warning(
                        "429 TooManyRequests removing power annotation from pod %s; "
                        "no Retry-After backoff, will retry on next sweep.",
                        pod.metadata.name,
                    )
                else:
                    logger.warning(
                        "Failed to remove power annotation from pod %s: %s",
                        pod.metadata.name,
                        e,
                    )
        if not any_annotation_found:
            self._disabled_power_removal_latched = True

    def _run_power_annotation_sweep(self, connector: KubernetesConnector) -> None:
        """Blocking power-annotation reconcile sweep (runs off the event loop).

        Invoked only via ``asyncio.to_thread`` from
        ``_apply_power_annotations`` after its guards pass. The connector is
        passed explicitly so the worker thread uses the locally narrowed
        ``KubernetesConnector`` rather than relying on a production assert.
        """

        # One DGD read for the whole sweep: resolving prefill and decode pods
        # each needs the deployment, so fetch it once and reuse it rather than
        # letting each get_component_pods() issue its own GET.
        #
        # The DGD read and pod listings hit the apiserver and can fail
        # transiently (ApiException — 5xx/timeout) or when the DGD is
        # momentarily unresolvable (DynamoGraphDeploymentNotFoundError — a
        # 404 blip). This reconciliation sweep is optional and idempotent, so
        # those must not propagate: run() wraps the tick loop in try/finally,
        # so an escaping exception would shut the engine down and exit the
        # planner. Log and skip — the next sweep retries. The catch stays
        # narrow (not the PlannerError base): component/validation errors are
        # already swallowed by get_component_pods(); other unexpected planner
        # subclasses still surface as bugs.
        try:
            deployment = connector.kube_api.get_graph_deployment(
                connector.graph_deployment_name
            )

            # Backend-default fallback names. If an aggregated DGD declares a
            # single generic ``type: worker`` component with a different name,
            # get_component_from_type_or_name resolves that actual DGD name.
            prefill_name, decode_name = get_planner_k8s_component_names(
                self.config.backend
            )

            pods_and_limits: list[tuple] = []
            if self.require_prefill:
                for pod in connector.get_component_pods(
                    SubComponentType.PREFILL,
                    deployment=deployment,
                    component_name=prefill_name,
                ):
                    pods_and_limits.append(
                        (pod, str(self.config.prefill_engine_gpu_power_limit))
                    )
            if self.require_decode:
                for pod in connector.get_component_pods(
                    SubComponentType.DECODE,
                    deployment=deployment,
                    component_name=decode_name,
                ):
                    pods_and_limits.append(
                        (pod, str(self.config.decode_engine_gpu_power_limit))
                    )
        except (ApiException, DynamoGraphDeploymentNotFoundError) as e:
            logger.warning(
                "Power-annotation sweep skipped: failed to read DGD or list "
                "worker pods (%s); retrying on the next sweep.",
                e,
            )
            return

        for pod, limit_str in pods_and_limits:
            current = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
            if current == limit_str:
                continue
            try:
                connector.kube_api.patch_pod_annotation(
                    pod.metadata.name, POWER_ANNOTATION_KEY, limit_str
                )
                logger.info(
                    "Annotated pod %s with %s=%s",
                    pod.metadata.name,
                    POWER_ANNOTATION_KEY,
                    limit_str,
                )
            except ApiException as e:
                if e.status == 409:
                    logger.debug(
                        "409 Conflict patching power annotation on pod %s; "
                        "will retry on next sweep.",
                        pod.metadata.name,
                    )
                elif e.status == 429:
                    logger.warning(
                        "429 TooManyRequests patching power annotation on pod %s; "
                        "no Retry-After backoff, will retry on next sweep.",
                        pod.metadata.name,
                    )
                else:
                    logger.warning(
                        "Failed to patch power annotation on pod %s: %s",
                        pod.metadata.name,
                        e,
                    )

    def _current_worker_counts(self) -> tuple[int, int]:
        """Best-known current (prefill, decode) ready worker counts.

        Reads ``self._last_worker_counts``, cached each tick in ``run()`` from
        the tick input — the count source the orchestrator engine exposes.
        Returns ``(0, 0)`` before the first worker-state tick populates it.
        Shared by ``_scaling_up`` and ``_log_decision_summary`` so the two
        can't drift.
        """
        if self._last_worker_counts is not None:
            return (
                self._last_worker_counts.ready_num_prefill or 0,
                self._last_worker_counts.ready_num_decode or 0,
            )
        return 0, 0

    def _scaling_up(self, effects: PlannerEffects) -> bool:
        """True when this tick's decision raises a managed replica count.

        A scale-up means the operator will create new worker pods, which start
        without the power-limit annotation; the caller uses this to force a
        prompt annotation sweep instead of waiting out the steady-state
        throttle. Reads current counts via ``_current_worker_counts``, the
        same source ``_log_decision_summary`` uses, so the force decision
        stays consistent with the logged summary.
        Scale-downs and holds return False because they create no pods.
        """
        if self._last_worker_counts is None:
            return False
        decision = effects.scale_to
        if decision is None:
            return False
        current_p, current_d = self._current_worker_counts()
        if decision.num_prefill is not None and decision.num_prefill > current_p:
            return True
        if decision.num_decode is not None and decision.num_decode > current_d:
            return True
        return False

    def _should_sweep_power_annotations(
        self, now_s: float, effects: PlannerEffects
    ) -> bool:
        """Decide whether to run a power-annotation sweep this tick.

        Gates both the enabled write sweep and the disabled removal sweep.
        Throttles steady-state sweeps to at most one per
        ``power_annotation_interval_seconds`` (a sweep costs a DGD read + a pod
        list per managed component, so per-tick sweeps multiply apiserver
        load). A scale-up (re)opens a force window of one interval during which
        every tick sweeps, so freshly-created pods are annotated without waiting
        out the throttle. When power awareness is disabled the scale-up force
        window is skipped. Removal sweeps run on the throttle until no
        planner-owned annotations remain, then latch off so default-off planners
        do zero apiserver work in steady state.
        Updates the sweep bookkeeping as a side effect when it returns True.
        """
        if self.config.advisory:
            return False
        if (
            not self.config.enable_power_awareness
            and self._disabled_power_removal_latched
        ):
            return False
        if self.config.enable_power_awareness and self._scaling_up(effects):
            self._force_power_annotations_until_s = (
                now_s + self.config.power_annotation_interval_seconds
            )
        within_force_window = now_s < self._force_power_annotations_until_s
        throttle_elapsed = (
            now_s - self._last_power_annotation_sweep_s
            >= self.config.power_annotation_interval_seconds
        )
        if within_force_window or throttle_elapsed:
            self._last_power_annotation_sweep_s = now_s
            return True
        return False

    def _resolve_power_projection_gpu_counts(self) -> None:
        """Resolve per-replica GPU totals for the power projection once.

        Called off the tick loop from ``_async_init``. The per-replica total
        (per-pod × multinode ``nodeCount``) is a static component spec, so a
        one-time resolve is sufficient and keeps ``_publish_power_budget_metrics``
        free of apiserver I/O on the asyncio loop. On failure we log and fall
        back to the per-pod config counts as the last known good value; because
        this runs once, the projection never silently flips units per tick.
        """
        p_gpu = self.config.prefill_engine_num_gpu or 0
        d_gpu = self.config.decode_engine_num_gpu or 0
        connector = getattr(self.environment, "controller", None)
        if connector is not None and hasattr(
            connector, "get_replica_gpu_counts_for_power_projection"
        ):
            state = self.environment.deployment_state()
            prefill_name = (
                state.prefill.info.k8s_name if state.prefill.info is not None else None
            )
            decode_name = (
                state.decode.info.k8s_name if state.decode.info is not None else None
            )
            try:
                (
                    p_gpu,
                    d_gpu,
                ) = connector.get_replica_gpu_counts_for_power_projection(
                    require_prefill=self.require_prefill,
                    require_decode=self.require_decode,
                    prefill_component_name=prefill_name,
                    decode_component_name=decode_name,
                )
            except (ApiException, PlannerError) as e:
                # Operational failures only (apiserver blip, missing/ambiguous
                # component, invalid GPU spec). Programming errors such as
                # AttributeError must propagate and abort init.
                logger.warning(
                    "Could not resolve per-replica GPU counts for power "
                    "projection (%s); power gauges will use per-pod counts "
                    "prefill=%s, decode=%s for this run.",
                    e,
                    p_gpu,
                    d_gpu,
                )
        self._power_projection_gpu_counts = (p_gpu, d_gpu)

    def _publish_power_budget_metrics(self, num_p: int, num_d: int) -> None:
        """Emit power budget gauges (Phase 1, dashboard observability only).

        These gauges are advisory: they expose the configured budget and the
        statically projected draw for dashboards. They do NOT gate scaling —
        budget enforcement is Phase 2 (PR #9684). Values come from static
        config, not DCGM, so the gauges stay meaningful even when DCGM is down.

        Per-replica GPU totals are read from the cache resolved once during
        ``_async_init`` — this method performs no apiserver I/O so it never
        blocks the tick loop.
        """
        if self.prometheus_port == 0 or not self.config.enable_power_awareness:
            return
        if self.config.total_gpu_power_limit is None:
            return

        if self._power_projection_gpu_counts is not None:
            p_gpu, d_gpu = self._power_projection_gpu_counts
        else:
            p_gpu = self.config.prefill_engine_num_gpu or 0
            d_gpu = self.config.decode_engine_num_gpu or 0
        if (self.require_prefill and p_gpu == 0) or (
            self.require_decode and d_gpu == 0
        ):
            if not self._power_projected_zero_warned:
                logger.warning(
                    "power_projected_watts will be zero or partial: "
                    "prefill_engine_num_gpu=%s, decode_engine_num_gpu=%s are not set. "
                    "Set these fields to get accurate power budget metrics.",
                    self.config.prefill_engine_num_gpu,
                    self.config.decode_engine_num_gpu,
                )
                self._power_projected_zero_warned = True
        projected = (
            num_p * self.config.prefill_engine_gpu_power_limit * p_gpu
            + num_d * self.config.decode_engine_gpu_power_limit * d_gpu
        )
        budget = self.config.total_gpu_power_limit
        pm = self.prometheus_metrics
        pm.power_budget_total_watts.set(budget)
        pm.power_projected_watts.set(projected)
        pm.power_budget_utilization.set(projected / budget if budget > 0 else 0.0)

    async def _apply_scaling_targets(
        self, targets: list[TargetReplica], blocking: bool = False
    ) -> None:
        if self.config.advisory or not targets:
            return
        await self.environment.apply_scaling(targets, blocking=blocking)

    def _log_decision_summary(self, effects: PlannerEffects) -> None:
        """Log a one-line summary of the scaling decision after each tick.

        Current worker counts come from ``_current_worker_counts`` — the
        cached ``self._last_worker_counts`` set in ``run()`` — shared with
        ``_scaling_up`` so the logged summary and the sweep-force decision
        cannot drift.
        """
        decision = effects.scale_to
        diag = effects.diagnostics

        current_p, current_d = self._current_worker_counts()

        rec_p = decision.num_prefill if decision else None
        rec_d = decision.num_decode if decision else None

        delta_p = (rec_p - current_p) if rec_p is not None else 0
        delta_d = (rec_d - current_d) if rec_d is not None else 0

        if decision is None or (delta_p == 0 and delta_d == 0):
            action = "hold"
        elif (delta_p > 0 or delta_d > 0) and (delta_p < 0 or delta_d < 0):
            action = "rebalance"
        elif delta_p > 0 or delta_d > 0:
            action = "scale_up"
        else:
            action = "scale_down"

        logger.info(
            "[summary] %s | current: prefill=%d decode=%d | "
            "recommended: prefill=%s decode=%s (delta: %+d / %+d) | "
            "load_reason=%s throughput_reason=%s | "
            "est_ttft=%.1fms est_itl=%.1fms",
            action.upper(),
            current_p,
            current_d,
            rec_p if rec_p is not None else "-",
            rec_d if rec_d is not None else "-",
            delta_p,
            delta_d,
            diag.load_decision_reason or "n/a",
            diag.throughput_decision_reason or "n/a",
            diag.estimated_ttft_ms or 0,
            diag.estimated_itl_ms or 0,
        )

    def _publish_inventory_and_gpu_hours(self, tick_input: TickInput) -> None:
        if tick_input.worker_counts is None:
            return
        num_p = tick_input.worker_counts.ready_num_prefill or 0
        num_d = tick_input.worker_counts.ready_num_decode or 0

        now = tick_input.now_s
        state = self.environment.deployment_state()
        prefill_gpus = state.prefill.num_gpus or 0
        decode_gpus = state.decode.num_gpus or 0
        if self._last_gpu_hours_update_ts is not None:
            dt_s = max(0.0, now - self._last_gpu_hours_update_ts)
            self._cumulative_gpu_hours += (
                (num_p * prefill_gpus + num_d * decode_gpus) * dt_s / 3600.0
            )
        self._last_gpu_hours_update_ts = now

        if self.prometheus_port == 0:
            return
        self.prometheus_metrics.num_prefill_replicas.set(num_p)
        self.prometheus_metrics.num_decode_replicas.set(num_d)
        self.prometheus_metrics.gpu_hours.set(self._cumulative_gpu_hours)
        self._publish_power_budget_metrics(num_p, num_d)

    @staticmethod
    def _set_if_observed(gauge, value: Optional[float]) -> None:
        """None = no new observation this tick -> leave the gauge unchanged.

        A concrete value (including 0.0) = asserted observation -> publish it.
        Mirrors the set/unset semantics documented on ``PredictionData``.
        """
        if value is not None:
            gauge.set(value)

    def _report_diagnostics(self, tick: ScheduledTick, diag: TickDiagnostics) -> None:
        if self.prometheus_port == 0:
            return
        pm = self.prometheus_metrics
        interval = self.config.throughput_adjustment_interval_seconds

        self._set_if_observed(pm.estimated_ttft_ms, diag.estimated_ttft_ms)
        self._set_if_observed(pm.estimated_itl_ms, diag.estimated_itl_ms)

        if tick.run_load_scaling:
            pm.load_scaling_decision.state(diag.load_decision_reason or "unset")

        # Predicted-load / engine-capacity gauges: ``PredictionData`` supports
        # partial predictions, so each gauge is gated on its own diagnostic --
        # a tick that asserts only some fields must not zero the others. Fields
        # are nulled by _reset_diag() at every tick start, so `is not None`
        # means "produced this tick" (builtin throughput loop or an
        # independently-scheduled PREDICT plugin).
        self._set_if_observed(
            pm.predicted_requests_per_second,
            (
                diag.predicted_num_req / interval
                if diag.predicted_num_req is not None and interval > 0
                else None
            ),
        )
        self._set_if_observed(pm.predicted_input_sequence_tokens, diag.predicted_isl)
        self._set_if_observed(pm.predicted_output_sequence_tokens, diag.predicted_osl)
        self._set_if_observed(
            pm.engine_prefill_capacity_requests_per_second, diag.engine_rps_prefill
        )
        self._set_if_observed(
            pm.engine_decode_capacity_requests_per_second, diag.engine_rps_decode
        )
        # The throughput-decision enum stays gated on the builtin throughput
        # cadence: it reflects a builtin-loop decision, not a plugin prediction.
        if tick.run_throughput_scaling:
            pm.throughput_scaling_decision.state(
                diag.throughput_decision_reason or "unset"
            )

    @staticmethod
    def _should_emit_tick_diagnostics(
        tick: ScheduledTick, effects: PlannerEffects
    ) -> bool:
        diag = effects.diagnostics
        return (
            tick.run_load_scaling
            or tick.run_throughput_scaling
            or effects.scale_to is not None
            or bool(diag.audit_events)
            or bool(diag.short_circuit_reason)
        )

    async def _run_one_tick(
        self, engine: EngineProtocol, tick: ScheduledTick
    ) -> ScheduledTick:
        """Execute one complete environment-to-scaling planner tick."""
        await self._refresh_and_update_capabilities()
        tick_input = await self._observe_tick(engine, tick)
        self._publish_observation_metrics(tick_input)
        self._publish_inventory_and_gpu_hours(tick_input)
        if tick_input.worker_counts is not None:
            self._last_worker_counts = tick_input.worker_counts

        effects = await engine.tick(tick, tick_input)
        await self._apply_effects(effects)
        if self._should_sweep_power_annotations(time.time(), effects):
            await self._apply_power_annotations()
        emit_diagnostics = self._should_emit_tick_diagnostics(tick, effects)
        if emit_diagnostics:
            self._report_diagnostics(tick, effects.diagnostics)
            self._log_decision_summary(effects)

        if self._recorder.enabled and emit_diagnostics:
            try:
                self._recorder.record(
                    tick_input,
                    effects,
                    self.environment.metrics_state(),
                    self._cumulative_gpu_hours,
                )
                if self._recorder.should_generate_report(tick_input.now_s):
                    self._recorder.generate_report()
            except Exception as exc:
                logger.error("Diagnostics report failed: %s", exc)

        assert effects.next_tick is not None
        return effects.next_tick

    async def run(self) -> None:
        engine = self._ensure_engine()
        next_tick = engine.initial_tick(time.time())
        poll_interval = self.config.load_adjustment_interval_seconds / 10

        try:
            while True:
                now = time.time()
                if now < next_tick.at_s:
                    await asyncio.sleep(min(next_tick.at_s - now, poll_interval))
                    continue

                next_tick = await self._run_one_tick(engine, next_tick)
        finally:
            self._recorder.finalize()
            await self.environment.shutdown()
            if self._dashboard_runner is not None:
                await self._dashboard_runner.cleanup()
            if self._engine is not None:
                await self._engine.shutdown()
