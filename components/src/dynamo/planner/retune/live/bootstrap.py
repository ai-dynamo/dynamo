# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the live retune supervisor of a Kubernetes planner.

``build_live_retune(host)`` runs in ``NativePlannerBase._async_init`` after
``environment.initialize()`` and before the engine is built. It reads the
planner's DGD, parses the frontend's startup router configuration with a
pure function of the DGD spec, evaluates the spec premises, and returns a
``LiveRetuneSupervisor`` whose ``retune_inputs()`` the engine's first
controller is built from (``None`` while P4 fails: the adapter then starts
with retune disabled) and whose ``hooks()`` every controller gets.

This module imports the whole live retune stack (Kubernetes client, router
actuator, reconciler, DGDR executor, window cutter).
``NativePlannerBase`` imports it only when the planner runs live retune or
must clean up after a removed one; ``build_supervisor`` takes the
collaborators injectable for tests.

``build_legacy_cleanup(host, kube)`` is the clean-up for a planner whose
retune was removed. ``NativePlannerBase`` calls it in the background, only
after finding that a previous live retune left its Leases behind; it returns
a supervisor in decommission mode that rolls back what that retune deployed,
then deletes its Requests and Leases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import socket
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from dynamo.llm import KvRouterConfig
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import (
    DgdrExecutorConfig,
    PlannerConfig,
    RetuneConfig,
    RetuneEngineSpec,
)
from dynamo.planner.core.types import RetuneInputs
from dynamo.planner.errors import ComponentError
from dynamo.planner.monitoring.dgd_services import get_component_from_type_or_name
from dynamo.planner.retune.domains import expand_router_domain
from dynamo.planner.retune.live import frontend_config, window_cut
from dynamo.planner.retune.live.actuator import (
    AiohttpRouterPolicyClient,
    RouterActuator,
)
from dynamo.planner.retune.live.contracts import (
    DgdRef,
    FrontendStartup,
    KubeAlreadyExists,
    KubeInvalid,
    PremiseReport,
    RetuneKube,
    RetuneMetricsSink,
    WindowCutter,
)
from dynamo.planner.retune.live.dgdr import DgdrContext, Shard, build_request
from dynamo.planner.retune.live.kube import KubernetesRetuneKube
from dynamo.planner.retune.live.metrics import RetuneMetrics
from dynamo.planner.retune.live.reconciler import Reconciler
from dynamo.planner.retune.live.state import fingerprint
from dynamo.planner.retune.live.supervisor import (
    ActuatorFactory,
    LiveRetuneSupervisor,
    SpecEvaluation,
    SupervisorSettings,
    worker_types_for,
)

logger = logging.getLogger(__name__)

# Shard id and window of the P11 canary Request (dry run only, never stored).
CANARY_SHARD_ID = "canary"
CANARY_WINDOW_END_MS = 0


# ------------------------------------------------------------- resolver


class KvRouterResolver:
    """``RouterResolver`` over the parsed startup ``KvRouterConfig`` JSON.

    Resolves exactly as the retune controller does: the startup value of a
    searched key is its serde value, or its aggregated resolution when the
    startup leaves it unset; a point's per worker type parameters are
    ``policy_parameters(wt)`` of ``startup ⊕ point``.
    """

    def __init__(
        self, router_config: Mapping[str, Any], worker_types: Sequence[str]
    ) -> None:
        self._router_config = dict(router_config)
        self._worker_types = tuple(worker_types)

    def startup_flat(self, keys: Sequence[str]) -> dict[str, float]:
        startup = KvRouterConfig.from_json(json.dumps(self._router_config))
        serialized = json.loads(startup.to_json())
        resolved = startup.policy_parameters("aggregated")
        values: dict[str, float] = {}
        for key in keys:
            value = serialized.get(key)
            values[key] = float(resolved[key] if value is None else value)
        return values

    def by_wt(self, flat: Mapping[str, float]) -> dict[str, dict[str, float]]:
        config = KvRouterConfig.from_json(json.dumps({**self._router_config, **flat}))
        return {
            worker_type: {
                key: float(value)
                for key, value in dict(config.policy_parameters(worker_type)).items()
            }
            for worker_type in self._worker_types
        }


# ------------------------------------------------------------ DGD reads


def worker_spec_replicas(
    dgd: Mapping[str, Any], planner_mode: str
) -> Optional[dict[str, int]]:
    """DGD spec replicas per worker type of the planner mode.

    ``None`` when a worker component cannot be resolved or has no replicas.
    An omitted ``replicas`` is the workload default of 1.
    """

    roles = (
        (("prefill", SubComponentType.PREFILL), ("decode", SubComponentType.DECODE))
        if planner_mode == "disagg"
        else (("aggregated", SubComponentType.DECODE),)
    )
    replicas: dict[str, int] = {}
    for worker_type, role in roles:
        try:
            service = get_component_from_type_or_name(dict(dgd), role)
        except ComponentError as error:
            logger.warning(
                "retune: cannot resolve the %s workers: %s", worker_type, error
            )
            return None
        raw = service.service.get("replicas")
        try:
            count = 1 if raw is None else int(raw)
        except (TypeError, ValueError):
            return None
        if count < 1:
            return None
        replicas[worker_type] = count
    return replicas


def build_retune_inputs(
    config: PlannerConfig,
    startup: FrontendStartup,
    *,
    dgd_name: str,
    model_name: Optional[str],
) -> RetuneInputs:
    """Live ``RetuneInputs``: the parsed startup, traces on the claim.

    The engine pins only the router (the DGDR Run's engine comes from
    ``retune.dgdr.engine``); ``prefill/decode_engine_args`` are empty maps in
    disagg so the controller's engine-shape check holds. Live trace time is
    epoch milliseconds (origin 0, speedup 1).
    """

    _, dgdr = _dgdr_retune(config)
    disagg = config.mode == "disagg"
    engine = RetuneEngineSpec(
        router_mode=startup.router_mode,  # type: ignore[arg-type]
        router_config=dict(startup.router_config),
        model_name=model_name or None,
        prefill_engine_args={} if disagg else None,
        decode_engine_args={} if disagg else None,
        extra_engine_args=None,
    )
    trace = posixpath.join(dgdr.mount_path, dgdr.segments_dir.strip("/"), dgd_name)
    return RetuneInputs(
        engine=engine,
        trace_paths=(trace,),
        trace_origin_ms=0.0,
        arrival_speedup_ratio=1.0,
    )


def evaluate_spec(
    dgd: Mapping[str, Any],
    *,
    config: PlannerConfig,
    k8s_namespace: str,
    dynamo_namespace: str,
    frontend: Any = None,
) -> SpecEvaluation:
    """Parse the frontend and evaluate the spec premises of one DGD read.

    ``frontend`` is the ``frontend_config`` module (injectable for tests).
    """

    if frontend is None:
        frontend = frontend_config
    metadata = dgd.get("metadata") or {}
    name = str(metadata.get("name") or "")
    uid = metadata.get("uid")
    if not name or not uid:
        raise ValueError("the DGD has no metadata.name or metadata.uid")
    ref = DgdRef(
        name=name,
        namespace=str(metadata.get("namespace") or k8s_namespace),
        uid=str(uid),
        dynamo_namespace=dynamo_namespace,
    )
    parsed = frontend.parse_frontend(dgd, planner_mode=config.mode)
    startup = parsed if isinstance(parsed, FrontendStartup) else None
    report: PremiseReport = frontend.evaluate_premises(dgd, planner_config=config)
    inputs = None
    if report.p4_ok and startup is not None:
        try:
            inputs = build_retune_inputs(
                config,
                startup,
                dgd_name=name,
                model_name=frontend.served_model_name(dgd),
            )
        except ValueError as error:
            logger.error("retune: building the live retune inputs failed: %s", error)
            report = PremiseReport(
                p4_ok=False,
                spec_ok=False,
                reasons=(*report.reasons, f"P4: {error}"),
                warnings=report.warnings,
                startup=report.startup,
            )
    generation = metadata.get("generation")
    return SpecEvaluation(
        report=report,
        startup=startup,
        inputs=inputs,
        spec_replicas=worker_spec_replicas(dgd, config.mode),
        dgd=ref,
        generation=generation if isinstance(generation, int) else None,
    )


# --------------------------------------------------------- DGDR pieces


def dgdr_context_factory(
    config: PlannerConfig, cut_window: Optional[WindowCutter] = None
) -> Callable[[LiveRetuneSupervisor], Any]:
    """Bind a ``dgdr.DgdrContext`` to the supervisor's current DGD."""

    def factory(supervisor: LiveRetuneSupervisor) -> DgdrContext:
        retune, dgdr = _dgdr_retune(config)
        return DgdrContext(
            dgd=supervisor.dgd,
            cfg=dgdr,
            retune=retune,
            backend=str(config.backend),
            kube=supervisor.kube,
            registry=supervisor.registry,
            cut_window=cut_window or window_cut.cut_window,
            is_leader=supervisor.is_leader,
        )

    return factory


def dgdr_canary(
    config: PlannerConfig, *, timeout_s: float = 10.0
) -> Callable[[LiveRetuneSupervisor], bool]:
    """P11: v1beta2 accepts ``workload.trace`` (a Strict dry-run create).

    The canary is the single-point Request of the startup router point that
    a real window would create, so a CRD or webhook that drops or rejects any
    field the executor relies on fails it. ``KubeInvalid`` is a failed
    premise; other API errors propagate (the supervisor counts them as
    failing and retries next round).
    """

    def canary(supervisor: LiveRetuneSupervisor) -> bool:
        retune, dgdr = _dgdr_retune(config)
        replicas = supervisor.evaluation.spec_replicas
        point = supervisor.startup_flat()
        if not replicas or not point:
            return False
        sla_ttft_ms, sla_itl_ms = retune.scoring_sla(config.ttft_ms, config.itl_ms)
        shard = Shard(
            id=CANARY_SHARD_ID,
            kind="point",
            choices={key: (value,) for key, value in point.items()},
            points=(dict(point),),
        )
        body = build_request(
            dgd=supervisor.dgd,
            cfg=dgdr,
            retune=retune,
            sla_ttft_ms=sla_ttft_ms,
            sla_itl_ms=sla_itl_ms,
            window_start_ms=CANARY_WINDOW_END_MS - 1,
            window_end_ms=CANARY_WINDOW_END_MS,
            as_of_ms=CANARY_WINDOW_END_MS,
            shard=shard,
            spec_replicas=replicas,
            backend=str(config.backend),
            attempt=0,
        )
        try:
            supervisor.kube.create_request(body, dry_run=True, timeout_s=timeout_s)
        except KubeAlreadyExists:
            return True
        except KubeInvalid as error:
            logger.error(
                "retune P11: the DGDR v1beta2 API rejects the Request: %s", error
            )
            return False
        return True

    return canary


def default_actuator_factory(
    kube: RetuneKube, metrics: RetuneMetricsSink
) -> ActuatorFactory:
    def factory(
        dgd: DgdRef, startup_maps: dict[str, dict[str, float]]
    ) -> RouterActuator:
        return RouterActuator(
            kube, AiohttpRouterPolicyClient(), dgd, startup_maps, metrics
        )

    return factory


def planner_identity() -> str:
    """``<pod>/<process nonce>``: the Lease holder identity of this process.

    The nonce makes a restarted container a new holder, so it never renews
    the Lease of its previous incarnation.
    """

    pod = (
        os.environ.get("POD_NAME")
        or os.environ.get("DYN_POD_NAME")
        or os.environ.get("HOSTNAME")
        or socket.gethostname()
    )
    return f"{pod}/{uuid.uuid4().hex[:12]}"


def searched_router_keys(config: PlannerConfig) -> tuple[str, ...]:
    """Live names of the searched router keys (independent of the startup)."""

    retune = config.retune
    if retune is None:
        return ()
    return tuple(sorted(expand_router_domain(retune)))


def _dgdr_retune(config: PlannerConfig) -> tuple[RetuneConfig, DgdrExecutorConfig]:
    """``config.retune`` and its DGDR executor section, which must be set."""

    retune = config.retune
    if retune is None or retune.dgdr is None:
        raise ValueError("live retune requires retune.executor 'dgdr'")
    return retune, retune.dgdr


def _runtime_namespace(host: Any) -> str:
    """The routers' dynamo namespace, else the planner's configured one."""

    value = host.environment.runtime_namespace()
    return str(value) if value else str(host.config.namespace)


# --------------------------------------------------------------- build


async def build_supervisor(
    config: PlannerConfig,
    *,
    kube: RetuneKube,
    dgd_name: str,
    namespace_fn: Callable[[], str],
    metrics: Optional[RetuneMetricsSink] = None,
    identity: Optional[str] = None,
    frontend: Any = None,
    actuator_factory: Optional[ActuatorFactory] = None,
    reconciler: Any = None,
    canary: Optional[Callable[[LiveRetuneSupervisor], bool]] = None,
    context_factory: Optional[Callable[[LiveRetuneSupervisor], Any]] = None,
    sweep_runner: Optional[Callable[..., Any]] = None,
    **supervisor_kwargs: Any,
) -> LiveRetuneSupervisor:
    """A supervisor for ``dgd_name`` with the production collaborators by default."""

    retune = config.retune
    if retune is None or retune.executor != "dgdr" or retune.dgdr is None:
        raise ValueError(
            "live retune requires retune.executor 'dgdr' (retune.executor "
            f"is {None if retune is None else retune.executor!r})"
        )
    dgdr = retune.dgdr
    if metrics is None:
        metrics = RetuneMetrics()
    dgd_obj = await asyncio.to_thread(kube.get_dgd, dgd_name)

    def evaluate(obj: dict) -> SpecEvaluation:
        return evaluate_spec(
            obj,
            config=config,
            k8s_namespace=kube.namespace,
            dynamo_namespace=namespace_fn(),
            frontend=frontend,
        )

    evaluation = evaluate(dgd_obj)
    if reconciler is None:
        reconciler = Reconciler(
            kube, evaluation.dgd, dgdr, retune, dgdr.mount_path, metrics
        )
    worker_types = worker_types_for(config.mode)
    settings = SupervisorSettings.from_config(
        config, router_keys=searched_router_keys(config)
    )
    supervisor = LiveRetuneSupervisor(
        settings=settings,
        kube=kube,
        evaluation=evaluation,
        identity=identity or planner_identity(),
        premise_evaluator=evaluate,
        actuator_factory=actuator_factory or default_actuator_factory(kube, metrics),
        fingerprint_fn=lambda startup: fingerprint(config, startup),
        resolver_factory=lambda startup: (
            None
            if startup is None
            else KvRouterResolver(startup.router_config, worker_types)
        ),
        reconciler=reconciler,
        metrics=metrics,
        canary=canary or dgdr_canary(config),
        sweep_runner=sweep_runner,
        context_factory=context_factory or dgdr_context_factory(config),
        **supervisor_kwargs,
    )
    report = evaluation.report
    logger.info(
        "live retune for DGD %s/%s (uid %s, dynamo namespace %s) as %s: "
        "P4 %s, spec_ok %s, spec replicas %s, advisory %s",
        evaluation.dgd.namespace,
        evaluation.dgd.name,
        evaluation.dgd.uid,
        evaluation.dgd.dynamo_namespace,
        supervisor.identity,
        "ok" if report.p4_ok else "failed",
        report.spec_ok,
        evaluation.spec_replicas,
        settings.advisory,
    )
    return supervisor


async def build_live_retune(host: Any) -> LiveRetuneSupervisor:
    """The supervisor of a ``NativePlannerBase`` with ``retune.executor: dgdr``.

    ``host.environment.controller`` is the planner's ``KubernetesConnector``;
    its ``kube_api`` client and ``parent_dgd_name`` name the DGD.
    """

    config = host.config
    retune = config.retune
    if retune is None or retune.executor != "dgdr":
        raise ValueError(
            "build_live_retune requires retune.executor 'dgdr'; executor "
            f"{None if retune is None else retune.executor!r} is replay-only"
        )
    connector = host.environment.controller
    kube = KubernetesRetuneKube.from_kube_api(connector.kube_api)
    return await build_supervisor(
        config,
        kube=kube,
        dgd_name=connector.parent_dgd_name,
        namespace_fn=lambda: _runtime_namespace(host),
    )


async def build_legacy_cleanup(
    host: Any, kube: RetuneKube, *, timeout_s: float = 10.0
) -> LiveRetuneSupervisor:
    """A decommission supervisor for the Leases a removed retune left.

    ``NativePlannerBase`` calls it once it found a Lease of the DGD. Run it
    like the live supervisor (``start``/``stop``): it rolls back what the
    old retune deployed, then deletes the DGD's retune Requests and both
    Leases and stops itself (``decommissioned``). Reading the DGD may raise
    ``KubeError``.
    """

    connector = host.environment.controller
    dgd_name = connector.parent_dgd_name
    dgd_obj = await asyncio.to_thread(kube.get_dgd, dgd_name, timeout_s=timeout_s)
    logger.info(
        "retune legacy clean-up: DGD %s has Leases of a removed live retune; "
        "rolling back what it deployed, then deleting its Requests and Leases",
        dgd_name,
    )
    return build_decommission_supervisor(
        host.config,
        kube=kube,
        dgd=dgd_obj,
        dynamo_namespace=_runtime_namespace(host),
    )


def build_decommission_supervisor(
    config: PlannerConfig,
    *,
    kube: RetuneKube,
    dgd: Mapping[str, Any],
    dynamo_namespace: str,
    metrics: Optional[RetuneMetricsSink] = None,
    actuator_factory: Optional[ActuatorFactory] = None,
    frontend: Any = None,
    identity: Optional[str] = None,
    **supervisor_kwargs: Any,
) -> LiveRetuneSupervisor:
    """A supervisor that only rolls back and deletes (no retune config)."""

    if frontend is None:
        frontend = frontend_config
    if metrics is None:
        metrics = RetuneMetrics()
    metadata = dgd.get("metadata") or {}
    ref = DgdRef(
        name=str(metadata.get("name")),
        namespace=str(metadata.get("namespace") or kube.namespace),
        uid=str(metadata.get("uid")),
        dynamo_namespace=dynamo_namespace,
    )
    parsed = frontend.parse_frontend(dgd, planner_mode=config.mode)
    startup = parsed if isinstance(parsed, FrontendStartup) else None

    def evaluate(obj: dict) -> SpecEvaluation:
        return SpecEvaluation(
            report=PremiseReport(p4_ok=False, spec_ok=True),
            startup=startup,
            inputs=None,
            spec_replicas=None,
            dgd=ref,
        )

    evaluation = evaluate(dict(dgd))
    settings = SupervisorSettings(
        interval_s=3600.0,
        grace_s=0.0,
        timeout_s=3000.0,
        advisory=True,
        worker_types=worker_types_for(config.mode),
        decommission=True,
    )
    return LiveRetuneSupervisor(
        settings=settings,
        kube=kube,
        evaluation=evaluation,
        identity=identity or planner_identity(),
        premise_evaluator=evaluate,
        actuator_factory=actuator_factory or default_actuator_factory(kube, metrics),
        fingerprint_fn=lambda _: "decommission",
        metrics=metrics,
        **supervisor_kwargs,
    )
