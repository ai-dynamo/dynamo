# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario events for fault tolerance testing.

Events are actions executed in sequence during a test scenario.
Each event has:
- execute(ctx): Perform the action
- stop(ctx): Optional cleanup, called after all events execute
- name: Event identifier (used to reference from other events)
- results: Optional results stored after execution
- description: Human-readable description for logging

To create a custom event, subclass Event and implement execute() and description.
"""

import asyncio
import random
import secrets
from datetime import datetime, timezone

__all__ = [
    "Event",
    "StartLoad",
    "StopLoad",
    "WaitForLoadCompletion",
    "Wait",
    "DeletePod",
    "WaitForRecovery",
    "RollingUpgrade",
    "WaitForLogPattern",
    "TerminateProcess",
    "StallProcess",
    "RunCommand",
    "NetworkPartition",
    "WaitForModelReady",
    "PrintProcessTree",
    "RstInjection",
    "RstFromInsidePod",
    "PodMemoryPoller",
    "RANDOM",
    "ALL",
]

# Sentinels for target selection on DeletePod / StallProcess /
# TerminateProcess. Resolution happens inside ``execute()`` so the
# choice is logged and reproducible from the test output.
#
#   RANDOM  -- pick exactly one item at random
#   ALL     -- target every item in the candidate set
#
# Apply to ``pod_indices`` (pod selection) and/or ``rank_index``
# (rank/process selection within each chosen pod). For ``rank_index``
# the legacy default of None means "first matching process" so existing
# tests are unchanged; pass ALL to fan out to every matching rank.
RANDOM = "random"
ALL = "all"


def _get_restart_count(pod) -> int:
    """Read containerStatuses[0].restartCount from a Pod, defaulting to 0."""
    if pod is None:
        return -1
    try:
        statuses = pod.raw.get("status", {}).get("containerStatuses") or []
        if not statuses:
            return 0
        return int(statuses[0].get("restartCount", 0))
    except Exception:
        return 0


def _get_terminated_reason(pod) -> str | None:
    """Return lastState.terminated.reason if present (e.g. 'OOMKilled', 'Error')."""
    if pod is None:
        return None
    try:
        statuses = pod.raw.get("status", {}).get("containerStatuses") or []
        if not statuses:
            return None
        last = (statuses[0].get("lastState") or {}).get("terminated") or {}
        return last.get("reason")
    except Exception:
        return None


def _write_verification_line(ctx, line: str) -> None:
    """Append a one-liner to ctx.log_dir/fault_verification.txt + ctx.logger."""
    import os

    ctx.logger.info(f"VERIFY: {line}")
    log_dir = getattr(ctx, "log_dir", None)
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "fault_verification.txt"), "a") as fh:
            fh.write(line + "\n")
    except Exception as e:
        ctx.logger.warning(f"could not write fault_verification.txt: {e}")


def _resolve_pod_selection(pods, pod_indices, logger=None):
    """Resolve ``pod_indices`` selection against the sorted pod list.

    ``pods`` is the input list (already sorted by name by callers).
    ``pod_indices`` may be:
      - None     → all pods (legacy default)
      - ALL      → all pods (explicit; same as None)
      - [i, ...] → the pods at those indices; out-of-range skipped
      - RANDOM   → exactly one pod, chosen uniformly at random
    """
    if pod_indices is None or pod_indices == ALL:
        return pods
    if pod_indices == RANDOM:
        if not pods:
            return []
        chosen = random.choice(pods)
        if logger is not None:
            logger.info(f"pod_indices=RANDOM resolved to pod '{chosen.name}'")
        return [chosen]
    return [pods[i] for i in pod_indices if 0 <= i < len(pods)]


def _resolve_rank_selection(matches, rank_index, pod_name, logger=None):
    """Resolve ``rank_index`` selection against the sorted-by-pid match list.

    ``rank_index`` may be:
      - None     → caller's legacy default: first match
      - n (int)  → the nth match; out-of-range returns []
      - RANDOM   → one match chosen uniformly at random
      - ALL      → every match
    Returns a list of zero or more chosen processes.
    """
    if not matches:
        return []
    if rank_index is None:
        return [matches[0]]
    if rank_index == ALL:
        return list(matches)
    if rank_index == RANDOM:
        chosen = random.choice(matches)
        if logger is not None:
            logger.info(
                f"rank_index=RANDOM resolved to pid={chosen.pid} " f"on pod {pod_name}"
            )
        return [chosen]
    if 0 <= rank_index < len(matches):
        return [matches[rank_index]]
    if logger is not None:
        logger.info(
            f"rank_index={rank_index} out of range for pod "
            f"{pod_name} ({len(matches)} matches); skipping"
        )
    return []


from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

import aiohttp  # noqa: E402
from kubernetes_asyncio import client  # noqa: E402
from kubernetes_asyncio.client import exceptions as k8s_exceptions  # noqa: E402

from tests.utils.managed_load import LoadConfig, ManagedLoad  # noqa: E402

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Event Base Class
# =============================================================================


@dataclass
class Event(ABC):
    """Base class for scenario events.

    Authoring an event: subclass and implement ``execute(ctx)`` (and a
    ``description`` property). Optionally override ``stop(ctx)`` for
    cleanup. The framework calls ``timed_execute(ctx)`` which wraps the
    subclass's ``execute`` with wall-clock timestamping so reports can
    bucket aiperf records by event boundary.

    Attributes set by the framework during execution:
    - started_at / ended_at: UTC-aware wall-clock bracket populated by
      ``timed_execute``. ``None`` until the event runs.
    """

    # Note: name and results are defined in subclasses since dataclasses
    # require fields with defaults to come after fields without defaults.

    @abstractmethod
    async def execute(self, ctx: "ScenarioContext") -> None:
        """Subclass override — perform the event's action."""
        pass

    async def timed_execute(self, ctx: "ScenarioContext") -> None:
        """Framework entry point — runs ``execute`` and records the
        wall-clock bracket on ``self.started_at``/``self.ended_at``.
        """
        self.started_at = datetime.now(timezone.utc)
        try:
            await self.execute(ctx)
        finally:
            self.ended_at = datetime.now(timezone.utc)

    async def stop(self, ctx: "ScenarioContext") -> None:
        """Optional stop/cleanup. Called after all events execute."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the event."""
        pass


# =============================================================================
# Load Events
# =============================================================================


@dataclass
class StartLoad(Event):
    """Start a load test.

    Creates and starts a ManagedLoad. Results are available from:
    - stop() method (auto-called after all events)
    - StopLoad event (for early termination)
    - WaitForLoadCompletion event (explicit wait)
    """

    load_config: LoadConfig
    name: str = "default"
    results: dict[str, Any] | None = field(default=None, init=False)
    _managed_load: ManagedLoad | None = field(default=None, init=False, repr=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(f"Creating load '{self.name}'...")

        # Auto-populate per-pod /metrics URLs. Workers expose dynamo_*
        # + vllm:* on system_port (9090). Frontends expose dynamo_frontend_*
        # on spec.port (8000). User-supplied extra_server_metrics_urls
        # overrides.
        #
        # Before 2026-05-12 the frontend was skipped here and only the
        # Service URL was scraped — that returns metrics from ONE pod
        # picked by Service round-robin per scrape, so we lost per-pod
        # attribution in the time-series. Including each FE pod's IP
        # gives a per-pod series like the worker side already does.
        if self.load_config.extra_server_metrics_urls is None:
            spec = ctx.deployment.deployment_spec
            urls: list[str] = []
            shortfall: list[str] = []
            for service in spec.services:
                expected = service.replicas
                if expected <= 0:
                    continue
                # Retry once with a short backoff to tolerate a transient
                # missing podIP — pods are Ready by here, but the kr8s
                # snapshot can race the status subresource update.
                pods: list = []
                for attempt in range(2):
                    pods_by_service = await asyncio.to_thread(
                        ctx.deployment.get_pods, [service.name]
                    )
                    pods = pods_by_service.get(service.name) or []
                    ready_with_ip = [
                        p for p in pods if p.raw.get("status", {}).get("podIP")
                    ]
                    if len(ready_with_ip) >= expected:
                        pods = ready_with_ip
                        break
                    if attempt == 0:
                        await asyncio.sleep(2)
                metrics_port = (
                    spec.port
                    if service.component_type == "frontend"
                    else spec.system_port
                )
                seen_for_service = 0
                for pod in pods:
                    pod_ip = pod.raw.get("status", {}).get("podIP")
                    if pod_ip:
                        urls.append(f"http://{pod_ip}:{metrics_port}/metrics")
                        seen_for_service += 1
                ctx.logger.info(
                    f"StartLoad: service={service.name} "
                    f"expected_replicas={expected} pods_with_podIP={seen_for_service}"
                )
                if seen_for_service < expected:
                    shortfall.append(f"{service.name}: {seen_for_service}/{expected}")
            if shortfall:
                raise RuntimeError(
                    "StartLoad: auto-discovery undercount — some replicas "
                    "missing /metrics URLs. AIPerf cannot scrape full fleet. "
                    f"Shortfalls: {shortfall}. "
                    "Pass an explicit LoadConfig.extra_server_metrics_urls "
                    "or wait for all pods Ready before this event."
                )
            if urls:
                ctx.logger.info(f"StartLoad: auto-discovered /metrics URLs: {urls}")
                self.load_config.extra_server_metrics_urls = urls

        self._managed_load = ManagedLoad(
            namespace=ctx.namespace,
            load_config=self.load_config,
            pvc_name=ctx.deployment.get_log_pvc_name(),
            pvc_run_id=ctx.deployment.get_log_run_id(),
            endpoint_url=ctx.deployment.deployment_spec.get_in_cluster_frontend_url(
                ctx.namespace
            ),
            log_dir=ctx.log_dir,
            job_name=f"load-{self.name}-{secrets.token_hex(4)}",
        )
        await self._managed_load._init_kubernetes()
        await self._managed_load.run(wait_for_completion=False)

        ctx.logger.info(f"Waiting for load '{self.name}' to start...")
        await self._managed_load.wait_for_started()
        ctx.logger.info(f"Load '{self.name}' started")

    async def stop(self, ctx: "ScenarioContext") -> None:
        """Wait for load to complete and collect results."""
        if self._managed_load:
            ctx.logger.info(f"Stopping load '{self.name}'...")
            await self._managed_load.wait_for_completion()
            self.results = await self._managed_load.get_results()
            await self._managed_load._cleanup()
            self._managed_load = None
            ctx.logger.info(f"Load '{self.name}' stopped")

    def is_active(self) -> bool:
        """Check if load is currently running."""
        return self._managed_load is not None

    @property
    def description(self) -> str:
        return f"Start load '{self.name}'"


@dataclass
class StopLoad(Event):
    """Stop a running load early and collect results.

    Use this to terminate a load before it completes naturally.
    For loads that complete on their own, use WaitForLoadCompletion instead.
    """

    name: str = "default"
    results: dict[str, Any] | None = field(default=None, init=False)

    def _get_start_load(self, ctx: "ScenarioContext") -> StartLoad:
        for event in ctx.events:
            if isinstance(event, StartLoad) and event.name == self.name:
                return event
        raise ValueError(f"Load '{self.name}' not found")

    async def execute(self, ctx: "ScenarioContext") -> None:
        start_load = self._get_start_load(ctx)
        if not start_load._managed_load:
            raise ValueError(f"Load '{self.name}' not active")

        ctx.logger.info(f"Stopping load '{self.name}'...")
        await start_load._managed_load.terminate()
        start_load.results = await start_load._managed_load.get_results()
        await start_load._managed_load._cleanup()
        start_load._managed_load = None
        ctx.logger.info(f"Load '{self.name}' stopped")

    @property
    def description(self) -> str:
        return f"Stop load '{self.name}'"


@dataclass
class WaitForLoadCompletion(Event):
    """Wait for a load to complete naturally and collect results.

    Use this after StartLoad to wait for the load to finish and get results.
    """

    name: str = "default"
    timeout: int | None = None
    results: dict[str, Any] | None = field(default=None, init=False)

    def _get_start_load(self, ctx: "ScenarioContext") -> StartLoad:
        for event in ctx.events:
            if isinstance(event, StartLoad) and event.name == self.name:
                return event
        raise ValueError(f"Load '{self.name}' not found")

    async def execute(self, ctx: "ScenarioContext") -> None:
        start_load = self._get_start_load(ctx)
        if not start_load._managed_load:
            raise ValueError(f"Load '{self.name}' not active")

        ctx.logger.info(f"Waiting for load '{self.name}' to complete...")
        await start_load._managed_load.wait_for_completion(timeout=self.timeout)
        start_load.results = await start_load._managed_load.get_results()
        await start_load._managed_load._cleanup()
        start_load._managed_load = None
        ctx.logger.info(f"Load '{self.name}' completed")

    @property
    def description(self) -> str:
        return f"Wait for load '{self.name}' completion"


# =============================================================================
# Basic Events
# =============================================================================


@dataclass
class Wait(Event):
    """Wait for a specified duration."""

    duration: int  # seconds
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(f"Waiting {self.duration}s...")
        await asyncio.sleep(self.duration)

    @property
    def description(self) -> str:
        return f"Wait {self.duration}s"


@dataclass
class SetBusyThreshold(Event):
    """POST /busy_threshold to the Frontend service. Affects only the named
    model. Subsequent requests will receive 503 once *all* workers for the
    model exceed the thresholds."""

    model_name: str
    active_decode_blocks_threshold: float = 0.85
    active_prefill_tokens_threshold_frac: float = 0.85
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx) -> None:
        import json

        pods = (await asyncio.to_thread(ctx.deployment.get_pods, ["Frontend"])).get(
            "Frontend"
        ) or []
        if not pods:
            raise RuntimeError("SetBusyThreshold: no Frontend pods")
        fe = pods[0]
        body = {
            "model": self.model_name,
            "active_decode_blocks_threshold": self.active_decode_blocks_threshold,
            "active_prefill_tokens_threshold_frac": self.active_prefill_tokens_threshold_frac,
        }
        pf = await asyncio.to_thread(ctx.deployment.port_forward, fe, 8000)
        if pf is None or pf.local_port == 0:
            raise RuntimeError(
                f"SetBusyThreshold: port-forward to {fe.name}:8000 failed"
            )
        url = f"http://127.0.0.1:{pf.local_port}/busy_threshold"
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=body, timeout=30) as r:
                txt = await r.text()
                if r.status != 200:
                    raise RuntimeError(
                        f"SetBusyThreshold: POST returned {r.status}: {txt}"
                    )
                ctx.logger.info(f"SetBusyThreshold: {self.model_name} -> {txt}")
                self.results = json.loads(txt)

    @property
    def description(self) -> str:
        return (
            f"SetBusyThreshold(model={self.model_name}, "
            f"decode_frac={self.active_decode_blocks_threshold}, "
            f"prefill_frac={self.active_prefill_tokens_threshold_frac})"
        )


@dataclass
class DeletePod(Event):
    """Delete pods for specified services."""

    services: list[str]
    force: bool = True
    # Pod selection within each service (after sort by name):
    #   None     -> all pods (legacy default)
    #   [i, j]   -> pods at those indices; out-of-range silently skipped
    #   RANDOM   -> exactly one pod, chosen uniformly at random per run
    pod_indices: list[int] | str | None = None
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(f"Deleting pods for services: {self.services}")
        # Pre-fault snapshot of pod-name set per service, so verify() can
        # confirm a replacement pod appears with a NEW name.
        baseline_names: dict[str, set] = {}
        deleted_names: list[str] = []
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            all_sorted = sorted(pods, key=lambda p: p.name)
            baseline_names[service_name] = {p.name for p in all_sorted}
            pods_to_delete = _resolve_pod_selection(
                all_sorted, self.pod_indices, ctx.logger
            )
            for pod in pods_to_delete:
                ctx.logger.info(f"Deleting pod {pod.name} (service: {service_name})")
                ctx.deployment._get_pod_manifest(pod, service_name, ".before_delete")
                await ctx.deployment._get_pod_metrics(
                    pod, service_name, ".before_delete"
                )
                pod.delete(force=self.force)
                deleted_names.append(pod.name)
        self._baseline_names = baseline_names
        self._deleted_names = deleted_names
        await self._verify(ctx)

    async def _verify(self, ctx: "ScenarioContext") -> None:
        if not self._deleted_names:
            _write_verification_line(
                ctx, f"DeletePod(name={self.name}): NO PODS MATCHED — no-op"
            )
            raise RuntimeError(
                f"DeletePod for services {self.services} matched zero pods"
            )
        # Poll for replacement up to 60s.
        deadline = asyncio.get_event_loop().time() + 60.0
        replaced = {n: False for n in self._deleted_names}
        while asyncio.get_event_loop().time() < deadline:
            live = ctx.deployment.get_pods(self.services)
            live_by_svc = {s: {p.name for p in pods} for s, pods in live.items()}
            for name in self._deleted_names:
                if replaced[name]:
                    continue
                svc = next(
                    (s for s, names in self._baseline_names.items() if name in names),
                    None,
                )
                if svc is None:
                    continue
                live_set = live_by_svc.get(svc, set())
                # A replacement is "any pod in this service whose name was
                # not in the pre-delete baseline and is not the deleted name".
                fresh = (live_set - self._baseline_names[svc]) - set(
                    self._deleted_names
                )
                if name not in live_set and fresh:
                    replaced[name] = True
            if all(replaced.values()):
                break
            await asyncio.sleep(2.0)
        for name, ok in replaced.items():
            _write_verification_line(
                ctx,
                f"DeletePod(name={self.name}) pod={name} "
                f"{'REPLACED' if ok else 'NO-REPLACEMENT-WITHIN-60s'}",
            )
        missing = [n for n, ok in replaced.items() if not ok]
        if missing:
            raise RuntimeError(
                f"DeletePod: no replacement appeared for {missing} within "
                f"60s — controller may be broken or pod is stuck terminating"
            )

    @property
    def description(self) -> str:
        return f"Delete pods: {', '.join(self.services)}"

    _baseline_names: dict = field(default_factory=dict, init=False, repr=False)
    _deleted_names: list = field(default_factory=list, init=False, repr=False)


@dataclass
class WaitForRecovery(Event):
    """Wait for deployment to recover after a failure."""

    timeout: int = 600
    unready_timeout: int = 60
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import time

        start_time = time.time()
        ctx.logger.info("Waiting for deployment to become unready...")
        await ctx.deployment.wait_for_unready(
            timeout=self.unready_timeout, log_interval=10
        )
        ctx.logger.info(f"Waiting for recovery (timeout: {self.timeout}s)...")
        await ctx.deployment.wait_for_ready(timeout=self.timeout)
        duration = time.time() - start_time
        ctx.logger.info(f"Deployment recovered in {duration:.1f}s")

    @property
    def description(self) -> str:
        return f"Wait for recovery (timeout: {self.timeout}s)"


@dataclass
class RollingUpgrade(Event):
    """Trigger a rolling upgrade for specified services."""

    services: list[str]
    name: str = ""
    unready_timeout: int = 60
    ready_timeout: int = 1800
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import time

        start_time = time.time()
        ctx.logger.info(f"Triggering rolling upgrade for: {self.services}")

        # Set trigger env var on each service
        for service_name in self.services:
            service = ctx.deployment.deployment_spec[service_name]
            service.set_env_var("TEST_ROLLING_UPDATE_TRIGGER", secrets.token_hex(8))

        await ctx.deployment.apply_service_changes(self.services)

        ctx.logger.info("Waiting for CR to become unready...")
        await ctx.deployment.wait_for_unready(
            timeout=self.unready_timeout, log_interval=10
        )

        ctx.logger.info("Waiting for CR to become ready...")
        await ctx.deployment.wait_for_ready(timeout=self.ready_timeout)

        duration = time.time() - start_time
        ctx.logger.info(f"Rolling upgrade completed in {duration:.1f}s")

        if self.name:
            self.results = {"services": self.services, "duration_seconds": duration}

    @property
    def description(self) -> str:
        return f"Rolling upgrade: {', '.join(self.services)}"


@dataclass
class WaitForLogPattern(Event):
    """Wait for a pattern to appear in a service's logs."""

    service: str
    pattern: str
    name: str = ""
    timeout: int = 300
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import re
        import time

        start_time = time.time()
        ctx.logger.info(
            f"Waiting for pattern '{self.pattern}' in {self.service} logs..."
        )

        # Get pods for service
        service_pods = ctx.deployment.get_pods([self.service])
        pods = service_pods.get(self.service, [])
        if not pods:
            raise ValueError(f"No pods found for service '{self.service}'")

        # Compile pattern
        regex = re.compile(self.pattern)

        # Poll logs until pattern found or timeout
        poll_interval = 2
        while time.time() - start_time < self.timeout:
            for pod in pods:
                try:
                    logs = pod.logs(since_seconds=10)
                    if regex.search(logs):
                        duration = time.time() - start_time
                        ctx.logger.info(
                            f"Pattern found in {self.service} after {duration:.1f}s"
                        )
                        if self.name:
                            self.results = {
                                "pattern": self.pattern,
                                "service": self.service,
                                "found_in_pod": pod.name,
                                "duration_seconds": duration,
                            }
                        return
                except Exception as e:
                    ctx.logger.debug(f"Error reading logs from {pod.name}: {e}")

            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Pattern '{self.pattern}' not found in {self.service} logs "
            f"after {self.timeout}s"
        )

    @property
    def description(self) -> str:
        return f"Wait for '{self.pattern}' in {self.service} logs"


@dataclass
class TerminateProcess(Event):
    """Terminate a process by name in service pods."""

    services: list[str]
    process_name: str  # e.g., "VLLM::Worker", "VLLM::EngineCore", "dynamo.runtime"
    signal: str = "SIGKILL"
    # Pod / rank selection — see DeletePod and the module-level
    # ``RANDOM`` sentinel for semantics. None = all pods / first match
    # respectively (legacy defaults).
    pod_indices: list[int] | str | None = None
    rank_index: int | str | None = None
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(
            f"Terminating process '{self.process_name}' in services: {self.services}"
        )
        # Pre-fault snapshot of (pod_name -> restartCount) so verify()
        # can detect whether the kill actually triggered a container
        # restart. If we never observe a count delta on any targeted
        # pod, the fault was a silent no-op (regex matched nothing,
        # or kubelet didn't notice).
        baseline: dict[str, int] = {}
        targeted: list[Any] = []
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            pods = sorted(pods, key=lambda p: p.name)
            pods = _resolve_pod_selection(pods, self.pod_indices, ctx.logger)
            for pod in pods:
                processes = ctx.deployment.get_processes(pod)
                matches = [p for p in processes if self.process_name in p.command]
                matches.sort(key=lambda p: p.pid)
                chosen = _resolve_rank_selection(
                    matches, self.rank_index, pod.name, ctx.logger
                )
                if not chosen:
                    continue
                baseline[pod.name] = _get_restart_count(pod)
                targeted.append(pod)
                for proc in chosen:
                    ctx.logger.info(
                        f"Killing process {proc.pid} ({proc.command[:50]}...) "
                        f"on pod {pod.name} with {self.signal}"
                    )
                    proc.kill(signal=self.signal)
        # Stash for verify()
        self._baseline_restarts = baseline
        self._targeted_pod_names = [p.name for p in targeted]
        # Block briefly so verify() has a fair chance to observe the
        # restart-count bump. Kubelet typically restarts within 1-5 s.
        if targeted:
            await asyncio.sleep(15.0)
        await self._verify(ctx)

    async def _verify(self, ctx: "ScenarioContext") -> None:
        if not self._targeted_pod_names:
            _write_verification_line(
                ctx,
                f"TerminateProcess(name={self.name or self.process_name}): "
                f"NO PROCESSES MATCHED — fault was a no-op",
            )
            raise RuntimeError(
                f"TerminateProcess '{self.process_name}' matched zero "
                f"processes; nothing was killed"
            )
        live = ctx.deployment.get_pods(self.services)
        live_by_name = {p.name: p for pods in live.values() for p in pods}
        results = []
        no_bump = []
        for name, before in self._baseline_restarts.items():
            pod = live_by_name.get(name)
            after = _get_restart_count(pod) if pod else -1
            reason = _get_terminated_reason(pod) if pod else None
            results.append((name, before, after, reason))
            if after <= before:
                no_bump.append(name)
        for name, before, after, reason in results:
            _write_verification_line(
                ctx,
                f"TerminateProcess(name={self.name or self.process_name}) "
                f"pod={name} restartCount {before}->{after} "
                f"lastTerminatedReason={reason}",
            )
        if no_bump:
            raise RuntimeError(
                f"TerminateProcess: kill sent but restartCount did NOT "
                f"increment on pods {no_bump} within 15s — likely silent "
                f"no-op (wrong process_name regex, or kubelet doesn't "
                f"track this child)"
            )

    @property
    def description(self) -> str:
        return f"Terminate '{self.process_name}' in {', '.join(self.services)}"

    _baseline_restarts: dict = field(default_factory=dict, init=False, repr=False)
    _targeted_pod_names: list = field(default_factory=list, init=False, repr=False)


@dataclass
class StallProcess(Event):
    """Pause a process by name in service pods (SIGSTOP), then resume (SIGCONT).

    Models a "hung worker": the process stays alive — TCP sockets,
    filesystem handles, conntrack entries are all preserved — but it
    stops servicing requests because it doesn't get scheduled. Useful
    for testing how the frontend handles a worker that accepts
    connections, never replies, and eventually un-hangs (e.g. a GPU
    deadlock that resolves itself, a long GC pause, a debugger).

    Differences vs. ``TerminateProcess`` (SIGKILL):
      * Pod IP and conntrack entries survive (no reconnect storm).
      * No container restart — kubelet doesn't notice.
      * The frontend sees idle TCP connections that never advance,
        so timeout / health-check behaviour is what's exercised
        (not crash recovery).

    Behaviour:
      * If ``duration`` is set, the event SIGSTOPs the matching
        processes, sleeps for ``duration`` seconds, then SIGCONTs them.
        Healing happens inside ``execute()`` so a transient stall reads
        as a single line in the scenario.
      * If ``duration`` is None, the stall holds until ``stop()`` runs
        at end-of-scenario.

    Example::

        StallProcess(
            services=["VllmDecodeWorker"],
            process_name="dynamo.vllm",
            duration=20,
        )
    """

    services: list[str]
    process_name: str
    duration: float | None = None  # seconds; None = hold until scenario stop()
    # Pod / rank selection — see DeletePod / TerminateProcess and the
    # module-level ``RANDOM`` sentinel for semantics.
    pod_indices: list[int] | str | None = None
    rank_index: int | str | None = None
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)
    _stalled_pids: list[tuple[Any, int]] = field(
        default_factory=list, init=False, repr=False
    )

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(
            f"Stalling process '{self.process_name}' (SIGSTOP) "
            f"in services: {self.services}"
        )
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            pods = sorted(pods, key=lambda p: p.name)
            pods = _resolve_pod_selection(pods, self.pod_indices, ctx.logger)
            for pod in pods:
                processes = ctx.deployment.get_processes(pod)
                matches = [p for p in processes if self.process_name in p.command]
                matches.sort(key=lambda p: p.pid)
                chosen = _resolve_rank_selection(
                    matches, self.rank_index, pod.name, ctx.logger
                )
                for proc in chosen:
                    ctx.logger.info(
                        f"SIGSTOP pid={proc.pid} ({proc.command[:50]}...) "
                        f"on pod {pod.name}"
                    )
                    proc.kill(signal="SIGSTOP")
                    self._stalled_pids.append((pod, proc.pid))

        if not self._stalled_pids:
            _write_verification_line(
                ctx,
                f"StallProcess(name={self.name or self.process_name}): "
                f"NO PROCESSES MATCHED — fault was a no-op",
            )
            raise RuntimeError(
                f"StallProcess '{self.process_name}' matched zero processes; "
                f"nothing was stalled"
            )

        # Verify state == 'T' on each stalled pid via /proc/<pid>/stat.
        # Tolerate transient state mismatches (e.g. 'Z' if a process
        # exited racy) by retrying up to 3 times.
        await self._verify_state(ctx, expected="T", phase="post-SIGSTOP")

        if self.duration is not None:
            ctx.logger.info(f"StallProcess: holding for {self.duration}s, then SIGCONT")
            try:
                await asyncio.sleep(self.duration)
            finally:
                await self._resume(ctx)

    async def stop(self, ctx: "ScenarioContext") -> None:
        # When duration is set, execute() already healed the stall.
        # Otherwise stop() resumes it at end of scenario.
        await self._resume(ctx)

    async def _resume(self, ctx: "ScenarioContext") -> None:
        if not self._stalled_pids:
            return
        resumed_snapshot = list(self._stalled_pids)
        for pod, pid in self._stalled_pids:
            try:
                await asyncio.to_thread(pod.exec, ["kill", "-SIGCONT", str(pid)])
                ctx.logger.info(f"SIGCONT pid={pid} on pod {pod.name}")
            except Exception as e:
                ctx.logger.warning(
                    f"StallProcess resume failed for pid={pid} on "
                    f"pod {pod.name}: {e}"
                )
        self._stalled_pids = []
        # Verify the resume: state should no longer be 'T'. Tolerant of
        # process exit (state missing — also acceptable since the
        # process is no longer stalled).
        await self._verify_state_pairs(
            ctx, resumed_snapshot, not_expected="T", phase="post-SIGCONT"
        )

    async def _verify_state(self, ctx, expected: str, phase: str) -> None:
        await self._verify_state_pairs(
            ctx, self._stalled_pids, expected=expected, phase=phase
        )

    async def _verify_state_pairs(
        self,
        ctx,
        pairs,
        *,
        expected: str | None = None,
        not_expected: str | None = None,
        phase: str,
    ) -> None:
        """Check /proc/<pid>/stat field 3 on each (pod, pid) tuple."""
        failures: list[str] = []
        for pod, pid in pairs:
            state = await self._read_proc_state(pod, pid)
            ok = True
            if expected is not None and state != expected:
                ok = False
            if not_expected is not None and state == not_expected:
                ok = False
            _write_verification_line(
                ctx,
                f"StallProcess(name={self.name or self.process_name}) "
                f"{phase} pod={pod.name} pid={pid} state={state} "
                f"{'OK' if ok else 'FAIL'}",
            )
            if not ok:
                failures.append(f"{pod.name}/pid={pid} state={state}")
        if failures:
            raise RuntimeError(
                f"StallProcess {phase} verification failed: "
                f"{failures} (expected {expected}, "
                f"not {not_expected})"
            )

    async def _read_proc_state(self, pod, pid: int) -> str:
        """Return /proc/<pid>/stat field-3 single-char state, or 'missing'."""
        try:
            result = await asyncio.to_thread(
                pod.exec, ["sh", "-c", f"cat /proc/{pid}/stat || echo MISSING"]
            )
            stdout = (
                result.stdout.decode() if hasattr(result, "stdout") else str(result)
            )
            if "MISSING" in stdout or not stdout.strip():
                return "missing"
            # /proc/<pid>/stat format: pid (comm) state ... — comm can contain spaces,
            # so split on ')' first to isolate the state field.
            tail = stdout.split(")", 1)
            if len(tail) < 2:
                return "?"
            after = tail[1].strip().split()
            return after[0] if after else "?"
        except Exception:
            return "missing"

    @property
    def description(self) -> str:
        suffix = f" for {self.duration}s" if self.duration else ""
        return f"Stall '{self.process_name}' in {', '.join(self.services)}" f"{suffix}"


@dataclass
class RunCommand(Event):
    """Run an arbitrary command in service pod(s).

    Use for custom fault injection that doesn't have a dedicated event.

    Example:
        RunCommand(services=["VllmWorker"], command="stress --vm 1 --vm-bytes 2G --timeout 30s")
    """

    services: list[str]
    command: str
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                ctx.logger.info(f"Running '{self.command}' in {pod.name}")
                await asyncio.to_thread(pod.exec, ["sh", "-c", self.command])

    @property
    def description(self) -> str:
        return f"Run '{self.command[:50]}' in {', '.join(self.services)}"


@dataclass
class PrintProcessTree(Event):
    """Dump the full process tree of one or more services' pods.

    Use this to identify the launcher vs the rank/worker subprocesses
    before targeting them with StallProcess / TerminateProcess. Runs
    ``ps -eo pid,ppid,pgid,cmd --forest`` inside each pod and logs the
    output line-by-line.

    Example::

        PrintProcessTree(services=["VllmDecodeWorker", "VllmPrefillWorker"])
    """

    services: list[str]
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import os

        out_path = None
        if getattr(ctx, "log_dir", None):
            os.makedirs(ctx.log_dir, exist_ok=True)
            out_path = os.path.join(ctx.log_dir, "process_tree.md")
            fh = open(out_path, "w")
            fh.write(f"# Process trees — {', '.join(self.services)}\n\n")
        else:
            fh = None

        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in sorted(pods, key=lambda p: p.name):
                header = f"=== process tree: {service_name} / {pod.name} ==="
                ctx.logger.info(header)
                if fh:
                    fh.write(f"\n## {service_name} / {pod.name}\n\n```\n")
                try:
                    result = await asyncio.to_thread(
                        pod.exec,
                        ["ps", "-eo", "pid,ppid,pgid,cmd", "--forest"],
                    )
                    stdout = (
                        result.stdout.decode()
                        if hasattr(result, "stdout")
                        else str(result)
                    )
                    for line in stdout.splitlines():
                        ctx.logger.info(f"  {line}")
                        if fh:
                            fh.write(line + "\n")
                except Exception as e:
                    ctx.logger.warning(f"ps failed on {pod.name}: {e}")
                    if fh:
                        fh.write(f"ps failed: {e}\n")
                if fh:
                    fh.write("```\n")

        if fh:
            fh.close()
            ctx.logger.info(f"Process trees written to {out_path}")

    @property
    def description(self) -> str:
        return f"Print process tree in {', '.join(self.services)}"


@dataclass
class NetworkPartition(Event):
    """Block traffic from one service's pods to another via a NetworkPolicy
    plus a one-shot conntrack flush that severs already-established sockets.

    Applies a NetworkPolicy that selects the *target* service's pods and
    permits ingress from every pod in the namespace EXCEPT the *source*
    service's pods, effectively cutting the source→target communication
    path. The policy is removed at ``stop()`` so the partition lasts only
    for the scenario.

    NetworkPolicy alone is connection-tracked: established TCP sockets
    survive policy creation and only *new* connections get blocked. The
    Dynamo TCP request plane keeps a long-lived pooled socket between
    Frontend and worker pods, so a partition mid-load would otherwise be
    a no-op. After applying the policy, this event schedules a
    privileged hostNetwork pod on the source pod's node and runs
    ``conntrack -D`` to flush flows for both directions, forcing the
    runtime to reconnect — and the reconnect is what the fresh policy
    actually denies.

    Requires a CNI that enforces NetworkPolicy (k3s with built-in
    kube-router, Calico, Cilium, etc.) and an image with conntrack-tools
    (default: ``localhost:32000/netshoot:latest``; override via
    ``DYN_CONNTRACK_IMAGE``).

    Example::

        NetworkPartition(source="decode", target="Frontend")
    """

    source: str
    target: str
    duration: float | None = None  # seconds; None means hold until scenario stop()
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)
    _policy_name: str = field(default="", init=False, repr=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        await self._apply_policy(ctx)
        if self.duration is not None:
            # Transient partition: create the cut, wait the requested
            # window, then heal — all inside a single event so the
            # scenario timeline stays linear ("Wait" + "DeletePod" + ...
            # without needing a separate HealNetworkPartition step).
            ctx.logger.info(
                f"NetworkPartition: holding for {self.duration}s, then healing"
            )
            try:
                await asyncio.sleep(self.duration)
            finally:
                await self._delete_policy(ctx)

    async def stop(self, ctx: "ScenarioContext") -> None:
        # When duration is set, execute() already healed the partition.
        # When duration is None, the policy stays active until end-of-
        # scenario and stop() is what tears it down.
        await self._delete_policy(ctx)

    async def _apply_policy(self, ctx: "ScenarioContext") -> None:
        # Scope the policy to this deployment via the operator's stable
        # graph-deployment-name label, then match the per-service
        # ``nvidia.com/dynamo-component`` label (which uses the spec's
        # service name verbatim, no per-deployment hash). The
        # ``nvidia.com/selector`` label looks tempting but the operator
        # appends a hash to it on workers, so cross-deployment lookups
        # using a constructed value silently miss.
        deployment = ctx.deployment.deployment_spec.name
        self._policy_name = (
            f"partition-{self.source.lower()}-{self.target.lower()}-"
            f"{secrets.token_hex(4)}"
        )
        body = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": self._policy_name,
                "namespace": ctx.namespace,
                "labels": {
                    "managed-by": "managed-deployment",
                    "purpose": "network-partition",
                },
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "nvidia.com/dynamo-graph-deployment-name": deployment,
                        "nvidia.com/dynamo-component": self.target,
                    }
                },
                "policyTypes": ["Ingress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "podSelector": {
                                    "matchExpressions": [
                                        {
                                            "key": "nvidia.com/dynamo-graph-deployment-name",
                                            "operator": "In",
                                            "values": [deployment],
                                        },
                                        {
                                            "key": "nvidia.com/dynamo-component",
                                            "operator": "NotIn",
                                            "values": [self.source],
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                ],
            },
        }
        api = client.NetworkingV1Api()
        await api.create_namespaced_network_policy(namespace=ctx.namespace, body=body)
        ctx.logger.info(
            f"Network partition active: {self.source} -X-> {self.target} "
            f"(policy={self._policy_name})"
        )

        # Sever any existing source<->target sockets so the policy
        # actually bites — see class docstring for why.
        await self._flush_conntrack(ctx)

    async def _flush_conntrack(self, ctx: "ScenarioContext") -> None:
        """Run ``conntrack -D`` on the source pod's node for source<->target flows.

        Picks the source pod's node, schedules a privileged hostNetwork
        pod on that node, and flushes both directions of the
        source<->target TCP flow from the kernel's conntrack table. The
        flush is fire-and-forget — we wait for the flusher pod to
        ``Succeeded`` once and then delete it. If conntrack isn't on
        the node or the image won't pull, we log the failure but don't
        fail the scenario; the policy is still in place and a strict
        test can assert separately.
        """
        import os

        src_pods = (
            await asyncio.to_thread(ctx.deployment.get_pods, [self.source])
        ).get(self.source) or []
        tgt_pods = (
            await asyncio.to_thread(ctx.deployment.get_pods, [self.target])
        ).get(self.target) or []
        if not src_pods or not tgt_pods:
            ctx.logger.warning(
                "NetworkPartition: cannot flush conntrack — missing pods "
                f"(source={len(src_pods)}, target={len(tgt_pods)})"
            )
            return

        src_pod = src_pods[0].raw
        tgt_ips = [
            p.raw.get("status", {}).get("podIP")
            for p in tgt_pods
            if p.raw.get("status", {}).get("podIP")
        ]
        src_ip = src_pod.get("status", {}).get("podIP")
        node = src_pod.get("spec", {}).get("nodeName")
        if not (src_ip and node and tgt_ips):
            ctx.logger.warning(
                "NetworkPartition: cannot flush conntrack — pod IP/node "
                f"not yet ready (src_ip={src_ip}, node={node}, "
                f"tgt_ips={tgt_ips})"
            )
            return

        flusher_image = os.environ.get(
            "DYN_CONNTRACK_IMAGE", "localhost:32000/netshoot:latest"
        )
        # One conntrack -D per direction per target IP. Exit 0 even if
        # nothing matched (-D returns non-zero when 0 rows deleted, so
        # we OR with true to keep the pod's exit code clean).
        cmds = []
        for tip in tgt_ips:
            cmds.append(
                f"conntrack -D -p tcp --orig-src {src_ip} --orig-dst {tip} || true"
            )
            cmds.append(
                f"conntrack -D -p tcp --orig-src {tip} --orig-dst {src_ip} || true"
            )
        cmds.append("echo conntrack-flush done")
        flusher_name = f"conntrack-flush-{secrets.token_hex(4)}"
        body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": flusher_name,
                "namespace": ctx.namespace,
                "labels": {
                    "managed-by": "managed-deployment",
                    "purpose": "conntrack-flush",
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "hostNetwork": True,
                "nodeName": node,
                "tolerations": [{"operator": "Exists"}],
                "containers": [
                    {
                        "name": "flusher",
                        "image": flusher_image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["sh", "-c", " ; ".join(cmds)],
                        "securityContext": {
                            "privileged": True,
                            "capabilities": {"add": ["NET_ADMIN"]},
                        },
                    }
                ],
            },
        }
        core = client.CoreV1Api()
        try:
            await core.create_namespaced_pod(namespace=ctx.namespace, body=body)
        except k8s_exceptions.ApiException as e:
            ctx.logger.warning(
                f"NetworkPartition: failed to schedule conntrack flusher: {e}"
            )
            return

        # Poll until the pod is Succeeded / Failed (cap at 30s — the
        # actual conntrack flush is sub-second; the rest is image pull
        # + scheduling).
        deadline = asyncio.get_event_loop().time() + 30
        terminal = ("Succeeded", "Failed")
        last_phase = "Pending"
        while asyncio.get_event_loop().time() < deadline:
            try:
                pod = await core.read_namespaced_pod(
                    name=flusher_name, namespace=ctx.namespace
                )
                last_phase = pod.status.phase
                if last_phase in terminal:
                    break
            except k8s_exceptions.ApiException:
                pass
            await asyncio.sleep(0.5)
        ctx.logger.info(
            f"NetworkPartition: conntrack flush pod {flusher_name} "
            f"phase={last_phase}"
        )
        try:
            await core.delete_namespaced_pod(
                name=flusher_name,
                namespace=ctx.namespace,
                grace_period_seconds=0,
            )
        except k8s_exceptions.ApiException:
            pass

    async def _delete_policy(self, ctx: "ScenarioContext") -> None:
        if not self._policy_name:
            return
        try:
            api = client.NetworkingV1Api()
            await api.delete_namespaced_network_policy(
                name=self._policy_name, namespace=ctx.namespace
            )
            ctx.logger.info(f"Network partition lifted: {self._policy_name}")
        except k8s_exceptions.ApiException as e:
            if e.status != 404:
                ctx.logger.warning(f"NetworkPartition cleanup: {e.status} {e.reason}")
        self._policy_name = ""

    @property
    def description(self) -> str:
        return f"Network partition: {self.source} -X-> {self.target}"


@dataclass
class RstInjection(Event):
    """Force TCP RSTs to/from one pod for ``duration`` seconds.

    Models the precondition of PR #8254: under high concurrency, a TCP
    stream read error (``ECONNRESET`` / "Connection reset by peer")
    used to ``panic!`` in ``tcp/client.rs`` / ``tcp/server.rs``, killing
    13–130 Tokio tasks per failing run. This event reproduces that
    precondition deterministically:

      1. Resolve a target pod (``pod_indices`` semantics same as the
         other targeting events; defaults to all pods of ``service``).
         For each chosen pod:
      2. Find the pod's IP and the node it's scheduled on.
      3. Schedule a privileged ``hostNetwork`` helper pod on that node
         that installs ``iptables -A FORWARD -d <pod_ip> -p tcp
         -j REJECT --reject-with tcp-reset`` (and a matching ``-s``
         rule for the return path). Every TCP packet to / from the
         target pod is replied to with an RST until the rule is
         removed.
      4. The helper pod sleeps for ``duration`` seconds while the
         rule is in place, then removes it (best-effort) and exits.
         ``stop()`` deletes the helper pod regardless.
      5. Verification: the event waits until the helper pod reaches
         Running (or fails loudly if it never does), and captures
         the helper's stdout (iptables exit codes + rule list) into
         ``ctx.log_dir/rst_injection_verification.txt`` after the
         duration elapses. If the iptables installation failed the
         event raises.

    History: an earlier iteration of this event used ``tcpkill`` from
    netshoot, which silently no-op'd because netshoot doesn't actually
    ship tcpkill. The ``|| echo`` fallback masked the missing-binary
    error and goodput came in *above* baseline — the giveaway that no
    RSTs had actually been injected. iptables is in the base image and
    works without external dependencies.

    Example::

        RstInjection(
            service="VllmDecodeWorker",
            pod_indices=[0],
            duration=30.0,
        )

    Requires a CNI where pod-to-pod traffic traverses the host's
    FORWARD chain. AWS VPC CNI does (pod traffic goes through the
    host's networking stack on the ENI). Most overlay CNIs (Calico,
    Cilium, kube-router) do too. Override the image via
    ``DYN_CONNTRACK_IMAGE``.
    """

    service: str
    duration: float = 30.0
    pod_indices: list[int] | str | None = None
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)
    _helper_pod_names: list[str] = field(default_factory=list, init=False, repr=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import os

        all_pods = (
            await asyncio.to_thread(ctx.deployment.get_pods, [self.service])
        ).get(self.service) or []
        all_pods = sorted(all_pods, key=lambda p: p.name)
        target_pods = _resolve_pod_selection(all_pods, self.pod_indices, ctx.logger)
        if not target_pods:
            ctx.logger.warning(
                f"RstInjection: no pods to target in service '{self.service}'"
            )
            return

        helper_image = os.environ.get(
            "DYN_CONNTRACK_IMAGE", "localhost:32000/netshoot:latest"
        )
        core = client.CoreV1Api()

        for pod in target_pods:
            pod_raw = pod.raw
            pod_ip = pod_raw.get("status", {}).get("podIP")
            node = pod_raw.get("spec", {}).get("nodeName")
            if not (pod_ip and node):
                ctx.logger.warning(
                    f"RstInjection: pod {pod.name} not ready "
                    f"(podIP={pod_ip}, nodeName={node}); skipping"
                )
                continue

            helper_name = f"rst-inject-{secrets.token_hex(4)}"
            duration_int = max(1, int(self.duration))
            # Install REJECT rules for both directions of the pod's TCP
            # flows in the FORWARD chain. set -e so an iptables failure
            # makes the pod exit non-zero and our verification catches
            # it. The verification block below greps the helper's
            # stdout for the RULE-INSTALLED line; missing → raise.
            cmd = (
                f"set -e; "
                f"iptables -I FORWARD -d {pod_ip} -p tcp -j REJECT --reject-with tcp-reset; "
                f"iptables -I FORWARD -s {pod_ip} -p tcp -j REJECT --reject-with tcp-reset; "
                f"echo RULE-INSTALLED for {pod_ip}; "
                f"iptables -S FORWARD | grep tcp-reset; "
                f"sleep {duration_int}; "
                f"iptables -D FORWARD -d {pod_ip} -p tcp -j REJECT --reject-with tcp-reset || true; "
                f"iptables -D FORWARD -s {pod_ip} -p tcp -j REJECT --reject-with tcp-reset || true; "
                f"echo RULE-REMOVED for {pod_ip}"
            )
            body = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": helper_name,
                    "namespace": ctx.namespace,
                    "labels": {
                        "managed-by": "managed-deployment",
                        "purpose": "rst-injection",
                        "target-pod": pod.name[:60],
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "hostNetwork": True,
                    "nodeName": node,
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "rst-inject",
                            "image": helper_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["sh", "-c", cmd],
                            "securityContext": {
                                "privileged": True,
                                "capabilities": {"add": ["NET_ADMIN", "NET_RAW"]},
                            },
                        }
                    ],
                },
            }
            try:
                await core.create_namespaced_pod(namespace=ctx.namespace, body=body)
                self._helper_pod_names.append(helper_name)
                ctx.logger.info(
                    f"RstInjection: launched {helper_name} on node {node} "
                    f"targeting pod {pod.name} ({pod_ip}) for {duration_int}s"
                )
            except k8s_exceptions.ApiException as e:
                ctx.logger.warning(
                    f"RstInjection: failed to schedule helper for "
                    f"pod {pod.name}: {e}"
                )

        # Verify every helper pod actually reached Running before we
        # begin "waiting for the duration". If a pod is stuck in
        # ImagePullBackOff or ContainerCreating, no RSTs are being
        # injected; we want a loud failure, not silent no-op.
        await self._wait_for_running(ctx, core)

        # Hold for ``duration`` seconds so the event blocks the
        # scenario timeline (mirrors StallProcess(duration=…)). The
        # helper pods self-timeout to the same value; stop() then
        # cleans them up.
        try:
            await asyncio.sleep(self.duration)
        finally:
            # Capture each helper's stdout for the verification record
            # BEFORE we delete the pod. Grep for the RULE-INSTALLED
            # / RULE-REMOVED sentinels.
            await self._capture_and_verify(ctx, core)
            await self._cleanup(ctx)

    async def _wait_for_running(self, ctx: "ScenarioContext", core) -> None:
        # 240s — AWS dev image pulls + Pod scheduling can take 60-90s
        # on first hit; 60s was too tight and produced false negatives.
        deadline = asyncio.get_event_loop().time() + 240.0
        not_running = list(self._helper_pod_names)
        while not_running and asyncio.get_event_loop().time() < deadline:
            still = []
            for name in not_running:
                try:
                    p = await core.read_namespaced_pod(
                        name=name, namespace=ctx.namespace
                    )
                    phase = p.status.phase
                    if phase == "Running" or phase == "Succeeded":
                        ctx.logger.info(f"RstInjection: helper {name} reached {phase}")
                        continue
                    if phase == "Failed":
                        ctx.logger.error(
                            f"RstInjection: helper {name} FAILED before "
                            f"installing rules; aborting event"
                        )
                        raise RuntimeError(
                            f"RstInjection helper {name} failed to start"
                        )
                except k8s_exceptions.ApiException:
                    pass
                still.append(name)
            not_running = still
            if not_running:
                await asyncio.sleep(1.0)
        if not_running:
            raise RuntimeError(
                f"RstInjection: helper pods {not_running} did not reach "
                f"Running within 60s; iptables rules never installed"
            )

    async def _capture_and_verify(self, ctx: "ScenarioContext", core) -> None:
        import os

        records = []
        for name in self._helper_pod_names:
            try:
                logs = await core.read_namespaced_pod_log(
                    name=name, namespace=ctx.namespace
                )
            except k8s_exceptions.ApiException as e:
                logs = f"<read_pod_log failed: {e}>"
            installed = "RULE-INSTALLED" in (logs or "")
            removed = "RULE-REMOVED" in (logs or "")
            records.append(
                {
                    "pod": name,
                    "installed": installed,
                    "removed": removed,
                    "logs": logs or "",
                }
            )
            ctx.logger.info(
                f"RstInjection: {name} installed={installed} " f"removed={removed}"
            )

        # Persist a verification record so a post-hoc reader can confirm
        # rules were installed without re-running anything.
        log_dir = getattr(ctx, "log_dir", None)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                with open(
                    os.path.join(log_dir, "rst_injection_verification.txt"), "w"
                ) as fh:
                    for r in records:
                        fh.write(
                            f"=== {r['pod']} installed={r['installed']} "
                            f"removed={r['removed']} ===\n{r['logs']}\n\n"
                        )
            except Exception as e:
                ctx.logger.warning(
                    f"RstInjection: could not write verification file: {e}"
                )

        # Hard fail the event if any helper pod did not install rules.
        not_installed = [r["pod"] for r in records if not r["installed"]]
        if not_installed:
            raise RuntimeError(
                f"RstInjection: iptables rules NOT installed on helper "
                f"pods {not_installed}; no RSTs were injected. "
                f"See rst_injection_verification.txt for stdout."
            )

    async def stop(self, ctx: "ScenarioContext") -> None:
        # Safety net if execute() exited early (exception path).
        await self._cleanup(ctx)

    async def _cleanup(self, ctx: "ScenarioContext") -> None:
        if not self._helper_pod_names:
            return
        core = client.CoreV1Api()
        for name in self._helper_pod_names:
            try:
                await core.delete_namespaced_pod(
                    name=name,
                    namespace=ctx.namespace,
                    grace_period_seconds=0,
                )
                ctx.logger.info(f"RstInjection: deleted helper pod {name}")
            except k8s_exceptions.ApiException:
                pass
        self._helper_pod_names = []

    @property
    def description(self) -> str:
        return (
            f"Inject TCP RSTs against {self.service} (pod_indices="
            f"{self.pod_indices}) for {self.duration}s"
        )


@dataclass
class WaitForModelReady(Event):
    """Wait until the frontend lists ``model_name`` in /v1/models.

    Bridges the gap between "deployment is Ready" (operator's view: all
    pods running) and "model is actually serving" (worker has registered
    with the frontend's discovery layer). For real workers (vllm,
    sglang, trtllm), engine startup + KV-cache warmup + worker-to-
    frontend discovery handshake can add tens of seconds beyond the pod
    Ready transition. Mocker is fast enough this barely matters.

    Pattern follows ``tests/router/helper.py:wait_for_frontend_ready``.

    Args:
        model_name: model identifier to look for in /v1/models. If
            ``None``, derives from the first worker service's ``--model``
            launch arg (i.e. ``deployment_spec[<worker>].model``).
        timeout: seconds to wait for the model to appear before raising
            ``TimeoutError``.
        poll_interval: seconds between /v1/models polls.
    """

    model_name: str | None = None
    timeout: int = 600
    poll_interval: float = 2.0
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import re

        spec = ctx.deployment.deployment_spec
        target_model = self.model_name
        if target_model is None:
            # First pass: clean ServiceSpec.model (works pre-log-wrap).
            for service in spec.services:
                if service.component_type != "frontend" and service.model:
                    target_model = service.model
                    break
        if target_model is None:
            # Second pass: enable_log_collection rewrites the worker
            # command into a bash heredoc, so --model lives inside a
            # string and ServiceSpec.model returns None. Grep the raw
            # command text for --model/--model-path instead.
            pattern = re.compile(r"--(?:model|model-path)[\s=]+(\S+)")
            for service in spec.services:
                if service.component_type == "frontend":
                    continue
                container = service._spec.get("extraPodSpec", {}).get(
                    "mainContainer", {}
                )
                cmd_text = " ".join(container.get("command", []) or []) + " "
                cmd_text += " ".join(container.get("args", []) or [])
                m = pattern.search(cmd_text)
                if m:
                    target_model = m.group(1)
                    break
        if not target_model:
            raise ValueError(
                "WaitForModelReady: no model_name passed and no worker "
                "service has --model/--model-path set"
            )

        # Port-forward to the frontend pod so this works whether the test
        # runs inside the cluster (in-cluster DNS) or outside (host
        # network — *.svc.cluster.local won't resolve). The port-forward
        # is owned by ManagedDeployment._active_port_forwards and gets
        # cleaned up on context exit.
        ctx.logger.info("WaitForModelReady: looking up frontend pod...")
        frontend_name = spec.frontend_service.name
        pods_by_service = await asyncio.to_thread(
            ctx.deployment.get_pods, [frontend_name]
        )
        frontend_pods = pods_by_service.get(frontend_name) or []
        if not frontend_pods:
            raise RuntimeError(
                f"WaitForModelReady: no pods found for frontend service "
                f"{frontend_name!r} in namespace {ctx.namespace}"
            )
        ctx.logger.info(
            f"WaitForModelReady: opening port-forward to "
            f"{frontend_pods[0].name}:{spec.port}..."
        )
        pf = await asyncio.to_thread(
            ctx.deployment.port_forward, frontend_pods[0], spec.port
        )
        if pf is None or pf.local_port == 0:
            raise RuntimeError(
                f"WaitForModelReady: port-forward to "
                f"{frontend_pods[0].name}:{spec.port} failed"
            )
        models_url = f"http://127.0.0.1:{pf.local_port}/v1/models"
        ctx.logger.info(
            f"Waiting for model '{target_model}' at {models_url} "
            f"(timeout={self.timeout}s)"
        )

        start = asyncio.get_event_loop().time()
        last_log = start
        async with aiohttp.ClientSession() as session:
            while True:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > self.timeout:
                    raise TimeoutError(
                        f"WaitForModelReady: '{target_model}' did not appear "
                        f"in /v1/models within {self.timeout}s"
                    )
                try:
                    async with session.get(
                        models_url,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            ids = [m.get("id") for m in data.get("data", [])]
                            if target_model in ids:
                                ctx.logger.info(
                                    f"Model '{target_model}' ready after "
                                    f"{elapsed:.1f}s (registered models: {ids})"
                                )
                                return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass

                # Heartbeat log every 10s so the user knows we're still polling.
                if asyncio.get_event_loop().time() - last_log >= 10:
                    ctx.logger.info(
                        f"  still waiting for '{target_model}' "
                        f"(elapsed {elapsed:.0f}s)"
                    )
                    last_log = asyncio.get_event_loop().time()
                await asyncio.sleep(self.poll_interval)

    @property
    def description(self) -> str:
        target = self.model_name or "(auto-detected)"
        return f"Wait for model '{target}' in /v1/models"


# =============================================================================
# RstFromInsidePod — true ECONNRESET via SO_LINGER=0 + partial-frame write.
#
# Schedules a helper Job in the namespace that runs a small Python client
# inline. The client opens a TCP connection to a target worker pod's
# system_port, half-writes a 24-byte TwoPartCodec frame, then closes
# with SO_LINGER=0 → kernel sends RST instead of FIN → worker's
# handle_reader hits Some(Err(_)) at tcp/client.rs:255 (v1.0.1) → panics.
# Post-PR-#8254: WARN line instead.
#
# This is the upstream-blessed shape from
# dynamo-observe/.../2026-05-03-tcp-client-panic-repro/arm_b_rst.py.
# iptables-FORWARD REJECT (RstInjection) is not on the packet path on
# AWS VPC CNI; this in-pod approach connects directly to the target
# pod IP and is reliable.
# =============================================================================


_RST_CLIENT_SCRIPT = r"""
import os, socket, struct, sys, time  # noqa: E402,E401
host = os.environ["TARGET_HOST"]
ports = [int(p) for p in os.environ["TARGET_PORTS"].split(",") if p.strip()]
count = int(os.environ.get("RST_COUNT", "100"))
inter = float(os.environ.get("RST_INTER_DELAY", "0.005"))
partial = b"\x00\x00\x00\x10" + b"x" * 8
print(f"[rst-client] target={host} ports={ports} count_per_port={count} delay={inter}s", flush=True)
started = time.monotonic()
sent = 0
for i in range(count):
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            s.settimeout(5.0)
            s.connect((host, port))
            s.sendall(partial)
            time.sleep(0.05)
            s.close()
            sent += 1
        except Exception as e:
            print(f"[rst-client] i={i} port={port} err: {e!r}", flush=True)
    time.sleep(inter)
elapsed = time.monotonic() - started
print(f"[rst-client] RST SENT count={sent} elapsed={elapsed:.1f}s "
      f"rate={sent/elapsed:.1f}/s", flush=True)
"""


@dataclass
class RstFromInsidePod(Event):
    """Drive real ECONNRESET to a worker by opening many TCP connections
    from inside the namespace and closing each with SO_LINGER=0 after a
    partial protocol-frame write. The remote socket's read loop sees a
    real ``Io(Os { code: 104, kind: ConnectionReset })`` — the exact
    shape that triggers PR #8254's panic site on v1.0.1.

    Distinct from ``RstInjection`` (iptables REJECT --reject-with
    tcp-reset on the host FORWARD chain): that approach is not on the
    packet path for AWS VPC CNI pod-to-pod traffic, so RSTs never
    actually reach the worker socket. This event makes a normal TCP
    socket from a helper pod IN the namespace; pod-to-pod IP routing
    is direct and the RST always lands.

    Example::

        RstFromInsidePod(
            service="VllmDecodeWorker",
            pod_indices=[0],
            count=200,
            inter_delay=0.005,
            target_port=9090,  # spec.system_port
        )
    """

    service: str
    pod_indices: list[int] | str | None = None
    count: int = 100
    inter_delay: float = 0.01
    # target_port: int|None — if None, auto-discover Dynamo TCP server ports
    # by reading /proc/net/tcp inside the target pod, excluding the framework's
    # status port (9090), NIXL UCX port (5600), and OpenAI port (8000).
    target_port: int | None = None
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)
    _helper_pod_names: list[str] = field(default_factory=list, init=False, repr=False)

    # Ports we know are NOT Dynamo's request-plane TcpStreamServer:
    #   9090 — system_status_server (axum, /live /health /metrics)
    #   5600 — NIXL UCX transport
    #   8000 — Frontend OpenAI HTTP port
    #   80, 443, 22, 53, 5601 etc — common service ports
    _EXCLUDED_PORTS = {9090, 5600, 8000, 80, 443, 22, 53, 5601, 9092, 9093}

    @staticmethod
    def _discover_dynamo_ports(pod) -> list[int]:
        """Read /proc/net/tcp inside the pod and return the set of TCP ports
        in LISTEN state minus framework-known ports. The remaining set is
        the Dynamo TcpStreamServer + dynamic etcd/NATS clients."""
        script = (
            "python3 -c \"with open('/proc/net/tcp') as f:\n"
            "  next(f)\n"
            "  for line in f:\n"
            "    p = line.split()\n"
            "    if p[3] == '0A':\n"
            "      print(int(p[1].split(':')[1], 16))\""
        )
        result = pod.exec(["sh", "-c", script])
        stdout = result.stdout.decode() if hasattr(result, "stdout") else str(result)
        ports = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                port = int(line)
            except ValueError:
                continue
            if port in RstFromInsidePod._EXCLUDED_PORTS:
                continue
            if port < 10000:
                # Below 10000 = framework/static; Dynamo TcpStreamServers
                # bind ephemeral ports in the 32768+ range.
                continue
            ports.append(port)
        return sorted(set(ports))

    async def execute(self, ctx: "ScenarioContext") -> None:
        import os

        all_pods = (
            await asyncio.to_thread(ctx.deployment.get_pods, [self.service])
        ).get(self.service) or []
        all_pods = sorted(all_pods, key=lambda p: p.name)
        target_pods = _resolve_pod_selection(all_pods, self.pod_indices, ctx.logger)
        if not target_pods:
            ctx.logger.warning(
                f"RstFromInsidePod: no target pods in service '{self.service}'"
            )
            return

        # Use a generic python image — no Dynamo dep needed for the client.
        helper_image = os.environ.get("DYN_RST_CLIENT_IMAGE", "python:3.12-slim")
        core = client.CoreV1Api()

        for pod in target_pods:
            pod_ip = pod.raw.get("status", {}).get("podIP")
            if not pod_ip:
                ctx.logger.warning(
                    f"RstFromInsidePod: {pod.name} has no podIP; skipping"
                )
                continue

            if self.target_port is not None:
                ports = [self.target_port]
            else:
                ports = await asyncio.to_thread(self._discover_dynamo_ports, pod)
                ctx.logger.info(
                    f"RstFromInsidePod: discovered {len(ports)} candidate "
                    f"Dynamo TCP ports on {pod.name}: {ports[:10]}"
                    f"{'...' if len(ports) > 10 else ''}"
                )
                if not ports:
                    raise RuntimeError(
                        f"RstFromInsidePod: no Dynamo TCP ports discovered on "
                        f"{pod.name}; pass target_port explicitly to override"
                    )

            name = f"rst-client-{secrets.token_hex(4)}"
            body = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": name,
                    "namespace": ctx.namespace,
                    "labels": {
                        "managed-by": "managed-deployment",
                        "purpose": "rst-from-inside-pod",
                        "target-pod": pod.name[:60],
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "rst-client",
                            "image": helper_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python3", "-c", _RST_CLIENT_SCRIPT],
                            "env": [
                                {"name": "TARGET_HOST", "value": pod_ip},
                                {
                                    "name": "TARGET_PORTS",
                                    "value": ",".join(str(p) for p in ports),
                                },
                                {"name": "RST_COUNT", "value": str(self.count)},
                                {
                                    "name": "RST_INTER_DELAY",
                                    "value": str(self.inter_delay),
                                },
                            ],
                        }
                    ],
                },
            }
            try:
                await core.create_namespaced_pod(namespace=ctx.namespace, body=body)
                self._helper_pod_names.append(name)
                ctx.logger.info(
                    f"RstFromInsidePod: launched {name} targeting {pod.name} "
                    f"({pod_ip}:{self.target_port}) count={self.count}"
                )
            except k8s_exceptions.ApiException as e:
                ctx.logger.warning(
                    f"RstFromInsidePod: failed to schedule helper for {pod.name}: {e}"
                )

        # Wait for each helper to finish (Succeeded or Failed). Cap based
        # on count*inter_delay + 60s buffer.
        deadline = asyncio.get_event_loop().time() + max(
            120.0, self.count * self.inter_delay + 60.0
        )
        pending = list(self._helper_pod_names)
        while pending and asyncio.get_event_loop().time() < deadline:
            still = []
            for name in pending:
                try:
                    p = await core.read_namespaced_pod(
                        name=name, namespace=ctx.namespace
                    )
                    phase = p.status.phase
                    if phase in ("Succeeded", "Failed"):
                        continue
                except k8s_exceptions.ApiException:
                    pass
                still.append(name)
            pending = still
            if pending:
                await asyncio.sleep(2.0)

        # Capture stdout for verification (grep for "RST SENT count=N").
        await self._capture(ctx, core)
        await self._cleanup(ctx, core)

    async def _capture(self, ctx, core) -> None:
        import os

        log_dir = getattr(ctx, "log_dir", None)
        records = []
        for name in self._helper_pod_names:
            try:
                logs = await core.read_namespaced_pod_log(
                    name=name, namespace=ctx.namespace
                )
            except k8s_exceptions.ApiException as e:
                logs = f"<read_pod_log failed: {e}>"
            sent_line = next(
                (ln for ln in (logs or "").splitlines() if "RST SENT count=" in ln), ""
            )
            records.append({"pod": name, "sent_line": sent_line, "logs": logs or ""})
            ctx.logger.info(
                f"RstFromInsidePod: {name} -> {sent_line or '<no SENT line>'}"
            )
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "rst_from_inside_pod.txt"), "w") as fh:
                    for r in records:
                        fh.write(
                            f"=== {r['pod']} ===\nSENT: {r['sent_line']}\n{r['logs']}\n\n"
                        )
            except Exception as e:
                ctx.logger.warning(
                    f"RstFromInsidePod: could not write verification file: {e}"
                )
        not_sent = [r["pod"] for r in records if not r["sent_line"]]
        if not_sent:
            raise RuntimeError(
                f"RstFromInsidePod: helper pods {not_sent} did not emit a "
                f"'RST SENT count=N' line; client never ran or failed to connect."
            )

    async def _cleanup(self, ctx, core) -> None:
        for name in list(self._helper_pod_names):
            try:
                await core.delete_namespaced_pod(
                    name=name, namespace=ctx.namespace, grace_period_seconds=0
                )
            except k8s_exceptions.ApiException:
                pass
        self._helper_pod_names = []

    async def stop(self, ctx: "ScenarioContext") -> None:
        core = client.CoreV1Api()
        await self._cleanup(ctx, core)

    @property
    def description(self) -> str:
        return (
            f"RstFromInsidePod: {self.count} SO_LINGER=0 closes against "
            f"{self.service}:{self.target_port}"
        )


# =============================================================================
# PodMemoryPoller — records BOTH cgroup container memory (metrics.k8s.io,
# same source kubelet uses for OOMKill decisions) AND per-process RSS
# of pid 1 (/proc/1/status:VmRSS, via pod.exec) for each pod, on every
# poll. Two columns let PodMemoryGrowth assert on the right signal:
#
#   - working_set_bytes: what kubelet sees → OOMKill-relevant.
#   - pid1_rss_bytes: what the main binary actually maps → attribution.
#
# The two are usually close on single-process containers like the
# Dynamo frontend, but diverge when child processes are large (e.g.
# vLLM workers).
# =============================================================================


def _parse_k8s_quantity(s: str) -> int:
    """Parse a k8s resource quantity like '123Mi', '12345Ki', '1Gi' to bytes."""
    if not s:
        return 0
    suffixes = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "K": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "T": 1000**4,
    }
    for suf, mult in suffixes.items():
        if s.endswith(suf):
            try:
                return int(float(s[: -len(suf)]) * mult)
            except ValueError:
                return 0
    try:
        return int(s)
    except ValueError:
        return 0


@dataclass
class PodMemoryPoller(Event):
    """Spawn a background task that on every poll, for each pod of
    ``services``, records two memory numbers and appends a row to
    ``ctx.log_dir/pod_memory_growth.tsv``:

    Columns: ``epoch_s, service, pod, container, working_set_bytes, pid1_rss_bytes``.

    - ``working_set_bytes``: from metrics.k8s.io (same as
      `kubectl top pod`; kubelet uses this for OOMKill).
    - ``pid1_rss_bytes``: from `/proc/1/status:VmRSS` inside the
      container (per-process attribution).

    The ``PodMemoryGrowth`` check reads either column.

    Best placed right after ``WaitForModelReady`` so polling starts
    before any fault is injected. Stops automatically when ``stop()``
    fires at scenario teardown.

    metrics-server's scrape interval is ~15s on most clusters — polling
    faster yields duplicate working-set samples but pid1 RSS is fresh
    every call. 15s default matches the metrics-server cadence.
    """

    services: list[str]
    interval_s: float = 15.0
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)
    _task: Any = field(default=None, init=False, repr=False)
    _stop: Any = field(default=None, init=False, repr=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        import os

        log_dir = getattr(ctx, "log_dir", None)
        if not log_dir:
            ctx.logger.warning("PodMemoryPoller: no ctx.log_dir; skipping")
            return
        os.makedirs(log_dir, exist_ok=True)
        out = os.path.join(log_dir, "pod_memory_growth.tsv")
        with open(out, "w") as fh:
            fh.write(
                "epoch_s\tservice\tpod\tcontainer\t"
                "working_set_bytes\tpid1_rss_bytes\n"
            )
        self._stop = asyncio.Event()

        custom = client.CustomObjectsApi()

        async def read_working_set(pod_name):
            """metrics.k8s.io -> dict[container_name, working_set_bytes]."""
            try:
                obj = await custom.get_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=ctx.namespace,
                    plural="pods",
                    name=pod_name,
                )
            except k8s_exceptions.ApiException:
                return {}
            return {
                c.get("name", "?"): _parse_k8s_quantity(
                    (c.get("usage") or {}).get("memory") or "0"
                )
                for c in (obj.get("containers") or [])
            }

        async def read_pid1_rss(pod):
            """`cat /proc/1/status:VmRSS` -> bytes (or 0)."""
            try:
                result = await asyncio.to_thread(
                    pod.exec,
                    ["sh", "-c", "grep VmRSS /proc/1/status 2>/dev/null || true"],
                )
                stdout = (
                    result.stdout.decode() if hasattr(result, "stdout") else str(result)
                )
                for ln in stdout.splitlines():
                    if "VmRSS" in ln:
                        for tok in ln.split():
                            if tok.isdigit():
                                return int(tok) * 1024  # kB → bytes
            except Exception:
                pass
            return 0

        async def loop():
            import time

            while not self._stop.is_set():
                try:
                    for svc in self.services:
                        pods_by_svc = await asyncio.to_thread(
                            ctx.deployment.get_pods, [svc]
                        )
                        for pod in pods_by_svc.get(svc) or []:
                            ws_by_container = await read_working_set(pod.name)
                            pid1_rss = await read_pid1_rss(pod)
                            ts = time.time()
                            if not ws_by_container:
                                ws_by_container = {"?": 0}
                            for cname, ws_bytes in ws_by_container.items():
                                with open(out, "a") as fh:
                                    fh.write(
                                        f"{ts:.0f}\t{svc}\t{pod.name}\t"
                                        f"{cname}\t{ws_bytes}\t{pid1_rss}\n"
                                    )
                except Exception as e:
                    ctx.logger.warning(f"PodMemoryPoller loop err: {e}")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except asyncio.TimeoutError:
                    pass

        self._task = asyncio.create_task(loop())
        ctx.logger.info(
            f"PodMemoryPoller(metrics.k8s.io): services={self.services} "
            f"interval={self.interval_s}s"
        )

    async def stop(self, ctx: "ScenarioContext") -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        ctx.logger.info("PodMemoryPoller: stopped")

    @property
    def description(self) -> str:
        return f"PodMemoryPoller {self.services} every {self.interval_s}s"
