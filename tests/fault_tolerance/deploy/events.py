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
    "RunCommand",
    "NetworkPartition",
    "WaitForModelReady",
]
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions as k8s_exceptions

from tests.utils.managed_load import LoadConfig, ManagedLoad

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

        self._managed_load = ManagedLoad(
            namespace=ctx.namespace,
            load_config=self.load_config,
            pvc_name=ctx.deployment.get_log_pvc_name(),
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
class DeletePod(Event):
    """Delete pods for specified services."""

    services: list[str]
    force: bool = True
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(f"Deleting pods for services: {self.services}")
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                ctx.logger.info(f"Deleting pod {pod.name} (service: {service_name})")
                # Pod manifest (status/conditions/restartCount) and metrics
                # endpoint scrape have to happen pre-delete; the PVC log
                # tee already captures the container's stdout/stderr so we
                # don't fetch kubectl logs here separately.
                ctx.deployment._get_pod_manifest(pod, service_name, ".before_delete")
                await ctx.deployment._get_pod_metrics(
                    pod, service_name, ".before_delete"
                )
                pod.delete(force=self.force)

    @property
    def description(self) -> str:
        return f"Delete pods: {', '.join(self.services)}"


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
    process_name: str  # e.g., "dynamo.runtime", "python", etc.
    signal: str = "SIGKILL"
    name: str = ""
    results: dict[str, Any] | None = field(default=None, init=False)

    async def execute(self, ctx: "ScenarioContext") -> None:
        ctx.logger.info(
            f"Terminating process '{self.process_name}' in services: {self.services}"
        )
        service_pod_dict = ctx.deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                processes = ctx.deployment.get_processes(pod)
                for proc in processes:
                    if self.process_name in proc.command:
                        ctx.logger.info(
                            f"Killing process {proc.pid} ({proc.command[:50]}...) "
                            f"with {self.signal}"
                        )
                        proc.kill(signal=self.signal)
                        break

    @property
    def description(self) -> str:
        return f"Terminate '{self.process_name}' in {', '.join(self.services)}"


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
