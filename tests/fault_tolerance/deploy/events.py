# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario events for fault tolerance testing.

Events have:
- execute(ctx): Perform the action
- stop(ctx): Optional cleanup, called after all events execute
- name: Event identifier
- results: Optional results stored after execution
"""

import asyncio
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.utils.managed_load import LoadConfig, ManagedLoad

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Event Base Class
# =============================================================================


@dataclass
class Event(ABC):
    """Base class for scenario events.

    Events have:
    - execute(ctx): Perform the action
    - stop(ctx): Optional cleanup, called after all events execute
    - name: Event identifier (used to find related events)
    - results: Optional results stored after execution
    """

    # Note: name and results are defined in subclasses since dataclasses
    # require fields with defaults to come after fields without defaults.

    @abstractmethod
    async def execute(self, ctx: "ScenarioContext") -> None:
        """Execute the event."""
        pass

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
                ctx.deployment.get_pod_manifest_logs_metrics(
                    service_name, pod, ".before_delete"
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
