# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Refactored test framework with explicit Event and Check classes.

Each test function explicitly defines:
1. Deployment configuration
2. Load configuration
3. Events (what happens during the test)
4. Checks (validation after test)

Then calls run_scenario() to execute the test.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pytest

from tests.utils.managed_aiperf_deployment import LoadConfig, ManagedAIPerfDeployment
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

# =============================================================================
# Event Base Class and Implementations
# =============================================================================


@dataclass
class Event(ABC):
    """Base class for test events that occur during a scenario."""

    @abstractmethod
    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        """Execute the event."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the event."""
        pass


@dataclass
class WaitEvent(Event):
    """Wait for a specified duration."""

    duration: int  # seconds

    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        logger.info(f"Waiting {self.duration}s...")
        await asyncio.sleep(self.duration)

    @property
    def description(self) -> str:
        return f"Wait {self.duration}s"


@dataclass
class DeletePodEvent(Event):
    """Delete pods for specified services."""

    services: list[str]
    force: bool = True  # force = no graceful termination

    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        logger.info(f"Deleting pods for services: {self.services}")
        service_pod_dict = deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                logger.info(f"Deleting pod {pod.name} (service: {service_name})")
                deployment.get_pod_manifest_logs_metrics(
                    service_name, pod, ".before_delete"
                )
                pod.delete(force=self.force)

    @property
    def description(self) -> str:
        return f"Delete pods: {', '.join(self.services)}"


@dataclass
class RollingUpgradeEvent(Event):
    """Trigger a rolling upgrade for specified services.

    PodClique pods get NEW NAMES when recreated (unlike StatefulSets).
    This tracks pod names to detect new pods:
    - Record pod names before upgrade
    - Trigger upgrade
    - Wait for pods with NEW names (not in original set) to become ready
    """

    services: list[str]
    wait_for_new_pod_timeout: int = 600  # seconds

    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        # Get pod names before the upgrade
        original_pod_names: set[str] = set()
        pods_dict = deployment.get_pods(self.services)
        for service_name, pods in pods_dict.items():
            for pod in pods:
                original_pod_names.add(pod.name)
        logger.info(f"Pods before rolling upgrade: {original_pod_names}")

        logger.info(f"Triggering rolling upgrade for: {self.services}")
        await deployment.trigger_rolling_upgrade(self.services)

        # Wait for new pods (different names) to appear and become ready
        logger.info(
            f"Waiting for new pods to appear (timeout: {self.wait_for_new_pod_timeout}s)..."
        )

        start_time = asyncio.get_event_loop().time()

        while (
            asyncio.get_event_loop().time() - start_time
        ) < self.wait_for_new_pod_timeout:
            # Get current pods
            current_pods_dict = deployment.get_pods(self.services)
            new_ready_pods = []
            current_pod_names = set()

            for service_name, pods in current_pods_dict.items():
                for pod in pods:
                    pod_name = pod.name
                    current_pod_names.add(pod_name)

                    # Check if this is a NEW pod (not in original set)
                    if pod_name not in original_pod_names:
                        # New pod - check if it's ready
                        if pod.status.phase == "Running":
                            if pod.status.container_statuses:
                                all_ready = all(
                                    cs.ready for cs in pod.status.container_statuses
                                )
                                if all_ready:
                                    new_ready_pods.append(pod_name)
                                    logger.info(f"New pod {pod_name} is ready")

            if new_ready_pods:
                logger.info(f"New pods ready: {new_ready_pods}")
                break

            # Log progress periodically
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            if elapsed % 10 == 0 and elapsed > 0:
                new_pods = current_pod_names - original_pod_names
                logger.info(
                    f"Waiting for new pods... elapsed={elapsed}s, "
                    f"new_pods_seen={new_pods}, current_pods={current_pod_names}"
                )

            await asyncio.sleep(2)
        else:
            raise TimeoutError(
                f"Timed out waiting for new pods after {self.wait_for_new_pod_timeout}s. "
                f"Original pods: {original_pod_names}"
            )

        logger.info("Rolling upgrade completed - new pods are ready")

    @property
    def description(self) -> str:
        return f"Rolling upgrade: {', '.join(self.services)}"


@dataclass
class TerminateProcessEvent(Event):
    """Terminate a specific process in pods."""

    services: list[str]
    process_name: str
    signal: str = "SIGKILL"

    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        logger.info(
            f"Terminating process '{self.process_name}' with {self.signal} in: {self.services}"
        )
        service_pod_dict = deployment.get_pods(self.services)
        for service_name, pods in service_pod_dict.items():
            for pod in pods:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if self.process_name in process.command:
                        logger.info(
                            f"Killing {service_name} pod {pod.name} PID {process.pid}"
                        )
                        process.kill(self.signal)

    @property
    def description(self) -> str:
        return f"Terminate {self.process_name} ({self.signal})"


@dataclass
class WaitForRecoveryEvent(Event):
    """Wait for deployment to recover after a failure."""

    timeout: int = 600  # seconds

    async def execute(
        self,
        deployment: ManagedDeployment,
        load: ManagedAIPerfDeployment,
        logger: logging.Logger,
    ) -> None:
        logger.info("Waiting for deployment to become unready...")
        await deployment.wait_for_unready(timeout=60, log_interval=10)
        logger.info(f"Waiting for recovery (timeout: {self.timeout}s)...")
        await deployment._wait_for_ready(timeout=self.timeout)
        logger.info("Deployment recovered")

    @property
    def description(self) -> str:
        return f"Wait for recovery (timeout: {self.timeout}s)"


# =============================================================================
# Check Base Class and Implementations
# =============================================================================


@dataclass
class Check(ABC):
    """Base class for result validation checks."""

    @abstractmethod
    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        """
        Validate results. Raises AssertionError on failure.

        Args:
            results: The aiperf results dictionary
            deployment: The managed deployment (for additional context)
            logger: Logger for output
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the check."""
        pass


@dataclass
class ZeroErrorsCheck(Check):
    """Assert that there are zero errors in the results."""

    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        error_result = results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        logger.info(f"ZeroErrorsCheck: error_count = {error_count}")
        assert error_count == 0, f"Expected 0 errors, got {error_count}"

    @property
    def description(self) -> str:
        return "Zero errors"


@dataclass
class MinRequestsCheck(Check):
    """Assert that at least a minimum number of requests succeeded."""

    min_count: int

    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        request_count = results.get("request_count", {}).get("avg", 0)

        logger.info(
            f"MinRequestsCheck: request_count = {request_count}, min = {self.min_count}"
        )
        assert (
            request_count >= self.min_count
        ), f"Expected at least {self.min_count} requests, got {request_count}"

    @property
    def description(self) -> str:
        return f"Min {self.min_count} requests"


@dataclass
class ThroughputCheck(Check):
    """Assert that throughput meets a minimum threshold."""

    min_throughput: float  # requests/sec

    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        throughput = results.get("request_throughput", {}).get("avg", 0)

        logger.info(
            f"ThroughputCheck: throughput = {throughput:.2f}, min = {self.min_throughput}"
        )
        assert (
            throughput >= self.min_throughput
        ), f"Expected throughput >= {self.min_throughput}, got {throughput:.2f}"

    @property
    def description(self) -> str:
        return f"Min throughput {self.min_throughput} req/s"


@dataclass
class LatencyP99Check(Check):
    """Assert that P99 latency is within bounds."""

    max_p99_ms: float  # milliseconds

    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        p99_latency = results.get("request_latency", {}).get("p99", 0)

        logger.info(
            f"LatencyP99Check: p99 = {p99_latency:.2f}ms, max = {self.max_p99_ms}ms"
        )
        assert (
            p99_latency <= self.max_p99_ms
        ), f"Expected P99 latency <= {self.max_p99_ms}ms, got {p99_latency:.2f}ms"

    @property
    def description(self) -> str:
        return f"P99 latency <= {self.max_p99_ms}ms"


@dataclass
class MaxErrorsCheck(Check):
    """Assert that errors are below a threshold (for fault tolerance tests)."""

    max_errors: int

    def validate(
        self,
        results: dict[str, Any],
        deployment: ManagedDeployment,
        logger: logging.Logger,
    ) -> None:
        error_result = results.get("error_request_count")
        error_count = error_result.get("avg", 0) if error_result else 0

        logger.info(
            f"MaxErrorsCheck: error_count = {error_count}, max = {self.max_errors}"
        )
        assert (
            error_count <= self.max_errors
        ), f"Expected at most {self.max_errors} errors, got {error_count}"

    @property
    def description(self) -> str:
        return f"Max {self.max_errors} errors"


# =============================================================================
# run_scenario() - Common Test Execution Function
# =============================================================================


async def run_scenario(
    request: Any,
    deployment_spec: DeploymentSpec,
    load_config: LoadConfig,
    events: list[Event],
    checks: list[Check],
    namespace: str,
    image: str | None = None,
    skip_service_restart: bool = False,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Run a fault tolerance test scenario.

    This is the common function that handles:
    1. Setting up deployment with log collection
    2. Starting load (non-blocking)
    3. Executing events in order
    4. Terminating load and collecting results
    5. Running all validation checks

    Args:
        request: pytest request fixture
        deployment_spec: The deployment configuration
        load_config: The load generator configuration
        events: List of events to execute during the test
        checks: List of checks to run after the test
        namespace: Kubernetes namespace
        image: Optional container image override
        skip_service_restart: Whether to skip service restart
        save_results: Whether to save results to a JSON file

    Returns:
        The aiperf results dictionary
    """
    logger = logging.getLogger(request.node.name)

    # Log scenario overview
    logger.info("=" * 60)
    logger.info("SCENARIO OVERVIEW")
    logger.info("=" * 60)
    logger.info(f"Events ({len(events)}):")
    for i, event in enumerate(events, 1):
        logger.info(f"  {i}. {event.description}")
    logger.info(f"Checks ({len(checks)}):")
    for i, check in enumerate(checks, 1):
        logger.info(f"  {i}. {check.description}")
    logger.info("=" * 60)

    # Apply image override if provided
    if image:
        deployment_spec.set_image(image)

    # Enable logging
    deployment_spec.set_logging(True, "info")

    # Enable PVC-based log collection
    deployment_spec.enable_log_collection(
        pvc_size="500Mi", container_log_dir="/tmp/service_logs"
    )

    # Compute endpoint URL
    endpoint_url = (
        f"http://{deployment_spec.name.lower()}-"
        f"{deployment_spec.frontend_service.name.lower()}."
        f"{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"
    )

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,
    ) as deployment:
        logger.info(f"Deployment ready, endpoint: {endpoint_url}")

        async with ManagedAIPerfDeployment(
            namespace=namespace,
            load_config=load_config,
            log_dir=request.node.name,
        ) as load:
            # Start load (non-blocking)
            logger.info("Starting load...")
            await load.run(wait_for_completion=False)
            await load._wait_for_started()
            logger.info("Load started")

            # Execute events in order
            logger.info("=" * 60)
            logger.info("EXECUTING EVENTS")
            logger.info("=" * 60)
            for i, event in enumerate(events, 1):
                logger.info(f"Event {i}/{len(events)}: {event.description}")
                await event.execute(deployment, load, logger)
                logger.info(f"Event {i} completed")

            # Terminate load and get results
            logger.info("=" * 60)
            logger.info("COLLECTING RESULTS")
            logger.info("=" * 60)
            await load.terminate()
            results = await load.get_results()

            # Save results if requested
            if save_results:
                results_file = os.path.join(request.node.name, "scenario_results.json")
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {results_file}")

            # Log summary metrics
            request_count = results.get("request_count", {}).get("avg", 0)
            error_result = results.get("error_request_count")
            error_count = error_result.get("avg", 0) if error_result else 0
            throughput = results.get("request_throughput", {}).get("avg", 0)

            logger.info("=== Results Summary ===")
            logger.info(f"Successful requests: {request_count}")
            logger.info(f"Error requests: {error_count}")
            logger.info(f"Throughput: {throughput:.2f} req/s")

            # Run all checks
            logger.info("=" * 60)
            logger.info("RUNNING CHECKS")
            logger.info("=" * 60)
            for i, check in enumerate(checks, 1):
                logger.info(f"Check {i}/{len(checks)}: {check.description}")
                check.validate(results, deployment, logger)
                logger.info(f"Check {i} PASSED")

            logger.info("=" * 60)
            logger.info("ALL CHECKS PASSED")
            logger.info("=" * 60)

            return results


# =============================================================================
# Example Test Functions
# =============================================================================


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_rolling_restart_decode_then_prefill(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    """
    DEP-709: Test rolling restart of disagg deployment with 0 errors.

    Scenario:
    1. Deploy disagg with 2 prefill + 2 decode replicas
    2. Start continuous load
    3. Rolling upgrade decode workers
    4. Rolling upgrade prefill workers
    5. Assert 0 errors
    """
    # 1. Setup deployment
    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )
    deployment_spec.set_service_replicas("TRTLLMPrefillWorker", 2)
    deployment_spec.set_service_replicas("TRTLLMDecodeWorker", 2)

    # 2. Setup load
    load_config = LoadConfig(
        endpoint_url="",  # Will be computed by run_scenario
        duration_minutes=10,
        concurrency=8,
    )

    # 3. Define events
    events = [
        WaitEvent(duration=30),  # Let load stabilize
        RollingUpgradeEvent(services=["TRTLLMDecodeWorker"]),
        WaitEvent(duration=30),  # Wait between restarts
        RollingUpgradeEvent(services=["TRTLLMPrefillWorker"]),
        WaitEvent(duration=30),  # Let load settle after restarts
    ]

    # 4. Define checks
    checks = [
        ZeroErrorsCheck(),
        MinRequestsCheck(min_count=100),
    ]

    # 5. Run scenario
    await run_scenario(
        request=request,
        deployment_spec=deployment_spec,
        load_config=load_config,
        events=events,
        checks=checks,
        namespace=namespace,
        image=image,
        skip_service_restart=skip_service_restart,
    )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_delete_frontend_pod_recovery(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    """
    Test that deleting the frontend pod recovers gracefully.

    Scenario:
    1. Deploy disagg
    2. Start continuous load
    3. Delete frontend pod
    4. Wait for recovery
    5. Assert minimal errors (some in-flight requests may fail)
    """
    # 1. Setup deployment
    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    # 2. Setup load
    load_config = LoadConfig(
        endpoint_url="",
        duration_minutes=5,
        concurrency=4,
    )

    # 3. Define events
    events = [
        WaitEvent(duration=30),  # Let load stabilize
        DeletePodEvent(services=["Frontend"]),
        WaitForRecoveryEvent(timeout=300),
        WaitEvent(duration=30),  # Let load settle after recovery
    ]

    # 4. Define checks (allow some errors during pod deletion)
    checks = [
        MaxErrorsCheck(max_errors=50),  # Allow some in-flight request failures
        MinRequestsCheck(min_count=50),
    ]

    # 5. Run scenario
    await run_scenario(
        request=request,
        deployment_spec=deployment_spec,
        load_config=load_config,
        events=events,
        checks=checks,
        namespace=namespace,
        image=image,
        skip_service_restart=skip_service_restart,
    )


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_basic_smoke_1000_requests(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    """
    Basic smoke test: 1000 requests with no failures injected.

    Scenario:
    1. Deploy disagg
    2. Run 1000 requests
    3. Assert 0 errors
    """
    # 1. Setup deployment
    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    # 2. Setup load for 1000 requests
    load_config = LoadConfig(
        endpoint_url="",
        request_count=1000,
        concurrency=10,
        input_tokens_mean=128,
        output_tokens_mean=32,
        warmup_requests=10,
    )

    # 3. No events for smoke test
    events: list[Event] = []

    # 4. Define checks
    checks = [
        ZeroErrorsCheck(),
        MinRequestsCheck(min_count=1000),
    ]

    # 5. Run scenario
    await run_scenario(
        request=request,
        deployment_spec=deployment_spec,
        load_config=load_config,
        events=events,
        checks=checks,
        namespace=namespace,
        image=image,
        skip_service_restart=skip_service_restart,
    )
