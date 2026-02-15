# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario framework for fault tolerance testing.

Provides:
- ScenarioContext: Runtime context holding deployment, events, checks, reports
- run_scenario(): Main entry point for running test scenarios

Usage:
    await run_scenario(
        request=request,
        deployment_spec=DeploymentSpec("..."),
        events=[StartLoad(...), Wait(...), StopLoad()],
        checks=[ZeroErrors(), MinRequests(min_count=50)],
    )
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from tests.fault_tolerance.deploy.checks import Check
from tests.fault_tolerance.deploy.events import Event, StartLoad
from tests.fault_tolerance.deploy.reports import Report
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

if TYPE_CHECKING:
    from tests.utils.resource_monitor import ResourceMonitorConfig, ResourceSnapshot

# =============================================================================
# ScenarioContext
# =============================================================================


@dataclass
class ScenarioContext:
    """Runtime context - holds deployment, events, checks, and reports.

    No helper methods - callers iterate/filter events themselves.

    Note: deployment may be None after context manager exits - only use events/reports
    data for checks and report generation.
    """

    deployment: Optional[ManagedDeployment]
    events: list[Event]
    checks: list[Check]
    reports: list[Report]
    logger: logging.Logger
    namespace: str
    log_dir: str
    resource_history: list["ResourceSnapshot"] = field(default_factory=list)


# =============================================================================
# run_scenario() - Main Entry Point
# =============================================================================


async def run_scenario(
    request: Any,
    deployment_spec: DeploymentSpec,
    events: list[Event],
    checks: list[Check],
    reports: list[Report] | None = None,
    resource_config: Optional["ResourceMonitorConfig"] = None,
) -> ScenarioContext:
    """
    Run a test scenario.

    Extracts common fixtures (namespace, image, skip_service_restart) from request.

    Flow:
    1. Setup deployment
    2. Start resource monitoring (if configured)
    3. Execute all events
    4. Stop all events (collects results from unfinished loads)
    5. Stop resource monitoring (collects metrics)
    6. [Deployment cleanup happens here - context manager exits]
    7. Generate reports (BEFORE checks, so failures don't block reports)
    8. Run checks (assertions happen last)
    """
    # Extract common fixtures from request
    namespace = request.getfixturevalue("namespace")
    image = request.getfixturevalue("image")
    skip_service_restart = request.getfixturevalue("skip_service_restart")
    storage_class = request.getfixturevalue("storage_class")
    log_dir = request.node.name
    logger = logging.getLogger(request.node.name)

    reports = reports or []

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
    if reports:
        logger.info(f"Reports ({len(reports)}):")
        for i, report in enumerate(reports, 1):
            logger.info(f"  {i}. {report.description}")
    if resource_config:
        logger.info("Resource monitoring: ENABLED")
    logger.info("=" * 60)

    # Apply image override
    if image:
        deployment_spec.set_image(image)

    # Enable logging
    deployment_spec.set_logging(True, "info")

    # Enable PVC-based log collection
    log_collection_kwargs = {
        "pvc_size": "500Mi",
        "container_log_dir": "/tmp/service_logs",
    }
    if storage_class:
        log_collection_kwargs["storage_class"] = storage_class
    deployment_spec.enable_log_collection(**log_collection_kwargs)

    # Create context (will be populated during deployment)
    ctx = ScenarioContext(
        deployment=None,
        events=events,
        checks=checks,
        reports=reports,
        logger=logger,
        namespace=namespace,
        log_dir=log_dir,
        resource_history=[],
    )

    # Phase 1: Deployment and event execution
    async with ManagedDeployment(
        namespace=namespace,
        log_dir=log_dir,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,
    ) as deployment:
        ctx.deployment = deployment

        # Start resource monitoring if configured
        monitor = None
        if resource_config:
            from tests.utils.resource_monitor import ResourceMonitor

            monitor = ResourceMonitor(deployment, resource_config)
            await monitor.start()

        try:
            # Execute all events
            logger.info("=" * 60)
            logger.info("EXECUTING EVENTS")
            logger.info("=" * 60)
            for i, event in enumerate(events, 1):
                logger.info(f"Event {i}/{len(events)}: {event.description}")
                await event.execute(ctx)
                logger.info(f"Event {i} completed")

            # Stop all events (collects results from unfinished loads)
            logger.info("=" * 60)
            logger.info("STOPPING EVENTS")
            logger.info("=" * 60)
            for event in reversed(events):
                await event.stop(ctx)

        finally:
            # Emergency cleanup for events
            for event in events:
                try:
                    await event.stop(ctx)
                except Exception as e:
                    logger.warning(f"Cleanup error for {event.description}: {e}")

            # Stop resource monitoring before deployment cleanup
            if monitor:
                logger.info("Stopping resource monitoring...")
                ctx.resource_history = await monitor.stop()
                await monitor.save_history_locally(log_dir)

        # Log results summary (while deployment still exists)
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        for event in events:
            if isinstance(event, StartLoad) and event.results:
                data = event.results
                request_count = data.get("request_count", {}).get("avg", 0)
                error_result = data.get("error_request_count")
                error_count = error_result.get("avg", 0) if error_result else 0
                throughput = data.get("request_throughput", {}).get("avg", 0)
                logger.info(f"Load '{event.name}':")
                logger.info(f"  Requests: {request_count}")
                logger.info(f"  Errors: {error_count}")
                logger.info(f"  Throughput: {throughput:.2f} req/s")

    # Deployment context has exited - cleanup completed
    ctx.deployment = None  # Mark that deployment is no longer active

    # Phase 2: Reports and checks (AFTER deployment cleanup)
    # Reports run first so check failures don't block report generation
    if reports:
        logger.info("=" * 60)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 60)
        for i, report in enumerate(reports, 1):
            logger.info(f"Report {i}/{len(reports)}: {report.description}")
            report.generate(ctx)
            logger.info(f"Report {i} generated")

    # Checks run last (may raise AssertionError)
    logger.info("=" * 60)
    logger.info("RUNNING CHECKS")
    logger.info("=" * 60)
    for i, check in enumerate(checks, 1):
        logger.info(f"Check {i}/{len(checks)}: {check.description}")
        check.validate(ctx)
        logger.info(f"Check {i} PASSED")

    logger.info("=" * 60)
    logger.info("ALL CHECKS PASSED")
    logger.info("=" * 60)

    return ctx
