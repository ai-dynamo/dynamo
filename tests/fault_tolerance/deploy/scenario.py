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
from dataclasses import dataclass
from typing import Any

from tests.fault_tolerance.deploy.checks import Check
from tests.fault_tolerance.deploy.events import Event, StartLoad
from tests.fault_tolerance.deploy.reports import Report
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

# =============================================================================
# ScenarioContext
# =============================================================================


@dataclass
class ScenarioContext:
    """Runtime context - holds deployment, events, checks, and reports.

    No helper methods - callers iterate/filter events themselves.
    """

    deployment: ManagedDeployment
    events: list[Event]
    checks: list[Check]
    reports: list[Report]
    logger: logging.Logger
    namespace: str
    log_dir: str


# =============================================================================
# run_scenario() - Main Entry Point
# =============================================================================


async def run_scenario(
    request: Any,
    deployment_spec: DeploymentSpec,
    events: list[Event],
    checks: list[Check],
    reports: list[Report] | None = None,
) -> ScenarioContext:
    """
    Run a test scenario.

    Extracts common fixtures (namespace, image, skip_service_restart) from request.

    Flow:
    1. Setup deployment
    2. Execute all events
    3. Stop all events (collects results from unfinished loads)
    4. Run checks
    5. Generate reports
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

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=log_dir,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
        enable_volume_log_collection=True,
    ) as deployment:
        ctx = ScenarioContext(
            deployment=deployment,
            events=events,
            checks=checks,
            reports=reports,
            logger=logger,
            namespace=namespace,
            log_dir=log_dir,
        )

        try:
            # 1. Execute all events
            logger.info("=" * 60)
            logger.info("EXECUTING EVENTS")
            logger.info("=" * 60)
            for i, event in enumerate(events, 1):
                logger.info(f"Event {i}/{len(events)}: {event.description}")
                await event.execute(ctx)
                logger.info(f"Event {i} completed")

            # 2. Stop all events (collects results from unfinished loads)
            logger.info("=" * 60)
            logger.info("STOPPING EVENTS")
            logger.info("=" * 60)
            for event in reversed(events):
                await event.stop(ctx)

        finally:
            # Emergency cleanup
            for event in events:
                try:
                    await event.stop(ctx)
                except Exception as e:
                    logger.warning(f"Cleanup error for {event.description}: {e}")

        # 3. Log results summary
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

        # 4. Run checks
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

        # 5. Generate reports (after checks pass)
        if reports:
            logger.info("=" * 60)
            logger.info("GENERATING REPORTS")
            logger.info("=" * 60)
            for i, report in enumerate(reports, 1):
                logger.info(f"Report {i}/{len(reports)}: {report.description}")
                report.generate(ctx)
                logger.info(f"Report {i} generated")

        return ctx
