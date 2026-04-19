# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GlobalPlanner - Centralized Scaling Execution Service

Entry point for the GlobalPlanner component.

Usage:
    DYN_NAMESPACE=global-infra python -m dynamo.global_planner

With authorization:
    DYN_NAMESPACE=global-infra python -m dynamo.global_planner \\
        --managed-namespaces app-ns-1 app-ns-2
"""

import asyncio
import logging
import os

from pydantic import BaseModel

from dynamo.global_planner.argparse_config import create_global_planner_parser
from dynamo.global_planner.scale_handler import ScaleRequestHandler
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class HealthCheckRequest(BaseModel):
    """Request type for health check endpoint"""

    text: str = "ping"


class HealthCheckResponse(BaseModel):
    """Response type for health check endpoint"""

    status: str
    component: str
    namespace: str
    managed_namespaces: object
    dgd_watch: dict
    message: str = ""


@dynamo_worker()
async def main(runtime: DistributedRuntime, args):
    """Initialize and run GlobalPlanner.

    The GlobalPlanner is a centralized scaling service that:
    1. Listens for scale requests from Planners
    2. Validates caller authorization (optional)
    3. Executes scaling via Kubernetes API
    4. Returns success/failure status

    Args:
        runtime: Dynamo runtime instance
        args: Parsed command-line arguments
    """
    # Get Dynamo namespace from environment variable
    namespace = os.environ.get("DYN_NAMESPACE")
    if not namespace:
        raise ValueError(
            "DYN_NAMESPACE environment variable is required but not set. "
            "Please set DYN_NAMESPACE to specify the Dynamo namespace for GlobalPlanner."
        )

    logger.info("=" * 60)
    logger.info("Starting GlobalPlanner")
    logger.info("=" * 60)
    logger.info(f"Namespace: {namespace}")
    logger.info(f"Environment: {args.environment}")

    if args.managed_namespaces:
        logger.info("Authorization: ENABLED")
        logger.info(f"Authorized namespaces: {args.managed_namespaces}")
    else:
        logger.info("Authorization: DISABLED (accepting all namespaces)")

    if args.no_operation:
        logger.info(
            "No-operation mode: ENABLED (scale requests will be logged, not executed)"
        )
    else:
        logger.info("No-operation mode: DISABLED")

    if args.max_total_gpus >= 0:
        logger.info(f"Max total GPUs: {args.max_total_gpus}")
    else:
        logger.info("Max total GPUs: UNLIMITED")

    logger.info("=" * 60)

    # Get K8s namespace (where GlobalPlanner pod is running)
    k8s_namespace = os.environ.get("POD_NAMESPACE", "default")
    logger.info(f"Running in Kubernetes namespace: {k8s_namespace}")

    # Create scale request handler
    handler = ScaleRequestHandler(
        runtime=runtime,
        managed_namespaces=args.managed_namespaces,
        k8s_namespace=k8s_namespace,
        no_operation=args.no_operation,
        max_total_gpus=args.max_total_gpus,
    )

    # Serve scale_request endpoint
    logger.info("Serving endpoints...")
    scale_endpoint = runtime.endpoint(f"{namespace}.GlobalPlanner.scale_request")
    asyncio.ensure_future(scale_endpoint.serve_endpoint(handler.scale_request))
    logger.info("  ✓ scale_request - Receives scaling requests from Planners")

    # Serve health check endpoint (includes DGD watch health)
    @dynamo_endpoint(HealthCheckRequest, HealthCheckResponse)
    async def health_check(request: HealthCheckRequest):
        """Health check endpoint for monitoring"""
        payload = {
            "status": "healthy",
            "component": "GlobalPlanner",
            "namespace": namespace,
            "managed_namespaces": args.managed_namespaces or "all",
        }
        dgd_watch_status = handler.get_dgd_watch_health_status()
        payload["dgd_watch"] = dgd_watch_status
        if dgd_watch_status.get("dgd_watch_enabled") and not dgd_watch_status.get(
            "dgd_watch_alive", True
        ):
            payload["status"] = "degraded"
            payload[
                "message"
            ] = "DGD watch thread not running; GPU budget cache may be stale"
        yield payload

    health_endpoint = runtime.endpoint(f"{namespace}.GlobalPlanner.health")
    asyncio.ensure_future(health_endpoint.serve_endpoint(health_check))
    logger.info("  ✓ health - Health check endpoint")

    logger.info("=" * 60)
    logger.info("GlobalPlanner is ready and waiting for scale requests")
    logger.info("=" * 60)

    if args.max_total_gpus >= 0:
        DGD_WATCH_LIVENESS_INTERVAL_SEC = 60

        async def watch_liveness_loop():
            while True:
                await asyncio.sleep(DGD_WATCH_LIVENESS_INTERVAL_SEC)
                if not handler.is_dgd_watch_healthy():
                    logger.critical(
                        "DGD watch thread is not running; DGD cache may be stale. "
                        "Scaling decisions may use outdated data."
                    )

        asyncio.create_task(watch_liveness_loop())
        logger.info(
            "DGD watch liveness check started (interval %ds)",
            DGD_WATCH_LIVENESS_INTERVAL_SEC,
        )

    # Keep running forever (process scale requests as they come)
    await asyncio.Event().wait()


if __name__ == "__main__":
    parser = create_global_planner_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
