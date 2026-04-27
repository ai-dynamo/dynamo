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
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class HealthCheckRequest(BaseModel):
    """Request type for health check endpoint"""

    text: str = "ping"


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

    logger.info("Serving endpoints...")
    scale_endpoint = runtime.endpoint(f"{namespace}.GlobalPlanner.scale_request")
    health_endpoint = runtime.endpoint(f"{namespace}.GlobalPlanner.health")

    async def health_check(request: HealthCheckRequest):
        """Health check endpoint for monitoring"""
        yield {
            "status": "healthy",
            "component": "GlobalPlanner",
            "namespace": namespace,
            "managed_namespaces": args.managed_namespaces or "all",
        }

    logger.info("  ✓ scale_request - Receives scaling requests from Planners")
    logger.info("  ✓ health - Health check endpoint")
    logger.info("=" * 60)
    logger.info("GlobalPlanner is ready and waiting for scale requests")
    logger.info("=" * 60)

    # serve_endpoint is a long-running task — it only returns on shutdown.
    # Awaiting them sequentially would block on scale_request forever and
    # never register the health endpoint, so system_health would never flip
    # to Ready and the operator-injected HTTP probes on :system/live and
    # :system/health would 503 indefinitely. Run concurrently via
    # asyncio.gather; pattern matches components/src/dynamo/planner/__main__.py.
    #
    # Passing health_check_payload to the health endpoint registers it as a
    # health-check target so system_health flips to Ready once the endpoint
    # is live. The payload shape matches HealthCheckRequest so if canary
    # probing is ever enabled it can deserialize cleanly.
    await asyncio.gather(
        scale_endpoint.serve_endpoint(handler.scale_request),
        health_endpoint.serve_endpoint(
            health_check,
            health_check_payload={"text": "health"},
        ),
    )


if __name__ == "__main__":
    parser = create_global_planner_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
