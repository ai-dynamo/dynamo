# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: Measure scale-up latency for ModelExpress P2P weight transfer
# vs. standard disk loading.
#
# Prerequisites:
#   - DGD "vllm-agg-mx-p2p" deployed with mx-source ready (replicas=1)
#   - mx-target at replicas=0 (starting state)
#   - DYN_PARENT_DGD_K8S_NAME env var set to the DGD name
#
# Usage:
#   export DYN_PARENT_DGD_K8S_NAME=vllm-agg-mx-p2p
#   python scaleup_bench.py --target-replicas 1 --k8s-namespace my-namespace
#
# Or use kubernetes_connector.py CLI directly:
#   python -m dynamo.planner.kubernetes_connector \
#       --action set --component-name mx-target --replicas 2 --blocking

import argparse
import asyncio
import logging
import time

from dynamo.planner.kubernetes_connector import KubernetesConnector, TargetReplica
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def measure_scaleup(
    connector: KubernetesConnector,
    target_replicas: int,
    service_name: str = "mx-target",
) -> float:
    """Scale a service to the desired replica count and measure time-to-ready.

    Returns:
        Wall-clock seconds from scale request to DGD ready.
    """
    targets = [
        TargetReplica(
            component_name=service_name,
            desired_replicas=target_replicas,
        ),
    ]

    logger.info(f"Scaling {service_name} to {target_replicas} replicas...")
    start = time.monotonic()
    await connector.set_component_replicas(targets, blocking=True)
    elapsed = time.monotonic() - start

    logger.info(f"Scale-up complete in {elapsed:.1f}s")
    return elapsed


async def run_benchmark(args: argparse.Namespace) -> None:
    connector = KubernetesConnector(
        dynamo_namespace=args.dynamo_namespace,
        k8s_namespace=args.k8s_namespace,
    )

    # Verify mx-source is up before scaling targets
    logger.info("Waiting for DGD to be ready (mx-source should be running)...")
    await connector.wait_for_deployment_ready()
    logger.info("DGD is ready. mx-source is serving.")

    # Scale up mx-target
    elapsed = await measure_scaleup(
        connector,
        target_replicas=args.target_replicas,
        service_name=args.service_name,
    )

    print(f"\n{'=' * 50}")
    print(f"Service:          {args.service_name}")
    print(f"Target replicas:  {args.target_replicas}")
    print(f"Time to ready:    {elapsed:.1f}s")
    print(f"{'=' * 50}")

    # Scale back down for next run
    if args.reset:
        logger.info(f"Resetting {args.service_name} to 0 replicas...")
        await measure_scaleup(
            connector, target_replicas=0, service_name=args.service_name
        )
        logger.info("Reset complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure scale-up latency for MX P2P weight transfer"
    )
    parser.add_argument(
        "--dynamo-namespace",
        type=str,
        default="dynamo",
        help="Dynamo logical namespace",
    )
    parser.add_argument(
        "--k8s-namespace",
        type=str,
        default=None,
        help="Kubernetes namespace (auto-detected if in-cluster)",
    )
    parser.add_argument(
        "--target-replicas",
        type=int,
        default=1,
        help="Number of mx-target replicas to scale to",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default="mx-target",
        help="DGD service name to scale",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Scale back to 0 after measurement",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
