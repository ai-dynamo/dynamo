# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging

from dynamo._core import DistributedRuntime, VirtualConnectorClient
from dynamo.planner import SubComponentType, TargetReplica, VirtualConnector

logger = logging.getLogger(__name__)


def get_runtime() -> DistributedRuntime:
    try:
        return DistributedRuntime.detached()
    except Exception:
        loop = asyncio.get_running_loop()
        return DistributedRuntime(loop, "etcd", "nats")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Manual virtual planner controller")
    parser.add_argument("--namespace", default="dynamo", help="Dynamo namespace")
    parser.add_argument(
        "--model-name",
        default="sglang",
        help="Model/backend name passed to VirtualConnector",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=1,
        help="Desired prefill replicas for the first scaling decision",
    )
    parser.add_argument(
        "--decode",
        type=int,
        default=1,
        help="Desired decode replicas for the first scaling decision",
    )
    args = parser.parse_args()

    runtime = get_runtime()
    connector = VirtualConnector(runtime, args.namespace, args.model_name)
    await connector._async_init()
    client = VirtualConnectorClient(runtime, args.namespace)

    replicas = [
        TargetReplica(
            sub_component_type=SubComponentType.PREFILL,
            desired_replicas=args.prefill,
        ),
        TargetReplica(
            sub_component_type=SubComponentType.DECODE,
            desired_replicas=args.decode,
        ),
    ]

    await connector.set_component_replicas(replicas, blocking=False)
    event = await client.get()
    logger.info(
        "Received scaling event decision_id=%s prefill=%s decode=%s",
        event.decision_id,
        event.num_prefill_workers,
        event.num_decode_workers,
    )
    await client.complete(event)
    await connector._wait_for_scaling_completion()


if __name__ == "__main__":
    asyncio.run(main())
