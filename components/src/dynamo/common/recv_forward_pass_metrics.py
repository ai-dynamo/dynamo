# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Receive ForwardPassMetrics via the Dynamo event plane.

Auto-discovers engine publishers through the discovery plane (K8s CRD /
etcd / file) and prints each metric message as JSON.

Usage:
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --component backend --endpoint generate \\
        [--discovery-backend etcd] [--request-plane nats]
"""

import argparse
import asyncio
import json
import logging
import os

import msgspec

from dynamo.common.forward_pass_metrics import decode
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive ForwardPassMetrics from the Dynamo event plane"
    )
    parser.add_argument(
        "--namespace", default="dynamo", help="Dynamo namespace (default: dynamo)"
    )
    parser.add_argument(
        "--component", default="backend", help="Dynamo component (default: backend)"
    )
    parser.add_argument(
        "--endpoint", default="generate", help="Dynamo endpoint (default: generate)"
    )
    parser.add_argument(
        "--discovery-backend",
        default=os.environ.get("DYN_DISCOVERY_BACKEND", "etcd"),
        help="Discovery backend (default: etcd)",
    )
    parser.add_argument(
        "--request-plane",
        default=os.environ.get("DYN_REQUEST_PLANE", "nats"),
        help="Request plane (default: nats)",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    event_plane = os.environ.get("DYN_EVENT_PLANE", "nats")
    enable_nats = args.request_plane == "nats" or event_plane == "nats"
    runtime = DistributedRuntime(
        loop, args.discovery_backend, args.request_plane, enable_nats
    )
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.{args.endpoint}")

    subscriber = FpmEventSubscriber(endpoint)
    json_encoder = msgspec.json.Encoder()

    logger.info(
        "Subscribed to forward-pass-metrics via event plane "
        "(namespace=%s, component=%s)  Ctrl+C to stop",
        args.namespace,
        args.component,
    )

    try:
        while True:
            data = await asyncio.to_thread(subscriber.recv)
            if data is None:
                logger.info("Stream closed.")
                break
            metrics = decode(data)
            if metrics is None:
                continue
            pretty = json.loads(json_encoder.encode(metrics))
            logger.info(
                "[worker=%s dp=%d counter=%d] %s",
                metrics.worker_id,
                metrics.dp_rank,
                metrics.counter_id,
                json.dumps(pretty, indent=2),
            )
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        subscriber.shutdown()


if __name__ == "__main__":
    main()
