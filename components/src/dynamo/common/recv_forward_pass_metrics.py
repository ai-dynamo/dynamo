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
import os
import sys

import msgspec

from dynamo.common.forward_pass_metrics import decode
from dynamo.runtime import DistributedRuntime


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
    from dynamo.llm import FpmEventSubscriber

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(
        loop, args.discovery_backend, args.request_plane, False
    )
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.{args.endpoint}")

    subscriber = FpmEventSubscriber(endpoint)
    json_encoder = msgspec.json.Encoder()

    print(
        f"Subscribed to forward-pass-metrics via event plane "
        f"(namespace={args.namespace}, component={args.component})  "
        f"Ctrl+C to stop",
        file=sys.stderr,
    )

    seq = 0
    try:
        while True:
            data = await asyncio.to_thread(subscriber.recv)
            if data is None:
                print("Stream closed.", file=sys.stderr)
                break
            metrics = decode(data)
            pretty = json.loads(json_encoder.encode(metrics))
            print(f"[seq={seq}] {json.dumps(pretty, indent=2)}", flush=True)
            seq += 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        subscriber.shutdown()


if __name__ == "__main__":
    main()
