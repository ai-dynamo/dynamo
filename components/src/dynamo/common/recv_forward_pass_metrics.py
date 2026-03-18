# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Receive ForwardPassMetrics via the Dynamo event plane.

Auto-discovers engine publishers through the discovery plane (K8s CRD /
etcd / file) and prints each metric message as JSON.

Supports two modes:

- **recv** (default): pull individual messages one at a time.
- **tracking**: periodically poll ``get_recent_stats()`` to print the
  latest snapshot keyed by ``(worker_id, dp_rank)``.

Usage:
    # recv mode (default)
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --component backend --endpoint generate

    # tracking mode (poll every 2 seconds)
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --component backend --endpoint generate \\
        --mode tracking --poll-interval 2.0
"""

import argparse
import asyncio
import json
import os
import sys
import time

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
    parser.add_argument(
        "--mode",
        choices=["recv", "tracking"],
        default="recv",
        help="Consumption mode: 'recv' for individual messages, "
        "'tracking' for latest-snapshot polling (default: recv)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for tracking mode (default: 2.0)",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


async def run(args: argparse.Namespace) -> None:
    from dynamo.llm import FpmEventSubscriber

    loop = asyncio.get_running_loop()
    event_plane = os.environ.get("DYN_EVENT_PLANE", "nats")
    enable_nats = args.request_plane == "nats" or event_plane == "nats"
    runtime = DistributedRuntime(
        loop, args.discovery_backend, args.request_plane, enable_nats
    )
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.{args.endpoint}")

    subscriber = FpmEventSubscriber(endpoint)

    print(
        f"Subscribed to forward-pass-metrics via event plane "
        f"(namespace={args.namespace}, component={args.component}, "
        f"mode={args.mode})  Ctrl+C to stop",
        file=sys.stderr,
    )

    try:
        if args.mode == "tracking":
            await _run_tracking(subscriber, args)
        else:
            await _run_recv(subscriber)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        subscriber.shutdown()


async def _run_recv(subscriber) -> None:
    """Pull individual FPM messages and print each as JSON."""
    json_encoder = msgspec.json.Encoder()
    seq = 0
    while True:
        data = await asyncio.to_thread(subscriber.recv)
        if data is None:
            print("Stream closed.", file=sys.stderr)
            break
        metrics = decode(data)
        pretty = json.loads(json_encoder.encode(metrics))
        print(f"[seq={seq}] {json.dumps(pretty, indent=2)}", flush=True)
        seq += 1


async def _run_tracking(subscriber, args: argparse.Namespace) -> None:
    """Poll get_recent_stats() and print the latest snapshot periodically."""
    json_encoder = msgspec.json.Encoder()
    subscriber.start_tracking()
    print(
        f"Tracking mode started (poll every {args.poll_interval}s)",
        file=sys.stderr,
    )

    poll = 0
    while True:
        await asyncio.sleep(args.poll_interval)
        stats = subscriber.get_recent_stats()

        if not stats:
            print(f"[poll={poll}] (no engines tracked)", flush=True)
        else:
            snapshot = {}
            for (worker_id, dp_rank), raw_bytes in stats.items():
                metrics = decode(raw_bytes)
                key = f"{worker_id}:dp{dp_rank}"
                snapshot[key] = json.loads(json_encoder.encode(metrics))

            ts = time.strftime("%H:%M:%S")
            print(
                f"[poll={poll} t={ts} engines={len(stats)}] "
                f"{json.dumps(snapshot, indent=2)}",
                flush=True,
            )
        poll += 1


if __name__ == "__main__":
    main()
