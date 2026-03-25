# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Receive ForwardPassMetrics via the Dynamo event plane.

Auto-discovers engine publishers through the discovery plane (K8s CRD /
etcd / file) and prints each metric message as JSON.

Usage:
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --component backend --endpoint generate \\
        [--discovery-backend etcd] [--request-plane nats] \\
        [--save-plot metrics.png]
"""

import argparse
import asyncio
import json
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import msgspec

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

matplotlib.use("Agg")

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _save_plot(path: str, history: list[tuple[float, ForwardPassMetrics]]) -> None:
    """Render 5-panel time-series plot and save to *path*."""
    if not history:
        logger.warning("No data collected, skipping plot.")
        return

    ts = [t for t, _ in history]
    num_prefill = [m.scheduled_requests.num_prefill_requests for _, m in history]
    sum_prefill = [m.scheduled_requests.sum_prefill_tokens for _, m in history]
    num_decode = [m.scheduled_requests.num_decode_requests for _, m in history]
    sum_kv = [m.scheduled_requests.sum_decode_kv_tokens for _, m in history]
    wall = [m.wall_time for _, m in history]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    panels = [
        (axes[0], num_prefill, "num_prefill_requests"),
        (axes[1], sum_prefill, "sum_prefill_tokens"),
        (axes[2], num_decode, "num_decode_requests"),
        (axes[3], sum_kv, "sum_decode_kv_tokens"),
        (axes[4], wall, "wall_time (s)"),
    ]

    for ax, data, label in panels:
        ax.plot(ts, data, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("ForwardPassMetrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s (%d data points)", path, len(history))


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
        "--save-plot",
        metavar="PATH",
        default=None,
        help="Save a time-series plot to the given PNG path on exit",
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
    json_encoder = msgspec.json.Encoder()

    logger.info(
        "Subscribed to forward-pass-metrics via event plane "
        "(namespace=%s, component=%s)  Ctrl+C to stop",
        args.namespace,
        args.component,
    )

    history: list[tuple[float, ForwardPassMetrics]] = []
    start_time: float | None = None

    try:
        while True:
            data = await asyncio.to_thread(subscriber.recv)
            if data is None:
                logger.info("Stream closed.")
                break
            metrics = decode(data)
            if metrics is None:
                continue

            now = time.monotonic()
            if start_time is None:
                start_time = now

            if args.save_plot:
                history.append((now - start_time, metrics))

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
        if args.save_plot and history:
            _save_plot(args.save_plot, history)


if __name__ == "__main__":
    main()
