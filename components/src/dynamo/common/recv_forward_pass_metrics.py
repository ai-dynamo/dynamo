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

    # recv mode with plot saving
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --component backend --endpoint generate \\
        --save-plot metrics.png

    # Buffered capture without per-message logging (existing files are refused)
    python -m dynamo.common.recv_forward_pass_metrics \\
        --namespace dynamo --output fpm.jsonl --flush-interval 1

Output is compact JSONL: {"received_at_ns": <Unix nanoseconds>, "metrics": <FPM>}.
The timestamp is taken by the receiver, not the engine. --log-metrics also prints
each message when recording. Flush writes Python buffers, not fsync; SIGKILL or
machine failure can lose buffered data. Counter gaps are diagnostics, not proof
of lossless delivery (publisher-side drops before sequence assignment are invisible).
"""

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import time
from contextlib import aclosing, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import msgspec

from dynamo.common.forward_pass_metrics import ForwardPassMetrics, decode
from dynamo.llm import FpmEventSubscriber
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _save_plot(path: str, history: list[tuple[float, ForwardPassMetrics]]) -> None:
    """Render 5-panel time-series plot and save to *path*."""
    if not history:
        logger.warning("No data collected, skipping plot.")
        return

    # Plotting is optional: runtime images used for capture need not contain
    # matplotlib or its dependencies. Load them only for --save-plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        default=None,
        help="Save a time-series plot to the given PNG path on exit (recv mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write compact JSONL to a new file (recv mode only); disables "
        "per-message logging unless --log-metrics is set",
    )
    parser.add_argument(
        "--flush-interval",
        type=_positive_seconds,
        default=1.0,
        help="Flush the output buffer every N seconds, including while idle (default: 1)",
    )
    parser.add_argument(
        "--log-metrics",
        action="store_true",
        help="Also log individual messages when --output is set",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.mode != "recv":
        parser.error("--output requires --mode recv; tracking only samples snapshots")
    return args


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run_with_signals(args))
    except KeyboardInterrupt:
        logger.info("Stopped.")


async def _run_with_signals(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None  # This coroutine is always executed inside a Task.
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await run(args)
    except asyncio.CancelledError:
        logger.info("Stopped.")
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


async def run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, args.discovery_backend, args.request_plane)
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.{args.endpoint}")

    subscriber = FpmEventSubscriber(endpoint)

    logger.info(
        "Subscribed to forward-pass-metrics via event plane "
        "(namespace=%s, component=%s, mode=%s)  Ctrl+C to stop",
        args.namespace,
        args.component,
        args.mode,
    )

    try:
        if args.mode == "tracking":
            await _run_tracking(subscriber, args)
        else:
            await _run_recv(subscriber, args)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        subscriber.shutdown()


async def _run_recv(subscriber, args: argparse.Namespace) -> None:
    """Pull individual FPM messages and print each as JSON."""
    json_encoder = msgspec.json.Encoder()
    history: list[tuple[float, ForwardPassMetrics]] = []
    start_time: float | None = None
    stats = _RecordingStats()
    output_context = (
        args.output.open("xb", buffering=1024 * 1024)
        if args.output is not None
        else nullcontext(None)
    )

    try:
        with output_context as output:
            async with aclosing(
                _receive(subscriber, output, args.flush_interval)
            ) as stream:
                async for data, received_at_ns in stream:
                    metrics = decode(data)
                    if metrics is None:
                        stats.rejected += 1
                        continue
                    stats.observe(metrics)
                    if output is not None:
                        output.write(
                            json_encoder.encode(
                                {
                                    "received_at_ns": received_at_ns,
                                    "metrics": metrics,
                                }
                            )
                            + b"\n"
                        )

                    if args.save_plot:
                        now = time.monotonic()
                        if start_time is None:
                            start_time = now
                        history.append((now - start_time, metrics))

                    if output is None or args.log_metrics:
                        pretty = json.loads(json_encoder.encode(metrics))
                        logger.info(
                            "[worker=%s dp=%d counter=%d] %s",
                            metrics.worker_id,
                            metrics.dp_rank,
                            metrics.counter_id,
                            json.dumps(pretty, indent=2),
                        )
    finally:
        logger.info(
            "FPM capture: messages=%d rejected=%d streams=%d counter_gaps=%d "
            "non_increasing_counters=%d (not a lossless-delivery guarantee)",
            stats.messages,
            stats.rejected,
            len(stats.last_counter),
            stats.counter_gaps,
            stats.non_increasing_counters,
        )
        if args.save_plot and history:
            _save_plot(args.save_plot, history)


@dataclass
class _RecordingStats:
    messages: int = 0
    rejected: int = 0
    counter_gaps: int = 0
    non_increasing_counters: int = 0
    last_counter: dict[tuple[str, int], int] = field(default_factory=dict)

    def observe(self, metrics: ForwardPassMetrics) -> None:
        self.messages += 1
        key = (metrics.worker_id, metrics.dp_rank)
        previous = self.last_counter.get(key)
        if previous is not None:
            if metrics.counter_id > previous + 1:
                self.counter_gaps += metrics.counter_id - previous - 1
            elif metrics.counter_id <= previous:
                # Could be a duplicate, reordering or producer restart. Keep
                # the high-water mark: a restart cannot be distinguished here,
                # and rebasing would turn delayed messages into artificial gaps.
                self.non_increasing_counters += 1
                return
        self.last_counter[key] = metrics.counter_id


async def _receive(subscriber, output: BinaryIO | None, flush_interval: float):
    """Keep one blocking receive in flight; flush even if no publisher is active."""
    pending = None
    next_flush = time.monotonic() + flush_interval
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(asyncio.to_thread(subscriber.recv))
            timeout = max(0.0, next_flush - time.monotonic()) if output else None
            done, _ = await asyncio.wait({pending}, timeout=timeout)
            if output is not None and time.monotonic() >= next_flush:
                output.flush()
                next_flush = time.monotonic() + flush_interval
            if done:
                data = pending.result()
                pending = None
                if data is None:
                    break
                yield data, time.time_ns()
    finally:
        # Cancelling to_thread alone does not stop its blocking Rust receive.
        # Unblock it before asyncio.run() joins the executor on process shutdown.
        subscriber.shutdown()
        if pending is not None:
            await pending


async def _run_tracking(subscriber, args: argparse.Namespace) -> None:
    """Poll get_recent_stats() and print the latest snapshot periodically."""
    json_encoder = msgspec.json.Encoder()
    subscriber.start_tracking()
    logger.info("Tracking mode started (poll every %.1fs)", args.poll_interval)

    poll = 0
    while True:
        await asyncio.sleep(args.poll_interval)
        stats = subscriber.get_recent_stats()

        if not stats:
            logger.info("[poll=%d] (no engines tracked)", poll)
        else:
            snapshot = {}
            for (worker_id, dp_rank), raw_bytes in stats.items():
                metrics = decode(raw_bytes)
                if metrics is None:
                    continue
                key = f"{worker_id}:dp{dp_rank}"
                snapshot[key] = json.loads(json_encoder.encode(metrics))

            ts = time.strftime("%H:%M:%S")
            logger.info(
                "[poll=%d t=%s engines=%d] %s",
                poll,
                ts,
                len(stats),
                json.dumps(snapshot, indent=2),
            )
        poll += 1


if __name__ == "__main__":
    main()
