# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare router backpressure using three CPU Mocker workers."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RECEIVED = "dynamo_component_requests_total"
INFLIGHT = "dynamo_component_inflight_requests"
PENDING = "dynamo_frontend_router_queue_pending_requests"


def metric(
    text: str, name: str, labels: tuple[str, ...] = (), *, required: bool = False
) -> float:
    """Sum only the requested Prometheus series, including optional exact labels."""
    values = [
        float(line.split()[-1])
        for line in text.splitlines()
        if (line.startswith(name + "{") or line.startswith(name + " "))
        and all(label in line.split()[0] for label in labels)
    ]
    if required and not values:
        raise RuntimeError(f"Required metric {name} with labels {labels} is missing")
    return sum(values)


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        candidates = iter(random.SystemRandom().sample(range(15000, 30000), 15000))
        for sock in sockets:
            for port in candidates:
                try:
                    sock.bind(("127.0.0.1", port))
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("No free ports in 15000..29999")
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


@dataclass
class Sample:
    seconds: float
    sent: int
    completed: int
    errors: int
    received_1: float
    received_2: float
    received_3: float
    inflight_1: float
    inflight_2: float
    inflight_3: float
    router_pending: float

    @property
    def received(self) -> tuple[float, float, float]:
        return self.received_1, self.received_2, self.received_3


class Demo:
    def __init__(self, args: argparse.Namespace, mode: str, output: Path):
        self.args, self.mode, self.output = args, mode, output
        self.namespace = "backpressure-" + uuid.uuid4().hex[:10]
        self.ports = free_ports(8)
        self.etcd_port, self.peer_port, self.nats_port, self.http_port = self.ports[:4]
        self.worker_ports = self.ports[4:7]
        self.processes: list[asyncio.subprocess.Process] = []
        self.workers: list[asyncio.subprocess.Process | None] = [None] * 3
        self.worker_ready = [False] * 3
        self.receipt_offset = [0.0] * 3
        self.last_receipts = [0.0] * 3
        self.samples: list[Sample] = []
        self.phases: list[dict] = []
        self.events: list[dict] = []
        self.worker_removed_at: float | None = None
        self.outcomes: list[dict] = []
        self.sent = self.completed = self.errors = 0
        self.pending: set[asyncio.Task] = set()
        self.stop_load = asyncio.Event()
        self.start_time = time.monotonic()
        self.commands: list[dict] = []
        # Exclude inherited router/admission overrides from this controlled comparison.
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("DYN_")}
        self.env.update(
            ETCD_ENDPOINTS=f"http://127.0.0.1:{self.etcd_port}",
            NATS_SERVER=f"nats://127.0.0.1:{self.nats_port}",
            DYN_REQUEST_PLANE="nats",
            DYN_LOG="warn",
            DYN_COMPUTE_THREADS="1",
            RAYON_NUM_THREADS="2",
            OMP_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )

    def seconds(self) -> float:
        return time.monotonic() - self.start_time

    async def spawn(
        self, name: str, command: list[str], **env
    ) -> asyncio.subprocess.Process:
        with (self.output / f"{name}.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=self.env | env,
            )
        self.processes.append(process)
        self.commands.append({"name": name, "pid": process.pid, "command": command})
        return process

    async def stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                process.kill()
                await process.wait()

    async def text(self, port: int, path: str = "/metrics") -> str:
        async with self.session.get(
            f"http://127.0.0.1:{port}{path}", timeout=aiohttp.ClientTimeout(total=2)
        ) as response:
            response.raise_for_status()
            return await response.text()

    async def wait_http(
        self, process: asyncio.subprocess.Process, port: int, path: str = "/metrics"
    ) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(
                    f"Process {process.pid} exited with {process.returncode}; inspect logs in {self.output}"
                )
            try:
                await self.text(port, path)
                return
            except (aiohttp.ClientError, TimeoutError):
                await asyncio.sleep(0.2)
        raise TimeoutError(f"Service on port {port} did not become ready")

    async def wait_workers(self, count: int) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            self.check_processes()
            data = await self.text(self.http_port)
            if (
                metric(data, "dynamo_frontend_router_workers", ('state="available"',))
                == count
            ):
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(f"Frontend did not discover {count} available workers")

    def check_processes(self) -> None:
        live_workers = [worker for worker in self.workers if worker is not None]
        for process in [*self.services, *live_workers]:
            if process.returncode is not None:
                raise RuntimeError(
                    f"Service {process.pid} exited unexpectedly; inspect {self.output}"
                )

    async def start_worker(self, index: int, speed: float, generation: str) -> None:
        process = await self.spawn(
            f"backend-{index + 1}-{generation}",
            [
                sys.executable,
                "-m",
                "dynamo.mocker",
                "--model-path",
                self.args.model,
                "--model-name",
                "backpressure",
                "--endpoint",
                f"dyn://{self.namespace}.mocker.generate",
                "--discovery-backend",
                "etcd",
                "--num-workers",
                "1",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "8192",
                "--no-enable-prefix-caching",
                "--speedup-ratio",
                str(speed),
            ],
            DYN_SYSTEM_PORT=str(self.worker_ports[index]),
        )
        self.workers[index] = process
        await self.wait_http(process, self.worker_ports[index])
        self.worker_ready[index] = True

    async def sample(self) -> Sample:
        self.check_processes()
        frontend = await self.text(self.http_port)
        received, inflight = [], []
        for index, ready in enumerate(self.worker_ready):
            if ready:
                body = await self.text(self.worker_ports[index])
                had_traffic = self.last_receipts[index] > self.receipt_offset[index]
                count = metric(
                    body,
                    RECEIVED,
                    ('dynamo_endpoint="generate"',),
                    required=had_traffic,
                )
                self.last_receipts[index] = self.receipt_offset[index] + count
                active = metric(
                    body,
                    INFLIGHT,
                    ('dynamo_endpoint="generate"',),
                    required=had_traffic,
                )
            else:
                active = 0.0
            received.append(self.last_receipts[index])
            inflight.append(active)
        point = Sample(
            self.seconds(),
            self.sent,
            self.completed,
            self.errors,
            *received,
            *inflight,
            metric(frontend, PENDING, required=True),
        )
        self.samples.append(point)
        return point

    async def send_request(self, number: int) -> None:
        started = self.seconds()
        self.sent += 1
        status, error = 0, ""
        try:
            async with self.session.post(
                f"http://127.0.0.1:{self.http_port}/v1/completions",
                json={
                    "model": "backpressure",
                    "prompt": [1000 + number % 10000] * 512,
                    "max_tokens": 1,
                    "stream": False,
                    "temperature": 0,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                status = response.status
                body = await response.text()
                if status != 200:
                    error = body[:200]
        except (aiohttp.ClientError, TimeoutError) as exc:
            status = 0
            error = str(exc)
        if status == 200:
            self.completed += 1
        else:
            self.errors += 1
        self.outcomes.append(
            {
                "started": started,
                "ended": self.seconds(),
                "status": status,
                "error": error,
            }
        )

    async def load(self) -> None:
        number, started = 0, time.monotonic()
        while not self.stop_load.is_set():
            if len(self.pending) >= 10000:
                raise RuntimeError(
                    "More than 10000 outstanding requests; reduce rate or run length"
                )
            task = asyncio.create_task(self.send_request(number))
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)
            number += 1
            await asyncio.sleep(
                max(0, started + number / self.args.rate - time.monotonic())
            )

    async def observe(self, duration: float) -> None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self.load_task.done():
                await self.load_task
                raise RuntimeError("Load generator stopped unexpectedly")
            await self.sample()
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))

    async def phase(self, name: str) -> None:
        await self.observe(self.args.settle)
        first = await self.sample()
        await self.observe(self.args.window)
        last = await self.sample()
        elapsed = last.seconds - first.seconds
        rates = [
            (end - start) / elapsed
            for start, end in zip(first.received, last.received, strict=True)
        ]
        points = [
            point
            for point in self.samples
            if first.seconds <= point.seconds <= last.seconds
        ]
        result = {
            "name": name,
            "start": first.seconds,
            "end": last.seconds,
            "arrival_rps": rates,
            "sent_rps": (last.sent - first.sent) / elapsed,
            "completed_rps": (last.completed - first.completed) / elapsed,
            "errors": last.errors - first.errors,
            "queue_mean": sum(point.router_pending for point in points) / len(points),
            "queue_delta": last.router_pending - first.router_pending,
            "inflight_end": [last.inflight_1, last.inflight_2, last.inflight_3],
            "inflight_delta": (
                last.inflight_1
                + last.inflight_2
                + last.inflight_3
                - first.inflight_1
                - first.inflight_2
                - first.inflight_3
            ),
            "inflight_max": max(
                max(p.inflight_1, p.inflight_2, p.inflight_3) for p in points
            ),
        }
        self.phases.append(result)
        print(
            f"{self.mode:3s} {name:12s}: backend arrivals {rates}, sent {result['sent_rps']:.1f}/s, router pending {result['queue_mean']:.1f}",
            flush=True,
        )

    async def run(self) -> dict:
        self.output.mkdir(parents=True)
        shutil.copyfile(HERE / f"{self.mode}.yaml", self.output / "policy.yaml")
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0)
        ) as self.session:
            self.services = []
            try:
                etcd = await self.spawn(
                    "etcd",
                    [
                        "etcd",
                        "--data-dir",
                        str(self.output / "etcd-data"),
                        "--listen-client-urls",
                        f"http://127.0.0.1:{self.etcd_port}",
                        "--advertise-client-urls",
                        f"http://127.0.0.1:{self.etcd_port}",
                        "--listen-peer-urls",
                        f"http://127.0.0.1:{self.peer_port}",
                        "--initial-advertise-peer-urls",
                        f"http://127.0.0.1:{self.peer_port}",
                        "--initial-cluster",
                        f"default=http://127.0.0.1:{self.peer_port}",
                    ],
                )
                await self.wait_http(etcd, self.etcd_port, "/health")
                nats = await self.spawn(
                    "nats",
                    [
                        "nats-server",
                        "-a",
                        "127.0.0.1",
                        "-p",
                        str(self.nats_port),
                        "-m",
                        str(self.ports[7]),
                        "-js",
                        "-sd",
                        str(self.output / "nats-data"),
                    ],
                )
                await self.wait_http(nats, self.ports[7], "/healthz")
                frontend = await self.spawn(
                    "frontend",
                    [
                        sys.executable,
                        "-m",
                        "dynamo.frontend",
                        "--http-port",
                        str(self.http_port),
                        "--namespace",
                        self.namespace,
                        "--discovery-backend",
                        "etcd",
                        "--router-mode",
                        "kv",
                        "--router-policy-config",
                        str(self.output / "policy.yaml"),
                    ],
                )
                self.services = [etcd, nats, frontend]
                await self.wait_http(frontend, self.http_port)
                for index in range(3):
                    await self.start_worker(index, self.args.speedup_ratio, "initial")
                await self.wait_workers(3)
                self.start_time = time.monotonic()
                self.load_task = asyncio.create_task(self.load())
                try:
                    await self.phase("healthy")
                    self.events.append(
                        {"seconds": self.seconds(), "action": "stop backend 3"}
                    )
                    self.worker_ready[2] = False
                    worker = self.workers[2]
                    self.workers[2] = None
                    if worker is None:
                        raise RuntimeError("Backend 3 was not running")
                    await self.stop_process(worker)
                    await self.wait_workers(2)
                    self.worker_removed_at = self.seconds()
                    await self.phase("removed")
                    self.events.append(
                        {
                            "seconds": self.seconds(),
                            "action": "restart backend 3 at half speed",
                        }
                    )
                    self.receipt_offset[2] = self.last_receipts[2]
                    await self.start_worker(2, self.args.speedup_ratio / 2, "slow")
                    await self.wait_workers(3)
                    await self.phase("slow")
                finally:
                    self.stop_load.set()
                    try:
                        await self.load_task
                    finally:
                        for task in list(self.pending):
                            task.cancel()
                        await asyncio.gather(*self.pending, return_exceptions=True)
            finally:
                for process in reversed(self.processes):
                    await self.stop_process(process)
                self.write_artifacts()
        return self.summary()

    def classify_outcome(self, outcome: dict) -> str:
        if outcome["status"] == 200:
            return "success"
        stopped = next(
            (
                event["seconds"]
                for event in self.events
                if event["action"] == "stop backend 3"
            ),
            None,
        )
        settled = next(
            (phase["start"] for phase in self.phases if phase["name"] == "removed"),
            None,
        )
        # Continuous traffic can still select the departing worker until discovery
        # confirms removal. Bound the 503 exception by that observation and settling.
        if (
            outcome["status"] == 503
            and stopped is not None
            and settled is not None
            and self.worker_removed_at is not None
            and outcome["started"] < self.worker_removed_at
            and stopped <= outcome["ended"] < settled
        ):
            return "shutdown_transition_failure"
        return "unexpected_failure"

    def summary(self) -> dict:
        checks = {}
        if len(self.phases) == 3:
            healthy, removed, slow = self.phases
            checks["client_rate_held"] = all(
                abs(p["sent_rps"] / self.args.rate - 1) < 0.05 for p in self.phases
            )
            checks["three_workers_received"] = all(
                rate > 0 for rate in healthy["arrival_rps"]
            )
            checks["removed_worker_received_zero"] = removed["arrival_rps"][2] == 0
            checks["responses_complete"] = all(
                p["completed_rps"] > 0 for p in self.phases
            )
            checks["no_unexpected_errors"] = not any(
                self.classify_outcome(outcome) == "unexpected_failure"
                for outcome in self.outcomes
            )
            if self.mode == "on":
                checks["healthy_workers_keep_rate"] = (
                    all(
                        abs(phase["arrival_rps"][i] / healthy["arrival_rps"][i] - 1)
                        <= 0.25
                        for phase in [removed, slow]
                        for i in [0, 1]
                    )
                    if all(healthy["arrival_rps"])
                    else False
                )
                ratio = (
                    slow["arrival_rps"][2] / healthy["arrival_rps"][2]
                    if healthy["arrival_rps"][2]
                    else 0
                )
                checks["slow_worker_half_rate"] = 0.35 <= ratio <= 0.65
                checks["excess_queued_at_router"] = all(
                    p["queue_mean"] > 1 for p in self.phases
                )
                checks["router_queue_grows"] = all(
                    p["queue_delta"] > 0 for p in self.phases
                )
                checks["dispatch_below_offered"] = all(
                    sum(p["arrival_rps"]) < 0.95 * p["sent_rps"] for p in self.phases
                )
                checks["backend_inflight_bounded"] = all(
                    p["inflight_max"] <= 6 for p in self.phases
                )
            else:
                checks["dispatch_tracks_offered"] = all(
                    abs(sum(p["arrival_rps"]) / p["sent_rps"] - 1) < 0.1
                    for p in self.phases
                )
                checks["backend_backlog_grows"] = all(
                    p["inflight_delta"] > 0 for p in self.phases
                )
                checks["no_router_backlog"] = all(
                    p["queue_mean"] < 1 for p in self.phases
                )
        return {
            "mode": self.mode,
            "phases": self.phases,
            "checks": checks,
            "passed": bool(checks) and all(checks.values()),
            "events": self.events,
            "worker_removed_at": self.worker_removed_at,
            "sent": self.sent,
            "completed": self.completed,
            "errors": self.errors,
            "shutdown_transition_errors": sum(
                self.classify_outcome(outcome) == "shutdown_transition_failure"
                for outcome in self.outcomes
            ),
            "unexpected_errors": sum(
                self.classify_outcome(outcome) == "unexpected_failure"
                for outcome in self.outcomes
            ),
            "cancelled_at_cleanup": self.sent - self.completed - self.errors,
        }

    def write_artifacts(self) -> None:
        (self.output / "metadata.json").write_text(
            json.dumps(
                {
                    "arguments": vars(self.args),
                    "namespace": self.namespace,
                    "python_version": sys.version,
                    "source_sha256": {
                        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                        for name in ["run.py", "off.yaml", "on.yaml"]
                    },
                    "commands": self.commands,
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
                    ).strip(),
                },
                indent=2,
            )
            + "\n"
        )
        (self.output / "summary.json").write_text(
            json.dumps(self.summary(), indent=2) + "\n"
        )
        for outcome in self.outcomes:
            outcome["classification"] = self.classify_outcome(outcome)
            for timestamp, label in [
                ("started", "send_phase"),
                ("ended", "completion_phase"),
            ]:
                outcome[label] = next(
                    (
                        phase["name"]
                        for phase in self.phases
                        if phase["start"] <= outcome[timestamp] <= phase["end"]
                    ),
                    "transition",
                )
            outcome["crossed_events"] = [
                event["action"]
                for event in self.events
                if outcome["started"] <= event["seconds"] <= outcome["ended"]
            ]
        (self.output / "outcomes.json").write_text(
            json.dumps(self.outcomes, indent=2) + "\n"
        )
        with (self.output / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(Sample.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(point) for point in self.samples)
        if len(self.samples) < 2:
            return
        figure = Figure(figsize=(10, 9), layout="constrained")
        FigureCanvasAgg(figure)
        top, middle, bottom = figure.subplots(3, 1, sharex=True)
        for index in range(1, 4):
            # One-second differences reduce burst noise while retaining the transitions.
            xs, ys = [], []
            for last in self.samples:
                prior = [
                    point for point in self.samples if point.seconds <= last.seconds - 1
                ]
                if prior:
                    first = prior[-1]
                    xs.append(last.seconds)
                    ys.append(
                        (last.received[index - 1] - first.received[index - 1])
                        / (last.seconds - first.seconds)
                    )
            top.plot(xs, ys, label=f"Backend {index}")
        send_x, send_y, received_y = [], [], []
        for last in self.samples:
            prior = [
                point for point in self.samples if point.seconds <= last.seconds - 1
            ]
            if prior:
                first = prior[-1]
                send_x.append(last.seconds)
                send_y.append((last.sent - first.sent) / (last.seconds - first.seconds))
                received_y.append(
                    (sum(last.received) - sum(first.received))
                    / (last.seconds - first.seconds)
                )
        top.plot(
            send_x, send_y, color="black", linestyle=":", label="Client sent total"
        )
        top.plot(
            send_x,
            received_y,
            color="purple",
            linestyle="--",
            label="Backend arrivals total",
        )
        top.set(
            ylabel="Request arrivals / second", title=f"Router admission {self.mode}"
        )
        top.legend()
        middle.plot(
            [p.seconds for p in self.samples],
            [p.router_pending for p in self.samples],
            label="Router pending",
        )
        middle.set(ylabel="Queued requests")
        middle.legend()
        for index, values in enumerate(
            (
                [p.inflight_1 for p in self.samples],
                [p.inflight_2 for p in self.samples],
                [p.inflight_3 for p in self.samples],
            ),
            1,
        ):
            bottom.plot(
                [p.seconds for p in self.samples], values, label=f"Backend {index}"
            )
        bottom.set(xlabel="Seconds", ylabel="Backend in-flight requests")
        bottom.legend()
        for event in self.events:
            for axis in [top, middle, bottom]:
                axis.axvline(event["seconds"], color="gray", linestyle="--")
            top.annotate(
                event["action"],
                (event["seconds"], 1),
                xycoords=("data", "axes fraction"),
                xytext=(3, -6),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=8,
            )
        for phase in self.phases:
            for axis in [top, middle, bottom]:
                axis.axvspan(phase["start"], phase["end"], color="green", alpha=0.08)
            middle.text(
                (phase["start"] + phase["end"]) / 2,
                0.96,
                phase["name"],
                transform=middle.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )
        figure.savefig(self.output / "plot.svg")
        figure.savefig(self.output / "plot.png", dpi=150)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["off", "on", "both"], default="both")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rate", type=float, default=80)
    parser.add_argument("--speedup-ratio", type=float, default=0.25)
    parser.add_argument("--window", type=float, default=5)
    parser.add_argument("--settle", type=float, default=3)
    args = parser.parse_args()
    if not all(
        math.isfinite(value) and value > 0
        for value in [args.rate, args.speedup_ratio, args.window, args.settle]
    ):
        parser.error(
            "Rate, speedup ratio, window, and settle time must be finite and positive"
        )
    for executable in ["etcd", "nats-server"]:
        if not shutil.which(executable):
            parser.error(f"Required executable is not on PATH: {executable}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ["off", "on"] if args.mode == "both" else [args.mode]:
        results.append(await Demo(args, mode, output / mode).run())
    passed = all(result["passed"] for result in results)
    (output / "summary.json").write_text(
        json.dumps({"passed": passed, "runs": results}, indent=2) + "\n"
    )
    print(f"{'PASS' if passed else 'FAIL'}: {output / 'summary.json'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
