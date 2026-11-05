# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show router backpressure while a CPU Mocker worker stops and returns slower."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import socket
import sys
import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-0.6B"
REQUEST_RATE = 90
INPUT_TOKENS = 512
MAX_BATCHED_TOKENS = 8192
SPEEDUP_RATIO = 0.25
WINDOW_SECONDS = 5
SETTLE_SECONDS = 3
SAMPLE_INTERVAL = 0.5
LIVE_RATE_SECONDS = 1.0


@dataclass(frozen=True)
class HttpOutcome:
    started: float
    ended: float
    status: int
    error: str


@dataclass(frozen=True)
class Sample:
    seconds: float
    sent: int
    completed: int
    received: tuple[float, ...]
    inflight: tuple[float, ...]
    router_pending: float


@dataclass(frozen=True)
class Window:
    name: str
    start: float
    end: float
    arrival_rps: tuple[float, ...]
    sent_rps: float
    completed_rps: float
    queue_mean: float
    queue_delta: float
    inflight_delta: float
    inflight_max: float
    queue_end: float
    inflight_end: tuple[float, ...]


@dataclass(frozen=True)
class ClientResult:
    windows: tuple[Window, ...]
    samples: tuple[Sample, ...]
    outcomes: tuple[HttpOutcome, ...]
    stopped_at: float
    worker_removed_at: float
    sent: int
    cancelled_at_cleanup: int
    output_dir: str
    process_pids: tuple[int, ...]

    def is_shutdown_failure(self, outcome: HttpOutcome) -> bool:
        # Discovery can briefly route requests to the worker being stopped.
        return (
            outcome.status == 503
            and outcome.started < self.worker_removed_at
            and self.stopped_at <= outcome.ended < self.windows[1].start
        )


def free_ports(count: int) -> list[int]:
    """Choose unused ports below the runtime metrics port's signed-16-bit limit."""
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        candidates = iter(random.SystemRandom().sample(range(15000, 30000), 15000))
        for sock in sockets:
            for port in candidates:
                try:
                    sock.bind(("127.0.0.1", port))
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("No available demo ports in 15000..29999")
        return [sock.getsockname()[1] for sock in sockets]


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop only this demo's process group, allowing graceful deregistration."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), 3)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # It exited between the timeout and the signal.
        await process.wait()


class Deployment:
    """Frontend and independently restartable workers, with one shared file store."""

    def __init__(
        self,
        output: Path,
        model: str,
        backpressure: bool,
        ports: tuple[int, int, int, int] | None = None,
    ):
        self.output, self.model = output, model
        # Pytest supplies reserved ports; standalone runs choose available ports.
        self.frontend_port, *self.worker_ports = (
            free_ports(4) if ports is None else ports
        )
        self.namespace = "backpressure-" + uuid.uuid4().hex[:10]
        self.frontend: asyncio.subprocess.Process | None = None
        self.workers: list[asyncio.subprocess.Process | None] = [None] * 3
        self.processes: list[asyncio.subprocess.Process] = []
        self.commands: list[dict] = []
        policy = HERE / ("on.yaml" if backpressure else "off.yaml")
        (output / "policy.yaml").write_text(policy.read_text())
        # Ignore inherited Dynamo overrides so only our deployment settings apply.
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DYN_")
        }
        self.env.update(
            DYN_FILE_KV=str(output / "discovery"),
            DYN_REQUEST_PLANE="tcp",
            DYN_EVENT_PLANE="zmq",
            DYN_RESPONSE_PLANE="tcp",
            DYN_LOG="warn",
            DYN_COMPUTE_THREADS="1",
            RAYON_NUM_THREADS="2",
            OMP_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )

    async def spawn(
        self, name: str, command: list[str], **env
    ) -> asyncio.subprocess.Process:
        with (self.output / f"{name}.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=self.env | env,
                start_new_session=True,
            )
        self.processes.append(process)
        self.commands.append({"name": name, "pid": process.pid, "command": command})
        (self.output / "processes.json").write_text(
            json.dumps(self.commands, indent=2) + "\n"
        )
        return process

    async def start_frontend(self) -> None:
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(self.frontend_port),
            "--namespace",
            self.namespace,
            "--discovery-backend",
            "file",
            "--router-mode",
            "kv",
            "--router-policy-config",
            str(self.output / "policy.yaml"),
        ]
        self.frontend = await self.spawn("frontend", command)

    async def start_worker(self, index: int, speed: float, generation: str) -> None:
        self.workers[index] = await self.spawn(
            f"worker-{index + 1}-{generation}",
            [
                sys.executable,
                "-m",
                "dynamo.mocker",
                "--model-path",
                self.model,
                "--model-name",
                "backpressure",
                "--endpoint",
                f"dyn://{self.namespace}.mocker.generate",
                "--discovery-backend",
                "file",
                "--num-workers",
                "1",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                str(MAX_BATCHED_TOKENS),
                "--no-enable-prefix-caching",
                "--speedup-ratio",
                str(speed),
            ],
            DYN_SYSTEM_PORT=str(self.worker_ports[index]),
        )

    async def stop_worker(self, index: int) -> None:
        process = self.workers[index]
        if process is None:
            raise RuntimeError(f"Worker {index + 1} is not running")
        self.workers[index] = None
        await stop_process(process)

    def check_alive(self) -> None:
        for process in [self.frontend, *self.workers]:
            if process is not None and process.returncode is not None:
                raise RuntimeError(
                    f"Process {process.pid} exited with {process.returncode}; see {self.output}"
                )

    async def close(self) -> None:
        for process in reversed(self.processes):
            await stop_process(process)


async def read_metrics(session: aiohttp.ClientSession, port: int) -> str:
    async with session.get(
        f"http://127.0.0.1:{port}/metrics", timeout=aiohttp.ClientTimeout(total=2)
    ) as response:
        response.raise_for_status()
        return await response.text()


def metric(text: str, name: str, label: str = "", *, required: bool = True) -> float:
    values = [
        float(line.split()[-1])
        for line in text.splitlines()
        if (line.startswith(name + "{") or line.startswith(name + " "))
        and label in line.split()[0]
    ]
    if required and not values:
        raise RuntimeError(f"Required metric {name} ({label}) is absent")
    return sum(values)


async def wait_for_workers(
    deployment: Deployment, session: aiohttp.ClientSession, count: int
) -> None:
    async with asyncio.timeout(90):
        while True:
            deployment.check_alive()
            try:
                text = await read_metrics(session, deployment.frontend_port)
                available = metric(
                    text,
                    "dynamo_frontend_router_workers",
                    'state="available"',
                    required=False,
                )
                if available == count:
                    for index, worker in enumerate(deployment.workers):
                        if worker is not None:
                            await read_metrics(session, deployment.worker_ports[index])
                    return
            except (aiohttp.ClientConnectionError, TimeoutError):
                pass  # The process may still be starting its HTTP server.
            await asyncio.sleep(0.2)


@dataclass
class Workload:
    session: aiohttp.ClientSession
    frontend_port: int
    started_at: float
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    outcomes: list[HttpOutcome] = field(default_factory=list)
    tasks: list[asyncio.Task] = field(default_factory=list)
    completed: int = 0

    def seconds(self) -> float:
        return asyncio.get_running_loop().time() - self.started_at


async def send_request(workload: Workload, number: int) -> None:
    started = workload.seconds()
    try:
        async with workload.session.post(
            f"http://127.0.0.1:{workload.frontend_port}/v1/completions",
            json={
                "model": "backpressure",
                "prompt": [1000 + number % 10000] * INPUT_TOKENS,
                "max_tokens": 1,
                "stream": False,
                "temperature": 0,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            body = await response.text()
            status, error = (
                response.status,
                body[:500] if response.status != 200 else "",
            )
    except (aiohttp.ClientError, TimeoutError) as exc:
        status, error = 0, str(exc)
    workload.outcomes.append(HttpOutcome(started, workload.seconds(), status, error))
    workload.completed += status == 200


async def send_fixed_rate(workload: Workload) -> None:
    """Use the same open-loop client in both modes, independent of response speed."""
    while not workload.stop.is_set():
        number = len(workload.tasks)
        workload.tasks.append(asyncio.create_task(send_request(workload, number)))
        await asyncio.sleep(max(0, (number + 1) / REQUEST_RATE - workload.seconds()))


async def sample(
    deployment: Deployment, workload: Workload, *, required: bool
) -> Sample:
    deployment.check_alive()
    received, inflight = [], []
    for index, worker in enumerate(deployment.workers):
        if worker is None:
            received.append(0.0)
            inflight.append(0.0)
            continue
        text = await read_metrics(workload.session, deployment.worker_ports[index])
        label = 'dynamo_endpoint="generate"'
        received.append(
            metric(text, "dynamo_component_requests_total", label, required=required)
        )
        inflight.append(
            metric(text, "dynamo_component_inflight_requests", label, required=required)
        )
    text = await read_metrics(workload.session, deployment.frontend_port)
    return Sample(
        seconds=workload.seconds(),
        sent=len(workload.tasks),
        completed=workload.completed,
        received=tuple(received),
        inflight=tuple(inflight),
        router_pending=metric(text, "dynamo_frontend_router_queue_pending_requests"),
    )


def print_rates(
    name: str,
    sent_rps: float,
    arrival_rps: tuple[float, ...],
    router_pending: float,
    inflight: tuple[float, ...],
    *,
    live: bool = False,
) -> None:
    rates = " ".join(f"{rate:7.1f}" for rate in arrival_rps)
    active = "/".join(str(int(value)) for value in inflight)
    row = f"{name:16s} {sent_rps:7.1f} {rates} {router_pending:12.0f} {active:>17s}"
    if live and sys.stdout.isatty():
        print("\r" + row + "\033[K", end="", flush=True)
    else:
        print(row, flush=True)


async def observe_window(
    deployment: Deployment,
    workload: Workload,
    load_task: asyncio.Task,
    name: str,
    samples: list[Sample],
) -> Window:
    async def observe(duration: float, *, required: bool) -> list[Sample]:
        loop = asyncio.get_running_loop()
        points, deadline = [], loop.time() + duration
        label = f"{name} {'live' if required else 'settling'}"
        try:
            while True:
                next_sample = min(loop.time() + SAMPLE_INTERVAL, deadline)
                if load_task.done():
                    await load_task
                    raise RuntimeError("Client stopped before the measurement finished")
                point = await sample(deployment, workload, required=required)
                points.append(point)
                samples.append(point)
                if len(points) > 1:
                    # Use roughly the latest second to smooth short dispatch bursts.
                    first = next(
                        (
                            previous
                            for previous in reversed(points[:-1])
                            if point.seconds - previous.seconds >= LIVE_RATE_SECONDS
                        ),
                        points[0],
                    )
                    elapsed = point.seconds - first.seconds
                    if elapsed > 0:
                        print_rates(
                            label,
                            (point.sent - first.sent) / elapsed,
                            tuple(
                                (end - start) / elapsed
                                for start, end in zip(
                                    first.received, point.received, strict=True
                                )
                            ),
                            point.router_pending,
                            point.inflight,
                            live=True,
                        )
                if loop.time() >= deadline:
                    return points
                await asyncio.sleep(max(0, next_sample - loop.time()))
        finally:
            if sys.stdout.isatty():
                # Clear the transient row before a summary, transition, or error.
                print("\r\033[K", end="", flush=True)

    await observe(SETTLE_SECONDS, required=False)
    points = await observe(WINDOW_SECONDS, required=True)
    first, last = points[0], points[-1]
    elapsed = last.seconds - first.seconds
    # Each phase gets its own baseline, including worker 3's reset counter.
    window = Window(
        name=name,
        start=first.seconds,
        end=last.seconds,
        arrival_rps=tuple(
            (end - start) / elapsed
            for start, end in zip(first.received, last.received, strict=True)
        ),
        sent_rps=(last.sent - first.sent) / elapsed,
        completed_rps=(last.completed - first.completed) / elapsed,
        queue_mean=sum(point.router_pending for point in points) / len(points),
        queue_delta=last.router_pending - first.router_pending,
        inflight_delta=sum(last.inflight) - sum(first.inflight),
        inflight_max=max(max(point.inflight) for point in points),
        queue_end=last.router_pending,
        inflight_end=last.inflight,
    )
    print_rates(
        f"{name} avg",
        window.sent_rps,
        window.arrival_rps,
        window.queue_end,
        window.inflight_end,
    )
    return window


async def drive_scenario(
    deployment: Deployment, session: aiohttp.ClientSession
) -> ClientResult:
    """Keep the client running through healthy, removed, and half-speed phases."""
    workload = Workload(
        session, deployment.frontend_port, asyncio.get_running_loop().time()
    )
    load_task = asyncio.create_task(send_fixed_rate(workload))
    windows, samples = [], []
    try:
        windows.append(
            await observe_window(deployment, workload, load_task, "healthy", samples)
        )
        print(
            f"Stopping worker 3; the client continues at {REQUEST_RATE} requests/s...",
            flush=True,
        )
        stopped_at = workload.seconds()
        await deployment.stop_worker(2)
        await wait_for_workers(deployment, session, 2)
        worker_removed_at = workload.seconds()
        windows.append(
            await observe_window(deployment, workload, load_task, "removed", samples)
        )
        print("Restarting worker 3 at half speed...", flush=True)
        await deployment.start_worker(2, SPEEDUP_RATIO / 2, "slow")
        await wait_for_workers(deployment, session, 3)
        windows.append(
            await observe_window(deployment, workload, load_task, "slow", samples)
        )
    finally:
        workload.stop.set()
        try:
            await load_task
        finally:
            pending = [task for task in workload.tasks if not task.done()]
            for task in pending:
                task.cancel()
            finished = await asyncio.gather(*workload.tasks, return_exceptions=True)
            failures = [
                item
                for item in finished
                if isinstance(item, BaseException)
                and not isinstance(item, asyncio.CancelledError)
            ]
            if failures:
                raise RuntimeError(f"Client task failed: {failures[0]}") from failures[
                    0
                ]
    return ClientResult(
        windows=tuple(windows),
        samples=tuple(samples),
        outcomes=tuple(workload.outcomes),
        stopped_at=stopped_at,
        worker_removed_at=worker_removed_at,
        sent=len(workload.tasks),
        cancelled_at_cleanup=len(pending),
        output_dir=str(deployment.output),
        process_pids=tuple(process.pid for process in deployment.processes),
    )


async def run_demo(
    *,
    backpressure: bool = True,
    output_dir: Path | None = None,
    model: str = MODEL,
    ports: tuple[int, int, int, int] | None = None,
) -> ClientResult:
    """Run the example and return measurements for its CI test.

    Optional ports are frontend HTTP, then worker 1/2/3 metrics. Pytest can
    reserve these across concurrent tests; the CLI chooses available ports.
    """
    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(tempfile.mkdtemp(prefix="dynamo-backpressure-", dir="/tmp"))
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    print(
        f"Backpressure {'ON' if backpressure else 'OFF'} | client {REQUEST_RATE} requests/s",
        flush=True,
    )
    print(f"Logs and discovery directory: {output}", flush=True)
    print("Starting one frontend and three CPU Mocker workers...", flush=True)
    deployment = Deployment(output, model, backpressure, ports=ports)
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            await deployment.start_frontend()
            for index in range(3):
                await deployment.start_worker(index, SPEEDUP_RATIO, "initial")
            await wait_for_workers(deployment, session, 3)
            print(
                f"Live rates refresh every {SAMPLE_INTERVAL}s over ~{LIVE_RATE_SECONDS:g}s; avg rows measure {WINDOW_SECONDS}s after {SETTLE_SECONDS}s settling.",
                flush=True,
            )
            print(
                f"{'Phase':16s} {'Sent/s':>7s} {'W1/s':>7s} {'W2/s':>7s} {'W3/s':>7s} {'Router queue':>12s} {'In-flight W1/2/3':>17s}",
                flush=True,
            )
            result = await drive_scenario(deployment, session)
    finally:
        await deployment.close()
    (output / "summary.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    shutdown = sum(result.is_shutdown_failure(item) for item in result.outcomes)
    unexpected = sum(
        item.status != 200 and not result.is_shutdown_failure(item)
        for item in result.outcomes
    )
    print(
        f"Stopped all servers. Shutdown 503s: {shutdown}; unexpected errors: {unexpected}; cancelled at cleanup: {result.cancelled_at_cleanup}.",
        flush=True,
    )
    print(f"Measurements: {output / 'summary.json'}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-backpressure",
        action="store_true",
        help="Use off.yaml with the same workers and client workload",
    )
    args = parser.parse_args()
    result = asyncio.run(run_demo(backpressure=not args.no_backpressure))
    return int(
        any(
            item.status != 200 and not result.is_shutdown_failure(item)
            for item in result.outcomes
        )
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
