# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage and live visualization of router overload protection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import textwrap
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.output_paths import resolve_test_output_path
from tests.utils.port_utils import ServicePorts
from tests.utils.prometheus import find_metric_samples

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(150),
]

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).with_name("configs") / "overload_protection"
REQUEST_RATE = 90
INPUT_TOKENS = 512
MAX_BATCHED_TOKENS = 8192
SPEEDUP_RATIO = 0.25
WINDOW_SECONDS = 5
SETTLE_SECONDS = 3
SAMPLE_INTERVAL = 0.5
LIVE_RATE_SECONDS = 1.0
# on.yaml uses fraction 1.0. The strict > check allows one more full request
# at the budget boundary; use that derived ceiling as 100% in both modes.
PREFILL_LIMIT_TOKENS = (MAX_BATCHED_TOKENS // INPUT_TOKENS + 1) * INPUT_TOKENS


class LiveDisplay:
    """Bypass pytest capture only for an interactive, single-process display."""

    def __init__(self, capsys: pytest.CaptureFixture[str]):
        self.capsys = capsys
        self.live_lines = 0
        # FD capture can hide the real terminal until capture is suspended.
        with capsys.disabled():
            self.enabled = (
                sys.stdout.isatty()
                and not os.environ.get("CI")
                and not os.environ.get("PYTEST_XDIST_WORKER")
            )

    def _clear(self) -> None:
        if self.live_lines:
            print(f"\033[{self.live_lines}A\r\033[J", end="", flush=True)
            self.live_lines = 0

    def write(self, text: str, *, live: bool = False) -> None:
        if text:
            logger.info("%s", text)
        if self.enabled:
            with self.capsys.disabled():
                self._clear()
                # Wrap explicitly so cursor movement counts physical rows, even
                # in a narrow terminal. Leave one column to avoid automatic wrap.
                width = max(1, shutil.get_terminal_size().columns - 1)
                lines = [
                    part
                    for line in text.split("\n")
                    for part in (
                        textwrap.wrap(
                            line,
                            width,
                            replace_whitespace=False,
                            drop_whitespace=False,
                        )
                        or [""]
                    )
                ]
                print("\n".join(lines), flush=True)
                self.live_lines = len(lines) if live else 0

    def clear(self) -> None:
        if self.enabled:
            with self.capsys.disabled():
                self._clear()


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
    prefill_tokens: tuple[float, ...]
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
    prefill_end: tuple[float, ...]


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


async def manage_process(callback):
    """Finish an in-progress lifecycle operation before cancellation tears down."""
    operation = asyncio.create_task(asyncio.to_thread(callback))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


class Deployment:
    """Compose shared process helpers with independently restartable workers."""

    def __init__(
        self,
        request: pytest.FixtureRequest,
        output: Path,
        overload_protection: bool,
        ports: ServicePorts,
        display: LiveDisplay,
    ):
        self.request, self.output, self.display = request, output, display
        self.frontend_port = ports.frontend_port
        self.worker_ports = ports.system_ports
        self.namespace = "overload-protection-" + uuid.uuid4().hex[:10]
        self.workers: list[MockerProcess | None] = [None] * 3
        self.worker_stacks = [ExitStack() for _ in range(3)]
        self.stack = ExitStack()
        self.process_pids: list[int] = []
        self.commands: list[dict] = []
        policy = CONFIG_DIR / ("on.yaml" if overload_protection else "off.yaml")
        (output / "policy.yaml").write_text(policy.read_text())
        # Isolate child settings from inherited Dynamo overrides.
        self.env = dict(
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
        self.frontend = DynamoFrontendProcess(
            request,
            frontend_port=self.frontend_port,
            router_mode="kv",
            extra_args=[
                "--namespace",
                self.namespace,
                "--discovery-backend",
                "file",
                "--router-policy-config",
                str(policy),
            ],
            extra_env=self.env,
            display_name="frontend",
        )
        self.frontend.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DYN_")
        } | self.env
        self.frontend.log_dir = str(output)
        self.frontend.display_output = False
        # Fixture teardown is a fallback if startup or the async scenario fails.
        request.addfinalizer(self.stack.close)

    def record_process(self, process: ManagedProcess) -> None:
        if process.proc is None:
            raise RuntimeError("Managed process did not start")
        self.process_pids.append(process.proc.pid)
        self.commands.append(
            {
                "name": process.display_name,
                "pid": process.proc.pid,
                "command": process.command,
            }
        )
        (self.output / "processes.json").write_text(
            json.dumps(self.commands, indent=2) + "\n"
        )

    async def start_frontend(self) -> None:
        await manage_process(lambda: self.stack.enter_context(self.frontend))
        self.record_process(self.frontend)

    async def start_worker(self, index: int, speed: float, generation: str) -> None:
        worker = MockerProcess(
            self.request,
            mocker_args={
                "max_num_seqs": 4,
                "max_num_batched_tokens": MAX_BATCHED_TOKENS,
                "enable_prefix_caching": False,
                "speedup_ratio": speed,
            },
            num_mockers=1,
            store_backend="file",
            request_plane="tcp",
            namespace=self.namespace,
            extra_env={key: None for key in os.environ if key.startswith("DYN_")}
            | self.env
            | {"DYN_SYSTEM_PORT": str(self.worker_ports[index])},
            display_name=f"worker-{index + 1}-{generation}",
        )
        worker.managed_process.log_dir = str(self.output)
        worker.managed_process.display_output = False
        worker_stack = ExitStack()
        self.worker_stacks[index] = worker_stack
        self.stack.callback(worker_stack.close)
        await manage_process(lambda: worker_stack.enter_context(worker))
        self.workers[index] = worker
        self.record_process(worker.managed_process)

    async def stop_worker(self, index: int) -> None:
        await manage_process(self.worker_stacks[index].close)
        self.workers[index] = None

    def check_alive(self) -> None:
        processes = [self.frontend] + [
            worker.managed_process for worker in self.workers if worker is not None
        ]
        for process in processes:
            if process.proc is None or process.proc.poll() is not None:
                raise RuntimeError(
                    f"{process.display_name} exited; see {process.log_path}"
                )

    async def close(self) -> None:
        await manage_process(self.stack.close)


async def read_metrics(session: aiohttp.ClientSession, port: int) -> str:
    async with session.get(
        f"http://127.0.0.1:{port}/metrics", timeout=aiohttp.ClientTimeout(total=2)
    ) as response:
        response.raise_for_status()
        return await response.text()


def metric(
    text: str, name: str, labels: dict[str, str] | None = None, *, required: bool = True
) -> float:
    values = find_metric_samples(text, name, labels)
    if required and not values:
        raise RuntimeError(f"Required metric {name} ({labels}) is absent")
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
                    {"state": "available"},
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
                "model": ROUTER_MODEL_NAME,
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


def worker_metric_id(text: str) -> str:
    """Match a worker's runtime identity to the frontend's per-worker metrics."""
    worker_ids = {
        point.labels["worker_id"]
        for family in text_string_to_metric_families(text)
        for point in family.samples
        if point.name == "dynamo_component_uptime_seconds"
    }
    if len(worker_ids) != 1:
        raise RuntimeError(f"Expected one worker identity, got {worker_ids}")
    # Runtime metric labels use hex; the frontend's load gauges use decimal.
    return str(int(worker_ids.pop(), 16))


async def sample(
    deployment: Deployment, workload: Workload, *, required: bool
) -> Sample:
    deployment.check_alive()
    received, inflight, worker_ids = [], [], []
    for index, worker in enumerate(deployment.workers):
        if worker is None:
            received.append(0.0)
            inflight.append(0.0)
            worker_ids.append(None)
            continue
        text = await read_metrics(workload.session, deployment.worker_ports[index])
        label = {"dynamo_endpoint": "generate"}
        received.append(
            metric(text, "dynamo_component_requests_total", label, required=required)
        )
        inflight.append(
            metric(text, "dynamo_component_inflight_requests", label, required=required)
        )
        # Read the identity again after restart; worker 3 gets a new runtime ID.
        worker_ids.append(worker_metric_id(text))
    text = await read_metrics(workload.session, deployment.frontend_port)
    return Sample(
        seconds=workload.seconds(),
        sent=len(workload.tasks),
        completed=workload.completed,
        received=tuple(received),
        inflight=tuple(inflight),
        prefill_tokens=tuple(
            metric(
                text,
                "dynamo_frontend_worker_active_prefill_tokens",
                {"worker_id": worker_id, "dp_rank": "0"},
                required=required,
            )
            if worker_id is not None
            else 0.0
            for worker_id in worker_ids
        ),
        router_pending=metric(text, "dynamo_frontend_router_queue_pending_requests"),
    )


def print_rates(
    display: LiveDisplay,
    name: str,
    sent_rps: float,
    arrival_rps: tuple[float, ...],
    router_pending: float,
    prefill_tokens: tuple[float, ...],
    *,
    live: bool = False,
) -> None:
    """Show equal-scale worker occupancy, arrivals, and work held at the router."""
    cells = 12  # Each half of the bar represents one admission ceiling.
    rows = [
        f"{name} | sent {sent_rps:.1f} req/s"
        + ("" if live else " | occupancy at phase end"),
        f"{'':12s}{'0%':<10s}{'100%':<12s}{'200%':>4s}"
        f" {'Load':>7s} {'ISL tok':>9s} {'Excess':>9s} {'In/s':>6s}",
    ]
    for index, (tokens, rate) in enumerate(
        zip(prefill_tokens, arrival_rps, strict=True)
    ):
        ratio = tokens / PREFILL_LIMIT_TOKENS
        within = round(min(ratio, 1.0) * cells)
        above = round(min(max(ratio - 1.0, 0.0), 1.0) * cells)
        # Preserve visible small overages; numbers always show the exact load.
        if ratio > 1.0:
            above = max(1, above)
        bar = "#" * within + "." * (cells - within)
        bar += "|" + "+" * above + "." * (cells - above)
        edge = ">" if ratio > 2.0 else "]"
        label = f"W{index + 1}"
        if index == 2:
            if name.startswith("removed"):
                label += " stopped"
            elif name.startswith("slow"):
                label += " 50% spd"
        excess = max(0, tokens - PREFILL_LIMIT_TOKENS)
        rows.append(
            f"{label:11s}[{bar}{edge} {ratio * 100:6.1f}%"
            f" {tokens:9,.0f} {excess:9,.0f} {rate:6.1f}"
        )
    rows.append(
        f"Router waiting: {router_pending:,.0f} requests"
        f" ({router_pending * INPUT_TOKENS:,.0f} ISL tokens)"
    )
    display.write("\n".join(rows), live=live)


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
                            deployment.display,
                            label,
                            (point.sent - first.sent) / elapsed,
                            tuple(
                                (end - start) / elapsed
                                for start, end in zip(
                                    first.received, point.received, strict=True
                                )
                            ),
                            point.router_pending,
                            point.prefill_tokens,
                            live=True,
                        )
                if loop.time() >= deadline:
                    return points
                await asyncio.sleep(max(0, next_sample - loop.time()))
        finally:
            deployment.display.clear()

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
        prefill_end=last.prefill_tokens,
    )
    print_rates(
        deployment.display,
        f"{name} avg",
        window.sent_rps,
        window.arrival_rps,
        window.queue_end,
        window.prefill_end,
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
        deployment.display.write(
            f"Stopping worker 3; the client continues at {REQUEST_RATE} requests/s..."
        )
        stopped_at = workload.seconds()
        await deployment.stop_worker(2)
        await wait_for_workers(deployment, session, 2)
        worker_removed_at = workload.seconds()
        windows.append(
            await observe_window(deployment, workload, load_task, "removed", samples)
        )
        deployment.display.write("Restarting worker 3 at half speed...")
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
        process_pids=tuple(deployment.process_pids),
    )


async def run_scenario(
    deployment: Deployment, overload_protection: bool
) -> ClientResult:
    display = deployment.display
    display.write(
        f"Overload protection {'ON' if overload_protection else 'OFF'} | client {REQUEST_RATE} requests/s"
    )
    display.write(f"Logs and measurements: {deployment.output}")
    display.write("Starting one frontend and three CPU Mocker workers...")
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0)
        ) as session:
            await deployment.start_frontend()
            for index in range(3):
                await deployment.start_worker(index, SPEEDUP_RATIO, "initial")
            await wait_for_workers(deployment, session, 3)
            display.write(
                f"Live rates refresh every {SAMPLE_INTERVAL}s over ~{LIVE_RATE_SECONDS:g}s; avg rows measure {WINDOW_SECONDS}s after {SETTLE_SECONDS}s settling."
            )
            display.write(
                f"100% = {PREFILL_LIMIT_TOKENS:,} outstanding prefill tokens"
                f" ({MAX_BATCHED_TOKENS:,} budget + one {INPUT_TOKENS}-token request)."
            )
            display.write(
                "# within admission limit | + excess ISL | > continues beyond 200%"
            )
            result = await drive_scenario(deployment, session)
            (deployment.output / "summary.json").write_text(
                json.dumps(asdict(result), indent=2) + "\n"
            )
    finally:
        await deployment.close()
    shutdown = sum(result.is_shutdown_failure(item) for item in result.outcomes)
    unexpected = sum(
        item.status != 200 and not result.is_shutdown_failure(item)
        for item in result.outcomes
    )
    display.write(
        f"Stopped all servers. Shutdown 503s: {shutdown}; unexpected errors: {unexpected}; cancelled at cleanup: {result.cancelled_at_cleanup}."
    )
    return result


@pytest.mark.usefixtures("predownload_tokenizers")
@pytest.mark.parametrize("num_system_ports", [3], indirect=True)
@pytest.mark.parametrize("overload_protection", [False, True], ids=["off", "on"])
def test_overload_protection(
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
    overload_protection: bool,
    dynamo_dynamic_ports: ServicePorts,
) -> None:
    """The same fixed-rate workload exercises admission on and off."""
    output = Path(resolve_test_output_path(request.node.name))
    output.mkdir(parents=True, exist_ok=True)
    run_output = Path(tempfile.mkdtemp(prefix="overload-", dir=output))
    deployment = Deployment(
        request,
        run_output,
        overload_protection,
        dynamo_dynamic_ports,
        LiveDisplay(capsys),
    )
    result = asyncio.run(run_scenario(deployment, overload_protection))

    healthy, removed, slow = result.windows
    reduced_capacity = (removed, slow)
    assert all(
        abs(window.sent_rps / REQUEST_RATE - 1) < 0.1 for window in result.windows
    ), result.windows
    healthy_mean_rps = sum(healthy.arrival_rps) / 3
    assert healthy_mean_rps > 0, healthy
    assert all(
        abs(rate / healthy_mean_rps - 1) < 0.1 for rate in healthy.arrival_rps
    ), healthy
    assert removed.arrival_rps[2] == 0, removed
    assert all(window.completed_rps > 0 for window in result.windows), result.windows
    failures = [
        outcome
        for outcome in result.outcomes
        if outcome.status != 200 and not result.is_shutdown_failure(outcome)
    ]
    assert not failures, failures[:5]
    assert not [
        pid for pid in result.process_pids if Path("/proc", str(pid)).exists()
    ], result.process_pids

    if overload_protection:
        assert all(
            abs(window.arrival_rps[index] / healthy.arrival_rps[index] - 1) <= 0.25
            for window in reduced_capacity
            for index in (0, 1)
        ), result.windows
        assert 0.35 <= slow.arrival_rps[2] / healthy.arrival_rps[2] <= 0.65, (
            healthy,
            slow,
        )
        assert all(window.queue_mean > 1 for window in result.windows), result.windows
        assert all(window.queue_delta > 0 for window in result.windows), result.windows
        assert all(
            sum(window.arrival_rps) < 0.95 * window.sent_rps
            for window in result.windows
        ), result.windows
        # Fraction 1.0 admits one prefill beyond the worker's token budget.
        # Allow two more requests for first-token/response accounting overlap.
        inflight_limit = MAX_BATCHED_TOKENS // INPUT_TOKENS + 1 + 2
        assert all(
            window.inflight_max <= inflight_limit for window in result.windows
        ), result.windows
    else:
        assert all(
            abs(rate / (healthy.sent_rps / 3) - 1) < 0.1 for rate in healthy.arrival_rps
        ), healthy
        assert all(
            abs(sum(window.arrival_rps) / window.sent_rps - 1) < 0.1
            for window in result.windows
        ), result.windows
        assert all(
            window.inflight_delta > 0 for window in result.windows
        ), result.windows
        assert all(window.queue_mean < 1 for window in result.windows), result.windows
