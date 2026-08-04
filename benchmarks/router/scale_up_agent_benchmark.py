# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run live, multi-turn Dynamo affinity lifecycle experiments.

The controller owns only the processes it starts. It deliberately uses a fresh
namespace and dynamically allocated ports so it can run beside another Dynamo
deployment on the same host. NATS and etcd are expected to be running already.

For scale-up it starts worker A, then starts worker B after every agent reaches
the configured trigger turn. For scale-down it starts both workers, stops worker
B at that trigger, and waits for frontend discovery to observe the removal. An
in-process workload sends ordinary OpenAI chat requests with one stable
``x-dynamo-session-id`` per agent and retries requests that race scale-down. The
resulting JSONL event streams and Dynamo structured logs are checked for
deterministic migration or rebinding as well as worker/rank stickiness.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_SERVED_MODEL = "dynamo-scale-up-agent-benchmark"
ROUTING_PATHS = ("token", "text")
SCENARIOS = ("a-only", "ab-start", "scale-up", "scale-down")
ROUTE_MESSAGES = {
    "token": "Selected token-input KV target",
    "text": "Observed text-input KV session target",
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record.setdefault("_line", line_number)
                records.append(record)
    return records


def git_state(repo: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    return {
        "commit": revision,
        "dirty": bool(status),
        "status": status,
    }


def http_get(url: str, timeout: float = 2.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "dynamo-scale-up"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def wait_for_model(
    frontend_port: int,
    model: str,
    timeout: float,
    processes: Sequence[ManagedCommand] = (),
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    url = f"http://127.0.0.1:{frontend_port}/v1/models"
    while time.monotonic() < deadline:
        for process in processes:
            process.require_running()
        try:
            payload = json.loads(http_get(url))
            models = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if model in models:
                return
            last_error = (
                f"available models: {sorted(value for value in models if value)}"
            )
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"model {model!r} did not appear at {url}: {last_error}")


class PortAllocator:
    def __init__(self) -> None:
        self._allocated: set[int] = set()

    @staticmethod
    def _available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def below(self, maximum: int) -> int:
        for port in range(10000, maximum):
            if port not in self._allocated and self._available(port):
                self._allocated.add(port)
                return port
        raise RuntimeError(f"could not allocate a port below {maximum}")

    def contiguous(self, count: int) -> int:
        for base in range(20000, 60000 - count):
            ports = range(base, base + count)
            if all(
                port not in self._allocated and self._available(port) for port in ports
            ):
                self._allocated.update(ports)
                return base
        raise RuntimeError(f"could not allocate {count} contiguous ports")


@dataclass
class ManagedCommand:
    name: str
    command: list[str]
    log_path: Path
    cwd: Path
    env: dict[str, str]
    process: subprocess.Popen[str] | None = field(default=None, init=False)
    _log_handle: Any = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError(f"{self.name} already started")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def require_running(self) -> None:
        return_code = self.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{self.name} exited unexpectedly with status {return_code}; "
                f"see {self.log_path}"
            )

    def wait(self, timeout: float | None = None) -> int:
        if self.process is None:
            raise RuntimeError(f"{self.name} has not started")
        return self.process.wait(timeout=timeout)

    def stop(self, timeout: float = 30.0) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class Sampler:
    def __init__(
        self,
        output_path: Path,
        endpoints: dict[str, str],
        gpu_ids: Sequence[int],
        interval: float,
    ) -> None:
        self.output_path = output_path
        self.endpoints = endpoints
        self.gpu_ids = list(gpu_ids)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False

    def start(self) -> None:
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.started:
            self.thread.join(timeout=max(5.0, self.interval * 4))

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                observed_at = utc_now()
                for name, url in self.endpoints.items():
                    try:
                        body = http_get(url, timeout=1.0)
                        lines = [
                            line
                            for line in body.splitlines()
                            if line
                            and not line.startswith("#")
                            and (
                                "dynamo_" in line
                                or "sglang:" in line
                                or "sglang_" in line
                            )
                        ]
                        record: dict[str, Any] = {
                            "timestamp": observed_at,
                            "source": name,
                            "metrics": lines,
                        }
                    except Exception as error:  # noqa: BLE001 - sampling is best effort
                        record = {
                            "timestamp": observed_at,
                            "source": name,
                            "error": str(error),
                        }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

                try:
                    query = (
                        "index,uuid,memory.used,memory.total,utilization.gpu,"
                        "utilization.memory,power.draw,temperature.gpu"
                    )
                    completed = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={','.join(str(gpu) for gpu in self.gpu_ids)}",
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "timestamp": observed_at,
                                "source": "nvidia-smi",
                                "returncode": completed.returncode,
                                "rows": completed.stdout.strip().splitlines(),
                                "stderr": completed.stderr.strip() or None,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                except Exception as error:  # noqa: BLE001 - sampling is best effort
                    handle.write(
                        json.dumps(
                            {
                                "timestamp": observed_at,
                                "source": "nvidia-smi",
                                "error": str(error),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                self.stop_event.wait(self.interval)


@dataclass(frozen=True)
class CasePorts:
    nats: int
    frontend: int
    engine_a: int
    engine_b: int
    dist_init_a: int
    dist_init_b: int
    system_a: int
    system_b: int
    kv_events_a: int
    kv_events_b: int
    forward_pass_a: int
    forward_pass_b: int


@dataclass
class CaseConfig:
    routing_path: str
    scenario: str
    output_dir: Path
    namespace: str
    ports: CasePorts


def allocate_case_ports() -> CasePorts:
    allocator = PortAllocator()
    return CasePorts(
        nats=allocator.below(32768),
        frontend=allocator.below(32768),
        engine_a=allocator.below(32768),
        engine_b=allocator.below(32768),
        # With DP attention SGLang consumes dist_init through dist_init + 5
        # for its controller/tokenizer ZMQ endpoints. Give each engine an
        # explicit, non-overlapping block instead of relying on the default
        # ``engine_port + ZMQ_TCP_PORT_DELTA`` calculation.
        dist_init_a=allocator.contiguous(6),
        dist_init_b=allocator.contiguous(6),
        system_a=allocator.below(32768),
        system_b=allocator.below(32768),
        kv_events_a=allocator.contiguous(2),
        kv_events_b=allocator.contiguous(2),
        forward_pass_a=allocator.below(32768),
        forward_pass_b=allocator.below(32768),
    )


def base_environment(args: argparse.Namespace, case: CaseConfig) -> dict[str, str]:
    env = os.environ.copy()
    # Do not resolve the venv's python symlink: its sibling directory contains
    # tools (notably ninja) that SGLang JIT subprocesses must inherit.
    python_bin = str(Path(args.python).expanduser().absolute().parent)
    env.update(
        {
            "DYN_NAMESPACE": case.namespace,
            "DYN_REQUEST_PLANE": "nats",
            "NATS_SERVER": args.nats_server or f"nats://127.0.0.1:{case.ports.nats}",
            "ETCD_ENDPOINTS": args.etcd_endpoints,
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "PATH": ":".join(
                value for value in (python_bin, env.get("PATH", "")) if value
            ),
        }
    )
    return env


def worker_command(
    args: argparse.Namespace,
    case: CaseConfig,
    label: str,
) -> ManagedCommand:
    is_a = label == "a"
    engine_port = case.ports.engine_a if is_a else case.ports.engine_b
    dist_init_port = case.ports.dist_init_a if is_a else case.ports.dist_init_b
    system_port = case.ports.system_a if is_a else case.ports.system_b
    kv_port = case.ports.kv_events_a if is_a else case.ports.kv_events_b
    fpm_port = case.ports.forward_pass_a if is_a else case.ports.forward_pass_b
    gpu_ids = args.gpu_a if is_a else args.gpu_b
    mem_fraction_static = (
        args.mem_fraction_static_a if is_a else args.mem_fraction_static_b
    )
    if mem_fraction_static is None:
        mem_fraction_static = args.mem_fraction_static
    env = base_environment(args, case)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
            "DYN_SYSTEM_PORT": str(system_port),
            "DYN_FORWARDPASS_METRIC_PORT": str(fpm_port),
            "DYN_OPENAI_BACKEND_LOAD_REPORT_INTERVAL_SECS": str(
                args.load_report_interval
            ),
            "LD_LIBRARY_PATH": ":".join(
                value
                for value in (
                    str(args.cuda_library_dir),
                    env.get("LD_LIBRARY_PATH", ""),
                )
                if value
            ),
        }
    )

    engine_args = [
        "--page-size",
        str(args.block_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--context-length",
        str(args.context_length),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--enable-metrics",
        "--dp-size",
        str(args.dp_size),
        "--tp-size",
        str(args.dp_size),
    ]
    if args.dp_size > 1:
        engine_args.extend(
            [
                "--enable-dp-attention",
                "--dist-init-addr",
                f"127.0.0.1:{dist_init_port}",
            ]
        )
    if case.routing_path == "token":
        kv_config = json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_port}",
            },
            separators=(",", ":"),
        )
        command = [
            args.python,
            "-m",
            "dynamo.sglang",
            "--model-path",
            args.model,
            "--served-model-name",
            args.served_model_name,
            "--port",
            str(engine_port),
            *engine_args,
            "--kv-events-config",
            kv_config,
        ]
    else:
        command = [
            args.python,
            "-m",
            "dynamo.openai_backend.sglang",
            "--model",
            args.model,
            "--served-model-name",
            args.served_model_name,
            "--engine-port",
            str(engine_port),
            "--",
            *engine_args,
        ]

    return ManagedCommand(
        name=f"worker-{label}",
        command=command,
        log_path=case.output_dir / f"worker-{label}.log",
        cwd=args.repo,
        env=env,
    )


def frontend_command(args: argparse.Namespace, case: CaseConfig) -> ManagedCommand:
    env = base_environment(args, case)
    env.update(
        {
            "DYN_ROUTER_MIN_INITIAL_WORKERS": (
                "2" if case.scenario in ("ab-start", "scale-down") else "1"
            ),
            "DYN_LOGGING_JSONL": "1",
            "DYN_LOG": "info,dynamo_llm=debug",
        }
    )
    command = [
        args.python,
        "-m",
        "dynamo.frontend",
        "--router-mode",
        "kv",
        "--http-port",
        str(case.ports.frontend),
        "--discovery-backend",
        "etcd",
        "--namespace",
        case.namespace,
        "--kv-cache-block-size",
        str(args.block_size),
        "--router-session-affinity-ttl-secs",
        str(args.affinity_ttl),
    ]
    return ManagedCommand(
        name="frontend",
        command=command,
        log_path=case.output_dir / "frontend.log",
        cwd=args.repo,
        env=env,
    )


def nats_command(args: argparse.Namespace, case: CaseConfig) -> ManagedCommand | None:
    if args.nats_server is not None:
        return None
    return ManagedCommand(
        name="nats-server",
        command=[args.nats_server_bin, "-p", str(case.ports.nats)],
        log_path=case.output_dir / "nats-server.log",
        cwd=args.repo,
        env=os.environ.copy(),
    )


SYNTHETIC_WORDS = (
    "amber",
    "bridge",
    "cedar",
    "delta",
    "ember",
    "forest",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "kinetic",
    "lantern",
    "meadow",
    "nectar",
    "orbit",
    "prairie",
    "quartz",
    "river",
    "summit",
    "timber",
    "umber",
    "valley",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
)


def bounded_lognormal(
    rng: random.Random, median: float, sigma: float, maximum: int
) -> int:
    """Draw a positive integer, treating a zero median as a disabled delay."""
    if median == 0:
        return 0
    value = math.exp(rng.normalvariate(math.log(median), sigma))
    return min(maximum, max(1, round(value)))


def synthetic_text(
    rng: random.Random, *, agent_id: int, turn: int, word_count: int
) -> str:
    """Build deterministic text without relying on an external dataset generator."""
    prefix = f"Agent {agent_id}, turn {turn}. "
    words = " ".join(rng.choice(SYNTHETIC_WORDS) for _ in range(word_count))
    return prefix + words


@dataclass(frozen=True)
class ChatResult:
    assistant_text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


class AsyncTurnBarrier:
    """Release every agent together once all have completed the trigger turn."""

    def __init__(self, parties: int, opened: threading.Event) -> None:
        self.parties = parties
        self.opened = opened
        self.arrived = 0
        self.condition = asyncio.Condition()

    async def wait(self) -> None:
        async with self.condition:
            self.arrived += 1
            if self.arrived == self.parties:
                self.opened.set()
                self.condition.notify_all()
                return
            await self.condition.wait_for(lambda: self.arrived == self.parties)


class AgentWorkload:
    """Run concurrent, stateful OpenAI chat sessions in a background thread."""

    def __init__(self, args: argparse.Namespace, case: CaseConfig) -> None:
        self.args = args
        self.case = case
        self.events_path = case.output_dir / "agent-events.jsonl"
        self.log_path = case.output_dir / "workload.log"
        self.trigger_reached = threading.Event()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._root_task: asyncio.Task[None] | None = None
        self._exit_code: int | None = None
        self._stopping = False
        self._error: str | None = None

    def start(self) -> None:
        self.thread.start()

    def poll(self) -> int | None:
        return None if self.thread.is_alive() else self._exit_code

    def stop(self) -> None:
        self._stopping = True
        if (
            self.thread.is_alive()
            and self._loop is not None
            and not self._loop.is_closed()
            and self._root_task is not None
        ):
            self._loop.call_soon_threadsafe(self._root_task.cancel)
        if self.thread.is_alive():
            self.thread.join(timeout=10)

    @property
    def error(self) -> str | None:
        return self._error

    def _thread_main(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            asyncio.run(self._run())
            self._exit_code = 0
        except asyncio.CancelledError as error:
            if self._stopping:
                self._exit_code = -signal.SIGTERM
                return
            self._record_failure(error)
        except Exception as error:
            self._record_failure(error)

    def _record_failure(self, error: BaseException) -> None:
        self._error = f"{type(error).__name__}: {error}"
        self._exit_code = 1
        with self.log_path.open("w", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._root_task = asyncio.current_task()
        self.events_path.unlink(missing_ok=True)
        barrier = AsyncTurnBarrier(self.args.agents, self.trigger_reached)
        timeout = aiohttp.ClientTimeout(total=self.args.request_timeout)
        connector = aiohttp.TCPConnector(limit=self.args.agents)
        headers = {
            "Authorization": "Bearer NOT USED",
            "Accept": "application/json",
        }
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as client:
            tasks = [
                asyncio.create_task(self._run_agent(client, barrier, agent_id))
                for agent_id in range(self.args.agents)
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    async def _run_agent(
        self,
        client: aiohttp.ClientSession,
        barrier: AsyncTurnBarrier,
        agent_id: int,
    ) -> None:
        rng = random.Random(self.args.seed + agent_id * 1_000_003)
        session_id = f"scale-agent-{agent_id}"
        initial_words = bounded_lognormal(
            rng,
            self.args.initial_words_median,
            self.args.initial_words_sigma,
            self.args.initial_words_max,
        )
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": synthetic_text(
                    rng, agent_id=agent_id, turn=1, word_count=initial_words
                ),
            }
        ]

        for invocation in range(1, self.args.turns + 1):
            output_tokens = bounded_lognormal(
                rng,
                self.args.output_tokens_median,
                self.args.output_tokens_sigma,
                self.args.output_tokens_max,
            )
            result = await self._request_turn(
                client,
                agent_id=agent_id,
                session_id=session_id,
                invocation=invocation,
                messages=messages,
                output_tokens=output_tokens,
            )
            messages.append({"role": "assistant", "content": result.assistant_text})

            if invocation == self.args.scale_after_turn:
                await barrier.wait()
            if invocation == self.args.turns:
                continue

            delay_ms = bounded_lognormal(
                rng,
                self.args.inter_turn_delay_median_ms,
                self.args.inter_turn_delay_sigma,
                self.args.inter_turn_delay_max_ms,
            )
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
            followup_words = bounded_lognormal(
                rng,
                self.args.followup_words_median,
                self.args.followup_words_sigma,
                self.args.followup_words_max,
            )
            messages.append(
                {
                    "role": "user",
                    "content": synthetic_text(
                        rng,
                        agent_id=agent_id,
                        turn=invocation + 1,
                        word_count=followup_words,
                    ),
                }
            )

    async def _request_turn(
        self,
        client: aiohttp.ClientSession,
        *,
        agent_id: int,
        session_id: str,
        invocation: int,
        messages: Sequence[dict[str, str]],
        output_tokens: int,
    ) -> ChatResult:
        url = f"http://127.0.0.1:{self.case.ports.frontend}/v1/chat/completions"
        payload = {
            "model": self.args.served_model_name,
            "messages": list(messages),
            "max_tokens": output_tokens,
            "temperature": self.args.temperature,
            "stream": False,
        }
        max_retries = (
            self.args.scale_down_max_retries
            if self.case.scenario == "scale-down"
            else 0
        )
        logical_started = time.monotonic()
        for attempt in range(max_retries + 1):
            attempt_started = time.monotonic()
            try:
                result = await self._post_chat(
                    client, url=url, session_id=session_id, payload=payload
                )
            except Exception as error:
                if attempt < max_retries:
                    append_jsonl(
                        self.events_path,
                        {
                            "timestamp": utc_now(),
                            "event": "request_retry",
                            "agent_id": agent_id,
                            "session_id": session_id,
                            "invocation": invocation,
                            "attempt": attempt + 1,
                            "latency_ms": (time.monotonic() - attempt_started) * 1000,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                    await asyncio.sleep(self.args.scale_down_retry_delay_ms / 1000)
                    continue
                append_jsonl(
                    self.events_path,
                    {
                        "timestamp": utc_now(),
                        "event": "request_complete",
                        "status": "failure",
                        "agent_id": agent_id,
                        "session_id": session_id,
                        "invocation": invocation,
                        "attempts": attempt + 1,
                        "latency_ms": (time.monotonic() - logical_started) * 1000,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise

            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "event": "request_complete",
                    "status": "success",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "invocation": invocation,
                    "attempts": attempt + 1,
                    "latency_ms": (time.monotonic() - logical_started) * 1000,
                    "prompt_tokens": result.prompt_tokens,
                    "cached_tokens": result.cached_tokens,
                    "completion_tokens": result.completion_tokens,
                    "requested_output_tokens": output_tokens,
                },
            )
            return result
        raise AssertionError("request retry loop did not return or raise")

    @staticmethod
    async def _post_chat(
        client: aiohttp.ClientSession,
        *,
        url: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> ChatResult:
        async with client.post(
            url,
            headers={"x-dynamo-session-id": session_id},
            json=payload,
        ) as response:
            body = await response.text()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"HTTP {response.status}: {body[:1000].replace(chr(10), ' ')}"
                )
        try:
            decoded = json.loads(body)
            message = decoded["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"invalid OpenAI chat response: {body[:1000]}"
            ) from error

        content = message.get("content")
        reasoning = message.get("reasoning_content")
        assistant_parts = [value for value in (reasoning, content) if value]
        assistant_text = "\n".join(
            value if isinstance(value, str) else json.dumps(value)
            for value in assistant_parts
        )
        usage = decoded.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return ChatResult(
            assistant_text=assistant_text or "No content returned.",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=int(
                prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0
            ),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def record_timeline(path: Path, event: str, **fields: Any) -> None:
    append_jsonl(path, {"timestamp": utc_now(), "event": event, **fields})


def discovery_removals(
    frontend_log: Path, *, after_line: int = 0
) -> list[dict[str, Any]]:
    removals: list[dict[str, Any]] = []
    for record in read_jsonl(frontend_log):
        line = int(record.get("_line", 0))
        if line <= after_line:
            continue
        message = log_message(record)
        match = re.search(r"removed=\[([^]]*)\]", message)
        if match is not None:
            worker_ids = [
                int(value, 16)
                for value in re.findall(r'"([0-9a-fA-F]+)"', match.group(1))
            ]
        else:
            # KvWorkerMonitor logs this once its runtime-config watch has
            # pruned the worker. This is the concrete frontend convergence
            # point after which stale affinity must no longer be reused.
            match = re.fullmatch(
                r"Removed Prometheus metrics for worker (\d+)", message
            )
            worker_ids = [int(match.group(1))] if match is not None else []
        if worker_ids:
            removals.append(
                {
                    "worker_ids": worker_ids,
                    "frontend_log_line": line,
                    "timestamp": nested_field(record, "time")
                    or nested_field(record, "timestamp"),
                    "message": message,
                }
            )
    return removals


def wait_for_discovery_removal(
    frontend_log: Path,
    *,
    after_line: int,
    timeout: float,
    processes: Sequence[ManagedCommand],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            process.require_running()
        removals = discovery_removals(frontend_log, after_line=after_line)
        if removals:
            return removals[-1]
        time.sleep(0.1)
    raise TimeoutError(
        f"frontend did not report a worker removal within {timeout}s; "
        f"see {frontend_log}"
    )


def run_workload(
    args: argparse.Namespace,
    case: CaseConfig,
    worker_a: ManagedCommand,
    worker_b: ManagedCommand,
    frontend: ManagedCommand,
    timeline_path: Path,
) -> None:
    workload = AgentWorkload(args, case)
    workload.start()
    record_timeline(
        timeline_path,
        "workload_started",
        agents=args.agents,
        turns=args.turns,
    )
    deadline = time.monotonic() + args.workload_timeout
    worker_b_started = case.scenario in ("ab-start", "scale-down")
    worker_b_stopped = False
    trigger_handled = False

    try:
        while True:
            if workload.trigger_reached.is_set() and not trigger_handled:
                record_timeline(
                    timeline_path,
                    "scale_trigger_reached",
                    trigger_turn=args.scale_after_turn,
                    completed_agents=args.agents,
                )
                if case.scenario == "scale-up":
                    worker_b.start()
                    worker_b_started = True
                    record_timeline(
                        timeline_path,
                        "worker_b_launched",
                        pid=worker_b.process.pid,
                    )
                elif case.scenario == "scale-down":
                    record_timeline(
                        timeline_path,
                        "scale_down_trigger_reached",
                        trigger_turn=args.scale_after_turn,
                        completed_agents=args.agents,
                    )
                    last_log_line = max(
                        (
                            int(record.get("_line", 0))
                            for record in read_jsonl(frontend.log_path)
                        ),
                        default=0,
                    )
                    record_timeline(timeline_path, "worker_b_stop_requested")
                    worker_b.stop(timeout=args.scale_down_stop_timeout)
                    worker_b_stopped = True
                    record_timeline(timeline_path, "worker_b_stopped")
                    removal = wait_for_discovery_removal(
                        frontend.log_path,
                        after_line=last_log_line,
                        timeout=args.scale_down_discovery_timeout,
                        processes=(worker_a, frontend),
                    )
                    record_timeline(
                        timeline_path,
                        "worker_b_discovery_removed",
                        **removal,
                    )
                trigger_handled = True

            return_code = workload.poll()
            if return_code is not None:
                if return_code != 0:
                    detail = f": {workload.error}" if workload.error else ""
                    raise RuntimeError(
                        f"agent workload exited with status {return_code}{detail}; "
                        f"see {workload.log_path}"
                    )
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"agent workload did not complete within {args.workload_timeout}s"
                )
            worker_a.require_running()
            frontend.require_running()
            if worker_b_started and not worker_b_stopped:
                worker_b.require_running()
            time.sleep(0.1)
    except Exception:
        workload.stop()
        raise

    if case.scenario == "scale-up" and not worker_b_started:
        raise RuntimeError(
            "workload ended before every agent completed the scale trigger turn"
        )
    if case.scenario == "scale-down" and not worker_b_stopped:
        raise RuntimeError(
            "workload ended before every agent completed the scale-down trigger turn"
        )
    record_timeline(timeline_path, "workload_completed")


def nested_field(record: dict[str, Any], name: str) -> Any:
    if name in record:
        return record[name]
    for container_name in ("fields", "event", "data"):
        container = record.get(container_name)
        if isinstance(container, dict) and name in container:
            return container[name]
    return None


def log_message(record: dict[str, Any]) -> str:
    value = nested_field(record, "message")
    return value if isinstance(value, str) else ""


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "None":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [
            number for item in value if (number := parse_optional_int(item)) is not None
        ]
    return [int(item) for item in re.findall(r"\d+", str(value))]


def parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def target_key(route: dict[str, Any]) -> tuple[int, int | None]:
    return int(route["worker_id"]), route.get("dp_rank")


def load_routes(
    frontend_log: Path, routing_path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    routes: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    scale_events: list[dict[str, Any]] = []
    route_message = ROUTE_MESSAGES[routing_path]
    for record in read_jsonl(frontend_log):
        message = log_message(record)
        if message == route_message:
            session_id = nested_field(record, "session_id")
            worker_id = parse_optional_int(nested_field(record, "worker_id"))
            if not session_id or worker_id is None:
                continue
            routes.append(
                {
                    "session_id": str(session_id),
                    "worker_id": worker_id,
                    "dp_rank": parse_optional_int(nested_field(record, "dp_rank")),
                    "affinity_action": str(
                        nested_field(record, "affinity_action") or "none"
                    ),
                    "timestamp": nested_field(record, "time")
                    or nested_field(record, "timestamp"),
                    "_line": record.get("_line", 0),
                }
            )
        elif message == "Committed affinity scale-up migration":
            migrations.append(
                {
                    "session_id": str(nested_field(record, "session_id") or ""),
                    "old_worker_id": parse_optional_int(
                        nested_field(record, "old_worker_id")
                    ),
                    "old_dp_rank": parse_optional_int(
                        nested_field(record, "old_dp_rank")
                    ),
                    "new_worker_id": parse_optional_int(
                        nested_field(record, "new_worker_id")
                    ),
                    "new_dp_rank": parse_optional_int(
                        nested_field(record, "new_dp_rank")
                    ),
                    "timestamp": nested_field(record, "time")
                    or nested_field(record, "timestamp"),
                }
            )
        elif message == "Detected affinity scale-up":
            scale_events.append(
                {
                    "generation": parse_optional_int(
                        nested_field(record, "generation")
                    ),
                    "added_workers": parse_int_list(
                        nested_field(record, "added_workers")
                    ),
                    "added_capacity": parse_optional_int(
                        nested_field(record, "added_capacity")
                    ),
                    "total_capacity": parse_optional_int(
                        nested_field(record, "total_capacity")
                    ),
                    "timestamp": nested_field(record, "time")
                    or nested_field(record, "timestamp"),
                }
            )
    return routes, migrations, scale_events


def selected_for_scale_up(
    scope: str,
    session_id: str,
    generation: int,
    added_capacity: int,
    total_capacity: int,
) -> bool:
    try:
        import xxhash  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "exact cohort validation requires the Python 'xxhash' package; "
            "install it with 'uv pip install xxhash'"
        ) from error
    payload = bytearray()
    scope_bytes = scope.encode()
    session_bytes = session_id.encode()
    payload.extend(len(scope_bytes).to_bytes(8, "little"))
    payload.extend(scope_bytes)
    payload.extend(len(session_bytes).to_bytes(8, "little"))
    payload.extend(session_bytes)
    payload.extend(generation.to_bytes(8, "little"))
    value = xxhash.xxh3_64_intdigest(payload)
    cutoff = ((1 << 64) * added_capacity) // total_capacity
    return value < cutoff


def latency_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(event["latency_ms"])
        for event in events
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def analyze_scale_down_routes(
    args: argparse.Namespace,
    routes_by_session: dict[str, list[dict[str, Any]]],
    timeline: Sequence[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    removal_record = next(
        (
            record
            for record in timeline
            if record.get("event") == "worker_b_discovery_removed"
        ),
        None,
    )
    if removal_record is None:
        errors.append("frontend discovery removal was not recorded")
        return {}

    removed_workers = [
        worker_id
        for value in removal_record.get("worker_ids", [])
        if (worker_id := parse_optional_int(value)) is not None
    ]
    removal_line = parse_optional_int(removal_record.get("frontend_log_line"))
    if len(removed_workers) != 1 or removal_line is None:
        errors.append(f"invalid discovery removal record: {removal_record}")
        return {}
    removed_worker = removed_workers[0]

    initial_targets: dict[str, tuple[int, int | None]] = {}
    for session, session_routes in routes_by_session.items():
        initial = session_routes[: args.scale_after_turn]
        targets = {target_key(route) for route in initial}
        if len(initial) != args.scale_after_turn or len(targets) != 1:
            errors.append(
                f"{session} did not have one stable target for all pre-removal turns"
            )
            continue
        initial_targets[session] = next(iter(targets))

    initial_workers = {target[0] for target in initial_targets.values()}
    if len(initial_workers) != 2:
        errors.append(
            "scale-down requires sessions on both initial workers; "
            f"observed workers {sorted(initial_workers)}"
        )
    if removed_worker not in initial_workers:
        errors.append(
            f"removed worker {removed_worker} had no pre-removal session affinity"
        )

    removed_sessions = {
        session
        for session, target in initial_targets.items()
        if target[0] == removed_worker
    }
    rebound_sessions: set[str] = set()
    post_removal_routes = 0
    for session, initial_target in initial_targets.items():
        session_routes = routes_by_session[session]
        if len(session_routes) < args.turns:
            errors.append(
                f"{session} has {len(session_routes)} routing decisions, expected at least "
                f"{args.turns}"
            )
        if len(session_routes) > args.turns + args.scale_down_max_retries:
            errors.append(
                f"{session} used more routing retries than configured: "
                f"{len(session_routes) - args.turns}"
            )

        routes_after_discovery = [
            route
            for route in session_routes
            if int(route.get("_line", 0)) > removal_line
        ]
        post_removal_routes += len(routes_after_discovery)
        if not routes_after_discovery:
            errors.append(f"{session} had no request after discovery removed worker B")
        if any(
            target_key(route)[0] == removed_worker for route in routes_after_discovery
        ):
            errors.append(
                f"{session} routed to removed worker after discovery converged"
            )

        if initial_target[0] != removed_worker:
            if any(target_key(route) != initial_target for route in session_routes):
                errors.append(
                    f"{session} was bound to the surviving worker but changed target"
                )
            continue

        later_routes = session_routes[args.scale_after_turn :]
        first_survivor = next(
            (
                index
                for index, route in enumerate(later_routes)
                if target_key(route)[0] != removed_worker
            ),
            None,
        )
        if first_survivor is None:
            errors.append(f"{session} never rebound from the removed worker")
            continue
        replacement = target_key(later_routes[first_survivor])
        if any(
            target_key(route) != replacement for route in later_routes[first_survivor:]
        ):
            errors.append(f"{session} did not stay sticky after scale-down rebinding")
            continue
        rebound_sessions.add(session)

    if rebound_sessions != removed_sessions:
        errors.append(
            "scale-down rebind cohort mismatch: "
            f"missing={sorted(removed_sessions - rebound_sessions)}, "
            f"unexpected={sorted(rebound_sessions - removed_sessions)}"
        )

    return {
        "removed_worker_id": removed_worker,
        "discovery_removal_log_line": removal_line,
        "expected_rebind_count": len(removed_sessions),
        "observed_rebind_count": len(rebound_sessions),
        "expected_rebind_sessions": sorted(removed_sessions),
        "observed_rebind_sessions": sorted(rebound_sessions),
        "post_removal_routing_decisions": post_removal_routes,
    }


def analyze_case(args: argparse.Namespace, case: CaseConfig) -> dict[str, Any]:
    errors: list[str] = []
    events = read_jsonl(case.output_dir / "agent-events.jsonl")
    completions = [
        event for event in events if event.get("event") == "request_complete"
    ]
    successes = [event for event in completions if event.get("status") == "success"]
    failures = [event for event in completions if event.get("status") != "success"]
    expected_requests = args.agents * args.turns
    if len(successes) != expected_requests:
        errors.append(
            f"expected {expected_requests} successful requests, observed {len(successes)}"
        )
    if failures:
        errors.append(f"observed {len(failures)} failed requests")

    expected_invocations = list(range(1, args.turns + 1))
    for agent_id in range(args.agents):
        invocations = sorted(
            int(event["invocation"])
            for event in successes
            if event.get("agent_id") == agent_id
        )
        if invocations != expected_invocations:
            errors.append(
                f"agent {agent_id} completed invocations {invocations}, expected "
                f"{expected_invocations}"
            )

    routes, migrations, scale_events = load_routes(
        case.output_dir / "frontend.log", case.routing_path
    )
    expected_sessions = {f"scale-agent-{agent_id}" for agent_id in range(args.agents)}
    routes_by_session: dict[str, list[dict[str, Any]]] = {
        session: [] for session in expected_sessions
    }
    for route in routes:
        if route["session_id"] in routes_by_session:
            routes_by_session[route["session_id"]].append(route)
    for session, session_routes in routes_by_session.items():
        session_routes.sort(
            key=lambda route: (
                parse_timestamp(route.get("timestamp")) or 0.0,
                int(route.get("_line", 0)),
            )
        )
        if case.scenario != "scale-down" and len(session_routes) != args.turns:
            errors.append(
                f"{session} has {len(session_routes)} routing decisions, expected {args.turns}"
            )

    observed_targets = {target_key(route) for route in routes}
    if case.scenario == "a-only":
        workers = {worker for worker, _ in observed_targets}
        if len(workers) != 1:
            errors.append(f"A-only case used {len(workers)} workers: {sorted(workers)}")
    elif case.scenario in ("ab-start", "scale-down"):
        workers = {worker for worker, _ in observed_targets}
        if len(workers) != 2:
            errors.append(f"A+B case used {len(workers)} workers: {sorted(workers)}")

    expected_migrations: set[str] = set()
    observed_migrations = {
        migration["session_id"]
        for migration in migrations
        if migration.get("session_id") in expected_sessions
    }
    scale_event: dict[str, Any] | None = None
    if case.scenario == "scale-up":
        if len(scale_events) != 1:
            errors.append(
                f"expected exactly one detected scale-up event, observed {len(scale_events)}"
            )
        if scale_events:
            scale_event = scale_events[-1]
            generation = scale_event.get("generation")
            added_capacity = scale_event.get("added_capacity")
            total_capacity = scale_event.get("total_capacity")
            if not all(
                isinstance(value, int) and value > 0
                for value in (generation, added_capacity, total_capacity)
            ):
                errors.append(
                    f"scale-up event has incomplete capacity data: {scale_event}"
                )
            else:
                expected_migrations = {
                    session
                    for session in expected_sessions
                    if selected_for_scale_up(
                        args.served_model_name,
                        session,
                        int(generation),
                        int(added_capacity),
                        int(total_capacity),
                    )
                }
                if observed_migrations != expected_migrations:
                    missing = sorted(expected_migrations - observed_migrations)
                    unexpected = sorted(observed_migrations - expected_migrations)
                    errors.append(
                        "migration cohort mismatch: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        if len(migrations) != len(observed_migrations):
            errors.append("one or more sessions committed migration more than once")
    elif migrations:
        errors.append(
            f"baseline case unexpectedly committed {len(migrations)} migrations"
        )

    if case.scenario != "scale-down":
        for session, session_routes in routes_by_session.items():
            if not session_routes:
                continue
            migration_indices = [
                index
                for index, route in enumerate(session_routes)
                if route.get("affinity_action") == "migrate"
            ]
            if session in expected_migrations:
                if len(migration_indices) != 1:
                    errors.append(
                        f"{session} has {len(migration_indices)} migrate decisions, "
                        "expected one"
                    )
                    continue
                migration_index = migration_indices[0]
                before = session_routes[:migration_index]
                after = session_routes[migration_index:]
                if before and len({target_key(route) for route in before}) != 1:
                    errors.append(f"{session} changed target before its migration")
                if len({target_key(route) for route in after}) != 1:
                    errors.append(f"{session} did not remain sticky after migration")
                if before and target_key(before[0]) == target_key(after[0]):
                    errors.append(
                        f"{session} migration did not change its exact target"
                    )
                if (
                    scale_event
                    and scale_event.get("added_workers")
                    and target_key(after[0])[0] not in scale_event["added_workers"]
                ):
                    errors.append(
                        f"{session} migrated to a worker that was not newly added"
                    )
            else:
                if migration_indices:
                    errors.append(f"{session} unexpectedly migrated")
                if len({target_key(route) for route in session_routes}) != 1:
                    errors.append(
                        f"{session} did not retain exact worker/rank affinity"
                    )

    before_events = [
        event
        for event in successes
        if isinstance(event.get("invocation"), int)
        and int(event["invocation"]) <= args.scale_after_turn
    ]
    after_events = [
        event
        for event in successes
        if isinstance(event.get("invocation"), int)
        and int(event["invocation"]) > args.scale_after_turn
    ]
    prompt_tokens = sum(int(event.get("prompt_tokens") or 0) for event in successes)
    cached_tokens = sum(int(event.get("cached_tokens") or 0) for event in successes)
    target_counts: dict[str, int] = {}
    for route in routes:
        key = f"worker={route['worker_id']},rank={route.get('dp_rank')}"
        target_counts[key] = target_counts.get(key, 0) + 1

    timeline = read_jsonl(case.output_dir / "timeline.jsonl")
    scale_down = (
        analyze_scale_down_routes(args, routes_by_session, timeline, errors)
        if case.scenario == "scale-down"
        else {}
    )
    if case.scenario == "scale-down" and scale_events:
        errors.append(
            f"worker removal unexpectedly produced {len(scale_events)} scale-up events"
        )
    launch_record = next(
        (record for record in timeline if record.get("event") == "worker_b_launched"),
        None,
    )
    launch_time = (
        parse_timestamp(launch_record.get("timestamp")) if launch_record else None
    )
    scale_time = (
        parse_timestamp(scale_event.get("timestamp"))
        if scale_event is not None
        else None
    )
    commit_times = [
        value
        for migration in migrations
        if (value := parse_timestamp(migration.get("timestamp"))) is not None
    ]

    analysis = {
        "passed": not errors,
        "errors": errors,
        "routing_path": case.routing_path,
        "scenario": case.scenario,
        "namespace": case.namespace,
        "expected_requests": expected_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "routing_decisions": len(routes),
        "target_counts": dict(sorted(target_counts.items())),
        "latency_before_trigger": latency_summary(before_events),
        "latency_after_trigger": latency_summary(after_events),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "reported_cache_fraction": (
            cached_tokens / prompt_tokens if prompt_tokens else None
        ),
        "scale_event": scale_event,
        "expected_migration_count": len(expected_migrations),
        "observed_migration_count": len(observed_migrations),
        "expected_migrations": sorted(expected_migrations),
        "observed_migrations": sorted(observed_migrations),
        "worker_b_registration_seconds": (
            scale_time - launch_time
            if scale_time is not None and launch_time is not None
            else None
        ),
        "first_migration_seconds_after_launch": (
            min(commit_times) - launch_time
            if commit_times and launch_time is not None
            else None
        ),
        "last_migration_seconds_after_launch": (
            max(commit_times) - launch_time
            if commit_times and launch_time is not None
            else None
        ),
        **scale_down,
    }
    write_json(case.output_dir / "analysis.json", analysis)
    return analysis


def preflight_gpu_memory(gpu_ids: Sequence[int], maximum_mib: int) -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={','.join(str(gpu) for gpu in gpu_ids)}",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    busy: list[str] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and int(float(fields[1])) > maximum_mib:
            busy.append(f"GPU {fields[0]} uses {fields[1]} MiB")
    if busy:
        raise RuntimeError("selected GPUs are not free: " + "; ".join(busy))


def check_runtime_service(endpoint: str, name: str) -> None:
    value = endpoint.removeprefix("http://").removeprefix("https://")
    value = value.removeprefix("nats://")
    host_port = value.split("/", 1)[0]
    host, port_text = host_port.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port_text)), timeout=3):
            return
    except OSError as error:
        raise RuntimeError(
            f"cannot connect to {name} at {endpoint}: {error}"
        ) from error


def wait_for_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} did not become ready")


def run_case(
    args: argparse.Namespace, routing_path: str, scenario: str
) -> dict[str, Any]:
    case_name = f"{routing_path}__{scenario}"
    case_dir = args.output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    case = CaseConfig(
        routing_path=routing_path,
        scenario=scenario,
        output_dir=case_dir,
        namespace=f"scale-bench-{uuid.uuid4().hex[:12]}",
        ports=allocate_case_ports(),
    )
    timeline_path = case_dir / "timeline.jsonl"
    worker_a = worker_command(args, case, "a")
    worker_b = worker_command(args, case, "b")
    frontend = frontend_command(args, case)
    nats = nats_command(args, case)
    processes = [
        process
        for process in (nats, worker_a, worker_b, frontend)
        if process is not None
    ]
    sampler = Sampler(
        case_dir / "samples.jsonl",
        {
            "frontend": f"http://127.0.0.1:{case.ports.frontend}/metrics",
            "worker-a": f"http://127.0.0.1:{case.ports.system_a}/metrics",
            "worker-b": f"http://127.0.0.1:{case.ports.system_b}/metrics",
            "engine-a": f"http://127.0.0.1:{case.ports.engine_a}/metrics",
            "engine-b": f"http://127.0.0.1:{case.ports.engine_b}/metrics",
        },
        [*args.gpu_a, *args.gpu_b],
        args.sample_interval,
    )
    manifest = {
        "started_at": utc_now(),
        "git": git_state(args.repo),
        "routing_path": routing_path,
        "scenario": scenario,
        "namespace": case.namespace,
        "ports": vars(case.ports),
        "commands": {process.name: process.command for process in processes},
        "workload": {
            "kind": "in-process OpenAI chat workload",
            "endpoint": f"http://127.0.0.1:{case.ports.frontend}/v1/chat/completions",
            "session_header": "x-dynamo-session-id",
            "session_template": "scale-agent-{agent_id}",
        },
        "configuration": serializable_args(args),
    }
    write_json(case_dir / "manifest.json", manifest)

    run_error: str | None = None
    try:
        record_timeline(timeline_path, "case_started")
        if nats is not None:
            nats.start()
            wait_for_port(case.ports.nats, 10)
            record_timeline(timeline_path, "nats_started", pid=nats.process.pid)
        worker_a.start()
        record_timeline(timeline_path, "worker_a_launched", pid=worker_a.process.pid)
        if scenario in ("ab-start", "scale-down"):
            worker_b.start()
            record_timeline(
                timeline_path, "worker_b_launched", pid=worker_b.process.pid
            )
        frontend.start()
        record_timeline(timeline_path, "frontend_launched", pid=frontend.process.pid)
        sampler.start()
        startup_processes = [worker_a, frontend]
        if scenario in ("ab-start", "scale-down"):
            startup_processes.append(worker_b)
        wait_for_model(
            case.ports.frontend,
            args.served_model_name,
            args.startup_timeout,
            startup_processes,
        )
        worker_a.require_running()
        if scenario in ("ab-start", "scale-down"):
            worker_b.require_running()
        frontend.require_running()
        record_timeline(timeline_path, "frontend_model_ready")
        # Let the runtime-config watch establish its startup baseline before
        # any affinity entries are created.
        time.sleep(args.baseline_settle_seconds)
        run_workload(
            args,
            case,
            worker_a,
            worker_b,
            frontend,
            timeline_path,
        )
    # Preserve artifacts and continue the matrix after any case-level failure.
    except Exception as error:  # noqa: BLE001
        run_error = f"{type(error).__name__}: {error}"
        record_timeline(timeline_path, "case_failed", error=run_error)
    finally:
        sampler.stop()
        for process in reversed(processes):
            process.stop()
        record_timeline(timeline_path, "case_stopped")

    if run_error is not None:
        analysis = {
            "passed": False,
            "routing_path": routing_path,
            "scenario": scenario,
            "errors": [run_error],
        }
        write_json(case_dir / "analysis.json", analysis)
        return analysis
    try:
        return analyze_case(args, case)
    except Exception as error:  # noqa: BLE001 - record analysis failures too
        analysis = {
            "passed": False,
            "routing_path": routing_path,
            "scenario": scenario,
            "errors": [f"analysis failed: {type(error).__name__}: {error}"],
        }
        write_json(case_dir / "analysis.json", analysis)
        return analysis


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            result[name] = str(value)
        else:
            result[name] = value
    return result


def parse_gpu_list(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU list must contain comma-separated integers"
        ) from error
    if not result or len(set(result)) != len(result) or any(gpu < 0 for gpu in result):
        raise argparse.ArgumentTypeError(
            "GPU lists must contain one or more distinct non-negative IDs"
        )
    return result


def expand_selection(value: str, choices: Sequence[str]) -> list[str]:
    return list(choices) if value == "all" else [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routing-path", choices=[*ROUTING_PATHS, "all"], default="all"
    )
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--cuda-library-dir",
        type=Path,
        help="Directory containing the CUDA runtime libraries needed by SGLang",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-a", type=parse_gpu_list, default=parse_gpu_list("0,1"))
    parser.add_argument("--gpu-b", type=parse_gpu_list, default=parse_gpu_list("2,3"))
    parser.add_argument(
        "--dp-size",
        type=int,
        choices=(1, 2),
        default=2,
        help="DP-attention ranks per worker; 1 runs a plain non-DP worker",
    )
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--scale-after-turn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-words-median", type=float, default=256)
    parser.add_argument("--initial-words-sigma", type=float, default=0.45)
    parser.add_argument("--initial-words-max", type=int, default=512)
    parser.add_argument("--output-tokens-median", type=float, default=48)
    parser.add_argument("--output-tokens-sigma", type=float, default=0.35)
    parser.add_argument("--output-tokens-max", type=int, default=96)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for every OpenAI request",
    )
    parser.add_argument("--followup-words-median", type=float, default=96)
    parser.add_argument("--followup-words-sigma", type=float, default=0.45)
    parser.add_argument("--followup-words-max", type=int, default=192)
    parser.add_argument("--inter-turn-delay-median-ms", type=float, default=400)
    parser.add_argument("--inter-turn-delay-sigma", type=float, default=0.5)
    parser.add_argument("--inter-turn-delay-max-ms", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument(
        "--mem-fraction-static-a",
        type=float,
        default=None,
        help="Optional worker-A override used for unequal-capacity tests",
    )
    parser.add_argument(
        "--mem-fraction-static-b",
        type=float,
        default=None,
        help="Optional worker-B override used for unequal-capacity tests",
    )
    parser.add_argument("--affinity-ttl", type=int, default=3600)
    parser.add_argument("--load-report-interval", type=float, default=0.25)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--baseline-settle-seconds", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--workload-timeout", type=float, default=7200)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--scale-down-max-retries", type=int, default=1)
    parser.add_argument("--scale-down-retry-delay-ms", type=int, default=1000)
    parser.add_argument("--scale-down-stop-timeout", type=float, default=30.0)
    parser.add_argument("--scale-down-discovery-timeout", type=float, default=30.0)
    parser.add_argument(
        "--nats-server",
        default=None,
        help="Use an existing NATS URL instead of launching a case-local server",
    )
    parser.add_argument("--nats-server-bin", default="nats-server")
    parser.add_argument("--etcd-endpoints", default="http://127.0.0.1:2379")
    parser.add_argument("--max-initial-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.agents <= 0 or args.turns <= 0:
        raise ValueError("agents and turns must be positive")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if not 1 <= args.scale_after_turn < args.turns:
        raise ValueError("scale-after-turn must be at least 1 and less than turns")
    for name in (
        "initial_words_median",
        "output_tokens_median",
        "followup_words_median",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "initial_words_sigma",
        "output_tokens_sigma",
        "followup_words_sigma",
        "inter_turn_delay_sigma",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "initial_words_max",
        "output_tokens_max",
        "followup_words_max",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(args.inter_turn_delay_median_ms)
        or args.inter_turn_delay_median_ms < 0
        or args.inter_turn_delay_max_ms < 0
    ):
        raise ValueError("inter-turn delay values cannot be negative")
    if set(args.gpu_a) & set(args.gpu_b):
        raise ValueError("worker A and worker B GPU lists must not overlap")
    if len(args.gpu_a) != args.dp_size or len(args.gpu_b) != args.dp_size:
        raise ValueError(
            f"each worker needs exactly {args.dp_size} GPU(s) for --dp-size "
            f"{args.dp_size}"
        )
    for name in (
        "mem_fraction_static",
        "mem_fraction_static_a",
        "mem_fraction_static_b",
    ):
        value = getattr(args, name)
        if value is not None and not 0 < value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.scale_down_max_retries < 0:
        raise ValueError("--scale-down-max-retries cannot be negative")
    if args.scale_down_retry_delay_ms < 0:
        raise ValueError("--scale-down-retry-delay-ms cannot be negative")
    if args.scale_down_stop_timeout <= 0 or args.scale_down_discovery_timeout <= 0:
        raise ValueError("scale-down stop and discovery timeouts must be positive")
    args.repo = args.repo.resolve()
    if not (args.repo / "pyproject.toml").is_file():
        raise ValueError(f"{args.repo} does not look like a Dynamo checkout")
    if args.cuda_library_dir is None:
        completed = subprocess.run(
            [
                args.python,
                "-c",
                ("import json,site; print(json.dumps(site.getsitepackages()))"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        site_packages = json.loads(completed.stdout)
        candidates = [
            Path(site_path) / "nvidia" / "cu13" / "lib" for site_path in site_packages
        ]
        args.cuda_library_dir = next(
            (path for path in candidates if (path / "libnvrtc.so.13").exists()),
            None,
        )
        if args.cuda_library_dir is None:
            raise ValueError("could not locate libnvrtc.so.13; pass --cuda-library-dir")
    args.cuda_library_dir = args.cuda_library_dir.resolve()
    if not args.cuda_library_dir.is_dir():
        raise ValueError(f"CUDA library directory not found: {args.cuda_library_dir}")
    if args.nats_server is None:
        resolved_nats = shutil.which(args.nats_server_bin)
        if resolved_nats is None:
            raise ValueError(f"NATS executable not found: {args.nats_server_bin}")
        args.nats_server_bin = resolved_nats
    else:
        check_runtime_service(args.nats_server, "NATS")
    check_runtime_service(args.etcd_endpoints, "etcd")


def write_matrix_csv(path: Path, analyses: Iterable[dict[str, Any]]) -> None:
    rows = list(analyses)
    fields = [
        "routing_path",
        "scenario",
        "passed",
        "successful_requests",
        "failed_requests",
        "routing_decisions",
        "expected_migration_count",
        "observed_migration_count",
        "reported_cache_fraction",
        "worker_b_registration_seconds",
        "first_migration_seconds_after_launch",
        "last_migration_seconds_after_launch",
        "removed_worker_id",
        "expected_rebind_count",
        "observed_rebind_count",
        "post_removal_routing_decisions",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for analysis in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(analysis.get(field))
                        if isinstance(analysis.get(field), (list, dict))
                        else analysis.get(field)
                    )
                    for field in fields
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except Exception as error:  # noqa: BLE001 - turn preflight into a concise CLI error
        print(f"preflight failed: {error}", file=sys.stderr)
        return 2

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path.home() / "benchmark-results" / "dynamo-scale-up" / stamp
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = serializable_args(args)
    configuration["git"] = git_state(args.repo)
    write_json(args.output_dir / "configuration.json", configuration)

    paths = expand_selection(args.routing_path, ROUTING_PATHS)
    scenarios = expand_selection(args.scenario, SCENARIOS)
    analyses: list[dict[str, Any]] = []
    for routing_path in paths:
        for scenario in scenarios:
            print(f"[{utc_now()}] starting {routing_path}/{scenario}", flush=True)
            if not args.skip_gpu_preflight:
                try:
                    preflight_gpu_memory(
                        [*args.gpu_a, *args.gpu_b],
                        args.max_initial_gpu_memory_mib,
                    )
                except Exception as error:  # noqa: BLE001 - preserve matrix summary
                    analysis = {
                        "passed": False,
                        "routing_path": routing_path,
                        "scenario": scenario,
                        "errors": [f"GPU preflight failed: {error}"],
                    }
                    analyses.append(analysis)
                    print(f"[{utc_now()}] failed {routing_path}/{scenario}: {error}")
                    continue
            analysis = run_case(args, routing_path, scenario)
            analyses.append(analysis)
            status = "passed" if analysis.get("passed") else "failed"
            print(f"[{utc_now()}] {status} {routing_path}/{scenario}", flush=True)

    write_json(args.output_dir / "matrix-analysis.json", analyses)
    write_matrix_csv(args.output_dir / "matrix-summary.csv", analyses)
    print(f"artifacts: {args.output_dir}")
    return 0 if analyses and all(item.get("passed") for item in analyses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
