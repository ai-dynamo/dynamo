# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned local serving contexts; run this module in the Dynamo serving environment."""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from .lifecycle import termination_signal as _termination_signal
from .workloads import MODEL_REVISION


@dataclass(frozen=True)
class ServingConfig:
    model_path: Path
    model: str
    log_dir: Path
    mode: str = "mocker"
    workers: int = 1
    python: str = sys.executable
    approve_gpu_hours: float | None = None
    gpu: str = "0"
    startup_timeout: int = 600


@dataclass(frozen=True)
class ServingHandle:
    base_url: str
    model: str
    mode: str
    namespace: str
    worker_count: int
    reservation_started_unix_seconds: float | None
    budget_deadline_unix_seconds: float | None
    monitor_pids: tuple[int, ...]


def _dependencies():
    # These are repository lifecycle helpers, not dependencies of the load client.
    from tests.conftest import EtcdServer, NatsServer
    from tests.utils.http_checks import (
        check_health_ready,
        check_http_ok,
        model_registered,
    )
    from tests.utils.managed_process import ManagedProcess
    from tests.utils.port_utils import reserved_ports

    return SimpleNamespace(
        EtcdServer=EtcdServer,
        NatsServer=NatsServer,
        ManagedProcess=ManagedProcess,
        reserved_ports=reserved_ports,
        check_health_ready=check_health_ready,
        check_http_ok=check_http_ok,
        model_registered=model_registered,
    )


def _validate(config):
    if config.mode not in ("mocker", "sglang", "native"):
        raise ValueError("mode must be mocker, sglang, or native")
    if not 1 <= config.workers <= 8 or not 1 <= config.startup_timeout <= 600:
        raise ValueError("workers must be 1..8 and startup timeout 1..600 seconds")
    if not config.model.strip():
        raise ValueError("registered model must not be blank")
    if config.mode != "mocker":
        budget = config.approve_gpu_hours
        if budget is None or not math.isfinite(budget) or not 0 < budget <= 4:
            raise ValueError(
                "GPU launch requires explicit --approve-gpu-hours in (0, 4]"
            )
        if config.workers != 1 or not config.gpu.isdigit():
            raise ValueError(
                "the qualified profile permits exactly one GPU and one worker"
            )
        if version("sglang").split("+")[0] != "0.5.19":
            raise ValueError(
                "GPU serving requires the separately qualified SGLang 0.5.19 environment"
            )
    path = config.model_path.resolve(strict=True)
    if not path.is_dir() or path.name != MODEL_REVISION:
        raise ValueError(
            f"model path must be the local snapshot revision {MODEL_REVISION}"
        )


@contextmanager
def _gpu_deadline(hours):
    """Interrupt the owning main thread so ExitStack performs its normal cleanup."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("GPU serving must be owned by the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("GPU deadline cannot replace an existing process alarm")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("approved GPU reservation deadline reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, hours * 3600)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _gpu_args(config):
    return [
        "--model-path",
        str(config.model_path.resolve()),
        "--served-model-name",
        config.model,
        "--context-length",
        "2048",
        "--max-total-tokens",
        "2048",
        "--mem-fraction-static",
        "0.75",
        "--page-size",
        "16",
        "--tp",
        "1",
        "--pp-size",
        "1",
        "--disable-piecewise-cuda-graph",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "1",
        "--enable-metrics",
    ]


def _local_infra(process):
    # Existing test helpers otherwise listen on all interfaces without authentication.
    process.command = [
        part.replace("http://0.0.0.0:", "http://127.0.0.1:") for part in process.command
    ]
    return process


@contextmanager
def _owned_process(process):
    try:
        entered = process.__enter__()
    except (KeyboardInterrupt, SystemExit):
        process.__exit__(*sys.exc_info())
        raise
    try:
        yield entered
    finally:
        process.__exit__(*sys.exc_info())


@contextmanager
def serve(config: ServingConfig):
    _validate(config)
    config.log_dir.mkdir(parents=True, exist_ok=False)
    dependencies = _dependencies()
    namespace = f"decision-perf-{uuid.uuid4().hex}"
    started = time.time()
    deadline = (
        started + config.approve_gpu_hours * 3600 if config.mode != "mocker" else None
    )
    budget = (
        _gpu_deadline(config.approve_gpu_hours)
        if deadline is not None
        else nullcontext()
    )
    with ExitStack() as stack, budget:
        serving_processes = []
        ports = stack.enter_context(
            dependencies.reserved_ports(config.workers + 1, 20000)
        )
        base_url = f"http://127.0.0.1:{ports[0]}"
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_NAMESPACE": namespace,
            "DYN_SYSTEM_HOST": "127.0.0.1",
            "DYN_SGLANG_ENABLE_GENERATE": "1",
            "CUDA_VISIBLE_DEVICES": "" if config.mode == "mocker" else config.gpu,
        }
        env = {key: value for key, value in env.items() if key != "DYN_SYSTEM_PORT"}
        common = {
            "terminate_all_matching_process_names": False,
            "log_dir": str(config.log_dir.resolve()),
            "timeout": config.startup_timeout,
            "display_output": False,
        }
        if config.mode != "native":
            request = SimpleNamespace(
                node=SimpleNamespace(name=str(config.log_dir.resolve()))
            )
            etcd = _local_infra(dependencies.EtcdServer(request, port=0, timeout=60))
            nats = dependencies.NatsServer(
                request, port=0, timeout=60, disable_jetstream=True
            )
            nats.command = [*nats.command, "--addr", "127.0.0.1"]
            stack.enter_context(_owned_process(etcd))
            stack.enter_context(_owned_process(nats))
            env = {
                **env,
                "ETCD_ENDPOINTS": f"http://127.0.0.1:{etcd.port}",
                "NATS_SERVER": f"nats://127.0.0.1:{nats.port}",
            }
            frontend = dependencies.ManagedProcess(
                command=[
                    config.python,
                    "-m",
                    "dynamo.frontend",
                    "--http-host",
                    "127.0.0.1",
                    "--http-port",
                    str(ports[0]),
                    "--router-mode",
                    "round-robin",
                    "--enable-systemone-api",
                ],
                env=env,
                health_check_ports=[ports[0]],
                display_name="decision-frontend",
                **common,
            )
            stack.enter_context(_owned_process(frontend))
            serving_processes.append(frontend)
        for index in range(config.workers):
            worker_env = {
                **env,
                "DYN_SYSTEM_PORT": str(ports[index + 1]),
                "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": '["generate"]',
            }
            if config.mode == "mocker":
                command = [
                    config.python,
                    "-m",
                    "dynamo.mocker",
                    "--model-path",
                    str(config.model_path.resolve()),
                    "--model-name",
                    config.model,
                    "--engine-type",
                    "sglang",
                    "--sglang-generate",
                    "--speedup-ratio",
                    "100",
                ]
            elif config.mode == "sglang":
                command = [config.python, "-m", "dynamo.sglang", *_gpu_args(config)]
            else:
                command = [
                    config.python,
                    "-m",
                    "sglang.launch_server",
                    *_gpu_args(config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ports[0]),
                ]
            checks = (
                [(f"{base_url}/health", dependencies.check_http_ok)]
                if config.mode == "native"
                else [
                    (
                        f"{base_url}/v1/models",
                        partial(dependencies.model_registered, model=config.model),
                    ),
                    (
                        f"http://127.0.0.1:{ports[index + 1]}/health",
                        dependencies.check_health_ready,
                    ),
                ]
            )
            worker = stack.enter_context(
                _owned_process(
                    dependencies.ManagedProcess(
                        command=command,
                        env=worker_env,
                        health_check_urls=checks,
                        display_name=f"decision-{config.mode}-{index}",
                        **common,
                    )
                )
            )
            serving_processes.append(worker)
        handle = ServingHandle(
            base_url,
            config.model,
            config.mode,
            namespace,
            config.workers,
            started if deadline is not None else None,
            deadline,
            tuple(process.proc.pid for process in serving_processes),
        )
        (config.log_dir / "serving_commands.json").write_text(
            json.dumps(
                [
                    {
                        "pid": process.proc.pid,
                        "command": process.command,
                        "kind": config.mode,
                    }
                    for process in serving_processes
                ],
                indent=2,
            )
            + "\n"
        )
        (config.log_dir / "ready.json").write_text(
            json.dumps(asdict(handle), indent=2) + "\n"
        )
        yield handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("mocker", "sglang", "native"), default="mocker"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--approve-gpu-hours", type=float)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--duration-seconds", type=float, default=300)
    args = parser.parse_args()
    if (
        not math.isfinite(args.duration_seconds)
        or not 0 < args.duration_seconds <= 14400
    ):
        parser.error("duration must be finite and in (0, 14400] seconds")
    config = ServingConfig(
        model_path=args.model_path,
        model=args.model,
        log_dir=args.log_dir,
        mode=args.mode,
        workers=args.workers,
        approve_gpu_hours=args.approve_gpu_hours,
        gpu=args.gpu,
    )
    with _termination_signal(), serve(config) as handle:
        print(json.dumps(asdict(handle)), flush=True)
        threading.Event().wait(args.duration_seconds)


if __name__ == "__main__":
    main()
