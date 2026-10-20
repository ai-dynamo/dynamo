# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node DP rank launch and fail-together process supervision."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dynamo.deepinfer.args import Config


def rank_environments(config: Config) -> list[dict[str, str]]:
    dp = config.engine_args["data_parallel_size"]
    cp = config.engine_args.get("context_parallel_size", 1)
    devices = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(devices) < dp * cp or any(not d for d in devices):
        raise ValueError("single-node DP requires at least DP * CP visible GPUs")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must not repeat GPUs")
    environments = []
    port = int(os.environ.get("DYN_SYSTEM_PORT", "0"))
    for rank in range(dp):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[rank * cp : (rank + 1) * cp])
        env["DYN_SYSTEM_PORT"] = str(port + rank if port else 0)
        environments.append(env)
    return environments


def stop_ranks(processes: list[subprocess.Popen]) -> None:
    # A cancelled Python future cannot interrupt a native collective. Bound teardown.
    for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    if any(process.poll() is None for process in processes):
        raise RuntimeError("DeepInfer ranks did not exit after SIGKILL")


def launch_ranks(config: Config, argv: list[str] | None = None) -> int:
    environments = rank_environments(config)
    argv = sys.argv[1:] if argv is None else argv
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    processes = []
    try:
        with tempfile.TemporaryDirectory(prefix="deepinfer-ranks-") as ready_dir:
            for rank, env in enumerate(environments):
                env["DYN_DEEPINFER_READY_DIR"] = ready_dir
                engine_args = {**config.engine_args, "engine_rank": rank}
                command = [
                    sys.executable,
                    "-m",
                    "dynamo.deepinfer",
                    *argv,
                    "--rank-worker",
                    "--engine-args",
                    json.dumps(engine_args),
                ]
                processes.append(
                    subprocess.Popen(command, env=env, start_new_session=True)
                )
            deadline = time.monotonic() + 3600
            ready = False
            while not stopping:
                for rank, process in enumerate(processes):
                    if process.poll() is not None:
                        print(
                            f"DeepInfer rank {rank} exited: {process.returncode}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 1
                if not ready:
                    ready = all(
                        Path(ready_dir, str(rank)).exists()
                        for rank in range(len(processes))
                    )
                    if ready:
                        print(
                            f"DeepInfer: all {len(processes)} DP ranks initialized",
                            flush=True,
                        )
                    elif time.monotonic() > deadline:
                        raise TimeoutError(
                            "DeepInfer rank initialization exceeded one hour"
                        )
                time.sleep(0.1)
            return 0
    finally:
        try:
            stop_ranks(processes)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
