#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a native engine and compare its mocker against it.

Runs inside the engine's upstream image. The test binary is the mocker crate's
`sidecar` integration test; see `parity_with_live_engine` in
`lib/mocker/servers/<engine>/tests/sidecar.rs`.
"""

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
CONTEXT_LENGTH = 1024
MAX_NUM_SEQS = 4
BLOCK_SIZE = 16
PACKAGES = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def engine_command(engine, model, grpc_port, http_port, kv_port, scratch):
    kv_events = json.dumps(
        {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{kv_port}",
        }
    )
    if engine == "vllm":
        return [
            "vllm-rs", "serve", model,
            "--host", "127.0.0.1",
            "--port", str(http_port),
            "--grpc-port", str(grpc_port),
            "--max-model-len", str(CONTEXT_LENGTH),
            "--reasoning-parser", "none",
            "--",
            "--enforce-eager",
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(CONTEXT_LENGTH),
            "--block-size", str(BLOCK_SIZE),
            "--kv-cache-memory-bytes", "1073741824",
            "--gpu-memory-utilization", "0.01",
            "--enable-prefix-caching",
            "--kv-events-config", kv_events,
            "--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        ]  # fmt: skip
    if engine == "sglang":
        return [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "127.0.0.1",
            "--port", str(http_port),
            "--grpc-port", str(grpc_port),
            "--incremental-streaming-output",
            "--context-length", str(CONTEXT_LENGTH),
            "--max-running-requests", str(MAX_NUM_SEQS),
            "--chunked-prefill-size", str(CONTEXT_LENGTH),
            "--page-size", str(BLOCK_SIZE),
            "--mem-fraction-static", "0.1",
            "--kv-events-config", kv_events,
        ]  # fmt: skip
    options = scratch / "engine.yaml"
    options.write_text(
        json.dumps(
            {
                "max_batch_size": MAX_NUM_SEQS,
                "max_num_tokens": CONTEXT_LENGTH,
                "kv_cache_config": {
                    "tokens_per_block": BLOCK_SIZE,
                    "free_gpu_memory_fraction": 0.1,
                },
                "cache_transceiver_config": {"backend": "NIXL"},
            }
        )
    )
    return [
        sys.executable, "-m", "tensorrt_llm.commands.serve", model,
        "--grpc", "--grpc-protocol", "openengine",
        "--host", "127.0.0.1",
        "--port", str(grpc_port),
        "--max_seq_len", str(CONTEXT_LENGTH),
        "--extra_llm_api_options", str(options),
    ]  # fmt: skip


def wait_ready(process, grpc_port, health_url, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"engine exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", grpc_port), timeout=1):
                if health_url is None:
                    return
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"engine not ready within {timeout} s")


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=PACKAGES, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--results", type=Path, default=Path("test-results/mocker-parity")
    )
    args = parser.parse_args()

    results = (args.results / args.engine).resolve()
    results.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version(PACKAGES[args.engine])
    print(f"{args.engine} {version}, model {args.model}", flush=True)
    grpc_port, http_port, kv_port = free_port(), free_port(), free_port()
    health_url = (
        None if args.engine == "trtllm" else f"http://127.0.0.1:{http_port}/health"
    )
    with tempfile.TemporaryDirectory() as scratch, (results / "engine.log").open(
        "w"
    ) as log:
        command = engine_command(
            args.engine, args.model, grpc_port, http_port, kv_port, Path(scratch)
        )
        print(" ".join(command), flush=True)
        engine = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            # The SGLang sidecar speaks sglang.api.v1, which SGLang's Rust server serves.
            env=dict(os.environ, SGLANG_RUST_SERVER="1"),
        )
        try:
            wait_ready(engine, grpc_port, health_url)
            test = subprocess.run(
                [
                    str(args.binary.resolve()),
                    "--exact",
                    "parity_with_live_engine",
                    "--ignored",
                    "--nocapture",
                    "--test-threads",
                    "1",
                ],  # fmt: skip
                env=dict(
                    os.environ,
                    MOCKER_PARITY_ENDPOINT=f"http://127.0.0.1:{grpc_port}",
                    MOCKER_PARITY_MODEL=args.model,
                    MOCKER_PARITY_ENGINE_VERSION=version,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )
        finally:
            stop(engine)
    (results / "test.log").write_text(test.stdout)
    print(test.stdout, flush=True)
    # libtest also exits 0 when the filter matches no test.
    if test.returncode != 0 or "1 passed; 0 failed; 0 ignored;" not in test.stdout:
        sys.exit(f"parity test failed; engine log: {results / 'engine.log'}")


if __name__ == "__main__":
    main()
