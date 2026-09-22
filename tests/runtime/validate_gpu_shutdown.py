# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual GPU validation: run inside a backend image with this checkout mounted."""

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import psutil
import requests

from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port


def validate(args):
    # Regression: engine teardown can interrupt admitted inference on SIGTERM;
    # verify the full HTTP stream and clean process exit with a real GPU engine.
    root = Path(__file__).resolve().parents[2]
    output = Path(tempfile.mkdtemp(prefix=f"shutdown-{args.backend}-", dir=args.output))
    output.chmod(0o755)
    port = allocate_port(DynamoPortRange.FRONTEND.value)
    env = {
        **os.environ,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_KV": str(output / "discovery"),
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_SYSTEM_PORT": "0",
        "DYN_TCP_RPC_PORT": "0",
        "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS": "1",
        "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT": "30",
    }
    command = [sys.executable, "-m", f"dynamo.{args.backend}"]
    if args.backend == "vllm":
        command += [
            "--model",
            args.model,
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.3",
            "--enforce-eager",
        ]
    elif args.backend == "sglang":
        command += [
            "--model-path",
            args.model,
            "--context-length",
            "2048",
            "--mem-fraction-static",
            "0.3",
            "--disable-cuda-graph",
        ]
    else:
        command += [
            "--model-path",
            args.model,
            "--extra-engine-args",
            str(root / "tests/serve/trtllm/engine_configs/qwen3/agg.yaml"),
        ]
    base_url = f"http://127.0.0.1:{port}"

    def model_ready():
        response = requests.get(f"{base_url}/v1/models", timeout=3)
        response.raise_for_status()
        assert any(model["id"] == args.model for model in response.json()["data"])
        return True

    common = dict(
        env=env,
        terminate_all_matching_process_names=False,
        log_dir=str(output),
        timeout=600,
    )
    print(f"RESULT_DIRECTORY={output}", flush=True)
    try:
        with ManagedProcess(
            command=[sys.executable, "-m", "dynamo.frontend", "--http-port", str(port)],
            display_name="frontend",
            health_check_urls=[f"{base_url}/health"],
            **common,
        ), ManagedProcess(
            command=command,
            display_name="worker",
            health_check_funcs=[model_ready],
            **common,
        ) as worker:
            assert worker.proc is not None
            payload = {
                "model": args.model,
                "prompt": "Count from one to one hundred:",
                "max_tokens": 512,
                "ignore_eos": True,
                "temperature": 0,
                "stream": True,
            }
            signalled = False
            finished = False
            done = False
            chunks_after_signal = 0
            children = []
            with requests.post(
                f"{base_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=(10, 45),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:]
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    assert "error" not in event, event
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    if signalled:
                        chunks_after_signal += 1
                    if choices[0].get("finish_reason") is not None:
                        assert choices[0]["finish_reason"] == "length", event
                        finished = True
                    if not signalled and choices[0].get("text"):
                        children = psutil.Process(worker.proc.pid).children(
                            recursive=True
                        )
                        started = time.monotonic()
                        os.kill(worker.proc.pid, signal.SIGTERM)
                        signalled = True
                        print("SIGTERM_DURING_INFERENCE", flush=True)
            assert signalled and done and finished and chunks_after_signal > 0
            code = worker.proc.wait(timeout=40)
            elapsed = time.monotonic() - started
            assert code == 0, f"worker exit={code}; logs={worker.log_path}"
            _, alive = psutil.wait_procs(children, timeout=5)
            alive = [p.pid for p in alive if p.status() != psutil.STATUS_ZOMBIE]
            assert not alive, f"surviving engine processes: {alive}"
            logs = worker.read_logs()
            assert "shutdown stage=cleanup reason=completed" in logs
            assert "shutdown stage=runtime reason=completed" in logs
            result = {
                "backend": args.backend,
                "model": args.model,
                "worker_exit": code,
                "shutdown_seconds": elapsed,
                "chunks_after_signal": chunks_after_signal,
                "stream_completed": done,
                "surviving_children": alive,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        deallocate_port(port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["vllm", "sglang", "trtllm"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default=None)
    logging.basicConfig(level=logging.INFO)
    validate(parser.parse_args())
