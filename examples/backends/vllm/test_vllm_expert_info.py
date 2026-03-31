# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for routed expert info return on the vLLM backend.

Starts a Dynamo frontend + vLLM backend with --enable-return-routed-experts
and verifies that expert routing info appears in response nvext.

Requires etcd and nats running (see deploy/docker-compose.yml).

Usage:
    python test_vllm_expert_info.py

Optional environment variables:
    MODEL_PATH: MoE model path or HF model id
    FRONTEND_PORT: frontend HTTP port
    DYN_SYSTEM_PORT: backend system port
    FRONTEND_EXTRA_ARGS: extra args appended to `python -m dynamo.frontend`
    BACKEND_EXTRA_ARGS: extra args appended to `python -m dynamo.vllm`
"""

import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import requests

# Configuration
# Use an MoE model by default. Override MODEL_PATH in your environment as needed.
MODEL = os.environ.get("MODEL_PATH", os.path.expanduser("~/proj/models/dsv2-lite-fp8"))
HOST = "127.0.0.1"
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "30080"))
SYSTEM_PORT = int(os.environ.get("DYN_SYSTEM_PORT", "8081"))
FRONTEND_URL = f"http://{HOST}:{FRONTEND_PORT}"
SYSTEM_URL = f"http://{HOST}:{SYSTEM_PORT}"
FRONTEND_EXTRA_ARGS = shlex.split(os.environ.get("FRONTEND_EXTRA_ARGS", ""))
BACKEND_EXTRA_ARGS = shlex.split(os.environ.get("BACKEND_EXTRA_ARGS", ""))
LOG_DIR = "/tmp/vllm_expert_info_test"


def wait_for_http_ready(url: str, *, name: str, timeout_s: int) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                print(f"  {name} is ready!")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise RuntimeError(f"{name} failed to start in time: {url}")


def start_frontend() -> subprocess.Popen[Any]:
    """Start the Dynamo frontend."""
    print("\nStarting Dynamo frontend...")
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(f"{LOG_DIR}/frontend.log", "w")

    cmd = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(FRONTEND_PORT),
        *FRONTEND_EXTRA_ARGS,
    ]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Logs: {LOG_DIR}/frontend.log")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        wait_for_http_ready(f"{FRONTEND_URL}/health", name="Frontend", timeout_s=30)
        return process
    except Exception:
        if process.poll() is None:
            process.kill()
        raise


def start_vllm_backend() -> subprocess.Popen[Any]:
    """Start the vLLM backend worker."""
    print("\nStarting vLLM backend...")
    log = open(f"{LOG_DIR}/backend.log", "w")

    env = os.environ.copy()
    env["DYN_SYSTEM_PORT"] = str(SYSTEM_PORT)

    cmd = [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        MODEL,
        "--enforce-eager",
        "--enable-return-routed-experts",
        *BACKEND_EXTRA_ARGS,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Logs: {LOG_DIR}/backend.log")
    process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    max_wait = 300
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{SYSTEM_URL}/health", timeout=1)
            if resp.status_code == 200:
                print("  Backend is ready!")
                return process
        except requests.exceptions.RequestException:
            pass
        if process.poll() is not None:
            raise RuntimeError(
                "Backend process died during startup. "
                f"Check logs: {LOG_DIR}/backend.log"
            )
        time.sleep(2)

    if process.poll() is None:
        process.kill()
    raise RuntimeError(f"Backend failed to start in time: {SYSTEM_URL}/health")


def validate_routed_experts(routed_experts: Any) -> np.ndarray:
    """Check that routed_experts is a base64-encoded string of int32 expert IDs."""
    assert isinstance(
        routed_experts, str
    ), f"Expected base64 string, got {type(routed_experts)}"
    decoded = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
    assert len(decoded) > 0, "routed_experts decoded to empty array"
    return decoded


def iter_sse_json(resp: requests.Response) -> list[dict[str, Any]]:
    chunks = []
    for line in resp.iter_lines():
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        chunks.append(json.loads(payload))
    return chunks


def assert_final_chunk_has_routed_experts(chunks: list[dict[str, Any]]) -> None:
    assert chunks, "Expected at least one streamed chunk"
    routed_expert_positions = []
    for idx, chunk in enumerate(chunks):
        nvext = chunk.get("nvext", {})
        if "routed_experts" in nvext:
            routed_expert_positions.append(idx)
            validate_routed_experts(nvext["routed_experts"])

    assert (
        routed_expert_positions
    ), "Expected routed_experts in at least one nvext chunk"
    assert routed_expert_positions == [len(chunks) - 1], (
        "Expected routed_experts only on the final streamed chunk, got positions "
        f"{routed_expert_positions} out of {len(chunks)} chunks"
    )


def test_completions_non_streaming() -> None:
    """Non-streaming completions should return routed_experts in nvext."""
    print("\n--- test_completions_non_streaming ---")

    resp = requests.post(
        f"{FRONTEND_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=30,
    )
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response keys: {list(data.keys())}")
    assert resp.status_code == 200
    assert "choices" in data
    assert len(data["choices"]) > 0

    nvext = data.get("nvext", {})
    assert (
        "routed_experts" in nvext
    ), f"Expected routed_experts in nvext, got keys: {list(nvext.keys())}"
    decoded = validate_routed_experts(nvext["routed_experts"])
    print(f"  routed_experts decoded ints: {len(decoded)}")
    print("  PASSED")


def test_completions_streaming() -> None:
    """Streaming completions should return routed_experts on the final chunk only."""
    print("\n--- test_completions_streaming ---")

    resp = requests.post(
        f"{FRONTEND_URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
        },
        timeout=30,
        stream=True,
    )
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 200

    chunks = iter_sse_json(resp)
    print(f"  Total chunks: {len(chunks)}")
    assert_final_chunk_has_routed_experts(chunks)
    print("  PASSED")


def test_chat_completions_streaming() -> None:
    """Streaming chat completions should return routed_experts on the final chunk only."""
    print("\n--- test_chat_completions_streaming ---")

    resp = requests.post(
        f"{FRONTEND_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
        },
        timeout=30,
        stream=True,
    )
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 200

    chunks = iter_sse_json(resp)
    print(f"  Total chunks: {len(chunks)}")
    assert_final_chunk_has_routed_experts(chunks)
    print("  PASSED")


def stop_process(name: str, proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    print(f"  Stopping {name}...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    frontend_process = None
    backend_process = None
    try:
        frontend_process = start_frontend()
        backend_process = start_vllm_backend()
        time.sleep(2)

        print("\n" + "=" * 60)
        print("Running vLLM expert info tests")
        print("=" * 60)

        test_completions_non_streaming()
        test_completions_streaming()
        test_chat_completions_streaming()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\nShutting down...")
        stop_process("backend", backend_process)
        stop_process("frontend", frontend_process)


if __name__ == "__main__":
    main()
