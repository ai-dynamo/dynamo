# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for routed expert info return.

Starts a Dynamo frontend + SGLang backend with --enable-return-routed-experts
and verifies that expert routing info appears in the response nvext.

Usage:
    python test_sglang_expert_info.py

Requires etcd and nats running (see deploy/docker-compose.yml).
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

# Configuration
MODEL = os.environ.get("MODEL_PATH", os.path.expanduser("~/proj/models/dsv2-lite-fp8"))
HOST = "127.0.0.1"
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "30080"))
SYSTEM_PORT = int(os.environ.get("DYN_SYSTEM_PORT", "9092"))
FRONTEND_URL = f"http://{HOST}:{FRONTEND_PORT}"
SYSTEM_URL = f"http://{HOST}:{SYSTEM_PORT}"
LOG_DIR = "/tmp/sglang_expert_info_test"


def start_frontend():
    """Start the Dynamo frontend."""
    print("\nStarting Dynamo frontend...")
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(f"{LOG_DIR}/frontend.log", "w")

    cmd = [sys.executable, "-m", "dynamo.frontend", "--http-port", str(FRONTEND_PORT)]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Logs: {LOG_DIR}/frontend.log")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{FRONTEND_URL}/health", timeout=1)
            if resp.status_code == 200:
                print("  Frontend is ready!")
                return process
        except requests.exceptions.RequestException:
            pass
        if process.poll() is not None:
            print("  Frontend process died!")
            sys.exit(1)
        time.sleep(1)

    print("  Frontend failed to start in time!")
    process.kill()
    sys.exit(1)


def start_sglang_backend():
    """Start the SGLang backend."""
    print("\nStarting SGLang backend...")
    log = open(f"{LOG_DIR}/backend.log", "w")

    env = os.environ.copy()
    env["DYN_SYSTEM_PORT"] = str(SYSTEM_PORT)

    cmd = [
        sys.executable,
        "-m",
        "dynamo.sglang",
        "--model-path",
        MODEL,
        "--tp",
        "1",
        "--mem-fraction-static",
        "0.8",
        "--enable-return-routed-experts",
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
            print("  Backend process died! Check logs:")
            print(f"    tail {LOG_DIR}/backend.log")
            sys.exit(1)
        time.sleep(2)

    print("  Backend failed to start in time!")
    process.kill()
    sys.exit(1)


def validate_routed_experts(routed_experts):
    """Check that routed_experts is a non-empty nested list of ints."""
    assert isinstance(
        routed_experts, list
    ), f"Expected list, got {type(routed_experts)}"
    assert len(routed_experts) > 0, "routed_experts is empty"
    # Shape should be (num_layers, seq_len, top_k)
    assert isinstance(routed_experts[0], list), "Expected nested list (layers)"
    assert isinstance(routed_experts[0][0], list), "Expected nested list (tokens)"
    assert isinstance(routed_experts[0][0][0], int), "Expected int expert IDs"


def test_completions_non_streaming():
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
    validate_routed_experts(nvext["routed_experts"])
    print(f"  routed_experts shape: {len(nvext['routed_experts'])} layers")
    print("  PASSED")


def test_completions_streaming():
    """Streaming completions should return routed_experts in final chunk's nvext."""
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

    chunks = []
    found_routed_experts = False
    for line in resp.iter_lines():
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        chunks.append(chunk)
        nvext = chunk.get("nvext", {})
        if "routed_experts" in nvext:
            found_routed_experts = True
            validate_routed_experts(nvext["routed_experts"])
            print(f"  routed_experts shape: {len(nvext['routed_experts'])} layers")

    print(f"  Total chunks: {len(chunks)}")
    assert len(chunks) > 0, "Expected at least one chunk"
    assert found_routed_experts, "Expected routed_experts in at least one nvext chunk."
    print("  PASSED")


def test_chat_completions_streaming():
    """Streaming chat completions should return routed_experts in final chunk's nvext."""
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

    chunks = []
    found_routed_experts = False
    for line in resp.iter_lines():
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        chunks.append(chunk)
        nvext = chunk.get("nvext", {})
        if "routed_experts" in nvext:
            found_routed_experts = True
            validate_routed_experts(nvext["routed_experts"])
            print(f"  routed_experts shape: {len(nvext['routed_experts'])} layers")

    print(f"  Total chunks: {len(chunks)}")
    assert len(chunks) > 0, "Expected at least one chunk"
    assert found_routed_experts, "Expected routed_experts in at least one nvext chunk."
    print("  PASSED")


def main():
    frontend_process = None
    backend_process = None
    try:
        frontend_process = start_frontend()
        backend_process = start_sglang_backend()
        time.sleep(2)

        print("\n" + "=" * 60)
        print("Running expert info tests")
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
        for name, proc in [
            ("backend", backend_process),
            ("frontend", frontend_process),
        ]:
            if proc:
                print(f"  Stopping {name}...")
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Done")


if __name__ == "__main__":
    main()
