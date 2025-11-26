# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for /engine/start_profile and /engine/stop_profile routes.

This script demonstrates the new custom engine route registration feature.
It starts a simple sglang server with dynamo and tests the profiling endpoints.

Usage:
    python test_sglang_profile.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# Configuration
MODEL = "Qwen/Qwen3-0.6B"  # Small model for quick testing
HOST = "127.0.0.1"
PORT = 30000
SYSTEM_PORT = 9090
PROFILER_OUTPUT_DIR = "/tmp/dynamo_profiler_test"


def cleanup_output_dir():
    """Clean up the profiler output directory"""
    import shutil

    if os.path.exists(PROFILER_OUTPUT_DIR):
        shutil.rmtree(PROFILER_OUTPUT_DIR)
    os.makedirs(PROFILER_OUTPUT_DIR, exist_ok=True)


def start_frontend():
    """Start the Dynamo frontend (HTTP server)"""
    print("\nStarting Dynamo frontend...")
    print(f"  - Frontend HTTP: http://{HOST}:{PORT}")

    cmd = [
        "python",
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(PORT),
    ]

    print(f"Command: {' '.join(cmd)}")
    print("(Output will appear below)\n")

    process = subprocess.Popen(cmd)

    # Wait for frontend to be ready
    max_wait = 30
    start_time = time.time()
    frontend_ready = False

    while time.time() - start_time < max_wait:
        try:
            # Check /health endpoint first
            response = requests.get(f"http://{HOST}:{PORT}/health", timeout=1)
            if response.status_code == 200:
                print("✓ Frontend is ready!")
                frontend_ready = True
                break
        except requests.exceptions.RequestException:
            pass

        if process.poll() is not None:
            print("✗ Frontend process died!")
            sys.exit(1)

        time.sleep(1)

    if not frontend_ready:
        print("✗ Frontend failed to start in time!")
        process.kill()
        sys.exit(1)

    return process


def start_sglang_backend():
    """Start the sglang backend (inference engine)"""
    print("\nStarting SGLang backend...")
    print(f"  - Model: {MODEL}")
    print(f"  - System server: http://{HOST}:{SYSTEM_PORT}")

    # Set environment variables
    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = PROFILER_OUTPUT_DIR
    env["DYN_SYSTEM_PORT"] = str(SYSTEM_PORT)

    cmd = [
        "python",
        "-m",
        "dynamo.sglang",
        "--model-path",
        MODEL,
        "--tp",
        "1",
        "--mem-fraction-static",
        "0.8",
    ]

    print(f"Command: {' '.join(cmd)}")
    print("(Output will appear below)")
    print("\nWaiting for backend to start...\n")

    process = subprocess.Popen(cmd, env=env)

    # Wait for backend to be ready (check system server health)
    max_wait = 120  # 2 minutes
    start_time = time.time()
    backend_ready = False

    while time.time() - start_time < max_wait:
        try:
            # Check system server health endpoint
            response = requests.get(f"http://{HOST}:{SYSTEM_PORT}/health", timeout=1)
            if response.status_code == 200:
                print("✓ Backend is ready!")
                backend_ready = True
                break
        except requests.exceptions.RequestException:
            pass

        # Check if process has died
        if process.poll() is not None:
            print("✗ Backend process died!")
            sys.exit(1)

        time.sleep(2)

    if not backend_ready:
        print("✗ Backend failed to start in time!")
        process.kill()
        sys.exit(1)

    return process


def test_profiling_endpoints():
    """Test the /engine/start_profile and /engine/stop_profile endpoints"""
    base_url = f"http://{HOST}:{SYSTEM_PORT}"

    print("\n" + "=" * 60)
    print("Testing /engine/start_profile and /engine/stop_profile")
    print("=" * 60)

    # Test 1: Start profiling with parameters (no num_steps so we control stop manually)
    print("\n1. Starting profiling with parameters...")
    response = requests.post(
        f"{base_url}/engine/start_profile",
        json={
            "output_dir": PROFILER_OUTPUT_DIR,
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": True,
        },
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json()["status"] == "ok", "Expected status 'ok'"

    # Check available models
    print("\n2. Checking available models...")
    response = requests.get(f"http://{HOST}:{PORT}/v1/models")
    if response.status_code == 200:
        models = response.json()
        print(f"   Available models: {models}")

    # Make a few inference requests to generate profiling data
    print("\n3. Making inference requests...")
    inference_url = f"http://{HOST}:{PORT}/v1/completions"
    for i in range(3):
        response = requests.post(
            inference_url,
            json={
                "model": MODEL,
                "prompt": f"Hello, this is test request {i+1}. ",
                "max_tokens": 10,
                "temperature": 0.8,
            },
        )
        print(f"   Request {i+1}: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response: {response.text[:200]}")
        time.sleep(0.5)

    # Test 2: Stop profiling
    print("\n4. Stopping profiling...")
    response = requests.post(f"{base_url}/engine/stop_profile")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json()["status"] == "ok", "Expected status 'ok'"

    # Test 3: Test with empty body (GET-like POST)
    print("\n5. Starting profiling with empty body...")
    response = requests.post(f"{base_url}/engine/start_profile")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Test 4: Test invalid route
    print("\n6. Testing invalid route...")
    response = requests.post(f"{base_url}/engine/nonexistent_route")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"

    # Stop profiling again
    response = requests.post(f"{base_url}/engine/stop_profile")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    # Check if profiling files were created
    print(f"\nChecking profiler output directory: {PROFILER_OUTPUT_DIR}")
    if os.path.exists(PROFILER_OUTPUT_DIR):
        files = list(Path(PROFILER_OUTPUT_DIR).rglob("*"))
        if files:
            print(f"✓ Found {len(files)} files in output directory")
            for f in files[:5]:  # Show first 5 files
                print(f"  - {f}")
        else:
            print("⚠ No files found (profiling may not have run long enough)")
    else:
        print("⚠ Output directory not created")


def main():
    """Main test function"""
    frontend_process = None
    backend_process = None
    try:
        # Clean up output directory
        cleanup_output_dir()

        # Start frontend first
        frontend_process = start_frontend()

        # Start backend
        backend_process = start_sglang_backend()

        # Run tests
        print("\n" + "=" * 60)
        print("Both frontend and backend are ready!")
        print("=" * 60)
        time.sleep(2)  # Give everything a moment to fully settle
        test_profiling_endpoints()

        print("\n✓ Test completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        print("\nShutting down servers...")
        if backend_process:
            print("  Stopping backend...")
            backend_process.send_signal(signal.SIGTERM)
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("  Force killing backend...")
                backend_process.kill()

        if frontend_process:
            print("  Stopping frontend...")
            frontend_process.send_signal(signal.SIGTERM)
            try:
                frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("  Force killing frontend...")
                frontend_process.kill()

        print("✓ Servers stopped")


if __name__ == "__main__":
    main()
