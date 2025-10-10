# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test configuration and fixtures for Dynamo Python bindings tests.

TWO MODES OF OPERATION:

1. Isolated Mode (ENABLE_ISOLATED_ETCD_AND_NATS=1):
   - Each test gets fresh NATS/ETCD on random ports
   - Requires: pytest-forked (uv pip install pytest-forked)
   - Tests using 'runtime' fixture MUST have @pytest.mark.forked
   - Safer, enables parallel execution
   - Run: ENABLE_ISOLATED_ETCD_AND_NATS=1 pytest tests/test_metrics_registry.py -n auto

2. Default Ports Mode (ENABLE_ISOLATED_ETCD_AND_NATS=0, default):
   - All tests share NATS/ETCD on default ports (4222, 2379)
   - No pytest-forked required
   - No @pytest.mark.forked required
   - Faster for sequential runs, but NO parallel execution
   - Run: pytest tests/test_metrics_registry.py

Performance comparison (32-core machine, 13 tests):
    Default ports (ENABLE_ISOLATED_ETCD_AND_NATS=0, default): 4.06s (sequential only)
    Isolated sequential (ENABLE_ISOLATED_ETCD_AND_NATS=1):    8.58s (2.1x slower, but safer)
    Isolated parallel -n 8:   2.82s (1.4x faster than default)
    Isolated parallel -n 16:  2.28s (1.8x faster than default, optimal)
    Isolated parallel -n 32:  2.74s (overhead dominates)

    Recommendation: Default mode for simplicity. Use ENABLE_ISOLATED_ETCD_AND_NATS=1
    with -n 8 to -n 16 when you need parallel execution and maximum test isolation.
"""

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time

import pytest

from dynamo.runtime import DistributedRuntime

# Configuration constants
# ENABLE_ISOLATED_ETCD_AND_NATS: When True, each test gets isolated NATS/ETCD instances
# on random ports with unique data directories. This enables parallel test execution.
# Set to False to use default ports (4222, 2379) for sequential execution.
# Can be overridden by environment variable: ENABLE_ISOLATED_ETCD_AND_NATS=0 or =1
ENABLE_ISOLATED_ETCD_AND_NATS = (
    os.environ.get("ENABLE_ISOLATED_ETCD_AND_NATS", "0") == "1"
)

# Check if pytest-forked is installed (only when using isolated NATS/ETCD)
# This is REQUIRED when ENABLE_ISOLATED_ETCD_AND_NATS=1 because each test gets
# fresh services and the DistributedRuntime singleton needs process isolation
if ENABLE_ISOLATED_ETCD_AND_NATS:
    try:
        import pytest_forked  # noqa: F401
    except ImportError:
        pytest.exit(
            """
pytest-forked is required when ENABLE_ISOLATED_ETCD_AND_NATS=1.
Install it with: uv pip install pytest-forked

This is needed because DistributedRuntime is a process-level singleton
and tests must run in separate processes to avoid 'Worker already initialized' errors.

Alternatively, set ENABLE_ISOLATED_ETCD_AND_NATS=0 to use default ports (slower, sequential only).
""",
            returncode=1,
        )

# Timeout constants
SERVICE_STARTUP_TIMEOUT = 5
SERVICE_SHUTDOWN_TIMEOUT = 5


def get_free_port():
    """Find and return an available port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_for_port(host, port, timeout: float = SERVICE_STARTUP_TIMEOUT):
    """Wait for a port to be available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            return True
        except (socket.error, ConnectionRefusedError):
            time.sleep(0.1)
    return False


def start_nats_and_etcd_default_ports():
    """
    Start NATS and ETCD on default ports (4222, 2379).

    Use this for sequential test execution or when running tests alone.
    Faster startup if services are already running.
    """
    # Use default ports
    nats_port = 4222
    etcd_client_port = 2379

    # No data directories needed - use defaults
    nats_data_dir = None
    etcd_data_dir = None

    # Check if ports are already in use (reuse them if so)
    # TODO: In the future, error out to ensure proper test isolation
    nats_already_running = wait_for_port("localhost", nats_port, timeout=0.1)
    etcd_already_running = wait_for_port("localhost", etcd_client_port, timeout=0.1)

    if nats_already_running and etcd_already_running:
        print(
            f"Reusing existing NATS on port {nats_port} and ETCD on port {etcd_client_port}"
        )
        # Set environment variables for the runtime to use
        os.environ["NATS_SERVER"] = f"nats://localhost:{nats_port}"
        os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_client_port}"
        # Return None for processes since we're reusing existing services
        return None, None, nats_port, etcd_client_port, None, None

    # Set environment variables for the runtime to use
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_client_port}"

    print(f"Using NATS on default port {nats_port}")
    print(f"Using ETCD on default client port {etcd_client_port}")

    # Start services with default ports
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])

    return nats_server, etcd, nats_port, etcd_client_port, nats_data_dir, etcd_data_dir


def start_nats_and_etcd_random_ports():
    """
    Start NATS and ETCD with random ports and unique data directories.

    This ensures test isolation by giving each test module (or parallel worker)
    its own NATS/ETCD instances on different ports with separate data directories.
    This allows tests to run in parallel without port or filesystem conflicts.

    Note: etcd uses port 0 (OS-assigned port) to eliminate race conditions.
    NATS uses get_free_port() with retry logic since it doesn't support port 0.
    Port collision probability per NATS attempt: ~1% (heavy parallel testing), ~0.05% (normal load).
    With 5 retries, probability of all NATS attempts failing: ~1.5e-10 (essentially never).
    """
    # Create unique temporary data directories
    nats_data_dir = tempfile.mkdtemp(prefix="nats_data_")
    etcd_data_dir = tempfile.mkdtemp(prefix="etcd_data_")

    # Start etcd first with port 0 (no race condition, no retries needed)
    print(f"Starting ETCD with port 0 (OS-assigned), data dir: {etcd_data_dir}")
    etcd = subprocess.Popen(
        [
            "etcd",
            "--data-dir",
            str(etcd_data_dir),
            "--listen-client-urls",
            "http://localhost:0",
            "--advertise-client-urls",
            "http://localhost:0",
            "--listen-peer-urls",
            "http://localhost:0",
            "--initial-advertise-peer-urls",
            "http://localhost:0",
            "--initial-cluster",
            "default=http://localhost:0",
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Parse etcd's stderr to discover the actual client port it bound to
    etcd_client_port = None
    timeout_at = time.time() + 5.0

    while time.time() < timeout_at:
        if etcd.poll() is not None:
            stderr = etcd.stderr.read() if etcd.stderr else ""
            shutil.rmtree(nats_data_dir, ignore_errors=True)
            shutil.rmtree(etcd_data_dir, ignore_errors=True)
            raise RuntimeError(f"ETCD failed to start: {stderr}")

        line = etcd.stderr.readline() if etcd.stderr else ""
        if not line:
            time.sleep(0.01)
            continue

        try:
            log = json.loads(line)
            msg = log.get("msg", "")

            # Look for the client port
            if "serving client traffic" in msg or "serving client" in msg:
                address = log.get("address", "")
                match = re.search(r":(\d+)$", address)
                if match:
                    etcd_client_port = int(match.group(1))
                    print(f"ETCD bound to client port: {etcd_client_port}")
                    break
        except (json.JSONDecodeError, ValueError):
            continue

    if etcd_client_port is None:
        etcd.terminate()
        etcd.wait()
        shutil.rmtree(nats_data_dir, ignore_errors=True)
        shutil.rmtree(etcd_data_dir, ignore_errors=True)
        raise RuntimeError("Failed to discover ETCD client port from logs")

    # Now start NATS with retry logic (up to 5 attempts due to race condition)
    max_nats_retries = 5
    nats_server = None
    nats_port = None
    last_error = None

    for attempt in range(max_nats_retries):
        try:
            nats_port = get_free_port()
            print(
                f"Attempt {attempt + 1}: Starting NATS on port {nats_port}, data dir: {nats_data_dir}"
            )

            nats_server = subprocess.Popen(
                ["nats-server", "-js", "-p", str(nats_port), "-sd", str(nats_data_dir)],
                stderr=subprocess.PIPE,
            )

            # Give NATS a moment to bind to the port
            time.sleep(0.1)

            # Check if NATS failed to start
            if nats_server.poll() is not None:
                stderr = (
                    nats_server.stderr.read().decode() if nats_server.stderr else ""
                )
                if "address already in use" in stderr.lower():
                    print(f"NATS port {nats_port} already in use, retrying...")
                    time.sleep(0.1)
                    continue
                etcd.terminate()
                etcd.wait()
                shutil.rmtree(nats_data_dir, ignore_errors=True)
                shutil.rmtree(etcd_data_dir, ignore_errors=True)
                raise RuntimeError(f"NATS failed to start: {stderr}")

            # Success - NATS started
            break

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_nats_retries - 1:
                time.sleep(0.2)
            else:
                etcd.terminate()
                etcd.wait()
                shutil.rmtree(nats_data_dir, ignore_errors=True)
                shutil.rmtree(etcd_data_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Failed to start NATS after {max_nats_retries} attempts: {last_error}"
                )

    # Set environment variables for the runtime to use
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_client_port}"

    return nats_server, etcd, nats_port, etcd_client_port, nats_data_dir, etcd_data_dir


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """
    Start NATS and ETCD for testing.

    Scope is "module" which means each test module shares the same NATS/ETCD instance.

    Behavior is controlled by ENABLE_ISOLATED_ETCD_AND_NATS constant:
    - True (default): Random ports + unique data dirs for parallel execution
    - False: Default ports (4222, 2379) for sequential execution
    """
    if ENABLE_ISOLATED_ETCD_AND_NATS:
        (
            nats_server,
            etcd,
            nats_port,
            etcd_client_port,
            nats_data_dir,
            etcd_data_dir,
        ) = start_nats_and_etcd_random_ports()
    else:
        (
            nats_server,
            etcd,
            nats_port,
            etcd_client_port,
            nats_data_dir,
            etcd_data_dir,
        ) = start_nats_and_etcd_default_ports()

    try:
        # Wait for services to be ready
        if not wait_for_port("localhost", nats_port, timeout=SERVICE_STARTUP_TIMEOUT):
            raise RuntimeError(f"NATS server failed to start on port {nats_port}")
        if not wait_for_port(
            "localhost", etcd_client_port, timeout=SERVICE_STARTUP_TIMEOUT
        ):
            raise RuntimeError(f"ETCD failed to start on port {etcd_client_port}")

        print(f"NATS ({nats_port}) and ETCD ({etcd_client_port}) services ready")
        yield
    finally:
        # Teardown code - always runs even if setup fails or tests error
        print("Tearing down resources")

        # Only terminate services if we started them (not reusing existing)
        if nats_server is None and etcd is None:
            print("Reused existing services, not stopping them")
        else:
            # Terminate both processes first (parallel shutdown)
            try:
                if nats_server:
                    nats_server.terminate()
            except Exception as e:
                print(f"Error terminating NATS: {e}")
            try:
                if etcd:
                    etcd.terminate()
            except Exception as e:
                print(f"Error terminating ETCD: {e}")

            # Wait for both processes to finish
            try:
                if nats_server:
                    nats_server.wait(timeout=SERVICE_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                print("NATS did not terminate gracefully, killing")
                try:
                    nats_server.kill()
                except Exception:
                    pass
            except Exception as e:
                print(f"Error waiting for NATS: {e}")

            try:
                if etcd:
                    etcd.wait(timeout=SERVICE_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                print("ETCD did not terminate gracefully, killing")
                try:
                    etcd.kill()
                except Exception:
                    pass
            except Exception as e:
                print(f"Error waiting for ETCD: {e}")

        # Clean up temporary data directories (if created)
        if nats_data_dir:
            try:
                shutil.rmtree(nats_data_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error removing NATS data dir: {e}")
        if etcd_data_dir:
            try:
                shutil.rmtree(etcd_data_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error removing ETCD data dir: {e}")


@pytest.fixture(scope="function", autouse=False)
async def runtime(request):
    """
    Create a DistributedRuntime for testing.

    IMPORTANT: DistributedRuntime is a process-level singleton. When using isolated
    NATS/ETCD (ENABLE_ISOLATED_ETCD_AND_NATS=1), tests using this fixture MUST be
    marked with `@pytest.mark.forked` to run in a separate process for isolation.

    Without @pytest.mark.forked in isolated mode, you will get "Worker already initialized"
    errors when multiple tests try to create runtimes in the same process.
    """
    # Check if the test is marked with @pytest.mark.forked (only in isolated mode)
    if ENABLE_ISOLATED_ETCD_AND_NATS:
        forked_marker = request.node.get_closest_marker("forked")
        if forked_marker is None:
            pytest.fail(
                f"""
Test '{request.node.name}' uses the 'runtime' fixture but is not marked with @pytest.mark.forked.
This is required when ENABLE_ISOLATED_ETCD_AND_NATS=1.

Add @pytest.mark.forked decorator to run this test in a separate process:
  @pytest.mark.forked
  async def test_my_test(runtime):
      ...

Or set ENABLE_ISOLATED_ETCD_AND_NATS=0 to use default ports (no forking needed).

This is required because DistributedRuntime is a process-level singleton.
"""
            )

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, True)
    yield runtime
    runtime.shutdown()
