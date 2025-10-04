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

To run tests with isolated NATS and ETCD (safer, but may or may not be faster):
    1. Install pytest-xdist: uv pip install pytest-xdist
    2. Run: ENABLE_ISOLATED_ETCD_AND_NATS=1 \
        pytest tests/test_metrics_registry.py -n auto --benchmark-disable

To run tests sequentially, run:
    ENABLE_ISOLATED_ETCD_AND_NATS=0 pytest tests/test_metrics_registry.py
"""

import asyncio
import os
import shutil
import socket
import subprocess
import tempfile
import time

import pytest

from dynamo.runtime import DistributedRuntime

# Configuration constants
# USE_PARALLEL_NATS_AND_ETCD: When True, each test gets isolated NATS/ETCD instances
# on random ports with unique data directories. This enables parallel test execution.
# Set to False to use default ports (4222, 2379) for sequential execution.
# Can be overridden by environment variable: ENABLE_ISOLATED_ETCD_AND_NATS=0 or =1
ENABLE_ISOLATED_ETCD_AND_NATS = (
    os.environ.get("ENABLE_ISOLATED_ETCD_AND_NATS", "1") == "1"
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

    # Check if ports are already in use (error out to ensure isolation)
    if wait_for_port("localhost", nats_port, timeout=0.1):
        raise RuntimeError(
            f"NATS port {nats_port} is already in use! Tests MUST run with isolated NATS/ETCD instances. "
            f"Please kill existing services or set ENABLE_ISOLATED_ETCD_AND_NATS=1"
        )
    if wait_for_port("localhost", etcd_client_port, timeout=0.1):
        raise RuntimeError(
            f"ETCD port {etcd_client_port} is already in use! Tests MUST run with isolated NATS/ETCD instances. "
            f"Please kill existing services or set ENABLE_ISOLATED_ETCD_AND_NATS=1"
        )

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
    """
    # Allocate random available ports
    nats_port = get_free_port()
    etcd_client_port = get_free_port()
    etcd_peer_port = get_free_port()

    # Create unique temporary data directories for NATS and ETCD
    nats_data_dir = tempfile.mkdtemp(prefix="nats_data_")
    etcd_data_dir = tempfile.mkdtemp(prefix="etcd_data_")

    # Set environment variables for the runtime to use
    os.environ["NATS_SERVER"] = f"nats://localhost:{nats_port}"
    os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_client_port}"

    print(f"Starting NATS on port {nats_port}, data dir: {nats_data_dir}")
    print(
        f"Starting ETCD on client port {etcd_client_port}, peer port {etcd_peer_port}, data dir: {etcd_data_dir}"
    )

    # Setup code - start services with allocated ports and unique data directories
    nats_server = subprocess.Popen(
        ["nats-server", "-js", "-p", str(nats_port), "-sd", str(nats_data_dir)]
    )
    etcd = subprocess.Popen(
        [
            "etcd",
            "--data-dir",
            str(etcd_data_dir),
            "--listen-client-urls",
            f"http://localhost:{etcd_client_port}",
            "--advertise-client-urls",
            f"http://localhost:{etcd_client_port}",
            "--listen-peer-urls",
            f"http://localhost:{etcd_peer_port}",
            "--initial-advertise-peer-urls",
            f"http://localhost:{etcd_peer_port}",
            "--initial-cluster",
            f"default=http://localhost:{etcd_peer_port}",
        ]
    )

    return nats_server, etcd, nats_port, etcd_client_port, nats_data_dir, etcd_data_dir


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """
    Start NATS and ETCD for testing.

    Behavior is controlled by USE_PARALLEL_NATS_AND_ETCD constant:
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

        print("Services ready")
        yield
    finally:
        # Teardown code - always runs even if setup fails or tests error
        print("Tearing down resources")
        # Terminate both processes first (parallel shutdown)
        try:
            nats_server.terminate()
        except Exception as e:
            print(f"Error terminating NATS: {e}")
        try:
            etcd.terminate()
        except Exception as e:
            print(f"Error terminating ETCD: {e}")

        # Wait for both processes to finish
        try:
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
async def runtime():
    """
    Create a DistributedRuntime for testing.
    DistributedRuntime has singleton requirements, so tests using this fixture should be
    marked with `@pytest.mark.forked` to run in a separate process for isolation.
    """
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, True)
    yield runtime
    runtime.shutdown()
