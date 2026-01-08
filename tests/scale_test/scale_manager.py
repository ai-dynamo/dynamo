# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core process management for scale testing.

This module manages the lifecycle of NATS, etcd, mocker, and frontend processes
for scale testing Dynamo deployments.
"""

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from tests.scale_test.utils import (
    generate_namespace,
    wait_for_all_ready,
    wait_for_all_workers_registered,
)
from tests.utils.managed_process import ManagedProcess, terminate_process_tree
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)


@dataclass
class ScaleManager:
    """
    Manages the lifecycle of scale test deployments.

    Creates and manages NATS/etcd infrastructure along with N mocker and frontend
    process pairs, each isolated by unique namespaces.
    """

    num_deployments: int
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    speedup_ratio: float = 10.0
    base_frontend_port: int = 8001
    display_output: bool = False
    timeout: int = 300

    # Process tracking
    mocker_processes: List[ManagedProcess] = field(default_factory=list)
    frontend_processes: List[ManagedProcess] = field(default_factory=list)
    nats_process: Optional[subprocess.Popen] = None
    etcd_process: Optional[subprocess.Popen] = None

    # Port tracking for cleanup
    _allocated_ports: List[int] = field(default_factory=list)

    # Infrastructure details
    nats_port: int = 4222
    etcd_port: int = 2379
    _nats_data_dir: Optional[str] = None
    _etcd_data_dir: Optional[str] = None
    _log_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize logging and create log directory."""
        self._log_dir = tempfile.mkdtemp(prefix="scale_test_logs_")
        logger.info(f"Scale test logs will be written to: {self._log_dir}")

    def start_infrastructure(self) -> None:
        """
        Start shared NATS and etcd servers.

        Creates temporary data directories and starts both services
        on default ports.
        """
        logger.info("Starting shared NATS and etcd infrastructure...")

        # Create data directories
        self._nats_data_dir = tempfile.mkdtemp(prefix="scale_test_nats_")
        self._etcd_data_dir = tempfile.mkdtemp(prefix="scale_test_etcd_")

        # Start NATS server with JetStream enabled
        logger.info(f"Starting NATS server on port {self.nats_port}...")
        self.nats_process = subprocess.Popen(
            [
                "nats-server",
                "-js",
                "-p",
                str(self.nats_port),
                "-sd",
                str(self._nats_data_dir),
            ],
            stdout=subprocess.PIPE if not self.display_output else None,
            stderr=subprocess.STDOUT if not self.display_output else None,
        )
        logger.info(f"NATS started with PID {self.nats_process.pid}")

        # Start etcd server
        logger.info(f"Starting etcd server on port {self.etcd_port}...")
        self.etcd_process = subprocess.Popen(
            [
                "etcd",
                "--data-dir",
                str(self._etcd_data_dir),
                "--listen-client-urls",
                f"http://localhost:{self.etcd_port}",
                "--advertise-client-urls",
                f"http://localhost:{self.etcd_port}",
            ],
            stdout=subprocess.PIPE if not self.display_output else None,
            stderr=subprocess.STDOUT if not self.display_output else None,
        )
        logger.info(f"etcd started with PID {self.etcd_process.pid}")

        # Wait for services to be ready
        self._wait_for_port(self.nats_port, timeout=30)
        self._wait_for_port(self.etcd_port, timeout=30)

        logger.info("Infrastructure started successfully!")

    def start_mockers(self) -> None:
        """
        Launch N mocker processes with unique namespaces.

        Each mocker gets its own DYN_NAMESPACE environment variable and
        connects to the shared NATS/etcd infrastructure.
        """
        logger.info(f"Starting {self.num_deployments} mocker processes...")

        for i in range(1, self.num_deployments + 1):
            namespace = generate_namespace(i)
            logger.info(f"Starting mocker {i} (namespace: {namespace})...")

            env = os.environ.copy()
            env["DYN_NAMESPACE"] = namespace
            env["NATS_SERVER"] = f"nats://localhost:{self.nats_port}"
            env["ETCD_ENDPOINTS"] = f"http://localhost:{self.etcd_port}"
            # Unset DYN_SYSTEM_PORT to avoid conflicts
            env.pop("DYN_SYSTEM_PORT", None)

            command = [
                "python",
                "-m",
                "dynamo.mocker",
                "--model-path",
                self.model_path,
                "--speedup-ratio",
                str(self.speedup_ratio),
                "--num-workers",
                "1",
            ]

            mocker = ManagedProcess(
                command=command,
                env=env,
                display_output=self.display_output,
                timeout=self.timeout,
                log_dir=os.path.join(self._log_dir, f"mocker_{i}"),
                terminate_existing=False,  # Don't kill other mockers
            )

            # Start the process using context manager protocol
            mocker.__enter__()
            self.mocker_processes.append(mocker)
            logger.info(f"Mocker {i} started (namespace: {namespace})")

        logger.info(f"All {self.num_deployments} mockers started!")

    def start_frontends(self) -> None:
        """
        Launch N frontend processes on unique ports.

        Each frontend connects to the shared NATS/etcd infrastructure and
        uses its corresponding namespace to find its mocker.
        """
        logger.info(f"Starting {self.num_deployments} frontend processes...")

        for i in range(1, self.num_deployments + 1):
            namespace = generate_namespace(i)
            port = self.base_frontend_port + i - 1

            # Try to allocate the port, or use the next available
            try:
                allocated_port = allocate_port(port)
                self._allocated_ports.append(allocated_port)
                port = allocated_port
            except RuntimeError:
                logger.warning(f"Could not allocate port {port}, using allocated port")

            logger.info(
                f"Starting frontend {i} on port {port} (namespace: {namespace})..."
            )

            env = os.environ.copy()
            env["DYN_NAMESPACE"] = namespace
            env["NATS_SERVER"] = f"nats://localhost:{self.nats_port}"
            env["ETCD_ENDPOINTS"] = f"http://localhost:{self.etcd_port}"
            # Unset DYN_SYSTEM_PORT to avoid conflicts
            env.pop("DYN_SYSTEM_PORT", None)

            command = [
                "python",
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(port),
                "--router-mode",
                "round-robin",
            ]

            frontend = ManagedProcess(
                command=command,
                env=env,
                display_output=self.display_output,
                timeout=self.timeout,
                log_dir=os.path.join(self._log_dir, f"frontend_{i}"),
                health_check_ports=[port],
                terminate_existing=False,  # Don't kill other frontends
            )

            # Start the process using context manager protocol
            frontend.__enter__()
            self.frontend_processes.append(frontend)
            logger.info(f"Frontend {i} started on port {port}")

        logger.info(f"All {self.num_deployments} frontends started!")

    def get_frontend_urls(self) -> List[str]:
        """
        Return list of frontend URLs for load generation.

        Returns:
            List of URLs like ['http://localhost:8001', 'http://localhost:8002', ...]
        """
        urls = []
        for i in range(self.num_deployments):
            if i < len(self._allocated_ports):
                port = self._allocated_ports[i]
            else:
                port = self.base_frontend_port + i
            urls.append(f"http://localhost:{port}")
        return urls

    def wait_for_all_ready(self, timeout: float = 120.0) -> bool:
        """
        Wait for all frontends to be ready to accept requests.

        This performs a two-phase check for each frontend:
        1. Wait for the frontend HTTP server to respond to /health
        2. Wait for workers to register (model appears in /v1/models)
        3. Verify the completions pipeline works (successful POST to /v1/completions)

        This is more robust than just checking /health because the frontend health
        check can pass before workers have registered via NATS/etcd, causing 503 errors.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all frontends are ready and have workers, False otherwise
        """
        frontend_urls = self.get_frontend_urls()

        # Phase 1: Wait for frontend HTTP servers to be up (quick health check)
        health_urls = [f"{url}/health" for url in frontend_urls]
        logger.info(f"Phase 1: Waiting for {len(health_urls)} frontend HTTP servers...")
        if not wait_for_all_ready(health_urls, timeout=timeout / 2):
            logger.error("Frontend HTTP servers did not become ready")
            return False

        # Phase 2: Wait for workers to register and pipelines to be functional
        logger.info("Phase 2: Waiting for workers to register with frontends...")
        return wait_for_all_workers_registered(
            frontend_urls, self.model_path, timeout=timeout / 2
        )

    def cleanup(self) -> None:
        """
        Terminate all processes in reverse order and clean up resources.

        Terminates frontends first, then mockers, and finally infrastructure.
        """
        logger.info("Cleaning up all processes...")

        # Terminate frontends first
        for i, frontend in enumerate(reversed(self.frontend_processes)):
            try:
                logger.info(f"Terminating frontend {self.num_deployments - i}...")
                frontend.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error terminating frontend: {e}")

        self.frontend_processes.clear()

        # Then mockers
        for i, mocker in enumerate(reversed(self.mocker_processes)):
            try:
                logger.info(f"Terminating mocker {self.num_deployments - i}...")
                mocker.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error terminating mocker: {e}")

        self.mocker_processes.clear()

        # Finally infrastructure
        if self.nats_process:
            try:
                logger.info("Terminating NATS server...")
                terminate_process_tree(self.nats_process.pid, logger)
                self.nats_process = None
            except Exception as e:
                logger.warning(f"Error terminating NATS: {e}")

        if self.etcd_process:
            try:
                logger.info("Terminating etcd server...")
                terminate_process_tree(self.etcd_process.pid, logger)
                self.etcd_process = None
            except Exception as e:
                logger.warning(f"Error terminating etcd: {e}")

        # Deallocate ports
        for port in self._allocated_ports:
            try:
                deallocate_port(port)
            except Exception as e:
                logger.warning(f"Error deallocating port {port}: {e}")
        self._allocated_ports.clear()

        # Clean up data directories
        if self._nats_data_dir:
            try:
                shutil.rmtree(self._nats_data_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error removing NATS data dir: {e}")
            self._nats_data_dir = None

        if self._etcd_data_dir:
            try:
                shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error removing etcd data dir: {e}")
            self._etcd_data_dir = None

        logger.info("All processes terminated.")

    def _wait_for_port(self, port: int, timeout: float = 30.0) -> None:
        """
        Wait for a port to become available.

        Args:
            port: Port number to check
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the port doesn't become available within timeout
        """
        import socket

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                sock.close()
                if result == 0:
                    return
            except socket.error:
                pass
            time.sleep(0.5)

        raise RuntimeError(f"Port {port} did not become available within {timeout}s")

    def __enter__(self):
        """Context manager entry - start all services."""
        self.start_infrastructure()
        self.start_mockers()
        self.start_frontends()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all services."""
        self.cleanup()
        return False

    def start_all(self) -> None:
        """
        Convenience method to start all components.

        Starts infrastructure, mockers, and frontends in order.
        """
        self.start_infrastructure()
        self.start_mockers()
        self.start_frontends()


def setup_signal_handlers(manager: ScaleManager) -> None:
    """
    Set up signal handlers for graceful cleanup on Ctrl+C.

    Args:
        manager: The ScaleManager instance to clean up on signal
    """

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        manager.cleanup()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
