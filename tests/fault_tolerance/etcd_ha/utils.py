# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import tempfile
import time
from typing import List, Optional

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend with ETCD HA support"""

    def __init__(self, request, etcd_endpoints: list):
        command = ["python", "-m", "dynamo.frontend"]

        # Set debug logging and ETCD endpoints
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["ETCD_ENDPOINTS"] = ",".join(etcd_endpoints)

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class EtcdReplicaServer(ManagedProcess):
    """Single ETCD replica server in a cluster"""

    def __init__(
        self,
        request,
        name: str,
        client_port: int,
        peer_port: int,
        initial_cluster: str,
        data_dir: str,
        log_dir: str,
        timeout: int = 30,
    ):
        self.name = name
        self.client_port = client_port
        self.peer_port = peer_port
        self.data_dir = data_dir

        etcd_env = os.environ.copy()
        etcd_env["ETCD_ENDPOINTS"] = ""  # Clear any inherited ETCD endpoints
        etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"

        command = [
            "etcd",
            "--name",
            name,
            "--data-dir",
            data_dir,
            "--listen-client-urls",
            f"http://0.0.0.0:{client_port}",
            "--advertise-client-urls",
            f"http://localhost:{client_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{peer_port}",
            "--initial-advertise-peer-urls",
            f"http://localhost:{peer_port}",
            "--initial-cluster",
            initial_cluster,
            "--initial-cluster-state",
            "new",
            "--initial-cluster-token",
            "etcd-cluster",
        ]

        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_existing=False,
            data_dir=data_dir,
            log_dir=log_dir,
        )

    def get_status(self) -> dict:
        """Get the status of this ETCD node"""
        try:
            response = requests.post(
                f"http://localhost:{self.client_port}/v3/maintenance/status",
                json={},
                timeout=2,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get status for {self.name}: {e}")
        return {}

    def is_leader(self) -> bool:
        """Check if this node is the current leader"""
        status = self.get_status()
        # In etcd v3 API, we check if this member ID matches the leader ID
        if status:
            member_id = status.get("header", {}).get("member_id", "")
            leader_id = status.get("leader", "")
            return member_id == leader_id
        return False


class EtcdCluster:
    """Manager for an ETCD cluster with configurable number of replicas"""

    def __init__(
        self,
        request,
        num_replicas: int = 3,
        base_client_port: int = 2379,
        base_peer_port: int = 12380,
    ):
        self.request = request
        self.num_replicas = num_replicas
        self.base_client_port = base_client_port
        self.base_peer_port = base_peer_port
        self.replicas: List[Optional[EtcdReplicaServer]] = []
        self.data_dirs: List[str] = []
        self.log_base_dir = f"{request.node.name}_etcd_cluster"

        # Clean up any existing log directory
        try:
            shutil.rmtree(self.log_base_dir)
            logger.info(f"Cleaned up existing log directory: {self.log_base_dir}")
        except FileNotFoundError:
            pass

        os.makedirs(self.log_base_dir, exist_ok=True)

    def start(self):
        """Start ETCD cluster with configured number of replicas"""
        logger.info(f"Starting {self.num_replicas}-node ETCD cluster")

        # Build initial cluster configuration
        initial_cluster_parts = []
        for i in range(self.num_replicas):
            name = f"etcd-{i}"
            peer_port = self.base_peer_port + i
            initial_cluster_parts.append(f"{name}=http://localhost:{peer_port}")

        initial_cluster = ",".join(initial_cluster_parts)

        # Start each replica
        for i in range(self.num_replicas):
            name = f"etcd-{i}"
            client_port = self.base_client_port + i
            peer_port = self.base_peer_port + i
            data_dir = tempfile.mkdtemp(prefix=f"etcd_{i}_")
            log_dir = os.path.join(self.log_base_dir, name)

            self.data_dirs.append(data_dir)
            os.makedirs(log_dir, exist_ok=True)

            logger.info(
                f"Starting {name} on client port {client_port}, peer port {peer_port}"
            )

            replica = EtcdReplicaServer(
                request=self.request,
                name=name,
                client_port=client_port,
                peer_port=peer_port,
                initial_cluster=initial_cluster,
                data_dir=data_dir,
                log_dir=log_dir,
            )

            replica.__enter__()
            self.replicas.append(replica)

        logger.info(f"All {self.num_replicas} ETCD replicas started successfully")

        # Wait for cluster to stabilize and elect a leader
        self._wait_for_healthy_cluster(timeout=30)

        leader_idx = self.find_leader()
        if leader_idx is not None:
            logger.info(f"Initial leader elected: etcd-{leader_idx}")
        else:
            logger.warning("No leader elected yet")

    def _wait_for_healthy_cluster(self, timeout: int = 30):
        """Wait for all replicas to be healthy and responsive.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If cluster doesn't become healthy within timeout
        """
        logger.info("Waiting for all replicas to be healthy...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            time.sleep(1)

            # Check if all replicas are responding
            all_healthy = True
            for i, replica in enumerate(self.replicas):
                if replica:
                    status = replica.get_status()
                    if not status:
                        logger.debug(f"etcd-{i} not yet responsive")
                        all_healthy = False
                        break

            if all_healthy:
                logger.info("All replicas are healthy")
                return

        raise RuntimeError(f"ETCD cluster failed to become healthy within {timeout}s")

    def find_leader(self) -> Optional[int]:
        """Find which replica is currently the leader"""
        for i, replica in enumerate(self.replicas):
            if replica and replica.is_leader():
                return i
        return None

    def terminate_leader(self) -> Optional[int]:
        """Terminate the current leader and return its index"""
        leader_idx = self.find_leader()

        if leader_idx is None:
            logger.warning("No leader found to terminate")
            return None

        logger.info(f"Terminating current leader: etcd-{leader_idx}")
        replica = self.replicas[leader_idx]

        if replica:
            replica.__exit__(None, None, None)
            self.replicas[leader_idx] = None
            logger.info(f"Leader etcd-{leader_idx} has been terminated")

        return leader_idx

    def get_client_endpoints(self) -> List[str]:
        """Get list of active client endpoints"""
        endpoints = []
        for i, replica in enumerate(self.replicas):
            if replica:  # Only include active replicas
                client_port = self.base_client_port + i
                endpoints.append(f"http://localhost:{client_port}")
        return endpoints

    def stop(self):
        """Clean up all replicas and temporary directories"""
        logger.info("Cleaning up ETCD cluster")

        # Stop all running replicas
        for replica in self.replicas:
            if replica:
                try:
                    replica.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error stopping replica: {e}")
        self.replicas = []

        # Clean up data directories
        for data_dir in self.data_dirs:
            try:
                shutil.rmtree(data_dir)
            except Exception as e:
                logger.warning(f"Error removing data directory {data_dir}: {e}")
        self.data_dirs = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def send_inference_request(prompt: str, max_tokens: int = 50) -> str:
    """Send a simple inference request to the frontend and return the generated text"""
    payload = {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Make output deterministic
    }

    headers = {"Content-Type": "application/json"}

    logger.info(f"Sending inference request: '{prompt}'")
    try:
        response = requests.post(
            f"http://localhost:{FRONTEND_PORT}/v1/completions",
            headers=headers,
            json=payload,
            timeout=round(max_tokens * 0.6),
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            logger.info(f"Inference generated text: '{text.strip()}'")
            return text
        else:
            pytest.fail(
                f"Inference request failed with code {response.status_code}: {response.text}"
            )
    except Exception as e:
        pytest.fail(f"Inference request failed: {e}")


def wait_for_processes_to_terminate(
    processes: dict, timeout: int = 30, poll_interval: int = 1
) -> None:
    """
    Wait for multiple processes to terminate and fail if they don't within timeout.

    Args:
        processes: Dictionary mapping process names to ManagedProcess instances
        timeout: Maximum time to wait in seconds
        poll_interval: Time between checks in seconds

    Raises:
        pytest.fail: If any process is still running after timeout
    """
    logger.info(f"Waiting for {len(processes)} process(es) to terminate")
    elapsed = 0
    terminated = {name: False for name in processes}

    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        # Check each process
        for name, process in processes.items():
            if (
                not terminated[name]
                and process.proc
                and process.proc.poll() is not None
            ):
                logger.info(f"{name} process has terminated after {elapsed}s")
                terminated[name] = True

        # Exit early if all processes have terminated
        if all(terminated.values()):
            return

    # Check for any processes still running and fail
    still_running = [name for name, term in terminated.items() if not term]
    if still_running:
        pytest.fail(
            f"Process(es) still running after {elapsed}s: {', '.join(still_running)}"
        )
