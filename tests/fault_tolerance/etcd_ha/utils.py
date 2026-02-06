# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional

import pytest
import requests

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import FRONTEND_PORT
from tests.utils.managed_process import (
    DynamoFrontendProcess as BaseDynamoFrontendProcess,
)
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(BaseDynamoFrontendProcess):
    """Process manager for Dynamo frontend with ETCD HA support."""

    def __init__(self, request, etcd_endpoints: list[str]):
        extra_env = {
            "DYN_LOG": "debug",
            "ETCD_ENDPOINTS": ",".join(etcd_endpoints),
        }
        # WARNING: terminate_all_matching_process_names=True is NOT pytest-xdist safe!
        # DANGER: Kills ALL dynamo-frontend processes system-wide, including other parallel tests.
        # For parallel-safe alternative, use terminate_all_matching_process_names=False.
        # See tests/kvbm_integration/common.py:llm_server_kvbm for example.
        # TODO: Switch to terminate_all_matching_process_names=False with dynamic ports
        super().__init__(
            request,
            router_mode="round-robin",
            extra_env=extra_env,
            terminate_all_matching_process_names=True,  # TODO: Change to False
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
        cluster_state: str = "new",
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
            f"http://127.0.0.1:{client_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{peer_port}",
            "--initial-advertise-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-cluster",
            initial_cluster,
            "--initial-cluster-state",
            cluster_state,
            "--initial-cluster-token",
            "etcd-cluster",
        ]

        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=False,
            data_dir=data_dir,
            log_dir=log_dir,
        )

    def get_status(self) -> dict:
        """Get the status of this ETCD node"""
        try:
            response = requests.post(
                f"http://127.0.0.1:{self.client_port}/v3/maintenance/status",
                json={},
                timeout=2,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get status for {self.name}: {e}")
        return {}

    def is_leader(self) -> Optional[bool]:
        """
        Check if this node is the current leader.

        Returns: True/False on is leader or None if status cannot be retrieved.
        """
        status = self.get_status()
        # In etcd v3 API, we check if this member ID matches the leader ID
        if status:
            member_id = status.get("header", {}).get("member_id", "")
            leader_id = status.get("leader", "")
            return member_id == leader_id
        return None


class EtcdCluster:
    """Manager for an ETCD cluster with configurable number of replicas"""

    def __init__(
        self,
        request,
        num_replicas: int = 3,
        base_port: int = 2379,
    ):
        self.request = request
        self.num_replicas = num_replicas
        self.base_port = base_port
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

    def _get_initial_cluster(self) -> str:
        """Build the initial cluster configuration string"""
        initial_cluster_parts = []
        for i in range(self.num_replicas):
            name = f"etcd-{i}"
            peer_port = self.base_port + (2 * i) + 1
            initial_cluster_parts.append(f"{name}=http://127.0.0.1:{peer_port}")
        return ",".join(initial_cluster_parts)

    def _start_replica(self, idx: int, cluster_state: str = "new") -> EtcdReplicaServer:
        """Start a single ETCD replica"""
        name = f"etcd-{idx}"
        # e.g. base_port = 2379 -> client_port = 2379, 2381, 2383
        # e.g. base_port = 2379 -> peer_port = 2380, 2382, 2384
        client_port = self.base_port + (2 * idx)
        peer_port = self.base_port + (2 * idx) + 1

        # Create data dir for the node
        data_dir = tempfile.mkdtemp(prefix=f"etcd_{idx}_")
        if idx < len(self.data_dirs):
            self.data_dirs[idx] = data_dir
        else:
            self.data_dirs.append(data_dir)

        log_dir = os.path.join(self.log_base_dir, name)
        os.makedirs(log_dir, exist_ok=True)

        logger.info(
            f"Starting {name} on client port {client_port}, peer port {peer_port}"
        )

        replica = EtcdReplicaServer(
            request=self.request,
            name=name,
            client_port=client_port,
            peer_port=peer_port,
            initial_cluster=self._get_initial_cluster(),
            data_dir=data_dir,
            log_dir=log_dir,
            cluster_state=cluster_state,
        )

        replica.__enter__()
        return replica

    def _wait_for_healthy_cluster(self, timeout: int = 30):
        """Wait for cluster to be healthy and elected leader."""
        logger.info("Waiting for cluster to become healthy...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if a leader is elected indicating cluster health
            is_healthy = True
            leader_id = None
            for i, replica in enumerate(self.replicas):
                if replica:
                    is_leader = replica.is_leader()
                    if is_leader is None:
                        is_healthy = False
                        break
                    if is_leader is True:
                        if leader_id is not None:
                            raise RuntimeError(
                                f"Multiple leaders detected in ETCD cluster etcd-{leader_id} and etcd-{i}"
                            )
                        leader_id = i

            if is_healthy and leader_id is not None:
                logger.info(f"Cluster is healthy with leader at etcd-{leader_id}")
                return

            time.sleep(1)

        raise RuntimeError(f"ETCD cluster failed to become healthy within {timeout}s")

    def _replace_member(self, idx: int):
        """Remove old member and add new member to the cluster using etcdctl"""
        # Find a healthy replica to perform member operations
        healthy_replica = None
        for i, r in enumerate(self.replicas):
            if r and i != idx:
                healthy_replica = r
                break

        if not healthy_replica:
            raise RuntimeError("No healthy replica found to perform member operations")

        name = f"etcd-{idx}"
        peer_port = self.base_port + (2 * idx) + 1
        peer_url = f"http://127.0.0.1:{peer_port}"

        # Set ETCDCTL_ENDPOINTS for etcdctl commands
        etcdctl_env = os.environ.copy()
        etcdctl_env[
            "ETCDCTL_ENDPOINTS"
        ] = f"http://127.0.0.1:{healthy_replica.client_port}"
        etcdctl_env["ETCDCTL_API"] = "3"

        # First, get member list to find the old member's ID
        logger.info(f"Getting member list to find {name}")
        try:
            result = subprocess.run(
                ["etcdctl", "member", "list", "--write-out=json"],
                env=etcdctl_env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                members = json.loads(result.stdout).get("members", [])
                old_member_id = None
                for member in members:
                    if member.get("name") == name:
                        old_member_id = member.get("ID")
                        break

                if old_member_id:
                    # Convert member ID to hex format (etcdctl expects hex)
                    hex_member_id = format(int(old_member_id), "x")
                    logger.info(
                        f"Removing member with ID {old_member_id} (hex: {hex_member_id})"
                    )
                    remove_result = subprocess.run(
                        ["etcdctl", "member", "remove", hex_member_id],
                        env=etcdctl_env,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if remove_result.returncode != 0:
                        raise RuntimeError(
                            f"Failed to remove old member: {remove_result.stderr}"
                        )
                    logger.info(f"Successfully removed old member {name}")
        except Exception as e:
            raise RuntimeError(f"Error during member removal: {e}")

        # Add the new member to the cluster
        logger.info(f"Adding new member {name} to cluster with peer URL {peer_url}")
        try:
            add_result = subprocess.run(
                ["etcdctl", "member", "add", name, f"--peer-urls={peer_url}"],
                env=etcdctl_env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if add_result.returncode != 0:
                raise RuntimeError(f"Failed to add new member: {add_result.stderr}")
            logger.info(f"Successfully added new member {name}")
        except Exception as e:
            raise RuntimeError(f"Error adding new member: {e}")

    def start(self):
        """Start ETCD cluster with configured number of replicas"""
        logger.info(f"Starting {self.num_replicas}-node ETCD cluster")

        # Start each replica
        for i in range(self.num_replicas):
            replica = self._start_replica(i, cluster_state="new")
            self.replicas.append(replica)

        logger.info(f"All {self.num_replicas} ETCD replicas started successfully")

        # Wait for cluster to stabilize
        self._wait_for_healthy_cluster()

    def get_client_endpoints(self) -> List[str]:
        """Get list of active client endpoints"""
        endpoints = []
        for i, replica in enumerate(self.replicas):
            if replica:  # Only include active replicas
                client_port = self.base_port + (2 * i)
                endpoints.append(f"http://127.0.0.1:{client_port}")
        return endpoints

    def terminate_replica(self, idx: int):
        """Terminate a specific replica by index."""
        if idx < 0 or idx >= self.num_replicas:
            raise RuntimeError(f"Invalid replica index: {idx}")

        replica = self.replicas[idx]
        if not replica:
            raise RuntimeError(f"Replica etcd-{idx} is already terminated")

        replica.__exit__(None, None, None)
        self.replicas[idx] = None

        logger.info(f"Terminated replica etcd-{idx}")

    def restart_replica(self, idx: int):
        """Restart a terminated replica"""
        if idx < 0 or idx >= self.num_replicas:
            raise RuntimeError(f"Invalid replica index: {idx}")

        if self.replicas[idx] is not None:
            raise RuntimeError(f"Replica etcd-{idx} is already running")

        # Make sure the cluster is healthy before restarting
        self._wait_for_healthy_cluster()

        # Remove old member and add new member
        self._replace_member(idx)

        # Start the replica with existing cluster state
        replica = self._start_replica(idx, cluster_state="existing")
        self.replicas[idx] = replica

        # Wait for cluster to stabilize
        self._wait_for_healthy_cluster()

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
                f"[ETCD HA regression?] Inference request failed with code {response.status_code}: {response.text}"
            )
    except Exception as e:
        pytest.fail(f"[ETCD HA regression?] Inference request failed: {e}")


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
