# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kubernetes Job-based load generator for scale testing.

This module creates and manages a Kubernetes Job that runs the load generator
inside the cluster, allowing it to access ClusterIP services.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions

# Path to the standalone load generator script
SCRIPT_PATH = Path(__file__).parent / "load_generator_script.py"

logger = logging.getLogger(__name__)


class LoadGeneratorJob:
    """
    Manages a Kubernetes Job that runs load generation inside the cluster.
    
    This allows the load generator to access ClusterIP services that are
    not reachable from outside the cluster.
    """

    def __init__(
        self,
        namespace: str,
        frontend_urls: List[str],
        model: str,
        duration_sec: int,
        qps: float,
        max_tokens: int = 30,
        image: str = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1",
        job_name: str = "scale-test-load-generator",
        num_pods: int = 1,
        num_processes_per_pod: int = 1,
    ):
        """
        Initialize the load generator job.

        Args:
            namespace: Kubernetes namespace to run the job in
            frontend_urls: List of frontend URLs to test
            model: Model name for requests
            duration_sec: Test duration in seconds
            qps: Queries per second (will be distributed across pods and processes)
            max_tokens: Maximum tokens per request
            image: Container image to use (must have Python and dependencies)
            job_name: Name for the Kubernetes Job
            num_pods: Number of parallel pods to run (for high QPS scaling)
            num_processes_per_pod: Number of Python processes per pod (to bypass GIL/asyncio limits)
        """
        self.namespace = namespace
        self.frontend_urls = frontend_urls
        self.model = model
        self.duration_sec = duration_sec
        self.qps = qps
        self.max_tokens = max_tokens
        self.image = image
        self.job_name = job_name
        self.num_pods = num_pods
        self.num_processes_per_pod = num_processes_per_pod

        self._batch_api: Optional[client.BatchV1Api] = None
        self._core_api: Optional[client.CoreV1Api] = None

    def _load_script(self) -> str:
        """Load the Python script from the external file."""
        if not SCRIPT_PATH.exists():
            raise FileNotFoundError(
                f"Load generator script not found at {SCRIPT_PATH}. "
                "Ensure load_generator_script.py exists in the same directory."
            )
        return SCRIPT_PATH.read_text()

    def _get_env_vars(self) -> list:
        """Build environment variables for the Job container."""
        qps_per_pod = self.qps / self.num_pods

        return [
            {"name": "LOAD_GEN_URLS", "value": json.dumps(self.frontend_urls)},
            {"name": "LOAD_GEN_MODEL", "value": self.model},
            {"name": "LOAD_GEN_DURATION", "value": str(self.duration_sec)},
            {"name": "LOAD_GEN_QPS_PER_POD", "value": str(qps_per_pod)},
            {"name": "LOAD_GEN_MAX_TOKENS", "value": str(self.max_tokens)},
            {"name": "LOAD_GEN_NUM_PROCESSES", "value": str(self.num_processes_per_pod)},
            {"name": "LOAD_GEN_TOTAL_PODS", "value": str(self.num_pods)},
        ]

    def _build_configmap_manifest(self) -> dict:
        """Build ConfigMap containing the load generator script."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.job_name}-script",
                "namespace": self.namespace,
                "labels": {
                    "app": "scale-test-load-generator",
                    "app.kubernetes.io/managed-by": "scale-test",
                },
            },
            "data": {
                "load_generator.py": self._load_script(),
            },
        }

    def _build_job_manifest(self) -> dict:
        """Build the Kubernetes Job manifest with optional parallelism."""
        # Calculate resource requests based on processes per pod
        cpu_request = str(max(1, self.num_processes_per_pod))
        cpu_limit = str(max(2, self.num_processes_per_pod * 2))
        memory_request = f"{max(1, self.num_processes_per_pod)}Gi"
        memory_limit = f"{max(2, self.num_processes_per_pod * 2)}Gi"
        
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "scale-test-load-generator",
                    "app.kubernetes.io/managed-by": "scale-test",
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,  # Auto-cleanup after 5 minutes
                "backoffLimit": 0,  # Don't retry on failure
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "scale-test-load-generator",
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "volumes": [
                            {
                                "name": "script-volume",
                                "configMap": {
                                    "name": f"{self.job_name}-script",
                                    "defaultMode": 0o755,
                                },
                            }
                        ],
                        "containers": [
                            {
                                "name": "load-generator",
                                "image": self.image,
                                "command": [
                                    "bash",
                                    "-c",
                                    "pip install -q openai && python3 /scripts/load_generator.py",
                                ],
                                "env": self._get_env_vars(),
                                "volumeMounts": [
                                    {
                                        "name": "script-volume",
                                        "mountPath": "/scripts",
                                        "readOnly": True,
                                    }
                                ],
                                "resources": {
                                    "requests": {"cpu": cpu_request, "memory": memory_request},
                                    "limits": {"cpu": cpu_limit, "memory": memory_limit},
                                },
                            }
                        ],
                    },
                },
            },
        }
        
        # Add parallelism and completions if using multiple pods
        if self.num_pods > 1:
            manifest["spec"]["parallelism"] = self.num_pods
            manifest["spec"]["completions"] = self.num_pods
            manifest["spec"]["completionMode"] = "Indexed"  # Gives each pod a unique index
        
        return manifest

    async def create_and_wait(
        self, batch_api: client.BatchV1Api, core_api: client.CoreV1Api, timeout: int = 600
    ) -> bool:
        """
        Create the Job and wait for it to complete.

        Args:
            batch_api: Kubernetes BatchV1Api client
            core_api: Kubernetes CoreV1Api client
            timeout: Maximum time to wait for job completion in seconds

        Returns:
            True if job completed successfully, False otherwise
        """
        self._batch_api = batch_api
        self._core_api = core_api

        # Create the ConfigMap with the script
        logger.info(f"Creating ConfigMap for load generator script")
        configmap_manifest = self._build_configmap_manifest()
        
        try:
            await self._core_api.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_manifest
            )
            logger.info(f"ConfigMap {self.job_name}-script created")
        except exceptions.ApiException as e:
            if e.status == 409:
                logger.warning(f"ConfigMap already exists, deleting and recreating...")
                await self.delete()
                await asyncio.sleep(2)
                await self._core_api.create_namespaced_config_map(
                    namespace=self.namespace, body=configmap_manifest
                )
            else:
                logger.error(f"Failed to create ConfigMap: {e}")
                return False

        # Create the job
        logger.info(f"Creating load generator job: {self.job_name}")
        job_manifest = self._build_job_manifest()

        try:
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_manifest
            )
            logger.info(f"Job {self.job_name} created successfully")
        except exceptions.ApiException as e:
            if e.status == 409:
                logger.warning(f"Job {self.job_name} already exists, deleting and recreating...")
                await self.delete()
                await asyncio.sleep(2)
                await self._core_api.create_namespaced_config_map(
                    namespace=self.namespace, body=configmap_manifest
                )
                await self._batch_api.create_namespaced_job(
                    namespace=self.namespace, body=job_manifest
                )
            else:
                logger.error(f"Failed to create job: {e}")
                await self._delete_configmap()
                return False

        # Wait for job to complete
        return await self._wait_for_completion(timeout)

    async def _wait_for_completion(self, timeout: int) -> bool:
        """Wait for the job to complete and print logs from all pods."""
        start_time = time.time()
        pod_names = []
        logged_pods = set()

        logger.info(f"Waiting for job {self.job_name} to complete (timeout: {timeout}s)...")
        if self.num_pods > 1:
            logger.info(f"Job will spawn {self.num_pods} parallel pods...")

        while time.time() - start_time < timeout:
            try:
                # Check job status
                job = await self._batch_api.read_namespaced_job_status(
                    name=self.job_name, namespace=self.namespace
                )

                # Get all pod names
                pods = await self._core_api.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                
                current_pod_names = [pod.metadata.name for pod in pods.items]
                
                # Log newly discovered pods
                for pod_name in current_pod_names:
                    if pod_name not in logged_pods:
                        logger.info(f"Job pod started: {pod_name}")
                        logged_pods.add(pod_name)
                
                pod_names = current_pod_names

                # Check if job completed
                if job.status.succeeded:
                    expected_successes = self.num_pods
                    if job.status.succeeded >= expected_successes:
                        logger.info(f"Load generator job completed successfully! ({job.status.succeeded}/{expected_successes} pods)")
                        await self._print_all_logs(pod_names)
                        return True
                elif job.status.failed:
                    logger.error(f"Load generator job failed! (failed pods: {job.status.failed})")
                    await self._print_all_logs(pod_names)
                    return False

                await asyncio.sleep(2)

            except exceptions.ApiException as e:
                logger.debug(f"Error checking job status: {e}")
                await asyncio.sleep(2)

        logger.error(f"Timeout waiting for job {self.job_name} to complete")
        if pod_names:
            await self._print_all_logs(pod_names)
        return False

    async def _print_logs(self, pod_name: str) -> None:
        """Print logs from a single pod."""
        try:
            logs = await self._core_api.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace
            )
            print("\n" + "=" * 70)
            print("LOAD GENERATOR JOB LOGS")
            print("=" * 70)
            print(logs)
            print("=" * 70 + "\n")
        except exceptions.ApiException as e:
            logger.error(f"Failed to get pod logs: {e}")
    
    async def _print_all_logs(self, pod_names: List[str]) -> None:
        """Print logs from all pods and aggregate results."""
        if not pod_names:
            logger.warning("No pods found to print logs from")
            return
        
        print("\n" + "=" * 70)
        print(f"LOAD GENERATOR JOB LOGS ({len(pod_names)} pod(s))")
        print("=" * 70)
        
        # Collect logs from all pods
        all_logs = []
        for pod_name in sorted(pod_names):
            try:
                logs = await self._core_api.read_namespaced_pod_log(
                    name=pod_name, namespace=self.namespace
                )
                all_logs.append((pod_name, logs))
            except exceptions.ApiException as e:
                logger.error(f"Failed to get logs from pod {pod_name}: {e}")
        
        # Print individual pod logs
        for pod_name, logs in all_logs:
            print(f"\n--- Pod: {pod_name} ---")
            print(logs)
            print("-" * 70)
        
        # If multiple pods, print aggregated summary
        if len(all_logs) > 1:
            self._print_aggregated_summary(all_logs)
        
        print("=" * 70 + "\n")
    
    def _print_aggregated_summary(self, all_logs: List[tuple]) -> None:
        """Parse and aggregate statistics from all pod logs."""
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS (All Pods)")
        print("=" * 70)
        
        total_requests = 0
        total_errors = 0
        total_duration = 0
        all_latencies = []
        
        for pod_name, logs in all_logs:
            # Parse key metrics from logs (simple text parsing)
            for line in logs.split('\n'):
                if "Total requests (this pod):" in line:
                    try:
                        total_requests += int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Total errors:" in line:
                    try:
                        total_errors += int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "Actual duration:" in line:
                    try:
                        duration_str = line.split(":")[-1].strip().rstrip("s")
                        total_duration = max(total_duration, float(duration_str))
                    except ValueError:
                        pass
        
        print(f"\nTotal requests (all pods): {total_requests}")
        print(f"Total errors (all pods): {total_errors}")
        print(f"Duration: {total_duration:.1f}s")
        
        if total_duration > 0:
            actual_qps = total_requests / total_duration
            target_qps = self.qps
            print(f"Target QPS: {target_qps:.1f}")
            print(f"Actual QPS: {actual_qps:.1f}")
            print(f"QPS achievement: {(actual_qps/target_qps*100):.1f}%")
        
        if total_requests > 0:
            error_rate = (total_errors / total_requests) * 100
            print(f"Overall error rate: {error_rate:.1f}%")
        
        print("=" * 70)

    async def _delete_configmap(self) -> None:
        """Delete the ConfigMap."""
        if self._core_api is None:
            return

        try:
            await self._core_api.delete_namespaced_config_map(
                name=f"{self.job_name}-script",
                namespace=self.namespace,
            )
            logger.info(f"ConfigMap {self.job_name}-script deleted")
        except exceptions.ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete ConfigMap: {e}")

    async def delete(self) -> None:
        """Delete the job and ConfigMap."""
        if self._batch_api is None:
            return

        # Delete Job
        try:
            logger.info(f"Deleting job {self.job_name}")
            await self._batch_api.delete_namespaced_job(
                name=self.job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
            logger.info(f"Job {self.job_name} deleted")
        except exceptions.ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete job: {e}")

        # Delete ConfigMap
        await self._delete_configmap()

