# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions

SCRIPT_PATH = Path(__file__).parent / "load_generator_script.py"
LOAD_GENERATOR_IMAGE = "python:3.11-slim"

logger = logging.getLogger(__name__)


class LoadGeneratorJob:
    """Manages a Kubernetes Job that runs load generation inside the cluster."""

    def __init__(
        self,
        namespace: str,
        frontend_urls: List[str],
        model: str,
        duration_sec: int,
        qps: float,
        max_tokens: int = 30,
        job_name: str = "scale-test-load-generator",
        num_pods: int = 1,
        num_processes_per_pod: int = 1,
        log_responses: bool = True,
        log_sample_rate: float = 1.0,
    ):
        self.namespace = namespace
        self.frontend_urls = frontend_urls
        self.model = model
        self.duration_sec = duration_sec
        self.qps = qps
        self.max_tokens = max_tokens
        self.job_name = job_name
        self.num_pods = num_pods
        self.num_processes_per_pod = num_processes_per_pod
        self.log_responses = log_responses
        self.log_sample_rate = max(0.0, min(1.0, log_sample_rate))
        if self.num_pods < 1:
            raise ValueError("num_pods must be at least 1")
        if self.num_processes_per_pod < 1:
            raise ValueError("num_processes_per_pod must be at least 1")

        self._batch_api: Optional[client.BatchV1Api] = None
        self._core_api: Optional[client.CoreV1Api] = None

    def _load_script(self) -> str:
        if not SCRIPT_PATH.exists():
            raise FileNotFoundError(f"Load generator script not found at {SCRIPT_PATH}")
        return SCRIPT_PATH.read_text()

    def _get_env_vars(self) -> list:
        qps_per_pod = self.qps / self.num_pods
        return [
            {"name": "LOAD_GEN_URLS", "value": json.dumps(self.frontend_urls)},
            {"name": "LOAD_GEN_MODEL", "value": self.model},
            {"name": "LOAD_GEN_DURATION", "value": str(self.duration_sec)},
            {"name": "LOAD_GEN_QPS_PER_POD", "value": str(qps_per_pod)},
            {"name": "LOAD_GEN_MAX_TOKENS", "value": str(self.max_tokens)},
            {
                "name": "LOAD_GEN_NUM_PROCESSES",
                "value": str(self.num_processes_per_pod),
            },
            {"name": "LOAD_GEN_TOTAL_PODS", "value": str(self.num_pods)},
            {"name": "LOAD_GEN_LOG_RESPONSES", "value": "1" if self.log_responses else "0"},
            {"name": "LOAD_GEN_LOG_SAMPLE_RATE", "value": str(self.log_sample_rate)},
        ]

    def _build_configmap_manifest(self) -> dict:
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.job_name}-script",
                "namespace": self.namespace,
                "labels": {"app": "scale-test-load-generator"},
            },
            "data": {"load_generator.py": self._load_script()},
        }

    def _build_job_manifest(self) -> dict:
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
                "labels": {"app": "scale-test-load-generator"},
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"app": "scale-test-load-generator"}},
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
                                "image": LOAD_GENERATOR_IMAGE,
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
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request,
                                    },
                                    "limits": {
                                        "cpu": cpu_limit,
                                        "memory": memory_limit,
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        }

        if self.num_pods > 1:
            manifest["spec"]["parallelism"] = self.num_pods
            manifest["spec"]["completions"] = self.num_pods
            manifest["spec"]["completionMode"] = "Indexed"

        return manifest

    async def create_and_wait(
        self,
        batch_api: client.BatchV1Api,
        core_api: client.CoreV1Api,
        timeout: int = 600,
    ) -> bool:
        self._batch_api = batch_api
        self._core_api = core_api

        logger.info("Creating ConfigMap for load generator script")
        configmap_manifest = self._build_configmap_manifest()

        try:
            await self._core_api.create_namespaced_config_map(
                namespace=self.namespace, body=configmap_manifest
            )
        except exceptions.ApiException as e:
            if e.status == 409:
                logger.warning("ConfigMap already exists, recreating...")
                await self._delete_configmap()
                await asyncio.sleep(2)
                await self._core_api.create_namespaced_config_map(
                    namespace=self.namespace, body=configmap_manifest
                )
            else:
                logger.error(f"Failed to create ConfigMap: {e}")
                return False

        logger.info(f"Creating job: {self.job_name}")
        job_manifest = self._build_job_manifest()

        try:
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_manifest
            )
        except exceptions.ApiException as e:
            if e.status == 409:
                logger.warning("Job already exists, recreating...")
                await self._delete_job()
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

        return await self._wait_for_completion(timeout)

    async def _wait_for_completion(self, timeout: int) -> bool:
        start_time = time.time()
        pod_names = []
        logged_pods = set()

        logger.info(f"Waiting for job to complete (timeout: {timeout}s)...")

        while time.time() - start_time < timeout:
            try:
                job = await self._batch_api.read_namespaced_job_status(
                    name=self.job_name, namespace=self.namespace
                )

                pods = await self._core_api.list_namespaced_pod(
                    namespace=self.namespace, label_selector=f"job-name={self.job_name}"
                )

                current_pod_names = [pod.metadata.name for pod in pods.items]

                for pod_name in current_pod_names:
                    if pod_name not in logged_pods:
                        logger.info(f"Pod started: {pod_name}")
                        logged_pods.add(pod_name)

                pod_names = current_pod_names

                if job.status.succeeded:
                    if job.status.succeeded >= self.num_pods:
                        logger.info(
                            f"Job completed ({job.status.succeeded}/{self.num_pods} pods)"
                        )
                        await self._print_all_logs(pod_names)
                        return True
                elif job.status.failed:
                    logger.error(f"Job failed ({job.status.failed} pods)")
                    await self._print_all_logs(pod_names)
                    return False

                await asyncio.sleep(2)

            except exceptions.ApiException:
                await asyncio.sleep(2)

        logger.error("Timeout waiting for job")
        if pod_names:
            await self._print_all_logs(pod_names)
        return False

    async def _print_all_logs(self, pod_names: List[str]) -> None:
        if not pod_names:
            return

        print("\n" + "=" * 70)
        print(f"LOAD GENERATOR LOGS ({len(pod_names)} pod(s))")
        print("=" * 70)

        all_logs = []
        for pod_name in sorted(pod_names):
            try:
                logs = await self._core_api.read_namespaced_pod_log(
                    name=pod_name, namespace=self.namespace
                )
                all_logs.append((pod_name, logs))
            except exceptions.ApiException as e:
                logger.error(f"Failed to get logs from {pod_name}: {e}")

        for pod_name, logs in all_logs:
            print(f"\n--- Pod: {pod_name} ---")
            print(logs)
            print("-" * 70)

        if len(all_logs) > 1:
            self._print_aggregated_summary(all_logs)

        print("=" * 70 + "\n")

    def _print_aggregated_summary(self, all_logs: List[tuple]) -> None:
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS")
        print("=" * 70)

        total_requests = 0
        total_errors = 0
        total_duration = 0

        for pod_name, logs in all_logs:
            for line in logs.split("\n"):
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

        print(f"\nTotal requests: {total_requests}")
        print(f"Total errors: {total_errors}")
        print(f"Duration: {total_duration:.1f}s")

        if total_duration > 0:
            actual_qps = total_requests / total_duration
            print(f"Target QPS: {self.qps:.1f}")
            print(f"Actual QPS: {actual_qps:.1f}")
            print(f"Achievement: {(actual_qps / self.qps * 100):.1f}%")

        if total_requests > 0:
            print(f"Error rate: {(total_errors / total_requests) * 100:.1f}%")

        print("=" * 70)

    async def delete(self) -> None:
        await self._delete_job()
        await self._delete_configmap()

    async def _delete_job(self) -> None:
        if self._batch_api is None:
            return
        try:
            await self._batch_api.delete_namespaced_job(
                name=self.job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
        except exceptions.ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete job: {e}")

    async def _delete_configmap(self) -> None:
        if self._core_api is None:
            return
        try:
            await self._core_api.delete_namespaced_config_map(
                name=f"{self.job_name}-script", namespace=self.namespace
            )
        except exceptions.ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete ConfigMap: {e}")
