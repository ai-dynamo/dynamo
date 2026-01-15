# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

from tests.scale_test.aiperf_load_generator_job import (
    AIPerfConfig,
    MultiTargetAIPerfJob,
)
from tests.scale_test.config import ScaleTestConfig
from tests.scale_test.dgd_builder import ScaleTestDGDBuilder
from tests.scale_test.load_generator_job import LoadGeneratorJob
from tests.utils.managed_deployment import DeploymentSpec

logger = logging.getLogger(__name__)


@dataclass
class ScaleManager:
    """Manages the lifecycle of scale test DGD deployments on Kubernetes."""

    num_deployments: int
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    speedup_ratio: float = 10.0
    kubernetes_namespace: str = "default"
    timeout: int = 600
    name_prefix: str = "scale-test"
    cleanup_on_exit: bool = True
    worker_replicas: int = 1

    _deployment_specs: List[DeploymentSpec] = field(default_factory=list)
    _deployment_names: List[str] = field(default_factory=list)
    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _log_dir: Optional[str] = None
    _initialized: bool = False
    _start_id: int = 1  # Starting deployment ID (auto-detected to avoid conflicts)

    def __post_init__(self):
        self._log_dir = tempfile.mkdtemp(prefix="scale_test_logs_")
        logger.info(f"Scale test logs: {self._log_dir}")

    @classmethod
    def from_config(
        cls, config: ScaleTestConfig, num_deployments: int
    ) -> "ScaleManager":
        return cls(
            num_deployments=num_deployments,
            model_path=config.model_path,
            speedup_ratio=config.speedup_ratio,
            kubernetes_namespace=config.kubernetes_namespace,
            timeout=config.deployment_timeout,
            name_prefix=config.name_prefix,
            cleanup_on_exit=config.cleanup_on_exit,
            worker_replicas=config.worker_replicas,
        )

    async def _init_kubernetes(self) -> None:
        try:
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except Exception:
            await config.load_kube_config()
            logger.info("Using kubeconfig file")

        k8s_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(k8s_client)
        self._core_api = client.CoreV1Api(k8s_client)
        self._initialized = True

    async def _find_next_available_start_id(self) -> int:
        """Find the next available deployment ID by checking existing DGDs.

        Scans existing DGDs with the same name prefix and returns the first
        ID that would not conflict with any existing deployment.
        """
        assert self._custom_api is not None

        try:
            dgds = await self._custom_api.list_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.kubernetes_namespace,
                plural="dynamographdeployments",
            )
        except exceptions.ApiException as e:
            logger.warning(f"Could not list existing DGDs: {e}")
            return 1

        # Find all existing IDs with our prefix
        existing_ids = set()
        prefix_with_dash = f"{self.name_prefix}-"

        for dgd in dgds.get("items", []):
            name = dgd["metadata"]["name"]
            if name.startswith(prefix_with_dash):
                suffix = name[len(prefix_with_dash):]
                try:
                    deployment_id = int(suffix)
                    existing_ids.add(deployment_id)
                except ValueError:
                    # Not a numeric suffix, ignore
                    pass

        if not existing_ids:
            return 1

        # Find the first ID that doesn't conflict with any of the IDs we need
        # We need num_deployments consecutive-ish slots, but simpler: start after max
        max_existing = max(existing_ids)
        next_id = max_existing + 1

        logger.info(
            f"Found {len(existing_ids)} existing DGDs with prefix '{self.name_prefix}' "
            f"(max ID: {max_existing}). Starting new deployments from ID {next_id}."
        )

        return next_id

    def _build_deployment_specs(self) -> List[DeploymentSpec]:
        specs = []
        for i in range(self.num_deployments):
            deployment_id = self._start_id + i
            builder = ScaleTestDGDBuilder(deployment_id=deployment_id, name_prefix=self.name_prefix)
            spec = (
                builder.set_kubernetes_namespace(self.kubernetes_namespace)
                .set_model(self.model_path)
                .set_speedup_ratio(self.speedup_ratio)
                .set_worker_replicas(self.worker_replicas)
                .build()
            )
            specs.append(spec)
            self._deployment_names.append(spec.name)
        return specs

    async def deploy_dgds(self) -> None:
        if not self._initialized:
            await self._init_kubernetes()

        # Find next available start ID to avoid conflicts with existing deployments
        self._start_id = await self._find_next_available_start_id()

        logger.info(f"Building {self.num_deployments} DGD specifications...")
        self._deployment_specs = self._build_deployment_specs()

        total_mockers = self.num_deployments * self.worker_replicas
        logger.info(
            f"Creating {self.num_deployments} DGD deployments "
            f"({total_mockers} mockers total)..."
        )

        for i, spec in enumerate(self._deployment_specs, 1):
            deployment_name = spec.name
            logger.info(f"Creating DGD {i}/{self.num_deployments}: {deployment_name}")

            try:
                assert self._custom_api is not None
                spec.namespace = self.kubernetes_namespace

                await self._custom_api.create_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.kubernetes_namespace,
                    plural="dynamographdeployments",
                    body=spec.spec(),
                )
                logger.info(f"DGD {deployment_name} created")

            except exceptions.ApiException as e:
                if e.status == 409:
                    logger.warning(f"DGD {deployment_name} already exists")
                else:
                    raise

        logger.info(f"All {self.num_deployments} DGDs created!")

    async def wait_for_dgds_ready(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            timeout = self.timeout

        assert self._custom_api is not None

        logger.info(
            f"Waiting for {len(self._deployment_names)} DGDs to become ready..."
        )
        start_time = time.time()
        pending_deployments = set(self._deployment_names)
        log_interval = 30
        last_log_time = start_time

        while pending_deployments and (time.time() - start_time) < timeout:
            for deployment_name in list(pending_deployments):
                try:
                    status = await self._custom_api.get_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.kubernetes_namespace,
                        plural="dynamographdeployments",
                        name=deployment_name,
                    )

                    status_obj = status.get("status", {})
                    conditions = status_obj.get("conditions", [])
                    state = status_obj.get("state", "unknown")

                    is_ready = any(
                        c.get("type") == "Ready" and c.get("status") == "True"
                        for c in conditions
                    )

                    if is_ready and state == "successful":
                        logger.info(f"DGD {deployment_name} is ready")
                        pending_deployments.discard(deployment_name)

                except exceptions.ApiException:
                    pass

            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                ready_count = len(self._deployment_names) - len(pending_deployments)
                logger.info(
                    f"Progress: {ready_count}/{len(self._deployment_names)} DGDs ready"
                )
                last_log_time = current_time

            if pending_deployments:
                await asyncio.sleep(2)

        if pending_deployments:
            logger.error(f"Timeout: {sorted(pending_deployments)} not ready")
            return False

        logger.info(f"All {len(self._deployment_names)} DGDs are ready!")
        return True

    async def get_frontend_urls(self) -> List[str]:
        assert self._core_api is not None

        urls = []
        for deployment_name in self._deployment_names:
            service_name = f"{deployment_name}-frontend"
            try:
                await self._core_api.read_namespaced_service(
                    name=service_name,
                    namespace=self.kubernetes_namespace,
                )
                url = f"http://{service_name}.{self.kubernetes_namespace}.svc.cluster.local:8000"
                urls.append(url)
            except exceptions.ApiException as e:
                if e.status == 404:
                    logger.warning(f"Service {service_name} not found")
                else:
                    logger.error(f"Error getting service {service_name}: {e}")

        return urls

    async def get_frontend_pods(self) -> Dict[str, str]:
        """Get frontend pod names for each deployment.

        Returns:
            Dict mapping deployment name to pod name.
        """
        assert self._core_api is not None

        pods = {}
        for deployment_name in self._deployment_names:
            # Frontend pods are labeled by the DGD operator
            label_selector = f"nvidia.com/dgd={deployment_name},nvidia.com/dgd-service=Frontend"
            try:
                pod_list = await self._core_api.list_namespaced_pod(
                    namespace=self.kubernetes_namespace,
                    label_selector=label_selector,
                )
                if pod_list.items:
                    # Take the first running pod
                    for pod in pod_list.items:
                        if pod.status.phase == "Running":
                            pods[deployment_name] = pod.metadata.name
                            break
                    else:
                        # No running pod, take the first one
                        pods[deployment_name] = pod_list.items[0].metadata.name
            except exceptions.ApiException as e:
                logger.warning(f"Error getting frontend pod for {deployment_name}: {e}")

        return pods

    async def collect_frontend_logs(
        self,
        output_dir: Optional[str] = None,
        since_seconds: Optional[int] = None,
    ) -> Dict[str, str]:
        """Collect logs from all frontend pods.

        Args:
            output_dir: Directory to save logs. Defaults to self._log_dir.
            since_seconds: Only return logs from the last N seconds.

        Returns:
            Dict mapping deployment name to log file path.
        """
        assert self._core_api is not None

        output_dir = output_dir or self._log_dir
        os.makedirs(output_dir, exist_ok=True)

        pods = await self.get_frontend_pods()
        log_files = {}

        for deployment_name, pod_name in pods.items():
            try:
                kwargs = {"name": pod_name, "namespace": self.kubernetes_namespace}
                if since_seconds:
                    kwargs["since_seconds"] = since_seconds

                logs = await self._core_api.read_namespaced_pod_log(**kwargs)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(
                    output_dir, f"{deployment_name}_frontend_{timestamp}.log"
                )

                with open(log_file, "w") as f:
                    f.write(logs)

                log_files[deployment_name] = log_file
                logger.info(f"Saved frontend logs for {deployment_name}: {log_file}")

            except exceptions.ApiException as e:
                logger.warning(f"Error getting logs for {deployment_name}: {e}")

        return log_files

    async def stream_frontend_logs_to_file(
        self,
        output_dir: Optional[str] = None,
        duration_sec: Optional[int] = None,
    ) -> Dict[str, str]:
        """Stream logs from all frontend pods to files in the background.

        Args:
            output_dir: Directory to save logs. Defaults to self._log_dir.
            duration_sec: How long to stream logs (optional, runs until cancelled).

        Returns:
            Dict mapping deployment name to log file path.
        """
        assert self._core_api is not None

        output_dir = output_dir or self._log_dir
        os.makedirs(output_dir, exist_ok=True)

        pods = await self.get_frontend_pods()
        log_files = {}
        tasks = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for deployment_name, pod_name in pods.items():
            log_file = os.path.join(
                output_dir, f"{deployment_name}_frontend_{timestamp}.log"
            )
            log_files[deployment_name] = log_file

            task = asyncio.create_task(
                self._stream_pod_logs(pod_name, log_file, duration_sec)
            )
            tasks.append(task)

        # Return immediately, logs are streamed in background
        # Caller can await these tasks or let them run
        self._log_stream_tasks = tasks
        return log_files

    async def _stream_pod_logs(
        self,
        pod_name: str,
        log_file: str,
        duration_sec: Optional[int] = None,
    ) -> None:
        """Stream logs from a pod to a file."""
        assert self._core_api is not None

        start_time = time.time()
        logger.info(f"Streaming logs from {pod_name} to {log_file}")

        try:
            with open(log_file, "w") as f:
                # Use follow=True for streaming
                # kubernetes_asyncio doesn't support true streaming easily,
                # so we poll with since_seconds
                last_log_time = start_time
                seen_lines = set()

                while True:
                    if duration_sec and (time.time() - start_time) >= duration_sec:
                        break

                    try:
                        # Get logs since last poll
                        since = int(time.time() - last_log_time) + 1
                        logs = await self._core_api.read_namespaced_pod_log(
                            name=pod_name,
                            namespace=self.kubernetes_namespace,
                            since_seconds=since,
                        )

                        if logs:
                            # Write new lines (dedupe using hash)
                            for line in logs.split("\n"):
                                if line and hash(line) not in seen_lines:
                                    seen_lines.add(hash(line))
                                    f.write(line + "\n")
                                    f.flush()

                        last_log_time = time.time()

                    except exceptions.ApiException as e:
                        if e.status == 404:
                            logger.warning(f"Pod {pod_name} no longer exists")
                            break
                        logger.debug(f"Error polling logs: {e}")

                    await asyncio.sleep(2)  # Poll every 2 seconds

        except asyncio.CancelledError:
            logger.info(f"Log streaming cancelled for {pod_name}")
        except Exception as e:
            logger.error(f"Error streaming logs from {pod_name}: {e}")

    async def stop_log_streaming(self) -> None:
        """Stop any background log streaming tasks."""
        if hasattr(self, "_log_stream_tasks"):
            for task in self._log_stream_tasks:
                task.cancel()
            await asyncio.gather(*self._log_stream_tasks, return_exceptions=True)
            self._log_stream_tasks = []

    async def run_load_generator_job(
        self,
        model: str,
        duration_sec: int,
        qps: float,
        max_tokens: int = 30,
        timeout: int = 600,
        num_pods: int = 1,
        num_processes_per_pod: int = 1,
    ) -> bool:
        assert self._core_api is not None

        urls = await self.get_frontend_urls()
        if not urls:
            logger.error("No frontend URLs found")
            return False

        logger.info(
            f"Running load generator: {duration_sec}s, {qps} QPS, {len(urls)} frontends"
        )
        logger.info(
            f"Parallelism: {num_pods} pod(s) x {num_processes_per_pod} process(es)"
        )

        k8s_client = client.ApiClient()
        batch_api = client.BatchV1Api(k8s_client)

        job = LoadGeneratorJob(
            namespace=self.kubernetes_namespace,
            frontend_urls=urls,
            model=model,
            duration_sec=duration_sec,
            qps=qps,
            max_tokens=max_tokens,
            num_pods=num_pods,
            num_processes_per_pod=num_processes_per_pod,
        )

        success = await job.create_and_wait(batch_api, self._core_api, timeout)
        await job.delete()
        return success

    async def run_aiperf_load_generator(
        self,
        model: str,
        duration_sec: int,
        config: Optional[AIPerfConfig] = None,
        timeout: int = 600,
        image: Optional[str] = None,
        tokenizer: Optional[str] = None,
    ) -> bool:
        """Run AIPerf-based load generation against all frontends."""
        assert self._core_api is not None

        urls = await self.get_frontend_urls()
        if not urls:
            logger.error("No frontend URLs found")
            return False

        aiperf_config = config or AIPerfConfig()

        logger.info(
            f"Running AIPerf load generator: {duration_sec}s, "
            f"concurrency={aiperf_config.concurrency}, {len(urls)} frontends"
        )
        logger.info(
            f"ISL: {aiperf_config.isl_mean} (stddev: {aiperf_config.isl_stddev}), "
            f"OSL: {aiperf_config.osl_mean} (stddev: {aiperf_config.osl_stddev})"
        )

        k8s_client = client.ApiClient()
        batch_api = client.BatchV1Api(k8s_client)

        job_kwargs = {
            "namespace": self.kubernetes_namespace,
            "frontend_urls": urls,
            "model": model,
            "duration_sec": duration_sec,
            "config": aiperf_config,
            "tokenizer": tokenizer or model,
        }
        if image:
            job_kwargs["image"] = image

        job = MultiTargetAIPerfJob(**job_kwargs)

        success = await job.create_and_wait(batch_api, self._core_api, timeout)
        await job.delete()
        return success

    async def cleanup(self) -> None:
        if not self._initialized or self._custom_api is None:
            return

        logger.info(f"Cleaning up {len(self._deployment_names)} DGD deployments...")

        for deployment_name in self._deployment_names:
            try:
                await self._custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.kubernetes_namespace,
                    plural="dynamographdeployments",
                    name=deployment_name,
                )
            except exceptions.ApiException as e:
                if e.status != 404:
                    logger.warning(f"Error deleting DGD {deployment_name}: {e}")

        logger.info("Waiting for deletions...")
        await self._wait_for_deletions(timeout=120)

        self._deployment_names.clear()
        self._deployment_specs.clear()
        logger.info("Cleanup complete")

    async def _wait_for_deletions(self, timeout: float = 120) -> None:
        assert self._custom_api is not None

        start_time = time.time()
        pending = set(self._deployment_names)

        while pending and (time.time() - start_time) < timeout:
            for name in list(pending):
                try:
                    await self._custom_api.get_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.kubernetes_namespace,
                        plural="dynamographdeployments",
                        name=name,
                    )
                except exceptions.ApiException as e:
                    if e.status == 404:
                        pending.discard(name)

            if pending:
                await asyncio.sleep(2)

        if pending:
            logger.warning(f"Some DGDs not fully deleted: {sorted(pending)}")

    async def start_all(self) -> None:
        await self._init_kubernetes()
        await self.deploy_dgds()
        if not await self.wait_for_dgds_ready():
            raise RuntimeError("Not all DGDs became ready within timeout")

    async def __aenter__(self):
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            await self.cleanup()
        return False
