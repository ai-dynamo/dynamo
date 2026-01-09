# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

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
    image: str = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"
    timeout: int = 600
    name_prefix: str = "scale-test"
    cleanup_on_exit: bool = True
    image_pull_secrets: List[str] = field(default_factory=list)

    _deployment_specs: List[DeploymentSpec] = field(default_factory=list)
    _deployment_names: List[str] = field(default_factory=list)
    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _log_dir: Optional[str] = None
    _initialized: bool = False

    def __post_init__(self):
        self._log_dir = tempfile.mkdtemp(prefix="scale_test_logs_")
        logger.info(f"Scale test logs: {self._log_dir}")

    @classmethod
    def from_config(cls, config: ScaleTestConfig, num_deployments: int) -> "ScaleManager":
        return cls(
            num_deployments=num_deployments,
            model_path=config.model_path,
            speedup_ratio=config.speedup_ratio,
            kubernetes_namespace=config.kubernetes_namespace,
            image=config.image,
            timeout=config.deployment_timeout,
            name_prefix=config.name_prefix,
            cleanup_on_exit=config.cleanup_on_exit,
            image_pull_secrets=config.image_pull_secrets,
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

    def _build_deployment_specs(self) -> List[DeploymentSpec]:
        specs = []
        for i in range(1, self.num_deployments + 1):
            builder = ScaleTestDGDBuilder(deployment_id=i, name_prefix=self.name_prefix)
            builder_chain = (
                builder.set_kubernetes_namespace(self.kubernetes_namespace)
                .set_model(self.model_path)
                .set_speedup_ratio(self.speedup_ratio)
                .set_image(self.image)
            )

            if self.image_pull_secrets:
                builder_chain = builder_chain.set_image_pull_secrets(self.image_pull_secrets)

            spec = builder_chain.build()
            specs.append(spec)
            self._deployment_names.append(spec.name)
        return specs

    async def deploy_dgds(self) -> None:
        if not self._initialized:
            await self._init_kubernetes()

        logger.info(f"Building {self.num_deployments} DGD specifications...")
        self._deployment_specs = self._build_deployment_specs()

        logger.info(f"Creating {self.num_deployments} DGD deployments...")

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

        logger.info(f"Waiting for {len(self._deployment_names)} DGDs to become ready...")
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
                logger.info(f"Progress: {ready_count}/{len(self._deployment_names)} DGDs ready")
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

        logger.info(f"Running load generator: {duration_sec}s, {qps} QPS, {len(urls)} frontends")
        logger.info(f"Parallelism: {num_pods} pod(s) x {num_processes_per_pod} process(es)")

        k8s_client = client.ApiClient()
        batch_api = client.BatchV1Api(k8s_client)

        job = LoadGeneratorJob(
            namespace=self.kubernetes_namespace,
            frontend_urls=urls,
            model=model,
            duration_sec=duration_sec,
            qps=qps,
            max_tokens=max_tokens,
            image=self.image,
            num_pods=num_pods,
            num_processes_per_pod=num_processes_per_pod,
        )

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
