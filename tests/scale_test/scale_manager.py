# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core Kubernetes deployment management for scale testing.

This module manages the lifecycle of DynamoGraphDeployment resources
for scale testing Dynamo deployments on Kubernetes.
"""

import asyncio
import logging
import signal
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

from tests.scale_test.config import ScaleTestConfig
from tests.scale_test.dgd_builder import ScaleTestDGDBuilder
from tests.utils.managed_deployment import DeploymentSpec

logger = logging.getLogger(__name__)


@dataclass
class ScaleManager:
    """
    Manages the lifecycle of scale test DGD deployments on Kubernetes.

    Creates and manages N DynamoGraphDeployment resources, each with its own
    Frontend and MockerWorker services. The Dynamo operator handles NATS/etcd
    infrastructure.
    """

    num_deployments: int
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    speedup_ratio: float = 10.0
    kubernetes_namespace: str = "default"
    image: str = "nvcr.io/nvidia/ai-dynamo/dynamo-base:latest"
    timeout: int = 600
    name_prefix: str = "scale-test"
    cleanup_on_exit: bool = True

    # Deployment tracking
    _deployment_specs: List[DeploymentSpec] = field(default_factory=list)
    _deployment_names: List[str] = field(default_factory=list)

    # Kubernetes client
    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _log_dir: Optional[str] = None
    _initialized: bool = False

    def __post_init__(self):
        """Initialize logging and create log directory."""
        self._log_dir = tempfile.mkdtemp(prefix="scale_test_logs_")
        logger.info(f"Scale test logs will be written to: {self._log_dir}")

    @classmethod
    def from_config(
        cls, config: ScaleTestConfig, num_deployments: int
    ) -> "ScaleManager":
        """Create a ScaleManager from a ScaleTestConfig."""
        return cls(
            num_deployments=num_deployments,
            model_path=config.model_path,
            speedup_ratio=config.speedup_ratio,
            kubernetes_namespace=config.kubernetes_namespace,
            image=config.image,
            timeout=config.deployment_timeout,
            name_prefix=config.name_prefix,
            cleanup_on_exit=config.cleanup_on_exit,
        )

    async def _init_kubernetes(self) -> None:
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()
            logger.info("Using kubeconfig file for Kubernetes configuration")

        k8s_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(k8s_client)
        self._core_api = client.CoreV1Api(k8s_client)
        self._initialized = True

    def _build_deployment_specs(self) -> List[DeploymentSpec]:
        """Build DGD specs for all deployments."""
        specs = []
        for i in range(1, self.num_deployments + 1):
            builder = ScaleTestDGDBuilder(
                deployment_id=i,
                name_prefix=self.name_prefix,
            )
            spec = (
                builder.set_kubernetes_namespace(self.kubernetes_namespace)
                .set_model(self.model_path)
                .set_speedup_ratio(self.speedup_ratio)
                .set_image(self.image)
                .build()
            )
            specs.append(spec)
            self._deployment_names.append(spec.name)
        return specs

    async def deploy_dgds(self) -> None:
        """
        Deploy N DynamoGraphDeployment resources to Kubernetes.

        Each deployment gets a unique name (e.g., scale-test-1, scale-test-2)
        and uses the Dynamo operator for infrastructure management.
        """
        if not self._initialized:
            await self._init_kubernetes()

        logger.info(f"Building {self.num_deployments} DGD specifications...")
        self._deployment_specs = self._build_deployment_specs()

        logger.info(f"Creating {self.num_deployments} DGD deployments...")

        for i, spec in enumerate(self._deployment_specs, 1):
            deployment_name = spec.name
            logger.info(f"Creating DGD {i}/{self.num_deployments}: {deployment_name}")

            try:
                assert self._custom_api is not None, "Kubernetes API not initialized"

                # Set namespace in spec
                spec.namespace = self.kubernetes_namespace

                await self._custom_api.create_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.kubernetes_namespace,
                    plural="dynamographdeployments",
                    body=spec.spec(),
                )
                logger.info(f"DGD {deployment_name} created successfully")

            except exceptions.ApiException as e:
                if e.status == 409:  # Already exists
                    logger.warning(f"DGD {deployment_name} already exists, skipping")
                else:
                    logger.error(f"Failed to create DGD {deployment_name}: {e}")
                    raise

        logger.info(f"All {self.num_deployments} DGDs created!")

    async def wait_for_dgds_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all DGD deployments to become ready.

        Args:
            timeout: Maximum time to wait in seconds (defaults to self.timeout)

        Returns:
            True if all deployments are ready, False otherwise
        """
        if timeout is None:
            timeout = self.timeout

        assert self._custom_api is not None, "Kubernetes API not initialized"

        logger.info(
            f"Waiting for {len(self._deployment_names)} DGDs to become ready..."
        )
        start_time = time.time()

        pending_deployments = set(self._deployment_names)
        log_interval = 30  # Log progress every 30 seconds
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

                    # Check if Ready condition is True and state is successful
                    is_ready = False
                    for condition in conditions:
                        if (
                            condition.get("type") == "Ready"
                            and condition.get("status") == "True"
                        ):
                            is_ready = True
                            break

                    if is_ready and state == "successful":
                        logger.info(f"DGD {deployment_name} is ready")
                        pending_deployments.discard(deployment_name)

                except exceptions.ApiException as e:
                    logger.debug(f"Error checking DGD {deployment_name}: {e}")

            # Log progress periodically
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                elapsed = current_time - start_time
                ready_count = len(self._deployment_names) - len(pending_deployments)
                logger.info(
                    f"Progress: {ready_count}/{len(self._deployment_names)} DGDs ready "
                    f"({elapsed:.0f}s/{timeout}s)"
                )
                last_log_time = current_time

            if pending_deployments:
                await asyncio.sleep(2)

        if pending_deployments:
            logger.error(
                f"Timeout waiting for DGDs: {sorted(pending_deployments)} not ready"
            )
            return False

        logger.info(f"All {len(self._deployment_names)} DGDs are ready!")
        return True

    async def get_frontend_urls(self) -> List[str]:
        """
        Get the list of frontend service URLs.

        Returns list of URLs for the Frontend services, obtained by querying
        Kubernetes Services created by the operator.

        Returns:
            List of URLs like ['http://scale-test-1-frontend:8000', ...]
        """
        assert self._core_api is not None, "Kubernetes API not initialized"

        urls = []
        for deployment_name in self._deployment_names:
            # The operator creates services with naming convention: {deployment}-{service}
            service_name = f"{deployment_name}-frontend"
            try:
                _ = await self._core_api.read_namespaced_service(
                    name=service_name,
                    namespace=self.kubernetes_namespace,
                )
                # Use cluster-internal DNS name
                # Format: {service}.{namespace}.svc.cluster.local
                url = f"http://{service_name}.{self.kubernetes_namespace}.svc.cluster.local:8000"
                urls.append(url)
            except exceptions.ApiException as e:
                if e.status == 404:
                    logger.warning(f"Service {service_name} not found")
                else:
                    logger.error(f"Error getting service {service_name}: {e}")

        return urls

    async def get_frontend_external_urls(self) -> List[str]:
        """
        Get external URLs for frontends (via NodePort or LoadBalancer).

        This is useful when running load tests from outside the cluster.

        Returns:
            List of external URLs if available
        """
        assert self._core_api is not None, "Kubernetes API not initialized"

        urls = []
        for deployment_name in self._deployment_names:
            service_name = f"{deployment_name}-frontend"
            try:
                service = await self._core_api.read_namespaced_service(
                    name=service_name,
                    namespace=self.kubernetes_namespace,
                )

                service_type = service.spec.type

                if service_type == "LoadBalancer":
                    # Get external IP from LoadBalancer
                    ingress = service.status.load_balancer.ingress
                    if ingress:
                        external_ip = ingress[0].ip or ingress[0].hostname
                        port = 8000
                        for port_spec in service.spec.ports:
                            if port_spec.port == 8000:
                                port = port_spec.port
                                break
                        urls.append(f"http://{external_ip}:{port}")

                elif service_type == "NodePort":
                    # Get NodePort
                    node_port = None
                    for port_spec in service.spec.ports:
                        if port_spec.port == 8000:
                            node_port = port_spec.node_port
                            break
                    if node_port:
                        # Use localhost for local testing
                        urls.append(f"http://localhost:{node_port}")
                else:
                    # ClusterIP - use port-forward or internal URL
                    logger.debug(
                        f"Service {service_name} is ClusterIP, external access requires port-forward"
                    )

            except exceptions.ApiException as e:
                logger.debug(f"Error getting external URL for {service_name}: {e}")

        return urls

    async def cleanup(self) -> None:
        """
        Delete all DGD deployments.

        Deletes each DynamoGraphDeployment CR, which triggers the operator
        to clean up all associated resources.
        """
        if not self._initialized or self._custom_api is None:
            logger.warning("Kubernetes not initialized, skipping cleanup")
            return

        logger.info(f"Cleaning up {len(self._deployment_names)} DGD deployments...")

        for deployment_name in self._deployment_names:
            try:
                logger.info(f"Deleting DGD: {deployment_name}")
                await self._custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.kubernetes_namespace,
                    plural="dynamographdeployments",
                    name=deployment_name,
                )
            except exceptions.ApiException as e:
                if e.status == 404:
                    logger.debug(f"DGD {deployment_name} already deleted")
                else:
                    logger.warning(f"Error deleting DGD {deployment_name}: {e}")

        # Wait for deletions to complete
        logger.info("Waiting for DGD deletions to complete...")
        await self._wait_for_deletions(timeout=120)

        self._deployment_names.clear()
        self._deployment_specs.clear()
        logger.info("Cleanup complete")

    async def _wait_for_deletions(self, timeout: float = 120) -> None:
        """Wait for all DGDs to be fully deleted."""
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
                    # Still exists
                except exceptions.ApiException as e:
                    if e.status == 404:
                        pending.discard(name)
                        logger.debug(f"DGD {name} deleted")

            if pending:
                await asyncio.sleep(2)

        if pending:
            logger.warning(f"Some DGDs not fully deleted: {sorted(pending)}")

    async def start_all(self) -> None:
        """
        Deploy all DGDs and wait for them to be ready.

        This is a convenience method that combines deploy_dgds() and
        wait_for_dgds_ready().
        """
        await self._init_kubernetes()
        await self.deploy_dgds()
        if not await self.wait_for_dgds_ready():
            raise RuntimeError("Not all DGDs became ready within timeout")

    async def __aenter__(self):
        """Async context manager entry - deploy all DGDs."""
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup all DGDs."""
        if self.cleanup_on_exit:
            await self.cleanup()
        return False


def setup_signal_handlers(
    manager: ScaleManager, loop: asyncio.AbstractEventLoop
) -> None:
    """
    Set up signal handlers for graceful cleanup on Ctrl+C.

    Args:
        manager: The ScaleManager instance to clean up on signal
        loop: The asyncio event loop
    """

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        # Schedule cleanup in the event loop
        asyncio.ensure_future(manager.cleanup(), loop=loop)
        loop.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_scale_test(
    num_deployments: int,
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    speedup_ratio: float = 10.0,
    namespace: str = "default",
    image: str = "nvcr.io/nvidia/ai-dynamo/dynamo-base:latest",
    timeout: int = 600,
) -> ScaleManager:
    """
    Convenience function to create and start a ScaleManager.

    Args:
        num_deployments: Number of DGD deployments to create
        model_path: Model path for mocker
        speedup_ratio: Mocker speedup ratio
        namespace: Kubernetes namespace
        image: Container image
        timeout: Deployment timeout

    Returns:
        Started ScaleManager instance
    """
    manager = ScaleManager(
        num_deployments=num_deployments,
        model_path=model_path,
        speedup_ratio=speedup_ratio,
        kubernetes_namespace=namespace,
        image=image,
        timeout=timeout,
    )
    await manager.start_all()
    return manager
