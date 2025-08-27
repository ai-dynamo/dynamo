# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
import time

import kubernetes_asyncio as kubernetes
import yaml
from kubernetes_asyncio import client, config


class VanillaBackendClient:
    def __init__(
        self,
        namespace: str,
        deployment_name: str = "vanilla-backend",
        service_name: str = "vanilla-backend",
        frontend_port: int = 8000,
    ):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.service_name = service_name
        self.frontend_port = frontend_port
        self.port_forward_process = None

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()

        self.k8s_client = client.ApiClient()
        self.apps_api = client.AppsV1Api(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

    async def create_deployment(self, manifest_path: str):
        """Create a vanilla Kubernetes deployment from yaml file"""
        await self._init_kubernetes()

        with open(manifest_path, "r") as f:
            manifests = list(yaml.safe_load_all(f))

        for manifest in manifests:
            if not manifest:
                continue

            # Set namespace
            manifest["metadata"]["namespace"] = self.namespace

            if manifest["kind"] == "Deployment":
                try:
                    await self.apps_api.create_namespaced_deployment(
                        namespace=self.namespace, body=manifest
                    )
                    print(f"Successfully created deployment {self.deployment_name}")
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 409:  # Already exists
                        print(f"Deployment {self.deployment_name} already exists")
                    else:
                        print(
                            f"Failed to create deployment {self.deployment_name}: {e}"
                        )
                        raise
            elif manifest["kind"] == "Service":
                try:
                    await self.core_api.create_namespaced_service(
                        namespace=self.namespace, body=manifest
                    )
                    print(f"Successfully created service {self.service_name}")
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 409:  # Already exists
                        print(f"Service {self.service_name} already exists")
                    else:
                        print(f"Failed to create service {self.service_name}: {e}")
                        raise

    async def wait_for_deployment_ready(self, timeout: int = 1800):
        """Wait for the deployment to be ready"""
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                deployment = await self.apps_api.read_namespaced_deployment(
                    name=self.deployment_name, namespace=self.namespace
                )

                status = deployment.status
                ready_replicas = status.ready_replicas or 0
                replicas = status.replicas or 0

                print(f"Deployment status: {ready_replicas}/{replicas} replicas ready")
                print(f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s")

                if ready_replicas == replicas and replicas > 0:
                    print("Deployment is ready: All replicas are ready")
                    return True
                else:
                    print("Deployment not ready yet")

            except kubernetes.client.rest.ApiException as e:
                print(f"API Exception while checking deployment status: {e}")
            except Exception as e:
                print(f"Unexpected exception while checking deployment status: {e}")

            await asyncio.sleep(20)

        raise TimeoutError("Deployment failed to become ready within timeout")

    def port_forward_frontend(self, local_port: int = 8000, quiet: bool = False) -> str:
        """
        Port forward the frontend service to a local port.

        Args:
            local_port: Local port to forward to (default: 8000)
            quiet: If True, suppress kubectl port-forward output messages (default: False)
        """
        cmd = [
            "kubectl",
            "port-forward",
            f"svc/{self.service_name}",
            f"{local_port}:{self.frontend_port}",
            "-n",
            self.namespace,
        ]

        print(f"Starting port forward: {' '.join(cmd)}")

        # Configure output redirection based on quiet flag
        if quiet:
            # Suppress kubectl's "Handling connection for..." messages
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None

        # Start port forward in background
        self.port_forward_process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

        # Wait a moment for port forward to establish
        print("Waiting for port forward to establish...")
        time.sleep(3)

        print(f"Port forward started with PID: {self.port_forward_process.pid}")
        return f"http://localhost:{local_port}"

    def stop_port_forward(self):
        """Stop the port forward process"""
        if self.port_forward_process:
            print(
                f"Stopping port forward process (PID: {self.port_forward_process.pid})"
            )
            self.port_forward_process.terminate()
            try:
                self.port_forward_process.wait(timeout=5)
                print("Port forward stopped")
            except subprocess.TimeoutExpired:
                print("Port forward process did not terminate, killing it")
                self.port_forward_process.kill()
                self.port_forward_process.wait()
            self.port_forward_process = None

    async def delete_deployment(self):
        """Delete the deployment and service"""
        try:
            # Delete deployment
            await self.apps_api.delete_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace
            )
            print(f"Successfully deleted deployment {self.deployment_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                print(f"Failed to delete deployment: {e}")

        try:
            # Delete service
            await self.core_api.delete_namespaced_service(
                name=self.service_name, namespace=self.namespace
            )
            print(f"Successfully deleted service {self.service_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                print(f"Failed to delete service: {e}")

        # Close the kubernetes client session to avoid warnings
        if hasattr(self, "k8s_client"):
            await self.k8s_client.close()
