#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "PyYAML",
#   "aiofiles",
#   "kubernetes-asyncio",
#   "kr8s",           # added
#   "httpx",          # added
# ]
# ///

import argparse
import asyncio
import os
import random
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import aiofiles
import httpx  # added for HTTP requests
import kubernetes_asyncio as kubernetes
import yaml
from kr8s.objects import Service
from kubernetes_asyncio import client, config

# Example chat completion request for testing deployments
EXAMPLE_CHAT_REQUEST = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {
            "role": "user",
            "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
        }
    ],
    "stream": False,
    "max_tokens": 30,
}


class DynamoDeploymentClient:
    def __init__(
        self,
        namespace: str,
        model_name: str = "Qwen/Qwen3-0.6B",
        deployment_name: str = "vllm-v1-agg",
        base_log_dir: Optional[str] = None,
    ):
        """
        Initialize the client with the namespace and deployment name.

        Args:
            namespace: The Kubernetes namespace
            deployment_name: Name of the deployment, defaults to vllm-v1-agg
            base_log_dir: Base directory for storing logs, defaults to ./logs if not specified
        """
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.components = []  # Will store component names from CR
        self.deployment_spec = None  # Will store the full deployment spec
        self.base_log_dir = Path(base_log_dir) if base_log_dir else Path("logs")

    def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
        except Exception:
            # Fallback to kube config file (for local development)
            config.load_kube_config()

        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

    def _is_running_in_kubernetes(self) -> bool:
        """
        Detect if we're running inside a Kubernetes cluster by checking for the service account token.
        """
        return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")

    def _get_service_url(self) -> str:
        """
        Get the service URL. Returns service DNS if running in Kubernetes, otherwise localhost with port forwarding.
        """
        if self._is_running_in_kubernetes():
            # Use Kubernetes service DNS: service-name.namespace.svc.cluster.local:port
            svc_name = f"{self.deployment_name}-frontend"
            return f"http://{svc_name}.{self.namespace}.svc.cluster.local:8000"
        else:
            # For local development, we need to use port forwarding
            raise RuntimeError(
                "Port forwarding is required when not running in Kubernetes. Use get_service_url_with_port_forward() instead."
            )

    async def create_deployment(self, deployment: Union[dict, str]):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.

        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """
        self._init_kubernetes()

        if isinstance(deployment, str):
            # Load from yaml file
            async with aiofiles.open(deployment, "r") as f:
                content = await f.read()
                self.deployment_spec = yaml.safe_load(content)
        else:
            self.deployment_spec = deployment

        # Extract component names
        self.components = [
            svc.lower() for svc in self.deployment_spec["spec"]["services"].keys()
        ]

        # Ensure name and namespace are set correctly
        self.deployment_spec["metadata"]["name"] = self.deployment_name
        self.deployment_spec["metadata"]["namespace"] = self.namespace

        try:
            await self.custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec,
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                print(f"Deployment {self.deployment_name} already exists")
            else:
                raise

    async def wait_for_deployment_ready(self, timeout: int = 300):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        # TODO: A little brittle, also should output intermediate status every so often.
        while (time.time() - start_time) < timeout:
            try:
                status = await self.custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name,
                )
                # Check both conditions:
                # 1. Ready condition is True
                # 2. State is successful
                status_obj = status.get("status", {})
                conditions = status_obj.get("conditions", [])
                current_state = status_obj.get("state", "unknown")

                print(f"Current deployment state: {current_state}")
                print(f"Current conditions: {conditions}")

                ready_condition = False
                for condition in conditions:
                    if (
                        condition.get("type") == "Ready"
                        and condition.get("status") == "True"
                    ):
                        ready_condition = True
                        break

                state_successful = status_obj.get("state") == "successful"

                if ready_condition and state_successful:
                    print(
                        "Deployment is ready: Ready condition is True and state is successful"
                    )
                    return True
                else:
                    print(
                        f"Deployment not ready yet - Ready condition: {ready_condition}, State successful: {state_successful}"
                    )

            except kubernetes.client.rest.ApiException:
                pass
            await asyncio.sleep(20)
        raise TimeoutError("Deployment failed to become ready within timeout")

    @contextmanager
    def get_service_url_with_port_forward(self, port: Optional[int] = None):
        """
        Get the service URL with automatic detection of environment.

        When running in Kubernetes: yields the service DNS URL directly (no port forwarding needed)
        When running locally: sets up port forwarding and yields localhost URL
        """
        if self._is_running_in_kubernetes():
            # No port forwarding needed, use service DNS directly
            yield self._get_service_url()
        else:
            # Use port forwarding for local development - delegate to existing method
            with self.port_forward(port) as forwarded_port:
                yield f"http://localhost:{forwarded_port}"

    @contextmanager
    def port_forward(self, port: Optional[int] = None):
        """
        Forward the service's HTTP port to a local port.
        """
        if port is None:
            # Find a free port in the ephemeral port range
            for _ in range(100):  # Try up to 100 times
                candidate_port = random.randint(49152, 65535)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("localhost", candidate_port))
                        port = candidate_port
                        break
                    except OSError:
                        continue  # Port is in use, try another
            if port is None:
                raise RuntimeError("Could not find a free port after 100 attempts")
        svc_name = f"{self.deployment_name}-frontend"
        # Get the Service and forward its HTTP port (8000)
        service = Service.get(svc_name, namespace=self.namespace)
        pf = service.portforward(remote_port=8000, local_port=port)
        pf.start()
        try:
            yield port
        finally:
            pf.stop()

    async def check_chat_completion(self):
        """
        Test the deployment with a chat completion request using httpx.
        """
        EXAMPLE_CHAT_REQUEST["model"] = self.model_name
        with self.get_service_url_with_port_forward() as base_url:
            url = f"{base_url}/v1/chat/completions"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=EXAMPLE_CHAT_REQUEST)
                response.raise_for_status()
                return response.text

    async def get_deployment_logs(self):
        """
        Get logs from all pods in the deployment, organized by component.
        """
        # Create logs directory
        base_dir = self.base_log_dir / self.deployment_name
        base_dir.mkdir(parents=True, exist_ok=True)

        for component in self.components:
            component_dir = base_dir / component
            component_dir.mkdir(exist_ok=True)

            # List pods for this component using the selector label
            # nvidia.com/selector: deployment-name-component
            label_selector = (
                f"nvidia.com/selector={self.deployment_name}-{component.lower()}"
            )

            pods = await self.core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=label_selector
            )

            # Get logs for each pod
            for i, pod in enumerate(pods.items):
                try:
                    logs = await self.core_api.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=self.namespace
                    )
                    async with aiofiles.open(component_dir / f"{i}.log", "w") as f:
                        await f.write(logs)
                except kubernetes.client.rest.ApiException as e:
                    print(f"Error getting logs for pod {pod.metadata.name}: {e}")

    async def delete_deployment(self):
        """
        Delete the DynamoGraphDeployment CR.
        """
        try:
            await self.custom_api.delete_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                name=self.deployment_name,
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise


async def main():
    parser = argparse.ArgumentParser(
        description="Deploy and manage DynamoGraphDeployment CRDs"
    )
    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Kubernetes namespace to deploy to (default: default)",
    )
    parser.add_argument(
        "--yaml-file",
        "-f",
        required=True,
        help="Path to the DynamoGraphDeployment YAML file",
    )
    parser.add_argument(
        "--log-dir",
        "-l",
        default="/tmp/dynamo_logs",
        help="Base directory for logs (default: /tmp/dynamo_logs)",
    )

    args = parser.parse_args()

    # Example usage with parsed arguments
    client = DynamoDeploymentClient(namespace=args.namespace, base_log_dir=args.log_dir)

    try:
        # Create deployment from yaml file
        await client.create_deployment(args.yaml_file)

        # Wait for deployment to be ready
        print("Waiting for deployment to be ready...")
        await client.wait_for_deployment_ready()
        print("Deployment is ready!")

        # Test chat completion
        print("Testing chat completion...")
        response = await client.check_chat_completion()
        print(f"Chat completion response: {response}")

        # Get logs
        print("Getting deployment logs...")
        await client.get_deployment_logs()
        print(
            f"Logs have been saved to {client.base_log_dir / client.deployment_name}!"
        )

    finally:
        # Cleanup
        print("Cleaning up deployment...")
        await client.delete_deployment()
        print("Deployment deleted!")


# run with:
# uv run benchmarks/profiler/utils/dynamo_deployment.py -n mo-dyn-cloud -f ./examples/vllm/deploy/agg.yaml -l ./client_logs
if __name__ == "__main__":
    asyncio.run(main())
