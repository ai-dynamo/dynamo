# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional, Union

import aiofiles
import httpx  # added for HTTP requests
import kubernetes_asyncio as kubernetes
import yaml
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
        frontend_port: int = 8000,
        base_log_dir: Optional[str] = None,
        service_name: Optional[str] = None,
    ):
        """
        Initialize the client with the namespace and deployment name.

        Args:
            namespace: The Kubernetes namespace
            deployment_name: Name of the deployment, defaults to vllm-v1-agg
            base_log_dir: Base directory for storing logs, defaults to ./logs if not specified
            service_name: Service name for connecting to the service, defaults to {deployment_name}-frontend
        """
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.service_name = service_name or f"{deployment_name}-frontend"
        self.components: list[str] = []  # Will store component names from CR
        self.deployment_spec: Optional[
            dict
        ] = None  # Will store the full deployment spec
        self.base_log_dir = Path(base_log_dir) if base_log_dir else Path("logs")
        self.frontend_port = frontend_port

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

    def get_service_url(self) -> str:
        """
        Get the service URL using Kubernetes service DNS.
        """
        service_url = f"http://{self.service_name}.{self.namespace}.svc.cluster.local:{self.frontend_port}"
        print(f"Using service URL: {service_url}")
        return service_url

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
            print(f"Successfully created deployment {self.deployment_name}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                print(f"Deployment {self.deployment_name} already exists")
            else:
                print(f"Failed to create deployment {self.deployment_name}: {e}")
                raise

    async def wait_for_deployment_ready(self, timeout: int = 1800):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
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
                print(f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s")

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

            except kubernetes.client.rest.ApiException as e:
                print(f"API Exception while checking deployment status: {e}")
                print(f"Status code: {e.status}, Reason: {e.reason}")
            except Exception as e:
                print(f"Unexpected exception while checking deployment status: {e}")
            await asyncio.sleep(20)
        raise TimeoutError("Deployment failed to become ready within timeout")

    async def check_chat_completion(self):
        """
        Test the deployment with a chat completion request using httpx.
        """
        EXAMPLE_CHAT_REQUEST["model"] = self.model_name
        base_url = self.get_service_url()
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


async def cleanup_remaining_deployments(deployment_clients, namespace):
    """Clean up any remaining tracked deployments, handling errors gracefully."""
    import logging

    logger = logging.getLogger(__name__)

    if not deployment_clients:
        logger.info("No deployments to clean up")
        return

    logger.info(f"Cleaning up {len(deployment_clients)} remaining deployments...")
    for deployment_client in deployment_clients:
        try:
            logger.info(
                f"Attempting to delete deployment {deployment_client.deployment_name}..."
            )
            await deployment_client.delete_deployment()
            logger.info(
                f"Successfully deleted deployment {deployment_client.deployment_name}"
            )
        except Exception as e:
            # If deployment doesn't exist (404), that's fine - it was already cleaned up
            if "404" in str(e) or "not found" in str(e).lower():
                logger.info(
                    f"Deployment {deployment_client.deployment_name} was already deleted"
                )
            else:
                logger.error(
                    f"Failed to delete deployment {deployment_client.deployment_name}: {e}"
                )


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
    parser.add_argument(
        "--service-name",
        "-s",
        help="Service name for connecting to the service (default: {deployment_name}-frontend)",
    )

    args = parser.parse_args()

    # Example usage with parsed arguments
    client = DynamoDeploymentClient(
        namespace=args.namespace,
        base_log_dir=args.log_dir,
        service_name=args.service_name,
    )

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
