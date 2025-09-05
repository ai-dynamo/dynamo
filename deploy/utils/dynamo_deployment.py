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
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles  # type: ignore[import-untyped]
import httpx  # added for HTTP requests
import kubernetes_asyncio as kubernetes
import yaml
from kubernetes_asyncio import client, config


def find_available_port(start_port: int = 8000) -> int:
    """Find the first available TCP port on 127.0.0.1 starting at start_port (inclusive), scanning up to start_port+99."""
    for port in range(
        start_port, start_port + 100
    ):  # Try ports start_port..start_port+99
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port+99}"
    )


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


class ProgressDisplay:
    """Helper class for cleaner progress display during deployment waiting"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_message = ""
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0

    def update(self, message: str, newline: bool = False):
        """Update progress display"""
        if self.verbose or newline:
            print(message)
        else:
            # Clear previous line and write new message
            sys.stdout.write(f"\r\033[K{message}")
            sys.stdout.flush()
            self.last_message = message

    def spinner(self) -> str:
        """Get next spinner character"""
        char = self.spinner_chars[self.spinner_idx]
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
        return char

    def finish(self, message: str):
        """Finish with a final message"""
        if not self.verbose and self.last_message:
            sys.stdout.write("\r\033[K")  # Clear the line
        print(message)


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
        self.components: List[str] = []  # Will store component names from CR
        self.deployment_spec: Optional[
            Dict[str, Any]
        ] = None  # Will store the full deployment spec
        self.base_log_dir = Path(base_log_dir) if base_log_dir else Path("logs")
        self.frontend_port = frontend_port
        self.port_forward_process: Optional[subprocess.Popen[bytes]] = None

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()

        self.k8s_client = client.ApiClient()
        self.custom_api = client.CustomObjectsApi(self.k8s_client)
        self.core_api = client.CoreV1Api(self.k8s_client)

    def port_forward_frontend(
        self, local_port: Optional[int] = None, quiet: bool = False
    ) -> str:
        """
        Port forward the frontend service to a local port.

        Args:
            local_port: Local port to forward to (if None, find first available port starting from 8000)
            quiet: If True, suppress kubectl port-forward output messages (default: False)
        """
        if local_port is None:
            local_port = find_available_port(8000)
            if not quiet:
                print(f"Using available local port: {local_port}")

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
        try:
            self.port_forward_process = subprocess.Popen(
                cmd, stdout=stdout, stderr=stderr
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "kubectl not found in PATH; required for port-forwarding"
            ) from e

        # Wait a moment for port forward to establish
        print("Waiting for port forward to establish...")
        time.sleep(3)

        print(f"Port forward started with PID: {self.port_forward_process.pid}")
        return f"http://localhost:{local_port}"

    def stop_port_forward(self):
        """
        Stop the port forward process.
        """
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
        await self._init_kubernetes()

        if isinstance(deployment, str):
            # Load from yaml file
            async with aiofiles.open(deployment, "r") as f:
                content = await f.read()
                self.deployment_spec = yaml.safe_load(content)
        else:
            self.deployment_spec = deployment

        # Ensure deployment_spec is properly loaded
        assert (
            self.deployment_spec is not None
        ), "Failed to load deployment specification"

        # Extract component names
        self.components = [
            svc.lower() for svc in self.deployment_spec["spec"]["services"].keys()
        ]

        # Ensure name and namespace are set correctly
        self.deployment_spec["metadata"]["name"] = self.deployment_name
        self.deployment_spec["metadata"]["namespace"] = self.namespace

        # Disable grove as it will cause the deployment to not report ready
        if "annotations" not in self.deployment_spec["metadata"]:
            self.deployment_spec["metadata"]["annotations"] = {}
        self.deployment_spec["metadata"]["annotations"][
            "nvidia.com/enable-grove"
        ] = "false"

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

    async def wait_for_deployment_ready(
        self, timeout: int = 1800, verbose: Optional[bool] = None
    ):
        """
        Wait for the custom resource to be ready with improved progress display.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
            verbose: If True, show detailed status updates. If None, uses DYNAMO_VERBOSE env var.
        """
        # Allow environment variable to control verbosity
        if verbose is None:
            verbose = os.environ.get("DYNAMO_VERBOSE", "false").lower() == "true"

        progress = ProgressDisplay(verbose=verbose)
        start_time = time.time()
        last_status = None
        last_conditions_str = ""
        check_interval = 20 if not verbose else 10

        # Initial message
        if not verbose:
            print(f"⏳ Waiting for deployment '{self.deployment_name}'...")

        while (time.time() - start_time) < timeout:
            try:
                status = await self.custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self.deployment_name,
                )

                status_obj = status.get("status", {})
                conditions = status_obj.get("conditions", [])
                current_state = status_obj.get("state", "unknown")
                elapsed = time.time() - start_time

                # Check readiness
                ready_condition = False
                ready_message = ""
                for condition in conditions:
                    if condition.get("type") == "Ready":
                        ready_condition = condition.get("status") == "True"
                        ready_message = condition.get("message", "")
                        break

                state_successful = current_state == "successful"

                # Extract not ready components from message
                not_ready_components = []
                if re.search(r"resources not ready:", ready_message, re.IGNORECASE):
                    match = re.search(r"\[(.*?)\]", ready_message)
                    if match:
                        items = match.group(1)
                        not_ready_components = [
                            s.strip() for s in re.split(r"[,\s]+", items) if s.strip()
                        ]

                # Format progress message based on mode
                if not verbose:
                    # Concise single-line progress with spinner
                    spinner = progress.spinner()

                    # Create status string
                    if not_ready_components:
                        # Show first 2 components, abbreviate if more
                        components_str = ", ".join(not_ready_components[:2])
                        if len(not_ready_components) > 2:
                            components_str += f" +{len(not_ready_components)-2} more"
                        status_str = f"Waiting for: {components_str}"
                    else:
                        status_str = f"State: {current_state}"

                    # Format time
                    time_str = f"[{elapsed:.0f}s]"

                    message = f"{spinner} {time_str} {status_str}"
                    progress.update(message)

                else:
                    # Verbose mode - show details when status changes
                    conditions_str = str(conditions)
                    if (
                        current_state != last_status
                        or conditions_str != last_conditions_str
                    ):
                        progress.update(f"Current deployment state: {current_state}")
                        progress.update(f"Current conditions: {conditions}")
                        progress.update(f"Elapsed time: {elapsed:.1f}s / {timeout}s")
                        progress.update(
                            f"Deployment not ready yet - Ready: {ready_condition}, "
                            f"State successful: {state_successful}"
                        )
                        last_status = current_state
                        last_conditions_str = conditions_str

                # Check if deployment is ready
                if ready_condition and state_successful:
                    progress.finish(
                        f"✅ Deployment '{self.deployment_name}' ready after {elapsed:.1f}s"
                    )
                    return True

            except kubernetes.client.rest.ApiException as e:
                if verbose:
                    progress.update(
                        f"API Exception while checking deployment status: {e}",
                        newline=True,
                    )
                    progress.update(
                        f"Status code: {e.status}, Reason: {e.reason}", newline=True
                    )
            except Exception as e:
                if verbose:
                    progress.update(
                        f"Unexpected exception while checking deployment status: {e}",
                        newline=True,
                    )

            await asyncio.sleep(check_interval)

        # Timeout reached
        progress.finish(
            f"❌ Deployment '{self.deployment_name}' failed to become ready within {timeout}s"
        )
        raise TimeoutError(f"Deployment failed to become ready within {timeout}s")

    async def check_chat_completion(
        self,
        use_port_forward: bool = False,
        local_port: int = 8000,
        quiet: bool = True,
        timeout_s: float = 30.0,
    ):
        """
        Test the deployment with a chat completion request using httpx.
        """
        EXAMPLE_CHAT_REQUEST["model"] = self.model_name

        # Use cluster DNS in-cluster; otherwise optionally port-forward
        inside_cluster = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
        base_url = self.get_service_url()
        if use_port_forward or not inside_cluster:
            base_url = self.port_forward_frontend(local_port=local_port, quiet=quiet)

        url = f"{base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=EXAMPLE_CHAT_REQUEST)
                response.raise_for_status()
                return response.text
        finally:
            if use_port_forward or not inside_cluster:
                self.stop_port_forward()

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
        finally:
            # Close the kubernetes client session to avoid warnings
            if hasattr(self, "k8s_client"):
                await self.k8s_client.close()


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
        response = await client.check_chat_completion(use_port_forward=True)
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
