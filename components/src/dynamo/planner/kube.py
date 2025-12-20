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

import asyncio
import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.utils.exceptions import DynamoGraphDeploymentNotFoundError
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def get_current_k8s_namespace() -> str:
    """Get the current namespace if running inside a k8s cluster"""
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to 'default' if not running in k8s
        return "default"


class KubernetesAPI:
    def __init__(self, k8s_namespace: Optional[str] = None):
        # Load kubernetes configuration
        try:
            config.load_incluster_config()  # for in-cluster deployment
        except ConfigException:
            config.load_kube_config()  # for out-of-cluster deployment

        self.custom_api = client.CustomObjectsApi()
        self.current_namespace = k8s_namespace or get_current_k8s_namespace()

    def _get_graph_deployment_from_name(self, graph_deployment_name: str) -> dict:
        """Get the graph deployment from the dynamo graph deployment name"""
        return self.custom_api.get_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
            name=graph_deployment_name,
        )

    def get_graph_deployment(self, graph_deployment_name: str) -> dict:
        """
        Get the parent DynamoGraphDeployment

        Returns:
            The DynamoGraphDeployment object

        Raises:
            DynamoGraphDeploymentNotFoundError: If the parent graph deployment is not found
        """
        try:
            return self._get_graph_deployment_from_name(graph_deployment_name)
        except client.ApiException as e:
            if e.status == 404:
                raise DynamoGraphDeploymentNotFoundError(
                    deployment_name=graph_deployment_name,
                    namespace=self.current_namespace,
                )
            raise

    def update_graph_replicas(
        self, graph_deployment_name: str, component_name: str, replicas: int
    ) -> None:
        """Update the replicas count for a component in a DynamoGraphDeployment"""
        patch = {"spec": {"services": {component_name: {"replicas": replicas}}}}
        self.custom_api.patch_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.current_namespace,
            plural="dynamographdeployments",
            name=graph_deployment_name,
            body=patch,
        )

    def is_deployment_ready(self, deployment: dict) -> bool:
        """Check if a graph deployment is ready"""

        conditions = deployment.get("status", {}).get("conditions", [])
        ready_condition = next(
            (c for c in conditions if c.get("type") == "Ready"), None
        )

        return ready_condition is not None and ready_condition.get("status") == "True"

    async def wait_for_graph_deployment_ready(
        self,
        graph_deployment_name: str,
        max_attempts: int = 180,  # default: 30 minutes total
        delay_seconds: int = 10,  # default: check every 10 seconds
    ) -> None:
        """Wait for a graph deployment to be ready"""

        for attempt in range(max_attempts):
            await asyncio.sleep(delay_seconds)

            graph_deployment = self.get_graph_deployment(graph_deployment_name)

            conditions = graph_deployment.get("status", {}).get("conditions", [])
            ready_condition = next(
                (c for c in conditions if c.get("type") == "Ready"), None
            )

            if ready_condition and ready_condition.get("status") == "True":
                return  # Deployment is ready

            logger.info(
                f"[Attempt {attempt + 1}/{max_attempts}] "
                f"(status: {ready_condition.get('status') if ready_condition else 'N/A'}, "
                f"message: {ready_condition.get('message') if ready_condition else 'no condition found'})"
            )

        # Raise after all attempts exhausted (without additional delay)
        raise TimeoutError(
            f"Graph deployment '{graph_deployment_name}' "
            f"is not ready after {max_attempts * delay_seconds} seconds"
        )
