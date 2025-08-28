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

import logging
from typing import Optional

from dynamo.planner.kube import KubernetesAPI
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class KubernetesConnector(PlannerConnector):
    def __init__(self, dynamo_namespace: str, k8s_namespace: Optional[str] = None):
        self.kube_api = KubernetesAPI(k8s_namespace)
        self.dynamo_namespace = dynamo_namespace

    async def add_component(self, component_name: str, blocking: bool = True):
        """Add a component by increasing its replica count by 1"""

        deployment = await self.kube_api.get_graph_deployment()
        if deployment is None:
            raise ValueError("Parent DynamoGraphDeployment not found")

        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        await self.kube_api.update_graph_replicas(
            self._get_graph_deployment_name(deployment),
            component_name,
            current_replicas + 1,
        )
        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self._get_graph_deployment_name(deployment)
            )

    async def remove_component(self, component_name: str, blocking: bool = True):
        """Remove a component by decreasing its replica count by 1"""

        deployment = await self.kube_api.get_graph_deployment()
        if deployment is None:
            raise ValueError("Parent DynamoGraphDeployment not found")

        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        if current_replicas > 0:
            await self.kube_api.update_graph_replicas(
                self._get_graph_deployment_name(deployment),
                component_name,
                current_replicas - 1,
            )
            if blocking:
                await self.kube_api.wait_for_graph_deployment_ready(
                    self._get_graph_deployment_name(deployment)
                )

    async def set_component_replicas(
        self, target_replicas: dict[str, int], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise ValueError("target_replicas cannot be empty")

        deployment = await self.kube_api.get_graph_deployment()
        if deployment is None:
            raise ValueError("Parent DynamoGraphDeployment not found")

        if not await self.kube_api.is_deployment_ready(
            self._get_graph_deployment_name(deployment)
        ):
            logger.warning(
                f"Deployment {self._get_graph_deployment_name(deployment)} is not ready, ignoring this scaling"
            )
            return

        for component_name, replicas in target_replicas.items():
            await self.kube_api.update_graph_replicas(
                self._get_graph_deployment_name(deployment),
                component_name,
                replicas,
            )

        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self._get_graph_deployment_name(deployment)
            )

    def _get_current_replicas(self, deployment: dict, component_name: str) -> int:
        """Get the current replicas for a component in a graph deployment"""
        return (
            deployment.get("spec", {})
            .get("services", {})
            .get(component_name, {})
            .get("replicas", 1)
        )

    def _get_graph_deployment_name(self, deployment: dict) -> str:
        """Get the name of the graph deployment"""
        return deployment["metadata"]["name"]


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamo_namespace", type=str, default="dynamo")
    parser.add_argument("--k8s_namespace", type=str, default="default")
    parser.add_argument("--action", type=str, choices=["add", "remove"])
    parser.add_argument("--component", type=str, default="planner")
    parser.add_argument("--blocking", action="store_true")
    args = parser.parse_args()
    connector = KubernetesConnector(args.dynamo_namespace, args.k8s_namespace)

    if args.action == "add":
        task = connector.add_component(args.component, args.blocking)
    elif args.action == "remove":
        task = connector.remove_component(args.component, args.blocking)
    asyncio.run(task)
