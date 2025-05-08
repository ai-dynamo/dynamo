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

from .kube import KubernetesAPI
from .planner_connector import PlannerConnector


class KubernetesConnector(PlannerConnector):
    def __init__(self, namespace: str):
        self.kube_api = KubernetesAPI()
        self.namespace = namespace

    async def add_component(self, component_name: str):
        """Add a component by increasing its replica count by 1"""
        deployment = await self.kube_api.get_graph_deployment(
            component_name, self.namespace
        )
        if deployment is None:
            raise ValueError(
                f"Graph not found for component {component_name} in dynamo namespace {self.namespace}"
            )
        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        await self.kube_api.update_graph_replicas(
            self._get_graph_deployment_name(deployment),
            component_name,
            current_replicas + 1,
        )

    async def remove_component(self, component_name: str):
        """Remove a component by decreasing its replica count by 1"""
        deployment = await self.kube_api.get_graph_deployment(
            component_name, self.namespace
        )
        if deployment is None:
            raise ValueError(
                f"Graph {component_name} not found for namespace {self.namespace}"
            )
        # get current replicas or 1 if not found
        current_replicas = self._get_current_replicas(deployment, component_name)
        if current_replicas > 0:
            await self.kube_api.update_graph_replicas(
                self._get_graph_deployment_name(deployment),
                component_name,
                current_replicas - 1,
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
