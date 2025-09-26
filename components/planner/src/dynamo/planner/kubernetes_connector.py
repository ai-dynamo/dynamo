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
import os
from typing import Optional

from pydantic import BaseModel

from dynamo.planner.defaults import (
    SubComponentType,
    get_service_from_sub_component_type_or_name,
)
from dynamo.planner.kube import KubernetesAPI
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.planner.utils.exceptions import (
    ComponentError,
    DeploymentValidationError,
    EmptyTargetReplicasError,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class TargetReplica(BaseModel):
    sub_component_type: SubComponentType
    component_name: Optional[str] = None
    desired_replicas: int


class KubernetesConnector(PlannerConnector):
    def __init__(self, dynamo_namespace: str, k8s_namespace: Optional[str] = None):
        self.kube_api = KubernetesAPI(k8s_namespace)
        self.dynamo_namespace = dynamo_namespace

        graph_deployment_name = os.getenv("DYN_PARENT_DGD_K8S_NAME")
        if not graph_deployment_name:
            raise DeploymentValidationError(
                ["DYN_PARENT_DGD_K8S_NAME environment variable is not set"]
            )

        self.graph_deployment_name = graph_deployment_name

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Add a component by increasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_service_from_sub_component_type_or_name(
            deployment, sub_component_type
        )
        self.kube_api.update_graph_replicas(
            self.graph_deployment_name,
            service.name,
            service.number_replicas() + 1,
        )
        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ):
        """Remove a component by decreasing its replica count by 1"""

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        service = get_service_from_sub_component_type_or_name(
            deployment, sub_component_type
        )
        if service.number_replicas() > 0:
            self.kube_api.update_graph_replicas(
                self.graph_deployment_name,
                service.name,
                service.number_replicas() - 1,
            )
            if blocking:
                await self.kube_api.wait_for_graph_deployment_ready(
                    self.graph_deployment_name,
                )

    async def verify_prefill_and_decode_components_exist(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ):
        """
        Verify that the deployment contains services with subComponentType prefill and decode.
        Will fallback to worker service names for backwards compatibility. (TODO: deprecate)

        Raises:
            DeploymentValidationError: If the deployment does not contain services with subComponentType prefill and decode
        """
        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        errors = []

        try:
            get_service_from_sub_component_type_or_name(
                deployment,
                SubComponentType.PREFILL,
                component_name=prefill_component_name,
            )
        except ComponentError as e:
            errors.append(str(e))

        try:
            get_service_from_sub_component_type_or_name(
                deployment,
                SubComponentType.DECODE,
                component_name=decode_component_name,
            )
        except ComponentError as e:
            errors.append(str(e))

        # Raise combined error if any issues found
        if errors:
            raise DeploymentValidationError(errors)

    async def wait_for_deployment_ready(
        self,
        max_attempts: int = 180,  # default: 30 minutes total
        delay_seconds: int = 10,  # default: check every 10 seconds
    ):
        """Wait for the deployment to be ready"""
        await self.kube_api.wait_for_graph_deployment_ready(
            self.graph_deployment_name,
            max_attempts,
            delay_seconds,
        )

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ):
        """Set the replicas for multiple components at once"""
        if not target_replicas:
            raise EmptyTargetReplicasError()

        deployment = self.kube_api.get_graph_deployment(self.graph_deployment_name)

        if not self.kube_api.is_deployment_ready(deployment):
            logger.warning(
                f"Deployment {self.graph_deployment_name} is not ready, ignoring this scaling"
            )
            return

        for target_replica in target_replicas:
            service = get_service_from_sub_component_type_or_name(
                deployment,
                target_replica.sub_component_type,
                component_name=target_replica.component_name,
            )
            current_replicas = service.number_replicas()
            if current_replicas != target_replica.desired_replicas:
                logger.info(
                    f"Updating {target_replica.sub_component_type.value} component {service.name} to desired replica count {target_replica.desired_replicas}"
                )
                self.kube_api.update_graph_replicas(
                    self.graph_deployment_name,
                    service.name,
                    target_replica.desired_replicas,
                )
            else:
                logger.info(
                    f"{target_replica.sub_component_type.value} component {service.name} already at desired replica count {target_replica.desired_replicas}, skipping"
                )

        if blocking:
            await self.kube_api.wait_for_graph_deployment_ready(
                self.graph_deployment_name,
            )


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamo_namespace", type=str, default="dynamo")
    parser.add_argument("--k8s_namespace", type=str, default="default")
    parser.add_argument("--action", type=str, choices=["add", "remove"])
    parser.add_argument(
        "--component",
        type=str,
        choices=[t.value for t in SubComponentType],
        default=SubComponentType.PREFILL.value,
        help="Target sub-component to scale",
    )
    parser.add_argument("--blocking", action="store_true")
    args = parser.parse_args()
    connector = KubernetesConnector(args.dynamo_namespace, args.k8s_namespace)

    if args.action == "add":
        task = connector.add_component(SubComponentType(args.component), args.blocking)
    elif args.action == "remove":
        task = connector.remove_component(
            SubComponentType(args.component), args.blocking
        )
    asyncio.run(task)
