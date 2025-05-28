# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing as t

from bentoml._internal.cloud import BentoCloudClient
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import CloudClientConfig, CloudClientContext
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException, CLIException, CloudRESTApiClientError
from rich.console import Console

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk.core.protocol.deployment import Deployment as ProtocolDeployment
from dynamo.sdk.core.protocol.deployment import (
    DeploymentManager,
    DeploymentResponse,
    DeploymentStatus,
)

# Configure logging to suppress INFO HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTP client library logs
logging.getLogger("httpcore").setLevel(logging.WARNING)  # HTTP core library logs
configure_dynamo_logging()

logger = logging.getLogger(__name__)
console = Console(highlight=False)


class BentoCloudDeploymentManager(DeploymentManager):
    """
    Implementation of DeploymentManager that talks to the BentoCloud deployment API.
    Handles all BentoCloud-specific config parameter building, error handling, and API calls.
    Accepts **kwargs for backend-specific options.
    Raises exceptions for errors; CLI handles user interaction.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self._cloud_client = self._login_to_cloud()

    def _login_to_cloud(self) -> "BentoCloudClient":
        """Connect to Dynamo Cloud and return an authenticated BentoCloudClient."""
        try:
            logger.info(f"Running against Dynamo Cloud at {self.endpoint}")
            api_token = ""  # Using empty string for now as it's not used
            cloud_rest_client = RestApiClient(self.endpoint, api_token)
            user = cloud_rest_client.v1.get_current_user()
            if user is None:
                raise CLIException("current user is not found")
            org = cloud_rest_client.v1.get_current_organization()
            if org is None:
                raise CLIException("current organization is not found")
            current_context_name = CloudClientConfig.get_config().current_context_name
            cloud_context = BentoMLContainer.cloud_context.get()
            ctx = CloudClientContext(
                name=cloud_context
                if cloud_context is not None
                else current_context_name,
                endpoint=self.endpoint,
                api_token=api_token,
                email=user.email,
            )
            ctx.save()
            logger.debug(
                f"Configured Dynamo Cloud credentials (current-context: {ctx.name})"
            )
            logger.debug(f"Logged in as {user.email} at {org.name} organization")

            return BentoCloudClient(endpoint=self.endpoint, api_key=api_token)
        except CloudRESTApiClientError as e:
            if e.error_code == 401:
                console.print(
                    f":police_car_light: Error validating token: HTTP 401: Bad credentials ({self.endpoint}/api-token)"
                )
            else:
                console.print(
                    f":police_car_light: Error validating token: HTTP {e.error_code}"
                )
            raise BentoMLException(f"Failed to login to Dynamo Cloud: {str(e)}") from e
        except Exception as e:
            console.print(
                f":police_car_light: Error connecting to Dynamo Cloud: {str(e)}"
            )
            raise BentoMLException(f"Failed to login to Dynamo Cloud: {str(e)}") from e

    def create_deployment(
        self, deployment: ProtocolDeployment, **kwargs
    ) -> DeploymentResponse:
        dev = kwargs.get("dev", False)

        config_params = DeploymentConfigParameters(
            name=deployment.name,
            bento=deployment.pipeline or deployment.namespace,
            envs=deployment.envs,
            secrets=None,
            cli=True,
            dev=dev,
        )
        try:
            config_params.verify()
        except BentoMLException as e:
            raise RuntimeError((400, f"Config verification error: {str(e)}", None))
        try:
            deployment_obj = self._cloud_client.deployment.create(
                deployment_config_params=config_params
            )
            return deployment_obj.to_dict()
        except BentoMLException as e:
            error_msg = str(e)
            if "already exists" in error_msg:
                raise RuntimeError((409, error_msg, None)) from e
            raise RuntimeError((500, error_msg, None)) from e

    def update_deployment(
        self, deployment_id: str, deployment: ProtocolDeployment
    ) -> DeploymentResponse:
        config_params = DeploymentConfigParameters(
            name=deployment_id,
            envs=deployment.envs,
            cli=True,
        )
        try:
            config_params.verify(create=False)
        except BentoMLException as e:
            raise RuntimeError((400, f"Config verification error: {str(e)}", None))
        try:
            deployment = self._cloud_client.deployment.update(
                deployment_config_params=config_params
            )
            return deployment.to_dict()
        except BentoMLException as e:
            raise RuntimeError((500, f"Deployment update error: {str(e)}", None)) from e

    def get_deployment(self, deployment_id: str) -> DeploymentResponse:
        try:
            deployment_obj = self._cloud_client.deployment.get(name=deployment_id)
            return deployment_obj.to_dict()
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((404, error_msg, None)) from e

    def list_deployments(self) -> list[DeploymentResponse]:
        try:
            deployments = self._cloud_client.deployment.list()
            return [
                d.to_dict() if hasattr(d, "to_dict") else vars(d) for d in deployments
            ]
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((500, error_msg, None)) from e

    def delete_deployment(self, deployment_id: str) -> None:
        try:
            self._cloud_client.deployment.delete(name=deployment_id)
        except BentoMLException as e:
            error_msg = str(e)
            raise RuntimeError((404, error_msg, None)) from e

    def get_status(
        self,
        deployment_id: str,
    ) -> DeploymentStatus:
        dep = self._cloud_client.deployment.get(deployment_id)
        status = dep._schema.status if dep._schema.status else "unknown"
        # Escape any characters that are interpreted as markup
        status = status.replace("[", "\\[")
        if status == "running":
            return DeploymentStatus.RUNNING
        elif status == "failed":
            return DeploymentStatus.FAILED
        elif status == "deploying":
            return DeploymentStatus.IN_PROGRESS
        elif status == "terminated":
            return DeploymentStatus.TERMINATED
        else:
            return DeploymentStatus.PENDING

    def wait_until_ready(
        self, deployment_id: str, timeout: int = 3600
    ) -> t.Tuple[DeploymentResponse, bool]:
        dep = self._cloud_client.deployment.get(name=deployment_id)
        retcode = dep.wait_until_ready(timeout=timeout)
        if retcode != 0:
            return dep.to_dict(), False
        return dep.to_dict(), True

    def get_endpoint_urls(
        self,
        deployment_id: str,
    ) -> list[str]:
        dep = self.get_deployment(deployment_id)
        latest = self._cloud_client.deployment._client.v2.get_deployment(
            dep["name"], dep["cluster"]
        )
        urls = latest.urls if hasattr(latest, "urls") else None
        return urls if urls is not None else []
