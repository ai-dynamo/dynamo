#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import os
import subprocess
import sys

import click
import rich
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import (
    DEFAULT_ENDPOINT,
    CloudClientConfig,
    CloudClientContext,
)
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import add_experimental_docstring
from bentoml._internal.utils.cattr import bentoml_cattr
from bentoml.exceptions import CLIException, CloudRESTApiClientError

from dynamo.sdk.lib.binary_manager import ensure_helm, ensure_kubectl


def build_cloud_command() -> click.Group:
    @click.group(name="server")
    @add_experimental_docstring
    def cloud_command():
        """Interact with your Dynamo Server"""

    @cloud_command.command()
    @click.argument("namespace", type=click.STRING)
    @click.option(
        "--sha",
        type=click.STRING,
        help="SHA of the commit to deploy",
        required=True,
    )
    @click.option(
        "--ngc-token",
        type=click.STRING,
        help="User's NGC API token",
        required=True,
    )
    @click.option(
        "--api-token",
        type=click.STRING,
        help="Dynamo Server user API token",
        required=True,
    )
    @click.option(
        "--endpoint-suffix",
        type=click.STRING,
        help="Dynamo Server endpoint suffix",
        default="dev.aire.nvidia.com",
        show_default=True,
        required=False,
    )
    def create(namespace: str, sha: str, ngc_token: str, api_token: str, endpoint_suffix: str) -> None:  # type: ignore
        """Create a deployment for a given namespace and SHA"""
        try:
            # Set up environment variables
            rich.print(":gear: Setting up environment variables...")
            os.environ["NAMESPACE"] = namespace
            os.environ["NGC_TOKEN"] = ngc_token
            os.environ["CI_COMMIT_SHA"] = sha
            os.environ["RELEASE_NAME"] = namespace
            os.environ["DYNAMO_INGRESS_SUFFIX"] = endpoint_suffix

            # Ensure helm and kubectl are available
            helm_path = ensure_helm()
            kubectl_path = ensure_kubectl()

            # Add helm repositories
            rich.print(":gear: Adding helm repositories...")
            subprocess.run(
                [
                    helm_path,
                    "repo",
                    "add",
                    "bitnami",
                    "https://charts.bitnami.com/bitnami",
                ],
                check=True,
            )
            subprocess.run(
                [
                    helm_path,
                    "repo",
                    "add",
                    "nats",
                    "https://nats-io.github.io/k8s/helm/charts/",
                ],
                check=True,
            )
            subprocess.run([helm_path, "repo", "update"], check=True)

            # Create namespace if it doesn't exist
            rich.print(f":gear: Creating namespace {namespace} if it doesn't exist...")
            try:
                subprocess.run(
                    [kubectl_path, "create", "namespace", namespace], check=True
                )
                rich.print(f":white_check_mark: Created namespace {namespace}")
            except subprocess.CalledProcessError as e:
                if "already exists" not in str(e):
                    raise
                rich.print(f":white_check_mark: Namespace {namespace} already exists")

            # Set current context to the namespace
            rich.print(f":gear: Setting current context to namespace {namespace}...")
            subprocess.run(
                [
                    kubectl_path,
                    "config",
                    "set-context",
                    "--current",
                    "--namespace",
                    namespace,
                ],
                check=True,
            )

            # Install etcd and nats
            rich.print(":gear: Installing etcd and nats...")
            subprocess.run(
                [
                    helm_path,
                    "install",
                    "etcd",
                    "bitnami/etcd",
                    "-n",
                    namespace,
                    "-f",
                    "etcd.yaml",
                ],
                check=True,
            )
            subprocess.run(
                [
                    helm_path,
                    "install",
                    "my-nats",
                    "nats/nats",
                    "--version",
                    "1.2.9",
                    "-f",
                    "nats.yaml",
                    "-n",
                    namespace,
                ],
                check=True,
            )

            # Deploy the helm charts
            rich.print(":gear: Deploying helm charts...")
            subprocess.run(["./deploy.sh"], check=True)

            # Verify user access at the end
            rich.print(":gear: Verifying user access...")
            endpoint = f"https://{namespace}.{endpoint_suffix}"
            login(endpoint, api_token)

            rich.print(":white_check_mark: Deployment completed successfully!")

        except subprocess.CalledProcessError as e:
            rich.print(
                f":police_car_light: Error during deployment: {str(e)}", file=sys.stderr
            )
            sys.exit(1)
        except (CLIException, CloudRESTApiClientError) as e:
            rich.print(
                f":police_car_light: Error during deployment: {str(e)}", file=sys.stderr
            )
            sys.exit(1)

    @cloud_command.command()
    @click.option(
        "--endpoint",
        type=click.STRING,
        help="Dynamo Server endpoint",
        default=DEFAULT_ENDPOINT,
        envvar="DYNAMO_SERVER_API_ENDPOINT",
        show_default=True,
        show_envvar=True,
        required=True,
    )
    @click.option(
        "--api-token",
        type=click.STRING,
        help="Dynamo Server user API token",
        envvar="DYNAMO_SERVER_API_KEY",
        show_envvar=True,
        required=True,
    )
    def login(endpoint: str, api_token: str) -> None:  # type: ignore
        """Connect to your Dynamo Server. You can find deployment instructions for this in our docs"""
        try:
            cloud_rest_client = RestApiClient(endpoint, api_token)
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
                endpoint=endpoint,
                api_token=api_token,
                email=user.email,
            )

            ctx.save()
            rich.print(
                f":white_check_mark: Configured Dynamo Cloud credentials (current-context: {ctx.name})"
            )
            rich.print(
                f":white_check_mark: Logged in as [blue]{user.email}[/] at [blue]{org.name}[/] organization"
            )
        except CloudRESTApiClientError as e:
            if e.error_code == 401:
                rich.print(
                    f":police_car_light: Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)",
                    file=sys.stderr,
                )
            else:
                rich.print(
                    f":police_car_light: Error validating token: HTTP {e.error_code}",
                    file=sys.stderr,
                )

    @cloud_command.command()
    def current_context() -> None:  # type: ignore
        """Get current cloud context."""
        rich.print_json(
            data=bentoml_cattr.unstructure(CloudClientConfig.get_config().get_context())
        )

    @cloud_command.command()
    def list_context() -> None:  # type: ignore
        """List all available context."""
        config = CloudClientConfig.get_config()
        rich.print_json(
            data=bentoml_cattr.unstructure([i.name for i in config.contexts])
        )

    @cloud_command.command()
    @click.argument("context_name", type=click.STRING)
    def update_current_context(context_name: str) -> None:  # type: ignore
        """Update current context"""
        ctx = CloudClientConfig.get_config().set_current_context(context_name)
        rich.print(f"Successfully switched to context: {ctx.name}")

    return cloud_command


cloud_command = build_cloud_command()
