#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
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
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import json
import logging
import re
import sys
import typing as t
from http import HTTPStatus

import click
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import Deployment, DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import add_experimental_docstring
from bentoml.exceptions import BentoMLException
from rich.console import Console
from simple_di import Provide, inject

from dynamo.sdk.lib.logging import configure_server_logging

from .utils import resolve_service_config

# Configure logging to suppress INFO HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTP client library logs
logging.getLogger("httpcore").setLevel(logging.WARNING)  # HTTP core library logs
configure_server_logging()

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* Dynamo Cloud API token is required for authorization. Run `dynamo cloud login` command to login"
        ) from None
    raise BentoMLException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


@click.command(name="deploy")
@click.argument(
    "bento",
    type=click.STRING,
    required=False,
)
@click.option(
    "-n",
    "--name",
    type=click.STRING,
    help="Deployment name",
)
@click.option(
    "-f",
    "--config-file",
    type=click.File(),
    help="Configuration file path",
    default=None,
)
@click.option(
    "--wait/--no-wait",
    type=click.BOOL,
    is_flag=True,
    help="Do not wait for deployment to be ready",
    default=True,
)
@click.option(
    "--timeout",
    type=click.INT,
    default=3600,
    help="Timeout for deployment to be ready in seconds",
)
@click.pass_context
@add_experimental_docstring
def deploy_command(
    ctx: click.Context,
    bento: str | None,
    name: str | None,
    config_file: str | t.TextIO | None,
    wait: bool,
    timeout: int,
) -> None:
    """Create a deployment on BentoCloud.

    \b
    Create a deployment using parameters, or using config yaml file.
    """
    create_deployment(
        bento=bento,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        args=ctx.args,
    )


def build_deployment_command() -> click.Group:
    @click.group(name="deployment")
    @add_experimental_docstring
    def deployment_command():
        """Deploy Dynamo applications to Kubernetes cluster"""

    @deployment_command.command()
    @click.argument(
        "bento",
        type=click.STRING,
        required=False,
    )
    @click.option(
        "-n",
        "--name",
        type=click.STRING,
        help="Deployment name",
    )
    @click.option(
        "-f",
        "--config-file",
        type=click.File(),
        help="Configuration file path",
        default=None,
    )
    @click.option(
        "--wait/--no-wait",
        type=click.BOOL,
        is_flag=True,
        help="Do not wait for deployment to be ready",
        default=True,
    )
    @click.option(
        "--timeout",
        type=click.INT,
        default=3600,
        help="Timeout for deployment to be ready in seconds",
    )
    @click.pass_context
    def create(
        ctx: click.Context,
        bento: str | None,
        name: str | None,
        config_file: str | t.TextIO | None,
        wait: bool,
        timeout: int,
    ) -> None:
        """Create a deployment on Dynamo Cloud.

        \b
        Create a deployment using parameters, or using config yaml file.
        """
        create_deployment(
            bento=bento,
            name=name,
            config_file=config_file,
            wait=wait,
            timeout=timeout,
            args=ctx.args,
        )

    @deployment_command.command()
    @click.argument("name", type=click.STRING)
    @click.option(
        "--cluster",
        type=click.STRING,
        help="Cluster name",
        default=None,
    )
    def get(name: str, cluster: str | None) -> None:
        """Get deployment details from Dynamo Cloud.

        \b
        Get deployment details by name.
        """
        get_deployment(name, cluster=cluster)

    @deployment_command.command()
    @click.option(
        "--cluster",
        type=click.STRING,
        help="Cluster name",
        default=None,
    )
    @click.option(
        "--search",
        type=click.STRING,
        help="Search query",
        default=None,
    )
    @click.option(
        "--dev",
        is_flag=True,
        help="List development deployments",
        default=False,
    )
    @click.option(
        "-q",
        "--query",
        type=click.STRING,
        help="Advanced query string",
        default=None,
    )
    def list(
        cluster: str | None, search: str | None, dev: bool, query: str | None
    ) -> None:
        """List all deployments from Dynamo Cloud.

        \b
        List and filter deployments.
        """
        list_deployments(cluster=cluster, search=search, dev=dev, q=query)

    @deployment_command.command()
    @click.argument("name", type=click.STRING)
    @click.option(
        "--cluster",
        type=click.STRING,
        help="Cluster name",
        default=None,
    )
    def delete(name: str, cluster: str | None) -> None:
        """Delete a deployment from Dynamo Cloud.

        \b
        Delete deployment by name.
        """
        delete_deployment(name, cluster=cluster)

    return deployment_command


deployment_command = build_deployment_command()


@inject
def create_deployment(
    bento: str | None = None,
    name: str | None = None,
    config_file: str | t.TextIO | None = None,
    wait: bool = True,
    timeout: int = 3600,
    dev: bool = False,
    args: list[str] | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    # Load config from file and serialize to env
    service_configs = resolve_service_config(config_file=config_file, args=args)
    env_dicts = []
    if service_configs:
        config_json = json.dumps(service_configs)
        logger.info(f"Deployment service configuration: {config_json}")
        env_dicts.append({"name": "DYN_DEPLOYMENT_CONFIG", "value": config_json})

    config_params = DeploymentConfigParameters(
        name=name,
        bento=bento,
        envs=env_dicts,
        secrets=None,
        cli=True,
        dev=dev,
    )

    try:
        config_params.verify()
    except BentoMLException as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        try:
            # Create deployment with initial status message
            spinner.update("Creating deployment on Dynamo Cloud...")
            deployment = _cloud_client.deployment.create(
                deployment_config_params=config_params
            )
            deployment.admin_console = _get_urls(deployment)  # remove dashboard url
            spinner.log(
                f':white_check_mark: Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
            )

            if wait:
                # Update spinner text for waiting phase
                spinner.log(
                    "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]"
                )
                retcode = deployment.wait_until_ready(timeout=timeout, spinner=spinner)
                if retcode != 0:
                    sys.exit(retcode)

            _display_deployment_info(spinner, deployment)
            return deployment

        except BentoMLException as e:
            error_msg = str(e)
            if "already exists" in error_msg:
                # Extract deployment name from error message and clean it
                match = re.search(r'"([^"]+?)(?:\\+)?" already exists', error_msg)
                dep_name = match.group(1).rstrip("\\") if match else name
                spinner.log(
                    "[red]:x: Error:[/] "
                    f'Deployment "{dep_name}" already exists. To create a new deployment:\n'
                    "  1. Use a different name with the --name flag\n"
                    f"  2. Or delete the existing deployment with: dynamo deployment delete {dep_name}"
                )
                sys.exit(1)
            spinner.log(f"[red]:x: Error:[/] {str(e)}")
            sys.exit(1)


def _get_urls(deployment: Deployment) -> list[str]:
    """Get URLs from deployment."""
    latest = deployment._client.v2.get_deployment(deployment.name, deployment.cluster)
    urls = latest.urls if hasattr(latest, "urls") else None
    return urls if urls is not None else []


def _display_deployment_info(spinner: Spinner, deployment: Deployment) -> None:
    """Helper function to display deployment status and URLs consistently."""
    # Get status directly from schema and escape any Rich markup
    status = deployment._schema.status if deployment._schema.status else "unknown"
    # Escape any characters that are interpreted as markup
    reformatted_status = status.replace("[", "\\[")
    spinner.log(f"[bold]Status:[/] {reformatted_status}")

    # Get URLs directly from schema
    spinner.log("[bold]Ingress URLs:[/]")
    try:
        # Get latest deployment info for URLs
        urls = _get_urls(deployment)
        if urls:
            for url in urls:
                spinner.log(f"  - {url}")
        else:
            spinner.log("    No URLs available")
    except Exception:
        # If refresh fails, fall back to existing URLs
        if deployment._urls:
            for url in deployment._urls:
                spinner.log(f"  - {url}")
        else:
            spinner.log("    No URLs available")


@inject
def get_deployment(
    name: str,
    cluster: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    """Get deployment details from Dynamo Cloud."""
    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        try:
            spinner.update(f'Getting deployment "{name}" from Dynamo Cloud...')
            deployment = _cloud_client.deployment.get(name=name, cluster=cluster)
            spinner.log(
                f':white_check_mark: Found deployment "{deployment.name}" in cluster "{deployment.cluster}"'
            )
            _display_deployment_info(spinner, deployment)
            return deployment
        except BentoMLException as e:
            error_msg = str(e)
            if "No cloud context default found" in error_msg:
                spinner.log(
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                )
                sys.exit(1)
            if "404 Not Found" in error_msg or "Deployment not found" in error_msg:
                cluster_msg = f" in cluster {cluster}" if cluster else ""
                spinner.log(f"[red]:x: Deployment '{name}' not found{cluster_msg}")
                sys.exit(1)
            spinner.log(f"[red]:x: Error:[/] Failed to get deployment: {error_msg}")
            sys.exit(1)


@inject
def delete_deployment(
    name: str,
    cluster: str | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """Delete a deployment from Dynamo Cloud."""
    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        try:
            spinner.update(f'Deleting deployment "{name}" from Dynamo Cloud...')
            _cloud_client.deployment.delete(name=name, cluster=cluster)
            spinner.log(f':white_check_mark: Successfully deleted deployment "{name}"')
        except BentoMLException as e:
            error_msg = str(e)
            if "No cloud context default found" in error_msg:
                spinner.log(
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                )
                sys.exit(1)
            if "404 Not Found" in error_msg or "Deployment not found" in error_msg:
                cluster_msg = f" in cluster {cluster}" if cluster else ""
                spinner.log(f"[red]:x: Deployment '{name}' not found{cluster_msg}")
                sys.exit(1)
            spinner.log(f"[red]:x: Error:[/] {error_msg}")
            sys.exit(1)


@inject
def list_deployments(
    cluster: str | None = None,
    search: str | None = None,
    dev: bool = False,
    q: str | None = None,
    labels: t.List[dict[str, t.Any]] | None = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """List all deployments from Dynamo Cloud."""
    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        try:
            # Handle label-based filtering
            if labels is not None:
                label_query = " ".join(f"label:{d['key']}={d['value']}" for d in labels)
                if q is not None:
                    q = f"{q} {label_query}"
                else:
                    q = label_query

            spinner.update("Getting deployments from Dynamo Cloud...")
            # Get all deployments in a single call by setting count=1000
            deployments = _cloud_client.deployment.list(
                cluster=cluster, search=search, dev=dev, q=q
            )

            if not deployments:
                spinner.log("No deployments found")
                return

            spinner.log(":white_check_mark: Found deployments:")
            for deployment in deployments:
                spinner.log(f"\n• {deployment.name} (cluster: {deployment.cluster})")
                _display_deployment_info(spinner, deployment)
        except BentoMLException as e:
            if "No cloud context default found" in str(e):
                spinner.log(
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                )
                sys.exit(1)
            spinner.log(f"[red]:x: Error:[/] Failed to list deployments: {str(e)}")
            sys.exit(1)
