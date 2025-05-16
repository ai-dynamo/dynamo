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
from typing import Any, Dict, List, Optional, TextIO

import typer
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import CloudClientConfig, CloudClientContext
from bentoml._internal.cloud.deployment import Deployment, DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException, CLIException, CloudRESTApiClientError
from rich.console import Console
from simple_di import Provide, inject

from dynamo.runtime.logging import configure_dynamo_logging

from .utils import resolve_service_config

# Configure logging to suppress INFO HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTP client library logs
logging.getLogger("httpcore").setLevel(logging.WARNING)  # HTTP core library logs
configure_dynamo_logging()

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Deploy Dynamo applications to Dynamo Cloud Kubernetes Platform",
    add_completion=True,
    no_args_is_help=True,
)

console = Console(highlight=False)

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* Dynamo Cloud API token is required for authorization. Please provide a valid endpoint with --endpoint option."
        ) from None
    raise BentoMLException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


def _get_urls(deployment: Deployment) -> List[str]:
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


def _build_env_dicts(
    config_file: Optional[TextIO] = None,
    args: Optional[list[str]] = None,
    envs: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build a list of environment variable dicts from config file, args, and env strings.

    Args:
        config_file: Optional configuration file
        args: Optional list of extra arguments
        envs: Optional list of environment variable strings (KEY=VALUE)

    Returns:
        List of dicts suitable for use as envs
    """
    service_configs = resolve_service_config(config_file=config_file, args=args)
    env_dicts = []
    if service_configs:
        config_json = json.dumps(service_configs)
        logger.info(f"Deployment service configuration: {config_json}")
        env_dicts.append({"name": "DYN_DEPLOYMENT_CONFIG", "value": config_json})
    if envs:
        for env in envs:
            if "=" not in env:
                raise CLIException(f"Invalid env format: {env}. Use KEY=VALUE.")
            key, value = env.split("=", 1)
            env_dicts.append({"name": key, "value": value})
    return env_dicts


@inject
def create_deployment(
    pipeline: Optional[str] = None,
    name: Optional[str] = None,
    config_file: Optional[TextIO] = None,
    wait: bool = True,
    timeout: int = 3600,
    dev: bool = False,
    args: Optional[List[str]] = None,
    envs: Optional[List[str]] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    # Build env_dicts from config_file, args, and envs
    env_dicts = _build_env_dicts(config_file=config_file, args=args, envs=envs)

    config_params = DeploymentConfigParameters(
        name=name,
        bento=pipeline,
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


@inject
def update_deployment(
    name: str,
    config_file: Optional[TextIO] = None,
    args: Optional[List[str]] = None,
    envs: Optional[List[str]] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    """Update an existing deployment on Dynamo Cloud.

    Args:
        name: The name of the deployment to update
        config_file: Optional configuration file for the update
        args: Optional extra arguments for config
        envs: Optional list of environment variables (KEY=VALUE)

    Returns:
        Deployment: The updated deployment object
    """
    # Build env_dicts from config_file, args, and envs
    env_dicts = _build_env_dicts(config_file=config_file, args=args, envs=envs)
    config_params = DeploymentConfigParameters(
        name=name,
        envs=env_dicts,
        cli=True,
    )
    try:
        config_params.verify(create=False)
    except BentoMLException as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    with Spinner(console=console) as spinner:
        try:
            spinner.update(f'Updating deployment "{name}" on Dynamo Cloud...')
            deployment = _cloud_client.deployment.update(
                deployment_config_params=config_params
            )
            spinner.log(
                f':white_check_mark: Updated deployment "{deployment.name}" in cluster "{deployment.cluster}"'
            )
            spinner.log(
                "[yellow]Update submitted. It may take a short time for the new pods to become active. Please wait a bit before accessing the deployment to ensure your changes are live.[/yellow]"
            )
            _display_deployment_info(spinner, deployment)
            return deployment
        except BentoMLException as e:
            spinner.log(f"[red]:x: Error:[/] Failed to update deployment: {str(e)}")
            sys.exit(1)


@inject
def get_deployment(
    name: str,
    cluster: Optional[str] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    """Get deployment details from Dynamo Cloud."""
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
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please provide a valid endpoint with --endpoint option."
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
    cluster: Optional[str] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """Delete a deployment from Dynamo Cloud."""
    with Spinner(console=console) as spinner:
        try:
            spinner.update(f'Deleting deployment "{name}" from Dynamo Cloud...')
            _cloud_client.deployment.delete(name=name, cluster=cluster)
            spinner.log(f':white_check_mark: Successfully deleted deployment "{name}"')
        except BentoMLException as e:
            error_msg = str(e)
            if "No cloud context default found" in error_msg:
                spinner.log(
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please provide a valid endpoint with --endpoint option."
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
    cluster: Optional[str] = None,
    search: Optional[str] = None,
    dev: bool = False,
    q: Optional[str] = None,
    labels: Optional[List[Dict[str, Any]]] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """List all deployments from Dynamo Cloud."""
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
                    "[red]:x: Error:[/] Not logged in to Dynamo Cloud. Please provide a valid endpoint with --endpoint option."
                )
                sys.exit(1)
            spinner.log(f"[red]:x: Error:[/] Failed to list deployments: {str(e)}")
            sys.exit(1)


@app.command()
def create(
    ctx: typer.Context,
    pipeline: Optional[str] = typer.Argument(..., help="Dynamo pipeline to deploy"),
    name: Optional[str] = typer.Option(..., "--name", "-n", help="Deployment name"),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: Optional[List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
) -> None:
    """Create a deployment on Dynamo Cloud.

    Create a deployment using parameters, or using config yaml file.
    """
    login_to_cloud(endpoint)
    create_deployment(
        pipeline=pipeline,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        args=ctx.args if hasattr(ctx, "args") else None,
        envs=envs,
    )


@app.command()
def get(
    name: str = typer.Argument(..., help="Deployment name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Get deployment details from Dynamo Cloud.

    Get deployment details by name.
    """
    login_to_cloud(endpoint)
    get_deployment(name, cluster=cluster)


@app.command("list")
def list_deployments_command(
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
    search: Optional[str] = typer.Option(None, "--search", help="Search query"),
    dev: bool = typer.Option(False, "--dev", help="List development deployments"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Advanced  query string"
    ),
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        "-e",
        help="Dynamo Cloud endpoint",
        envvar="DYNAMO_CLOUD",
    ),
) -> None:
    """List all deployments from Dynamo Cloud.

    List and filter deployments.
    """
    login_to_cloud(endpoint)
    list_deployments(cluster=cluster, search=search, dev=dev, q=query)


@app.command()
def update(
    name: str = typer.Argument(..., help="Deployment name to update"),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    envs: Optional[List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Update an existing deployment on Dynamo Cloud.

    Update a deployment using parameters or a config yaml file.
    """
    login_to_cloud(endpoint)
    update_deployment(
        name=name,
        config_file=config_file,
        envs=envs,
    )


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        "-e",
        help="Dynamo Cloud endpoint",
        envvar="DYNAMO_CLOUD",
    ),
) -> None:
    """Delete a deployment from Dynamo Cloud.

    Delete deployment by name.
    """
    login_to_cloud(endpoint)
    delete_deployment(name, cluster=cluster)


def deploy(
    ctx: typer.Context,
    pipeline: Optional[str] = typer.Argument(..., help="Dynamo pipeline to deploy"),
    name: Optional[str] = typer.Option(..., "--name", "-n", help="Deployment name"),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ...,
        "--endpoint",
        "-e",
        help="Dynamo Cloud endpoint",
        envvar="DYNAMO_CLOUD",
    ),
) -> None:
    """Create a deployment on Dynamo Cloud.

    Create a deployment using parameters, or using config yaml file.
    """
    login_to_cloud(endpoint)
    create_deployment(
        pipeline=pipeline,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        args=ctx.args if hasattr(ctx, "args") else None,
    )


def login_to_cloud(endpoint: str) -> None:
    """Connect to Dynamo Cloud silently using logging for success and console for errors."""
    try:
        logger.info(f"Running against Dynamo Cloud at {endpoint}")
        api_token = ""  # Using empty string for now as it's not used
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
            name=cloud_context if cloud_context is not None else current_context_name,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        ctx.save()
        logger.debug(
            f"Configured Dynamo Cloud credentials (current-context: {ctx.name})"
        )
        logger.debug(f"Logged in as {user.email} at {org.name} organization")
    except CloudRESTApiClientError as e:
        if e.error_code == 401:
            console.print(
                f":police_car_light: Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)"
            )
        else:
            console.print(
                f":police_car_light: Error validating token: HTTP {e.error_code}"
            )
        raise BentoMLException(f"Failed to login to Dynamo Cloud: {str(e)}") from e
    except Exception as e:
        console.print(f":police_car_light: Error connecting to Dynamo Cloud: {str(e)}")
        raise BentoMLException(f"Failed to login to Dynamo Cloud: {str(e)}") from e
