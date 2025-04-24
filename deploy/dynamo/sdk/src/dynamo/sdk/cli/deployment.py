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
from bentoml._internal.cloud.deployment import Deployment, DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import add_experimental_docstring
from bentoml.exceptions import BentoMLException
from rich.console import Console
from simple_di import Provide, inject

from dynamo.sdk.lib.logging import configure_server_logging

from .utils import resolve_service_config

configure_server_logging()

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Deploy Dynamo applications to Kubernetes cluster",
    add_completion=True,
    no_args_is_help=True,
)
console = Console(highlight=False)

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* Dynamo Cloud API token is required for authorization. Run `dynamo cloud login` command to login"
        ) from None
    raise BentoMLException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


@inject
def create_deployment(
    bento: Optional[str] = None,
    name: Optional[str] = None,
    config_file: Optional[TextIO] = None,
    wait: bool = True,
    timeout: int = 3600,
    dev: bool = False,
    args: Optional[List[str]] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    # Load config from file and serialize to env
    service_configs = resolve_service_config(config_file=config_file, args=args)
    print(f"service_configs: {service_configs}")
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

    with Spinner(console=console) as spinner:
        spinner.update("Creating deployment on Dynamo Cloud")
        try:
            deployment = _cloud_client.deployment.create(
                deployment_config_params=config_params
            )
            spinner.log(
                f':white_check_mark: Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
            )
            if wait:
                spinner.update(
                    "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]",
                )
                retcode = deployment.wait_until_ready(timeout=timeout, spinner=spinner)
                if retcode != 0:
                    sys.exit(retcode)
            return deployment
        except BentoMLException as e:
            error_msg = str(e)
            if "already exists" in error_msg:
                # Extract deployment name from error message and clean it
                match = re.search(r'"([^"]+?)(?:\\+)?" already exists', error_msg)
                dep_name = match.group(1).rstrip("\\") if match else name
                error_msg = (
                    f'Error: Deployment "{dep_name}" already exists. To create a new deployment:\n'
                    f"1. Use a different name with the --name flag\n"
                    f"2. Or delete the existing deployment with: dynamo deployment delete {dep_name}"
                )
                print(error_msg)
                sys.exit(1)
            print(f"Error: {str(e)}")
            sys.exit(1)


@inject
def get_deployment(
    name: str,
    cluster: Optional[str] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    """Get deployment details from Dynamo Cloud."""
    with Spinner(console=console) as spinner:
        spinner.update(f'Getting deployment "{name}" from Dynamo Cloud')
        try:
            deployment = _cloud_client.deployment.get(name=name, cluster=cluster)
            spinner.log(
                f':white_check_mark: Found deployment "{deployment.name}" in cluster "{deployment.cluster}"'
            )
            return deployment
        except BentoMLException as e:
            if "No cloud context default found" in str(e):
                raise BentoMLException(
                    "Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                ) from None
            raise_deployment_config_error(e, "get")


@inject
def delete_deployment(
    name: str,
    cluster: Optional[str] = None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """Delete a deployment from Dynamo Cloud."""
    with Spinner(console=console) as spinner:
        spinner.update(f'Deleting deployment "{name}" from Dynamo Cloud')
        try:
            _cloud_client.deployment.delete(name=name, cluster=cluster)
            spinner.log(f':white_check_mark: Deleted deployment "{name}"')
        except BentoMLException as e:
            if "No cloud context default found" in str(e):
                raise BentoMLException(
                    "Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                ) from None
            raise_deployment_config_error(e, "delete")


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
        spinner.update("Getting all deployments from Dynamo Cloud")
        try:
            # Handle label-based filtering
            if labels is not None:
                label_query = " ".join(f"label:{d['key']}={d['value']}" for d in labels)
                if q is not None:
                    q = f"{q} {label_query}"
                else:
                    q = label_query

            deployments = _cloud_client.deployment.list(
                cluster=cluster, search=search, dev=dev, q=q
            )

            if not deployments:
                spinner.log("No deployments found")
                return

            spinner.log(":white_check_mark: Found deployments:")
            for deployment in deployments:
                spinner.log(f"  • {deployment.name} (cluster: {deployment.cluster})")
        except BentoMLException as e:
            if "No cloud context default found" in str(e):
                raise BentoMLException(
                    "Not logged in to Dynamo Cloud. Please run 'dynamo cloud login' first."
                ) from None
            raise_deployment_config_error(e, "list")


@app.command()
@add_experimental_docstring
def create(
    bento: Optional[str] = typer.Argument(None, help="Bento to deploy"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    ctx=typer.Context,
) -> None:
    """Create a deployment on Dynamo Cloud.

    Create a deployment using parameters, or using config yaml file.
    """
    create_deployment(
        bento=bento,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        args=ctx.args if hasattr(ctx, "args") else None,
    )


@app.command()
def get(
    name: str = typer.Argument(..., help="Deployment name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
) -> None:
    """Get deployment details from Dynamo Cloud.

    Get deployment details by name.
    """
    get_deployment(name, cluster=cluster)


@app.command()
def list(
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
    search: Optional[str] = typer.Option(None, "--search", help="Search query"),
    dev: bool = typer.Option(False, "--dev", help="List development deployments"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Advanced query string"
    ),
) -> None:
    """List all deployments from Dynamo Cloud.

    List and filter deployments.
    """
    list_deployments(cluster=cluster, search=search, dev=dev, q=query)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    cluster: Optional[str] = typer.Option(None, "--cluster", help="Cluster name"),
) -> None:
    """Delete a deployment from Dynamo Cloud.

    Delete deployment by name.
    """
    delete_deployment(name, cluster=cluster)


@add_experimental_docstring
def deploy(
    bento: Optional[str] = typer.Argument(None, help="Bento to deploy"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    ctx=typer.Context,
) -> None:
    """Create a deployment on Dynamo Cloud.

    Create a deployment using parameters, or using config yaml file.
    """
    create_deployment(
        bento=bento,
        name=name,
        config_file=config_file,
        wait=wait,
        timeout=timeout,
        args=ctx.args if hasattr(ctx, "args") else None,
    )
