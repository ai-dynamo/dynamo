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
import os
import typing as t
from http import HTTPStatus

import click
import yaml
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import Deployment, DeploymentConfigParameters
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import add_experimental_docstring
from bentoml.exceptions import BentoMLException
from bentoml_cli.utils import BentoMLCommandGroup
from rich.console import Console
from simple_di import Provide, inject

from dynamo.sdk.lib.logging import configure_server_logging
configure_server_logging()

from .utils import resolve_service_config

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud API token is required for authorization. Run `bentoml cloud login` command to login"
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
    @click.group(name="deployment", cls=BentoMLCommandGroup)
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
    print(f"service_configs: {service_configs}")
    env_dicts = []
    if service_configs:
        config_json = json.dumps(service_configs)
        logger.info(f"Deployment service configuration: {config_json}")
        # TODO: The logger line above this is not working so print is current WAR
        print(f"Deployment service configuration: {config_json}")
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
        raise_deployment_config_error(e, "create")

    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        spinner.update("Creating deployment on BentoCloud")
        deployment = _cloud_client.deployment.create(
            deployment_config_params=config_params
        )
        spinner.log(
            f':white_check_mark: Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
        )
        spinner.log(f":laptop_computer: View Dashboard: {deployment.admin_console}")
        if wait:
            spinner.update(
                "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]",
            )
            retcode = deployment.wait_until_ready(timeout=timeout, spinner=spinner)
            if retcode != 0:
                raise SystemExit(retcode)
        return deployment
