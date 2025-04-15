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
import sys
import typing as t
from typing import Optional

import click
import rich

from .utils import resolve_service_config

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")  # type: ignore
    F = t.Callable[P, t.Any]  # type: ignore

logger = logging.getLogger(__name__)


def build_serve_command() -> click.Group:
    from dynamo.sdk.lib.logging import configure_server_logging

    @click.group(name="serve")
    def cli():
        pass

    @cli.command(
        context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,
        ),
    )
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--service-name",
        type=click.STRING,
        required=False,
        default="",
        envvar="BENTOML_SERVE_SERVICE_NAME",
        help="Only serve the specified service. Don't serve any dependencies of this service.",
    )
    @click.option(
        "--depends",
        type=click.STRING,
        multiple=True,
        envvar="BENTOML_SERVE_DEPENDS",
        help="list of runners map",
    )
    @click.option(
        "-f",
        "--file",
        type=click.Path(exists=True),
        help="Path to YAML config file for service configuration",
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        help="The port to listen on for the REST api server if you are not using a dynamo service",
        envvar="BENTOML_PORT",
        show_envvar=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        help="The host to bind for the REST api server if you are not using a dynamo service",
        envvar="BENTOML_HOST",
        show_envvar=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Print the final service configuration and exit without starting the server",
        default=False,
    )
    @click.option(
        "--enable-planner",
        is_flag=True,
        help="Save a snapshot of your service state to a file that allows planner to edit your deployment configuration",
        default=False,
    )
    @click.pass_context
    def serve(
        ctx: click.Context,
        bento: str,
        service_name: str,
        depends: Optional[list[str]],
        dry_run: bool,
        port: int,
        host: str,
        file: str | None,
        working_dir: str | None,
        enable_planner: bool,
        **attrs: t.Any,
    ) -> None:
        """Locally run connected Dynamo services. You can pass service-specific configuration options using --ServiceName.param=value format."""
        # WARNING: internal
        from bentoml._internal.service.loader import load

        from dynamo.sdk.lib.service import LinkedServices

        # Resolve service configs from yaml file, command line args into a python dict
        service_configs = resolve_service_config(file, ctx.args)

        # Process depends
        if depends:
            runner_map_dict = dict([s.split("=", maxsplit=2) for s in depends or []])
        else:
            runner_map_dict = {}

        if dry_run:
            rich.print("[bold]Service Configuration:[/bold]")
            rich.print(json.dumps(service_configs, indent=2))
            rich.print("\n[bold]Environment Variable that would be set:[/bold]")
            rich.print(f"DYNAMO_SERVICE_CONFIG={json.dumps(service_configs)}")
            sys.exit(0)

        configure_server_logging()
        # Set environment variable with service configuration
        if service_configs:
            logger.info(f"Running dynamo serve with service configs {service_configs}")
            os.environ["DYNAMO_SERVICE_CONFIG"] = json.dumps(service_configs)

        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)
        svc = load(bento_identifier=bento, working_dir=working_dir)

        LinkedServices.remove_unused_edges()

        from dynamo.sdk.cli.serving import serve_http  # type: ignore

        svc.inject_config()
        serve_http(
            bento,
            working_dir=working_dir,
            host=host,
            port=port,
            dependency_map=runner_map_dict,
            service_name=service_name,
            enable_planner=enable_planner,
        )

    return cli


serve_command = build_serve_command()
