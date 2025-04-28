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
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .utils import resolve_service_config

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")  # type: ignore
    F = t.Callable[P, t.Any]  # type: ignore

logger = logging.getLogger(__name__)
console = Console()


def serve(
    ctx: typer.Context,
    dynamo_pipeline: str = typer.Argument(
        ..., help="The path to the Dynamo pipeline to serve"
    ),
    service_name: str = typer.Option(
        "",
        help="Only serve the specified service. Don't serve any dependencies of this service.",
        envvar="DYNAMO_SERVICE_NAME",
    ),
    depends: List[str] = typer.Option(
        [],
        help="List of runner dependencies in name=value format",
        envvar="DYNAMO_SERVE_DEPENDS",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config-file",
        "-f",
        help="Path to YAML config file for service configuration",
        exists=True,
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="The port to listen on for the REST API server",
        envvar="DYNAMO_PORT",
    ),
    host: Optional[str] = typer.Option(
        None,
        help="The host to bind for the REST API server",
        envvar="DYNAMO_HOST",
    ),
    working_dir: Optional[Path] = typer.Option(
        None,
        help="When loading from source code, specify the directory to find the Service instance",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Print the final service configuration and exit without starting the server",
    ),
    enable_local_planner: bool = typer.Option(
        False,
        help="Save a snapshot of your service state to a file that allows planner to edit your deployment configuration",
    ),
):
    """Locally serve a Dynamo pipeline.

    Starts a local server for the specified Dynamo pipeline.
    """

    from dynamo.runtime.logging import configure_dynamo_logging
    from dynamo.sdk.lib.loader import find_and_load_service
    from dynamo.sdk.lib.service import LinkedServices

    # Extract extra arguments not captured by typer
    service_configs = resolve_service_config(config_file, ctx.args)

    # Process depends
    runner_map_dict = {}
    if depends:
        try:
            runner_map_dict = dict([s.split("=", maxsplit=2) for s in depends or []])
        except ValueError:
            console.print(
                "[bold red]Error:[/bold red] Invalid format for --depends option. Use format 'name=value'"
            )
            raise typer.Exit(code=1)

    if dry_run:
        console.print("[bold green]Service Configuration:[/bold green]")
        console.print_json(json.dumps(service_configs))
        console.print(
            "\n[bold green]Environment Variable that would be set:[/bold green]"
        )
        console.print(f"DYNAMO_SERVICE_CONFIG={json.dumps(service_configs)}")
        raise typer.Exit()

    configure_dynamo_logging()

    if service_configs:
        logger.info(f"Running dynamo serve with service configs {service_configs}")
        os.environ["DYNAMO_SERVICE_CONFIG"] = json.dumps(service_configs)

    if working_dir is None:
        if os.path.isdir(os.path.expanduser(dynamo_pipeline)):
            working_dir = Path(os.path.expanduser(dynamo_pipeline))
        else:
            working_dir = Path(".")

    # Convert Path objects to strings where string is required
    working_dir_str = str(working_dir)

    if sys.path[0] != working_dir_str:
        sys.path.insert(0, working_dir_str)

    svc = find_and_load_service(dynamo_pipeline, working_dir=working_dir)
    logger.info(f"Loaded service: {svc.name}")
    logger.info("Dependencies: %s", [dep.on.name for dep in svc.dependencies.values()])
    LinkedServices.remove_unused_edges()

    from dynamo.sdk.cli.serving import serve_dynamo_graph  # type: ignore

    svc.inject_config()

    # Start the service
    console.print(
        Panel.fit(
            f"[bold]Starting Dynamo service:[/bold] [cyan]{dynamo_pipeline}[/cyan]",
            title="[bold green]Dynamo Serve[/bold green]",
            border_style="green",
        )
    )

    serve_dynamo_graph(
        dynamo_pipeline,
        working_dir=working_dir_str,
        # host=host,
        # port=port,
        dependency_map=runner_map_dict,
        service_name=service_name,
        enable_local_planner=enable_local_planner,
    )
