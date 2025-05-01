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

import importlib.metadata

import typer
from rich.console import Console

from dynamo.sdk.cli.deployment import app as deployment_app
from dynamo.sdk.cli.deployment import deploy
from dynamo.sdk.cli.env import env
from dynamo.sdk.cli.pipeline import build, get
from dynamo.sdk.cli.run import run
from dynamo.sdk.cli.serve import serve

console = Console()

cli = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    name="dynamo",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def version_callback(value: bool):
    if value:
        version = importlib.metadata.version("ai-dynamo")
        console.print(
            f"[bold green]Dynamo CLI[/bold green] version: [cyan]{version}[/cyan]"
        )
        raise typer.Exit()


@cli.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the application version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    The Dynamo CLI is a CLI for serving, containerizing, and deploying Dynamo applications.
    It takes inspiration from and leverages core pieces of the BentoML deployment stack.

    At a high level, you use `serve` to run a set of dynamo services locally,
    `build` and `containerize` to package them up for deployment, and then `cloud`
    and `deploy` to deploy them to a K8s cluster running the Dynamo Cloud
    """


cli.command()(env)
cli.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(serve)
cli.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)(run)
cli.add_typer(deployment_app, name="deployment")
cli.command()(deploy)
cli.command()(build)
cli.command()(get)

if __name__ == "__main__":
    cli()
