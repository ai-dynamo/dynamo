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

import click
import psutil


def create_bentoml_cli() -> click.Command:
    from bentoml._internal.configuration import BENTOML_VERSION
    from bentoml._internal.context import server_context
    from bentoml_cli.bentos import bento_command

    # from bentoml_cli.containerize import containerize_command
    from bentoml_cli.utils import get_entry_points

    # from dynamo.sdk.cli.deploy import deploy_command
    from dynamo.sdk.cli.run import run_command
    from dynamo.sdk.cli.serve import serve_command

    # from dynamo.sdk.cli.server import cloud_command
    from dynamo.sdk.cli.start import start_command
    from dynamo.sdk.cli.utils import DynamoCommandGroup

    server_context.service_type = "cli"

    CONTEXT_SETTINGS = {"help_option_names": ("-h", "--help")}

    @click.group(cls=DynamoCommandGroup, context_settings=CONTEXT_SETTINGS)
    @click.version_option(BENTOML_VERSION, "-v", "--version")
    def bentoml_cli():  # TODO: to be renamed to something....
        """
        The Dynamo CLI is a CLI for serving, containerizing, and deploying Dynamo applications.
        It takes inspiration from and leverages core pieces of the BentoML deployment stack.

        At a high level, you use `serve` to run a set of dynamo services locally,
        `build` and `containerize` to package them  up for deployment, and then `server`
        and `deploy` to deploy them to a K8s cluster running the Dynamo Server
        """

    # Add top-level CLI commands
    # bentoml_cli.add_command(cloud_command)
    bentoml_cli.add_single_command(bento_command, "build")
    bentoml_cli.add_subcommands(start_command)
    bentoml_cli.add_single_command(bento_command, "get")
    bentoml_cli.add_subcommands(serve_command)
    bentoml_cli.add_subcommands(run_command)
    # bentoml_cli.add_command(containerize_command)
    # bentoml_cli.add_command(deploy_command)
    # Load commands from extensions
    for ep in get_entry_points("bentoml.commands"):
        bentoml_cli.add_command(ep.load())

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return bentoml_cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()
