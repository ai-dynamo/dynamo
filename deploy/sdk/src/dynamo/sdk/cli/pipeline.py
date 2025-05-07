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
import subprocess
import typing as t

import attr
import typer
import yaml
from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILES
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils.args import set_arguments
from bentoml._internal.utils.filesystem import resolve_user_filepath
from bentoml.exceptions import InvalidArgument
from rich.console import Console
from rich.syntax import Syntax
from simple_di import Provide, inject

from dynamo.sdk.cli.bento_util import Bento

if t.TYPE_CHECKING:
    from bentoml._internal.bento import BentoStore
    from bentoml._internal.container import DefaultBuilder

logger = logging.getLogger(__name__)
console = Console()

DYNAMO_FIGLET = """
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗ ██████╗
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██╔═══██╗
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║   ██║
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║   ██║
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║╚██████╔╝
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝
"""


@inject
def build_bentofile(
    bentofile: str | None = None,
    *,
    service: str | None = None,
    name: str | None = None,
    version: str | None = None,
    labels: dict[str, str] | None = None,
    build_ctx: str | None = None,
    platform: str | None = None,
    bare: bool = False,
    reload: bool = False,
    args: dict[str, t.Any] | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Build a Dynamo pipeline based on options specified in a bentofile.yaml file.
    """
    if args is not None:
        set_arguments(**args)
    if bentofile:
        try:
            bentofile = resolve_user_filepath(bentofile, None)
        except FileNotFoundError:
            raise InvalidArgument(f'bentofile "{bentofile}" not found')
        else:
            build_config = BentoBuildConfig.from_file(bentofile)
    else:
        for filename in DEFAULT_BENTO_BUILD_FILES:
            try:
                bentofile = resolve_user_filepath(filename, build_ctx)
            except FileNotFoundError:
                pass
            else:
                build_config = BentoBuildConfig.from_file(bentofile)
                break
        else:
            build_config = BentoBuildConfig(service=service or "")

    new_attrs: dict[str, t.Any] = {}
    if name is not None:
        new_attrs["name"] = name
    if labels:
        # Ensure both dictionaries are of type dict[str, str]
        existing_labels: dict[str, str] = build_config.labels or {}
        new_attrs["labels"] = {**existing_labels, **labels}

    if new_attrs:
        build_config = attr.evolve(build_config, **new_attrs)

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
        platform=platform,
        bare=bare,
        reload=reload,
    )
    if not bare:
        return bento.save(_bento_store)
    return bento


def get(
    pipeline_tag: str = typer.Argument(
        ..., help="The tag of the Dynamo pipeline to display"
    ),
    output: str = typer.Option(
        "yaml",
        "--output",
        "-o",
        help="Output format (json, yaml, or path)",
        show_default=True,
    ),
) -> None:
    """Display Dynamo pipeline details.

    Prints information about a Dynamo pipeline by its tag.
    """

    # Validate output format
    valid_outputs = ["json", "yaml", "path"]
    if output not in valid_outputs:
        console.print(f"[red]Error: Output format must be one of {valid_outputs}[/red]")
        raise typer.Exit(code=1)

    bento_store = BentoMLContainer.bento_store.get()
    bento = bento_store.get(pipeline_tag)

    if output == "path":
        console.print(bento.path)
    elif output == "json":
        info = json.dumps(bento.info.to_dict(), indent=2, default=str)
        console.print_json(info)
    else:
        info = yaml.dump(bento.info.to_dict(), indent=2, sort_keys=False)
        console.print(Syntax(info, "yaml", background_color="default"))


def build(
    dynamo_pipeline: str = typer.Argument(
        ..., help="Path to the Dynamo pipeline to build"
    ),
    output: str = typer.Option(
        "default",
        "--output",
        "-o",
        help="Output log format. Use 'tag' to display only pipeline tag.",
        show_default=True,
    ),
    containerize: bool = typer.Option(
        False,
        "--containerize",
        help="Containerize the Dynamo pipeline after building. Shortcut for 'dynamo build && dynamo containerize'.",
    ),
    platform: str = typer.Option(None, "--platform", help="Platform to build for"),
) -> None:
    """Build a new Dynamo pipeline from the specified path.

    Creates a packaged Dynamo pipeline ready for deployment. Optionally builds a docker container.
    """
    from bentoml._internal.configuration import get_quiet_mode, set_quiet_mode
    from bentoml._internal.log import configure_logging

    # Validate output format
    valid_outputs = ["tag", "default"]
    if output not in valid_outputs:
        console.print(f"[red]Error: Output format must be one of {valid_outputs}[/red]")
        raise typer.Exit(code=1)

    if output == "tag":
        set_quiet_mode()
        configure_logging()

    service: str | None = None
    build_ctx = "."
    if ":" in dynamo_pipeline:
        service = dynamo_pipeline
    else:
        build_ctx = dynamo_pipeline

    bento = build_bentofile(
        service=service,
        build_ctx=build_ctx,
        platform=platform,
    )

    containerize_cmd = f"dynamo containerize {bento.tag}"

    if output == "tag":
        console.print(f"__tag__:{bento.tag}")
    else:
        if not get_quiet_mode():
            console.print(DYNAMO_FIGLET)
            console.print(f"[green]Successfully built {bento.tag}.")
            next_steps = []
            if not containerize:
                next_steps.append(
                    "\n\n* Containerize your Dynamo pipeline with `dynamo containerize`:\n"
                    f"    $ {containerize_cmd} [or dynamo build --containerize]"
                )

            if next_steps:
                console.print(f"\n[blue]Next steps: {''.join(next_steps)}[/]")

    if containerize:
        backend: DefaultBuilder = t.cast(
            "DefaultBuilder", os.getenv("BENTOML_CONTAINERIZE_BACKEND", "docker")
        )
        try:
            import bentoml

            bentoml.container.health(backend)
        except subprocess.CalledProcessError:
            from bentoml.exceptions import BentoMLException

            raise BentoMLException(f"Backend {backend} is not healthy")
        bentoml.container.build(bento.tag, backend=backend)
