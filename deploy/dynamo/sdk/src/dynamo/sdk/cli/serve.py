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

import collections
import json
import logging
import os
import sys
import typing as t
from typing import Optional

import click
import rich
import yaml

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")  # type: ignore
    F = t.Callable[P, t.Any]  # type: ignore

logger = logging.getLogger(__name__)


def _parse_service_arg(arg_name: str, arg_value: str) -> tuple[str, str, t.Any]:
    """Parse a single CLI argument into service name, key, and value."""

    parts = arg_name.split(".")
    service = parts[0]
    nested_keys = parts[1:]

    # Special case: if this is a ServiceArgs.envs.* path, keep value as string
    if (
        len(nested_keys) >= 2
        and nested_keys[0] == "ServiceArgs"
        and nested_keys[1] == "envs"
    ):
        value: t.Union[str, int, float, bool, dict, list] = arg_value
    else:
        # Parse value based on type for non-env vars
        try:
            value = json.loads(arg_value)
        except json.JSONDecodeError:
            if arg_value.isdigit():
                value = int(arg_value)
            elif arg_value.replace(".", "", 1).isdigit() and arg_value.count(".") <= 1:
                value = float(arg_value)
            elif arg_value.lower() in ("true", "false"):
                value = arg_value.lower() == "true"
            else:
                value = arg_value

    # Build nested dict structure
    result = value
    for key in reversed(nested_keys[1:]):
        result = {key: result}

    return service, nested_keys[0], result


def _parse_service_args(args: list[str]) -> t.Dict[str, t.Any]:
    service_configs: t.DefaultDict[str, t.Dict[str, t.Any]] = collections.defaultdict(
        dict
    )

    def deep_update(d: dict, key: str, value: t.Any):
        """
        Recursively updates nested dictionaries. We use this to process arguments like

        ---Worker.ServiceArgs.env.CUDA_VISIBLE_DEVICES="0,1"

        The _parse_service_arg function will parse this into:
        service = "Worker"
        nested_keys = ["ServiceArgs", "envs", "CUDA_VISIBLE_DEVICES"]

        And returns returns: ("VllmWorker", "ServiceArgs", {"envs": {"CUDA_VISIBLE_DEVICES": "0,1"}})

        We then use deep_update to update the service_configs dictionary with this nested value.
        """
        if isinstance(value, dict) and key in d and isinstance(d[key], dict):
            for k, v in value.items():
                deep_update(d[key], k, v)
        else:
            d[key] = value

    index = 0
    while index < len(args):
        next_arg = args[index]

        if not (next_arg.startswith("--") or "." not in next_arg):
            continue
        try:
            if "=" in next_arg:
                arg_name, arg_value = next_arg.split("=", 1)
                index += 1
            elif args[index + 1] == "=":
                arg_name = next_arg
                arg_value = args[index + 2]
                index += 3
            else:
                arg_name = next_arg
                arg_value = args[index + 1]
                index += 2
            if arg_value.startswith("-"):
                raise ValueError("Service arg value can not start with -")
            arg_name = arg_name[2:]
            service, key, value = _parse_service_arg(arg_name, arg_value)
            deep_update(service_configs[service], key, value)
        except Exception:
            raise ValueError(f"Error parsing service arg: {args[index]}")

    return service_configs


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
        **attrs: t.Any,
    ) -> None:
        """Locally run connected Dynamo services. You can pass service-specific configuration options using --ServiceName.param=value format."""
        # WARNING: internal
        from bentoml._internal.service.loader import load

        from dynamo.sdk.lib.service import LinkedServices

        service_configs: dict[str, dict[str, t.Any]] = {}

        # Load file if provided
        if file:
            with open(file) as f:
                yaml_configs = yaml.safe_load(f)
                # Initialize service_configs as empty dict if it's None
                # Convert nested YAML structure to flat dict with dot notation
                for service, configs in yaml_configs.items():
                    if service not in service_configs:
                        service_configs[service] = {}
                    for key, value in configs.items():
                        service_configs[service][key] = value

        # Process service-specific options
        cmdline_overrides: t.Dict[str, t.Any] = _parse_service_args(ctx.args)
        for service, configs in cmdline_overrides.items():
            if service not in service_configs:
                service_configs[service] = {}
            for key, value in configs.items():
                service_configs[service][key] = value

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
        )

    return cli


serve_command = build_serve_command()
