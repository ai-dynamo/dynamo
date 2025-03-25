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

import collections
import json
import logging
import os
import sys
import typing as t

import click
import rich
import yaml

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")  # type: ignore
    F = t.Callable[P, t.Any]  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"


def deprecated_option(*param_decls: str, **attrs: t.Any):
    """Marks a given options as deprecated, and omit a warning when it's used"""
    deprecated = attrs.pop("deprecated", True)
    new_behaviour = attrs.pop("current_behaviour", None)
    assert new_behaviour is not None, "current_behaviour is required"

    def show_deprecated_callback(
        ctx: click.Context, param: click.Parameter, value: t.Any
    ):
        if value is not param.default and deprecated:
            name = "'--%(name)s'" if attrs.get("is_flag", False) else "'%(name)s'"
            DEPRECATION_WARNING = f"[yellow]DeprecationWarning: The parameter {name} is deprecated and will be removed in the future. (Current behaviour: %(new_behaviour)s)[/]"
            rich.print(
                DEPRECATION_WARNING
                % {"name": param.name, "new_behaviour": new_behaviour},
                file=sys.stderr,
            )

    def decorator(f: F[t.Any]) -> t.Callable[[F[t.Any]], click.Command]:  # type: ignore
        msg = attrs.pop("help", "")
        msg += " (Deprecated)" if msg else "(Deprecated)"
        attrs.setdefault("help", msg)
        attrs.setdefault("callback", show_deprecated_callback)
        return click.option(*param_decls, **attrs)(f)

    return decorator


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
    from bentoml._internal.log import configure_server_logging
    from bentoml_cli.env_manager import env_manager
    from bentoml_cli.utils import AliasCommand

    @click.group(name="serve")
    def cli():
        pass

    @cli.command(
        context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,
        ),
        aliases=["serve-http"],
        cls=AliasCommand,
    )
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "-f",
        "--file",
        type=click.Path(exists=True),
        help="Path to YAML config file for service configuration",
    )
    @click.option(
        "--development",
        type=click.BOOL,
        help="Run the BentoServer in development mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @deprecated_option(
        "--production",
        type=click.BOOL,
        help="Run BentoServer in production mode",
        current_behaviour="This is enabled by default. To run in development mode, use '--development'.",
        is_flag=True,
        default=True,
        show_default=False,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_envvar=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        help="The host to bind for the REST api server",
        envvar="BENTOML_HOST",
        show_envvar=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
        show_envvar=True,
        hidden=True,
    )
    @click.option(
        "--timeout",
        type=click.INT,
        help="Specify the timeout (seconds) for API server and runners",
        envvar="BENTOML_TIMEOUT",
        hidden=True,
    )
    @click.option(
        "--backlog",
        type=click.INT,
        help="The maximum number of pending connections.",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected",
        default=False,
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        help="SSL certificate file",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        help="SSL key file",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-keyfile-password",
        type=str,
        help="SSL keyfile password",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-version",
        type=int,
        help="SSL version to use (see stdlib 'ssl' module)",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        help="Whether client certificate is required (see stdlib 'ssl' module)",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        help="CA certificates file",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--ssl-ciphers",
        type=str,
        help="Ciphers to use (see stdlib 'ssl' module)",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--timeout-keep-alive",
        type=int,
        help="Close Keep-Alive connections if no new data is received within this timeout.",
        hidden=True,
    )
    @click.option(
        "--timeout-graceful-shutdown",
        type=int,
        default=None,
        help="Maximum number of seconds to wait for graceful shutdown. After this timeout, the server will start terminating requests.",
        show_default=True,
        hidden=True,
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Print the final service configuration and exit without starting the server",
        default=False,
    )
    @click.pass_context
    @env_manager
    def serve(
        ctx: click.Context,
        bento: str,
        dry_run: bool,
        development: bool,
        port: int,
        host: str,
        file: str | None,
        api_workers: int,
        timeout: int | None,
        backlog: int,
        reload: bool,
        working_dir: str | None,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
        timeout_keep_alive: int | None,
        timeout_graceful_shutdown: int | None,
        **attrs: t.Any,
    ) -> None:
        """Locally run connected Dynamo services

        \b
        You can also pass service-specific configuration options using --ServiceName.param=value format.

        \b
        BENTO is the serving target, it can be the import as:
        - the import path of a 'bentoml.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'bentofile.yaml' build file with a 'service' field, which provides the import path of a 'bentoml.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            'bentoml serve fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'bentoml serve fraud_detector:4tht2icroji6zput3suqi5nl2'
            'bentoml serve fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'bentoml serve ./fraud_detector_bento'

        \b
        If '--reload' is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'bentofile.yaml' as well as the '.bentoignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        from bentoml import Service
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
                    for key, value in configs.items():
                        if service not in service_configs:
                            service_configs[service] = {}
                        service_configs[service][key] = value

        # Process service-specific options
        cmdline_overrides: t.Dict[str, t.Any] = _parse_service_args(ctx.args)
        for service, configs in cmdline_overrides.items():
            for key, value in configs.items():
                if service not in service_configs:
                    service_configs[service] = {}
                service_configs[service][key] = value

        if dry_run:
            rich.print("[bold]Service Configuration:[/bold]")
            rich.print(json.dumps(service_configs, indent=2))
            rich.print("\n[bold]Environment Variable that would be set:[/bold]")
            rich.print(f"DYNAMO_SERVICE_CONFIG={json.dumps(service_configs)}")
            sys.exit(0)

        # Set environment variable with service configuration
        if service_configs:
            os.environ["DYNAMO_SERVICE_CONFIG"] = json.dumps(service_configs)

        configure_server_logging()
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)
        svc = load(bento_identifier=bento, working_dir=working_dir)

        LinkedServices.remove_unused_edges()
        if isinstance(svc, Service):
            # bentoml<1.2
            from bentoml.serving import serve_http_production

            if development:
                serve_http_production(
                    bento,
                    working_dir=working_dir,
                    port=port,
                    host=DEFAULT_DEV_SERVER_HOST if not host else host,
                    backlog=backlog,
                    api_workers=1,
                    timeout=timeout,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                    reload=reload,
                    development_mode=True,
                    timeout_keep_alive=timeout_keep_alive,
                    timeout_graceful_shutdown=timeout_graceful_shutdown,
                )
            else:
                serve_http_production(
                    bento,
                    working_dir=working_dir,
                    port=port,
                    host=host,
                    api_workers=api_workers,
                    timeout=timeout,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                    reload=reload,
                    development_mode=False,
                    timeout_keep_alive=timeout_keep_alive,
                    timeout_graceful_shutdown=timeout_graceful_shutdown,
                )
        else:
            # bentoml>=1.2
            # from _bentoml_impl.server import serve_http
            from dynamo.sdk.cli.serving import serve_http  # type: ignore

            svc.inject_config()
            serve_http(
                bento,
                working_dir=working_dir,
                host=host,
                port=port,
                backlog=backlog,
                timeout=timeout,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
                development_mode=development,
                reload=reload,
                timeout_keep_alive=timeout_keep_alive,
                timeout_graceful_shutdown=timeout_graceful_shutdown,
            )

    return cli


serve_command = build_serve_command()
