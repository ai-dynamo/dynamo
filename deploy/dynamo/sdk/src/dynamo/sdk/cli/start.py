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

import json
import logging
import os
import sys
import typing as t
from typing import Optional
from urllib.parse import urlparse

import click
import rich
import yaml

from dynamo.sdk.cli.serve import _parse_service_args

logger = logging.getLogger(__name__)


def build_start_command() -> click.Group:
    from bentoml._internal.utils import add_experimental_docstring

    @click.group(name="start")
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
        help="specify the runner name to serve",
    )
    @click.option(
        "-f",
        "--file",
        type=click.Path(exists=True),
        help="Path to YAML config file for service configuration",
    )
    @click.option(
        "--depends",
        type=click.STRING,
        multiple=True,
        envvar="BENTOML_SERVE_DEPENDS",
        help="list of runners map",
    )
    @click.option(
        "--runner-map",
        type=click.STRING,
        envvar="BENTOML_SERVE_RUNNER_MAP",
        help="[Deprecated] use --depends instead. "
        "JSON string of runners map. For backword compatibility for yatai < 1.0.0",
    )
    @click.option(
        "--bind",
        type=click.STRING,
        help="[Deprecated] use --host and --port instead."
        "Bind address for the server. For backword compatibility for yatai < 1.0.0",
        required=False,
    )
    @click.option(
        "--port",
        type=click.INT,
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_envvar=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        help="The host to bind for the REST api server [defaults: 127.0.0.1(dev), 0.0.0.0(production)]",
        show_envvar="BENTOML_HOST",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        help="The maximum number of pending connections.",
        show_envvar=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
    )
    @click.option(
        "--timeout",
        type=click.INT,
        help="Specify the timeout (seconds) for API server",
        envvar="BENTOML_TIMEOUT",
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option("--ssl-certfile", type=str, help="SSL certificate file")
    @click.option("--ssl-keyfile", type=str, help="SSL key file")
    @click.option("--ssl-keyfile-password", type=str, help="SSL keyfile password")
    @click.option(
        "--ssl-version", type=int, help="SSL version to use (see stdlib 'ssl' module)"
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        help="Whether client certificate is required (see stdlib 'ssl' module)",
    )
    @click.option("--ssl-ca-certs", type=str, help="CA certificates file")
    @click.option(
        "--ssl-ciphers", type=str, help="Ciphers to use (see stdlib 'ssl' module)"
    )
    @click.option(
        "--timeout-keep-alive",
        type=int,
        help="Close Keep-Alive connections if no new data is received within this timeout.",
    )
    @click.option(
        "--timeout-graceful-shutdown",
        type=int,
        default=None,
        help="Maximum number of seconds to wait for graceful shutdown. After this timeout, the server will start terminating requests.",
    )
    @click.option(
        "--reload",
        is_flag=True,
        help="Reload Service when code changes detected",
        default=False,
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Print the final service configuration and exit without starting the server",
        default=False,
    )
    @click.pass_context
    @add_experimental_docstring
    def start(
        ctx: click.Context,
        bento: str,
        service_name: str,
        dry_run: bool,
        depends: Optional[list[str]],
        runner_map: Optional[str],
        bind: Optional[str],
        port: Optional[int],
        host: Optional[str],
        file: str | None,
        backlog: Optional[int],
        working_dir: Optional[str],
        api_workers: Optional[int],
        timeout: Optional[int],
        ssl_certfile: Optional[str],
        ssl_keyfile: Optional[str],
        ssl_keyfile_password: Optional[str],
        ssl_version: Optional[int],
        ssl_cert_reqs: Optional[int],
        ssl_ca_certs: Optional[str],
        ssl_ciphers: Optional[str],
        timeout_keep_alive: Optional[int],
        timeout_graceful_shutdown: Optional[int],
        reload: bool = False,
    ) -> None:
        """
        Start a single Dynamo service. This will be used inside Yatai.
        """
        from bentoml import Service
        from bentoml._internal.service.loader import load

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

        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)
        if depends:
            runner_map_dict = dict([s.split("=", maxsplit=2) for s in depends or []])
        elif runner_map:
            runner_map_dict = json.loads(runner_map)
        else:
            runner_map_dict = {}

        if bind is not None:
            parsed = urlparse(bind)
            assert parsed.scheme == "tcp"
            host = parsed.hostname or host
            port = parsed.port or port

        svc = load(bento, working_dir=working_dir)
        if isinstance(svc, Service):
            if reload:
                logger.warning("--reload does not work with legacy style services")
            # for <1.2 bentos
            if not service_name or service_name == svc.name:
                from bentoml.start import start_http_server

                for dep in depends or []:
                    rich.print(f"Using remote: {dep}")
                start_http_server(
                    bento,
                    runner_map=runner_map_dict,
                    working_dir=working_dir,
                    port=port,
                    host=host,
                    backlog=backlog,
                    api_workers=api_workers or 1,
                    timeout=timeout,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                    timeout_keep_alive=timeout_keep_alive,
                    timeout_graceful_shutdown=timeout_graceful_shutdown,
                )
            else:
                from bentoml.start import start_runner_server

                if bind is not None:
                    parsed = urlparse(bind)
                    assert parsed.scheme == "tcp"
                    host = parsed.hostname or host
                    port = parsed.port or port

                start_runner_server(
                    bento,
                    runner_name=service_name,
                    working_dir=working_dir,
                    timeout=timeout,
                    port=port,
                    host=host,
                    backlog=backlog,
                )
        else:
            # for >=1.2 bentos
            from dynamo.sdk.cli.serving import serve_http

            print(f"Starting service {service_name}")
            svc.inject_config()
            serve_http(
                bento,
                working_dir=working_dir,
                port=port,
                host=host,
                backlog=backlog,
                timeout=timeout,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
                timeout_keep_alive=timeout_keep_alive,
                timeout_graceful_shutdown=timeout_graceful_shutdown,
                dependency_map=runner_map_dict,
                service_name=service_name,
                reload=reload,
            )

    return cli


start_command = build_start_command()
