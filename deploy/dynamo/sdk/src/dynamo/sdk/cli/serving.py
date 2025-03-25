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

import contextlib
import ipaddress
import json
import logging
import os
import pathlib
import platform
import shutil
import socket
import tempfile
import typing as t
from typing import Any, Dict, Optional, Protocol, TypeVar

from _bentoml_sdk import Service
from bentoml._internal.container import BentoMLContainer
from bentoml._internal.utils.circus import Server
from bentoml.exceptions import BentoMLConfigException
from circus.sockets import CircusSocket
from circus.watcher import Watcher
from simple_di import Provide, inject

from .allocator import ResourceAllocator


# Define a Protocol for services to ensure type safety
class ServiceProtocol(Protocol):
    name: str
    inner: Any
    models: list[Any]
    bento: Any

    def is_dynamo_component(self) -> bool:
        ...


# Use Protocol as the base for type alias
AnyService = TypeVar("AnyService", bound=ServiceProtocol)

POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
IS_WSL = "microsoft-standard" in platform.release()
API_SERVER_NAME = "_bento_api_server"

MAX_AF_UNIX_PATH_LENGTH = 103
logger = logging.getLogger("bentoml.serve")

if POSIX and not IS_WSL:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from bentoml._internal.utils.uri import path_to_uri
        from circus.sockets import CircusSocket

        socket_path = os.path.join(uds_path, f"{id(service)}.sock")
        assert len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
        return path_to_uri(socket_path), CircusSocket(
            name=service.name, path=socket_path, backlog=backlog
        )

elif WINDOWS or IS_WSL:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from bentoml._internal.utils import reserve_free_port
        from circus.sockets import CircusSocket

        runner_port = port_stack.enter_context(reserve_free_port())
        runner_host = "127.0.0.1"

        return f"tcp://{runner_host}:{runner_port}", CircusSocket(
            name=service.name,
            host=runner_host,
            port=runner_port,
            backlog=backlog,
        )

else:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
        backlog: int,
    ) -> tuple[str, CircusSocket]:
        from bentoml.exceptions import BentoMLException

        raise BentoMLException("Unsupported platform")


_SERVICE_WORKER_SCRIPT = "_bentoml_impl.worker.service"


def create_dependency_watcher(
    bento_identifier: str,
    svc: ServiceProtocol,
    uds_path: str,
    port_stack: contextlib.ExitStack,
    backlog: int,
    scheduler: ResourceAllocator,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Watcher, CircusSocket, str]:
    from bentoml.serving import create_watcher

    num_workers, worker_envs = scheduler.get_worker_env(svc)
    uri, socket = _get_server_socket(svc, uds_path, port_stack, backlog)
    args = [
        "-m",
        _SERVICE_WORKER_SCRIPT,
        bento_identifier,
        "--service-name",
        svc.name,
        "--fd",
        f"$(circus.sockets.{svc.name})",
        "--worker-id",
        "$(CIRCUS.WID)",
    ]

    if worker_envs:
        args.extend(["--worker-env", json.dumps(worker_envs)])

    watcher = create_watcher(
        name=f"service_{svc.name}",
        args=args,
        numprocesses=num_workers,
        working_dir=working_dir,
        env=env,
    )
    return watcher, socket, uri


def create_dynamo_watcher(
    bento_identifier: str,
    svc: ServiceProtocol,
    uds_path: str,
    port_stack: contextlib.ExitStack,
    backlog: int,
    scheduler: ResourceAllocator,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Watcher, CircusSocket, str]:
    """Create a watcher for a Dynamo service in the dependency graph"""
    from bentoml.serving import create_watcher

    # Get socket for this service
    uri, socket = _get_server_socket(svc, uds_path, port_stack, backlog)

    # Get worker configuration
    num_workers, worker_envs = scheduler.get_worker_env(svc)

    # Create Dynamo-specific worker args
    args = [
        "-m",
        "dynamo.sdk.cli.serve_dynamo",  # Use our Dynamo worker module
        bento_identifier,
        "--service-name",
        svc.name,
        "--worker-id",
        "$(CIRCUS.WID)",
    ]

    if worker_envs:
        args.extend(["--worker-env", json.dumps(worker_envs)])

    # Update env to include ServiceConfig and service-specific environment variables
    worker_env = env.copy() if env else {}

    # Pass through the main service config
    if "DYNAMO_SERVICE_CONFIG" in os.environ:
        worker_env["DYNAMO_SERVICE_CONFIG"] = os.environ["DYNAMO_SERVICE_CONFIG"]

    # Get service-specific environment variables from DYNAMO_SERVICE_ENVS
    if "DYNAMO_SERVICE_ENVS" in os.environ:
        try:
            service_envs = json.loads(os.environ["DYNAMO_SERVICE_ENVS"])
            if svc.name in service_envs:
                service_args = service_envs[svc.name].get("ServiceArgs", {})
                if "envs" in service_args:
                    worker_env.update(service_args["envs"])
                    logger.info(
                        f"Added service-specific environment variables for {svc.name}"
                    )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse DYNAMO_SERVICE_ENVS: {e}")

    # Create the watcher with updated environment
    watcher = create_watcher(
        name=f"dynamo_service_{svc.name}",
        args=args,
        numprocesses=num_workers,
        working_dir=working_dir,
        env=worker_env,
    )

    return watcher, socket, uri


@inject
def server_on_deployment(
    svc: ServiceProtocol, result_file: str = Provide[BentoMLContainer.result_store_file]
) -> None:
    # Resolve models before server starts.
    if hasattr(svc, "bento") and (bento := getattr(svc, "bento")):
        for model in bento.info.all_models:
            model.to_model().resolve()
    elif hasattr(svc, "models"):
        for model in svc.models:
            model.resolve()

    if hasattr(svc, "inner"):
        inner = svc.inner
        for name in dir(inner):
            member = getattr(inner, name)
            if callable(member) and getattr(
                member, "__bentoml_deployment_hook__", False
            ):
                member()

    if os.path.exists(result_file):
        os.remove(result_file)


@inject(squeeze_none=True)
def serve_http(
    bento_identifier: str | AnyService,
    working_dir: str | None = None,
    host: str = Provide[BentoMLContainer.http.host],
    port: int = Provide[BentoMLContainer.http.port],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    timeout: int | None = None,
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    development_mode: bool = False,
    reload: bool = False,
    timeout_keep_alive: int | None = None,
    timeout_graceful_shutdown: int | None = None,
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
    threaded: bool = False,
) -> Server:
    from _bentoml_impl.loader import import_service, normalize_identifier
    from bentoml._internal.log import SERVER_LOGGING_CONFIG
    from bentoml._internal.utils import reserve_free_port
    from bentoml._internal.utils.analytics.usage_stats import track_serve
    from bentoml._internal.utils.circus import create_standalone_arbiter
    from bentoml.serving import (
        construct_ssl_args,
        construct_timeouts_args,
        create_watcher,
        ensure_prometheus_dir,
        make_reload_plugin,
    )
    from circus.sockets import CircusSocket

    from .allocator import ResourceAllocator

    bento_id: str = ""
    env = {"PROMETHEUS_MULTIPROC_DIR": ensure_prometheus_dir()}
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
        bento_id = svc.import_string
        assert (
            working_dir is None
        ), "working_dir should not be set when passing a service in process"
        # use cwd
        bento_path = pathlib.Path(".")
    else:
        bento_id, bento_path = normalize_identifier(bento_identifier, working_dir)

        svc = import_service(bento_id, bento_path)

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    allocator = ResourceAllocator()
    if dependency_map is None:
        dependency_map = {}

    # TODO: Only for testing, this will prevent any other dep services from getting started, relying entirely on configured deps in the runner-map
    standalone = False
    if service_name:
        print("Running in standalone mode")
        print(f"service_name: {service_name}")
        standalone = True

    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    num_workers, worker_envs = allocator.get_worker_env(svc)
    server_on_deployment(svc)
    uds_path = tempfile.mkdtemp(prefix="bentoml-uds-")
    try:
        if not service_name and not development_mode and not standalone:
            with contextlib.ExitStack() as port_stack:
                for name, dep_svc in svc.all_services().items():
                    if name == svc.name:
                        continue
                    if name in dependency_map:
                        continue

                    # Check if this is a Dynamo service
                    if (
                        hasattr(dep_svc, "is_dynamo_component")
                        and dep_svc.is_dynamo_component()
                    ):
                        new_watcher, new_socket, uri = create_dynamo_watcher(
                            bento_id,
                            dep_svc,
                            uds_path,
                            port_stack,
                            backlog,
                            allocator,
                            str(bento_path.absolute()),
                            env=env,
                        )
                    else:
                        # Regular BentoML service
                        new_watcher, new_socket, uri = create_dependency_watcher(
                            bento_id,
                            dep_svc,
                            uds_path,
                            port_stack,
                            backlog,
                            allocator,
                            str(bento_path.absolute()),
                            env=env,
                        )

                    watchers.append(new_watcher)
                    sockets.append(new_socket)
                    dependency_map[name] = uri
                    server_on_deployment(dep_svc)
                # reserve one more to avoid conflicts
                port_stack.enter_context(reserve_free_port())

        try:
            ipaddr = ipaddress.ip_address(host)
            if ipaddr.version == 4:
                family = socket.AF_INET
            elif ipaddr.version == 6:
                family = socket.AF_INET6
            else:
                raise BentoMLConfigException(
                    f"Unsupported host IP address version: {ipaddr.version}"
                )
        except ValueError as e:
            raise BentoMLConfigException(f"Invalid host IP address: {host}") from e

        if not svc.is_dynamo_component():
            sockets.append(
                CircusSocket(
                    name=API_SERVER_NAME,
                    host=host,
                    port=port,
                    family=family,
                    backlog=backlog,
                )
            )
        if BentoMLContainer.ssl.enabled.get() and not ssl_certfile:
            raise BentoMLConfigException("ssl_certfile is required when ssl is enabled")

        ssl_args = construct_ssl_args(
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_ciphers=ssl_ciphers,
        )
        timeouts_args = construct_timeouts_args(
            timeout_keep_alive=timeout_keep_alive,
            timeout_graceful_shutdown=timeout_graceful_shutdown,
        )
        timeout_args = ["--timeout", str(timeout)] if timeout else []

        server_args = [
            "-m",
            _SERVICE_WORKER_SCRIPT,
            bento_identifier,
            "--fd",
            f"$(circus.sockets.{API_SERVER_NAME})",
            "--service-name",
            svc.name,
            "--backlog",
            str(backlog),
            "--worker-id",
            "$(CIRCUS.WID)",
            *ssl_args,
            *timeouts_args,
            *timeout_args,
        ]
        if worker_envs:
            server_args.extend(["--worker-env", json.dumps(worker_envs)])
        if development_mode:
            server_args.append("--development-mode")

        scheme = "https" if BentoMLContainer.ssl.enabled.get() else "http"

        # Check if this is a Dynamo service
        if hasattr(svc, "is_dynamo_component") and svc.is_dynamo_component():
            # Create Dynamo-specific watcher using existing socket
            args = [
                "-m",
                "dynamo.sdk.cli.serve_dynamo",  # Use our Dynamo worker module
                bento_identifier,
                "--service-name",
                svc.name,
                "--worker-id",
                "$(CIRCUS.WID)",
            ]
            watcher = create_watcher(
                name=f"dynamo_service_{svc.name}",
                args=args,
                numprocesses=num_workers,
                working_dir=str(bento_path.absolute()),
                close_child_stdin=not development_mode,
                env=env,  # Dependency map will be injected by serve_http
            )
            watchers.append(watcher)
            print(f"dynamo_service_{svc.name} entrypoint created")
        else:
            # Create regular BentoML service watcher
            watchers.append(
                create_watcher(
                    name="service",
                    args=server_args,
                    working_dir=str(bento_path.absolute()),
                    numprocesses=num_workers,
                    close_child_stdin=not development_mode,
                    env=env,
                )
            )

        log_host = "localhost" if host in ["0.0.0.0", "::"] else host
        dependency_map[svc.name] = f"{scheme}://{log_host}:{port}"

        # inject runner map now
        inject_env = {"BENTOML_RUNNER_MAP": json.dumps(dependency_map)}

        for watcher in watchers:
            if watcher.env is None:
                watcher.env = inject_env
            else:
                watcher.env.update(inject_env)

        arbiter_kwargs: dict[str, t.Any] = {
            "watchers": watchers,
            "sockets": sockets,
            "threaded": threaded,
        }

        if reload:
            reload_plugin = make_reload_plugin(str(bento_path.absolute()), bentoml_home)
            arbiter_kwargs["plugins"] = [reload_plugin]

        if development_mode:
            arbiter_kwargs["debug"] = True
            arbiter_kwargs["loggerconfig"] = SERVER_LOGGING_CONFIG

        arbiter = create_standalone_arbiter(**arbiter_kwargs)
        arbiter.exit_stack.enter_context(
            track_serve(svc, production=not development_mode)
        )
        arbiter.exit_stack.callback(shutil.rmtree, uds_path, ignore_errors=True)
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                (
                    "Starting Dynamo Service %s (%s/%s) listening on %s://%s:%d (Press CTRL+C to quit)"
                    if (
                        hasattr(svc, "is_dynamo_component")
                        and svc.is_dynamo_component()
                    )
                    else "Starting %s (Press CTRL+C to quit)"
                ),
                *(
                    (svc.name, *svc.dynamo_address(), scheme, log_host, port)
                    if (
                        hasattr(svc, "is_dynamo_component")
                        and svc.is_dynamo_component()
                    )
                    else (bento_identifier,)
                ),
            ),
        )
        return Server(url=f"{scheme}://{log_host}:{port}", arbiter=arbiter)
    except Exception:
        shutil.rmtree(uds_path, ignore_errors=True)
        raise
