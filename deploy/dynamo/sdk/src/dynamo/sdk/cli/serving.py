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

# WARNING: internal
from _bentoml_sdk import Service

# WARNING: internal
from bentoml._internal.container import BentoMLContainer

# WARNING: internal
from bentoml._internal.utils.circus import Server
from bentoml.exceptions import BentoMLConfigException
from circus.sockets import CircusSocket
from circus.watcher import Watcher
from simple_di import Provide, inject

from .allocator import ResourceAllocator
from .utils import path_to_uri, reserve_free_port


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
logger = logging.getLogger(__name__)

if POSIX and not IS_WSL:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
    ) -> tuple[str, CircusSocket]:
        from circus.sockets import CircusSocket

        socket_path = os.path.join(uds_path, f"{id(service)}.sock")
        assert len(socket_path) < MAX_AF_UNIX_PATH_LENGTH
        return path_to_uri(socket_path), CircusSocket(
            name=service.name, path=socket_path
        )

elif WINDOWS or IS_WSL:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
    ) -> tuple[str, CircusSocket]:
        from circus.sockets import CircusSocket

        runner_port = port_stack.enter_context(reserve_free_port())
        runner_host = "127.0.0.1"

        return f"tcp://{runner_host}:{runner_port}", CircusSocket(
            name=service.name,
            host=runner_host,
            port=runner_port,
        )

else:

    def _get_server_socket(
        service: ServiceProtocol,
        uds_path: str,
        port_stack: contextlib.ExitStack,
    ) -> tuple[str, CircusSocket]:
        from bentoml.exceptions import BentoMLException

        raise BentoMLException("Unsupported platform")


# WARNING: internal
_SERVICE_WORKER_SCRIPT = "_bentoml_impl.worker.service"


def create_dependency_watcher(
    bento_identifier: str,
    svc: ServiceProtocol,
    uds_path: str,
    port_stack: contextlib.ExitStack,
    scheduler: ResourceAllocator,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Watcher, CircusSocket, str]:
    from bentoml.serving import create_watcher

    num_workers, resource_envs = scheduler.get_resource_envs(svc)
    uri, socket = _get_server_socket(svc, uds_path, port_stack)
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

    if resource_envs:
        args.extend(["--worker-env", json.dumps(resource_envs)])

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
    scheduler: ResourceAllocator,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Watcher, CircusSocket, str]:
    """Create a watcher for a Dynamo service in the dependency graph"""
    from bentoml.serving import create_watcher

    # Get socket for this service
    uri, socket = _get_server_socket(svc, uds_path, port_stack)

    # Get worker configuration
    num_workers, resource_envs = scheduler.get_resource_envs(svc)

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

    if resource_envs:
        args.extend(["--worker-env", json.dumps(resource_envs)])

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
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
) -> Server:
    # WARNING: internal
    from _bentoml_impl.loader import load

    # WARNING: internal
    from bentoml._internal.utils.circus import create_standalone_arbiter
    from bentoml.serving import create_watcher
    from circus.sockets import CircusSocket

    from .allocator import ResourceAllocator

    bento_id: str = ""
    env: dict[str, Any] = {}
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
        bento_id = svc.import_string
        assert (
            working_dir is None
        ), "working_dir should not be set when passing a service in process"
        # use cwd
        bento_path = pathlib.Path(".")
    else:
        svc = load(bento_identifier, working_dir)
        bento_id = str(bento_identifier)
        bento_path = pathlib.Path(working_dir or ".")

    watchers: list[Watcher] = []
    sockets: list[CircusSocket] = []
    allocator = ResourceAllocator()
    if dependency_map is None:
        dependency_map = {}

    # TODO: Only for testing, this will prevent any other dep services from getting started, relying entirely on configured deps in the runner-map
    standalone = False
    if service_name:
        logger.info("Running in standalone mode")
        logger.info(f"service_name: {service_name}")
        standalone = True

    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    num_workers, resource_envs = allocator.get_resource_envs(svc)
    server_on_deployment(svc)
    uds_path = tempfile.mkdtemp(prefix="bentoml-uds-")
    try:
        if not service_name and not standalone:
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
                )
            )

        server_args = [
            "-m",
            _SERVICE_WORKER_SCRIPT,
            bento_identifier,
            "--fd",
            f"$(circus.sockets.{API_SERVER_NAME})",
            "--service-name",
            svc.name,
            "--worker-id",
            "$(CIRCUS.WID)",
        ]
        if resource_envs:
            server_args.extend(["--worker-env", json.dumps(resource_envs)])

        scheme = "http"

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
            # resource_envs is the resource allocation (ie CUDA_VISIBLE_DEVICES) for each worker created by the allocator
            # these resource_envs are passed to each individual worker's environment which is set in serve_dynamo
            if resource_envs:
                args.extend(["--worker-env", json.dumps(resource_envs)])
            # env is the base bentoml environment variables. We make a copy and update it to add any service configurations and additional env vars
            worker_env = env.copy() if env else {}

            # Pass through the main service config
            if "DYNAMO_SERVICE_CONFIG" in os.environ:
                worker_env["DYNAMO_SERVICE_CONFIG"] = os.environ[
                    "DYNAMO_SERVICE_CONFIG"
                ]

            # Get service-specific environment variables from DYNAMO_SERVICE_ENVS
            if "DYNAMO_SERVICE_ENVS" in os.environ:
                try:
                    service_envs = json.loads(os.environ["DYNAMO_SERVICE_ENVS"])
                    if svc.name in service_envs:
                        service_args = service_envs[svc.name].get("ServiceArgs", {})
                        if "envs" in service_args:
                            worker_env.update(service_args["envs"])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse DYNAMO_SERVICE_ENVS: {e}")

            watcher = create_watcher(
                name=f"dynamo_service_{svc.name}",
                args=args,
                numprocesses=num_workers,
                working_dir=str(bento_path.absolute()),
                env=worker_env,  # Dependency map will be injected by serve_http
            )
            watchers.append(watcher)
            logger.info(f"dynamo_service_{svc.name} entrypoint created")
        else:
            # Create regular BentoML service watcher
            watchers.append(
                create_watcher(
                    name="service",
                    args=server_args,
                    working_dir=str(bento_path.absolute()),
                    numprocesses=num_workers,
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
        }

        arbiter = create_standalone_arbiter(**arbiter_kwargs)
        arbiter.exit_stack.callback(shutil.rmtree, uds_path, ignore_errors=True)
        arbiter.start(
            cb=lambda _: logger.info(  # type: ignore
                (
                    "Starting Dynamo Service %s (Press CTRL+C to quit)"
                    if (
                        hasattr(svc, "is_dynamo_component")
                        and svc.is_dynamo_component()
                    )
                    else "Starting %s (Press CTRL+C to quit)"
                ),
                *(
                    (svc.name,)
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
