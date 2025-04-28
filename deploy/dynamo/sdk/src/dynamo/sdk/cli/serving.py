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
import json
import logging
import os
import pathlib
import shutil
import tempfile
from typing import Any, Dict, Optional, TypeVar

# TODO: WARNING: internal but only for type checking in the deploy path i believe
from _bentoml_sdk import Service
from circus.sockets import CircusSocket
from circus.watcher import Watcher
from simple_di import inject

from dynamo.sdk.cli.circus import CircusRunner

from .allocator import NVIDIA_GPU, ResourceAllocator
from .circus import _get_server_socket
from .utils import (
    DYN_LOCAL_STATE_DIR,
    ServiceProtocol,
    reserve_free_port,
    save_dynamo_state,
)

# WARNING: internal


# Use Protocol as the base for type alias
AnyService = TypeVar("AnyService", bound=ServiceProtocol)


logger = logging.getLogger(__name__)

_DYNAMO_WORKER_SCRIPT = "dynamo.sdk.cli.serve_dynamo"


def _get_dynamo_worker_script(bento_identifier: str, svc_name: str) -> list[str]:
    args = [
        "-m",
        _DYNAMO_WORKER_SCRIPT,
        bento_identifier,
        "--service-name",
        svc_name,
        "--worker-id",
        "$(CIRCUS.WID)",
    ]
    return args


def create_dynamo_watcher(
    bento_identifier: str,
    svc: ServiceProtocol,
    uds_path: str,
    scheduler: ResourceAllocator,
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Watcher, CircusSocket, str]:
    """Create a watcher for a Dynamo service in the dependency graph"""
    from dynamo.sdk.cli.circus import create_circus_watcher

    num_workers, resource_envs = scheduler.get_resource_envs(svc)
    uri, socket = _get_server_socket(svc, uds_path)
    args = _get_dynamo_worker_script(bento_identifier, svc.name)
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

    # use namespace from the service
    namespace, _ = svc.dynamo_address()

    # Create the watcher with updated environment
    watcher = create_circus_watcher(
        name=f"{namespace}_{svc.name}",
        args=args,
        numprocesses=num_workers,
        working_dir=working_dir,
        env=worker_env,
    )

    logger.info(f"Created watcher for {svc.name}'s in the {namespace} namespace")

    return watcher, socket, uri


@inject(squeeze_none=True)
def serve_dynamo_graph(
    bento_identifier: str | AnyService,
    working_dir: str | None = None,
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
    enable_local_planner: bool = False,
) -> CircusRunner:
    from dynamo.runtime.logging import configure_dynamo_logging
    from dynamo.sdk.cli.circus import create_arbiter, create_circus_watcher
    from dynamo.sdk.lib.loader import find_and_load_service

    from .allocator import ResourceAllocator

    configure_dynamo_logging(service_name=service_name)

    bento_id: str = ""
    namespace: str = ""
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
        svc = find_and_load_service(bento_identifier, working_dir)
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
        logger.info(f"Service '{service_name}' running in standalone mode")
        standalone = True

    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    num_workers, resource_envs = allocator.get_resource_envs(svc)
    uds_path = tempfile.mkdtemp(prefix="dynamo-uds-")
    try:
        if not service_name and not standalone:
            with contextlib.ExitStack() as port_stack:
                for name, dep_svc in svc.all_services().items():
                    if name == svc.name:
                        continue
                    if name in dependency_map:
                        continue
                    if not (
                        hasattr(dep_svc, "is_dynamo_component")
                        and dep_svc.is_dynamo_component()
                    ):
                        raise RuntimeError(
                            f"Service {dep_svc.name} is not a Dynamo component"
                        )
                    new_watcher, new_socket, uri = create_dynamo_watcher(
                        bento_id,
                        dep_svc,
                        uds_path,
                        allocator,
                        str(bento_path.absolute()),
                        env=env,
                    )
                    namespace, _ = dep_svc.dynamo_address()
                    watchers.append(new_watcher)
                    sockets.append(new_socket)
                    dependency_map[name] = uri
                # reserve one more to avoid conflicts
                port_stack.enter_context(reserve_free_port())

        dynamo_args = [
            "-m",
            _DYNAMO_WORKER_SCRIPT,
            bento_identifier,
            "--service-name",
            svc.name,
            "--worker-id",
            "$(CIRCUS.WID)",
        ]

        if hasattr(svc, "is_dynamo_component") and svc.is_dynamo_component():
            # resource_envs is the resource allocation (ie CUDA_VISIBLE_DEVICES) for each worker created by the allocator
            # these resource_envs are passed to each individual worker's environment which is set in serve_dynamo
            if resource_envs:
                dynamo_args.extend(["--worker-env", json.dumps(resource_envs)])
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

            watcher = create_circus_watcher(
                name=f"{namespace}_{svc.name}",
                args=dynamo_args,
                numprocesses=num_workers,
                working_dir=str(bento_path.absolute()),
                env=worker_env,
            )
            watchers.append(watcher)
            logger.info(
                f"Created watcher for {svc.name} with {num_workers} workers in the {namespace} namespace"
            )

        # inject runner map now
        inject_env = {"BENTOML_RUNNER_MAP": json.dumps(dependency_map)}

        for watcher in watchers:
            if watcher.env is None:
                watcher.env = inject_env
            else:
                watcher.env.update(inject_env)

        arbiter_kwargs: dict[str, Any] = {
            "watchers": watchers,
            "sockets": sockets,
        }

        arbiter = create_arbiter(**arbiter_kwargs)
        arbiter.exit_stack.callback(shutil.rmtree, uds_path, ignore_errors=True)
        if enable_local_planner:
            arbiter.exit_stack.callback(
                shutil.rmtree,
                os.environ.get(
                    DYN_LOCAL_STATE_DIR, os.path.expanduser("~/.dynamo/state")
                ),
                ignore_errors=True,
            )
            logger.warn(f"arbiter: {arbiter.endpoint}")

            # save deployment state for planner
            if not namespace:
                raise ValueError("No namespace found for service")

            # Track GPU allocation for each component
            component_resources = {}
            logger.info(f"Building component resources for {len(watchers)} watchers")

            for watcher in watchers:
                component_name = watcher.name
                logger.info(f"Processing watcher: {component_name}")

                # Extract worker info including GPU allocation
                worker_gpu_info: dict[str, Any] = {}

                # Extract service name from watcher name
                service_name = ""
                if component_name.startswith(f"{namespace}"):
                    service_name = component_name.replace(f"{namespace}_", "", 1)

                # Get GPU allocation from ResourceAllocator
                if (
                    not worker_gpu_info
                    and hasattr(allocator, "_service_gpu_allocations")
                    and service_name
                ):
                    gpu_allocations = getattr(allocator, "_service_gpu_allocations", {})
                    if service_name in gpu_allocations:
                        logger.info(
                            f"Found GPU allocation for {service_name} in ResourceAllocator: {gpu_allocations[service_name]}"
                        )
                        worker_gpu_info["allocated_gpus"] = gpu_allocations[
                            service_name
                        ]

                # Store final worker GPU info
                component_resources[component_name] = worker_gpu_info
                logger.info(f"Final GPU info for {component_name}: {worker_gpu_info}")

            logger.info(f"Completed component resources: {component_resources}")

            # Now create components dict with resources included
            components_dict = {
                watcher.name: {
                    "watcher_name": watcher.name,
                    "cmd": watcher.cmd
                    + " -m "
                    + " ".join(
                        watcher.args[1:]
                    )  # WAR because it combines python-m into 1 word
                    if hasattr(watcher, "args")
                    else watcher.cmd,
                    "resources": component_resources.get(watcher.name, {}),
                }
                for watcher in watchers
            }

            save_dynamo_state(
                namespace,
                arbiter.endpoint,
                components=components_dict,
                environment={
                    "DYNAMO_SERVICE_CONFIG": os.environ["DYNAMO_SERVICE_CONFIG"],
                    "SYSTEM_RESOURCES": {
                        "total_gpus": len(allocator.system_resources[NVIDIA_GPU]),
                        "gpu_info": [
                            str(gpu) for gpu in allocator.system_resources[NVIDIA_GPU]
                        ],
                    },
                },
            )

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
        return CircusRunner(arbiter=arbiter)
    except Exception:
        shutil.rmtree(uds_path, ignore_errors=True)
        raise
