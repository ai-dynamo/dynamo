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


def extract_gpu_ids(gpu_resources: dict[str, Any]) -> list[str]:
    """Extract GPU IDs from resource dict containing CUDA_VISIBLE_DEVICES.

    Args:
        gpu_resources: Dict containing CUDA_VISIBLE_DEVICES key

    Returns:
        List of GPU IDs as strings
    """
    if not gpu_resources or "CUDA_VISIBLE_DEVICES" not in gpu_resources:
        return []

    gpu_str = gpu_resources["CUDA_VISIBLE_DEVICES"]
    return gpu_str.split(",")


def create_dynamo_watcher(
    bento_identifier: str,
    svc: ServiceProtocol,
    uds_path: str,
    scheduler: ResourceAllocator,
    component_resources: Dict[str, Any],
    working_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[list[Watcher], list[CircusSocket], dict[str, str]]:
    """Create a watcher for a Dynamo service in the dependency graph"""
    from .circus import _get_server_socket
    from dynamo.sdk.cli.circus import create_circus_watcher

    num_workers, resource_envs = scheduler.get_resource_envs(svc)
    namespace, comp_name = svc.dynamo_address()

    watchers = []
    sockets = []
    worker_uris = {}

    if num_workers > 0 and not resource_envs:
        resource_envs = [{} for _ in range(num_workers)]

    # create singleton watcher per worker
    for worker_idx in range(num_workers):
        uri, socket = _get_server_socket(svc, uds_path, worker_idx)
        sockets.append(socket)

        watcher_name = f"{namespace}_{comp_name}_{worker_idx}"
        worker_env_dict = resource_envs[worker_idx] if resource_envs else {}

        # store a mapping of watcher_name: gpu_resources
        component_resources[watcher_name] = worker_env_dict

        args = _get_dynamo_worker_script(bento_identifier, svc.name)
        args.extend(["--custom-component-name", watcher_name])
        args.extend(["--worker-env", json.dumps(worker_env_dict)])
        watcher_env = env.copy() if env else {}

        # Pass through the main service config
        if "DYNAMO_SERVICE_CONFIG" in os.environ:
            watcher_env["DYNAMO_SERVICE_CONFIG"] = os.environ["DYNAMO_SERVICE_CONFIG"]

        # Get service-specific environment variables from DYNAMO_SERVICE_ENVS
        if "DYNAMO_SERVICE_ENVS" in os.environ:
            try:
                service_envs = json.loads(os.environ["DYNAMO_SERVICE_ENVS"])
                if svc.name in service_envs:
                    service_args = service_envs[svc.name].get("ServiceArgs", {})
                    if "envs" in service_args:
                        watcher_env.update(service_args["envs"])
                        logger.info(
                            f"Added service-specific environment variables for {svc.name}"
                        )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse DYNAMO_SERVICE_ENVS: {e}")

        # watcher
        watcher = create_circus_watcher(
            name=watcher_name,
            args=args,
            singleton=True,
            working_dir=working_dir,
            env=watcher_env,
        )
        watchers.append(watcher)
        worker_uris[watcher_name] = uri

        logger.info(
            f"Created watcher {watcher_name} for worker {worker_idx} of service {svc.name}"
        )

    return watchers, sockets, worker_uris


@inject(squeeze_none=True)
def serve_dynamo_graph(
    bento_identifier: str | AnyService,
    working_dir: str | None = None,
    dependency_map: dict[str, str] | None = None,
    service_name: str = "",
    enable_local_planner: bool = False,
) -> CircusRunner:
    from dynamo.runtime.logging import configure_dynamo_logging
    from dynamo.sdk.cli.circus import create_arbiter
    from dynamo.sdk.lib.loader import find_and_load_service

    from .allocator import ResourceAllocator

    configure_dynamo_logging(service_name=service_name)

    bento_id: str = ""
    namespace: str = ""
    env: dict[str, Any] = {}
    component_resources: dict[str, Any] = {}
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

    if service_name and service_name != svc.name:
        svc = svc.find_dependent_by_name(service_name)
    uds_path = tempfile.mkdtemp(prefix="dynamo-uds-")
    try:
        with contextlib.ExitStack() as port_stack:
            services_to_process = {}
            if service_name:
                logger.info(f"Service '{service_name}' running in standalone mode")
                services_to_process[service_name] = svc
            else:
                services_to_process = svc.all_services()

            for name, service_to_run in services_to_process.items():
                if name in dependency_map:
                    continue

                if not (
                    hasattr(service_to_run, "is_dynamo_component")
                    and service_to_run.is_dynamo_component()
                ):
                    continue

                (
                    service_watchers,
                    service_sockets,
                    service_uris,
                ) = create_dynamo_watcher(
                    bento_id,
                    service_to_run,
                    uds_path,
                    allocator,
                    component_resources,
                    str(bento_path.absolute()),
                    env=env,
                )
                namespace, _ = service_to_run.dynamo_address()

                watchers.extend(service_watchers)
                sockets.extend(service_sockets)

                # Store the primary URI for service discovery
                if service_uris:
                    primary_uri = next(iter(service_uris.values()))
                    dependency_map[name] = primary_uri
                    dependency_map[f"{name}_workers"] = json.dumps(service_uris)

            # reserve one more to avoid conflicts
            port_stack.enter_context(reserve_free_port())

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

                # Extract the CUDA_VISIBLE_DEVICES from the key of component_resources
                if component_name in component_resources:
                    worker_gpu_info["allocated_gpus"] = extract_gpu_ids(
                        component_resources[component_name]
                    )
                    worker_gpu_info["required_gpus"] = len(
                        worker_gpu_info["allocated_gpus"]
                    )

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
