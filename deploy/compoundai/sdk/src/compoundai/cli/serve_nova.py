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

import asyncio
import inspect
import json
import logging
import os
import random
import string
import typing as t
from typing import Any

import click
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker

logger = logging.getLogger("compoundai.serve.nova")


def generate_run_id():
    """Generate a random 6-character run ID"""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--service-name", type=click.STRING, required=False, default="")
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--worker-env", type=click.STRING, default=None, help="Environment variables"
)
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
def main(
    bento_identifier: str,
    service_name: str,
    runner_map: str | None,
    worker_env: str | None,
    worker_id: int | None,
) -> None:
    """Start a worker for the given service - either Nova or regular service"""
    from _bentoml_impl.loader import import_service
    from bentoml._internal.container import BentoMLContainer
    from bentoml._internal.context import server_context
    from bentoml._internal.log import configure_server_logging

    run_id = generate_run_id()

    # print the contents of the environment variable BENTOML_RUNNER_MAP
    print(f"BENTOML_RUNNER_MAP: {os.environ['BENTOML_RUNNER_MAP']}")

    # Import service first to check configuration
    service = import_service(bento_identifier)
    if service_name and service_name != service.name:
        service = service.find_dependent_by_name(service_name)

    # Handle worker environment if specified
    if worker_env:
        env_list: list[dict[str, t.Any]] = json.loads(worker_env)
        if worker_id is not None:
            worker_key = worker_id - 1
            if worker_key >= len(env_list):
                raise IndexError(
                    f"Worker ID {worker_id} is out of range, "
                    f"the maximum worker ID is {len(env_list)}"
                )
            os.environ.update(env_list[worker_key])

    configure_server_logging()
    if runner_map:
        BentoMLContainer.remote_runner_mapping.set(
            t.cast(t.Dict[str, str], json.loads(runner_map))
        )

    # Check if Nova is enabled for this service
    if service.is_nova_component():
        if worker_id is not None:
            server_context.worker_index = worker_id
        class_instance = service.inner()

        @triton_worker()
        async def worker(runtime: DistributedRuntime):
            if service_name and service_name != service.name:
                server_context.service_type = "service"
            else:
                server_context.service_type = "entry_service"

            server_context.service_name = service.name

            # Get Nova configuration and create component
            namespace, component_name = service.nova_address()
            logger.info(
                f"[{run_id}] Registering component {namespace}/{component_name}"
            )
            component = runtime.namespace(namespace).component(component_name)

            try:
                # Create service first
                await component.create_service()
                logger.info(f"[{run_id}] Created {service.name} component")

                # Run startup hooks before setting up endpoints
                for name, member in vars(class_instance.__class__).items():
                    if callable(member) and getattr(
                        member, "__bentoml_startup_hook__", False
                    ):
                        logger.info(f"[{run_id}] Running startup hook: {name}")
                        result = getattr(class_instance, name)()
                        if inspect.isawaitable(result):
                            await result
                            logger.info(
                                f"[{run_id}] Completed async startup hook: {name}"
                            )
                        else:
                            logger.info(f"[{run_id}] Completed startup hook: {name}")

                # Set runtime on all dependencies
                for dep in service.dependencies.values():
                    dep.set_runtime(runtime)
                    logger.info(f"[{run_id}] Set runtime for dependency: {dep}")

                # Then register all Nova endpoints
                nova_endpoints = service.get_nova_endpoints()
                if not nova_endpoints:
                    error_msg = f"[{run_id}] FATAL ERROR: No Nova endpoints found in service {service.name}!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                print(f"[{run_id}] Nova endpoints: {nova_endpoints}")
                for name, endpoint in nova_endpoints.items():
                    td_endpoint = component.endpoint(name)
                    logger.info(f"[{run_id}] Registering endpoint '{name}'")
                    # Bind an instance of inner to the endpoint
                    bound_method = endpoint.func.__get__(class_instance)
                    # Only pass request type for now, use Any for response
                    # TODO: Handle a triton_endpoint not having types
                    # TODO: Handle multiple endpoints in a single component
                    triton_wrapped_method = triton_endpoint(endpoint.request_type, Any)(
                        bound_method
                    )
                    result = await td_endpoint.serve_endpoint(triton_wrapped_method)
                    # WARNING: unreachable code :( because serve blocks
                    logger.info(f"[{run_id}] Result: {result}")
                    logger.info(f"[{run_id}] Registered endpoint '{name}'")

                logger.info(
                    f"[{run_id}] Started {service.name} instance with all endpoints registered"
                )
                logger.info(
                    f"[{run_id}] Available endpoints: {service.list_nova_endpoints()}"
                )
            except Exception as e:
                logger.error(f"[{run_id}] Error in Nova component setup: {str(e)}")
                raise

        asyncio.run(worker())


if __name__ == "__main__":
    main()
