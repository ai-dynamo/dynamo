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

import asyncio
import inspect
import json
import logging
import os
import time
import typing as t
from typing import Any

import click
import uvicorn
import uvloop
from fastapi.responses import StreamingResponse

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker
from dynamo.sdk import dynamo_context
from dynamo.sdk.core.protocol.interface import DynamoTransport, LinkedServices
from dynamo.sdk.lib.loader import find_and_load_service
from dynamo.sdk.lib.utils import get_host_port

logger = logging.getLogger(__name__)


def add_fastapi_routes(app, service, class_instance):
    """
    Add FastAPI routes for Dynamo endpoints supporting HTTP transport.

    Args:
        app: FastAPI app instance
        service: Dynamo service instance
        class_instance: Instance of the service class
    """

    added_routes = []
    for name, endpoint in service.get_dynamo_endpoints().items():
        if DynamoTransport.HTTP in endpoint.transports:
            path = name if name.startswith("/") else f"/{name}"
            # Bind the method to the class instance
            bound_method = endpoint.func.__get__(class_instance)

            # Check if the method is a generator or async generator
            is_streaming = inspect.isasyncgenfunction(
                bound_method
            ) or inspect.isgeneratorfunction(bound_method)

            # Set up appropriate response model and response class
            if is_streaming:
                logger.info(f"Registering streaming endpoint {path}")
                app.add_api_route(
                    path,
                    bound_method,
                    methods=["POST"],
                    response_class=StreamingResponse,
                )
            else:
                logger.info(f"Registering regular endpoint {path}")
                app.add_api_route(
                    path,
                    bound_method,
                    methods=["POST"],
                )

            added_routes.append(path)
            logger.info(f"Added API route {path} to FastAPI app")
    return added_routes


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
@click.option(
    "--custom-component-name",
    required=False,
    type=click.STRING,
    default=None,
    help="If set, use this custom component name instead of the default service name",
)
@click.option(
    "--target",
    type=click.STRING,
    default="dynamo",
    help="Specify the target: 'dynamo' or 'bento'.",
)
def main(
    bento_identifier: str,
    service_name: str,
    runner_map: str | None,
    worker_env: str | None,
    worker_id: int | None,
    custom_component_name: str | None,
    target: str,
) -> None:
    # hack to avoid bentoml from respawning the workers after their leases are revoked
    os.environ["BENTOML_CONTAINERIZED"] = "true"

    """Start a worker for the given service - either Dynamo or regular service"""
    from bentoml._internal.container import BentoMLContainer
    from bentoml._internal.context import server_context

    from dynamo.runtime.logging import configure_dynamo_logging
    from dynamo.sdk.cli.utils import configure_target_environment
    from dynamo.sdk.core.runner import TargetEnum

    configure_target_environment(TargetEnum(target))

    dynamo_context["service_name"] = service_name
    dynamo_context["runner_map"] = runner_map
    dynamo_context["worker_id"] = worker_id

    # Ensure environment variables are set before we initialize
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
    service = find_and_load_service(bento_identifier)
    if service_name and service_name != service.name:
        service = service.find_dependent_by_name(service_name)

    # Set namespace in dynamo_context if service is a dynamo component
    namespace, _ = service.dynamo_address()
    dynamo_context["namespace"] = namespace

    configure_dynamo_logging(service_name=service_name, worker_id=worker_id)
    if runner_map:
        BentoMLContainer.remote_runner_mapping.set(
            t.cast(t.Dict[str, str], json.loads(runner_map))
        )

    # TODO: test this with a deep chain of services
    LinkedServices.remove_unused_edges()
    # Check if Dynamo is enabled for this service
    if worker_id is not None:
        server_context.worker_index = worker_id

    @dynamo_worker()
    async def worker(runtime: DistributedRuntime):
        global dynamo_context
        dynamo_context["runtime"] = runtime
        if service_name and service_name != service.name:
            server_context.service_type = "service"
        else:
            server_context.service_type = "entry_service"

        server_context.service_name = service.name
        # Get Dynamo configuration and create component
        namespace, component_name = service.dynamo_address()
        logger.info(f"Registering component {namespace}/{component_name}")
        component = runtime.namespace(namespace).component(component_name)

        try:
            # Create service first
            await component.create_service()
            logger.info(f"Created {service.name} component")

            # Set runtime on all dependencies
            for dep in service.dependencies.values():
                dep.set_runtime(runtime)
                logger.debug(f"Set runtime for dependency: {dep}")

            # Then register all Dynamo endpoints
            dynamo_endpoints = service.get_dynamo_endpoints()

            endpoints = []
            for name, endpoint in dynamo_endpoints.items():
                td_endpoint = component.endpoint(name)
                logger.debug(f"Registering endpoint '{name}'")
                endpoints.append(td_endpoint)
                # Bind an instance of inner to the endpoint
            dynamo_context["component"] = component
            dynamo_context["endpoints"] = endpoints
            class_instance = service.inner()
            dynamo_handlers = []
            for name, endpoint in dynamo_endpoints.items():
                bound_method = endpoint.func.__get__(class_instance)
                # Only pass request type for now, use Any for response
                # TODO: Handle a dynamo_endpoint not having types
                # TODO: Handle multiple endpoints in a single component
                dynamo_wrapped_method = dynamo_endpoint(endpoint.request_type, Any)(
                    bound_method
                )
                dynamo_handlers.append(dynamo_wrapped_method)
            # Run startup hooks before setting up endpoints
            for name, member in vars(class_instance.__class__).items():
                if callable(member) and getattr(
                    member, "__dynamo_startup_hook__", False
                ):
                    logger.debug(f"Running startup hook: {name}")
                    result = getattr(class_instance, name)()
                    if inspect.isawaitable(result):
                        # await on startup hook async_on_start
                        await result
                        logger.debug(f"Completed async startup hook: {name}")
                    else:
                        logger.info(f"Completed startup hook: {name}")
            logger.info(
                f"Starting {service.name} instance with all registered endpoints"
            )
            # Launch serve_endpoint for all endpoints concurrently
            tasks = [
                endpoint.serve_endpoint(handler)
                for endpoint, handler in zip(endpoints, dynamo_handlers)
            ]
            if tasks:
                # Wait for all tasks to complete
                await asyncio.gather(*tasks)
            else:
                msg = f"No Dynamo endpoints found in service {service.name} but keeping service alive"
                logger.info(msg)
                # Even with no endpoints, we should keep the service running
                # until explicitly terminated
                try:
                    # Create an event to wait on indefinitely until interrupted
                    stop_event = asyncio.Event()
                    # Wait for the event that will never be set unless interrupted
                    await stop_event.wait()
                except asyncio.CancelledError:
                    logger.info("Service execution cancelled")
                except KeyboardInterrupt:
                    logger.info("Service interrupted by user")
                except Exception as e:
                    logger.exception(
                        f"Unexpected error while keeping service alive: {e}"
                    )
                finally:
                    logger.info("Service shutting down")
        except Exception as e:
            logger.error(f"Error in Dynamo component setup: {str(e)}")
            raise

    # if the service has a FastAPI app, add the worker as an event handler
    def web_worker():
        if not service.app:
            return

        # Create the class instance
        class_instance = service.inner()
        # TODO: init hooks
        # Add API routes to the FastAPI app
        added_routes = add_fastapi_routes(service.app, service, class_instance)
        if added_routes:
            # Configure uvicorn with graceful shutdown
            host, port = get_host_port()
            config = uvicorn.Config(service.app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)

            # Start the server with graceful shutdown handling
            logger.info(
                f"Starting FastAPI server on {config.host}:{config.port} with routes: {added_routes}"
            )
            server.run()
        else:
            logger.warning("No API routes found, not starting FastAPI server")
            # Keep the process running until interrupted
            logger.info("Service is running, press Ctrl+C to stop")
            while True:
                try:
                    # Sleep in small increments to respond to signals quickly
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    logger.info("Gracefully shutting down FastAPI process")
                    break

    uvloop.install()
    start_http_server = False
    for endpoint in service.get_dynamo_endpoints().values():
        logger.debug(f"Checking transports for endpoint: {endpoint.transports}")
        if DynamoTransport.HTTP in endpoint.transports:
            start_http_server = True
            break
    if start_http_server:
        web_worker()
    else:
        asyncio.run(worker())


if __name__ == "__main__":
    main()
