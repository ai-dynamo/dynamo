# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import logging
import signal
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.planner_factory import construct_planner
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

# How long a stopped planner waits for its endpoint to stop serving.
_SERVE_STOP_TIMEOUT_S = 10.0


class RequestType(BaseModel):
    text: str


async def start_planner(runtime: DistributedRuntime, config: PlannerConfig):
    planner = construct_planner(runtime=runtime, config=config)
    await planner._async_init()

    async def generate(request: RequestType):
        yield "mock endpoint"

    generate_endpoint = runtime.endpoint(f"{config.namespace}.Planner.generate")

    # serve_endpoint registers a health check target and sets HealthStatus::Ready
    # once the handler is registered with the transport.  Running it concurrently
    # with planner.run() ensures the system server (/health, /live) reports the
    # planner as ready only after _async_init() has completed.
    serve = generate_endpoint.serve_endpoint(
        generate,  # type: ignore[arg-type]
        health_check_payload={"text": "health"},
    )
    if config.retune is None:
        await asyncio.gather(serve, planner.run())
        return
    await _run_until_sigterm(runtime, serve, planner.run())


async def _run_until_sigterm(
    runtime: Any, serve: Awaitable[Any], run: Awaitable[None]
) -> None:
    """Serve and run the planner; SIGTERM stops the planner cleanly.

    Live retune holds a leadership Lease and may have a sweep in flight.
    Without a handler SIGTERM kills the process before ``run()``'s
    ``finally`` closes the retune controller and releases the Lease, so the
    next leader would wait out the Lease. The handler cancels only the run
    task; once its shutdown finished, the runtime is shut down to stop
    serving the endpoint and the call returns normally.
    """

    loop = asyncio.get_running_loop()
    run_task = asyncio.ensure_future(run)
    serve_task = asyncio.ensure_future(serve)
    terminating = False

    def on_sigterm() -> None:
        nonlocal terminating
        if terminating:
            return  # a second SIGTERM must not interrupt the shutdown
        terminating = True
        logger.info("SIGTERM received; stopping the planner")
        run_task.cancel()

    try:
        loop.add_signal_handler(signal.SIGTERM, on_sigterm)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        # No signal handlers on this platform or off the main thread.
        logger.warning("Cannot install a planner SIGTERM handler: %s", exc)
        installed = False
    try:
        await asyncio.gather(serve_task, run_task)
    except asyncio.CancelledError:
        if not (terminating and run_task.done() and run_task.cancelled()):
            raise
        logger.info("Planner stopped; shutting down the runtime")
        runtime.shutdown()
        done, _ = await asyncio.wait({serve_task}, timeout=_SERVE_STOP_TIMEOUT_S)
        if not done:
            serve_task.cancel()
        elif not serve_task.cancelled() and serve_task.exception() is not None:
            logger.warning(
                "Planner endpoint stopped with an error: %s", serve_task.exception()
            )
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)


async def init_planner(runtime: DistributedRuntime, config: PlannerConfig):
    await start_planner(runtime, config)


def _parse_config() -> PlannerConfig:
    parser = argparse.ArgumentParser(description="Dynamo Planner")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to a JSON/YAML config file",
    )
    args = parser.parse_args()
    return PlannerConfig.from_config_arg(args.config)


@dynamo_worker()
async def worker(runtime: DistributedRuntime, config: PlannerConfig):
    await init_planner(runtime, config)


def main():
    config = _parse_config()
    asyncio.run(worker(config))  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
