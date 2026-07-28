# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process management for OpenAI-compatible backend launchers."""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

LOGGER = logging.getLogger("dynamo.openai_backend.launcher")


def add_shared_launcher_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--model", required=True, help="Model path or identifier.")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional public model name to register with Dynamo and the engine.",
    )
    parser.add_argument("--engine-host", default="127.0.0.1")
    parser.add_argument("--engine-port", type=int, default=30000)
    parser.add_argument("--api-prefix", default="/v1")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument(
        "engine_args",
        nargs=argparse.REMAINDER,
        help="Additional engine args after '--', passed through unchanged.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def strip_remainder_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_health_url(args: argparse.Namespace) -> str:
    return f"http://{args.engine_host}:{args.engine_port}{args.health_path}"


def build_worker_command(
    args: argparse.Namespace,
    *,
    priority_multiplier: int | None = None,
    upstream_port: int | None = None,
) -> list[str]:
    served_model_name = args.served_model_name or args.model
    port = upstream_port if upstream_port is not None else args.engine_port
    upstream_base_url = (
        f"http://{args.engine_host}:{port}{args.api_prefix.rstrip('/')}"
    )
    command = [
        sys.executable,
        "-m",
        "dynamo.openai_backend._worker",
        "--model",
        args.model,
        "--served-model-name",
        served_model_name,
        "--upstream-base-url",
        upstream_base_url,
        "--upstream-health-path",
        args.health_path,
    ]
    if priority_multiplier is not None:
        command.extend(["--priority-multiplier", str(priority_multiplier)])
    return command


async def wait_for_health(
    health_url: str,
    stop_event: asyncio.Event,
    name: str = "engine",
) -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            try:
                response = await client.get(health_url, timeout=5.0)
                if response.is_success:
                    LOGGER.info("%s became healthy at %s", name.capitalize(), health_url)
                    return
            except httpx.HTTPError:
                LOGGER.debug("%s is not healthy yet", name.capitalize(), exc_info=True)

            await asyncio.sleep(2.0)

    raise asyncio.CancelledError(f"shutdown requested while waiting for {name} health")


async def terminate_process(
    process: asyncio.subprocess.Process,
    name: str,
    timeout: float = 20.0,
) -> None:
    if process.returncode is not None:
        return

    LOGGER.info("Terminating %s process pid=%s", name, process.pid)
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        LOGGER.warning("Killing unresponsive %s process pid=%s", name, process.pid)
        process.kill()
        await process.wait()


async def cancel_task(task: asyncio.Task[object]) -> None:
    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _gate_until_healthy(
    *,
    name: str,
    health_url: str,
    process_wait_task: asyncio.Task[int],
    stop_event: asyncio.Event,
) -> int | None:
    health_task = asyncio.create_task(wait_for_health(health_url, stop_event, name))
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        [health_task, process_wait_task, stop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if stop_task in done:
        for task in pending:
            if task is not process_wait_task:
                await cancel_task(task)
        LOGGER.info("Shutdown requested before %s became healthy", name)
        return 0

    if process_wait_task in done:
        for task in pending:
            await cancel_task(task)
        return_code = process_wait_task.result()
        LOGGER.error(
            "%s exited with status %s before becoming healthy",
            name.capitalize(),
            return_code,
        )
        return return_code or 1

    health_task.result()
    await cancel_task(stop_task)
    return None


async def run_launcher(
    *,
    engine_command: list[str],
    worker_command: list[str],
    health_url: str,
    router_command: list[str] | None = None,
    router_health_url: str | None = None,
) -> int:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)

    engine_process = await asyncio.create_subprocess_exec(*engine_command)
    router_process = None
    worker_process = None

    try:
        engine_wait_task = asyncio.create_task(engine_process.wait())
        exit_code = await _gate_until_healthy(
            name="engine",
            health_url=health_url,
            process_wait_task=engine_wait_task,
            stop_event=stop_event,
        )
        if exit_code is not None:
            return exit_code

        watchers = [("engine", engine_wait_task)]

        if router_command is not None:
            router_process = await asyncio.create_subprocess_exec(*router_command)
            router_wait_task = asyncio.create_task(router_process.wait())
            exit_code = await _gate_until_healthy(
                name="router",
                health_url=router_health_url or health_url,
                process_wait_task=router_wait_task,
                stop_event=stop_event,
            )
            if exit_code is not None:
                return exit_code
            watchers.append(("router", router_wait_task))

        worker_process = await asyncio.create_subprocess_exec(*worker_command)
        watchers.append(("worker", asyncio.create_task(worker_process.wait())))

        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            [task for _, task in watchers] + [stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if stop_task in done:
            LOGGER.info("Shutdown requested")
            return 0

        for name, task in watchers:
            if task in done:
                return_code = task.result()
                LOGGER.error("%s exited with status %s", name.capitalize(), return_code)
                return return_code or 1

        return 1
    finally:
        if worker_process is not None:
            await terminate_process(worker_process, "worker")
        if router_process is not None:
            await terminate_process(router_process, "router")
        await terminate_process(engine_process, "engine")
