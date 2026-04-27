# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Shared process management for OpenAI-compatible backend launchers."""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

LOGGER = logging.getLogger("dynamo.openai_backend.launcher")


def add_shared_launcher_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def build_worker_command(args: argparse.Namespace) -> list[str]:
    served_model_name = args.served_model_name or args.model
    upstream_base_url = (
        f"http://{args.engine_host}:{args.engine_port}{args.api_prefix.rstrip('/')}"
    )
    return [
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


async def wait_for_health(health_url: str, stop_event: asyncio.Event) -> None:
    import httpx

    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            try:
                response = await client.get(health_url, timeout=5.0)
                if response.is_success:
                    LOGGER.info("Engine became healthy at %s", health_url)
                    return
            except httpx.HTTPError:
                LOGGER.debug("Engine is not healthy yet", exc_info=True)

            await asyncio.sleep(2.0)

    raise asyncio.CancelledError("shutdown requested while waiting for engine health")


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


async def run_launcher(
    *,
    engine_command: list[str],
    worker_command: list[str],
    health_url: str,
) -> int:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)

    engine_process = await asyncio.create_subprocess_exec(*engine_command)
    worker_process = None

    try:
        health_task = asyncio.create_task(wait_for_health(health_url, stop_event))
        engine_startup_task = asyncio.create_task(engine_process.wait())
        stop_task = asyncio.create_task(stop_event.wait())
        startup_done, startup_pending = await asyncio.wait(
            [health_task, engine_startup_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop_task in startup_done:
            for task in startup_pending:
                await cancel_task(task)
            LOGGER.info("Shutdown requested before engine became healthy")
            return 0

        if engine_startup_task in startup_done:
            for task in startup_pending:
                await cancel_task(task)
            return_code = engine_startup_task.result()
            LOGGER.error(
                "Engine exited with status %s before becoming healthy",
                return_code,
            )
            return return_code or 1

        health_task.result()
        await cancel_task(stop_task)
        worker_process = await asyncio.create_subprocess_exec(*worker_command)

        wait_tasks = [
            engine_startup_task,
            asyncio.create_task(worker_process.wait()),
            asyncio.create_task(stop_event.wait()),
        ]
        done, pending = await asyncio.wait(
            wait_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if wait_tasks[2] in done:
            LOGGER.info("Shutdown requested")
            return 0

        if wait_tasks[0] in done:
            return_code = wait_tasks[0].result()
            LOGGER.error("Engine exited with status %s", return_code)
            return return_code or 1

        return_code = wait_tasks[1].result()
        LOGGER.error("Worker exited with status %s", return_code)
        return return_code or 1
    finally:
        if worker_process is not None:
            await terminate_process(worker_process, "worker")
        await terminate_process(engine_process, "engine")
