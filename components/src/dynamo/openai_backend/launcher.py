# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Launch a local OpenAI-compatible engine and the Dynamo backend worker."""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Sequence

LOGGER = logging.getLogger("dynamo.openai_backend.launcher")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an OpenAI-compatible engine and Dynamo worker together."
    )
    parser.add_argument(
        "--engine",
        default="sglang",
        choices=["sglang"],
        help="Inference engine to launch. Currently only 'sglang' is supported.",
    )
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
    parser.add_argument("--upstream-api-key", default=None)
    parser.add_argument(
        "engine_args",
        nargs=argparse.REMAINDER,
        help="Additional engine args after '--', passed through unchanged.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _strip_remainder_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _engine_command(args: argparse.Namespace) -> list[str]:
    served_model_name = args.served_model_name or args.model
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--served-model-name",
        served_model_name,
        "--host",
        args.engine_host,
        "--port",
        str(args.engine_port),
    ]
    command.extend(_strip_remainder_separator(list(args.engine_args)))
    return command


def _worker_command(args: argparse.Namespace) -> list[str]:
    served_model_name = args.served_model_name or args.model
    upstream_base_url = (
        f"http://{args.engine_host}:{args.engine_port}{args.api_prefix.rstrip('/')}"
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
    if args.upstream_api_key:
        command.extend(["--upstream-api-key", args.upstream_api_key])
    return command


async def _wait_for_health(args: argparse.Namespace, stop_event: asyncio.Event) -> None:
    import httpx

    health_url = f"http://{args.engine_host}:{args.engine_port}{args.health_path}"

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


async def _terminate_process(
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


async def _cancel_task(task: asyncio.Task[object]) -> None:
    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _run(args: argparse.Namespace) -> int:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)

    engine_process = await asyncio.create_subprocess_exec(*_engine_command(args))
    worker_process = None

    try:
        health_task = asyncio.create_task(_wait_for_health(args, stop_event))
        engine_startup_task = asyncio.create_task(engine_process.wait())
        stop_task = asyncio.create_task(stop_event.wait())
        startup_done, startup_pending = await asyncio.wait(
            [health_task, engine_startup_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop_task in startup_done:
            for task in startup_pending:
                await _cancel_task(task)
            LOGGER.info("Shutdown requested before engine became healthy")
            return 0

        if engine_startup_task in startup_done:
            for task in startup_pending:
                await _cancel_task(task)
            return_code = engine_startup_task.result()
            LOGGER.error(
                "Engine exited with status %s before becoming healthy",
                return_code,
            )
            return return_code or 1

        health_task.result()
        await _cancel_task(stop_task)
        worker_process = await asyncio.create_subprocess_exec(*_worker_command(args))

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
            await _terminate_process(worker_process, "worker")
        await _terminate_process(engine_process, "engine")


def launch_main(argv: Sequence[str] | None = None) -> None:
    _configure_logging()
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    import uvloop

    raise SystemExit(uvloop.run(_run(args)))
