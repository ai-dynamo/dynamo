# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone, multi-slot KV state-agent host."""

import argparse
import asyncio
import logging
import os
from typing import Any

import uvloop

from dynamo.llm import KvStateAgentHost
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamo KV state-agent host")
    parser.add_argument("--max-slots", type=int, default=8)
    return parser.parse_args()


class HostDiagnostics:
    def __init__(self, host: KvStateAgentHost):
        self._host = host

    async def health(self, _request):
        status = await self._host.status()
        if not status.get("healthy", False):
            raise RuntimeError(
                status.get("error", "KV state-agent host supervisor is not running")
            )
        yield status


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = parse_args()
    if args.max_slots <= 0:
        raise ValueError("--max-slots must be greater than zero")

    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    host = KvStateAgentHost(
        runtime.endpoint(f"{namespace}.kv_state_agent.control"),
        args.max_slots,
    )
    await host.start()
    health_task: asyncio.Future[Any] | None = None
    termination_task: asyncio.Future[Any] | None = None
    try:
        diagnostics = HostDiagnostics(host)
        health_task = asyncio.ensure_future(
            runtime.endpoint(f"{namespace}.kv_state_agent.health").serve_endpoint(
                diagnostics.health,
                graceful_shutdown=True,
                metrics_labels=[("service", "kv_state_agent")],
                health_check_payload={"text": "health"},
            )
        )
        # PyO3 exposes this as an awaitable Future, not necessarily a coroutine.
        termination_task = asyncio.ensure_future(host.wait_terminated())
        logger.info("KV state-agent host started with max_slots=%d", args.max_slots)
        done, _ = await asyncio.wait(
            {health_task, termination_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if termination_task in done:
            raise RuntimeError("KV state-agent host intent watch terminated")
        await health_task
    finally:
        tasks = [task for task in (health_task, termination_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await host.shutdown()
        logger.info("KV state-agent host stopped")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
