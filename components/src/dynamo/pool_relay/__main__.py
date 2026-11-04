# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool Relay component for Dynamo inference endpoints."""

import asyncio
import hashlib
import logging
import os
import sys
from collections.abc import Mapping

import uvloop

from dynamo.llm import PoolRelay
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .cli import monitor_relay, parse_args, schedule_awaitable

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class PoolRelayDiagnostics:
    def __init__(self, relay: PoolRelay):
        self._relay = relay

    async def stats(self, _request):
        yield await self._relay.stats()

    async def snapshot(self, request):
        serving_endpoint = request.get("serving_endpoint")
        if not serving_endpoint:
            raise ValueError("snapshot requests require serving_endpoint")
        yield await self._relay.snapshot(serving_endpoint)

    async def health(self, _request):
        yield await self._relay.health()


def _handle_relay_shutdown(health: Mapping[str, object]) -> None:
    host_error = health.get("host_last_error")
    wan_error = health.get("wan_last_error")
    if host_error is None and wan_error is None:
        logger.info("Pool Relay host stopped during runtime shutdown")
        return

    terminal_error = host_error if host_error is not None else wan_error
    logger.error("Pool Relay stopped after terminal failure: %s", terminal_error)
    raise SystemExit(1)


async def _shutdown_relay(relay: PoolRelay, *, classify_host_failure: bool) -> None:
    try:
        await relay.shutdown()
    except BaseException:
        if classify_host_failure:
            raise
        logger.exception(
            "Pool Relay shutdown also failed while preserving an endpoint exception"
        )
        return
    if classify_host_failure:
        _handle_relay_shutdown(await relay.health())


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = parse_args()
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    relay_endpoint = runtime.endpoint(f"{namespace}.pool_relay.control")
    relay = PoolRelay(
        relay_endpoint,
        args.cluster_id,
        namespaces=list(args.namespaces),
        endpoint_prefixes=list(args.endpoint_prefixes),
        watch_all=args.watch_all,
        expected_unique_blocks=args.expected_unique_blocks,
        bind=args.bind,
        tuning=dict(args.tuning) or None,
        sources_file=args.sources_file,
        connection_revision=args.connection_revision,
        global_discovery_url=args.global_discovery_url,
        routing_pool=args.routing_pool,
    )
    await relay.start()
    diagnostics = PoolRelayDiagnostics(relay)
    relay_identity = hashlib.sha256(args.cluster_id.encode()).hexdigest()[:32]
    diagnostics_component = f"pool_relay_{relay_identity}"

    logger.info(
        "Pool Relay started for cluster_id=%s namespaces=%s watch_all=%s "
        "endpoint_prefixes=%s wan_bind=%s global_discovery_url=%s",
        args.cluster_id,
        args.namespaces,
        args.watch_all,
        args.endpoint_prefixes,
        args.bind,
        args.global_discovery_url,
    )
    endpoint_tasks: list[asyncio.Future[object]] = []
    # A terminal host failure only resolves wait_for_shutdown(); the diagnostics
    # endpoints would otherwise keep the process alive and answering health checks
    # for a relay that is no longer ingesting anything.
    try:
        if hasattr(relay, "stats") and hasattr(relay, "snapshot"):
            endpoint_tasks.append(
                schedule_awaitable(
                    runtime.endpoint(
                        f"{namespace}.{diagnostics_component}.stats"
                    ).serve_endpoint(
                        diagnostics.stats,
                        graceful_shutdown=True,
                        metrics_labels=[("service", "pool_relay")],
                    )
                )
            )
            endpoint_tasks.append(
                schedule_awaitable(
                    runtime.endpoint(
                        f"{namespace}.{diagnostics_component}.snapshot"
                    ).serve_endpoint(
                        diagnostics.snapshot,
                        graceful_shutdown=True,
                        metrics_labels=[("service", "pool_relay")],
                    )
                )
            )
        else:
            logger.info(
                "Pool Relay rich diagnostics are disabled in this build; "
                "enable the ckf-diagnostics Cargo feature to expose them"
            )
        endpoint_tasks.append(
            schedule_awaitable(
                runtime.endpoint(
                    f"{namespace}.{diagnostics_component}.health"
                ).serve_endpoint(
                    diagnostics.health,
                    graceful_shutdown=True,
                    metrics_labels=[("service", "pool_relay")],
                    health_check_payload={"text": "health"},
                )
            )
        )
        await monitor_relay(relay, endpoint_tasks)
    finally:
        classify_host_failure = sys.exc_info()[0] is None
        for task in endpoint_tasks:
            task.cancel()
        await asyncio.gather(*endpoint_tasks, return_exceptions=True)
        await _shutdown_relay(relay, classify_host_failure=classify_host_failure)
        logger.info("Pool Relay stopped")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
