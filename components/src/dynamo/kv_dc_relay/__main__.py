# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DC-scoped, multi-endpoint Dynamo KV Relay component."""

import asyncio
import hashlib
import logging
import os

import uvloop

from dynamo.llm import KvDcRelay
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .cli import monitor_relay, parse_args, schedule_awaitable

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class KvDcRelayDiagnostics:
    def __init__(self, relay: KvDcRelay):
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


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = parse_args()
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    relay_endpoint = runtime.endpoint(f"{namespace}.kv_dc_relay.control")
    relay = KvDcRelay(
        relay_endpoint,
        args.dc_id,
        namespaces=list(args.namespaces),
        endpoint_prefixes=list(args.endpoint_prefixes),
        watch_all=args.watch_all,
        expected_unique_blocks=args.expected_unique_blocks,
        bind=args.bind,
        tls_server_cert=args.tls_server_cert,
        tls_server_key=args.tls_server_key,
        tls_client_ca=args.tls_client_ca,
        tuning=dict(args.tuning) or None,
    )
    await relay.start()
    diagnostics = KvDcRelayDiagnostics(relay)
    relay_identity = hashlib.sha256(args.dc_id.encode()).hexdigest()[:32]
    diagnostics_component = f"kv_dc_relay_{relay_identity}"

    logger.info(
        "KV DC Relay started for dc_id=%s namespaces=%s watch_all=%s "
        "endpoint_prefixes=%s wan_bind=%s",
        args.dc_id,
        args.namespaces,
        args.watch_all,
        args.endpoint_prefixes,
        args.bind,
    )
    endpoint_tasks = []
    try:
        if hasattr(relay, "stats") and hasattr(relay, "snapshot"):
            endpoint_tasks.append(
                schedule_awaitable(
                    runtime.endpoint(
                        f"{namespace}.{diagnostics_component}.stats"
                    ).serve_endpoint(
                        diagnostics.stats,
                        graceful_shutdown=True,
                        metrics_labels=[("service", "kv_dc_relay")],
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
                        metrics_labels=[("service", "kv_dc_relay")],
                    )
                )
            )
        else:
            logger.info(
                "KV DC Relay rich diagnostics are disabled in this build; "
                "enable the ckf-diagnostics Cargo feature to expose them"
            )
        endpoint_tasks.append(
            schedule_awaitable(
                runtime.endpoint(
                    f"{namespace}.{diagnostics_component}.health"
                ).serve_endpoint(
                    diagnostics.health,
                    graceful_shutdown=True,
                    metrics_labels=[("service", "kv_dc_relay")],
                    health_check_payload={"text": "health"},
                )
            )
        )
        await monitor_relay(relay, endpoint_tasks)
    finally:
        for task in endpoint_tasks:
            task.cancel()
        await asyncio.gather(*endpoint_tasks, return_exceptions=True)
        await relay.shutdown()
        logger.info("KV DC Relay stopped")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
