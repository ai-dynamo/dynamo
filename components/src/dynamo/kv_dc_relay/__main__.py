# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-scoped, single-DC Dynamo KV DC Relay component."""

import argparse
import asyncio
import logging
import os

import uvloop

from dynamo.llm import KvDcRelay
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamo model-scoped KV DC Relay")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Worker endpoint in namespace.component.endpoint form",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dc-id", required=True)
    return parser.parse_args()


class KvDcRelayDiagnostics:
    def __init__(self, relay: KvDcRelay):
        self._relay = relay

    async def stats(self, _request):
        yield await self._relay.stats()

    async def snapshot(self, _request):
        yield await self._relay.snapshot()

    async def health(self, _request):
        yield await self._relay.health()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = parse_args()
    if len(args.endpoint.split(".")) != 3:
        raise ValueError("--endpoint must use namespace.component.endpoint form")

    worker_endpoint = runtime.endpoint(args.endpoint)
    relay = KvDcRelay(worker_endpoint, args.model_name, args.dc_id)
    await relay.start()
    diagnostics = KvDcRelayDiagnostics(relay)
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    logger.info(
        "KV DC Relay started for model=%s dc_id=%s endpoint=%s",
        args.model_name,
        args.dc_id,
        args.endpoint,
    )
    try:
        await asyncio.gather(
            runtime.endpoint(f"{namespace}.kv_dc_relay.stats").serve_endpoint(
                diagnostics.stats,
                graceful_shutdown=True,
                metrics_labels=[("service", "kv_dc_relay")],
            ),
            runtime.endpoint(f"{namespace}.kv_dc_relay.snapshot").serve_endpoint(
                diagnostics.snapshot,
                graceful_shutdown=True,
                metrics_labels=[("service", "kv_dc_relay")],
            ),
            runtime.endpoint(f"{namespace}.kv_dc_relay.health").serve_endpoint(
                diagnostics.health,
                graceful_shutdown=True,
                metrics_labels=[("service", "kv_dc_relay")],
                health_check_payload={"text": "health"},
            ),
        )
    finally:
        await relay.shutdown()
        logger.info("KV DC Relay stopped")


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
