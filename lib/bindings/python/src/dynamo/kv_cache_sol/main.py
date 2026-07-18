# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os

import uvloop

from dynamo.llm import KvCacheSolEstimator
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only workload KV-cache speed-of-light estimator"
    )
    parser.add_argument(
        "--namespace", default=os.environ.get("DYN_NAMESPACE", "dynamo")
    )
    parser.add_argument("--component", default="kv-cache-sol")
    parser.add_argument("--horizon-secs", type=int, default=3600)
    parser.add_argument("--max-cache-blocks", type=int, default=5_000_000)
    parser.add_argument("--max-pending-requests", type=int, default=100_000)
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    args = parse_args()
    endpoint = runtime.endpoint(f"{args.namespace}.{args.component}.metrics")
    estimator = KvCacheSolEstimator(
        endpoint,
        horizon_secs=args.horizon_secs,
        max_cache_blocks=args.max_cache_blocks,
        max_pending_requests=args.max_pending_requests,
    )
    logger.info(
        "KV-cache speed-of-light estimator started: namespace=%s horizon=%ss",
        args.namespace,
        args.horizon_secs,
    )
    if not os.environ.get("DYN_SYSTEM_PORT"):
        logger.warning(
            "DYN_SYSTEM_PORT is not set; set it to expose estimator Prometheus metrics"
        )
    try:
        await asyncio.Event().wait()
    finally:
        estimator.shutdown()


def main() -> None:
    uvloop.run(worker())
