# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import logging

import uvloop
from dynamo.llm import KvRouter, KvRouterConfig
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

try:
    from .thompson_sampling import ThompsonSamplingRouter
except ImportError:
    from thompson_sampling import ThompsonSamplingRouter

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class CustomRouterService:
    """Serve a Python-defined routing strategy as a Dynamo component."""

    def __init__(self, runtime: DistributedRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self.strategy = None

    async def initialize(self) -> None:
        worker_endpoint = self.runtime.endpoint(self.args.worker_endpoint)
        kv_router = KvRouter(
            endpoint=worker_endpoint,
            block_size=self.args.router_block_size,
            kv_router_config=KvRouterConfig(
                overlap_score_weight=self.args.overlap_score_weight,
                router_temperature=self.args.router_temperature,
                use_kv_events=not self.args.no_router_kv_events,
                router_track_active_blocks=not self.args.no_router_track_active_blocks,
                router_assume_kv_reuse=not self.args.no_router_assume_kv_reuse,
            ),
        )
        self.strategy = ThompsonSamplingRouter(
            kv_router,
            block_size=self.args.router_block_size,
            initial_reuse_budget=self.args.initial_reuse_budget,
            overlap_weight=self.args.overlap_weight,
            sticky_bonus=self.args.sticky_bonus,
            switch_penalty=self.args.switch_penalty,
            prefill_penalty_weight=self.args.prefill_penalty_weight,
            decode_penalty_weight=self.args.decode_penalty_weight,
            reuse_penalty_weight=self.args.reuse_penalty_weight,
            bandit_seed=self.args.bandit_seed,
        )
        logger.info(
            "Initialized custom Thompson router for %s -> %s.%s",
            self.args.worker_endpoint,
            self.args.namespace,
            self.args.component_name,
        )

    async def generate(self, request: dict):
        if self.strategy is None:
            raise RuntimeError("router service not initialized")
        async for output in self.strategy.generate(request):
            yield output

    async def select_worker(self, request: dict):
        if self.strategy is None:
            raise RuntimeError("router service not initialized")
        yield await self.strategy.inspect_selection(request)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example Thompson router built on top of Dynamo's KvRouter bindings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--worker-endpoint",
        required=True,
        help="Full worker endpoint path in namespace.component.endpoint form.",
    )
    parser.add_argument(
        "--namespace",
        default="dynamo",
        help="Namespace where this example router will serve its endpoints.",
    )
    parser.add_argument(
        "--component-name",
        default="custom_router",
        help="Component name used for the example router endpoints.",
    )
    parser.add_argument(
        "--router-block-size",
        type=int,
        default=128,
        help="KV block size used by the underlying KvRouter.",
    )
    parser.add_argument(
        "--overlap-score-weight",
        type=float,
        default=1.0,
        help="Forwarded into KvRouterConfig for its internal overlap scorer.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.0,
        help="Forwarded into KvRouterConfig for its built-in worker sampler.",
    )
    parser.add_argument(
        "--no-router-kv-events",
        action="store_true",
        help="Disable KV events in the underlying KvRouter.",
    )
    parser.add_argument(
        "--no-router-track-active-blocks",
        action="store_true",
        help="Disable active block tracking in the underlying KvRouter.",
    )
    parser.add_argument(
        "--no-router-assume-kv-reuse",
        action="store_true",
        help="Disable KvRouter's optimistic KV reuse assumption.",
    )
    parser.add_argument("--initial-reuse-budget", type=int, default=3)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--sticky-bonus", type=float, default=0.35)
    parser.add_argument("--switch-penalty", type=float, default=0.15)
    parser.add_argument("--prefill-penalty-weight", type=float, default=1.0)
    parser.add_argument("--decode-penalty-weight", type=float, default=0.35)
    parser.add_argument("--reuse-penalty-weight", type=float, default=0.25)
    parser.add_argument("--bandit-seed", type=int, default=7)
    args = parser.parse_args(argv)
    if len(args.worker_endpoint.split(".")) != 3:
        raise ValueError(
            "--worker-endpoint must use namespace.component.endpoint format"
        )
    if args.router_block_size <= 0:
        raise ValueError("--router-block-size must be > 0")
    return args


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    args = parse_args()
    service = CustomRouterService(runtime, args)
    await service.initialize()

    component = runtime.namespace(args.namespace).component(args.component_name)
    generate_endpoint = component.endpoint("generate")
    select_endpoint = component.endpoint("select_worker")

    await asyncio.gather(
        generate_endpoint.serve_endpoint(
            service.generate,
            graceful_shutdown=True,
            metrics_labels=[("service", "custom_router")],
        ),
        select_endpoint.serve_endpoint(
            service.select_worker,
            graceful_shutdown=True,
            metrics_labels=[("service", "custom_router")],
        ),
    )


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
