# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Thompson Sampling Router Service

Usage: python -m dynamo.thompson_router --endpoint <namespace.component.endpoint> [args]

This service provides a standalone KV-aware router with Thompson Sampling
(Beta bandits + LinTS contextual bandits) for any set of workers in a Dynamo
deployment. It layers learning-based worker selection on top of Dynamo's
native KvRouter for KV cache overlap and load signals.
"""

import asyncio
import logging
from typing import Optional

import uvloop

from dynamo.llm import KvRouter, KvRouterConfig
from dynamo.router.args import build_kv_router_config
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.thompson_router.args import ThompsonRouterConfig
from dynamo.thompson_router.args import parse_args as parse_thompson_args
from dynamo.thompson_router.hints import extract_hints
from dynamo.thompson_router.management import RouterManagementServer
from dynamo.thompson_router.router import KvThompsonRouter

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class ThompsonRouterHandler:
    """Handles routing requests to workers using Thompson Sampling + KV-aware routing.

    Reads agent hints from native Dynamo PreprocessedRequest fields (routing +
    annotations) so no custom processor is needed. Clients send hints via the
    standard nvext.agent_hints and nvext.annotations API.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        worker_endpoint_path: str,
        block_size: int,
        kv_router_config: KvRouterConfig,
        thompson_config: dict,
    ):
        self.runtime = runtime
        self.worker_endpoint_path = worker_endpoint_path
        self.block_size = block_size
        self.kv_router_config = kv_router_config
        self.thompson_config = thompson_config
        self.kv_router: Optional[KvRouter] = None
        self.thompson: Optional[KvThompsonRouter] = None

    async def initialize(self):
        """Initialize the KV router and Thompson layer."""
        try:
            parts = self.worker_endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path format: {self.worker_endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )

            worker_endpoint = self.runtime.endpoint(self.worker_endpoint_path)

            self.kv_router = KvRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
            )

            # Get live worker load monitor (created during KvRouter init)
            worker_load_monitor = self.kv_router.worker_load_monitor
            if worker_load_monitor is not None:
                logger.info("WorkerLoadMonitor available — live utilization metrics enabled")
            else:
                logger.warning("WorkerLoadMonitor unavailable — falling back to decode_blocks proxy")

            self.thompson = KvThompsonRouter(
                self.kv_router,
                self.thompson_config,
                worker_load_monitor=worker_load_monitor,
            )

            logger.info(
                "ThompsonRouterHandler initialized for %s (block_size=%d)",
                self.worker_endpoint_path,
                self.block_size,
            )

        except Exception as e:
            logger.error("Failed to initialize ThompsonRouterHandler: %s", e)
            raise

    async def generate(self, request):
        """Route via Thompson Sampling, forward to kv_router.generate(worker_id=chosen).

        Hint extraction:
          osl            <- routing.expected_output_tokens (from nvext.agent_hints.osl)
                            or annotations "osl:<n>"
          iat            <- routing.priority_jump * 1000 (from nvext.agent_hints.latency_sensitivity)
                            or annotations "iat:<ms>"
          prefix_id      <- annotations "prefix_id:<id>"
          total_requests <- annotations "total_requests:<n>"
          reuse_budget   <- total_requests - 1
        """
        if self.thompson is None or self.kv_router is None:
            raise RuntimeError("Router not initialized")

        token_ids = request["token_ids"]
        hints = extract_hints(request)

        decision = await self.thompson.pick_worker(
            token_ids=token_ids,
            prefix_id=hints["prefix_id"],
            reuse_budget=hints["reuse_budget"],
            osl=hints["osl"],
            iat=hints["iat"],
            tokens_in=hints["tokens_in"],
            latency_sensitivity=hints.get("latency_sensitivity", 2.0),
        )

        logger.debug(
            "Thompson decision: prefix=%s chosen=%d native=%d osl=%d iat=%d reuse=%d",
            hints["prefix_id"],
            decision.chosen,
            decision.native_pick,
            hints["osl"],
            hints["iat"],
            hints["reuse_budget"],
        )

        start_ms = asyncio.get_event_loop().time() * 1000
        tokens_out = 0

        async for worker_output in await self.kv_router.generate(
            token_ids=token_ids,
            model=request.get("model", "unknown"),
            stop_conditions=request.get("stop_conditions"),
            sampling_options=request.get("sampling_options"),
            output_options=request.get("output_options"),
            worker_id=decision.chosen,
        ):
            token_ids_out = worker_output.get("token_ids", [])  # type: ignore[attr-defined]
            tokens_out += len(token_ids_out)

            yield {
                "token_ids": token_ids_out,
                "tokens": worker_output.get("tokens"),  # type: ignore[attr-defined]
                "text": worker_output.get("text"),  # type: ignore[attr-defined]
                "cum_log_probs": worker_output.get("cum_log_probs"),  # type: ignore[attr-defined]
                "log_probs": worker_output.get("log_probs"),  # type: ignore[attr-defined]
                "top_logprobs": worker_output.get("top_logprobs"),  # type: ignore[attr-defined]
                "finish_reason": worker_output.get("finish_reason"),  # type: ignore[attr-defined]
                "stop_reason": worker_output.get("stop_reason"),  # type: ignore[attr-defined]
                "index": worker_output.get("index"),  # type: ignore[attr-defined]
                "disaggregated_params": worker_output.get("disaggregated_params"),  # type: ignore[attr-defined]
                "extra_args": worker_output.get("extra_args"),  # type: ignore[attr-defined]
                "completion_usage": worker_output.get("completion_usage"),  # type: ignore[attr-defined]
            }

        elapsed_ms = asyncio.get_event_loop().time() * 1000 - start_ms
        self.thompson.update_feedback(decision, elapsed_ms, tokens_out)


def parse_args(argv=None) -> ThompsonRouterConfig:
    """Parse router CLI arguments."""
    return parse_thompson_args(argv)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the standalone Thompson router service."""

    config = parse_args()

    logger.info("Starting Thompson Sampling Router Service")
    logger.info(
        "Configuration: endpoint=%s, block_size=%d, ts_weight=%.3f, "
        "beta_decay=%.3f, enable_lints=%s",
        config.endpoint,
        config.router_block_size,
        config.ts_weight,
        config.beta_decay,
        config.enable_lints,
    )

    kv_router_config = build_kv_router_config(config)
    thompson_config = config.to_thompson_config()

    handler = ThompsonRouterHandler(
        runtime,
        config.endpoint,
        config.router_block_size,
        kv_router_config,
        thompson_config,
    )
    await handler.initialize()

    mgmt = RouterManagementServer(handler.thompson, port=config.mgmt_port)
    await mgmt.start()

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.thompson_router.generate"
    )

    logger.info("Serving Thompson router endpoints...")

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("service", "thompson_router")],
        )
    except Exception as e:
        logger.error("Failed to serve endpoint: %s", e)
        raise
    finally:
        await mgmt.stop()
        logger.info("Thompson Sampling Router Service shutting down")


def main():
    """Entry point for the standalone Thompson router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
