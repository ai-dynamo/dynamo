# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator

from dynamo._core import Context

from .engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine
from .worker import WorkerConfig


class SampleLLMEngine(LLMEngine):
    """Reference LLMEngine implementation.

    Generates rotating token IDs with configurable per-token latency.
    Useful for testing the Worker lifecycle end-to-end
    and as a template for engine leads implementing real backends.
    """

    def __init__(
        self,
        model_name: str = "sample-model",
        max_tokens: int = 16,
        delay: float = 0.01,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.delay = delay

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[SampleLLMEngine, WorkerConfig]:
        parser = argparse.ArgumentParser(description="Sample Dynamo backend")
        parser.add_argument("--model-name", default="sample-model")
        parser.add_argument("--namespace", default="dynamo")
        parser.add_argument("--component", default="sample")
        parser.add_argument("--endpoint", default="generate")
        parser.add_argument("--max-tokens", type=int, default=16)
        parser.add_argument("--delay", type=float, default=0.01)
        parser.add_argument("--endpoint-types", default="chat,completions")
        parser.add_argument("--discovery-backend", default="etcd")
        parser.add_argument("--request-plane", default="tcp")
        parser.add_argument("--event-plane", default="nats")
        args = parser.parse_args(argv)

        engine = cls(
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            delay=args.delay,
        )
        worker_config = WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=args.model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        return EngineConfig(
            model=self.model_name,
            served_model_name=self.model_name,
            context_length=2048,
            kv_cache_block_size=16,
            total_kv_blocks=1000,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        token_ids = request.get("token_ids", [])
        prompt_len = len(token_ids)
        stop_conditions = request.get("stop_conditions", {})
        max_new = stop_conditions.get("max_tokens") or self.max_tokens

        for i in range(max_new):
            if context.is_stopped():
                yield {
                    "token_ids": [],
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": i,
                        "total_tokens": prompt_len + i,
                    },
                }
                break
            await asyncio.sleep(self.delay)
            token_id = (i + 1) % 32000
            out: GenerateChunk = {"token_ids": [token_id]}
            if i == max_new - 1:
                out["finish_reason"] = "length"
                out["completion_usage"] = {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": max_new,
                    "total_tokens": prompt_len + max_new,
                }
            yield out

    async def cleanup(self) -> None:
        pass
