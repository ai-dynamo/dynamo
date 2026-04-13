# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample backend — CPU-only reference implementation.

Usage:
    python -m dynamo.common.backend.sample_main [--model-name ...]
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator

from dynamo._core import Context

from .serve import EngineConfig, WorkerConfig, run, serve


async def sample_main(argv=None):
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

    async def generate(request: dict, context: Context) -> AsyncGenerator[dict, None]:
        token_ids = request.get("token_ids", [])
        prompt_len = len(token_ids)
        max_new = (
            request.get("stop_conditions", {}).get("max_tokens") or args.max_tokens
        )

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
                return
            await asyncio.sleep(args.delay)
            token_id = (i + 1) % 32000
            out: dict = {"token_ids": [token_id]}
            if i == max_new - 1:
                out["finish_reason"] = "length"
                out["completion_usage"] = {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": max_new,
                    "total_tokens": prompt_len + max_new,
                }
            yield out

    await serve(
        worker_config=WorkerConfig(
            namespace=args.namespace,
            component=args.component,
            endpoint=args.endpoint,
            model_name=args.model_name,
            served_model_name=args.model_name,
            endpoint_types=args.endpoint_types,
            discovery_backend=args.discovery_backend,
            request_plane=args.request_plane,
            event_plane=args.event_plane,
        ),
        engine_config=EngineConfig(
            model=args.model_name,
            served_model_name=args.model_name,
            context_length=2048,
            kv_cache_block_size=16,
            total_kv_blocks=1000,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
        ),
        generate=generate,
    )


def main():
    run(sample_main)


if __name__ == "__main__":
    main()
