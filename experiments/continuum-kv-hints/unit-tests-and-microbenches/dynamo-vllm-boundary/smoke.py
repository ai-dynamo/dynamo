# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Dynamo-to-vLLM KV hints against a local vLLM engine."""

import argparse
import asyncio
import json
from contextlib import closing
from types import SimpleNamespace
from typing import Any

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.vllm.handlers import DecodeWorkerHandler, _build_vllm_kv_hints


def raw_hint(case: str) -> dict[str, Any] | None:
    if case == "baseline":
        return None
    payload: dict[str, Any] = {
        "block_hashes": [],
        "execute_at": "request_completion",
        "include_current_request": True,
    }
    if case == "retain":
        payload.update(priority=10, ttl_seconds=300)
    return {
        "protocol_version": "0.1",
        "message_id": f"dynamo-{case}-message",
        "actions": [
            {
                "action_id": f"dynamo-{case}-action",
                "action_type": f"kv.{case}",
                "action_version": "1.0",
                "payload": payload,
            }
        ],
    }


def convert_hint(case: str) -> tuple[Any | None, bool]:
    raw = raw_hint(case)
    if raw is None:
        return None, True

    envelope = _build_vllm_kv_hints({"kv_hint": raw})
    action = envelope.actions[0]
    preserved = (
        envelope.protocol_version == raw["protocol_version"]
        and envelope.message_id == raw["message_id"]
        and action.action_id == raw["actions"][0]["action_id"]
        and action.action_type == raw["actions"][0]["action_type"]
        and action.action_version == raw["actions"][0]["action_version"]
        and action.payload == raw["actions"][0]["payload"]
    )
    return envelope, preserved


def make_handler(engine: AsyncLLM) -> DecodeWorkerHandler:
    handler = object.__new__(DecodeWorkerHandler)
    handler.engine_client = engine
    handler.runtime = SimpleNamespace(shutdown=lambda: None)
    return handler


async def generate(
    handler: DecodeWorkerHandler,
    request_id: str,
    token_ids: list[int],
    kv_hints: Any | None = None,
) -> tuple[int, list[int]]:
    final = None
    async for chunk in handler.generate_tokens(
        TokensPrompt(prompt_token_ids=token_ids),
        SamplingParams(
            max_tokens=1,
            temperature=0,
            ignore_eos=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        ),
        request_id,
        session_id="dynamo-smoke-session",
        kv_hints=kv_hints,
    ):
        final = chunk

    assert final is not None
    usage = final["completion_usage"]
    return usage["prompt_tokens_details"]["cached_tokens"], final["token_ids"]


async def run_case(args: argparse.Namespace) -> dict[str, Any]:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            enforce_eager=True,
            enable_prefix_caching=True,
            num_gpu_blocks_override=args.num_gpu_blocks,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
        )
    )
    handler = make_handler(engine)
    target = [args.target_token_id] * args.prompt_tokens
    hint, envelope_preserved = convert_hint(args.case)
    assert envelope_preserved

    try:
        warm_cached, warm_output = await generate(
            handler,
            f"{args.case}-target-warm",
            target,
            hint,
        )
        pressure_cached = []
        if args.case != "evict":
            for index in range(args.pressure_requests):
                cached, _ = await generate(
                    handler,
                    f"{args.case}-pressure-{index}",
                    [args.pressure_token_id + index] * args.prompt_tokens,
                )
                pressure_cached.append(cached)
        replay_cached, replay_output = await generate(
            handler,
            f"{args.case}-target-replay",
            target,
        )
        return {
            "case": args.case,
            "envelope_preserved": envelope_preserved,
            "warm_cached_tokens": warm_cached,
            "pressure_cached_tokens": pressure_cached,
            "replay_cached_tokens": replay_cached,
            "outputs_match": warm_output == replay_output,
        }
    finally:
        engine.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("baseline", "evict", "retain"), required=True
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-gpu-blocks", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--pressure-requests", type=int, default=6)
    parser.add_argument("--target-token-id", type=int, default=1000)
    parser.add_argument("--pressure-token-id", type=int, default=2000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    with closing(asyncio.Runner()) as runner:
        print(json.dumps(runner.run(run_case(parse_args())), sort_keys=True))
