# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct CPU smoke client for the sample multimodal worker roles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from dynamo.runtime import DistributedRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("aggregated", "epd"), required=True)
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--model-name", default="sample-model")
    parser.add_argument("--aggregated-component", default="sample-multimodal-agg")
    parser.add_argument("--encode-component", default="sample-multimodal-encode")
    parser.add_argument("--prefill-component", default="sample-multimodal-prefill")
    parser.add_argument("--decode-component", default="sample-multimodal-decode")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


async def connect(runtime: DistributedRuntime, endpoint_name: str, timeout: float):
    endpoint = runtime.endpoint(endpoint_name)
    client = await endpoint.client()
    await asyncio.wait_for(client.wait_for_instances(), timeout=timeout)
    return client


async def collect(
    client, request: dict[str, Any], timeout: float
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        stream = await client.generate(request)
        async for response in stream:
            if response.is_error():
                comments = response.comments() or []
                raise RuntimeError("worker returned an error: " + "; ".join(comments))
            data = response.data()
            if isinstance(data, str):
                data = json.loads(data)
            if data is not None:
                chunks.append(data)
    return chunks


def multimodal_request(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "output_options": {},
        "multi_modal_data": {"image_url": [{"Url": "data:image/png;base64,AA=="}]},
        "mm_processor_kwargs": {"min_pixels": 64},
    }


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


async def run_aggregated(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    client = await connect(
        runtime,
        f"{args.namespace}.{args.aggregated_component}.generate",
        args.timeout,
    )
    request = multimodal_request(args.model_name)
    chunks = await collect(client, request, args.timeout)
    terminal = chunks[-1]
    observed = terminal["engine_data"]["sample_multimodal"]["multimodal_kwargs"]
    _require_equal(
        observed["multi_modal_data"],
        request["multi_modal_data"],
        "aggregated multi_modal_data",
    )
    _require_equal(
        observed["mm_processor_kwargs"],
        request["mm_processor_kwargs"],
        "aggregated mm_processor_kwargs",
    )


async def run_epd(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    encode, prefill, decode = await asyncio.gather(
        connect(
            runtime,
            f"{args.namespace}.{args.encode_component}.generate",
            args.timeout,
        ),
        connect(
            runtime,
            f"{args.namespace}.{args.prefill_component}.generate",
            args.timeout,
        ),
        connect(
            runtime,
            f"{args.namespace}.{args.decode_component}.generate",
            args.timeout,
        ),
    )
    request = multimodal_request(args.model_name)

    [encode_terminal] = await collect(encode, request, args.timeout)
    _require_equal(encode_terminal["token_ids"], [], "encode token_ids")
    _require_equal(encode_terminal["finish_reason"], "stop", "encode finish_reason")

    [prefill_terminal] = await collect(
        prefill,
        {**request, "encoder_result": encode_terminal["encoder_result"]},
        args.timeout,
    )
    _require_equal(
        prefill_terminal["engine_data"]["sample_multimodal"],
        encode_terminal["encoder_result"],
        "prefill encoder_result handoff",
    )
    _require_equal(
        encode_terminal["encoder_result"]["multimodal_kwargs"],
        {
            "multi_modal_data": request["multi_modal_data"],
            "mm_processor_kwargs": request["mm_processor_kwargs"],
        },
        "encode multimodal_kwargs",
    )

    decode_request = {
        "model": args.model_name,
        "token_ids": request["token_ids"],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "output_options": {},
        "prefill_result": {
            "disaggregated_params": prefill_terminal["disaggregated_params"]
        },
    }
    decode_chunks = await collect(decode, decode_request, args.timeout)
    _require_equal(decode_chunks[-1]["finish_reason"], "length", "decode finish_reason")


async def main() -> None:
    args = parse_args()
    # Launch scripts assign DYN_SYSTEM_PORT to workers. The direct client is a
    # separate runtime and must allocate its own system-server port.
    os.environ.pop("DYN_SYSTEM_PORT", None)
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")
    try:
        # Bound the whole smoke, rather than granting each sequential EPD stage
        # a fresh timeout that can exceed the outer test harness deadline.
        async with asyncio.timeout(args.timeout):
            if args.mode == "aggregated":
                await run_aggregated(runtime, args)
            else:
                await run_epd(runtime, args)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
