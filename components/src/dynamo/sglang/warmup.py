# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang Engine warmup helpers used by Dynamo workers."""

import asyncio
import inspect
import logging
from typing import Any

import sglang as sgl

logger = logging.getLogger(__name__)


class _WarmupContext:
    trace_id = "dynamo-sglang-warmup"
    span_id = None

    def id(self) -> str:
        return self.trace_id

    def is_stopped(self) -> bool:
        return False

    def async_killed_or_stopped(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()


async def _freeze_gc_after_warmup(engine: Any, *, label: str) -> None:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    freeze_gc = getattr(tokenizer_manager, "freeze_gc", None)
    if freeze_gc is None:
        freeze_gc = getattr(engine, "freeze_gc", None)
    if freeze_gc is None:
        logger.info("Skipping %s GC freeze: freeze_gc is unavailable", label)
        return

    logger.info("Freezing %s GC after warmup", label)
    result = freeze_gc()
    if inspect.isawaitable(result):
        await result
    logger.info("%s GC freeze completed", label)


_DECODE_WARMUP_TOKEN_IDS = list(range(18))


def _decode_warmup_request() -> dict[str, Any]:
    return {
        "token_ids": list(_DECODE_WARMUP_TOKEN_IDS),
        "stop_conditions": {
            "max_tokens": 8,
            "stop": [],
            "stop_token_ids": [],
            "min_tokens": 0,
            "ignore_eos": False,
        },
        "sampling_options": {
            "n": 1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "seed": None,
            "guided_decoding": None,
        },
        "output_options": {
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
            "return_tokens_as_token_ids": None,
        },
        "eos_token_ids": [],
        "annotations": [],
        "routing": None,
    }


async def warmup_generation_engine(
    engine: sgl.Engine,
    server_args,
    *,
    timeout: float = 1800,
    label: str = "decode",
) -> None:
    if server_args.disaggregation_mode != "null":
        logging.info(
            "Skipping %s warmup for disaggregation_mode=%s",
            label,
            server_args.disaggregation_mode,
        )
        return

    logging.info("Starting %s warmup request", label)
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 8,
        "ignore_eos": True,
    }

    async def _do_warmup():
        results = await engine.async_generate(
            input_ids=_DECODE_WARMUP_TOKEN_IDS,
            sampling_params=sampling_params,
            stream=True,
        )
        async for _ in results:
            pass

    await asyncio.wait_for(_do_warmup(), timeout=timeout)
    await asyncio.wait_for(
        _freeze_gc_after_warmup(engine, label=label), timeout=timeout
    )
    logging.info("%s warmup completed", label)


async def warmup_runtime_endpoint(
    endpoint: Any,
    engine: Any,
    server_args,
    *,
    timeout: float = 1800,
    label: str = "decode runtime endpoint",
) -> None:
    if server_args.disaggregation_mode != "null":
        logging.info(
            "Skipping %s warmup for disaggregation_mode=%s",
            label,
            server_args.disaggregation_mode,
        )
        return

    logging.info("Starting %s warmup request", label)

    async def _do_warmup():
        client = await endpoint.client()
        instance_id = endpoint.connection_id()
        await client.wait_for_instances()

        while instance_id not in client.instance_ids():
            await asyncio.sleep(0.05)

        stream = await client.direct(
            _decode_warmup_request(),
            instance_id,
            annotated=False,
        )
        async for _ in stream:
            pass

    await asyncio.wait_for(_do_warmup(), timeout=timeout)
    await asyncio.wait_for(
        _freeze_gc_after_warmup(engine, label=label), timeout=timeout
    )
    logging.info("%s warmup completed", label)


async def warmup_decode_handler(
    handler: Any,
    server_args,
    *,
    timeout: float = 1800,
    label: str = "decode handler",
) -> None:
    if server_args.disaggregation_mode != "null":
        logging.info(
            "Skipping %s warmup for disaggregation_mode=%s",
            label,
            server_args.disaggregation_mode,
        )
        return

    logging.info("Starting %s warmup request", label)

    async def _do_warmup():
        async for _ in handler.generate(_decode_warmup_request(), _WarmupContext()):
            pass

    await asyncio.wait_for(_do_warmup(), timeout=timeout)
    await asyncio.wait_for(
        _freeze_gc_after_warmup(handler.engine, label=label),
        timeout=timeout,
    )
    logging.info("%s warmup completed", label)
