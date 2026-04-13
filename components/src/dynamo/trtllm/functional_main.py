# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional entry point for the TensorRT-LLM backend.

Usage:
    python -m dynamo.trtllm.functional_main <trtllm args>
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.sampling_params import GuidedDecodingParams
from torch.cuda import device_count

from dynamo._core import Context
from dynamo.common.backend.serve import EngineConfig, WorkerConfig, run, serve
from dynamo.llm import ModelInput
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine

logger = logging.getLogger(__name__)


def _override_sampling_params(
    sampling_params: SamplingParams, request: dict
) -> SamplingParams:
    """Apply request-level sampling overrides to a base SamplingParams."""
    overrides = {
        key: value
        for key, value in request["sampling_options"].items()
        if value is not None
    }

    guided_decoding = overrides.pop("guided_decoding", None)
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        regex = guided_decoding.get("regex")
        choice = guided_decoding.get("choice")
        if choice and not regex:
            valid_choices = [c for c in choice if c is not None]
            if valid_choices:
                regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"
        overrides["guided_decoding"] = GuidedDecodingParams(
            json=guided_decoding.get("json"),
            regex=regex,
            grammar=guided_decoding.get("grammar"),
            json_object=guided_decoding.get("json_object", False),
            structural_tag=guided_decoding.get("structural_tag"),
        )

    return dataclasses.replace(sampling_params, **overrides)


async def trtllm_main(argv=None):
    # -- Parse args --
    config = parse_args(argv)

    gpus_per_node = config.gpus_per_node or device_count()

    engine_args = {
        "model": str(config.model),
        "scheduler_config": SchedulerConfig(),
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "backend": Backend.PYTORCH,
        "kv_cache_config": KvCacheConfig(
            free_gpu_memory_fraction=config.free_gpu_memory_fraction,
        ),
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
    }

    worker_config = WorkerConfig.from_runtime_config(
        config,
        model_name=config.model,
        served_model_name=config.served_model_name,
        model_input=ModelInput.Tokens,
    )

    # -- Start engine --
    trtllm_engine = TensorRTLLMEngine(engine_args)
    await trtllm_engine.initialize()

    max_seq_len = config.max_seq_len
    max_batch_size = config.max_batch_size
    max_num_tokens = config.max_num_tokens
    kv_block_size = config.kv_block_size
    default_sampling_params = SamplingParams(detokenize=False)
    active_requests: dict[str, Any] = {}

    # -- Callbacks --
    async def generate(request: dict, context: Context) -> AsyncGenerator[dict, None]:
        token_ids = request.get("token_ids", [])
        sampling_params = _override_sampling_params(default_sampling_params, request)

        max_tokens = request["stop_conditions"].get("max_tokens")
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        elif max_seq_len is not None:
            sampling_params.max_tokens = max(1, max_seq_len - len(token_ids))

        ignore_eos = request["stop_conditions"].get("ignore_eos")
        if ignore_eos:
            sampling_params.ignore_eos = ignore_eos

        generation_result = trtllm_engine.llm.generate_async(
            inputs=token_ids,
            sampling_params=sampling_params,
            streaming=True,
        )

        request_id = context.id()
        if request_id is not None:
            active_requests[request_id] = generation_result

        try:
            num_output_tokens_so_far = 0
            async for res in generation_result:
                if not res.outputs and not res.finished:
                    yield {"finish_reason": "error", "token_ids": []}
                    break

                output = res.outputs[0]
                next_total = len(output.token_ids)
                out: dict = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                if output.finish_reason:
                    out["finish_reason"] = str(output.finish_reason)

                if out.get("finish_reason") or res.finished:
                    if not out.get("finish_reason"):
                        out["finish_reason"] = "unknown"
                    prompt_tokens = len(token_ids)
                    out["completion_usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": next_total,
                        "total_tokens": prompt_tokens + next_total,
                    }

                yield out
                num_output_tokens_so_far = next_total
        finally:
            if request_id is not None:
                active_requests.pop(request_id, None)

    async def abort(context: Context) -> None:
        request_id = context.id()
        if request_id is not None:
            generation_result = active_requests.get(request_id)
            if generation_result is not None:
                generation_result.abort()
                logger.debug("Aborted request %s", request_id)

    async def cleanup() -> None:
        await trtllm_engine.cleanup()
        logger.info("TensorRT-LLM engine shutdown")

    # -- Serve --
    await serve(
        worker_config=worker_config,
        engine_config=EngineConfig(
            model=config.model,
            served_model_name=config.served_model_name,
            context_length=max_seq_len,
            kv_cache_block_size=kv_block_size,
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=max_num_tokens,
        ),
        generate=generate,
        abort=abort,
        cleanup=cleanup,
    )


def main():
    run(trtllm_main)


if __name__ == "__main__":
    main()
