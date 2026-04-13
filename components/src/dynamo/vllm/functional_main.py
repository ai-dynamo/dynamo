# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional entry point for the vLLM backend.

Usage:
    python -m dynamo.vllm.functional_main <vllm args>
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import AsyncGenerator

from vllm.inputs import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo._core import Context
from dynamo.common.backend.serve import EngineConfig, WorkerConfig, run, serve
from dynamo.llm import ModelInput
from dynamo.vllm.args import parse_args

from .handlers import build_sampling_params

logger = logging.getLogger(__name__)


async def vllm_main(argv=None):
    # -- Parse args --
    config = parse_args()  # TODO: forward argv when vllm supports it
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    worker_config = WorkerConfig.from_runtime_config(
        config,
        model_name=config.model,
        served_model_name=config.served_model_name,
        model_input=ModelInput.Tokens,
    )

    # -- Start engine --
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    prometheus_temp_dir = None
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prometheus_temp_dir = tempfile.TemporaryDirectory(prefix="vllm_prometheus_")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_temp_dir.name

    default_sampling_params = (
        config.engine_args.create_model_config().get_diff_sampling_param()
    )
    vllm_config = config.engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    model_max_len = getattr(
        getattr(vllm_config, "model_config", None), "max_model_len", None
    )
    num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
    block_size = vllm_config.cache_config.block_size

    # -- Callbacks --
    async def generate(request: dict, context: Context) -> AsyncGenerator[dict, None]:
        token_ids = request.get("token_ids", [])
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        sampling_params = build_sampling_params(
            request, default_sampling_params, model_max_len
        )

        num_so_far = 0
        async for res in engine_client.generate(prompt, sampling_params, context.id()):
            if not res.outputs:
                yield {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "token_ids": [],
                }
                break

            output = res.outputs[0]
            next_total = len(output.token_ids)
            out: dict = {"token_ids": output.token_ids[num_so_far:]}

            if output.finish_reason:
                out["finish_reason"] = str(output.finish_reason)
                prompt_tokens = len(res.prompt_token_ids) if res.prompt_token_ids else 0
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": next_total,
                    "total_tokens": prompt_tokens + next_total,
                }

            yield out
            num_so_far = next_total

    async def abort(context: Context) -> None:
        request_id = context.id()
        if engine_client is not None and request_id is not None:
            await engine_client.abort(request_id)
            logger.debug("Aborted request %s", request_id)

    async def cleanup() -> None:
        engine_client.shutdown()
        if prometheus_temp_dir is not None:
            prometheus_temp_dir.cleanup()
        logger.info("vLLM engine shutdown")

    # -- Serve --
    await serve(
        worker_config=worker_config,
        engine_config=EngineConfig(
            model=config.engine_args.model,
            served_model_name=config.engine_args.served_model_name,
            context_length=model_max_len,
            kv_cache_block_size=block_size,
            total_kv_blocks=num_gpu_blocks,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        ),
        generate=generate,
        abort=abort,
        cleanup=cleanup,
    )


def main():
    run(vllm_main)


if __name__ == "__main__":
    main()
