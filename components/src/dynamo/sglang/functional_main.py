# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional entry point for the SGLang backend.

Usage:
    python -m dynamo.sglang.functional_main <sglang args>
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, Dict

import sglang as sgl

from dynamo._core import Context
from dynamo.common.backend.serve import EngineConfig, WorkerConfig, run, serve
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import ModelInput
from dynamo.sglang.args import parse_args

logger = logging.getLogger(__name__)


async def sglang_main(argv=None):
    # -- Parse args --
    config = await parse_args(argv if argv is not None else sys.argv[1:])
    server_args = config.server_args
    dynamo_args = config.dynamo_args

    skip_tokenizer_init = server_args.skip_tokenizer_init
    model_input = ModelInput.Text if not skip_tokenizer_init else ModelInput.Tokens

    worker_config = WorkerConfig.from_runtime_config(
        dynamo_args,
        model_name=server_args.model_path,
        served_model_name=server_args.served_model_name,
        model_input=model_input,
    )

    # -- Start engine --
    engine = sgl.Engine(server_args=server_args)

    tokenizer = engine.tokenizer_manager.tokenizer if not skip_tokenizer_init else None
    input_param_manager = InputParamManager(tokenizer)

    # -- Helpers --
    def _build_sampling_params(request: dict) -> dict:
        if skip_tokenizer_init:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}

    def _get_input_param(request: dict) -> dict:
        request_input = input_param_manager.get_input_param(
            request, use_tokenizer=not skip_tokenizer_init
        )
        return {
            "prompt" if isinstance(request_input, str) else "input_ids": request_input
        }

    # -- Callbacks --
    async def generate(request: dict, context: Context) -> AsyncGenerator[dict, None]:
        sampling_params = _build_sampling_params(request)
        input_param = _get_input_param(request)

        stream = await engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            rid=context.trace_id,
        )

        async for res in stream:
            out: Dict[str, Any] = {}
            meta_info = res["meta_info"]
            finish_reason = meta_info["finish_reason"]

            output_ids = res.get("output_ids", [])
            if not output_ids and not finish_reason:
                if context.is_stopped():
                    prompt_tokens = meta_info.get("prompt_tokens", 0)
                    completion_tokens = meta_info.get("completion_tokens", 0)
                    yield {
                        "token_ids": [],
                        "finish_reason": "cancelled",
                        "completion_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    break
                continue

            out["token_ids"] = output_ids

            if finish_reason:
                prompt_tokens = meta_info["prompt_tokens"]
                completion_tokens = meta_info["completion_tokens"]
                out["finish_reason"] = finish_reason["type"]
                out["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            if context.is_stopped():
                prompt_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                yield {
                    "token_ids": output_ids,
                    "finish_reason": "cancelled",
                    "completion_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                break

            yield out

    async def abort(context: Context) -> None:
        rid = context.trace_id
        if engine is not None and rid is not None:
            if hasattr(engine, "tokenizer_manager") and engine.tokenizer_manager:
                engine.tokenizer_manager.abort_request(rid=rid, abort_all=False)
                logger.debug("Aborted request %s", rid)

    async def cleanup() -> None:
        if engine is not None:
            engine.shutdown()
            logger.info("SGLang engine shutdown")

    # -- Serve --
    await serve(
        worker_config=worker_config,
        engine_config=EngineConfig(
            model=server_args.model_path,
            served_model_name=server_args.served_model_name,
            context_length=server_args.context_length,
            kv_cache_block_size=server_args.page_size,
        ),
        generate=generate,
        abort=abort,
        cleanup=cleanup,
    )


def main():
    run(sglang_main)


if __name__ == "__main__":
    main()
