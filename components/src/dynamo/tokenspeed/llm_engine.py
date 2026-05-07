# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed LLMEngine implementation for the unified backend."""

from __future__ import annotations

import importlib
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

from dynamo._core import Context
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.args import parse_args

logger = logging.getLogger(__name__)


class TokenspeedLLMEngine(LLMEngine):
    def __init__(self, server_args: Any):
        self.server_args = server_args
        self.engine = None
        self._model_max_len: int | None = None
        self._active_rids_by_context: dict[str, list[str]] = {}

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TokenspeedLLMEngine, WorkerConfig]:
        config = parse_args(argv)
        engine = cls(config.server_args)
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        # The Dynamo response layer expects per-chunk token deltas.
        self.server_args.stream_output = True
        self.engine = _tokenspeed_engine_cls()(server_args=self.server_args)

        scheduler_info = getattr(self.engine, "scheduler_info", {}) or {}
        self._model_max_len = _optional_int(
            scheduler_info.get("max_model_len")
            or getattr(
                getattr(self.engine, "tokenizer_manager", None), "context_len", None
            )
            or getattr(self.server_args, "max_model_len", None)
        )

        block_size = _optional_int(getattr(self.server_args, "block_size", None))
        max_total_tokens = _optional_int(
            scheduler_info.get("max_total_num_tokens")
            or getattr(self.server_args, "max_total_tokens", None)
        )
        total_kv_blocks = (
            (max_total_tokens + block_size - 1) // block_size
            if max_total_tokens is not None and block_size
            else None
        )

        max_num_batched_tokens = _optional_int(
            scheduler_info.get("chunked_prefill_size")
            or getattr(self.server_args, "chunked_prefill_size", None)
            or getattr(self.server_args, "max_prefill_tokens", None)
        )

        return EngineConfig(
            model=self.server_args.model,
            served_model_name=self.server_args.served_model_name,
            context_length=self._model_max_len,
            kv_cache_block_size=block_size,
            total_kv_blocks=total_kv_blocks,
            max_num_seqs=_optional_int(
                scheduler_info.get("max_num_seqs")
                or getattr(self.server_args, "max_num_seqs", None)
            ),
            max_num_batched_tokens=max_num_batched_tokens,
        )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None, "Engine not initialized"

        _validate_single_choice_sampling(request)
        sampling_params = build_sampling_params(request, self._model_max_len)
        token_ids = request.get("token_ids", [])
        obj = _generate_req_input_cls()(
            input_ids=token_ids,
            sampling_params=sampling_params,
            stream=True,
        )

        request_id = context.id()
        if request_id is not None:
            obj.rid = request_id
            self._active_rids_by_context[request_id] = [request_id]

        emitted_completion_tokens = 0
        try:
            async for out in self.engine.tokenizer_manager.generate_request(obj):
                delta_out, emitted_completion_tokens = _completion_delta_output(
                    out, emitted_completion_tokens
                )
                yield convert_output_to_chunk(delta_out)
        finally:
            if request_id is not None:
                self._active_rids_by_context.pop(request_id, None)

    async def abort(self, context: Context) -> None:
        request_id = context.id()
        if self.engine is None or request_id is None:
            return

        rids = self._active_rids_by_context.get(request_id, [request_id])
        for rid in rids:
            self.engine.tokenizer_manager.abort_request(rid)
            logger.debug("Aborted TokenSpeed request %s", rid)

    async def cleanup(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            logger.info("TokenSpeed engine shutdown")


def build_sampling_params(
    request: GenerateRequest,
    model_max_len: int | None = None,
) -> dict[str, Any]:
    sampling_options = request.get("sampling_options", {}) or {}
    stop_conditions = request.get("stop_conditions", {}) or {}

    params: dict[str, Any] = {}

    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "logit_bias",
        "n",
    ):
        value = sampling_options.get(key)
        if value is not None:
            params[key] = value

    guided_decoding = sampling_options.get("guided_decoding")
    if isinstance(guided_decoding, dict):
        params.update(_guided_decoding_params(guided_decoding))

    max_tokens = stop_conditions.get("max_tokens")
    if max_tokens is not None:
        params["max_new_tokens"] = max_tokens
    elif model_max_len is not None:
        params["max_new_tokens"] = max(
            1, model_max_len - len(request.get("token_ids", []))
        )

    min_tokens = stop_conditions.get("min_tokens")
    if min_tokens is not None:
        params["min_new_tokens"] = min_tokens

    ignore_eos = stop_conditions.get("ignore_eos")
    if ignore_eos is not None:
        params["ignore_eos"] = ignore_eos

    stop_token_ids = _merge_stop_token_ids(
        stop_conditions.get("stop_token_ids_hidden"),
        stop_conditions.get("stop_token_ids"),
    )
    if stop_token_ids:
        params["stop_token_ids"] = stop_token_ids

    return params


def convert_output_to_chunk(out: dict[str, Any]) -> GenerateChunk:
    meta_info = out.get("meta_info", {}) or {}
    output_idx = out.get("index") or 0
    chunk: GenerateChunk = {
        "index": output_idx,
        "token_ids": out.get("output_ids", []) or [],
    }

    finish_reason = meta_info.get("finish_reason")
    if finish_reason is not None:
        chunk["finish_reason"] = normalize_finish_reason(
            _finish_reason_type(finish_reason)
        )
        prompt_tokens = int(meta_info.get("prompt_tokens") or 0)
        completion_tokens = int(meta_info.get("completion_tokens") or 0)
        chunk["completion_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    return chunk


def _completion_delta_output(
    out: dict[str, Any],
    previously_emitted: int,
) -> tuple[dict[str, Any], int]:
    meta_info = out.get("meta_info", {}) or {}
    completion_tokens = meta_info.get("completion_tokens")
    if completion_tokens is None:
        return out, previously_emitted

    try:
        total_emitted = int(completion_tokens)
    except (TypeError, ValueError):
        return out, previously_emitted

    delta_count = max(0, total_emitted - previously_emitted)
    output_ids = out.get("output_ids", []) or []
    if delta_count == 0:
        delta_ids: list[int] = []
    elif len(output_ids) >= delta_count:
        # TokenSpeed's first streamed output can include echoed prompt/context
        # tokens even though meta_info.completion_tokens only counts newly
        # generated tokens. Dynamo expects token deltas, so keep the newest
        # completion-token suffix.
        delta_ids = output_ids[-delta_count:]
    else:
        delta_ids = output_ids

    delta_out = dict(out)
    delta_out["output_ids"] = delta_ids
    return delta_out, total_emitted


def _guided_decoding_params(guided_decoding: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_schema = guided_decoding.get("json")
    regex = guided_decoding.get("regex")
    choice = guided_decoding.get("choice")
    grammar = guided_decoding.get("grammar")
    structural_tag = guided_decoding.get("structural_tag")

    if regex is None and choice:
        valid_choices = [str(c) for c in choice if c is not None]
        if valid_choices:
            regex = "(" + "|".join(re.escape(c) for c in valid_choices) + ")"

    constraints = {
        "json": json_schema,
        "regex": regex,
        "grammar": grammar,
        "structural_tag": structural_tag,
    }
    active_constraints = [
        name for name, value in constraints.items() if value is not None
    ]
    if len(active_constraints) > 1:
        raise InvalidArgument(
            "TokenSpeed guided decoding supports one constraint at a time; "
            f"got {', '.join(active_constraints)}"
        )

    if json_schema is not None:
        params["json_schema"] = (
            json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
        )

    if regex is not None:
        params["regex"] = regex

    if grammar is not None:
        params["ebnf"] = grammar

    if structural_tag is not None:
        if hasattr(structural_tag, "model_dump"):
            structural_tag = structural_tag.model_dump()
        params["structural_tag"] = (
            structural_tag
            if isinstance(structural_tag, str)
            else json.dumps(structural_tag)
        )

    return params


def _finish_reason_type(finish_reason: Any) -> str:
    if hasattr(finish_reason, "to_json"):
        finish_reason = finish_reason.to_json()
    if isinstance(finish_reason, dict):
        return str(finish_reason.get("type") or "unknown")
    return str(finish_reason)


def _merge_stop_token_ids(*token_id_lists: Any) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for token_ids in token_id_lists:
        for token_id in token_ids or []:
            if token_id not in seen:
                seen.add(token_id)
                merged.append(token_id)
    return merged


def _validate_single_choice_sampling(request: GenerateRequest) -> None:
    sampling_options = request.get("sampling_options", {}) or {}
    n = sampling_options.get("n", 1)
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        raise InvalidArgument(
            f"TokenSpeed Dynamo backend does not support n={n}; only n=1 is supported"
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _tokenspeed_engine_cls() -> Any:
    module = importlib.import_module("tokenspeed.runtime.entrypoints.engine")
    return module.Engine


def _generate_req_input_cls() -> Any:
    module = importlib.import_module("tokenspeed.runtime.engine.io_struct")
    return module.GenerateReqInput
