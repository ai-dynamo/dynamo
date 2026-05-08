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
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.llm import ModelInput, ModelType
from dynamo.llm.exceptions import InvalidArgument
from dynamo.tokenspeed.args import (
    kv_events_config_dict,
    kv_events_enabled,
    parse_args,
)
from dynamo.tokenspeed.disagg import (
    runtime_disaggregated_endpoint,
    validate_disagg_compatibility,
)

logger = logging.getLogger(__name__)


class TokenspeedLLMEngine(LLMEngine):
    def __init__(
        self,
        server_args: Any,
        dynamo_config: Any | None = None,
        disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED,
    ):
        self.server_args = server_args
        self.dynamo_config = dynamo_config
        self.disaggregation_mode = disaggregation_mode
        self.engine = None
        self._model_max_len: int | None = None
        self._active_rids_by_context: dict[str, list[str]] = {}
        self._kv_publishers: list[Any] = []

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[TokenspeedLLMEngine, WorkerConfig]:
        config = parse_args(argv)
        engine = cls(
            config.server_args,
            dynamo_config=config,
            disaggregation_mode=config.disaggregation_mode,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self) -> EngineConfig:
        validate_disagg_compatibility(self.disaggregation_mode, self.server_args)

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

        disagg_kwargs: dict[str, Any] = {}
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Prefill workers register on the prefill router and advertise the
            # Mooncake bootstrap endpoint so the router can synthesize
            # disaggregated_state into each decode request.
            disagg_kwargs["model_type"] = ModelType.Prefill
            disagg_kwargs["disaggregated_endpoint"] = runtime_disaggregated_endpoint(
                self.server_args
            )
            # Prefill workers ship KV out-of-band; a local prefix-match indexer
            # would just track blocks that are already gone.
            disagg_kwargs["enable_local_indexer"] = False

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
            **disagg_kwargs,
        )

    async def start_kv_events(self, endpoint: Any, engine_config: EngineConfig) -> None:
        if not kv_events_enabled(getattr(self.server_args, "kv_events_config", None)):
            return

        if not getattr(self.server_args, "enable_prefix_caching", True):
            logger.info(
                "TokenSpeed KV event publishing skipped: prefix caching disabled"
            )
            return

        _assert_tokenspeed_kv_events_supported()

        kv_block_size = engine_config.kv_cache_block_size or _optional_int(
            getattr(self.server_args, "block_size", None)
        )
        if not kv_block_size:
            raise RuntimeError("TokenSpeed KV events require a non-zero block size")

        config = kv_events_config_dict(self.server_args.kv_events_config)
        base_zmq_endpoint = config.get("endpoint") or "tcp://*:5557"
        publisher_cls = _kv_event_publisher_cls()

        for dp_rank in _local_dp_rank_range(self.server_args):
            zmq_endpoint = _format_zmq_connect_endpoint(
                _offset_zmq_endpoint_port(base_zmq_endpoint, dp_rank)
            )
            logger.info(
                "TokenSpeed KV event publisher for dp_rank=%s subscribing to %s",
                dp_rank,
                zmq_endpoint,
            )
            self._kv_publishers.append(
                publisher_cls(
                    endpoint=endpoint,
                    kv_block_size=kv_block_size,
                    zmq_endpoint=zmq_endpoint,
                    zmq_topic="",
                    enable_local_indexer=getattr(
                        self.dynamo_config, "enable_local_indexer", False
                    ),
                    dp_rank=dp_rank,
                )
            )

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        assert self.engine is not None, "Engine not initialized"

        _validate_single_choice_sampling(request)
        sampling_params = build_sampling_params(request, self._model_max_len)
        token_ids = request.get("token_ids", [])

        bootstrap_kwargs = _bootstrap_kwargs_for_request(
            self.disaggregation_mode, request
        )
        obj = _generate_req_input_cls()(
            input_ids=token_ids,
            sampling_params=sampling_params,
            stream=True,
            **bootstrap_kwargs,
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
        for publisher in self._kv_publishers:
            try:
                publisher.shutdown()
            except Exception:
                logger.warning(
                    "Failed to shut down TokenSpeed KV publisher", exc_info=True
                )
        self._kv_publishers.clear()
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


_BOOTSTRAP_KEYS = ("bootstrap_host", "bootstrap_port", "bootstrap_room")


def _bootstrap_kwargs_for_request(
    mode: DisaggregationMode, request: GenerateRequest
) -> dict[str, Any]:
    """Extract Mooncake bootstrap kwargs for ``GenerateReqInput``.

    Both prefill and decode TokenSpeed workers receive the same
    ``{bootstrap_host, bootstrap_port, bootstrap_room}`` triple from
    Dynamo's ``PrefillRouter`` (router-resolved bootstrap mode). The room
    is per-request; the host/port identify the prefill side's Mooncake
    server. Aggregated workers don't see disaggregated_state at all.
    """
    if mode == DisaggregationMode.AGGREGATED:
        return {}

    # Read the canonical disaggregated_state field first; fall back to the
    # legacy SGLang wire key bootstrap_info that Dynamo's Rust PrefillRouter
    # writes today. Lets this PR work end-to-end before the parallel
    # Rust-side normalization to disaggregated_state lands.
    state = request.get("disaggregated_state") or request.get("bootstrap_info") or {}  # type: ignore[typeddict-item]
    if not state:
        raise InvalidArgument(
            f"TokenSpeed worker in disaggregation_mode={mode.value} requires "
            "disaggregated_state (or legacy bootstrap_info) on the request "
            "(bootstrap_host/bootstrap_port/bootstrap_room)"
        )

    missing = [key for key in _BOOTSTRAP_KEYS if state.get(key) is None]
    if missing:
        raise InvalidArgument(
            f"TokenSpeed worker missing bootstrap fields {missing} in "
            f"disaggregated_state; got keys={sorted(state.keys())}"
        )

    return {key: state[key] for key in _BOOTSTRAP_KEYS}


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


def _kv_event_publisher_cls() -> Any:
    from dynamo.llm import KvEventPublisher

    return KvEventPublisher


def _assert_tokenspeed_kv_events_supported() -> None:
    try:
        module = importlib.import_module("tokenspeed.runtime.pd.kv_events")
    except ImportError as exc:
        raise RuntimeError(
            "TokenSpeed KV events require tokenspeed.runtime.pd.kv_events"
        ) from exc

    factory = getattr(module, "EventPublisherFactory", None)
    config_cls = getattr(module, "KVEventsConfig", None)
    fields = getattr(config_cls, "model_fields", {}) if config_cls is not None else {}
    if not hasattr(factory, "is_enabled") or "enable_kv_cache_events" not in fields:
        raise RuntimeError(
            "TokenSpeed KV events require a TokenSpeed build with "
            "KVEventsConfig.enable_kv_cache_events and "
            "EventPublisherFactory.is_enabled support"
        )


def _local_dp_rank_range(server_args: Any) -> range:
    mapping = getattr(server_args, "mapping", None)
    attn_mapping = getattr(mapping, "attn", None)
    dp_size = (
        _optional_int(getattr(attn_mapping, "dp_size", None))
        or _optional_int(getattr(server_args, "data_parallel_size", None))
        or 1
    )
    if dp_size <= 1:
        return range(0, 1)

    nnodes = _optional_int(getattr(mapping, "nnodes", None)) or _optional_int(
        getattr(server_args, "nnodes", None)
    )
    nnodes = nnodes or 1
    node_rank = _optional_int(getattr(server_args, "node_rank", None)) or 0

    dp_ranks_per_node = dp_size // nnodes
    start = node_rank * dp_ranks_per_node
    end = min(dp_size, start + dp_ranks_per_node)
    return range(start, end)


def _offset_zmq_endpoint_port(endpoint: str | None, dp_rank: int) -> str | None:
    if not endpoint or dp_rank == 0:
        return endpoint

    if "inproc" in endpoint:
        return f"{endpoint}_dp{dp_rank}"
    if "tcp" in endpoint:
        last_colon_idx = endpoint.rfind(":")
        if last_colon_idx < 0:
            return endpoint
        base_addr = endpoint[:last_colon_idx]
        base_port = int(endpoint[last_colon_idx + 1 :])
        return f"{base_addr}:{base_port + dp_rank}"
    raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


def _format_zmq_connect_endpoint(endpoint: str | None) -> str:
    if not endpoint:
        raise ValueError("TokenSpeed kv_events_config is missing an endpoint")
    return endpoint.replace("*", "127.0.0.1")
