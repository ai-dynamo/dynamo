# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-specific adaptation of vLLM generation outputs."""

import base64
import io
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)

MIN_FINITE_LOGPROB = -1e30


def finite_logprob(value: Any) -> float:
    logprob = float(value)
    return logprob if math.isfinite(logprob) else MIN_FINITE_LOGPROB


def serialize_prompt_logprobs(raw_prompt_logprobs: list) -> list:
    """Convert vLLM prompt logprobs into Dynamo's transport-safe shape."""
    result: list = []
    for entry in raw_prompt_logprobs:
        if entry is None:
            result.append(None)
            continue

        converted: Dict[str, Dict[str, Any]] = {}
        for token_id, logprob_obj in entry.items():
            try:
                key = str(int(token_id))
            except (TypeError, ValueError):
                continue
            value: Dict[str, Any] = {
                "logprob": finite_logprob(logprob_obj.logprob),
            }
            rank = getattr(logprob_obj, "rank", None)
            if rank is not None:
                value["rank"] = int(rank)
            decoded = getattr(logprob_obj, "decoded_token", None)
            if decoded is not None:
                value["decoded_token"] = decoded
            converted[key] = value
        result.append(converted)
    return result


def attach_prompt_logprobs_engine_data(
    output: Dict[str, Any], prompt_logprobs: list
) -> None:
    engine_data = output.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["prompt_logprobs"] = prompt_logprobs


def attach_routed_experts_engine_data(
    output: Dict[str, Any], routed_experts: Dict[str, Any]
) -> None:
    engine_data = output.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["routed_experts"] = routed_experts


def serialize_legacy_routed_experts(
    routed_experts: Any, start: int = 0
) -> Optional[Dict[str, Any]]:
    if routed_experts is None:
        return None
    shape = getattr(routed_experts, "shape", None)
    tobytes = getattr(routed_experts, "tobytes", None)
    if shape is None or not callable(tobytes):
        logger.warning(
            "Unable to serialize routed_experts of type %s",
            type(routed_experts).__name__,
        )
        return None
    return {
        "data": base64.b64encode(tobytes()).decode("ascii"),
        "shape": [int(dim) for dim in shape],
        "start": int(start),
        "dtype": str(getattr(routed_experts, "dtype", "")),
    }


def serialize_vllm_routed_experts(routed_experts: Any) -> Optional[str]:
    if routed_experts is None:
        return None
    try:
        buffer = io.BytesIO()
        np.save(buffer, routed_experts)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        logger.warning(
            "Unable to encode routed_experts for generate API",
            exc_info=True,
        )
        return None


def build_completion_usage(
    request_output: RequestOutput,
    embedding_sequence_length: int | None = None,
    completion_token_counts: dict[int, int] | None = None,
) -> Dict[str, Any]:
    if embedding_sequence_length is not None:
        prompt_tokens = embedding_sequence_length
    elif request_output.prompt_token_ids:
        prompt_tokens = len(request_output.prompt_token_ids)
    else:
        prompt_tokens = None

    if completion_token_counts is not None:
        completion_tokens = sum(completion_token_counts.values())
    else:
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
        if prompt_tokens is not None
        else None,
        "prompt_tokens_details": (
            {"cached_tokens": num_cached}
            if (num_cached := getattr(request_output, "num_cached_tokens", None))
            else None
        ),
    }


@dataclass(frozen=True)
class GenerationResponseContext:
    request_output: RequestOutput
    output_index: int
    finished: bool
    sampling_params: Any
    embedding_sequence_length: int | None
    completion_token_counts: dict[int, int]
    prompt_logprobs: list | None
    routed_experts: Any
    kv_transfer_params: Any


class ResponseAdapter(Protocol):
    def adapt(self, output: Dict[str, Any], context: GenerationResponseContext) -> None:
        ...


class LegacyResponseAdapter:
    """Adapt the common token stream to Dynamo's legacy internal protocol."""

    def adapt(
        self,
        output: Dict[str, Any],
        context: GenerationResponseContext,
    ) -> None:
        if not context.finished:
            return
        output["completion_usage"] = build_completion_usage(
            context.request_output,
            context.embedding_sequence_length,
            context.completion_token_counts,
        )
        if context.prompt_logprobs is not None:
            attach_prompt_logprobs_engine_data(output, context.prompt_logprobs)
        raw_start = int(
            getattr(context.sampling_params, "routed_experts_prompt_start", 0) or 0
        )
        prompt_len = len(
            getattr(context.request_output, "prompt_token_ids", None) or []
        )
        serialized = serialize_legacy_routed_experts(
            context.routed_experts,
            start=min(raw_start, prompt_len),
        )
        if serialized is not None:
            attach_routed_experts_engine_data(output, serialized)


class EngineGenerateResponseAdapter:
    """Adapt the common token stream to vLLM's engine-generate contract."""

    def adapt(
        self,
        output: Dict[str, Any],
        context: GenerationResponseContext,
    ) -> None:
        output["completion_usage"] = build_completion_usage(
            context.request_output,
            context.embedding_sequence_length,
            context.completion_token_counts,
        )
        if not context.finished:
            return

        metadata: Dict[str, Any] = {}
        if context.prompt_logprobs is not None:
            metadata["prompt_logprobs"] = context.prompt_logprobs
        serialized = serialize_vllm_routed_experts(context.routed_experts)
        if serialized is not None:
            metadata["routed_experts"] = serialized
        if context.kv_transfer_params is not None:
            metadata["kv_transfer_params"] = context.kv_transfer_params
        if metadata:
            output["generate_metadata"] = metadata
