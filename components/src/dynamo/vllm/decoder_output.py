# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared conversion from native vLLM outputs to Dynamo token chunks."""

from __future__ import annotations

import base64
import logging
import math
from collections.abc import Callable
from typing import Any, Dict, Optional

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from dynamo.common.utils.engine_response import normalize_finish_reason

logger = logging.getLogger(__name__)

# Logprobs can be -inf (log of probability 0) for masked/disallowed tokens.
# JSON has no inf/nan, so clamp non-finite values before Rust deserialization.
_MIN_FINITE_LOGPROB = -1e30


def _finite_logprob(value: Any) -> float:
    lp = float(value)
    return lp if math.isfinite(lp) else _MIN_FINITE_LOGPROB


def _serialize_prompt_logprobs(raw_prompt_logprobs: list) -> list:
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
            lp_dict: Dict[str, Any] = {
                "logprob": _finite_logprob(logprob_obj.logprob),
            }
            rank = getattr(logprob_obj, "rank", None)
            if rank is not None:
                lp_dict["rank"] = int(rank)
            decoded = getattr(logprob_obj, "decoded_token", None)
            if decoded is not None:
                lp_dict["decoded_token"] = decoded
            converted[key] = lp_dict
        result.append(converted)
    return result


def _attach_prompt_logprobs_engine_data(
    chunk: Dict[str, Any], prompt_logprobs: list
) -> None:
    engine_data = chunk.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["prompt_logprobs"] = prompt_logprobs


def _serialize_routed_experts(
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


def _attach_routed_experts_engine_data(
    chunk: Dict[str, Any], routed_experts: Dict[str, Any]
) -> None:
    engine_data = chunk.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["routed_experts"] = routed_experts


def build_prompt_tokens_details(
    num_cached_tokens: int | None,
) -> dict[str, int] | None:
    """Preserve the distinction between unavailable and zero cached tokens."""

    if num_cached_tokens is None:
        return None
    return {"cached_tokens": num_cached_tokens}


def build_completion_usage(
    request_output: RequestOutput,
    completion_token_counts: dict[int, int] | None = None,
) -> Dict[str, Any]:
    """Build Dynamo completion usage from a native vLLM output."""

    prompt_tokens = (
        len(request_output.prompt_token_ids)
        if request_output.prompt_token_ids
        else None
    )
    if completion_token_counts is not None:
        completion_tokens = sum(completion_token_counts.values())
    else:
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": build_prompt_tokens_details(
            getattr(request_output, "num_cached_tokens", None)
        ),
    }


ExtractLogprobs = Callable[
    [Any, int, Any], tuple[list[float] | None, list[list[dict]] | None]
]


class _VllmDecodeOutputAdapter:
    """Statefully convert one native vLLM request stream into Dynamo chunks."""

    def __init__(
        self,
        sampling_params: SamplingParams,
        *,
        tokenizer: Any,
        extract_logprobs: ExtractLogprobs,
    ) -> None:
        self._sampling_params = sampling_params
        self._tokenizer = tokenizer
        self._extract_logprobs = extract_logprobs
        self._completion_tokens_by_index: dict[int, int] = {}
        self._routed_experts_by_index: dict[int, Any] = {}
        self._prompt_logprobs: list | None = None

    def completion_tokens(self, output_index: int) -> int:
        return self._completion_tokens_by_index.get(output_index, 0)

    def convert(self, response: RequestOutput) -> list[Dict[str, Any]]:
        """Convert one native response, retaining state needed by final chunks."""

        if (
            self._prompt_logprobs is None
            and getattr(response, "prompt_logprobs", None) is not None
        ):
            self._prompt_logprobs = _serialize_prompt_logprobs(response.prompt_logprobs)

        if not response.outputs:
            return [
                {
                    "finish_reason": "error: No outputs from vLLM engine",
                    "index": 0,
                    "token_ids": [],
                }
            ]

        # Account for the complete native response before emitting any of its
        # chunks. A single response can finish one output index while advancing
        # another, and completion usage is request-wide rather than per-index.
        for output in response.outputs:
            output_index = getattr(output, "index", 0) or 0
            self._completion_tokens_by_index[
                output_index
            ] = self._completion_tokens_by_index.get(output_index, 0) + len(
                output.token_ids or []
            )

        chunks: list[Dict[str, Any]] = []
        for output in response.outputs:
            output_index = getattr(output, "index", 0) or 0
            token_ids = list(output.token_ids or [])
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
            if not token_ids and not finish_reason and not stop_reason:
                continue

            chunk: Dict[str, Any] = {
                "index": output_index,
                "token_ids": token_ids,
            }
            routed_experts = getattr(output, "routed_experts", None)
            if routed_experts is not None:
                self._routed_experts_by_index[output_index] = routed_experts

            log_probs, top_logprobs = self._extract_logprobs(output, 0, self._tokenizer)
            if log_probs is not None:
                chunk["log_probs"] = log_probs
            if top_logprobs is not None:
                chunk["top_logprobs"] = top_logprobs

            if finish_reason:
                chunk["finish_reason"] = normalize_finish_reason(finish_reason)
                chunk["completion_usage"] = build_completion_usage(
                    response,
                    completion_token_counts=self._completion_tokens_by_index,
                )
                if self._prompt_logprobs is not None:
                    _attach_prompt_logprobs_engine_data(chunk, self._prompt_logprobs)
                raw_start = int(
                    getattr(
                        self._sampling_params,
                        "routed_experts_prompt_start",
                        0,
                    )
                    or 0
                )
                prompt_length = len(getattr(response, "prompt_token_ids", None) or [])
                serialized_routed_experts = _serialize_routed_experts(
                    self._routed_experts_by_index.get(output_index),
                    start=min(raw_start, prompt_length),
                )
                if serialized_routed_experts is not None:
                    _attach_routed_experts_engine_data(chunk, serialized_routed_experts)
            if stop_reason:
                chunk["stop_reason"] = stop_reason
            chunks.append(chunk)

        return chunks
