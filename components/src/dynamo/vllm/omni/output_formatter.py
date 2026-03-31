# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modality-specific output formatters for vLLM-Omni.

Extracted from OmniHandler and AudioGenerationHandler so that any consumer
(aggregated handler, disaggregated router, test harness) can format engine
output without creating an engine or loading model weights.
"""

import time
from typing import Any, Dict

from dynamo.common.utils.engine_response import normalize_finish_reason


class TextFormatter:
    """Formats LLM text output as OpenAI chat completion chunks."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def format(
        self,
        request_output: Any,
        request_id: str,
        *,
        previous_text: str = "",
    ) -> Dict[str, Any] | None:
        if not request_output.outputs:
            return _error_chunk(request_id, self._model_name, "No outputs from engine")

        output = request_output.outputs[0]
        delta_text = output.text[len(previous_text) :]

        chunk: Dict[str, Any] = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": (
                        normalize_finish_reason(output.finish_reason)
                        if output.finish_reason
                        else None
                    ),
                }
            ],
        }

        if output.finish_reason:
            chunk["usage"] = _build_completion_usage(request_output)

        return chunk


def _error_chunk(
    request_id: str, model_name: str, error_message: str
) -> Dict[str, Any]:
    """Error response in OpenAI chat.completion.chunk format."""
    return {
        "id": request_id,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"Error: {error_message}"},
                "finish_reason": "error",
            }
        ],
    }


def _build_completion_usage(request_output: Any) -> Dict[str, Any]:
    """Build completion usage stats from a vLLM RequestOutput."""
    prompt_tokens = (
        len(request_output.prompt_token_ids)
        if getattr(request_output, "prompt_token_ids", None)
        else None
    )
    completion_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": (
            {"cached_tokens": num_cached}
            if (num_cached := getattr(request_output, "num_cached_tokens", None))
            else None
        ),
    }
