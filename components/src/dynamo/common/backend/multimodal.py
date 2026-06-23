# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal helpers shared across unified-backend engines."""

from __future__ import annotations

from typing import Any, Optional

from dynamo.common.backend.engine import GenerateChunk, GenerateRequest
from dynamo.common.constants import DisaggregationMode


def extract_multimodal_inputs(
    request: GenerateRequest,
) -> Optional[dict[str, Any]]:
    """Return ``multi_modal_data`` from the request, or ``None`` if absent."""
    return request.get("multi_modal_data")  # type: ignore[return-value]


def require_encoder_result(request: GenerateRequest, mode: DisaggregationMode) -> Any:
    """Return ``encoder_result``, raising if absent.

    Prefill/Aggregated workers in E/PD topologies must call this before
    consuming the encoder payload so misconfigured pipelines fail loudly.
    """
    result = request.get("encoder_result")
    if result is None:
        raise ValueError(
            f"{mode.value} worker received multimodal request with no encoder_result; "
            "expected the Encode worker to populate this field before forwarding "
            "the request to the downstream prefill/aggregated peer"
        )
    return result


def encoder_terminal_chunk(
    encoder_result: Any,
    prompt_len: int,
    index: int = 0,
) -> GenerateChunk:
    """Build the terminal chunk emitted by an Encode worker."""
    return GenerateChunk(
        token_ids=[],
        index=index,
        finish_reason="encode",
        completion_usage={
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "total_tokens": prompt_len,
        },
        encoder_result=encoder_result,
    )
