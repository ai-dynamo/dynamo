# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract helpers for the unified-backend multimodal wire shape.

Three small helpers, parallel to the prefill/decode pair in
``dynamo.common.backend.disagg``:

* :func:`extract_multimodal_kwargs` — pull ``multi_modal_data`` and
  ``mm_processor_kwargs`` off a request into a flat ``**kwargs`` dict
  suitable for engine APIs that accept those names.
* :func:`require_encoder_result` — read ``encoder_result`` off a
  downstream Prefill/Aggregated request; raise when missing or wrongly
  shaped (the encoder handoff contract).
* :func:`encoder_terminal_chunk` — build the terminal ``GenerateChunk``
  an Encode worker yields.

These are utilities, not abstractions. Backends are free to inline the
behavior if their generate path is shaped differently.

The wire shape (matching the Rust producer side
``LLMEngineOutput::encode_terminal``) is **engine-opaque**: the framework
guarantees the payload is a JSON object, not the keys inside it. Engines
own content validation.
"""

from __future__ import annotations

from typing import Any, Optional

from dynamo.common.backend.engine import GenerateChunk, GenerateRequest
from dynamo.common.constants import DisaggregationMode


def extract_multimodal_kwargs(request: GenerateRequest) -> Optional[dict[str, Any]]:
    """Return ``{"multi_modal_data": ..., "mm_processor_kwargs": ...}``
    with keys omitted when absent OR ``None``-valued.

    Suitable for splatting into engine APIs that accept those kwarg
    names. Returns ``None`` when neither key has a non-None value so
    callers can use ``extract_multimodal_kwargs(req) or {}`` to spread
    onto a kwargs dict without a branch.

    Both keys are passed through unchanged (including empty ``{}``):
    the framework guarantees the shape (object), engines own content
    validation.
    """
    result: dict[str, Any] = {}
    for key in ("multi_modal_data", "mm_processor_kwargs"):
        value = request.get(key)
        if value is not None:
            result[key] = value
    return result or None


def require_encoder_result(
    request: GenerateRequest, mode: DisaggregationMode
) -> dict[str, Any]:
    """Return the request's ``encoder_result`` dict, raising on missing
    or wrongly shaped.

    Analog of :func:`dynamo.common.backend.disagg.require_prefill_result`.
    The frontend is expected to forward the Encode worker's terminal-chunk
    ``encoder_result`` onto the downstream
    ``PreprocessedRequest.encoder_result``; missing means the request
    bypassed encoder routing (or the engine is misconfigured).

    Raises:
        ValueError: when ``encoder_result`` is absent.
        ValueError: when ``encoder_result`` is not a ``dict`` (the wire
            shape is object-only by contract).
    """
    encoder_result = request.get("encoder_result")
    if encoder_result is None:
        raise ValueError(
            f"{mode.value} worker received request with no encoder_result; "
            "expected the frontend to forward the encoder handoff payload "
            "from an Encode peer"
        )
    if not isinstance(encoder_result, dict):
        raise ValueError(
            f"encoder_result must be a JSON object (dict); got "
            f"{type(encoder_result).__name__}({encoder_result!r}). The wire "
            "shape is object-only by contract."
        )
    return encoder_result


def encoder_terminal_chunk(
    encoder_result: dict[str, Any],
    *,
    completion_usage: Optional[dict[str, int]] = None,
) -> GenerateChunk:
    """Build the terminal ``GenerateChunk`` an Encode worker yields.

    Exact shape (matching Rust ``LLMEngineOutput::encode_terminal``):
    empty ``token_ids``, ``index=0``, ``finish_reason="stop"``,
    ``encoder_result`` carried through unchanged.

    ``completion_usage`` is **omitted** from the returned dict when
    ``None`` (matches the ``GenerateChunk`` ``total=False`` semantics and
    the Rust ``Option`` / ``skip_serializing_if = "Option::is_none"``
    style); only included as a key when the caller passes a non-None
    dict.

    Raises:
        TypeError: when ``encoder_result`` is not a ``dict`` (the wire
            shape is object-only by contract).
    """
    if not isinstance(encoder_result, dict):
        raise TypeError(
            f"encoder_result must be a dict; got "
            f"{type(encoder_result).__name__}({encoder_result!r}). The wire "
            "shape is object-only by contract."
        )
    chunk: GenerateChunk = {
        "token_ids": [],
        "index": 0,
        "finish_reason": "stop",
        "encoder_result": encoder_result,
    }
    if completion_usage is not None:
        chunk["completion_usage"] = completion_usage
    return chunk
