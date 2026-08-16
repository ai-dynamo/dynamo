# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opaque SGLang request handling for Dynamo native Generate API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from pydantic import TypeAdapter
from sglang.srt.managers.io_struct import GenerateReqInput

from dynamo.common.backend import logprobs as _shared_logprobs

SGLANG_GENERATE_CAPABILITY = "sglang_generate"
_PAYLOAD_KEY = "sglang_tito"
_GENERATE_REQUEST_ADAPTER = TypeAdapter(GenerateReqInput)


def native_generate_payload(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the opaque native SGLang body carried by a canonical request."""
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    payload = extra_args.get(_PAYLOAD_KEY)
    return payload if isinstance(payload, dict) else None


def build_native_generate_request(
    native_payload: Mapping[str, Any],
    *,
    input_ids: list[int],
    fallback_rid: str,
    priority: int | None,
    sampling_overrides: Mapping[str, Any] | None = None,
    bootstrap_host: str | None = None,
    bootstrap_port: int | None = None,
    bootstrap_room: int | None = None,
    external_trace_header: dict[str, str] | None = None,
    routed_dp_rank: int | None = None,
    lora_path: str | None = None,
    cache_salt: str | None = None,
) -> GenerateReqInput:
    """Reconstruct the installed SGLang version native request.

    The Rust frontend preserves the public request opaquely under
    ``extra_args.sglang_tito``. Dynamo replaces only canonical input,
    routing state, and fields supplied by the selected worker. SGLang owns
    all remaining validation.
    """
    payload = dict(native_payload)
    payload["input_ids"] = input_ids
    payload["rid"] = payload.get("rid") or fallback_rid
    payload["stream"] = True
    if priority is None:
        payload.pop("priority", None)
    else:
        payload["priority"] = priority

    if sampling_overrides:
        sampling_params = payload.get("sampling_params")
        if sampling_params is None:
            sampling_params = {}
        if not isinstance(sampling_params, dict):
            raise ValueError("sampling_params must be an object")
        payload["sampling_params"] = {
            **sampling_params,
            **sampling_overrides,
        }

    for name, value in (
        ("bootstrap_host", bootstrap_host),
        ("bootstrap_port", bootstrap_port),
        ("bootstrap_room", bootstrap_room),
        ("external_trace_header", external_trace_header),
        ("routed_dp_rank", routed_dp_rank),
        ("lora_path", lora_path),
        ("cache_salt", cache_salt),
    ):
        if value is not None:
            payload[name] = value

    native_request = _GENERATE_REQUEST_ADAPTER.validate_python(payload)
    _shared_logprobs.validate_sglang_top_logprobs(
        native_request.top_logprobs_num,
        allow_top_logprobs=_shared_logprobs.sglang_top_logprobs_allowed(),
    )
    return native_request


async def native_generate_stream(
    engine: Any, request: GenerateReqInput
) -> AsyncIterator[dict[str, Any]]:
    """Dispatch exactly as SGLang native ``/generate`` handler does."""
    async for response in engine.tokenizer_manager.generate_request(request, None):
        yield {"token_ids": [], "engine_data": {"sglang_response": response}}
