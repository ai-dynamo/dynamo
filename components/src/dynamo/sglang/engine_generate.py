# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opaque SGLang request handling for Dynamo native Generate API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from pydantic import TypeAdapter
from sglang.srt.managers.io_struct import GenerateReqInput

SGLANG_GENERATE_CAPABILITY = "sglang_generate"
_PAYLOAD_KEY = "sglang_tito"
_GENERATE_REQUEST_ADAPTER = TypeAdapter(GenerateReqInput)


def is_native_generate_request(request: Mapping[str, Any]) -> bool:
    """Return whether the canonical request carries a native SGLang body."""
    extra_args = request.get("extra_args")
    return isinstance(extra_args, dict) and isinstance(
        extra_args.get(_PAYLOAD_KEY), dict
    )


def build_native_generate_request(
    request: Mapping[str, Any],
    *,
    input_ids: list[int],
    fallback_rid: str,
    priority: int | None,
    sampling_overrides: Mapping[str, Any] | None = None,
    internal_fields: Mapping[str, Any] | None = None,
) -> GenerateReqInput | None:
    """Reconstruct the installed SGLang version native request.

    The Rust frontend preserves the public request opaquely under
    ``extra_args.sglang_tito``. Dynamo replaces only canonical input,
    routing state, and fields supplied by the selected worker. SGLang owns
    all remaining validation.
    """
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    native_payload = extra_args.get(_PAYLOAD_KEY)
    if not isinstance(native_payload, dict):
        return None

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

    if internal_fields:
        payload.update(
            {
                name: value
                for name, value in internal_fields.items()
                if value is not None
            }
        )

    return _GENERATE_REQUEST_ADAPTER.validate_python(payload)


def native_generate_stream(
    engine: Any, request: GenerateReqInput
) -> AsyncIterator[dict[str, Any]]:
    """Dispatch exactly as SGLang native ``/generate`` handler does."""
    return engine.tokenizer_manager.generate_request(request, None)
