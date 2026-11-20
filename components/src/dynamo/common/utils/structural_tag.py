# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from dynamo.llm.exceptions import InvalidArgument

VLLM_STRUCTURED_OUTPUT_BACKEND = "vllm_structured_output_backend"
VLLM_STRUCTURAL_TAG_FORMAT_SUPPORTED = "vllm_structural_tag_format_supported"
TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY = (
    "tool_call_structural_tag_excludes_reasoning"
)


def serialize_structural_tag(structural_tag: Any) -> str | None:
    """Return structural tag as the JSON string expected by backends."""
    if structural_tag is None:
        return None
    if hasattr(structural_tag, "model_dump"):
        structural_tag = structural_tag.model_dump()
    return (
        structural_tag
        if isinstance(structural_tag, str)
        else json.dumps(structural_tag)
    )


def validate_structural_tag_backend(
    structural_tag: str,
    backend: str,
    *,
    backend_label: str,
    supported_backends: tuple[str, ...],
    legacy_backend: str,
) -> None:
    """Reject unsupported tags before handing them to the selected engine."""
    if backend in supported_backends or (
        backend == legacy_backend and _accepts_legacy_tag(structural_tag)
    ):
        return

    supported = " or ".join(repr(name) for name in supported_backends)
    raise InvalidArgument(
        f"{backend_label} {backend!r} does not support this structural tag. "
        f"Configure the worker to use {supported}."
    )


def _accepts_legacy_tag(tag: Any) -> bool:
    if isinstance(tag, str):
        try:
            tag = json.loads(tag)
        except json.JSONDecodeError:
            # Leave malformed grammar syntax to the engine.
            return True
    return (
        isinstance(tag, dict)
        and tag.get("triggers") is not None
        and tag.get("structures") is not None
    )


def supports_structural_tag_format(backend: str | None) -> bool | None:
    """Support for the modern format tag in the supported vLLM release."""
    if backend in ("auto", "xgrammar"):
        return True
    if backend in ("guidance", "outlines", "lm-format-enforcer"):
        return False
    return None


def structural_tag_backend_compatible(tag: Any, backend: str | None) -> bool:
    return supports_structural_tag_format(backend) is True or (
        backend == "guidance" and _accepts_legacy_tag(tag)
    )


def structural_tag_backend_error(backend: str | None) -> str:
    return (
        f"vLLM structured-output backend {backend!r} does not support this "
        "structural tag, or its support is unknown. This request needs a tag "
        "to enforce the output format and reasoning boundary. Configure the "
        "worker to use 'auto' or 'xgrammar' and upgrade workers before the frontend."
    )


def publish_vllm_structural_tag_backend(runtime_config: Any, vllm_config: Any) -> None:
    backend = vllm_config.structured_outputs_config.backend
    runtime_config.set_engine_specific(
        VLLM_STRUCTURED_OUTPUT_BACKEND, json.dumps(backend)
    )
    runtime_config.set_engine_specific(
        VLLM_STRUCTURAL_TAG_FORMAT_SUPPORTED,
        json.dumps(supports_structural_tag_format(backend)),
    )


def publish_vllm_structural_tag_reasoning_policy(
    runtime_config: Any, vllm_config: Any
) -> None:
    """Publish whether the vLLM engine holds grammar back until reasoning ends."""
    config = vllm_config.structured_outputs_config
    runtime_config.set_engine_specific(
        TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
        json.dumps(bool(config.reasoning_parser) and not config.enable_in_reasoning),
    )
