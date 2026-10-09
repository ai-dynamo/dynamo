# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

ToolChoiceKind = Literal["none", "auto", "required", "named", "other"]


def runtime_structural_tag_options(
    runtime_config: Any,
) -> tuple[str, str, str]:
    """Read structural-tag policy from a model-card runtime config."""
    if not isinstance(runtime_config, dict):
        return "off", "auto", "auto"
    # New cards project legacy fields for old readers. Prefer the canonical
    # setting here, including its explicit disabled value, when present.
    if "structural_tag" in runtime_config:
        config = runtime_config["structural_tag"]
        if not isinstance(config, dict):
            return "off", "auto", "auto"
        return "on", config.get("scope", "auto"), config.get("schema", "auto")
    return (
        runtime_config.get("structural_tag_mode", "off"),
        runtime_config.get("structural_tag_scope", "auto"),
        runtime_config.get("structural_tag_schema", "auto"),
    )


# Published by the vLLM worker (dynamo.vllm.main.publish_vllm_structural_tag_reasoning_policy)
# and read by the Rust preprocessor (lib/llm/src/preprocessor/structural_tag.rs).
TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY = (
    "tool_call_structural_tag_excludes_reasoning"
)


def runtime_structural_tag_excludes_reasoning(runtime_config: Any) -> bool:
    """Whether the worker engine applies tool grammars only after reasoning ends.

    True only when the vLLM worker runs its own reasoning parser with
    ``enable_in_reasoning`` off. Missing or invalid metadata keeps the
    compatibility behavior (the grammar must model the open reasoning block),
    as in the Rust preprocessor.
    """
    if not isinstance(runtime_config, dict):
        return False
    runtime_data = runtime_config.get("runtime_data")
    if not isinstance(runtime_data, dict):
        return False
    return (
        runtime_data.get(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)
        is True
    )


def should_attempt_structural_tag(
    *,
    mode: str,
    scope: str,
    tool_choice_kind: ToolChoiceKind,
    has_tools: bool,
    any_explicit_strict: bool,
    parallel_tool_calls_explicitly_false: bool,
) -> bool:
    """Return whether frontend preprocessing should request a structural tag."""
    if mode != "on" or not has_tools or tool_choice_kind == "none":
        return False
    if tool_choice_kind in {"required", "named"}:
        return True
    if tool_choice_kind != "auto":
        return False
    if scope == "always":
        return True
    return any_explicit_strict or parallel_tool_calls_explicitly_false


def effective_tool_strict(request_strict: bool | None, schema_mode: str) -> bool:
    """Resolve request strictness for grammar construction.

    In ``auto`` schema mode, omitted ``strict`` is schema-enforced and only an
    explicit ``strict: false`` opts out. ``strict`` schema mode overrides that
    request-level opt-out.
    """
    return schema_mode == "strict" or request_strict is not False
