# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

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
    """Whether tool-call structural tags must leave reasoning to the backend.

    Mirrors ``resolve_reasoning_boundary`` in the Rust preprocessor: the
    canonical ``structural_tag.reasoning_boundary`` setting (``auto`` by
    default) decides, and ``auto`` follows the vLLM worker's published
    ``tool_call_structural_tag_excludes_reasoning`` flag. Missing or invalid
    worker metadata keeps the compatibility behavior (the tag models reasoning).
    """
    if not isinstance(runtime_config, dict):
        return False
    runtime_data = runtime_config.get("runtime_data")
    backend_excludes = (
        isinstance(runtime_data, dict)
        and runtime_data.get(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)
        is True
    )
    config = runtime_config.get("structural_tag")
    boundary = (
        config.get("reasoning_boundary", "auto") if isinstance(config, dict) else "auto"
    )
    if boundary == "backend":
        return True
    if boundary == "structural_tag" and backend_excludes:
        # Rust rejects this pairing at startup; follow the backend, which
        # consumes the reasoning block before the grammar sees any token.
        logger.warning(
            "structural_tag.reasoning_boundary=structural_tag conflicts with the "
            "backend's reasoning-aware guided-decoding policy; using 'backend'"
        )
    return backend_excludes


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


def unsupported_regex_construct(pattern: str) -> str | None:
    """Return XGrammar's error for ``pattern``, or ``None`` when it compiles.

    Asks the installed XGrammar instead of mirroring its regex syntax, so the
    verdict follows the version the vLLM or SGLang backend ships with. Its regex
    converter is the strictest path any tag format takes. The Rust preprocessor
    has no XGrammar and keeps a list of constructs instead.
    """
    import xgrammar

    try:
        xgrammar.Grammar.from_regex(pattern)
    except RuntimeError as error:
        return str(error).splitlines()[0]
    return None


_INSTANCE_KEYWORDS = frozenset({"const", "default", "enum", "examples"})
_SUBSCHEMA_MAPS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependencies",
    }
)


def unsupported_schema_construct(schema: Any) -> str | None:
    """Return XGrammar's error for the first regex in ``pattern`` values or
    ``patternProperties`` keys it cannot compile. Subschema maps are searched by
    value only, so a property named ``default`` is still checked.
    """
    if isinstance(schema, list):
        for item in schema:
            if construct := unsupported_schema_construct(item):
                return construct
        return None
    if not isinstance(schema, dict):
        return None
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and (construct := unsupported_regex_construct(pattern)):
        return construct
    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict):
        for key in pattern_properties:
            if construct := unsupported_regex_construct(key):
                return construct
    for key, value in schema.items():
        if key in _INSTANCE_KEYWORDS:
            continue
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            value = list(value.values())
        if construct := unsupported_schema_construct(value):
            return construct
    return None


def _strip_unsupported_patterns(schema: Any) -> str | None:
    """Remove ``pattern`` values XGrammar cannot compile, in place. Returns XGrammar's
    error for an unsupported ``patternProperties`` key, which is kept, since
    removing it would change which property names are allowed.
    """
    if isinstance(schema, list):
        for item in schema:
            if construct := _strip_unsupported_patterns(item):
                return construct
        return None
    if not isinstance(schema, dict):
        return None
    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict):
        for key in pattern_properties:
            if construct := unsupported_regex_construct(key):
                return construct
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and unsupported_regex_construct(pattern):
        del schema["pattern"]
    for key, value in schema.items():
        if key in _INSTANCE_KEYWORDS:
            continue
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            value = list(value.values())
        if construct := _strip_unsupported_patterns(value):
            return construct
    return None


def effective_tool_strict(request_strict: bool | None, schema_mode: str) -> bool:
    """Resolve request strictness for grammar construction.

    In ``auto`` schema mode, omitted ``strict`` is schema-enforced and only an
    explicit ``strict: false`` opts out. ``strict`` schema mode overrides that
    request-level opt-out.
    """
    return schema_mode == "strict" or request_strict is not False


def guidance_tool_schema(
    request_strict: bool | None,
    schema_mode: str,
    parameters: Any,
    *,
    tool_name: str | None = None,
) -> tuple[Any, bool]:
    """Return the argument schema and strictness to build a tool's grammar from.

    Leaves regexes XGrammar cannot compile out of schemas Dynamo enforces on its
    own: omitted ``strict`` under schema mode ``auto``, which OpenAI treats as
    non-strict. Otherwise one bad ``pattern`` fails the whole request. Explicit
    ``strict`` and schema mode ``strict`` are left for the backend to reject.
    """
    strict = effective_tool_strict(request_strict, schema_mode)
    if schema_mode == "strict" or request_strict is not None:
        return parameters, strict
    construct = unsupported_schema_construct(parameters)
    if construct is None:
        return parameters, strict
    stripped = copy.deepcopy(parameters)
    if property_name_construct := _strip_unsupported_patterns(stripped):
        logger.debug(
            "Tool %r: the grammar backend cannot compile a patternProperties "
            "key (%s); not enforcing its arguments",
            tool_name,
            property_name_construct,
        )
        return parameters, False
    logger.debug(
        "Tool %r: leaving a pattern the grammar backend cannot compile (%s) "
        "out of its argument grammar",
        tool_name,
        construct,
    )
    return stripped, strict
