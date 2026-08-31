# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the Inkling vLLM runtime markers baked into this image.

Asserts the POSTCONDITIONS of
`container/deps/vllm/patches/v0.28.0/inkling/*.patch`, not merely that `patch`
exited zero. The failure this guards against is silent: without a registered
structural tag, Dynamo installs no constraint for a tool request and degrades to
`tool_choice="auto"` with HTTP 200 and no warning, so a forced tool choice is
simply not honoured.
"""

from __future__ import annotations

import importlib.metadata as metadata

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.structural_tag_registry import (
    _VLLM_STRUCTURAL_TAG_REGISTRY,
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
)

EXPECTED_VLLM_VERSION = "0.28.0"


def _check_version() -> list[str]:
    installed = metadata.version("vllm")
    if installed != EXPECTED_VLLM_VERSION:
        return [
            f"vllm {installed!r} != expected {EXPECTED_VLLM_VERSION!r}; the Inkling "
            "patches are cut against that exact base"
        ]
    return []


def _check_structural_tag() -> list[str]:
    errors: list[str] = []
    if "inkling" not in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS:
        errors.append(
            "'inkling' missing from VLLM_BUILTIN_STRUCTURAL_TAG_MODELS -- the "
            "structural-tag patch did not apply"
        )
    if "inkling" not in _VLLM_STRUCTURAL_TAG_REGISTRY:
        errors.append(
            "no structural-tag builder registered for 'inkling' in "
            "_VLLM_STRUCTURAL_TAG_REGISTRY"
        )
    return errors


def _check_tool_parser() -> list[str]:
    """The parser must ADVERTISE the tag, and must NOT hand-set the strict flag.

    `AbstractToolParser.__init_subclass__` forces `supports_required_and_named`
    to False whenever `structural_tag_model` is set and
    VLLM_ENFORCE_STRICT_TOOL_CALLING is on (the default). Setting it in the
    subclass is therefore both redundant and silently overwritten; asserting the
    derived value here is what keeps that from being reintroduced.
    """
    import vllm.envs as envs

    errors: list[str] = []
    parser = ToolParserManager.get_tool_parser("inkling")
    if getattr(parser, "structural_tag_model", None) != "inkling":
        errors.append(
            f"InklingToolParser.structural_tag_model = "
            f"{getattr(parser, 'structural_tag_model', None)!r}, expected 'inkling'"
        )
    expected = not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING
    actual = parser.supports_required_and_named
    if actual is not expected:
        errors.append(
            f"InklingToolParser.supports_required_and_named = {actual!r}, expected "
            f"{expected!r} (derived from VLLM_ENFORCE_STRICT_TOOL_CALLING)"
        )
    return errors


def main() -> int:
    errors = _check_version() + _check_structural_tag() + _check_tool_parser()
    if errors:
        print("Inkling runtime validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"Inkling runtime validation passed (vllm {EXPECTED_VLLM_VERSION}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
