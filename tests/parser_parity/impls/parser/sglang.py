# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for SGLang's Python tool detectors (in-process import)."""

from __future__ import annotations

import json
from typing import Any

# SGLang re-exports detectors through `function_call_parser`; we go through
# that umbrella so the import is stable across per-module renames.
from sglang.srt.function_call.function_call_parser import (  # type: ignore[import-untyped]
    DeepSeekV31Detector,
    Glm47MoeDetector,
)
from sglang.srt.function_call.gpt_oss_detector import (
    GptOssDetector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.kimik2_detector import (
    KimiK2Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.qwen3_coder_detector import (  # type: ignore[import-untyped]
    Qwen3CoderDetector,
)

from tests.parser_parity.impls.common import ParseResult

# Maps parser_family → SGLang detector class. SGLang doesn't have a registry-by-name
# like vLLM; the class is imported directly.
_FAMILY_TO_SGLANG_DETECTOR = {
    "kimi_k2": KimiK2Detector,
    "qwen3_coder": Qwen3CoderDetector,
    "glm47": Glm47MoeDetector,
    "deepseek_v3_1": DeepSeekV31Detector,
    "harmony": GptOssDetector,
}

# Families with no SGLang detector today: minimax_m2, nemotron_deci.


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    detector_cls = _FAMILY_TO_SGLANG_DETECTOR.get(parser_family)
    if detector_cls is None:
        return ParseResult(error=f"SGLang has no detector for family={parser_family!r}")

    detector = detector_cls()

    # SGLang's BaseFormatDetector exposes detect_and_parse(text, tools) where
    # tools is the OpenAI Tool[] shape. Build that wrapper if our fixture
    # passed flat-shape definitions.
    sg_tools = _build_tools(tools)

    try:
        info = detector.detect_and_parse(raw_text, sg_tools)
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    calls = []
    # SGLang returns StreamingParseResult-like objects with .calls (list of ToolCallItem)
    for tc in info.calls or []:
        args_str = tc.parameters if hasattr(tc, "parameters") else tc.arguments
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError):
            args = args_str
        calls.append({"name": tc.name, "arguments": args})

    normal_text = info.normal_text if hasattr(info, "normal_text") else None
    return ParseResult(calls=calls, normal_text=normal_text)


class _Function:
    def __init__(self, name: str, parameters: Any, strict: bool = False) -> None:
        self.name = name
        self.parameters = parameters
        self.strict = strict


class _Tool:
    def __init__(self, name: str, parameters: Any) -> None:
        self.function = _Function(name, parameters)


def _build_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    """Wrap flat tool defs in pydantic-like objects that SGLang expects.

    SGLang detectors access `tool.function.name` / `tool.function.parameters`
    via attribute access, not dict subscript, so plain dicts fail with
    AttributeError. Provide minimal duck-typed objects.
    """
    if not tools:
        return None
    wrapped = []
    for t in tools:
        if "function" in t:
            inner = t["function"]
            wrapped.append(_Tool(inner["name"], inner.get("parameters")))
        else:
            wrapped.append(_Tool(t["name"], t.get("parameters")))
    return wrapped
