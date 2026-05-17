# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parser-mode wrapper for SGLang's Python tool detectors (in-process import)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sglang.srt.function_call.deepseekv3_detector import (
    DeepSeekV3Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,  # type: ignore[import-untyped]
)

# SGLang re-exports detectors through `function_call_parser`; we go through
# that umbrella so the import is stable across per-module renames.
from sglang.srt.function_call.function_call_parser import (  # type: ignore[import-untyped]
    DeepSeekV31Detector,
    Glm47MoeDetector,
)
from sglang.srt.function_call.gemma4_detector import (
    Gemma4Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.gpt_oss_detector import (
    GptOssDetector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.hermes_detector import (
    HermesDetector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.kimik2_detector import (
    KimiK2Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.llama32_detector import (
    Llama32Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.minimax_m2 import (
    MinimaxM2Detector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.mistral_detector import (
    MistralDetector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.pythonic_detector import (
    PythonicDetector,  # type: ignore[import-untyped]
)
from sglang.srt.function_call.qwen3_coder_detector import (  # type: ignore[import-untyped]
    Qwen3CoderDetector,
)
from sglang.srt.function_call.qwen25_detector import (
    Qwen25Detector,  # type: ignore[import-untyped]
)

from tests.parity.common import ParseResult, decode_arguments

# Maps parser_family → SGLang detector class. SGLang doesn't have a registry-by-name
# like vLLM; the class is imported directly.
_FAMILY_TO_SGLANG_DETECTOR = {
    "kimi_k2": KimiK2Detector,
    "qwen3_coder": Qwen3CoderDetector,
    "qwen25": Qwen25Detector,
    "glm47": Glm47MoeDetector,
    "deepseek_v3_1": DeepSeekV31Detector,
    "deepseek_v3_2": DeepSeekV32Detector,
    "deepseek_v3": DeepSeekV3Detector,
    "gemma4": Gemma4Detector,
    "harmony": GptOssDetector,
    "llama3_json": Llama32Detector,
    "minimax_m2": MinimaxM2Detector,
    "pythonic": PythonicDetector,
    "hermes": HermesDetector,
    "mistral": MistralDetector,
}

# Families with no SGLang detector today: nemotron_deci, nemotron_nano,
# deepseek_v4, jamba, phi4.


def parse(
    parser_family: str,
    raw_text: str,
    tools: list[dict[str, Any]] | None,
) -> ParseResult:
    detector_cls = _FAMILY_TO_SGLANG_DETECTOR.get(parser_family)
    if detector_cls is None:
        return ParseResult(
            error=f"UNAVAILABLE: SGLang has no detector for family={parser_family!r}"
        )

    try:
        detector = detector_cls()
        # SGLang's BaseFormatDetector exposes detect_and_parse(text, tools) where
        # tools is the OpenAI Tool[] shape. Build that wrapper if our fixture
        # passed flat-shape definitions.
        sg_tools = _build_tools(tools)
        info = detector.detect_and_parse(raw_text, sg_tools)
    except Exception as e:
        return ParseResult(error=f"{type(e).__name__}: {e}")

    # SGLang returns StreamingParseResult with .calls; each item exposes
    # arguments via either .parameters or .arguments depending on version.
    calls = [
        {
            "name": tc.name,
            "arguments": decode_arguments(
                getattr(tc, "parameters", None) or tc.arguments
            ),
        }
        for tc in info.calls or []
    ]
    return ParseResult(calls=calls, normal_text=info.normal_text)


def _build_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    """Wrap flat tool defs as duck-typed objects with `.function.name` /
    `.function.parameters`. SGLang detectors access these via attribute,
    not dict subscript, so plain dicts fail with AttributeError.
    """
    if not tools:
        return None
    return [
        SimpleNamespace(
            type="function",
            function=SimpleNamespace(
                name=(t["function"] if "function" in t else t)["name"],
                parameters=(t["function"] if "function" in t else t).get("parameters"),
                strict=False,
            ),
        )
        for t in tools
    ]
