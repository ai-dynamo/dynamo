# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning parity wrapper for SGLang's reasoning parser."""

from __future__ import annotations

from typing import Any

from sglang.srt.parser.reasoning_parser import ReasoningParser

from tests.parity.common import ReasoningResult

_FAMILY_TO_SGLANG_REASONING = {
    "deepseek_r1": "deepseek-r1",
    "gpt_oss": "gpt-oss",
    "kimi": "kimi",
    "qwen3": "qwen3",
}


def _make_parser(parser_name: str, fixture: dict[str, Any]) -> ReasoningParser:
    return ReasoningParser(
        model_type=parser_name,
        stream_reasoning=True,
        force_reasoning=fixture.get("force_reasoning", False),
    )


def parse(
    parser_family: str,
    fixture: dict[str, Any],
    mode: str,
) -> ReasoningResult:
    parser_name = fixture.get("sglang_parser") or _FAMILY_TO_SGLANG_REASONING.get(
        parser_family
    )
    if not parser_name:
        return ReasoningResult(
            error=f"UNAVAILABLE: SGLang has no reasoning parser for family={parser_family!r}"
        )

    try:
        parser = _make_parser(parser_name, fixture)
        if mode == "stream":
            chunks = fixture["chunks"]
        elif mode == "batch":
            chunks = [fixture["model_text"]]
        else:
            raise ValueError(f"unsupported reasoning mode: {mode!r}")
        reasoning_text = ""
        normal_text = ""
        for chunk in chunks:
            reasoning_delta, normal_delta = parser.parse_stream_chunk(chunk)
            reasoning_text += reasoning_delta or ""
            normal_text += normal_delta or ""
    except Exception as e:
        return ReasoningResult(error=f"{type(e).__name__}: {e}")

    return ReasoningResult(reasoning_text=reasoning_text, normal_text=normal_text)
