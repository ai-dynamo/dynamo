# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning parity wrapper for Dynamo's Rust parsers, via the PyO3 binding."""

from __future__ import annotations

import json
from typing import Any

from dynamo._core import parse_reasoning_batch, parse_reasoning_stream
from tests.parity.common import ReasoningResult


def parse(
    parser_family: str,
    fixture: dict[str, Any],
    mode: str,
) -> ReasoningResult:
    try:
        if mode == "stream":
            raw_json = parse_reasoning_stream(
                parser_family,
                fixture["chunks"],
                fixture.get("token_chunks"),
                fixture.get("in_reasoning", False),
            )
        elif mode == "batch":
            raw_json = parse_reasoning_batch(
                parser_family,
                fixture["model_text"],
                fixture.get("token_ids"),
                fixture.get("in_reasoning", False),
            )
        else:
            raise ValueError(f"unsupported reasoning mode: {mode!r}")
        raw = json.loads(raw_json)
    except Exception as e:
        return ReasoningResult(error=f"{type(e).__name__}: {e}")

    return ReasoningResult(
        reasoning_text=raw.get("reasoning_text"),
        normal_text=raw.get("normal_text"),
    )
