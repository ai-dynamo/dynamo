# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang unified engine's disagg dispatch."""

from __future__ import annotations

import pytest

pytest.importorskip("sglang", reason="sglang not installed in this container")

from dynamo.common.constants import DisaggregationMode  # noqa: E402
from dynamo.sglang.llm_engine import SglangLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _engine_with_cached_bootstrap() -> SglangLLMEngine:
    # __new__ bypasses real engine init; we only exercise pure-Python helpers.
    engine = SglangLLMEngine.__new__(SglangLLMEngine)
    engine._bootstrap_host = "10.0.0.5"
    engine._bootstrap_port = 12345
    engine.serving_mode = DisaggregationMode.PREFILL
    return engine


def test_prefill_bootstrap_honours_router_provided_triple():
    # Router-resolved triple must be passed through unchanged so the
    # decode peer (which the router writes the same triple onto) finds
    # the matching room.
    engine = _engine_with_cached_bootstrap()
    out = engine._resolve_prefill_bootstrap(
        {
            "token_ids": [1, 2, 3],
            "bootstrap_info": {
                "bootstrap_host": "router.host",
                "bootstrap_port": 9999,
                "bootstrap_room": 42,
            },
        }
    )
    assert out == {
        "bootstrap_host": "router.host",
        "bootstrap_port": 9999,
        "bootstrap_room": 42,
    }


def test_decode_bootstrap_raises_when_router_left_no_info():
    # Both router paths missing → loud failure. Silent zero-fill would
    # leave the decode worker waiting on a phantom KV transfer.
    with pytest.raises(ValueError, match="bootstrap"):
        SglangLLMEngine._resolve_decode_bootstrap({"token_ids": [1, 2, 3]})
