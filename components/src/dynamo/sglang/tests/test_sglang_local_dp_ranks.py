# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


def _slice(**overrides):
    from dynamo.sglang.llm_engine import _local_dp_rank_range as helper

    fields = {
        "dp_size": 1,
        "enable_dp_attention": False,
        "nnodes": 1,
        "node_rank": 0,
        **overrides,
    }
    return helper(SimpleNamespace(**fields))


def test_dp_attention_disabled_short_circuits_to_rank_zero():
    assert _slice() == (0, 1)
    assert _slice(dp_size=8) == (0, 1)


def test_dp_attention_multi_node_slices_ranks_by_node_index():
    assert _slice(dp_size=8, enable_dp_attention=True, nnodes=2, node_rank=0) == (0, 4)
    assert _slice(dp_size=8, enable_dp_attention=True, nnodes=2, node_rank=1) == (4, 8)


def test_missing_attributes_use_safe_defaults():
    from dynamo.sglang.llm_engine import _local_dp_rank_range as helper

    assert helper(SimpleNamespace()) == (0, 1)
