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


def test_model_card_registration_keeps_global_dp_range():
    from dynamo.sglang.capacity import model_card_dp_rank_bounds

    server_args = SimpleNamespace(
        dp_size=16,
        enable_dp_attention=True,
        nnodes=4,
        node_rank=0,
    )

    assert model_card_dp_rank_bounds(server_args) == (0, 16)
