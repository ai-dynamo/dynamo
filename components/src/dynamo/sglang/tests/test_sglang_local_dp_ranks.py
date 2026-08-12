# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import (
    local_dp_rank_bounds,
    model_card_dp_rank_bounds,
    publishes_kv_events,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


def test_model_card_registration_keeps_global_dp_range():
    server_args = SimpleNamespace(
        dp_size=16,
        enable_dp_attention=True,
        nnodes=4,
        node_rank=0,
    )

    assert model_card_dp_rank_bounds(server_args) == (0, 16)


def _args(**kwargs) -> SimpleNamespace:
    base = dict(dp_size=1, enable_dp_attention=False, nnodes=1, node_rank=0)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_single_node_publishes_kv_events():
    assert publishes_kv_events(_args()) is True


def test_multinode_without_dp_attention_publishes_only_from_leader():
    """TP-only multinode must advertise one source per logical worker."""
    leader = _args(nnodes=2, node_rank=0)
    follower = _args(nnodes=2, node_rank=1)

    # Precondition for the collision this guards against.
    assert local_dp_rank_bounds(leader) == local_dp_rank_bounds(follower) == (0, 1)

    assert publishes_kv_events(leader) is True
    assert publishes_kv_events(follower) is False


def test_dp_attention_publishes_from_every_node():
    """Each node owns a distinct slice when DP attention is enabled."""
    nodes = [
        _args(dp_size=4, enable_dp_attention=True, nnodes=2, node_rank=rank)
        for rank in (0, 1)
    ]
    assert local_dp_rank_bounds(nodes[0]) != local_dp_rank_bounds(nodes[1])
    assert all(publishes_kv_events(node) is True for node in nodes)


def test_dp_size_one_with_dp_attention_still_leader_only():
    """DP size one keeps the shared [0, 1) slice even with the flag set."""
    assert (
        publishes_kv_events(_args(enable_dp_attention=True, nnodes=2, node_rank=1))
        is False
    )
