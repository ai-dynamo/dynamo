# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import (
    local_dp_rank_bounds,
    model_card_dp_rank_bounds,
    per_rank_max_running_requests,
    publishes_kv_events,
    sglang_dp_layout,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
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


def test_single_node_pure_dp_exposes_every_local_rank():
    assert local_dp_rank_bounds(_args(dp_size=4)) == (0, 4)


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


def test_pure_dp_keeps_per_scheduler_max_running_requests():
    server_args = _args(dp_size=4, max_running_requests=128)

    assert per_rank_max_running_requests(server_args) == 128


def test_dp_attention_splits_global_max_running_requests():
    server_args = _args(
        dp_size=4,
        enable_dp_attention=True,
        max_running_requests=128,
    )

    assert per_rank_max_running_requests(server_args) == 32


# SGLang >= #41818 resolves `--dp-size N --enable-dp-attention` to
# `attn_dp_size=N, dp_size=1, enable_dp_attention=False`.
def _attn_dp_args(attn_dp_size: int, **kwargs) -> SimpleNamespace:
    return _args(attn_dp_size=attn_dp_size, **kwargs)


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        (_args(), (1, False)),
        (_args(dp_size=4), (4, False)),
        (_args(dp_size=4, enable_dp_attention=True), (4, True)),
        (_attn_dp_args(4), (4, True)),
        (_attn_dp_args(1, dp_size=2), (2, False)),
        # Pre-#41818 SGLang declares attn_dp_size as a derived-field descriptor.
        (_args(dp_size=4, enable_dp_attention=True, attn_dp_size=object()), (4, True)),
        (_args(dp_size=2, attn_dp_size=object()), (2, False)),
        (SimpleNamespace(), (1, False)),
    ],
)
def test_sglang_dp_layout(server_args, expected):
    assert sglang_dp_layout(server_args) == expected


def test_attn_dp_size_exposes_every_local_rank():
    server_args = _attn_dp_args(2)

    assert local_dp_rank_bounds(server_args) == (0, 2)
    assert model_card_dp_rank_bounds(server_args) == (0, 2)
    assert publishes_kv_events(server_args) is True


def test_attn_dp_size_multinode_slices_by_node():
    nodes = [_attn_dp_args(8, nnodes=2, node_rank=rank) for rank in (0, 1)]

    assert [local_dp_rank_bounds(node) for node in nodes] == [(0, 4), (4, 8)]
    assert all(model_card_dp_rank_bounds(node) == (0, 8) for node in nodes)
    assert all(publishes_kv_events(node) is True for node in nodes)


def test_attn_dp_size_splits_global_max_running_requests():
    server_args = _attn_dp_args(2, max_running_requests=320)

    assert per_rank_max_running_requests(server_args) == 160
