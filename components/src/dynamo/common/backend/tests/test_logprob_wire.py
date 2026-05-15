# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared logprob wire-builder."""

from __future__ import annotations

import pytest

from dynamo.common.backend.logprob_wire import TopLogprob, build_chunk

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_build_chunk_renders_top_logprobs_with_optional_bytes():
    log_probs, top = build_chunk(
        selected=[-0.1, -0.2],
        top_per_position=[
            [
                TopLogprob(rank=1, token_id=11, token="a", logprob=-0.1, bytes_=[97]),
                TopLogprob(rank=2, token_id=22, token=None, logprob=-0.5),
            ],
            [TopLogprob(rank=1, token_id=33, token="b", logprob=-0.2)],
        ],
    )
    assert log_probs == [-0.1, -0.2]
    assert top == [
        [
            {"rank": 1, "token_id": 11, "token": "a", "logprob": -0.1, "bytes": [97]},
            {"rank": 2, "token_id": 22, "token": None, "logprob": -0.5},
        ],
        [{"rank": 1, "token_id": 33, "token": "b", "logprob": -0.2}],
    ]


def test_build_chunk_empty_selected_returns_none():
    log_probs, top = build_chunk(selected=[], top_per_position=None)
    assert log_probs is None
    assert top is None
