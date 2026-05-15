# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for vLLM's ``extract_logprobs`` wire shape."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Skip the whole module outside the vLLM CI image — `dynamo.vllm.handlers`
# imports torch + vllm at load.
pytest.importorskip("torch")
pytest.importorskip("vllm")

from dynamo.vllm.handlers import extract_logprobs  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def _logprob(logprob: float, rank: int, decoded_token: str | None) -> SimpleNamespace:
    return SimpleNamespace(logprob=logprob, rank=rank, decoded_token=decoded_token)


def test_extract_logprobs_emits_dynamo_wire_shape():
    output = SimpleNamespace(
        token_ids=[11, 22],
        logprobs=[
            {11: _logprob(-0.1, 1, "a"), 99: _logprob(-0.5, 2, "x")},
            {22: _logprob(-0.2, 1, "b"), 98: _logprob(-0.7, 2, "y")},
        ],
    )
    log_probs, top_logprobs = extract_logprobs(output, num_output_tokens_so_far=0)
    assert log_probs == [-0.1, -0.2]
    assert top_logprobs is not None and len(top_logprobs) == 2
    assert top_logprobs[0][0] == {
        "rank": 1,
        "token_id": 11,
        "token": "a",
        "logprob": -0.1,
        "bytes": list(b"a"),
    }
    assert top_logprobs[1][0]["token_id"] == 22


def test_extract_logprobs_returns_none_when_engine_omitted_logprobs():
    output = SimpleNamespace(token_ids=[11], logprobs=None)
    assert extract_logprobs(output, num_output_tokens_so_far=0) == (None, None)


def test_extract_logprobs_slices_by_num_output_tokens_so_far():
    output = SimpleNamespace(
        token_ids=[11, 22, 33],
        logprobs=[
            {11: _logprob(-0.1, 1, "a")},
            {22: _logprob(-0.2, 1, "b")},
            {33: _logprob(-0.3, 1, "c")},
        ],
    )
    log_probs, top_logprobs = extract_logprobs(output, num_output_tokens_so_far=2)
    assert log_probs == [-0.3]
    assert top_logprobs is not None and len(top_logprobs) == 1
    assert top_logprobs[0][0]["token_id"] == 33
