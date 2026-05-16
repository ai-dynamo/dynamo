# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for TRT-LLM's ``extract_logprobs`` wire shape (covers
both the dict and the float-list fallback paths)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "tensorrt_llm import requires CUDA; skipping under -m gpu_0 collection.",
        allow_module_level=True,
    )
pytest.importorskip("tensorrt_llm")

from dynamo.trtllm.request_handlers.handler_base import extract_logprobs  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def _logprob(
    logprob: float, rank: int, decoded_token: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(logprob=logprob, rank=rank, decoded_token=decoded_token)


def test_extract_logprobs_dict_path_emits_dynamo_wire_shape():
    output = SimpleNamespace(
        token_ids=[11, 22],
        logprobs=[
            {11: _logprob(-0.1, 1, "a"), 99: _logprob(-0.5, 2, "x")},
            {22: _logprob(-0.2, 1, "b")},
        ],
    )
    log_probs, top_logprobs = extract_logprobs(output, num_output_tokens_so_far=0)
    assert log_probs == [-0.1, -0.2]
    assert top_logprobs is not None and len(top_logprobs) == 2
    # TRT-LLM does not populate UTF-8 bytes; the wire dict omits "bytes".
    assert top_logprobs[0][0] == {
        "rank": 1,
        "token_id": 11,
        "token": "a",
        "logprob": -0.1,
    }


def test_extract_logprobs_float_list_fallback():
    output = SimpleNamespace(token_ids=[11, 22], logprobs=[-0.1, -0.2])
    log_probs, top_logprobs = extract_logprobs(output, num_output_tokens_so_far=0)
    assert log_probs == [-0.1, -0.2]
    assert top_logprobs is None


def test_extract_logprobs_returns_none_when_engine_omitted_logprobs():
    output = SimpleNamespace(token_ids=[11], logprobs=None)
    assert extract_logprobs(output, num_output_tokens_so_far=0) == (None, None)
