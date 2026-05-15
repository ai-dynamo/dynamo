# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``dynamo.common.backend.sglang_logprobs``."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from dynamo.common.backend.sglang_logprobs import build_logprob_kwargs, extract_logprobs

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@mock.patch.dict(os.environ, {"DYN_SGL_ALLOW_TOP_LOGPROBS": "1"})
def test_build_logprob_kwargs_maps_logprobs_and_prompt_logprobs():
    kwargs = build_logprob_kwargs(
        {"output_options": {"logprobs": 2, "prompt_logprobs": 5}}
    )
    assert kwargs == {
        "return_logprob": True,
        "top_logprobs_num": 5,
        "logprob_start_len": 0,
    }


def test_build_logprob_kwargs_gates_top_logprobs_when_env_unset():
    with mock.patch.dict(os.environ, {"DYN_SGL_ALLOW_TOP_LOGPROBS": ""}, clear=False):
        with pytest.raises(ValueError, match="DYN_SGL_ALLOW_TOP_LOGPROBS"):
            build_logprob_kwargs({"output_options": {"logprobs": 5}})


def test_extract_logprobs_slices_cumulative_meta_info_and_threads_offset():
    meta = {
        "output_token_logprobs": [
            (-0.1, 11, "a"),
            (-0.2, 22, "b"),
            (-0.3, 33, "c"),
        ],
        "output_top_logprobs": [
            [(-0.1, 11, "a"), (-0.5, 99, "x")],
            [(-0.2, 22, "b"), (-0.7, 98, "y")],
            [(-0.3, 33, "c"), (-0.9, 97, "z")],
        ],
    }
    log, top, new_total = extract_logprobs(meta, 1)
    assert log == [-0.2, -0.3]
    assert top is not None and top[0][0] == {
        "rank": 1,
        "token_id": 22,
        "token": "b",
        "logprob": -0.2,
    }
    assert new_total == 3
