# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.backend.logprobs import (
    extract_from_completion_output,
    validate_sampled_logprobs,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "entries",
    [
        [{2: SimpleNamespace(logprob=-1.0)}],
        [{1: SimpleNamespace(logprob=float("nan"))}],
        [{1: SimpleNamespace(logprob=float("-inf"))}],
        [],
        [None],
    ],
)
def test_strict_selected_logprob_rejects_invalid_output(entries):
    output = SimpleNamespace(token_ids=[1], logprobs=entries)
    with pytest.raises(ValueError):
        validate_sampled_logprobs(output)


def test_strict_selected_logprob_requires_count_alignment():
    output = SimpleNamespace(
        token_ids=[1, 2], logprobs=[{1: SimpleNamespace(logprob=-0.2)}]
    )
    with pytest.raises(ValueError):
        validate_sampled_logprobs(output)


def test_strict_selected_logprob_keeps_zero_and_missing_unrequested_output():
    output = SimpleNamespace(
        token_ids=[1], logprobs=[{1: SimpleNamespace(logprob=0.0)}]
    )
    validate_sampled_logprobs(output)
    assert extract_from_completion_output(output, 0)[0] == [0.0]
    assert extract_from_completion_output(
        SimpleNamespace(token_ids=[1], logprobs=None), 0
    ) == (None, None)


def test_legacy_fallback_remains_unchanged():
    output = SimpleNamespace(
        token_ids=[1], logprobs=[{2: SimpleNamespace(logprob=-0.5)}]
    )
    assert extract_from_completion_output(output, 0, fallback_to_first_on_missing=True)[
        0
    ] == [-0.5]
