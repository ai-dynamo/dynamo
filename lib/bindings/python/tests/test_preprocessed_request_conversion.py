# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoutedEngine.generate's direct read of token_ids and extra_args equals pythonize's."""

import json

import pytest

from dynamo._core import _convert_preprocessed_request

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def request(token_ids=(1, 2, 3), **fields):
    return {"model": "m", "token_ids": list(token_ids), **fields}


class TokenList(list):
    pass


ACCEPTED = [
    request(),
    request(token_ids=[]),
    request(extra_args=None),
    request(
        stop_conditions={"max_tokens": 8},
        sampling_options={"temperature": 0.5},
        annotations=["nemo_gym:rollout"],
        extra_args={
            "nemo_rl": {
                "template_token_ids": [7, 8, 9],
                "request": {"max_tokens": None, "tools": [{"type": "function"}]},
                "splice": None,
                "verify": False,
            }
        },
    ),
    request(
        extra_args={"ints": [0, -1, -(2**63), 2**63 - 1, 2**63, 2**64 - 1]}
    ),
    request(extra_args={"floats": [1.5, -0.0, float("nan"), float("inf")]}),
    request(extra_args={"z": (1, "x"), "a": [], "m": {}, "s": "héllo", "b": True}),
]

DECLINED = [
    {"model": "m", "token_ids": TokenList([1, 2])},
    request(token_ids=[True, 2]),
    request(extra_args={"big": 2**64}),
    request(extra_args={"$serde_json::private::RawValue": "[1]"}),
]


@pytest.mark.parametrize("preprocessed", ACCEPTED)
def test_direct_read_matches_pythonize(preprocessed):
    fast = _convert_preprocessed_request(preprocessed, fast_path=True)
    assert fast is not None
    slow = _convert_preprocessed_request(preprocessed, fast_path=False)
    # JSON text also pins key order and NaN -> null.
    assert json.dumps(fast) == json.dumps(slow)


@pytest.mark.parametrize("preprocessed", DECLINED)
def test_direct_read_declines_to_pythonize(preprocessed):
    assert _convert_preprocessed_request(preprocessed, fast_path=True) is None
