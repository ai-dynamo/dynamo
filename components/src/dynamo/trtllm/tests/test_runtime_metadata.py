# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.trtllm.utils.trtllm_utils import (
    engine_max_num_seqs,
    get_spec_decode_runtime_data,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_engine_max_num_seqs_uses_post_init_engine_value():
    engine = SimpleNamespace(
        llm=SimpleNamespace(args=SimpleNamespace(max_batch_size=37))
    )

    assert engine_max_num_seqs(engine) == 37


@pytest.mark.parametrize("value", [None, 0, -1, True, "37"])
def test_engine_max_num_seqs_ignores_invalid_values(value):
    engine = SimpleNamespace(
        llm=SimpleNamespace(args=SimpleNamespace(max_batch_size=value))
    )

    assert engine_max_num_seqs(engine) is None


def test_spec_decode_runtime_data_uses_max_draft_len():
    engine_args = {
        "speculative_config": {
            "max_draft_len": "6",
            "num_nextn_predict_layers": 99,
            "decoding_type": "EAGLE",
        }
    }

    assert get_spec_decode_runtime_data(engine_args) == {
        "nextn": 6,
        "method": "EAGLE",
        "source": "backend_config",
    }


def test_spec_decode_runtime_data_falls_back_to_num_nextn_predict_layers():
    engine_args = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_nextn_predict_layers=2,
            decoding_type="NEXTN",
        )
    )

    assert get_spec_decode_runtime_data(engine_args) == {
        "nextn": 2,
        "method": "NEXTN",
        "source": "backend_config",
    }


@pytest.mark.parametrize(
    "speculative_config",
    [
        None,
        {},
        {"max_draft_len": 0},
        {"max_draft_len": "bad"},
        {"num_nextn_predict_layers": 0},
    ],
)
def test_spec_decode_runtime_data_ignores_invalid_nextn(speculative_config):
    engine_args = {"speculative_config": speculative_config}

    assert get_spec_decode_runtime_data(engine_args) is None
