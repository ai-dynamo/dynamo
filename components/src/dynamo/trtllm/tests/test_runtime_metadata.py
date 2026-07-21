# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.utils.trtllm_utils import (
    get_spec_decode_runtime_data,
    per_dp_rank_max_num_seqs,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("max_batch_size", "data_parallel_size", "expected"),
    [
        (37, 1, 37),
        (36, 4, 9),
        (37, 4, 9),
        (None, 1, None),
        (0, 1, None),
        (-1, 1, None),
        (True, 1, None),
        ("37", 1, None),
        (37, None, None),
        (37, 0, None),
        (37, -1, None),
        (37, True, None),
        (37, "4", None),
        (3, 4, None),
    ],
)
def test_max_num_seqs_is_reported_per_dp_rank(
    max_batch_size, data_parallel_size, expected
):
    engine = SimpleNamespace(
        llm=SimpleNamespace(args=SimpleNamespace(max_batch_size=max_batch_size))
    )

    result = per_dp_rank_max_num_seqs(engine, data_parallel_size)

    assert result == expected
    if result is not None:
        assert result * data_parallel_size <= max_batch_size


def test_encode_registration_without_local_scheduler_reports_no_max_num_seqs():
    engine = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.ENCODE,
        encoder_available=False,
    )

    assert per_dp_rank_max_num_seqs(engine, 1) is None


def test_per_dp_rank_max_num_seqs_fails_if_engine_contract_changes():
    with pytest.raises(AttributeError):
        per_dp_rank_max_num_seqs(SimpleNamespace(), 1)


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
