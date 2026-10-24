# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.deepinfer.args import parse_args

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_defaults_describe_a_single_attention_dp_rank():
    config = parse_args(["--model", "m"])

    assert config.model == "m"
    assert config.served_model_name == "m"
    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
    assert config.context_parallel_size % config.expert_parallel_size == 0
    assert config.endpoint == "generate"
    assert config.component == "backend"


def test_served_model_name_overrides_model():
    config = parse_args(["--model", "m", "--served-model-name", "alias"])

    assert config.served_model_name == "alias"


def test_cp_must_be_a_multiple_of_ep():
    with pytest.raises(ValueError, match="multiple of"):
        parse_args(
            [
                "--model",
                "m",
                "--expert-parallel-size",
                "16",
                "--context-parallel-size",
                "8",
            ]
        )


@pytest.mark.parametrize(
    "flag", ["--max-model-len", "--max-num-seqs", "--expert-parallel-size"]
)
def test_sizes_must_be_positive(flag):
    with pytest.raises(ValueError, match=flag):
        parse_args(["--model", "m", flag, "0"])


def test_gpu_memory_utilization_must_be_a_fraction():
    with pytest.raises(ValueError, match="gpu-memory-utilization"):
        parse_args(["--model", "m", "--gpu-memory-utilization", "1.5"])
