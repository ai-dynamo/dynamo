# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from dynamo.planner.core.throughput.interpolation import EncodeInterpolator

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _profile_dir() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "profiling_results",
        "H200_encode_TP1",
    )


def test_encode_interpolator_loads_json_fixture():
    interpolator = EncodeInterpolator(_profile_dir())
    assert interpolator.interpolate_thpt_per_gpu(2.0) == pytest.approx(2.0)


def test_encode_interpolator_clamps_lower_bound():
    interpolator = EncodeInterpolator(_profile_dir())
    assert interpolator.interpolate_thpt_per_gpu(0.1) == pytest.approx(1.0)


def test_encode_interpolator_clamps_upper_bound():
    interpolator = EncodeInterpolator(_profile_dir())
    assert interpolator.interpolate_thpt_per_gpu(10.0) == pytest.approx(2.5)


def test_encode_interpolator_missing_files_raise_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Encode interpolation files not found"):
        EncodeInterpolator(str(tmp_path))


@pytest.mark.parametrize(
    "raw_data,error_match",
    [
        (
            {"encode_request_rate": [], "encode_thpt_per_gpu": []},
            "cannot be empty",
        ),
        (
            {"encode_request_rate": [0.5], "encode_thpt_per_gpu": [1.0]},
            "at least two samples",
        ),
        (
            {"encode_request_rate": [-1.0, 1.0], "encode_thpt_per_gpu": [1.0, 2.0]},
            "must be non-negative",
        ),
        (
            {"encode_request_rate": [1.0, 2.0], "encode_thpt_per_gpu": [1.0, -2.0]},
            "must be positive",
        ),
    ],
)
def test_encode_interpolator_rejects_invalid_data(raw_data, error_match):
    with pytest.raises(ValueError, match=error_match):
        EncodeInterpolator(raw_data=raw_data)
