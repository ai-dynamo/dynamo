# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from dynamo.profiler.utils.profile_prefill import (
    _profile_prefill_helper,
    profile_prefill,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _run(tmp_path, ttfts):
    # max_context_length 4612 and granularity 4 sweep ISL 100, 1100, 2100, 3100.
    results = iter(ttfts)
    _profile_prefill_helper(
        str(tmp_path),
        num_gpus=1,
        max_context_length=4612,
        interpolation_granularity=4,
        get_ttft=lambda isl: next(results),
    )


@pytest.mark.parametrize(
    "ttfts",
    [
        [None, None, None, None],
        [10.0, 20.0, None, None],
    ],
    ids=["all-points-failed", "two-points"],
)
def test_prefill_profile_fails_with_fewer_than_three_points(tmp_path, ttfts):
    with pytest.raises(RuntimeError, match="need at least 3"):
        _run(tmp_path, ttfts)

    assert not (tmp_path / "raw_data.npz").exists()


def test_prefill_profile_saves_data_with_three_points(tmp_path):
    _run(tmp_path, [10.0, 20.0, 30.0, None])

    assert (tmp_path / "raw_data.npz").exists()


def test_profile_prefill_fails_when_every_aiperf_run_exits_nonzero(tmp_path):
    aiperf = MagicMock(returncode=1)
    aiperf.communicate.return_value = ("", "endpoint returned 503")

    with patch(
        "dynamo.profiler.utils.aiperf.subprocess.Popen", return_value=aiperf
    ) as popen:
        with pytest.raises(RuntimeError, match="measured 0 TTFT data points"):
            profile_prefill(
                str(tmp_path),
                "test-model",
                "test-tokenizer",
                "http://mock:8000",
                num_gpus=1,
                max_context_length=4612,
                interpolation_granularity=4,
            )

    # Every ISL point was attempted before the sweep failed.
    assert popen.call_count == 4
    assert not (tmp_path / "raw_data.npz").exists()
