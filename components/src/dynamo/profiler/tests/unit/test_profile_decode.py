# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import numpy as np
import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.monitoring.perf_metrics import fetch_pre_deployment_metrics
from dynamo.planner.monitoring.worker_info import WorkerInfo
from dynamo.profiler.utils import profile_decode
from dynamo.profiler.utils.dgd_generation import add_profile_data_to_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize("sparse", [False, True], ids=["full", "collinear"])
def test_decode_profile_data_reaches_planner(tmp_path, monkeypatch, sparse):
    def measure_decode(isl, _osl, concurrency, *args, **kwargs):
        if sparse and isl != 100:
            return None, None
        return 5.0 + concurrency / 100, 125.0

    monkeypatch.setattr(
        profile_decode, "get_decode_itl_and_thpt_per_gpu", measure_decode
    )
    decode_dir = tmp_path / "selected_decode_interpolation"
    decode_dir.mkdir()
    profile_decode.profile_decode(
        work_dir=str(decode_dir),
        model_name="model",
        tokenizer="tokenizer",
        url="http://unused",
        num_gpus=1,
        max_kv_tokens=100_000,
        max_context_length=2_048,
        interpolation_granularity=6,
        attention_dp_size=1,
    )

    config_map = add_profile_data_to_config({"spec": {"components": []}}, str(tmp_path))
    assert config_map is not None
    mount_dir = tmp_path / "mounted"
    mount_dir.mkdir()
    (mount_dir / "decode_raw_data.json").write_text(
        config_map["data"]["decode_raw_data.json"], encoding="utf-8"
    )
    fpms = asyncio.run(
        fetch_pre_deployment_metrics(
            runtime=None,
            namespace="test",
            worker_info=WorkerInfo(),
            profile_results_dir=str(mount_dir),
            component_type=SubComponentType.DECODE,
        )
    )
    with np.load(decode_dir / "raw_data.npz") as data:
        assert len(fpms) == len(data["z_itl"]) == (6 if sparse else 36)
        assert [fpm.wall_time for fpm in fpms] == pytest.approx(data["z_itl"] / 1000)
    for filename in (
        "decode_itl_interpolation.png",
        "decode_throughput_interpolation.png",
    ):
        assert (decode_dir / filename).exists() == (not sparse)


def test_decode_profiling_fails_without_valid_measurements(tmp_path):
    with pytest.raises(
        RuntimeError, match="Decode profiling produced no valid measurements"
    ):
        profile_decode._profile_decode_helper(
            work_dir=str(tmp_path),
            num_gpus=1,
            max_kv_tokens=10_000,
            max_context_length=2_048,
            interpolation_granularity=6,
            get_itl_and_thpt_per_gpu=lambda *_: (None, None),
            attention_dp_size=1,
        )
