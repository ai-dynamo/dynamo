# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import UserDict
from types import MappingProxyType

import pytest

from dynamo._core import EngineType, EntrypointArgs, run_mocker_synthetic_trace_replay
from dynamo._internal import ais
from dynamo.mocker.config import normalize_mocker_config

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.mark.parametrize("mapping_type", [MappingProxyType, UserDict])
def test_mapping_inputs_reach_native_mocker_entrypoints(mapping_type):
    raw = {"engine": {"num_gpu_blocks": 64}}
    config = mapping_type(raw)
    assert normalize_mocker_config(config) == normalize_mocker_config(raw)
    EntrypointArgs(engine_type=EngineType.Mocker, mocker_engine_args=config)
    report = run_mocker_synthetic_trace_replay(
        32, 2, 1, extra_engine_args=config, replay_concurrency=1
    )
    assert report.summary["completed_requests"] == 1
    assert raw == {"engine": {"num_gpu_blocks": 64}}


def test_mapping_input_does_not_accept_a_sequence_of_pairs():
    with pytest.raises(TypeError):
        normalize_mocker_config([("engine", {"num_gpu_blocks": 64})])


def _config(**overrides):
    return {
        "model": "Qwen/Qwen3-32B",
        "system": "h200_sxm",
        "backend": "vllm",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        **overrides,
    }


def test_automatic_capacity_roundtrip_and_unbounded_limits(monkeypatch):
    calls = []

    def estimate(config, **options):
        calls.append(options)
        return 3904

    monkeypatch.setattr(ais, "estimate_canonical_num_gpu_blocks", estimate)
    raw = {
        "engine": {
            "max_num_batched_tokens": None,
            "max_num_seqs": None,
            "timing_model": {
                "type": "external",
                "provider": "ais",
                "config": _config(),
            },
        }
    }
    original = normalize_mocker_config(raw)
    restored = normalize_mocker_config(json.dumps(original))
    assert restored == original
    assert restored["engine"]["num_gpu_blocks"] == 3904
    assert restored["num_gpu_blocks_is_explicit"] is False
    assert len(calls) == 2
    assert all(
        "max_num_batched_tokens" not in call and "max_num_seqs" not in call
        for call in calls
    )
    for config in (original, restored):
        report = run_mocker_synthetic_trace_replay(
            32,
            2,
            1,
            extra_engine_args=config,
            replay_concurrency=1,
            capture_telemetry=True,
        )
        assert report.summary["completed_requests"] == 1
        assert (
            report.telemetry["samples"][0]["decode_scheduler_metrics"][0][
                "total_blocks"
            ]
            == 3904
        )


def test_explicit_capacity_skips_estimation_and_preserves_controls(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("explicit capacity must not be estimated")

    monkeypatch.setattr(ais, "estimate_canonical_num_gpu_blocks", unexpected)
    perf = _config(estimator_config={"correction": {"enabled": False}})
    raw = {
        "engine": {
            "num_gpu_blocks": 1000,
            "timing_model": {"type": "external", "provider": "ais", "config": perf},
        }
    }
    config = normalize_mocker_config(raw)
    assert config["num_gpu_blocks_is_explicit"] is True
    assert (
        config["engine"]["timing_model"]["config"]["estimator_config"]
        == perf["estimator_config"]
    )
    assert normalize_mocker_config(json.dumps(config)) == config


@pytest.mark.parametrize(
    "field,value", [("dp_size", 0), ("gpu_memory_utilization", 1.1)]
)
def test_validation_errors_preserve_field_paths(field, value):
    with pytest.raises(ValueError, match=field):
        normalize_mocker_config({field: value})
    with pytest.raises(ValueError, match="num_gpu_blocks"):
        normalize_mocker_config({"engine": {"num_gpu_blocks": "bad"}})
