# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo._internal.ais as ais_helpers
from dynamo._core import MockEngineArgs, run_mocker_synthetic_trace_replay

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _config(**overrides):
    return {
        "model": "Qwen/Qwen3-32B",
        "system": "h200_sxm",
        "backend": "vllm",
        "worker_type": "aggregated",
        **overrides,
    }


def test_canonical_config_owns_identity_and_topology():
    args = MockEngineArgs(
        ais_perf_config=_config(tp=2, attention_dp=4), num_gpu_blocks=1000
    )
    assert args.ais_tp_size == 2
    assert args.dp_size == 4
    assert args.num_gpu_blocks == 1000
    assert args.ais_perf_config["attention_dp"] == 4
    with pytest.raises(Exception, match="conflicts"):
        MockEngineArgs(ais_perf_config=_config(attention_dp=4), dp_size=2)
    with pytest.raises(Exception, match="worker_type"):
        MockEngineArgs(ais_perf_config=_config(worker_type="decode"))


def test_removed_sdk_names_are_rejected():
    import dynamo.llm as llm

    assert not hasattr(llm, "AicPerfConfig")
    with pytest.raises(TypeError):
        MockEngineArgs(aic_backend="vllm")
    with pytest.raises(Exception, match="unknown field"):
        MockEngineArgs.from_json(json.dumps({"aic_backend": "vllm"}))


def test_direct_replay_preserves_capacity_errors(monkeypatch):
    def invalid_capacity(*_args, **_kwargs):
        raise ValueError("invalid capacity request")

    monkeypatch.setattr(
        ais_helpers, "estimate_canonical_num_gpu_blocks", invalid_capacity
    )
    args = MockEngineArgs(ais_perf_config=_config())
    with pytest.raises(Exception, match="invalid capacity request"):
        run_mocker_synthetic_trace_replay(
            32, 1, 1, extra_engine_args=args, replay_concurrency=1
        )


def test_invalid_json_num_gpu_blocks_type_is_rejected():
    with pytest.raises(Exception, match="num_gpu_blocks"):
        MockEngineArgs.from_json(
            json.dumps({"ais_perf_config": _config(), "num_gpu_blocks": "bad"})
        )


def test_memory_fraction_setters_validate_range():
    engine_args = MockEngineArgs(gpu_memory_utilization=0.8, mem_fraction_static=0.7)

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        engine_args.gpu_memory_utilization = 1.1
    assert engine_args.gpu_memory_utilization == 0.8

    with pytest.raises(ValueError, match="mem_fraction_static"):
        engine_args.mem_fraction_static = -0.1
    assert engine_args.mem_fraction_static == 0.7

    engine_args.gpu_memory_utilization = None
    engine_args.mem_fraction_static = None

    assert engine_args.gpu_memory_utilization is None
    assert engine_args.mem_fraction_static is None


def test_json_rejects_invalid_memory_fraction_types():
    with pytest.raises(Exception, match="gpu_memory_utilization"):
        MockEngineArgs.from_json(json.dumps({"gpu_memory_utilization": "bad"}))

    with pytest.raises(Exception, match="mem_fraction_static"):
        MockEngineArgs.from_json(json.dumps({"mem_fraction_static": {}}))
