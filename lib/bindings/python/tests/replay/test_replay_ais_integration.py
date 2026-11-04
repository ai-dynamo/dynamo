# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, network-free coverage of Dynamo replay against the pinned AISimulate release."""

from __future__ import annotations

import asyncio
import json

import pytest

from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.aiconfigurator,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.mocker,
    pytest.mark.pre_merge,
]

AIS_MODEL = "Qwen/Qwen3-32B"
AIS_SYSTEM = "h200_sxm"
AIS_BACKEND_VERSION = "current"


@pytest.fixture(autouse=True)
def _offline_ais(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _engine_args(worker_type: str | None = None):
    from dynamo.mocker.config import normalize_mocker_config

    return normalize_mocker_config(
        {
            "engine": {
                "backend": "vllm",
                "worker_type": worker_type or "aggregated",
                "block_size": 64,
                "max_num_batched_tokens": 4096,
                "max_num_seqs": 128,
                "timing_model": {
                    "type": "external",
                    "provider": "ais",
                    "config": {
                        "model": AIS_MODEL,
                        "system": AIS_SYSTEM,
                        "backend": "vllm",
                        "backend_version": AIS_BACKEND_VERSION,
                        "worker_type": worker_type or "aggregated",
                    },
                },
            }
        }
    )


def test_real_ais_memory_estimates_gpu_blocks() -> None:
    from aisimulate_core.sdk.memory import estimate_num_gpu_blocks

    blocks = estimate_num_gpu_blocks(
        model_path=AIS_MODEL,
        system=AIS_SYSTEM,
        backend="vllm",
        backend_version=AIS_BACKEND_VERSION,
        tp_size=1,
        scheduler_block_size=64,
        max_num_tokens=4096,
        max_batch_size=128,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
    )
    assert blocks > 0


def test_default_ais_capacity_uses_queryable_version() -> None:
    args = _engine_args()
    assert args["engine"]["num_gpu_blocks"] > 0
    assert args["engine"]["timing_model"]["config"]["backend_version"] == "current"


def test_aggregated_replay_uses_native_ais_engine() -> None:
    from aisimulate_core.sdk.engine import compile_engine

    from dynamo.replay import run_synthetic_trace_replay

    assert callable(compile_engine)
    report = run_synthetic_trace_replay(
        128,
        8,
        2,
        extra_engine_args=_engine_args(),
        replay_concurrency=1,
        replay_mode="offline",
    )
    report = report.summary
    assert report["num_requests"] == 2
    assert report["mean_ttft_ms"] > 0.0
    assert report["mean_tpot_ms"] > 0.0


def test_disaggregated_replay_uses_native_ais_engine() -> None:
    from dynamo.replay import run_synthetic_trace_replay

    report = run_synthetic_trace_replay(
        128,
        8,
        2,
        prefill_engine_args=_engine_args("prefill"),
        decode_engine_args=_engine_args("decode"),
        num_prefill_workers=1,
        num_decode_workers=1,
        replay_concurrency=1,
        replay_mode="offline",
    )
    report = report.summary
    assert report["num_requests"] == 2
    assert report["mean_ttft_ms"] > 0.0
    assert report["mean_tpot_ms"] > 0.0


@pytest.mark.parametrize(
    "input_kind,policy",
    [("mapping", "off"), ("json", "balanced"), ("external_json", None)],
)
def test_canonical_python_presets_survive_mocker_round_trip_and_replay(
    input_kind, policy
) -> None:
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    from dynamo._core import AisPerfConfig, run_mocker_synthetic_trace_replay
    from dynamo.mocker.config import normalize_mocker_config

    # Load packaged model metadata/performance tables only, without model weights.
    payload = {
        "model": AIS_MODEL,
        "system": AIS_SYSTEM,
        "backend": "vllm",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        "transfer_policy": policy,
        "systems_paths": ["default"],
    }
    expected = ForwardPassPerfModelConfig(**payload).to_dict()
    assert AisPerfConfig(payload).to_dict() == expected
    raw = {
        "engine": {
            "num_gpu_blocks": 1000,
            "timing_model": {
                "type": "external",
                "provider": "aic" if input_kind == "external_json" else "ais",
                "config": payload,
            },
        }
    }
    args = normalize_mocker_config(raw if input_kind == "mapping" else json.dumps(raw))
    assert args["engine"]["timing_model"]["config"] == expected
    args["engine"]["num_gpu_blocks"] = 1001
    restored = normalize_mocker_config(json.dumps(args))
    assert restored["engine"]["timing_model"]["config"] == expected
    assert restored["engine"]["num_gpu_blocks"] == 1001
    report = run_mocker_synthetic_trace_replay(
        128, 2, 1, extra_engine_args=restored, replay_concurrency=1
    ).summary
    assert report["num_requests"] == 1
    assert report["mean_ttft_ms"] > 0
    assert report["mean_tpot_ms"] > 0


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.timeout(30)
@pytest.mark.parametrize("num_gpu_blocks", [None, 1000])
async def test_live_mocker_file_normalizes_canonical_python_presets(
    tmp_path, num_gpu_blocks
) -> None:
    from dynamo._core import EngineType, EntrypointArgs, make_engine
    from dynamo.mocker.config import normalize_mocker_config

    payload = {
        "engine": {
            "timing_model": {
                "type": "external",
                "provider": "ais",
                "config": {
                    "model": AIS_MODEL,
                    "system": AIS_SYSTEM,
                    "backend": "vllm",
                    "worker_type": "aggregated",
                    "estimation_mode": "op_level",
                    "transfer_policy": "balanced",
                    "systems_paths": ["default"],
                },
            }
        }
    }
    if num_gpu_blocks is not None:
        payload["engine"]["num_gpu_blocks"] = num_gpu_blocks
    path = tmp_path / "mocker.json"
    path.write_text(json.dumps(payload))
    parsed = normalize_mocker_config(path.read_text())
    assert parsed["engine"]["num_gpu_blocks"] > 0
    if num_gpu_blocks is not None:
        assert parsed["engine"]["num_gpu_blocks"] == num_gpu_blocks
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    try:
        engine = await make_engine(
            runtime,
            EntrypointArgs(
                engine_type=EngineType.Mocker,
                model_name="ais-file-config-test",
                extra_engine_args=str(path),
            ),
        )
        assert engine is not None
    finally:
        runtime.shutdown()
