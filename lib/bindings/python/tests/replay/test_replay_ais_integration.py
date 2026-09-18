# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, network-free coverage of Dynamo replay against the pinned AISimulate release."""

from __future__ import annotations

import json

import pytest

from dynamo.replay.config import load_engine_args

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
    from dynamo.mocker import MockEngineArgs

    worker_options = {"worker_type": worker_type} if worker_type is not None else {}
    return MockEngineArgs(
        ais_perf_config={
            "model": AIS_MODEL,
            "system": AIS_SYSTEM,
            "backend": "vllm",
            "backend_version": AIS_BACKEND_VERSION,
            "worker_type": worker_type or "aggregated",
        },
        block_size=64,
        max_num_batched_tokens=4096,
        max_num_seqs=128,
        **worker_options,
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
    args = load_engine_args(
        {
            "ais_perf_config": {
                "model": AIS_MODEL,
                "system": AIS_SYSTEM,
                "backend": "vllm",
                "worker_type": "aggregated",
                "backend_version": "current",
            },
            "block_size": 64,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 128,
        }
    )
    assert args is not None
    assert args.num_gpu_blocks > 0
    from aisimulate_core.sdk import perf_database

    assert args.ais_backend_version == perf_database.resolve_query_version(
        AIS_SYSTEM, "vllm", "current"
    )


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

    from dynamo._core import (
        AisPerfConfig,
        MockEngineArgs,
        run_mocker_synthetic_trace_replay,
    )

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
    if input_kind == "mapping":
        args = MockEngineArgs(ais_perf_config=payload, num_gpu_blocks=1000)
    else:
        timing = (
            {"ais_perf_config": payload}
            if input_kind == "json"
            else {
                "timing_model": {
                    "type": "external",
                    "provider": "aic",
                    "config": payload,
                }
            }
        )
        args = MockEngineArgs.from_json(json.dumps({"num_gpu_blocks": 1000, **timing}))
    assert args.ais_perf_config == expected
    updated = args.with_overrides(num_gpu_blocks=1001)
    assert updated.ais_perf_config == expected
    restored = MockEngineArgs.from_json(
        json.dumps(
            {
                "ais_perf_config": updated.ais_perf_config,
                "num_gpu_blocks": updated.num_gpu_blocks,
            }
        )
    )
    assert restored.ais_perf_config == expected
    assert restored.num_gpu_blocks == 1001
    report = run_mocker_synthetic_trace_replay(
        128, 2, 1, extra_engine_args=restored, replay_concurrency=1
    ).summary
    assert report["num_requests"] == 1
    assert report["mean_ttft_ms"] > 0
    assert report["mean_tpot_ms"] > 0


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.parametrize(
    "discovery_backend,request_plane", [("mem", "tcp")], indirect=True
)
@pytest.mark.parametrize("num_gpu_blocks", [None, 1000])
async def test_live_mocker_file_normalizes_canonical_python_presets(
    runtime, tmp_path, num_gpu_blocks
) -> None:
    from dynamo._core import EngineType, EntrypointArgs, MockEngineArgs, make_engine

    payload = {
        "ais_perf_config": {
            "model": AIS_MODEL,
            "system": AIS_SYSTEM,
            "backend": "vllm",
            "worker_type": "aggregated",
            "estimation_mode": "op_level",
            "transfer_policy": "balanced",
            "systems_paths": ["default"],
        }
    }
    if num_gpu_blocks is not None:
        payload["num_gpu_blocks"] = num_gpu_blocks
    path = tmp_path / "mocker.json"
    path.write_text(json.dumps(payload))
    parsed = MockEngineArgs.from_json(path.read_text())
    assert parsed.num_gpu_blocks == (
        16384 if num_gpu_blocks is None else num_gpu_blocks
    )
    engine = await make_engine(
        runtime,
        EntrypointArgs(
            engine_type=EngineType.Mocker,
            model_name="ais-file-config-test",
            extra_engine_args=str(path),
        ),
    )
    assert engine is not None
