# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused wiring tests for Match Config simulation and real backends."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autoscaling_arena import match_runner
from autoscaling_arena.match_config import (
    EvaluationConfig,
    MatchConfig,
    ReplicaCounts,
    SimBackendConfig,
    parse_match_config,
)


@pytest.mark.parametrize("telemetry_interval", [5.0, None])
def test_run_sim_item_passes_exact_native_shards_without_materializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    telemetry_interval: float | None,
):
    config = _sim_config(tmp_path, "agg")
    config = replace(
        config,
        backend=replace(
            config.backend,
            replay=replace(
                config.backend.replay,
                telemetry_sample_interval_s=telemetry_interval,
            ),
        ),
    )
    first = tmp_path / "recorded-02.jsonl.gz"
    second = tmp_path / "recorded-01.jsonl.gz"
    first.write_bytes(b"first-native-shard")
    second.write_bytes(b"second-native-shard")
    config = replace(
        config,
        evaluations=(
            EvaluationConfig(
                workload="recorded-native",
                seed=11,
                max_requests=None,
                arrival_speedup=1.0,
                trace_paths=(first.resolve(), second.resolve()),
                trace_format="dynamo",
                trace_block_size=None,
            ),
        ),
    )
    item = next(config.iter_runs())
    replay_calls: list[dict[str, Any]] = []

    report = SimpleNamespace(
        timeline=[],
        trace_report={
            "prefix_cache_reused_ratio": 0.25,
            "total_input_tokens": 400,
        },
        per_request=None,
        telemetry_artifact=(
            {
                "contract": "dynamo.replay.telemetry.v1",
                "sample_count": 7,
                "sha256": "a" * 64,
            }
            if telemetry_interval is not None
            else None
        ),
    )

    fake_sims = types.ModuleType("autoscaling_arena.runners.sims")
    fake_sims.run_arena_replay = lambda **kwargs: (
        replay_calls.append(kwargs) or report
    )
    monkeypatch.setitem(sys.modules, "autoscaling_arena.runners.sims", fake_sims)

    import autoscaling_arena.scorecard as scorecard_module

    monkeypatch.setattr(
        scorecard_module,
        "scorecard",
        lambda replay_report, **kwargs: {
            "gpu_hours": 1.0,
            "profiles": {kwargs["sla_profile"].name: {"goodput_rps": 2.0}},
        },
    )
    monkeypatch.setattr(
        match_runner,
        "_materialize_trace",
        lambda *args, **kwargs: pytest.fail(
            "exact native traces must not be materialized"
        ),
    )
    monkeypatch.setattr(
        match_runner,
        "_build_sim_factory",
        lambda autoscaler, *, topology: object(),
    )
    metadata_calls: list[dict[str, Any]] = []

    def fake_metadata(received_item, **kwargs):
        assert received_item is item
        metadata_calls.append(kwargs)
        return {"sla": {"name": item.sla}}

    monkeypatch.setattr(match_runner, "_evaluation_metadata", fake_metadata)

    context = match_runner._ExecutionContext(
        session_id="native-session",
        session_root=tmp_path / "native-session",
    )
    result = match_runner._run_sim_item(config, item, context)

    assert len(replay_calls) == 1
    replay = replay_calls[0]
    assert replay["trace_files"] == [str(first.resolve()), str(second.resolve())]
    assert replay["trace_format"] == "dynamo"
    assert replay["trace_block_size"] is None
    assert replay["arrival_speedup_ratio"] == 1.0
    assert replay["performance_model_metadata"] == {
        "aggregated": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "backend_version": "0.19.0",
                "system": "h200_sxm",
                "model_path": "openai/gpt-oss-120b",
                "tp_size": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "attention_dp_size": 1,
            },
        }
    }
    expected_telemetry = context.session_root / "runs" / item.run_id / "telemetry.jsonl"
    assert replay["telemetry_jsonl_path"] == (
        str(expected_telemetry) if telemetry_interval is not None else None
    )
    assert "trace_file" not in replay
    assert metadata_calls[0]["trace_path"] is None
    assert metadata_calls[0]["trace_paths"] == (
        first.resolve(),
        second.resolve(),
    )
    assert result["timeline_semantics"] == "arriving_and_completed_v2"
    assert result["cache"]["prefix_cache_reused_ratio"] == pytest.approx(0.25)
    assert result["cache"]["timeline_available"] is False
    assert result["runtime"]["replay"] == {
        "telemetry_sample_interval_s": telemetry_interval,
        "telemetry_contract": (
            "dynamo.replay.telemetry.v1" if telemetry_interval is not None else None
        ),
        "telemetry_sample_count": (7 if telemetry_interval is not None else None),
        "telemetry_sha256": ("a" * 64 if telemetry_interval is not None else None),
    }
    if telemetry_interval is not None:
        assert result["artifacts"]["telemetry"] == str(expected_telemetry)
        assert result["artifacts"]["telemetry_metadata"] == {
            "contract": "dynamo.replay.telemetry.v1",
            "sample_count": 7,
            "sha256": "a" * 64,
        }
    else:
        assert "telemetry" not in result["artifacts"]


def _sim_config(
    tmp_path: Path,
    topology: str,
    *,
    deployment: dict[str, Any] | None = None,
) -> MatchConfig:
    static_config: dict[str, Any] = {
        "num_decode": 3,
        "poll_interval_s": 2.5,
    }
    if topology == "disagg":
        static_config["num_prefill"] = 2
    backend: dict[str, Any] = {
        "type": "sim",
        "topology": topology,
        "gpu_budget": 32,
        "router": {"mode": "round_robin"},
        "planner_config": {"load_adjustment_interval_seconds": 9},
        "replay": {
            "ais_bootstrap": False,
            "concurrency": 7,
        },
        "autoscalers": [
            {
                "name": "fixed",
                "type": "static",
                "config": static_config,
            }
        ],
    }
    if deployment is None:
        backend["substrate"] = "gpt_oss"
        backend["replay"]["model_name"] = "configured-model"
    else:
        backend.update(deployment)
    return parse_match_config(
        {
            "schema_version": 1,
            "name": f"sim-{topology}-wiring",
            "backend": backend,
            "evaluations": {
                "workloads": ["flat"],
                "defaults": {
                    "seed": 11,
                    "max_requests": 4,
                    "arrival_speedup": 4.0,
                },
            },
            "slo_profiles": [
                {
                    "name": "evaluation-sla",
                    "ttft_ms": 111,
                    "itl_ms": 22,
                    "e2e_ms": 333,
                }
            ],
            "metrics": {
                "rank_by": "goodput_rps",
                "include": ["goodput_rps", "gpu_hours"],
            },
            "publish": {
                "artifact_root": "artifacts",
                "destinations": [{"type": "console"}],
            },
        },
        source_path=tmp_path / "match.yaml",
    )


def _run_sim_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: MatchConfig,
) -> SimpleNamespace:
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    item = next(config.iter_runs())

    trace_path = tmp_path / f"{backend.topology}-trace.jsonl"
    trace_path.write_text('{"timestamp": 0}\n')
    materialize_calls: list[dict[str, Any]] = []
    replay_calls: list[dict[str, Any]] = []
    scorecard_calls: list[dict[str, Any]] = []
    factory_sentinel = object()
    report_sentinel = object()

    def fake_materialize(context, **kwargs):
        del context
        materialize_calls.append(kwargs)
        return trace_path.resolve()

    def fake_replay(**kwargs):
        replay_calls.append(kwargs)
        return report_sentinel

    def fake_scorecard(report, **kwargs):
        assert report is report_sentinel
        scorecard_calls.append(kwargs)
        profile = kwargs["sla_profile"]
        return {
            "gpu_hours": 2.25,
            "profiles": {profile.name: {"goodput_rps": 8.5}},
        }

    fake_sims = types.ModuleType("autoscaling_arena.runners.sims")
    fake_sims.run_arena_replay = fake_replay  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoscaling_arena.runners.sims", fake_sims)

    import autoscaling_arena.scorecard as scorecard_module

    monkeypatch.setattr(scorecard_module, "scorecard", fake_scorecard)
    monkeypatch.setattr(
        match_runner,
        "get_workload",
        lambda name: SimpleNamespace(block_size=64),
    )
    monkeypatch.setattr(match_runner, "_materialize_trace", fake_materialize)
    monkeypatch.setattr(
        match_runner,
        "_build_sim_factory",
        lambda autoscaler, *, topology: factory_sentinel,
    )

    context = match_runner._ExecutionContext(
        session_id="sim-session",
        session_root=tmp_path / "sim-session",
    )
    result = match_runner._run_sim_item(config, item, context)

    return SimpleNamespace(
        item=item,
        result=result,
        trace_path=trace_path.resolve(),
        materialize_calls=materialize_calls,
        replay_calls=replay_calls,
        scorecard_calls=scorecard_calls,
        factory=factory_sentinel,
    )


def _assert_common_sim_wiring(
    run: SimpleNamespace,
    *,
    expected_model_name: str,
) -> dict[str, Any]:
    item = run.item
    assert run.materialize_calls == [
        {
            "workload_name": "flat",
            "seed": 11,
            "max_requests": 4,
            "arrival_speedup": 1.0,
            "speedup_is_materialized": False,
        }
    ]
    assert len(run.replay_calls) == 1
    replay_kwargs = run.replay_calls[0]
    assert replay_kwargs["trace_file"] == str(run.trace_path)
    assert replay_kwargs["autoscaler"] is run.factory
    assert replay_kwargs["arrival_speedup_ratio"] == 4.0
    assert replay_kwargs["trace_block_size"] == 64
    assert replay_kwargs["router_mode"] == "round_robin"
    assert replay_kwargs["replay_concurrency"] == 7
    assert replay_kwargs["model_name"] == expected_model_name
    assert replay_kwargs["ais_bootstrap"] is False
    assert replay_kwargs["sla_ttft_ms"] == 111.0
    assert replay_kwargs["sla_itl_ms"] == 22.0
    assert replay_kwargs["sla_e2e_ms"] == 333.0

    assert len(run.scorecard_calls) == 1
    scored_profile = run.scorecard_calls[0]["sla_profile"]
    assert run.scorecard_calls[0]["profiles"] == (scored_profile,)
    assert scored_profile.name == "evaluation-sla"
    assert scored_profile.ttft_ms == 111.0
    assert scored_profile.itl_ms == 22.0
    assert scored_profile.e2e_ms == 333.0
    assert run.result["status"] == "ok"
    assert run.result["metrics"] == {
        "goodput_rps": 8.5,
        "gpu_hours": 2.25,
    }
    assert run.result["evaluation"]["sla"] == {
        "name": "evaluation-sla",
        "source_name": "evaluation-sla",
        "ttft_ms": 111.0,
        "itl_ms": 22.0,
        "e2e_ms": 333.0,
    }
    assert run.result["run_id"] == item.run_id
    return replay_kwargs


def _assert_engine_provenance_matches_replay(
    result: dict[str, Any],
    *,
    role: str,
    rendered_args: str,
    num_gpus: int,
) -> None:
    metadata = result["runtime"]["engines"][role]
    assert metadata["config"]["num_gpus"] == num_gpus
    assert (
        metadata["rendered_args_sha256"]
        == hashlib.sha256(rendered_args.encode()).hexdigest()
    )


def test_run_sim_item_wires_legacy_preset_model_engine_speedup_and_sla(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _sim_config(tmp_path, "disagg")
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.autoscalers[0].start == ReplicaCounts(prefill=2, decode=3)

    run = _run_sim_with_fakes(tmp_path, monkeypatch, config)
    replay_kwargs = _assert_common_sim_wiring(
        run, expected_model_name="configured-model"
    )

    expected_engine_args = {
        "engine_type": "vllm",
        "tensor_parallel_size": 1,
        "dp_size": 1,
        "ais_perf_config": {
            "backend": "vllm",
            "system": "h200_sxm",
            "model": "openai/gpt-oss-120b",
            "tp": 1,
            "backend_version": "0.19.0",
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "attention_dp": 1,
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
    }
    assert json.loads(replay_kwargs["prefill_engine_args"]) == expected_engine_args
    assert json.loads(replay_kwargs["decode_engine_args"]) == expected_engine_args
    assert replay_kwargs["num_prefill_workers"] == 2
    assert replay_kwargs["num_decode_workers"] == 3
    assert {"extra_engine_args", "num_workers"}.isdisjoint(replay_kwargs)
    assert replay_kwargs["substrate_config"]["prefill_engine_num_gpu"] == 1
    assert replay_kwargs["substrate_config"]["decode_engine_num_gpu"] == 1

    runtime = run.result["runtime"]
    assert runtime["substrate_preset"] == "gpt_oss"
    assert runtime["model"] == {
        "name": "configured-model",
        "ais_model_path": "openai/gpt-oss-120b",
    }
    assert set(runtime["engines"]) == {"prefill", "decode"}
    _assert_engine_provenance_matches_replay(
        run.result,
        role="prefill",
        rendered_args=replay_kwargs["prefill_engine_args"],
        num_gpus=1,
    )
    _assert_engine_provenance_matches_replay(
        run.result,
        role="decode",
        rendered_args=replay_kwargs["decode_engine_args"],
        num_gpus=1,
    )


def test_run_sim_item_wires_explicit_disagg_role_engines_and_gpu_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _sim_config(
        tmp_path,
        "disagg",
        deployment={
            "gpu_budget": 14,
            "model": {
                "name": "request-model-alias",
                "ais_model_path": "weights/model-for-ais",
            },
            "engines": {
                "common": {
                    "system": "b200_sxm",
                    "backend": "sglang",
                    "backend_version": "0.5.10",
                    "ais_backend": "vllm",
                    "ais_backend_version": "0.19.0",
                    "attention_dp_size": 1,
                    "runtime": {
                        "cold_start_delay_s": 12.5,
                        "kv_transfer_bandwidth_gbps": 800,
                        "kv_bytes_per_token": 98304,
                    },
                    "extra_args": {
                        "block_size": 64,
                        "num_gpu_blocks": 999,
                    },
                },
                "prefill": {
                    "tp_size": 4,
                    "moe_tp_size": 2,
                    "moe_ep_size": 2,
                    "runtime": {"cold_start_delay_s": 25},
                    "extra_args": {
                        "num_gpu_blocks": 111,
                        "max_num_seqs": 101,
                    },
                },
                "decode": {
                    "tp_size": 2,
                    "moe_tp_size": 1,
                    "moe_ep_size": 2,
                    "extra_args": {
                        "num_gpu_blocks": 222,
                        "max_num_seqs": 202,
                    },
                },
            },
        },
    )
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.engines.prefill is not None
    assert backend.engines.decode is not None
    assert backend.engines.prefill.num_gpus == 4
    assert backend.engines.decode.num_gpus == 2
    assert backend.engines.prefill.backend == "sglang"
    assert backend.engines.prefill.ais_backend == "vllm"
    assert backend.engines.prefill.ais_backend_version == "0.19.0"

    run = _run_sim_with_fakes(tmp_path, monkeypatch, config)
    replay_kwargs = _assert_common_sim_wiring(
        run, expected_model_name="request-model-alias"
    )

    assert json.loads(replay_kwargs["prefill_engine_args"]) == {
        "engine_type": "sglang",
        "startup_time": 25.0,
        "kv_transfer_bandwidth": 800.0,
        "kv_bytes_per_token": 98304,
        "block_size": 64,
        "num_gpu_blocks": 111,
        "max_num_seqs": 101,
        "tensor_parallel_size": 4,
        "dp_size": 1,
        "ais_perf_config": {
            "backend": "vllm",
            "system": "b200_sxm",
            "model": "weights/model-for-ais",
            "tp": 4,
            "backend_version": "0.19.0",
            "moe_tp_size": 2,
            "moe_ep_size": 2,
            "attention_dp": 1,
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
    }
    assert json.loads(replay_kwargs["decode_engine_args"]) == {
        "engine_type": "sglang",
        "startup_time": 12.5,
        "kv_transfer_bandwidth": 800.0,
        "kv_bytes_per_token": 98304,
        "block_size": 64,
        "num_gpu_blocks": 222,
        "max_num_seqs": 202,
        "tensor_parallel_size": 2,
        "dp_size": 1,
        "ais_perf_config": {
            "backend": "vllm",
            "system": "b200_sxm",
            "model": "weights/model-for-ais",
            "tp": 2,
            "backend_version": "0.19.0",
            "moe_tp_size": 1,
            "moe_ep_size": 2,
            "attention_dp": 1,
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
    }
    assert replay_kwargs["performance_model_metadata"] == {
        "prefill": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "backend_version": "0.19.0",
                "system": "b200_sxm",
                "model_path": "weights/model-for-ais",
                "tp_size": 4,
                "moe_tp_size": 2,
                "moe_ep_size": 2,
                "attention_dp_size": 1,
            },
        },
        "decode": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "backend_version": "0.19.0",
                "system": "b200_sxm",
                "model_path": "weights/model-for-ais",
                "tp_size": 2,
                "moe_tp_size": 1,
                "moe_ep_size": 2,
                "attention_dp_size": 1,
            },
        },
    }
    assert replay_kwargs["num_prefill_workers"] == 2
    assert replay_kwargs["num_decode_workers"] == 3
    assert {"extra_engine_args", "num_workers"}.isdisjoint(replay_kwargs)
    assert replay_kwargs["substrate_config"]["prefill_engine_num_gpu"] == 4
    assert replay_kwargs["substrate_config"]["decode_engine_num_gpu"] == 2
    assert replay_kwargs["substrate_config"]["max_gpu_budget"] == 14

    runtime = run.result["runtime"]
    assert runtime["substrate_preset"] is None
    assert runtime["model"] == {
        "name": "request-model-alias",
        "ais_model_path": "weights/model-for-ais",
    }
    assert set(runtime["engines"]) == {"prefill", "decode"}
    assert runtime["engines"]["prefill"]["config"]["extra_args"] == {
        "block_size": 64,
        "num_gpu_blocks": 111,
        "max_num_seqs": 101,
    }
    assert runtime["engines"]["prefill"]["config"]["runtime"] == {
        "cold_start_delay_s": 25.0,
        "kv_transfer_bandwidth_gbps": 800.0,
        "kv_bytes_per_token": 98304,
    }
    assert runtime["engines"]["decode"]["config"]["extra_args"] == {
        "block_size": 64,
        "num_gpu_blocks": 222,
        "max_num_seqs": 202,
    }
    assert runtime["engines"]["decode"]["config"]["runtime"] == {
        "cold_start_delay_s": 12.5,
        "kv_transfer_bandwidth_gbps": 800.0,
        "kv_bytes_per_token": 98304,
    }
    _assert_engine_provenance_matches_replay(
        run.result,
        role="prefill",
        rendered_args=replay_kwargs["prefill_engine_args"],
        num_gpus=4,
    )
    _assert_engine_provenance_matches_replay(
        run.result,
        role="decode",
        rendered_args=replay_kwargs["decode_engine_args"],
        num_gpus=2,
    )


def test_run_sim_item_wires_explicit_agg_engine_and_gpu_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _sim_config(
        tmp_path,
        "agg",
        deployment={
            "gpu_budget": 24,
            "model": {"name": "aggregate-request-model"},
            "engines": {
                "common": {
                    "system": "h100_sxm",
                    "backend": "vllm",
                    "backend_version": "0.20.0",
                    "tp_size": 2,
                    "attention_dp_size": 1,
                    "runtime": {"cold_start_delay_s": 7},
                    "extra_args": {
                        "block_size": 32,
                        "num_gpu_blocks": 900,
                    },
                },
                "aggregate": {
                    "tp_size": 4,
                    "attention_dp_size": 2,
                    "runtime": {
                        "kv_transfer_bandwidth_gbps": 400,
                        "kv_bytes_per_token": 65536,
                    },
                    "extra_args": {
                        "num_gpu_blocks": 333,
                        "max_num_seqs": 42,
                    },
                },
            },
        },
    )
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.engines.aggregate is not None
    assert backend.engines.aggregate.num_gpus == 8
    assert backend.autoscalers[0].start == ReplicaCounts(prefill=0, decode=3)

    run = _run_sim_with_fakes(tmp_path, monkeypatch, config)
    replay_kwargs = _assert_common_sim_wiring(
        run, expected_model_name="aggregate-request-model"
    )

    assert json.loads(replay_kwargs["extra_engine_args"]) == {
        "engine_type": "vllm",
        "startup_time": 7.0,
        "kv_transfer_bandwidth": 400.0,
        "kv_bytes_per_token": 65536,
        "block_size": 32,
        "num_gpu_blocks": 333,
        "max_num_seqs": 42,
        "tensor_parallel_size": 4,
        "dp_size": 2,
        "ais_perf_config": {
            "backend": "vllm",
            "system": "h100_sxm",
            "model": "aggregate-request-model",
            "tp": 4,
            "backend_version": "0.20.0",
            "attention_dp": 2,
            "estimation_mode": "auto",
            "fallback_policy": "deny",
        },
    }
    assert replay_kwargs["num_workers"] == 3
    assert {
        "prefill_engine_args",
        "decode_engine_args",
        "num_prefill_workers",
        "num_decode_workers",
    }.isdisjoint(replay_kwargs)
    assert replay_kwargs["substrate_config"]["prefill_engine_num_gpu"] == 8
    assert replay_kwargs["substrate_config"]["decode_engine_num_gpu"] == 8
    assert replay_kwargs["substrate_config"]["max_gpu_budget"] == 24

    runtime = run.result["runtime"]
    assert runtime["substrate_preset"] is None
    assert runtime["model"] == {
        "name": "aggregate-request-model",
        "ais_model_path": "aggregate-request-model",
    }
    assert set(runtime["engines"]) == {"aggregate"}
    assert runtime["engines"]["aggregate"]["config"]["extra_args"] == {
        "block_size": 32,
        "num_gpu_blocks": 333,
        "max_num_seqs": 42,
    }
    assert runtime["engines"]["aggregate"]["config"]["runtime"] == {
        "cold_start_delay_s": 7.0,
        "kv_transfer_bandwidth_gbps": 400.0,
        "kv_bytes_per_token": 65536,
    }
    _assert_engine_provenance_matches_replay(
        run.result,
        role="aggregate",
        rendered_args=replay_kwargs["extra_engine_args"],
        num_gpus=8,
    )


def test_performance_model_metadata_preserves_unpinned_aggregate_identity(
    tmp_path: Path,
) -> None:
    config = _sim_config(tmp_path, "agg")
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.engines.aggregate is not None
    unpinned = replace(
        backend,
        engines=replace(
            backend.engines,
            aggregate=replace(backend.engines.aggregate, ais_backend_version=None),
        ),
    )

    assert match_runner._sim_performance_model_metadata(unpinned) == {
        "aggregated": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "system": "h200_sxm",
                "model_path": "openai/gpt-oss-120b",
                "tp_size": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "attention_dp_size": 1,
            },
        }
    }


def test_performance_model_metadata_preserves_distinct_unpinned_disagg_roles(
    tmp_path: Path,
) -> None:
    config = _sim_config(
        tmp_path,
        "disagg",
        deployment={
            "model": {
                "name": "request-model",
                "ais_model_path": "weights/ais-model",
            },
            "engines": {
                "prefill": {
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "ais_backend": "vllm",
                    "tp_size": 2,
                    "attention_dp_size": 1,
                },
                "decode": {
                    "system": "b200_sxm",
                    "backend": "sglang",
                    "ais_backend": "sglang",
                    "tp_size": 4,
                    "attention_dp_size": 1,
                },
            },
        },
    )
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.engines.prefill is not None
    assert backend.engines.decode is not None
    assert backend.engines.prefill.ais_backend_version is None
    assert backend.engines.decode.ais_backend_version is None

    assert match_runner._sim_performance_model_metadata(backend) == {
        "prefill": {
            "provider": "ais",
            "config": {
                "backend": "vllm",
                "system": "h200_sxm",
                "model_path": "weights/ais-model",
                "tp_size": 2,
                "attention_dp_size": 1,
            },
        },
        "decode": {
            "provider": "ais",
            "config": {
                "backend": "sglang",
                "system": "b200_sxm",
                "model_path": "weights/ais-model",
                "tp_size": 4,
                "attention_dp_size": 1,
            },
        },
    }


@pytest.mark.parametrize(
    ("topology", "expected_start", "expected_parameters"),
    [
        (
            "disagg",
            ReplicaCounts(prefill=2, decode=3),
            {
                "num_prefill": 2,
                "num_decode": 3,
                "poll_interval_s": 2.5,
            },
        ),
        (
            "agg",
            ReplicaCounts(prefill=0, decode=3),
            {
                "num_prefill": 0,
                "num_decode": 3,
                "poll_interval_s": 2.5,
            },
        ),
    ],
)
def test_static_config_derives_matching_start_and_builds_fresh_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    expected_start: ReplicaCounts,
    expected_parameters: dict[str, Any],
):
    config = _sim_config(tmp_path, topology)
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    autoscaler = backend.autoscalers[0]

    assert autoscaler.start == expected_start
    assert autoscaler.config == expected_parameters

    class FakeStaticAutoscaler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # The production adapters import Dynamo types. Replace only the lazy import
    # boundary so this factory contract remains runnable in a lightweight env.
    fake_adapters = types.ModuleType("autoscaling_arena.adapters")
    fake_adapters.StaticAutoscaler = FakeStaticAutoscaler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoscaling_arena.adapters", fake_adapters)

    factory = match_runner._build_sim_factory(autoscaler, topology=topology)
    first = factory(object(), object())
    second = factory(object(), object())

    assert first is not second
    assert first.kwargs == {"mode": topology, **expected_parameters}
    assert second.kwargs == first.kwargs


def test_run_real_item_materializes_speedup_resolves_endpoint_and_projects_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = parse_match_config(
        {
            "schema_version": 1,
            "name": "real-wiring",
            "backend": {
                "type": "real",
                "endpoints": [
                    {
                        "name": "not-selected",
                        "url": "https://unused.example/v1",
                        "model": "unused-model",
                    },
                    {
                        "name": "serving-b",
                        "url": "https://selected.example/v1",
                        "model": "selected-model",
                        "description": "the selected endpoint",
                        "endpoint_type": "completions",
                        "declared_deployment": {
                            "topology": "disagg",
                            "model": {
                                "name": "weights/model",
                                "revision": "deployment-revision",
                            },
                            "engines": {
                                "prefill": {"tp_size": 2},
                                "decode": {"tp_size": 4},
                            },
                        },
                    },
                ],
                "aiperf": {
                    "executable": "custom-aiperf",
                    "tokenizer": "custom-tokenizer",
                    "streaming": False,
                    "timeout_s": 45,
                    "extra_args": ["--request-rate", "9"],
                },
                "autoscalers": [
                    {
                        "name": "autoscaler-b",
                        "endpoint": "serving-b",
                        "autoscaler_type": "keda",
                        "declared_config": {"min_replicas": 2},
                    }
                ],
            },
            "evaluations": {
                "workloads": ["flat"],
                "defaults": {
                    "seed": 13,
                    "max_requests": 6,
                    "arrival_speedup": 5.0,
                },
            },
            "slo_profiles": [{"name": "online-sla", "ttft_ms": 250, "itl_ms": 40}],
            "metrics": {
                "rank_by": "good_request_fraction",
                "include": [
                    "good_request_fraction",
                    "request_count",
                    "benchmark_duration_s",
                    "mean_ttft_ms",
                ],
            },
            "publish": {
                "artifact_root": "artifacts",
                "destinations": [{"type": "console"}],
            },
        },
        source_path=tmp_path / "match.yaml",
    )
    assert config.metrics.rank_by == "good_rate"
    assert config.metrics.include == (
        "good_rate",
        "completed_requests",
        "duration_s",
        "mean_ttft_ms",
    )
    item = next(config.iter_runs())

    trace_path = tmp_path / "real-trace.jsonl"
    trace_path.write_text('{"timestamp": 0}\n')
    materialize_calls: list[dict[str, Any]] = []
    endpoint_calls: list[tuple[Any, Path, Any, dict[str, Any]]] = []

    def fake_materialize(context, **kwargs):
        del context
        materialize_calls.append(kwargs)
        return trace_path.resolve()

    def fake_run_endpoint_match(endpoint, trace_file, profile, **kwargs):
        endpoint_calls.append((endpoint, trace_file, profile, kwargs))
        return SimpleNamespace(
            metrics={
                "good_request_fraction": 0.875,
                "request_count": 16.0,
                "benchmark_duration_s": 12.5,
                "mean_ttft_ms": 91.0,
            },
            returncode=0,
            artifact_dir=str(tmp_path / "aiperf-artifacts"),
            stderr_tail="",
        )

    import autoscaling_arena.runners.real as real_runner

    monkeypatch.setattr(real_runner, "run_endpoint_match", fake_run_endpoint_match)
    monkeypatch.setattr(
        match_runner,
        "get_workload",
        lambda name: SimpleNamespace(block_size=128),
    )
    monkeypatch.setattr(match_runner, "_materialize_trace", fake_materialize)

    context = match_runner._ExecutionContext(
        session_id="real-session",
        session_root=tmp_path / "real-session",
    )
    result = match_runner._run_real_item(config, item, context)

    assert materialize_calls == [
        {
            "workload_name": "flat",
            "seed": 13,
            "max_requests": 6,
            "arrival_speedup": 5.0,
            "speedup_is_materialized": True,
        }
    ]
    assert len(endpoint_calls) == 1
    endpoint, passed_trace, profile, aiperf_kwargs = endpoint_calls[0]
    assert endpoint.name == "autoscaler-b"
    assert endpoint.url == "https://selected.example/v1"
    assert endpoint.model == "selected-model"
    assert endpoint.description == "the selected endpoint"
    assert endpoint.endpoint_type == "completions"
    assert passed_trace == trace_path.resolve()
    assert profile.name == "online-sla"
    assert profile.ttft_ms == 250.0
    assert profile.itl_ms == 40.0
    assert aiperf_kwargs == {
        "artifact_root": tmp_path / "real-session" / "runs" / item.run_id,
        "block_size": 128,
        "streaming": False,
        "tokenizer": "custom-tokenizer",
        "aiperf_bin": "custom-aiperf",
        "timeout_s": 45.0,
        "extra_args": ("--request-rate", "9"),
        "workload_name": "flat",
    }
    assert "arrival_speedup" not in aiperf_kwargs

    assert result["status"] == "ok"
    assert result["endpoint"] == "serving-b"
    assert result["runtime"]["endpoint"] == {
        "name": "serving-b",
        "url": "https://selected.example/v1",
        "model": "selected-model",
        "endpoint_type": "completions",
        "declared_deployment": {
            "topology": "disagg",
            "model": {
                "name": "weights/model",
                "revision": "deployment-revision",
            },
            "engines": {
                "prefill": {"tp_size": 2},
                "decode": {"tp_size": 4},
            },
        },
    }
    assert result["metrics"] == {
        "good_rate": 0.875,
        "completed_requests": 16.0,
        "duration_s": 12.5,
        "mean_ttft_ms": 91.0,
    }


def test_real_match_config_preserves_mixed_trace_block_sizes(tmp_path, monkeypatch):
    import subprocess

    from autoscaling_arena.runners import real
    from autoscaling_arena.workloads import validate_mooncake_trace

    traces = []
    for block_size in (16, 512):
        trace = tmp_path / f"trace-{block_size}.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "timestamp": 0,
                    "input_length": block_size * 2,
                    "output_length": 2,
                    "hash_ids": [1, 2],
                }
            )
            + "\n"
        )
        traces.append(
            {
                "name": f"blocks-{block_size}",
                "path": str(trace),
                "block_size": block_size,
                "presorted": True,
            }
        )
    config = parse_match_config(
        {
            "schema_version": 1,
            "name": "mixed-block-sizes",
            "backend": {
                "type": "real",
                "endpoints": [
                    {"name": "serving", "url": "http://unused", "model": "example"}
                ],
                "autoscalers": [{"name": "static", "endpoint": "serving"}],
            },
            "evaluations": {"traces": traces},
            "slo_profiles": [{"name": "test", "ttft_ms": 1000}],
            "metrics": {"rank_by": "goodput_rps", "include": ["goodput_rps"]},
            "publish": {
                "artifact_root": str(tmp_path / "artifacts"),
                "destinations": [{"type": "console"}],
            },
        },
        source_path=tmp_path / "match.yaml",
    )
    observed_sizes = []

    def run(command, **kwargs):
        block_size = int(command[command.index("--isl-block-size") + 1])
        trace = Path(command[command.index("--input-file") + 1])
        assert validate_mooncake_trace(trace, block_size=block_size) == 1
        observed_sizes.append(block_size)
        output = Path(command[command.index("--artifact-dir") + 1])
        (output / "profile_export_aiperf.json").write_text('{"goodput": {"avg": 2}}')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(real.subprocess, "run", run)
    monkeypatch.setattr(match_runner, "_git_commit", lambda: "test-commit")
    report = match_runner.execute_match_config(config)
    assert report["summary"]["succeeded_runs"] == 2
    assert observed_sizes == [16, 512]
    assert [
        result["evaluation"]["trace_block_size"] for result in report["results"]
    ] == observed_sizes
