# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""CPU-only end-to-end coverage for the public Dynamo-stack CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip(
    "aisimulate.config.cli",
    reason="AISimulate is an optional Dynamo simulation dependency",
)

from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config.common import split_config_sections

pytestmark = [
    # Release-gating CPU smoke for plugin discovery, replay composition, and
    # recommendation-to-prediction round trips in the shipped Planner image.
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_REPO_ROOT = Path(__file__).resolve().parents[6]
_CONFIG_ROOT = Path("components/src/dynamo/replay/tests/e2e/configs/unified_cli")
_PREDICT_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "predict/dynamo").glob("*.yaml"))
)
_RECOMMEND_CASES = tuple(
    sorted((_REPO_ROOT / _CONFIG_ROOT / "recommend/dynamo").glob("*.yaml"))
)


def _run_cli(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *args],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"aisimulate {' '.join(args)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _assert_concrete(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        assert "choices" not in value, f"{path} still contains a choices domain"
        assert "range" not in value, f"{path} still contains a range domain"
        assert "preset" not in value, f"{path} still contains a preset"
        for key, child in value.items():
            _assert_concrete(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_concrete(child, path=f"{path}[{index}]")


@pytest.mark.parametrize("config_path", _PREDICT_CASES, ids=lambda path: path.stem)
def test_dynamo_predict_cli_cases(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "predict",
        "--stack",
        "dynamo",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    summary = json.loads(result.stdout)
    report = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    assert summary["completed_requests"] > 0
    assert report["summary"]["completed_requests"] == summary["completed_requests"]

    expected_trace_requests = {
        "05-trace-standard-round-robin.yaml": 2,
        "09-trace-mooncake-speedup.yaml": 2,
        "10-trace-mooncake-delta-concurrency.yaml": 2,
        "11-trace-agentic-mooncake.yaml": 3,
        "12-trace-applied-compute-agentic.yaml": 3,
        "13-trace-dynamo-agentic.yaml": 4,
    }
    if config_path.name in expected_trace_requests:
        assert (
            summary["completed_requests"] == expected_trace_requests[config_path.name]
        )

    if config_path.name == "08-synthetic-throughput-planner.yaml":
        assert report["planner"]["metadata"]["bootstrap"]["status"] == "installed"
        assert "falling back to load-based scaling only" not in result.stderr


@pytest.mark.parametrize("throughput_planner", [False, True])
def test_legacy_default_aic_timing_cli(tmp_path, throughput_planner):
    name = (
        "08-synthetic-throughput-planner.yaml"
        if throughput_planner
        else "01-synthetic-no-adapters.yaml"
    )
    config = yaml.safe_load(
        (_REPO_ROOT / _CONFIG_ROOT / "predict/dynamo" / name).read_text()
    )
    config["engine"]["model"] = "Qwen/Qwen3-32B"
    config["engine"]["workers"]["aggregated"]["timing"] = {"type": "default"}
    path = tmp_path / "aic-timing.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "aic-output"
    result = _run_cli(
        "predict",
        "--stack",
        "dynamo",
        "--config",
        str(path),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )
    summary = json.loads(result.stdout)
    assert summary["completed_requests"] == 6
    # Real AIC decode timing for this BF16 model differs from the template's
    # fixed one-millisecond engine; the test executes the compiled estimator.
    assert summary["mean_tpot_ms"] > 1
    if throughput_planner:
        report = json.loads((output / "prediction.json").read_text())
        assert report["planner"]["metadata"]["bootstrap"]["status"] == "installed"
        assert "falling back to load-based scaling only" not in result.stderr
    else:
        # Sweeper passes the SDK's resolved op_level identity, with expanded
        # estimator defaults and system roots, rather than authored auto values.
        recommend = yaml.safe_load(
            (
                _REPO_ROOT
                / _CONFIG_ROOT
                / "recommend/dynamo/01-no-adapters-throughput.yaml"
            ).read_text()
        )
        recommend["engine"]["model"] = config["engine"]["model"]
        recommend["engine"]["workers"]["aggregated"]["timing"] = {
            "type": "default",
            "estimation_mode": "op_level",
        }
        recommend_path = tmp_path / "aic-recommend.yaml"
        recommend_path.write_text(yaml.safe_dump(recommend))
        recommended = _run_cli(
            "recommend",
            "--stack",
            "dynamo",
            "--config",
            str(recommend_path),
            "--output-dir",
            str(tmp_path / "recommend"),
            "--format",
            "json",
        )
        assert json.loads(recommended.stdout)


@pytest.mark.parametrize("config_path", _RECOMMEND_CASES, ids=lambda path: path.stem)
def test_dynamo_recommend_cli_cases_round_trip(
    config_path: Path, tmp_path: Path
) -> None:
    output = tmp_path / config_path.stem
    result = _run_cli(
        "recommend",
        "--stack",
        "dynamo",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
    )

    rows = json.loads(result.stdout)
    recommendation_paths = sorted((output / "recommendations").glob("*.yaml"))
    assert rows
    assert len(recommendation_paths) == len(rows)
    assert len({path.read_bytes() for path in recommendation_paths}) == len(
        recommendation_paths
    )
    if config_path.name != "05-router-planner-pareto.yaml":
        assert [row["score"] for row in rows] == sorted(
            (row["score"] for row in rows), reverse=True
        )

    generated = []
    for index, recommendation_path in enumerate(recommendation_paths):
        raw = yaml.safe_load(recommendation_path.read_text(encoding="utf-8"))
        generated.append(raw)
        _assert_concrete(raw)
        core, _ = split_config_sections(raw, command="predict")
        CorePredictionConfig.model_validate(core)

        prediction_output = tmp_path / f"{config_path.stem}-predict-{index}"
        prediction = _run_cli(
            "predict",
            "--stack",
            "dynamo",
            "--config",
            str(recommendation_path),
            "--output-dir",
            str(prediction_output),
            "--format",
            "json",
        )
        assert json.loads(prediction.stdout)["completed_requests"] > 0
        assert "falling back to load-based scaling only" not in prediction.stderr

    disabled_planners = [
        config
        for config in generated
        if config.get("planner", {}).get("policy") == "disabled"
    ]
    assert len(disabled_planners) <= 1
    assert "router AIC payload requires" not in result.stderr


def _conversation_prediction(tmp_path: Path, backend: str, topology: str, mode: str):
    """Self-authored two-play tree; no weights, profiles, or network access."""
    baseline = _REPO_ROOT / _CONFIG_ROOT / "predict/dynamo/13-trace-dynamo-agentic.yaml"
    config = yaml.safe_load(baseline.read_text())
    rows = [
        {
            "schema": "dynamo.agentic_mooncake",
            "version": 2,
            "block_size": 64,
            "hash_id_scope": "local",
            "source": {
                "format": "dynamo-existing-adapter-test",
                "digest": "two-branches-two-plays-v1",
            },
        }
    ]
    for play in ("first", "second"):
        for index, session, predecessor, relation in (
            (0, "owner", None, None),
            (1, "branch-a", 0, "spawn"),
            (2, "branch-b", 0, "spawn"),
            (3, "branch-a", 1, "sequence"),
            (4, "branch-b", 2, "sequence"),
        ):
            rows.append(
                {
                    "request_id": f"{play}-{index}",
                    "play_id": play,
                    "session_id": session,
                    "model": config["engine"]["model"],
                    "input_length": 128,
                    "output_length": 2,
                    "hash_ids": [101, 202],
                    "not_before_ms": index * 20,
                    "recorded_api_time_ms": 2,
                    "dependencies": []
                    if predecessor is None
                    else [
                        {
                            "request_id": f"{play}-{predecessor}",
                            "relation": relation,
                            "trigger": "dispatch"
                            if relation == "spawn"
                            else "completion",
                            "delay_ms": 20,
                        }
                    ],
                }
            )
    trace = tmp_path / "conversation-tree.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["traffic"] = {
        "source": {
            "type": "trace",
            "format": "agentic_mooncake",
            "paths": [str(trace)],
            "block_size": 64,
        },
        "load": {"type": "trace_timestamps"},
    }
    worker = config["engine"]["workers"]["aggregated"]
    worker["parallelism"].update(replicas=2, attention_data=2)
    config["engine"].update(mode=topology, backend=backend)
    config["engine"]["workers"] = {
        role: json.loads(json.dumps(worker))
        for role in (
            ["aggregated"] if topology == "aggregated" else ["prefill", "decode"]
        )
    }
    config["router"] = {
        "policy": "kv_router",
        "affinity": {"mode": mode, "ttl_seconds": 3600},
    }
    return config


def _predict_conversations(tmp_path: Path, config: dict, name: str, *, stack=None):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / name
    args = [
        "predict",
        "--config",
        str(path),
        "--output-dir",
        str(output),
        "--capture-per-request",
        "--format",
        "json",
    ]
    if stack is not None:
        args.extend(["--stack", stack])
    _run_cli(*args)
    report = json.loads((output / "prediction.json").read_text())
    assert report["summary"]["completed_requests"] == report["completed_requests"]
    assert report["coverage"]["per_request_records"] == len(report["per_request"])
    assert report["per_request"] == [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    return report


def _assert_conversation_bindings(report: dict, topology: str, mode: str):
    policy = report["dynamo_policy"]
    assert policy["native_policy"] is True
    assert policy["routing_provider"] == "dynamo.DefaultWorkerSelector"
    assert policy["physical_kv_events"] > 0
    assert policy["post_dispatch_checks"] == policy["decision_count"]
    assert policy["dispatch_aborts"] == 0
    roles = {"aggregated"} if topology == "aggregated" else {"prefill", "decode"}
    decisions = {(row["request_id"], row["role"]): row for row in policy["decisions"]}
    bindings: dict[tuple, set] = {}
    group_keys: dict[tuple, set] = {}
    observations = 0
    for record in report["per_request"]:
        assert record["terminal_status"] == "completed"
        identity = record["agentic"]
        lineage = identity["lineage"]
        conversation = identity["conversation_id"]
        group = (
            (identity["play_id"], conversation)
            if mode == "session"
            else (
                identity["play_id"],
                lineage["root_conversation_id"],
                lineage.get("parent_conversation_id") or conversation,
                bool(lineage.get("parent_conversation_id")),
            )
        )
        observed_roles = set()
        for route in record["routing_history"]:
            role = "aggregated" if route["pool"] == "agg" else route["pool"]
            observed_roles.add(role)
            decision = decisions[(record["uuid"], role)]
            pair = decision["worker_id"], decision["dp_rank"]
            assert pair == (route["logical_worker_id"], route["dp_rank"])
            assert all(value in (0, 1) for value in pair)
            bindings.setdefault((group, role), set()).add(pair)
            group_keys.setdefault(group, set()).add(decision["group_key"])
            observations += 1
        assert observed_roles == roles
    assert observations > len(bindings)
    assert all(len(pairs) == 1 for pairs in bindings.values())
    assert all(len(keys) == 1 for keys in group_keys.values())
    assert len({next(iter(keys)) for keys in group_keys.values()}) == len(group_keys)
    assert any(row["binding_reused"] for row in policy["decisions"])
    assert report["first_admission_prefix_cache_reused_ratio"] > 0
    assert any(row["reused_input_tokens"] > 0 for row in report["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_conversation_yaml_uses_existing_native_router(
    tmp_path, backend, topology, mode
):
    config = _conversation_prediction(tmp_path, backend, topology, mode)
    report = _predict_conversations(tmp_path, config, "affinity")
    assert report["completed_requests"] == 10
    _assert_conversation_bindings(report, topology, mode)
    for worker in config["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = False
    cold = _predict_conversations(tmp_path, config, "cache-disabled", stack="dynamo")
    assert cold["completed_requests"] == report["completed_requests"]
    assert cold["first_admission_prefix_cache_reused_ratio"] == 0
    assert all(row["reused_input_tokens"] == 0 for row in cold["per_request"])


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_conversation_duration_reuses_snapshot_warmup_pipeline(
    tmp_path, backend, topology, mode
):
    config = _conversation_prediction(tmp_path, backend, topology, mode)
    config["traffic"]["load"].update(
        agentic_lanes=2,
        agentic_snapshot={"seed": 42},
        agentic_warmup=True,
        agentic_profile={"duration_seconds": 0.4},
    )
    report = _predict_conversations(tmp_path, config, "duration", stack="dynamo")
    _assert_conversation_bindings(report, topology, mode)
    profile = report["agentic_profile"]
    phases = report["agentic_phases"]
    assert profile["admission_closed"]
    assert profile["profile_start_ms"] == phases["profile_start_ms"]
    assert profile["admission_cutoff_ms"] - profile[
        "profile_start_ms"
    ] == pytest.approx(400)
    assert profile["plays_started"] > 2
    assert profile["unsettled_server_requests"] == 0
    assert not profile["cancel_drain_timed_out"]
    # Warmup requests traverse the same native policy before measured traffic.
    measured = {row["uuid"] for row in report["per_request"]}
    prepared = {row["uuid"] for row in phases["requests"]}
    assert prepared - measured
    decisions = {row["request_id"] for row in report["dynamo_policy"]["decisions"]}
    assert measured | prepared <= decisions


def test_native_report_storage_exhaustion_returns_resource_exit(tmp_path, monkeypatch):
    config = _conversation_prediction(tmp_path, "vllm", "aggregated", "session")
    config["router"] = {"policy": "kv_router"}
    config["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 2},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 4100},
    }
    path = tmp_path / "storage.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "storage-output"
    # Exact report samples spill to native temporary storage after 4096 results.
    monkeypatch.setenv("TMPDIR", str(tmp_path / "absent-temp-directory"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--config",
            str(path),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 3, result.stderr
    assert "report storage" in result.stderr
    assert not (output / "prediction.json").exists()
    assert (
        json.loads((output / "resource-runtime.json").read_text())["status"]
        == "resource_limited"
    )
