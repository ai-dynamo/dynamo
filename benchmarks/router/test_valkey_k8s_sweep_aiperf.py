# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.router.kubernetes.valkey_sweep import aiperf_runner
from benchmarks.router.kubernetes.valkey_sweep import model as sweep_model
from benchmarks.router.kubernetes.valkey_sweep import sweep
from benchmarks.router.valkey_k8s_sweep_test_support import (
    BINDING,
    DIGEST_IMAGE,
    IMAGE_ID,
    aiperf_metrics,
    completed,
)


def test_aiperf_summary_rejects_missing_nonfinite_and_wrong_lengths() -> None:
    point = sweep_model.MatrixPoint(1, 4096, 10)
    metrics = aiperf_metrics(sweep_model.request_count(point.concurrency))
    missing = {
        key: value for key, value in metrics.items() if key != "request_throughput"
    }
    with pytest.raises(RuntimeError, match="invalid required metrics"):
        aiperf_runner.validate_metrics(missing, point)
    nonfinite = {**metrics, "request_throughput": {"avg": float("nan")}}
    with pytest.raises(RuntimeError, match="invalid required metrics"):
        aiperf_runner.validate_metrics(nonfinite, point)
    wrong_isl = deepcopy(metrics)
    wrong_isl["input_sequence_length"]["avg"] = 800
    with pytest.raises(RuntimeError, match="actual_isl_avg"):
        aiperf_runner.validate_metrics(wrong_isl, point)
    wrong_osl = deepcopy(metrics)
    wrong_osl["output_sequence_length"]["avg"] = 1300
    with pytest.raises(RuntimeError, match="calibrated AIPerf chat range"):
        aiperf_runner.validate_metrics(wrong_osl, point)
    with pytest.raises(RuntimeError, match="cancelled"):
        aiperf_runner.validate_metrics({**metrics, "was_cancelled": True}, point)
    with pytest.raises(RuntimeError, match="completion counts"):
        aiperf_runner.validate_metrics(
            {**metrics, "error_summary": [{"message": "failed"}]}, point
        )


def test_aiperf_command_run_point_and_image_binding(
    monkeypatch, tmp_path: Path
) -> None:
    point = sweep.MatrixPoint(1, 4096, 10)
    requests = sweep_model.request_count(point.concurrency)
    summary = json.dumps(aiperf_metrics(requests, ["http://10.0.0.1:8000"]))
    client_calls: list[tuple[str, ...]] = []

    class FakeCluster:
        def client_exec(self, arguments, **_kwargs):
            client_calls.append(tuple(arguments))
            if arguments[0] == "cat":
                return completed(summary)
            if arguments[0] == "/app/.venv/bin/python":
                return completed(
                    json.dumps(
                        {
                            "revision": "c" * 40,
                            "dirty": False,
                            "valkey_revision": sweep_model.VALKEY_GIT_REVISION,
                        }
                    )
                )
            return completed("benchmark-output")

        def valkey(self, _host, command, *arguments):
            if command == "INFO":
                return ["# section", "role:master", "used_memory:42", "ignored:value"]
            if command == "DBSIZE":
                return ["7"]
            raise AssertionError((command, arguments))

    fake = FakeCluster()
    snapshot = {"pod": {"uid": "1"}}
    ha_evidence = {
        group: {
            "sentinel_master": [f"{group}-master", 6379],
            "elected_identity": f"{group}-primary",
            "replica_identity": f"{group}-replica",
            "identities": {},
        }
        for group in ("dynamo-router", "dynamo-tokenizer")
    }
    monkeypatch.setattr(aiperf_runner, "wait_for_ha", lambda *_: None)
    monkeypatch.setattr(aiperf_runner, "verify_active_images", lambda *_: snapshot)
    monkeypatch.setattr(aiperf_runner, "ha_snapshot", lambda *_: ha_evidence)
    monkeypatch.setattr(
        aiperf_runner, "registered_mocker_stats", lambda *_: [0, 10, 10]
    )
    topology = {
        "runtime_namespace": "runtime",
        "attempt_generation": "generation",
        "registered_index_stats": [0, 10, 10],
        "frontend_pods": ["frontend"],
        "frontend_urls": ["http://10.0.0.1:8000"],
        "pre_runtime_snapshot": snapshot,
        "pre_ha_snapshot": ha_evidence,
        "load_generator_capacity": {
            "fd_limit": "1048576",
            "ephemeral_port_start": 1024,
            "ephemeral_port_end": 65535,
            "ephemeral_ports": 64512,
            "required_connections_with_headroom": 8192,
        },
        "router_reset": {
            "flush": ["OK"],
            "primary_dbsize": ["0"],
            "replica_dbsize": ["0"],
        },
        "router_write_durability": sweep.router_write_durability(),
        "tokenizer_reset": {
            "flush": ["OK"],
            "primary_dbsize": ["0"],
            "replica_dbsize": ["0"],
        },
    }
    command = aiperf_runner.aiperf_command(
        point, topology["frontend_urls"], "/data/results"
    )
    assert "--request-rate" not in command
    assert command.count("--url") == 1
    assert command[command.index("--export-level") + 1] == "records"
    result = aiperf_runner.run_point(
        fake, "campaign", "sha256:manifest", point, topology, tmp_path, BINDING
    )
    assert result["metrics"]["request_throughput_rps"] == pytest.approx(
        255.191, abs=0.001
    )
    assert result["metrics"]["actual_isl_avg"] == pytest.approx(1024.001)
    assert result["metrics"]["actual_osl_avg"] == pytest.approx(1049.653)
    assert result["post_registered_index_stats"] == [0, 10, 10]
    assert result["remote_artifacts_retained"] is False
    assert any(call[:3] == ("rm", "-rf", "--") for call in client_calls)
    assert json.loads((tmp_path / "result.json").read_text()) == result
    monkeypatch.setattr(sweep, "active_image_inventory", lambda *_: {"pod": [IMAGE_ID]})
    monkeypatch.setattr(
        sweep,
        "pods",
        lambda *_: [
            {
                "status": {
                    "containerStatuses": [{"image": DIGEST_IMAGE}],
                    "initContainerStatuses": [],
                }
            }
        ],
    )
    assert sweep.image_binding(fake, DIGEST_IMAGE) == BINDING


def test_aiperf_failure_still_cleans_remote_records(
    monkeypatch, tmp_path: Path
) -> None:
    point = sweep.MatrixPoint(1, 4096, 10)
    client_calls: list[tuple[str, ...]] = []

    class FailingCluster:
        def client_exec(self, arguments, **_kwargs):
            client_calls.append(tuple(arguments))
            if arguments[0] == "/opt/aiperf-venv/bin/aiperf":
                return completed("partial output", 2)
            if arguments[0] == "cat":
                return completed('{"partial": true}\n')
            return completed()

    topology = {
        "runtime_namespace": "runtime",
        "attempt_generation": "generation",
        "registered_index_stats": [0, 10, 10],
        "frontend_pods": ["frontend"],
        "frontend_urls": ["http://10.0.0.1:8000"],
        "pre_runtime_snapshot": {"pod": {"uid": "1"}},
        "pre_ha_snapshot": {"ha": "healthy"},
        "load_generator_capacity": {},
        "router_reset": {},
        "router_write_durability": sweep.router_write_durability(),
        "tokenizer_reset": {},
    }
    monkeypatch.setattr(aiperf_runner, "valkey_telemetry", lambda *_: {})
    with pytest.raises(RuntimeError, match="exit code 2"):
        aiperf_runner.run_point(
            FailingCluster(),
            "campaign",
            "sha256:manifest",
            point,
            topology,
            tmp_path,
            BINDING,
        )
    assert any(call[:3] == ("rm", "-rf", "--") for call in client_calls)
    assert json.loads((tmp_path / "profile_export_aiperf.json").read_text()) == {
        "partial": True
    }


def test_selection_cli_and_resume_main(monkeypatch, tmp_path: Path) -> None:
    args = SimpleNamespace(frontends=[1], concurrencies=[4096], mockers=[10])
    assert sweep.selected_points(args) == [sweep.MatrixPoint(1, 4096, 10)]
    with pytest.raises(ValueError, match="frontends"):
        sweep.selected_points(
            SimpleNamespace(frontends=[3], concurrencies=None, mockers=None)
        )
    parsed = sweep.parse_args(
        ["--image", DIGEST_IMAGE, "--campaign", "campaign", "--results", str(tmp_path)]
    )
    assert parsed.campaign == "campaign"

    monkeypatch.setattr(sweep, "Cluster", lambda *_: object())
    monkeypatch.setattr(sweep, "wait_for_ha", lambda *_: None)
    monkeypatch.setattr(sweep, "image_binding", lambda *_: BINDING)
    methodology = {
        "git_revision": BINDING["core_revision"],
        "git_dirty": False,
        "methodology_digest": "sha256:methodology",
    }
    monkeypatch.setattr(sweep, "methodology_binding", lambda *_: methodology)
    monkeypatch.setattr(sweep, "verify_methodology", lambda *_: None)
    monkeypatch.setattr(
        sweep, "sentinel_policy_snapshot", lambda *_: {"status": "expected"}
    )
    monkeypatch.setattr(
        sweep, "prove_network_isolation", lambda *_: {"status": "passed"}
    )
    monkeypatch.setattr(
        sweep,
        "configure_topology",
        lambda _cluster, _campaign, _point, _binding, generation: {
            "runtime_namespace": "runtime",
            "attempt_generation": generation,
            "frontend_urls": ["http://127.0.0.1:18100"],
        },
    )

    def fake_run_point(
        _cluster, campaign, manifest_digest, point, topology, local_dir, binding
    ):
        summary = aiperf_metrics(sweep_model.request_count(point.concurrency))
        summary_path = local_dir / "profile_export_aiperf.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validated = aiperf_runner.validate_metrics(
            summary, point, topology["frontend_urls"]
        )
        result = {
            "status": "ok",
            "campaign": campaign,
            "manifest_digest": manifest_digest,
            "attempt_generation": topology["attempt_generation"],
            "point": point._asdict(),
            "provenance": binding,
            "aiperf_summary_digest": sweep.file_digest(summary_path),
            "frontend_urls": topology["frontend_urls"],
            "metrics": validated,
        }
        sweep.write_json_atomic(local_dir / "result.json", result)
        return result

    monkeypatch.setattr(sweep, "run_point", fake_run_point)
    invocation = [
        "--image",
        DIGEST_IMAGE,
        "--campaign",
        "campaign",
        "--results",
        str(tmp_path),
        "--frontends",
        "1",
        "--concurrencies",
        "4096",
        "--mockers",
        "10",
    ]
    assert sweep.main(invocation) == 0
    assert sweep.main([*invocation, "--resume"]) == 0
    assert json.loads((tmp_path / "results.json").read_text())[0]["status"] == "ok"
    attempt_path = tmp_path / "m1-c4096-n10" / "attempt.json"
    completed_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    sweep.write_json_atomic(
        attempt_path,
        {
            key: value
            for key, value in completed_attempt.items()
            if key not in {"result", "result_digest", "aiperf_summary_digest"}
        }
        | {"status": "starting"},
    )
    assert sweep.main([*invocation, "--resume"]) == 0
    rerun_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert rerun_attempt["status"] == "complete"
    assert rerun_attempt["generation"] != completed_attempt["generation"]
    with pytest.raises(ValueError, match="campaign contract"):
        sweep.main([*invocation[:-1], "40", "--resume"])

    result_path = tmp_path / "m1-c4096-n10" / "result.json"
    original_result = json.loads(result_path.read_text(encoding="utf-8"))
    tampered_result = deepcopy(original_result)
    tampered_result["metrics"]["request_throughput_rps"] += 1
    sweep.write_json_atomic(result_path, tampered_result)
    with pytest.raises(ValueError, match="not bound"):
        sweep.main([*invocation, "--resume"])
    sweep.write_json_atomic(result_path, original_result)

    summary_path = tmp_path / "m1-c4096-n10" / "profile_export_aiperf.json"
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tampered_summary = deepcopy(original_summary)
    tampered_summary["request_throughput"]["avg"] += 1
    summary_path.write_text(
        json.dumps(tampered_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not bound"):
        sweep.main([*invocation, "--resume"])
