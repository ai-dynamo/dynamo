# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import compile_planner_gym_batch_impact as compiler
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def _timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _metric(value: float) -> dict[str, object]:
    return {"unit": "synthetic", "avg": value}


def _write_outer_manifest(run_dir: Path) -> None:
    manifest = {
        str(path.relative_to(run_dir)): compiler._sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "artifact-checksums.json"
    }
    _write_json(run_dir / "artifact-checksums.json", manifest)


def _pod(name: str, uid: str, image_id: str) -> dict[str, object]:
    return {
        "name": name,
        "uid": uid,
        "phase": "Running",
        "ready": True,
        "restart_count": 0,
        "image_ids": [image_id],
        "deleting": False,
    }


def _runtime_evidence(
    arm: str, *, native_metrics_interval_seconds: int = 5
) -> dict[str, object]:
    native_snapshot_count = len(range(0, 181, native_metrics_interval_seconds))
    pods = [
        _pod(
            "async-dispatch-llm-d-async-abc",
            "async-uid",
            "containerd://async-image-digest",
        ),
        _pod(
            "qwen3-0-6b-batch-frontend-abc",
            "frontend-uid",
            "containerd://frontend-image-digest",
        ),
        _pod(
            "qwen3-0-6b-batch-worker-abc",
            "worker-uid",
            "containerd://worker-image-digest",
        ),
    ]
    native = None
    if arm == "planner-native":
        pods.append(
            _pod(
                "qwen3-0-6b-batch-planner-abc",
                "planner-uid",
                "containerd://planner-image-digest",
            )
        )
        native = {
            "minimum_decision_logs": 2,
            "decision_log_match_count": native_snapshot_count - 1,
            "log_commands": {"logs-qwen3-0-6b-batch-planner-abc": 0},
        }
    start = {
        "phase": "start",
        "pod_status": [dict(pod) for pod in pods],
        "native_planner": native,
    }
    end = {
        "phase": "end",
        "pod_status": [dict(pod) for pod in pods],
        "native_planner": native,
    }
    return {
        "kubernetes_start": start,
        "kubernetes_end": end,
        "metrics_summary": {
            "enabled": True,
            "samples": native_snapshot_count * 2 if arm == "planner-native" else 4,
            "failed_samples": 0,
            "error": None,
            "endpoints": [
                {
                    "name": "frontend",
                    "successful_samples": (
                        native_snapshot_count if arm == "planner-native" else 4
                    ),
                    "failed_samples": 0,
                },
                *(
                    [
                        {
                            "name": "async",
                            "successful_samples": native_snapshot_count,
                            "failed_samples": 0,
                        }
                    ]
                    if arm == "planner-native"
                    else []
                ),
            ],
        },
    }


def _runtime_images(arm: str) -> list[dict[str, object]]:
    images: list[dict[str, object]] = [
        {
            "pod": "async-dispatch-llm-d-async-abc",
            "phase": "Running",
            "container": "llm-d-async",
            "image": "ghcr.io/llm-d/llm-d-async:v0.10.0",
            "image_id": "containerd://async-image-digest",
            "ready": True,
            "restart_count": 0,
        },
        {
            "pod": "qwen3-0-6b-batch-frontend-abc",
            "phase": "Running",
            "container": "main",
            "image": "registry.example/dynamo:test",
            "image_id": "containerd://frontend-image-digest",
            "ready": True,
            "restart_count": 0,
        },
        {
            "pod": "qwen3-0-6b-batch-worker-abc",
            "phase": "Running",
            "container": "main",
            "image": "registry.example/dynamo:test",
            "image_id": "containerd://worker-image-digest",
            "ready": True,
            "restart_count": 0,
        },
    ]
    if arm == "planner-native":
        images.append(
            {
                "pod": "qwen3-0-6b-batch-planner-abc",
                "phase": "Running",
                "container": "main",
                "image": "registry.example/dynamo:test",
                "image_id": "containerd://planner-image-digest",
                "ready": True,
                "restart_count": 0,
            }
        )
    return images


def _write_runtime_images(evidence_root: Path, arm: str) -> None:
    images = _runtime_images(arm)
    for phase in ("start", "end"):
        _write_json(evidence_root / "kubernetes" / phase / "images.json", images)


def _run_evidence_root(run_dir: Path) -> Path:
    arm = json.loads((run_dir / "metadata.json").read_text())["arm"]
    if arm == "online-only":
        return run_dir / "online-evidence"
    return next((run_dir / "batch-harness" / "results" / "raw").iterdir())


def _redact_image_id(run_dir: Path, pod_prefix: str) -> None:
    evidence_root = _run_evidence_root(run_dir)
    metadata_path = (
        run_dir / "metadata.json"
        if evidence_root.name == "online-evidence"
        else evidence_root / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    for phase in ("start", "end"):
        path = evidence_root / "kubernetes" / phase / "images.json"
        images = json.loads(path.read_text())
        for item in images:
            if item["pod"].startswith(pod_prefix):
                item["image_id"] = "<redacted-url>"
        _write_json(path, images)
        for pod in metadata[f"kubernetes_{phase}"]["pod_status"]:
            if pod["name"].startswith(pod_prefix):
                pod["image_ids"] = ["<redacted-url>"]
    _write_json(metadata_path, metadata)
    _write_outer_manifest(run_dir)


def _write_frontend_metrics(evidence_root: Path) -> None:
    payload = (
        "process_start_time_seconds 1750000000\n"
        'dynamo_frontend_model_ready{model="Qwen/Qwen3-0.6B"} 1\n'
    )
    metrics = evidence_root / "metrics" / "frontend"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "20260918T000010.000000Z.prom").write_text(payload, encoding="utf-8")
    (metrics / "20260918T000110.000000Z.prom").write_text(payload, encoding="utf-8")


def _write_metric_snapshot(
    directory: Path,
    observed_at: dt.datetime,
    payload: str,
    *,
    url: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = observed_at.strftime("%Y%m%dT%H%M%S.%fZ")
    (directory / f"{stem}.prom").write_text(payload, encoding="utf-8")
    _write_json(
        directory / f"{stem}.json",
        {
            "observed_at": _timestamp(observed_at),
            "url": url,
            "http_status": 200,
        },
    )


def _write_native_control_evidence(
    evidence_root: Path,
    online_start: dt.datetime,
    *,
    high_cap: float = 0.0,
    final_cap: float = 0.5,
    lease_valid: bool = True,
    lease_expired: bool = False,
    interval_seconds: int = 5,
) -> None:
    frontend_counter = 1_000.0
    planner_lines: list[str] = []
    for elapsed in range(0, 181, interval_seconds):
        observed_at = online_start + dt.timedelta(seconds=elapsed)
        if elapsed:
            prior = elapsed - interval_seconds
            rate = 5.0 if prior < 60 or prior >= 120 else 8.0
            frontend_counter += rate * interval_seconds
        frontend_payload = (
            "process_start_time_seconds 1750000000\n"
            'dynamo_frontend_model_ready{model="Qwen/Qwen3-0.6B"} 1\n'
            "dynamo_frontend_requests_started_total{"
            'model="Qwen/Qwen3-0.6B",endpoint="chat_completions",'
            f'request_type="stream"}} {frontend_counter}\n'
        )
        _write_metric_snapshot(
            evidence_root / "metrics" / "frontend",
            observed_at,
            frontend_payload,
            url="http://frontend:8081/metrics",
        )

        cap = 0.5 if elapsed <= 60 else high_cap if elapsed <= 120 else final_cap
        expiry = observed_at.timestamp() + (-1.0 if lease_expired else 90.0)
        async_payload = (
            'llm_d_async_async_broker_backlog{pool_name="dynamo-batch"} 1000\n'
            "llm_d_async_async_broker_backlog_source_available{"
            'pool_name="dynamo-batch"} 1\n'
            'llm_d_async_async_queue_depth{pool_name="dynamo-batch"} 3\n'
            "llm_d_async_async_drain_limit_rps{"
            f'pool_name="dynamo-batch"}} {cap}\n'
            "llm_d_async_async_drain_limit_lease_valid{"
            f'pool_name="dynamo-batch"}} {1 if lease_valid else 0}\n'
            "llm_d_async_async_drain_limit_valid_until_seconds{"
            f'pool_name="dynamo-batch"}} {expiry}\n'
        )
        _write_metric_snapshot(
            evidence_root / "metrics" / "async",
            observed_at,
            async_payload,
            url="http://async:9090/metrics",
        )
        if elapsed:
            planner_lines.append(
                f"{_timestamp(observed_at)} INFO Batch scheduling decision: "
                "pipeline_action=noop pool_id=dynamo-batch replica_floor=1 "
                f"max_admission_rps={cap} decision_id=decision-{elapsed} "
                f"valid_until_s={observed_at.timestamp() + 90.0}\n"
            )
    log_path = (
        evidence_root
        / "kubernetes"
        / "end"
        / "logs-qwen3-0-6b-batch-planner-abc.stdout"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(planner_lines), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    arm: str,
    repetition: int,
    goodput: float,
    p99_ttft: float,
    errors: int,
    nonoverlap: bool = False,
    native_high_cap: float = 0.0,
    native_final_cap: float = 0.5,
    native_lease_valid: bool = True,
    native_lease_expired: bool = False,
    native_metrics_interval_seconds: int = 5,
) -> Path:
    arm_index = compiler.ARMS.index(arm)
    hour = repetition * 3 + arm_index
    started = dt.datetime(2026, 9, 18, hour, tzinfo=dt.timezone.utc)
    run_id = (
        started.strftime("%Y%m%dT%H%M%SZ")
        + f"-planner-gym-batch-impact-{repetition}{arm_index}abcd"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    online_start = started + dt.timedelta(seconds=200 if nonoverlap else 10)
    online_end = online_start + dt.timedelta(seconds=180)
    batch_required = arm != "online-only"
    synchronization = {
        "batch_required": batch_required,
        "batch_id": f"batch-{repetition}-{arm_index}" if batch_required else None,
        "batch_created_observed_at": (
            _timestamp(started + dt.timedelta(seconds=5)) if batch_required else None
        ),
        "batch_active_observed_at": (
            _timestamp(started + dt.timedelta(seconds=6)) if batch_required else None
        ),
        "batch_active_status": "in_progress" if batch_required else None,
        "batch_dispatch_active_at": (
            _timestamp(started + dt.timedelta(seconds=6, milliseconds=500))
            if batch_required
            else None
        ),
        "batch_dispatch_active_observed_at": (
            _timestamp(started + dt.timedelta(seconds=7)) if batch_required else None
        ),
        "batch_dispatch_active_status": ("in_progress" if batch_required else None),
        "batch_dispatch_completed": 1 if batch_required else None,
        "batch_dispatch_failed": 0 if batch_required else None,
        "batch_dispatch_total": 100 if batch_required else None,
        "planner_gym_started_at": _timestamp(online_start),
    }
    metadata = {
        "schema_version": "1.0",
        "run_id": run_id,
        "kind": "planner-gym-batch-impact",
        "arm": arm,
        "status": "completed",
        "exit_code": 0,
        "started": _timestamp(started),
        "ended": _timestamp(online_end + dt.timedelta(seconds=1)),
        "configuration": {
            "model": "Qwen/Qwen3-0.6B",
            "workload": "bounded-impact",
            "seed": 17,
            "max_requests": 100,
            "expected_worker_pool_id": "dynamo-batch",
        },
        "source": {
            "dynamo": {"revision": "dynamo-commit"},
            "planner_gym": {"revision": "gym-commit"},
        },
        "inputs": {"dataset": {"sha256": "dataset-sha"}},
        "synchronization": synchronization,
    }
    if arm == "online-only":
        metadata.update(_runtime_evidence(arm))
        _write_frontend_metrics(run_dir / "online-evidence")
        _write_runtime_images(run_dir / "online-evidence", arm)
    _write_json(
        run_dir / "metadata.json",
        metadata,
    )

    successful = 100.0
    error_rate_percent = 100.0 * errors / (successful + errors)
    good_rate = (successful - 5.0) / (successful + errors)
    metrics = {
        "goodput_rps": goodput,
        "good_rate": good_rate,
        "request_throughput_rps": 1.0,
        "completed_requests": successful,
        "duration_s": 180.0,
        "mean_ttft_ms": p99_ttft / 2.0,
        "p99_ttft_ms": p99_ttft,
        "mean_itl_ms": 20.0,
        "p99_itl_ms": 30.0,
        "mean_e2e_ms": 500.0,
        "p99_e2e_ms": 700.0,
    }
    raw_metrics = {
        "goodput_rps": goodput,
        "good_request_fraction": good_rate,
        "request_throughput_rps": 1.0,
        "request_count": successful,
        "benchmark_duration_s": 180.0,
        "mean_ttft_ms": p99_ttft / 2.0,
        "p99_ttft_ms": p99_ttft,
        "mean_itl_ms": 20.0,
        "p99_itl_ms": 30.0,
        "mean_e2e_ms": 500.0,
        "p99_e2e_ms": 700.0,
    }
    evaluation = {
        "workload": "bounded-impact",
        "seed": 17,
        "repetition": 0,
        "max_requests": 100,
        "arrival_speedup": 1.0,
        "trace_block_size": 512,
        "trace": {
            "sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "size_bytes": 1234,
        },
        "sla": {
            "name": "impact-slo",
            "ttft_ms": 2000.0,
            "itl_ms": 100.0,
            "e2e_ms": None,
        },
    }
    _write_json(
        run_dir / "planner-gym" / "results.json",
        {
            "match": {"name": f"impact-{arm}", "backend": "real"},
            "summary": {"status": "ok"},
            "provenance": {
                "schema_version": 1,
                "config_sha256": "c" * 64,
                "git_commit": "gym-commit",
                "started_at": _timestamp(online_start),
                "finished_at": _timestamp(online_end),
            },
            "results": [
                {
                    "run_id": f"one-cell-{arm}",
                    "status": "ok",
                    "metrics": metrics,
                    "raw_metrics": raw_metrics,
                    "evaluation": evaluation,
                }
            ],
        },
    )
    aiperf = {
        "request_count": _metric(successful),
        "request_error_rate": _metric(error_rate_percent),
    }
    if errors:
        aiperf["error_request_count"] = _metric(float(errors))
    _write_json(
        run_dir
        / "planner-gym"
        / "artifacts"
        / "session"
        / "profile_export_aiperf.json",
        aiperf,
    )

    if arm != "online-only":
        child_kind = "baseline" if arm == "stock" else "planner-native"
        child_id = f"20260918T{hour:02d}0001Z-{child_kind}-{repetition}{arm_index}cdef"
        child = run_dir / "batch-harness" / "results" / "raw" / child_id
        child_metadata = {
            "schema_version": "1.0",
            "run_id": child_id,
            "kind": child_kind,
            "requested_run_kind": child_kind,
            "status": "completed",
            "exit_code": 0,
            "source": {"revision": "dynamo-commit"},
            "configuration": {
                "model": "Qwen/Qwen3-0.6B",
                "batch_size": 100,
                "start_index": 0,
                "max_tokens": 128,
                "temperature": 0.0,
            },
            "inputs": {
                "dataset_sha256": "dataset-sha",
                "submitted_sha256": f"{child_kind}-submitted-sha",
            },
            "control_plane": {"mode": child_kind},
        }
        child_metadata.update(
            _runtime_evidence(
                arm,
                native_metrics_interval_seconds=native_metrics_interval_seconds,
            )
        )
        if arm == "planner-native":
            _write_native_control_evidence(
                child,
                online_start,
                high_cap=native_high_cap,
                final_cap=native_final_cap,
                lease_valid=native_lease_valid,
                lease_expired=native_lease_expired,
                interval_seconds=native_metrics_interval_seconds,
            )
        else:
            _write_frontend_metrics(child)
        _write_runtime_images(child, arm)
        _write_json(child / "metadata.json", child_metadata)
        _write_json(
            child / "workload-manifest.json",
            {
                "schema_version": "1.0",
                "source": "gsm8k.jsonl",
                "source_sha256": "dataset-sha",
                "output": "batch-input.jsonl",
                # The normalized output differs intentionally because the harness
                # gives each arm a distinct submitted custom_id prefix.
                "output_sha256": f"{child_kind}-submitted-sha",
                "run_kind": child_kind,
                "custom_id_prefix": f"gsm8k-{child_kind}-",
                "selection": {
                    "start_index": 0,
                    "count": 100,
                    "order": "source file order",
                },
                "fixed_request_configuration": {
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "model": "Qwen/Qwen3-0.6B",
                    "max_tokens": 128,
                    "temperature": 0.0,
                    "stream": False,
                },
                "id_mapping": [
                    {
                        "source_custom_id": "source-000001",
                        "submitted_custom_id": f"gsm8k-{child_kind}-000001",
                    }
                ],
            },
        )
        _write_json(
            child / "result-validation.json",
            {
                "expected_total": 100,
                "completed": 100,
                "failed": 0,
                "valid": True,
            },
        )
        _write_jsonl(
            child / "progress.jsonl",
            [
                {
                    "observed_at": _timestamp(started + dt.timedelta(seconds=6)),
                    "elapsed_seconds": 1.0,
                    "status": "in_progress",
                    "total": 100,
                    "completed": 1,
                    "failed": 0,
                    "interval_completion_rate_rps": 1.0,
                },
                {
                    "observed_at": _timestamp(started + dt.timedelta(seconds=190)),
                    "elapsed_seconds": 185.0,
                    "status": "in_progress",
                    "total": 100,
                    "completed": 91,
                    "failed": 0,
                    "interval_completion_rate_rps": 90.0 / 184.0,
                },
                {
                    "observed_at": _timestamp(started + dt.timedelta(seconds=200)),
                    "elapsed_seconds": 195.0,
                    "status": "completed",
                    "total": 100,
                    "completed": 100,
                    "failed": 0,
                    "interval_completion_rate_rps": 9.0 / 10.0,
                },
            ],
        )
    _write_outer_manifest(run_dir)
    return run_dir


def _balanced_campaign(root: Path) -> list[Path]:
    runs: list[Path] = []
    values = {
        "online-only": ([9.9, 10.0, 10.1], 100.0, 0),
        "stock": ([7.9, 8.0, 8.1], 160.0, 2),
        "planner-native": ([9.4, 9.5, 9.6], 115.0, 1),
    }
    for repetition in range(3):
        for arm in compiler.ARMS:
            goodputs, p99_ttft, errors = values[arm]
            runs.append(
                _make_run(
                    root,
                    arm=arm,
                    repetition=repetition,
                    goodput=goodputs[repetition],
                    p99_ttft=p99_ttft,
                    errors=errors,
                )
            )
    return runs


def _cli_args(runs: list[Path], output: Path, *extra: str) -> list[str]:
    args: list[str] = []
    for run in runs:
        args.extend(["--run-directory", str(run)])
    args.extend(["--output-directory", str(output), *extra])
    return args


def test_compile_campaign_summarizes_arms_pairs_batch_and_provenance(
    tmp_path: Path,
) -> None:
    runs = _balanced_campaign(tmp_path / "raw")

    report = compiler.compile_campaign(runs)
    output = compiler.write_campaign(report, tmp_path / "compiled" / "impact")

    assert report["status"] == "valid"
    assert report["campaign"]["valid_run_count"] == 9
    assert report["arms"]["stock"]["valid_run_count"] == 3
    assert report["arms"]["stock"]["online"]["goodput_rps"]["mean"] == pytest.approx(
        8.0
    )
    assert report["arms"]["stock"]["online"]["error_rate"]["mean"] == pytest.approx(
        2 / 102
    )
    assert report["arms"]["stock"]["batch"]["completion_rate_during_online_rps"][
        "mean"
    ] == pytest.approx(0.5)

    comparison = report["comparisons"]["planner-native_vs_stock"]
    assert comparison["valid_pair_count"] == 3
    assert comparison["online"]["goodput_rps"]["treatment_minus_reference"][
        "mean"
    ] == pytest.approx(1.5)
    assert comparison["online"]["p99_ttft_ms"]["improvement"]["mean"] == pytest.approx(
        45.0
    )
    assert report["stopping_assessment"]["extra_repetitions_recommended"] is False
    assert report["stopping_assessment"]["target_valid_repetitions_per_arm"] == 3
    assert report["runs"][0]["source_files"]["metadata.json"]["sha256"]
    assert (
        report["runs"][0]["provenance"]["outer_artifact_manifest"][
            "verified_source_file_count"
        ]
        >= 3
    )
    assert report["provenance"]["compiler"]["sha256"]
    stock_run = next(item for item in report["runs"] if item["arm"] == "stock")
    native_run = next(
        item for item in report["runs"] if item["arm"] == "planner-native"
    )
    assert stock_run["batch"]["created_observed_at"].endswith("05+00:00")
    assert stock_run["batch"]["in_progress_observed_at"].endswith("06+00:00")
    assert stock_run["batch"]["active_started_at"].endswith("07+00:00")
    assert stock_run["batch"]["active_status"] == "in_progress"
    assert stock_run["batch"]["dispatch_active_counts"] == {
        "completed": 1,
        "failed": 0,
        "total": 100,
    }
    assert stock_run["deployment_signature_sha256"]
    assert stock_run["serving_signature_sha256"]
    assert stock_run["async_signature_sha256"]
    assert stock_run["planner_signature_sha256"] is None
    assert native_run["planner_signature_sha256"]
    assert stock_run["evidence"]["deployment_identity"]["control_plane"] == {
        "async": {
            "image_ref": "ghcr.io/llm-d/llm-d-async:v0.10.0",
            "image_id": "containerd://async-image-digest",
            "observed_image_id": "containerd://async-image-digest",
            "image_id_attested": False,
        }
    }
    assert native_run["evidence"]["deployment_identity"]["control_plane"][
        "planner"
    ] == {
        "image_ref": "registry.example/dynamo:test",
        "image_id": "containerd://planner-image-digest",
        "observed_image_id": "containerd://planner-image-digest",
        "image_id_attested": False,
    }
    assert report["campaign"]["canonical_serving_signature_sha256"]
    assert report["campaign"]["canonical_batch_workload_signature_sha256"]
    assert (
        stock_run["batch_workload_signature_sha256"]
        == native_run["batch_workload_signature_sha256"]
    )
    assert set(report["campaign"]["canonical_async_signature_sha256_by_arm"]) == set(
        compiler.ARMS
    )
    assert set(
        report["campaign"]["canonical_control_plane_signature_sha256_by_arm"]
    ) == set(compiler.ARMS)
    assert set(report["campaign"]["canonical_planner_signature_sha256_by_arm"]) == {
        "planner-native"
    }
    assert report["campaign"]["canonical_image_identities"]["serving"]
    assert stock_run["evidence"]["frontend_metric_snapshot_count"] == 2
    assert any(name.endswith(".prom") for name in stock_run["source_files"])
    control = native_run["evidence"]["batch_drain_control"]
    assert control["valid"] is True
    assert control["phases"]["low_1"]["planner_desired_cap_rps"]["median"] == 0.5
    assert control["phases"]["high"]["planner_desired_cap_rps"]["median"] == 0.0
    assert control["phases"]["low_2"]["async_applied_cap_rps"]["median"] == 0.5
    assert control["checks"]["planner_cap_decreased_for_high_load"] is True
    assert control["checks"]["async_cap_resumed_for_final_low_load"] is True

    published = json.loads((output / "summary.json").read_text())
    assert published["campaign"]["valid_run_count"] == 9
    assert "Paired online effects" in (output / "README.md").read_text()
    assert not list((tmp_path / "compiled").glob(".impact.staging-*"))


def test_compile_campaign_rejects_mismatched_batch_workloads(tmp_path: Path) -> None:
    runs = _balanced_campaign(tmp_path / "raw")
    native_runs = [
        run
        for run in runs
        if json.loads((run / "metadata.json").read_text())["arm"] == "planner-native"
    ]
    for run in native_runs:
        child = _run_evidence_root(run)
        manifest_path = child / "workload-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fixed_request_configuration"]["max_tokens"] = 256
        manifest["output_sha256"] = "planner-native-different-submitted-sha"
        _write_json(manifest_path, manifest)

        metadata_path = child / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["configuration"]["max_tokens"] = 256
        metadata["inputs"][
            "submitted_sha256"
        ] = "planner-native-different-submitted-sha"
        _write_json(metadata_path, metadata)
        _write_outer_manifest(run)

    report = compiler.compile_campaign(runs)

    assert report["status"] == "invalid-inputs-present"
    comparison = report["comparisons"]["planner-native_vs_stock"]
    assert comparison["valid_pair_count"] == 0
    assert all(
        "Batch workload signatures differ" in pair["invalid_reasons"]
        for pair in comparison["pairs"]
    )
    batch_signatures = {
        record["arm"]: record["batch_workload_signature_sha256"]
        for record in report["runs"]
        if record["arm"] in {"stock", "planner-native"}
    }
    assert batch_signatures["stock"] != batch_signatures["planner-native"]


def test_compile_campaign_requires_async_and_native_planner_image_identity(
    tmp_path: Path,
) -> None:
    stock = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    stock_root = _run_evidence_root(stock)
    for phase in ("start", "end"):
        path = stock_root / "kubernetes" / phase / "images.json"
        images = json.loads(path.read_text())
        _write_json(
            path,
            [
                item
                for item in images
                if not item["pod"].startswith(compiler.ASYNC_POD_PREFIX)
            ],
        )
    _write_outer_manifest(stock)

    native = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
    )
    native_root = _run_evidence_root(native)
    end_images_path = native_root / "kubernetes" / "end" / "images.json"
    end_images = json.loads(end_images_path.read_text())
    for item in end_images:
        if item["pod"].startswith(compiler.PLANNER_POD_PREFIX):
            item["image"] = "registry.example/dynamo:unexpected"
    _write_json(end_images_path, end_images)
    _write_outer_manifest(native)

    report = compiler.compile_campaign([stock, native])
    by_arm = {record["arm"]: record for record in report["runs"]}
    assert any(
        "Async pod" in reason and "image record" in reason
        for reason in by_arm["stock"]["invalid_reasons"]
    )
    assert any(
        "Planner image identity changed" in reason
        for reason in by_arm["planner-native"]["invalid_reasons"]
    )


def test_compile_campaign_attests_sanitized_image_id_and_records_provenance(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
    )
    _redact_image_id(run, compiler.PLANNER_POD_PREFIX)

    report = compiler.compile_campaign(
        [run],
        image_id_attestations={
            "registry.example/dynamo:test": "sha256:attested-planner-image"
        },
    )

    compiled = report["runs"][0]
    assert compiled["valid"] is True
    planner = compiled["evidence"]["deployment_identity"]["control_plane"]["planner"]
    assert planner == {
        "image_ref": "registry.example/dynamo:test",
        "image_id": "sha256:attested-planner-image",
        "observed_image_id": "<redacted-url>",
        "image_id_attested": True,
    }
    assert report["provenance"]["image_id_attestations"] == [
        {
            "image_ref": "registry.example/dynamo:test",
            "image_id": "sha256:attested-planner-image",
            "used_for_sanitized_observation": True,
        }
    ]


def test_compile_campaign_rejects_missing_and_mismatched_image_attestations(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
    )
    _redact_image_id(run, compiler.PLANNER_POD_PREFIX)

    missing = compiler.compile_campaign([run])
    assert missing["runs"][0]["valid"] is False
    assert any(
        "provide --image-id-attestation" in reason
        for reason in missing["runs"][0]["invalid_reasons"]
    )

    with pytest.raises(compiler.CampaignCompileError, match="was not used"):
        compiler.compile_campaign(
            [run],
            image_id_attestations={"registry.example/dynamo:wrong": "sha256:wrong-ref"},
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["ref=id", "ref=id"], "duplicate"),
        (["ref=id-one", "ref=id-two"], "conflicting"),
        (["missing-separator"], "must use"),
    ],
)
def test_parse_image_id_attestations_rejects_invalid_entries(
    values: list[str], message: str
) -> None:
    with pytest.raises(compiler.CampaignCompileError, match=message):
        compiler._parse_image_id_attestations(values)


def test_compile_campaign_enforces_shared_serving_and_per_arm_control_plane(
    tmp_path: Path,
) -> None:
    runs = _balanced_campaign(tmp_path / "raw")
    online_outlier = next(run for run in runs if "-00abcd" in run.name)
    for phase in ("start", "end"):
        path = _run_evidence_root(online_outlier) / "kubernetes" / phase / "images.json"
        images = json.loads(path.read_text())
        for item in images:
            if item["pod"].startswith(compiler.FRONTEND_POD_PREFIX):
                item["image"] = "registry.example/dynamo:serving-outlier"
        _write_json(path, images)
    _write_outer_manifest(online_outlier)

    stock_runs = [
        run
        for run in runs
        if json.loads((run / "metadata.json").read_text())["arm"] == "stock"
    ]
    async_outlier = stock_runs[0]
    for phase in ("start", "end"):
        path = _run_evidence_root(async_outlier) / "kubernetes" / phase / "images.json"
        images = json.loads(path.read_text())
        for item in images:
            if item["pod"].startswith(compiler.ASYNC_POD_PREFIX):
                item["image"] = "ghcr.io/llm-d/llm-d-async:outlier"
        _write_json(path, images)
    _write_outer_manifest(async_outlier)

    report = compiler.compile_campaign(runs)
    compiled_online = next(
        record for record in report["runs"] if record["run_id"] == online_outlier.name
    )
    compiled_stock = next(
        record for record in report["runs"] if record["run_id"] == async_outlier.name
    )
    assert any(
        "frontend/worker image identity differs" in reason
        for reason in compiled_online["invalid_reasons"]
    )
    assert any(
        "Async image identity differs from the stock arm majority" in reason
        for reason in compiled_stock["invalid_reasons"]
    )


def test_compile_campaign_excludes_missing_and_nonoverlapping_runs(
    tmp_path: Path,
) -> None:
    baseline = _make_run(
        tmp_path,
        arm="online-only",
        repetition=0,
        goodput=10.0,
        p99_ttft=100.0,
        errors=0,
    )
    missing = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    (missing / "planner-gym" / "results.json").unlink()
    nonoverlap = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
        nonoverlap=True,
    )

    report = compiler.compile_campaign([baseline, missing, nonoverlap])

    assert report["status"] == "invalid-inputs-present"
    assert report["campaign"]["valid_run_count"] == 1
    reasons = {
        issue["arm"]: "; ".join(issue["reasons"])
        for issue in report["data_quality"]["issues"]
    }
    assert "missing planner-gym/results.json" in reasons["stock"]
    assert "do not overlap" in reasons["planner-native"]
    assert report["comparisons"]["stock_vs_online-only"]["valid_pair_count"] == 0
    assert report["stopping_assessment"]["extra_repetitions_recommended"] is True
    markdown = compiler.render_markdown(report)
    assert "- comparison `stock_vs_online-only`: ordinal 1:" in markdown
    assert "- `<missing>` (None): ordinal 1:" not in markdown
    insufficient = [
        finding
        for finding in report["stopping_assessment"]["reasons"]
        if finding["code"] == "insufficient-valid-repetitions"
    ]
    assert {finding["arm"] for finding in insufficient} == set(compiler.ARMS)


def test_native_control_accepts_rfc3339nano_planner_timestamps(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
    )
    log_path = (
        _run_evidence_root(run)
        / "kubernetes"
        / "end"
        / "logs-qwen3-0-6b-batch-planner-abc.stdout"
    )
    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace("Z INFO", ".123456789Z INFO"),
        encoding="utf-8",
    )
    _write_outer_manifest(run)

    report = compiler.compile_campaign([run])

    compiled = report["runs"][0]
    assert compiled["valid"] is True
    assert compiled["evidence"]["batch_drain_control"]["valid"] is True


@pytest.mark.parametrize(
    ("kwargs", "failed_check"),
    [
        (
            {"native_high_cap": 0.5},
            "planner_cap_decreased_for_high_load",
        ),
        (
            {"native_final_cap": 0.0},
            "planner_cap_resumed_for_final_low_load",
        ),
    ],
    ids=("no-decrease", "no-resume"),
)
def test_native_control_requires_decrease_and_resume_in_both_surfaces(
    tmp_path: Path,
    kwargs: dict[str, float],
    failed_check: str,
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
        **kwargs,
    )

    report = compiler.compile_campaign([run])

    compiled = report["runs"][0]
    assert compiled["valid"] is False
    control = compiled["evidence"]["batch_drain_control"]
    assert control["valid"] is False
    assert failed_check in control["failed_checks"]
    assert any(
        "native batch drain control evidence failed checks" in reason
        for reason in compiled["invalid_reasons"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"native_lease_valid": False},
        {"native_lease_expired": True},
    ],
    ids=("invalid-lease", "expired-lease"),
)
def test_native_control_requires_valid_unexpired_async_leases(
    tmp_path: Path, kwargs: dict[str, bool]
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
        **kwargs,
    )

    report = compiler.compile_campaign([run])

    compiled = report["runs"][0]
    assert compiled["valid"] is False
    control = compiled["evidence"]["batch_drain_control"]
    assert control["valid"] is False
    assert "high_async_leases_valid_and_unexpired" in control["failed_checks"]


def test_native_control_requires_three_samples_per_stable_phase(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="planner-native",
        repetition=0,
        goodput=9.5,
        p99_ttft=115.0,
        errors=1,
        native_metrics_interval_seconds=15,
    )

    report = compiler.compile_campaign([run])

    compiled = report["runs"][0]
    assert compiled["valid"] is False
    control = compiled["evidence"]["batch_drain_control"]
    assert control["valid"] is False
    assert "only 2 stable high frontend samples" in control["error"]


def test_stopping_assessment_flags_high_cv_and_effect_direction_changes(
    tmp_path: Path,
) -> None:
    runs: list[Path] = []
    stock_goodput = [5.0, 15.0, 5.0]
    for repetition in range(3):
        runs.extend(
            [
                _make_run(
                    tmp_path,
                    arm="online-only",
                    repetition=repetition,
                    goodput=10.0,
                    p99_ttft=100.0,
                    errors=0,
                ),
                _make_run(
                    tmp_path,
                    arm="stock",
                    repetition=repetition,
                    goodput=stock_goodput[repetition],
                    p99_ttft=160.0,
                    errors=2,
                ),
                _make_run(
                    tmp_path,
                    arm="planner-native",
                    repetition=repetition,
                    goodput=9.5,
                    p99_ttft=115.0,
                    errors=1,
                ),
            ]
        )

    report = compiler.compile_campaign(runs)
    findings = report["stopping_assessment"]["reasons"]

    assert report["arms"]["stock"]["online"]["goodput_rps"]["cv_percent"] > 5.0
    assert (
        report["comparisons"]["stock_vs_online-only"]["online"]["goodput_rps"][
            "direction_changed_across_repetitions"
        ]
        is True
    )
    assert any(
        finding["code"] == "cv-above-threshold"
        and finding.get("arm") == "stock"
        and finding.get("metric") == "goodput_rps"
        for finding in findings
    )
    assert any(
        finding["code"] == "paired-effect-direction-changed"
        and finding.get("comparison") == "stock_vs_online-only"
        and finding.get("metric") == "goodput_rps"
        for finding in findings
    )
    assert report["stopping_assessment"]["instability_triggered"] is True
    assert report["stopping_assessment"]["target_valid_repetitions_per_arm"] == 5
    assert report["stopping_assessment"]["minimum_additional_balanced_repetitions"] == 2


def test_write_campaign_refuses_to_overwrite_existing_analysis(
    tmp_path: Path,
) -> None:
    report = compiler.compile_campaign(_balanced_campaign(tmp_path / "raw"))
    output = tmp_path / "compiled"
    compiler.write_campaign(report, output)

    with pytest.raises(FileExistsError):
        compiler.write_campaign(report, output)

    assert json.loads((output / "summary.json").read_text())["schema_version"] == "1.0"


def test_compile_campaign_flags_artifact_changed_after_finalization(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="online-only",
        repetition=0,
        goodput=10.0,
        p99_ttft=100.0,
        errors=0,
    )
    results_path = run / "planner-gym" / "results.json"
    results_path.write_text(results_path.read_text() + "\n", encoding="utf-8")

    report = compiler.compile_campaign([run])

    compiled_run = next(item for item in report["runs"] if item["run_id"] == run.name)
    assert compiled_run["valid"] is False
    assert any(
        "artifact checksum manifest disagrees" in reason
        for reason in compiled_run["invalid_reasons"]
    )


def test_compile_campaign_rejects_short_batch_exposure_by_default(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    started = compiler._parse_time(
        json.loads((run / "metadata.json").read_text())["started"], "started"
    )
    progress_path = next(
        (run / "batch-harness" / "results" / "raw").glob("*/progress.jsonl")
    )
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    progress[1].update(
        observed_at=_timestamp(started + dt.timedelta(seconds=30)),
        elapsed_seconds=25.0,
    )
    progress[2].update(
        observed_at=_timestamp(started + dt.timedelta(seconds=82)),
        elapsed_seconds=77.0,
    )
    _write_jsonl(progress_path, progress)
    _write_outer_manifest(run)

    strict = compiler.compile_campaign([run])
    permissive = compiler.compile_campaign([run], minimum_batch_overlap_fraction=0.3)

    strict_run = strict["runs"][0]
    assert strict_run["valid"] is False
    assert any("overlap fraction" in reason for reason in strict_run["invalid_reasons"])
    assert permissive["runs"][0]["valid"] is True
    assert permissive["runs"][0]["batch"][
        "overlap_fraction_of_online"
    ] == pytest.approx(0.4)


def test_compile_campaign_requires_observed_in_progress_synchronization(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["synchronization"]["batch_active_status"] = "validating"
    _write_json(metadata_path, metadata)
    _write_outer_manifest(run)

    report = compiler.compile_campaign([run])

    assert report["runs"][0]["valid"] is False
    assert any(
        "does not prove an in_progress Batch job" in reason
        for reason in report["runs"][0]["invalid_reasons"]
    )


def test_compile_campaign_requires_positive_dispatch_synchronization(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["synchronization"]["batch_dispatch_completed"] = 0
    _write_json(metadata_path, metadata)
    _write_outer_manifest(run)

    report = compiler.compile_campaign([run])

    assert report["runs"][0]["valid"] is False
    assert any(
        "do not prove positive bounded progress" in reason.lower()
        for reason in report["runs"][0]["invalid_reasons"]
    )


def test_compile_campaign_rejects_dispatch_observed_after_online_start(
    tmp_path: Path,
) -> None:
    run = _make_run(
        tmp_path,
        arm="stock",
        repetition=0,
        goodput=8.0,
        p99_ttft=160.0,
        errors=2,
    )
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    online_started = compiler._parse_time(
        json.loads((run / "planner-gym" / "results.json").read_text())["provenance"][
            "started_at"
        ],
        "online started",
    )
    metadata["synchronization"]["batch_dispatch_active_observed_at"] = _timestamp(
        online_started + dt.timedelta(seconds=1)
    )
    _write_json(metadata_path, metadata)
    _write_outer_manifest(run)

    report = compiler.compile_campaign([run])

    assert report["runs"][0]["valid"] is False
    assert any(
        "follows the online interval start" in reason
        for reason in report["runs"][0]["invalid_reasons"]
    )


def test_cli_fails_closed_for_invalid_or_incomplete_campaigns(
    tmp_path: Path,
) -> None:
    runs: list[Path] = []
    for repetition in range(2):
        for arm, goodput, ttft, errors in (
            ("online-only", 10.0, 100.0, 0),
            ("stock", 8.0, 160.0, 2),
            ("planner-native", 9.5, 115.0, 1),
        ):
            runs.append(
                _make_run(
                    tmp_path / "raw",
                    arm=arm,
                    repetition=repetition,
                    goodput=goodput,
                    p99_ttft=ttft,
                    errors=errors,
                )
            )

    incomplete_output = tmp_path / "incomplete"
    assert compiler.main(_cli_args(runs, incomplete_output)) == 2
    incomplete = json.loads((incomplete_output / "summary.json").read_text())
    assert incomplete["status"] == "valid"
    assert (
        incomplete["stopping_assessment"]["minimum_additional_balanced_repetitions"]
        == 1
    )

    invalid_runs = [runs[0]]
    invalid_output = tmp_path / "invalid"
    assert compiler.main(_cli_args(invalid_runs, invalid_output)) == 2
    assert (
        json.loads((invalid_output / "summary.json").read_text())["status"]
        == "invalid-inputs-present"
    )

    accepted_output = tmp_path / "accepted-invalid"
    assert (
        compiler.main(
            _cli_args(
                invalid_runs,
                accepted_output,
                "--allow-invalid-or-incomplete",
            )
        )
        == 0
    )
