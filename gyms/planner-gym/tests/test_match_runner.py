# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure and mocked contract tests for Match Config execution and publication."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from autoscaling_arena import match_runner
from autoscaling_arena.html_report import _replay_command
from autoscaling_arena.match_config import (
    EngineConfig,
    EvaluationConfig,
    ExecutionConfig,
    MatchConfig,
    MatchRun,
    MetricsConfig,
    ModelConfig,
    PublishConfig,
    PublishDestination,
    ReplayConfig,
    ReplicaCounts,
    RouterConfig,
    SimAutoscalerConfig,
    SimBackendConfig,
    SimEnginesConfig,
    SLAProfileConfig,
    load_match_config,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

ARENA_ROOT = Path(__file__).resolve().parents[1]
RUN_MATCH_CONFIG = ARENA_ROOT / "scripts" / "run_match_config.py"
IDENTITY_FIELDS = (
    "run_id",
    "backend",
    "autoscaler",
    "workload",
    "sla",
    "repetition",
    "seed",
)


def test_prefix_cache_telemetry_is_token_weighted_and_completion_attributed():
    report = SimpleNamespace(
        trace_report={
            "prefix_cache_reused_ratio": 0.5,
            "first_admission_prefix_cache_reused_ratio": 0.4,
            "total_input_tokens": 300,
        },
        per_request=[
            {
                "terminal_status": "completed",
                "first_admit_ms": 100.0,
                "terminal_time_ms": 500.0,
                "input_length": 100,
                "reused_input_tokens": 20,
                "decode_reused_input_tokens": 50,
            },
            {
                "terminal_status": "completed",
                "first_admit_ms": 200.0,
                "terminal_time_ms": 1_200.0,
                "input_length": 200,
                "reused_input_tokens": 100,
                "decode_reused_input_tokens": None,
            },
            {
                "terminal_status": "rejected",
                "first_admit_ms": 300.0,
                "terminal_time_ms": 900.0,
                "input_length": 1_000,
                "reused_input_tokens": 1_000,
            },
        ],
    )

    cache = match_runner._prefix_cache_telemetry(report)

    assert cache["prefix_cache_reused_ratio"] == pytest.approx(0.5)
    assert cache["first_admission_prefix_cache_reused_ratio"] == pytest.approx(0.4)
    assert cache["reused_input_tokens"] == 150
    assert cache["timeline_available"] is True
    assert cache["timeline_source"] == "completed_requests"
    assert cache["timeline"] == [
        {
            "window_start_s": 0.0,
            "time_s": 1.0,
            "input_tokens": 100,
            "reused_input_tokens": 50,
            "completed_requests": 1,
            "prefix_cache_reused_ratio": 0.5,
        },
        {
            "window_start_s": 1.0,
            "time_s": 2.0,
            "input_tokens": 200,
            "reused_input_tokens": 100,
            "completed_requests": 1,
            "prefix_cache_reused_ratio": 0.5,
        },
    ]


def _sim_config(
    tmp_path: Path,
    *,
    autoscalers: tuple[str, ...] = ("static-a", "static-b"),
    fail_fast: bool = False,
    destinations: tuple[PublishDestination, ...] | None = None,
) -> MatchConfig:
    engine = EngineConfig(
        system="h200_sxm",
        backend="vllm",
        backend_version="0.19.0",
        ais_backend="vllm",
        ais_backend_version="0.19.0",
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        num_gpus=1,
        extra_args={},
    )
    backend = SimBackendConfig(
        type="sim",
        substrate="gpt_oss",
        model=ModelConfig(
            name="openai/gpt-oss-120b",
            ais_model_path="openai/gpt-oss-120b",
        ),
        engines=SimEnginesConfig(prefill=engine, decode=engine),
        topology="disagg",
        gpu_budget=8,
        router=RouterConfig(),
        planner_config={},
        replay=ReplayConfig(),
        autoscalers=tuple(
            SimAutoscalerConfig(
                name=name,
                type="static",
                start=ReplicaCounts(prefill=1, decode=1),
                config={"num_prefill": 1, "num_decode": 1},
            )
            for name in autoscalers
        ),
    )
    return MatchConfig(
        schema_version=1,
        name="runner-unit",
        description="runner contract test",
        labels={"suite": "unit"},
        backend=backend,
        evaluations=(
            EvaluationConfig(
                workload="flat",
                seed=17,
                max_requests=5,
                arrival_speedup=2.0,
            ),
        ),
        sla_profiles=(
            SLAProfileConfig(
                name="interactive",
                source_name="interactive",
                ttft_ms=300.0,
                itl_ms=50.0,
            ),
        ),
        metrics=MetricsConfig(
            rank_by="goodput_rps",
            include=("goodput_rps", "p99_ttft_ms"),
        ),
        execution=ExecutionConfig(
            repetitions=1,
            fail_fast=fail_fast,
            max_runs=100,
        ),
        publish=PublishConfig(
            artifact_root=tmp_path / "artifacts",
            destinations=destinations
            if destinations is not None
            else (PublishDestination(type="console"),),
        ),
        source_path=tmp_path / "match.yaml",
    )


def _identity(item: MatchRun) -> dict[str, Any]:
    return {field: getattr(item, field) for field in IDENTITY_FIELDS}


def _ok_result(item: MatchRun, **metrics: Any) -> dict[str, Any]:
    return {
        **_identity(item),
        "status": "ok",
        "metrics": metrics or {"goodput_rps": float(item.index)},
    }


def _json_config(
    tmp_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> MatchConfig:
    return _sim_config(
        tmp_path,
        destinations=(
            PublishDestination(
                type="json",
                path=output_path,
                overwrite=overwrite,
            ),
        ),
    )


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "client_secret",
        "access_token",
        "secret_key",
        "aws_secret_access_key",
    ],
)
def test_sim_engine_secret_is_redacted_from_failure_messages(
    tmp_path: Path, secret_key: str
):
    config = _sim_config(tmp_path)
    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    assert backend.engines.prefill is not None
    secret_engine = replace(
        backend.engines.prefill,
        extra_args={secret_key: "sim-engine-secret"},
    )
    config = replace(
        config,
        backend=replace(
            backend,
            engines=replace(backend.engines, prefill=secret_engine),
        ),
    )

    message = match_runner._redact_exception_message(
        config, "engine rejected sim-engine-secret"
    )

    assert message == "engine rejected <redacted>"


def _write_cli_sim_config(
    tmp_path: Path,
    *,
    name: str,
    destinations: str,
) -> Path:
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            schema_version: 1
            name: {name}
            backend:
              type: sim
              substrate: gpt_oss
              topology: disagg
              autoscalers:
                - name: fixed
                  type: static
                  config:
                    num_prefill: 1
                    num_decode: 1
            evaluations:
              workloads: [flat]
            slo_profiles:
              - name: interactive
                ttft_ms: 300
                itl_ms: 50
            metrics:
              rank_by: goodput_rps
              include: [goodput_rps]
            execution:
              repetitions: 1
            publish:
              artifact_root: artifacts
              destinations:
            """
        )
        + textwrap.indent(
            textwrap.dedent(destinations).strip(),
            "    ",
        )
        + "\n"
    )
    return config_path


def _write_cli_dynamo_trace_config(tmp_path: Path) -> tuple[Path, Path]:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"timestamp":0,"input_length":8,"output_length":4}\n')
    config_path = tmp_path / "exact-trace.yaml"
    config_path.write_text(
        textwrap.dedent(
            """\
            schema_version: 1
            name: exact-trace
            backend:
              type: sim
              substrate: gpt_oss
              topology: disagg
              autoscalers:
                - name: fixed
                  type: static
                  config: {num_prefill: 1, num_decode: 1}
            evaluations:
              traces:
                - name: exact
                  format: dynamo
                  paths: [trace.jsonl]
            slo_profiles:
              - name: interactive
                ttft_ms: 300
                itl_ms: 50
            metrics:
              rank_by: goodput_rps
              include: [goodput_rps]
            execution:
              repetitions: 1
            publish:
              artifact_root: artifacts
              destinations:
                - type: console
            """
        )
    )
    return config_path, trace_path


def _run_guarded_cli(
    tmp_path: Path,
    config_path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    guard = textwrap.dedent(
        """\
        import builtins
        import runpy
        import sys

        script, *arguments = sys.argv[1:]
        original_import = builtins.__import__
        forbidden = (
            "dynamo",
            "aiperf",
            "autoscaling_arena.runners.sims",
            "autoscaling_arena.runners.real",
        )

        def guarded_import(name, *args, **kwargs):
            if any(name == item or name.startswith(item + ".") for item in forbidden):
                raise AssertionError(f"CLI imported runtime module: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import
        sys.argv = [script, *arguments]
        runpy.run_path(script, run_name="__main__")
        """
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            guard,
            str(RUN_MATCH_CONFIG),
            str(config_path),
            *arguments,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


@pytest.mark.timeout(30)
def test_validate_only_does_not_import_runtime_backends_or_create_artifacts(
    tmp_path: Path,
):
    config_path = _write_cli_sim_config(
        tmp_path,
        name="validate-only-unit",
        destinations="""
        - type: console
        """,
    )
    proc = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--print-matrix",
    )

    assert proc.returncode == 0, proc.stderr
    assert "Valid Match Config: 'validate-only-unit'" in proc.stdout
    assert "0001-sim-fixed-flat-interactive-r0-s0" in proc.stdout
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.timeout(30)
def test_cli_rejects_unknown_exact_replay_run_before_runtime_import(
    tmp_path: Path,
):
    config_path = _write_cli_sim_config(
        tmp_path,
        name="exact-replay-unit",
        destinations="""
        - type: console
        """,
    )

    proc = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--run-id",
        "missing-run",
    )

    assert proc.returncode == 2
    assert "unknown --run-id: missing-run" in proc.stderr
    assert "CLI imported runtime module" not in proc.stderr
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.timeout(30)
def test_cli_validates_expected_config_digest_before_runtime_import(
    tmp_path: Path,
):
    config_path = _write_cli_sim_config(
        tmp_path,
        name="digest-unit",
        destinations="""
        - type: console
        """,
    )
    digest = match_runner.match_replay_sha256(load_match_config(config_path))

    accepted = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--expect-config-sha256",
        digest,
    )
    rejected = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--expect-config-sha256",
        "0" * 64,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert rejected.returncode == 2
    assert "Match Config digest mismatch" in rejected.stderr
    assert "CLI imported runtime module" not in rejected.stderr
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.timeout(30)
def test_cli_validates_selected_source_trace_digest_before_runtime_import(
    tmp_path: Path,
):
    config_path, trace_path = _write_cli_dynamo_trace_config(tmp_path)
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    run_id = "0001-sim-fixed-exact-interactive-r0-s0"

    accepted = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--run-id",
        run_id,
        "--expect-trace-sha256",
        digest,
    )
    trace_path.write_text("changed\n")
    rejected = _run_guarded_cli(
        tmp_path,
        config_path,
        "--validate-only",
        "--run-id",
        run_id,
        "--expect-trace-sha256",
        digest,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert rejected.returncode == 2
    assert "source trace digest mismatch" in rejected.stderr
    assert "CLI imported runtime module" not in rejected.stderr
    assert not (tmp_path / "artifacts").exists()


def test_replay_config_digest_is_portable_across_workspace_roots(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_config, _ = _write_cli_dynamo_trace_config(left)
    right_config, _ = _write_cli_dynamo_trace_config(right)
    left_match = load_match_config(left_config)
    right_match = load_match_config(right_config)

    assert match_runner.match_config_sha256(
        left_match
    ) != match_runner.match_config_sha256(right_match)
    assert match_runner.match_replay_sha256(
        left_match
    ) == match_runner.match_replay_sha256(right_match)


@pytest.mark.timeout(30)
def test_cli_preflights_json_collision_before_importing_sim_runtime(
    tmp_path: Path,
):
    output_path = tmp_path / "results.json"
    output_path.write_text("previous result\n")
    config_path = _write_cli_sim_config(
        tmp_path,
        name="preflight-unit",
        destinations="""
        - type: json
          path: results.json
        """,
    )

    proc = _run_guarded_cli(tmp_path, config_path)

    assert proc.returncode == 2
    assert "Publish error: JSON destination already exists" in proc.stderr
    assert "CLI imported runtime module" not in proc.stderr
    assert output_path.read_text() == "previous result\n"
    assert not (tmp_path / "artifacts").exists()


def test_json_publisher_refuses_existing_destination_without_overwrite(
    tmp_path: Path,
):
    output_path = tmp_path / "published" / "results.json"
    output_path.parent.mkdir()
    output_path.write_text("original\n")
    config = _json_config(tmp_path, output_path)

    with pytest.raises(
        match_runner.MatchPublishError,
        match="JSON destination already exists",
    ):
        match_runner.publish_match_results(config, {"new": "report"})

    assert output_path.read_text() == "original\n"
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_json_publisher_late_collision_preserves_winner_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_path = tmp_path / "published" / "results.json"
    config = _json_config(tmp_path, output_path)

    def concurrent_winner(source: str | Path, destination: str | Path) -> None:
        del source
        Path(destination).write_text("concurrent result\n")
        raise FileExistsError("destination won by another publisher")

    monkeypatch.setattr(match_runner.os, "link", concurrent_winner)

    with pytest.raises(
        match_runner.MatchPublishError,
        match="created concurrently",
    ):
        match_runner.publish_match_results(config, {"new": "report"})

    assert output_path.read_text() == "concurrent result\n"
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_json_publisher_rolls_back_first_destination_when_second_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "published"
    first_path = output_dir / "first.json"
    second_path = output_dir / "second.json"
    config = _sim_config(
        tmp_path,
        destinations=(
            PublishDestination(type="json", path=first_path),
            PublishDestination(type="json", path=second_path),
        ),
    )
    original_link = match_runner.os.link

    def fail_second_commit(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        if Path(destination) == second_path:
            raise OSError("second commit failed")
        original_link(source, destination)

    monkeypatch.setattr(match_runner.os, "link", fail_second_commit)

    with pytest.raises(
        match_runner.MatchPublishError,
        match="second commit failed",
    ):
        match_runner.publish_match_results(config, {"new": "report"})

    assert not first_path.exists()
    assert not second_path.exists()
    assert list(output_dir.glob(".*.tmp")) == []
    assert list(output_dir.glob(".*.backup")) == []


@pytest.mark.parametrize(
    ("configured_overwrite", "force_overwrite"),
    [(True, False), (False, True)],
)
def test_json_publisher_overwrites_via_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_overwrite: bool,
    force_overwrite: bool,
):
    output_path = tmp_path / "published" / "results.json"
    output_path.parent.mkdir()
    output_path.write_text("original\n")
    config = _json_config(
        tmp_path,
        output_path,
        overwrite=configured_overwrite,
    )
    report = {"summary": {"status": "ok"}, "rows": [1, 2]}
    original_replace = match_runner.os.replace
    replacements: list[tuple[Path, Path]] = []

    def observed_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == output_path
        assert destination_path.read_text() == "original\n"
        assert source_path.parent == output_path.parent
        assert json.loads(source_path.read_text()) == report
        replacements.append((source_path, destination_path))
        original_replace(source, destination)

    monkeypatch.setattr(match_runner.os, "replace", observed_replace)

    written = match_runner.publish_match_results(
        config,
        report,
        force_overwrite=force_overwrite,
    )

    assert written == [output_path]
    assert json.loads(output_path.read_text()) == report
    assert len(replacements) == 1
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


def test_json_publisher_replace_failure_preserves_old_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_path = tmp_path / "published" / "results.json"
    output_path.parent.mkdir()
    output_path.write_text("original\n")
    config = _json_config(tmp_path, output_path, overwrite=True)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(match_runner.os, "replace", fail_replace)

    with pytest.raises(
        (OSError, match_runner.MatchPublishError),
        match="replace failed",
    ):
        match_runner.publish_match_results(config, {"new": "report"})

    assert output_path.read_text() == "original\n"
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    (
        "fail_fast",
        "expected_calls",
        "expected_succeeded",
        "expected_skipped",
    ),
    [
        (False, 3, 2, 0),
        (True, 1, 0, 2),
    ],
)
def test_execute_match_continues_or_stops_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_fast: bool,
    expected_calls: int,
    expected_succeeded: int,
    expected_skipped: int,
):
    config = _sim_config(
        tmp_path,
        autoscalers=("broken", "healthy-a", "healthy-b"),
        fail_fast=fail_fast,
    )
    calls: list[str] = []
    progress: list[tuple[str, str]] = []

    def fake_run(
        config: MatchConfig,
        item: MatchRun,
        context: match_runner._ExecutionContext,
    ) -> dict[str, Any]:
        del config, context
        calls.append(item.autoscaler)
        if item.autoscaler == "broken":
            raise RuntimeError("synthetic runner failure")
        return _ok_result(item, goodput_rps=float(item.index))

    monkeypatch.setattr(match_runner, "_run_sim_item", fake_run)
    monkeypatch.setattr(match_runner, "_git_commit", lambda: None)

    report = match_runner.execute_match_config(
        config,
        on_progress=lambda phase, item, result: progress.append((phase, item.run_id)),
    )

    assert calls == ["broken", "healthy-a", "healthy-b"][:expected_calls]
    assert report["summary"] == {
        "status": "failed",
        "planned_runs": 3,
        "executed_runs": expected_calls,
        "succeeded_runs": expected_succeeded,
        "failed_runs": 1,
        "skipped_runs": expected_skipped,
        "rank_by": "goodput_rps",
        "metrics": ["goodput_rps", "p99_ttft_ms"],
    }
    assert report["results"][0]["error"] == {
        "type": "RuntimeError",
        "message": "synthetic runner failure",
    }
    assert [phase for phase, _ in progress] == [
        phase for _ in range(expected_calls) for phase in ("started", "finished")
    ]


def test_execute_match_normalizes_json_and_preserves_matrix_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _sim_config(tmp_path)

    def fake_run(
        config: MatchConfig,
        item: MatchRun,
        context: match_runner._ExecutionContext,
    ) -> dict[str, Any]:
        del config
        return {
            **_ok_result(
                item,
                goodput_rps=float("nan") if item.index == 1 else 2.0,
                p99_ttft_ms=float("inf"),
            ),
            "runtime": {
                "session_root": context.session_root,
                "tags": ("mocked", item.autoscaler),
            },
        }

    monkeypatch.setattr(match_runner, "_run_sim_item", fake_run)
    monkeypatch.setattr(match_runner, "_git_commit", lambda: "deadbeef")

    report = match_runner.execute_match_config(config)

    matrix_identities = [
        {field: row[field] for field in IDENTITY_FIELDS} for row in report["matrix"]
    ]
    result_identities = [
        {field: row[field] for field in IDENTITY_FIELDS} for row in report["results"]
    ]
    assert result_identities == matrix_identities
    assert [row["run_id"] for row in report["matrix"]] == [
        item.run_id for item in config.iter_runs()
    ]
    assert report["results"][0]["metrics"]["goodput_rps"] is None
    assert report["results"][0]["metrics"]["p99_ttft_ms"] is None
    assert isinstance(report["results"][0]["runtime"]["session_root"], str)
    assert report["results"][0]["runtime"]["tags"] == ["mocked", "static-a"]
    assert report["provenance"]["git_commit"] == "deadbeef"
    json.dumps(report, allow_nan=False)


def test_execute_match_can_replay_one_exact_matrix_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _sim_config(tmp_path)
    selected = list(config.iter_runs())[1]
    calls: list[str] = []

    def fake_run(
        config: MatchConfig,
        item: MatchRun,
        context: match_runner._ExecutionContext,
    ) -> dict[str, Any]:
        del config, context
        calls.append(item.run_id)
        return _ok_result(item, goodput_rps=2.0, p99_ttft_ms=10.0)

    monkeypatch.setattr(match_runner, "_run_sim_item", fake_run)
    monkeypatch.setattr(match_runner, "_git_commit", lambda: None)

    report = match_runner.execute_match_config(config, run_ids={selected.run_id})

    assert calls == [selected.run_id]
    assert [row["run_id"] for row in report["matrix"]] == [selected.run_id]
    assert [row["run_id"] for row in report["results"]] == [selected.run_id]
    assert report["summary"]["planned_runs"] == 1
    assert report["summary"]["executed_runs"] == 1
    assert report["summary"]["skipped_runs"] == 0
    assert report["provenance"]["config_file"] == "match.yaml"
    assert report["provenance"]["replay"] == {
        "kind": "match_config",
        "config_path": None,
    }

    with pytest.raises(ValueError, match="unknown Match Config run id"):
        match_runner.execute_match_config(config, run_ids={"missing-run"})


def test_existing_external_match_config_uses_portable_replay_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _sim_config(tmp_path, autoscalers=("static",))
    config.source_path.write_text("schema_version: 1\n")
    monkeypatch.setattr(
        match_runner,
        "_run_sim_item",
        lambda config, item, context: _ok_result(item),
    )

    report = match_runner.execute_match_config(config)

    assert report["provenance"]["replay"] == {
        "kind": "match_config",
        "config_path": "$MATCH_CONFIG",
    }


def test_failed_native_trace_run_redacts_every_shard_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "traces" / "recorded-00.jsonl.gz"
    second = tmp_path / "traces" / "recorded-01.jsonl.gz"
    base = _sim_config(tmp_path, autoscalers=("static",))
    config = replace(
        base,
        source_path=tmp_path / "configs" / "match.yaml",
        evaluations=(
            replace(
                base.evaluations[0],
                workload="recorded-native",
                max_requests=None,
                arrival_speedup=1.0,
                trace_path=None,
                trace_paths=(first, second),
                trace_format="dynamo",
                trace_block_size=None,
            ),
        ),
    )

    def fail_with_paths(config, item, context):
        del config, item, context
        raise RuntimeError(f"could not open {first} or {second}")

    monkeypatch.setattr(match_runner, "_run_sim_item", fail_with_paths)

    report = match_runner.execute_match_config(config)
    serialized = json.dumps(report)

    assert str(first) not in serialized
    assert str(second) not in serialized
    assert report["results"][0]["error"]["message"] == (
        "could not open <redacted> or <redacted>"
    )


@pytest.mark.parametrize(
    ("rank_by", "values"),
    [
        ("goodput_rps", {"best": 10.0, "second": 5.0}),
        ("p99_ttft_ms", {"best": 10.0, "second": 50.0}),
        ("duration_s", {"best": 10.0, "second": 20.0}),
    ],
)
def test_console_ranking_orders_best_first_and_missing_or_failed_last(
    rank_by: str,
    values: dict[str, float],
):
    def result(
        autoscaler: str,
        *,
        status: str = "ok",
        value: float | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "autoscaler": autoscaler,
            "workload": "flat",
            "sla": "interactive",
            "repetition": 0,
            "status": status,
            "metrics": {rank_by: value},
        }
        if status != "ok":
            row["error"] = {"type": "MockFailure", "message": "boom"}
        return row

    report = {
        "match": {"name": "ranking-unit", "backend": "sim"},
        "summary": {
            "succeeded_runs": 3,
            "failed_runs": 1,
            "skipped_runs": 0,
            "planned_runs": 4,
            "rank_by": rank_by,
            "metrics": [rank_by],
        },
        "results": [
            result("second", value=values["second"]),
            result("missing", value=None),
            result("crashed", status="failed"),
            result("best", value=values["best"]),
        ],
    }

    output = match_runner.format_match_results(report)

    positions = [
        output.index(name) for name in ("best", "second", "missing", "crashed")
    ]
    assert positions == sorted(positions)
    assert "flat | interactive | repetition 0" in output
    assert "crashed              FAILED MockFailure: boom" in output


def test_aiperf_header_credentials_are_redacted_from_config_and_errors():
    args = (
        "--header",
        "Authorization:Bearer top-secret",
        "X-Trace-ID:also-private",
        "--request-rate",
        "4",
        "-H=X-API-Key:joined-secret",
    )

    assert match_runner._redacted_extra_args(args) == [
        "--header",
        "<redacted>",
        "<redacted>",
        "--request-rate",
        "4",
        "-H=<redacted>",
    ]
    message = f"failed command: {' '.join(args)}"
    redacted = match_runner._redact_aiperf_text(message, args)
    assert "top-secret" not in redacted
    assert "also-private" not in redacted
    assert "joined-secret" not in redacted
    assert redacted.count("<redacted>") == 3


def test_common_aiperf_credential_flags_are_redacted():
    args = (
        "--client-secret=oauth-secret",
        "--access-token",
        "access-secret",
        "--request-rate",
        "4",
    )

    assert match_runner._redacted_extra_args(args) == [
        "--client-secret=<redacted>",
        "--access-token",
        "<redacted>",
        "--request-rate",
        "4",
    ]
    redacted = match_runner._redact_aiperf_text("oauth-secret access-secret", args)
    assert redacted == "<redacted> <redacted>"


def test_user_supplied_trace_is_materialized_with_declared_metadata(tmp_path: Path):
    source = tmp_path / "private" / "traffic.jsonl"
    source.parent.mkdir()
    source.write_text(
        '{"timestamp": 200, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
        '{"timestamp": 400, "input_length": 8, "output_length": 2, "hash_ids": [2]}\n'
    )
    item = MatchRun(
        index=1,
        run_id="0001-sim-static-private-traffic-interactive-r0-s0",
        backend="sim",
        autoscaler="static",
        workload="private-traffic",
        sla="interactive",
        slo_profile=SLAProfileConfig(
            name="interactive",
            source_name="interactive",
            ttft_ms=300.0,
        ),
        repetition=0,
        seed=0,
        max_requests=1,
        arrival_speedup=2.0,
        trace_path=source,
        trace_block_size=64,
        trace_presorted=True,
    )

    workload = match_runner._workload_for_item(item)
    assert workload.static_trace == source
    assert workload.block_size == 64
    assert workload.presorted is True

    context = match_runner._ExecutionContext(
        session_id="external-trace",
        session_root=tmp_path / "session",
    )
    materialized = match_runner._materialize_trace(
        context,
        workload_name=item.workload,
        seed=item.seed,
        max_requests=item.max_requests,
        arrival_speedup=item.arrival_speedup,
        speedup_is_materialized=True,
        **match_runner._external_trace_options(item),
    )

    assert materialized != source
    assert json.loads(materialized.read_text()) == {
        "timestamp": 100,
        "input_length": 8,
        "output_length": 2,
        "hash_ids": [1],
    }


def test_external_trace_copy_stays_outside_artifact_tree_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import autoscaling_arena.html_report as html_report

    source = tmp_path / "private-inputs" / "sensitive-source-name.jsonl"
    source.parent.mkdir()
    source.write_text(
        '{"timestamp": 200, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
        '{"timestamp": 400, "input_length": 8, "output_length": 2, "hash_ids": [2]}\n'
    )
    base = _sim_config(tmp_path, autoscalers=("static-a", "static-b"))
    config = replace(
        base,
        evaluations=(
            EvaluationConfig(
                workload="recorded-traffic",
                seed=0,
                max_requests=None,
                arrival_speedup=1.0,
                trace_path=source,
                trace_block_size=64,
                trace_presorted=True,
            ),
        ),
    )
    observed: dict[str, Path] = {}
    arrival_calls = 0
    original_arrival_series = html_report.trace_arrival_series

    def counted_arrival_series(path: Path, *, speedup: float = 1.0):
        nonlocal arrival_calls
        arrival_calls += 1
        return original_arrival_series(path, speedup=speedup)

    monkeypatch.setattr(html_report, "trace_arrival_series", counted_arrival_series)

    def fake_run(
        value: MatchConfig, item: MatchRun, context: match_runner._ExecutionContext
    ) -> dict[str, Any]:
        del value
        trace = match_runner._materialize_trace(
            context,
            workload_name=item.workload,
            seed=item.seed,
            max_requests=item.max_requests,
            arrival_speedup=item.arrival_speedup,
            speedup_is_materialized=False,
            **match_runner._external_trace_options(item),
        )
        assert context.external_trace_root is not None
        assert trace.is_relative_to(context.external_trace_root)
        assert not trace.is_relative_to(context.session_root)
        assert trace.name == "recorded-trace.jsonl"
        observed["external_root"] = context.external_trace_root
        return {
            **_identity(item),
            "status": "ok",
            "metrics": {"goodput_rps": 1.0, "p99_ttft_ms": 1.0},
            "evaluation": match_runner._evaluation_metadata(
                item, block_size=64, trace_path=trace, context=context
            ),
        }

    monkeypatch.setattr(match_runner, "_run_sim_item", fake_run)

    report = match_runner.execute_match_config(config)

    assert not observed["external_root"].exists()
    assert arrival_calls == 1
    serialized = json.dumps(report)
    assert str(source) not in serialized
    assert str(tmp_path) not in serialized
    trace_metadata = report["results"][0]["evaluation"]["trace"]
    assert "path" not in trace_metadata
    assert trace_metadata["source_sha256"]
    assert trace_metadata["sha256"]
    assert trace_metadata["staged"] is True
    assert trace_metadata["transformed"] is False
    assert report["resolved_config"]["evaluations"][0]["trace_path"] == "<redacted>"
    assert report["results"][0]["evaluation"]["arrival_series"]["status"] == "ok"
    assert report["provenance"]["config_path"] == "<config>"
    assert report["provenance"]["artifact_root"] == "<artifact-session>"


def test_publish_preflight_protects_user_supplied_trace(tmp_path: Path):
    source = tmp_path / "traffic.jsonl"
    source.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
    )
    base = _sim_config(tmp_path, autoscalers=("static",))
    config = replace(
        base,
        evaluations=(
            replace(
                base.evaluations[0],
                workload="recorded-traffic",
                trace_path=source,
                trace_block_size=64,
            ),
        ),
        publish=replace(
            base.publish,
            destinations=(PublishDestination(type="json", path=source),),
        ),
    )

    with pytest.raises(match_runner.MatchPublishError, match="refusing to overwrite"):
        match_runner.preflight_publish_destinations(config, force_overwrite=True)


def test_publish_preflight_protects_every_native_trace_shard(tmp_path: Path):
    first = tmp_path / "recorded-00.jsonl.gz"
    second = tmp_path / "recorded-01.jsonl.gz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    base = _sim_config(tmp_path, autoscalers=("static",))
    config = replace(
        base,
        evaluations=(
            replace(
                base.evaluations[0],
                workload="recorded-native",
                max_requests=None,
                arrival_speedup=1.0,
                trace_path=None,
                trace_paths=(first, second),
                trace_format="dynamo",
                trace_block_size=None,
            ),
        ),
        publish=replace(
            base.publish,
            destinations=(PublishDestination(type="json", path=second),),
        ),
    )

    with pytest.raises(match_runner.MatchPublishError, match="refusing to overwrite"):
        match_runner.preflight_publish_destinations(config, force_overwrite=True)


def test_publish_preflight_resolves_trace_path_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "traffic.jsonl"
    source.write_text(
        '{"timestamp": 0, "input_length": 8, "output_length": 2, "hash_ids": [1]}\n'
    )
    monkeypatch.chdir(tmp_path)
    base = _sim_config(tmp_path, autoscalers=("static",))
    config = replace(
        base,
        evaluations=(
            replace(
                base.evaluations[0],
                workload="recorded-traffic",
                trace_path=Path("traffic.jsonl"),
                trace_block_size=64,
            ),
        ),
        publish=replace(
            base.publish,
            destinations=(PublishDestination(type="json", path=source.resolve()),),
        ),
    )

    with pytest.raises(match_runner.MatchPublishError, match="refusing to overwrite"):
        match_runner.preflight_publish_destinations(config, force_overwrite=True)


def test_publisher_writes_json_and_standalone_html_in_one_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import autoscaling_arena.html_report as html_report

    json_path = tmp_path / "published" / "results.json"
    html_path = tmp_path / "published" / "report.html"
    config = _sim_config(
        tmp_path,
        destinations=(
            PublishDestination(type="json", path=json_path),
            PublishDestination(type="html", path=html_path),
        ),
    )
    report = {
        "match": {"name": "publisher-unit", "backend": "sim"},
        "summary": {"status": "ok"},
        "results": [],
    }
    expected_html = "<!doctype html><title>Arena report</title>"
    render_calls: list[dict[str, Any]] = []

    def fake_render(value: dict[str, Any]) -> str:
        render_calls.append(value)
        return expected_html

    monkeypatch.setattr(html_report, "render_match_report", fake_render)

    written = match_runner.publish_match_results(config, report)

    assert written == [json_path, html_path]
    assert json.loads(json_path.read_text()) == report
    assert html_path.read_text() == expected_html
    assert render_calls == [report]
    assert list(json_path.parent.glob(".*.tmp")) == []
    assert list(json_path.parent.glob(".*.backup")) == []


def test_publisher_refuses_existing_html_without_overwrite(tmp_path: Path):
    html_path = tmp_path / "published" / "report.html"
    html_path.parent.mkdir()
    html_path.write_text("original report\n")
    config = _sim_config(
        tmp_path,
        destinations=(PublishDestination(type="html", path=html_path),),
    )

    with pytest.raises(
        match_runner.MatchPublishError,
        match="HTML destination already exists",
    ):
        match_runner.publish_match_results(config, {"results": []})

    assert html_path.read_text() == "original report\n"
    assert list(html_path.parent.glob(f".{html_path.name}.*.tmp")) == []


@pytest.mark.timeout(30)
def test_generated_replay_command_validates_from_documented_gym_directory():
    config = load_match_config(ARENA_ROOT / "configs/match.quickstart.yaml")
    run = next(config.iter_runs())
    config_reference = match_runner._safe_replay_config_path(config)
    assert config_reference == "configs/match.quickstart.yaml"
    report = {
        "provenance": {
            "replay_config_sha256": match_runner.match_replay_sha256(config),
            "replay": {"kind": "match_config", "config_path": config_reference},
        }
    }
    command = _replay_command(report, {"run_id": run.run_id}, replay_config_path=None)
    assert command is not None
    arguments = shlex.split(command)
    arguments[0] = sys.executable
    result = subprocess.run(
        [*arguments, "--validate-only"],
        cwd=ARENA_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid Match Config" in result.stdout
