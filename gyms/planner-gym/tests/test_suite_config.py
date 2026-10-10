# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for independent-case manifests without a simulation runtime."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from autoscaling_arena import match_runner, suite_runner
from autoscaling_arena.suite_config import (
    SuiteConfigError,
    load_suite_config,
    parse_suite_config,
    suite_config_sha256,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def suite_files(tmp_path: Path) -> tuple[Path, dict]:
    config = {
        "schema_version": 1,
        "name": "synthetic-case",
        "backend": {
            "type": "sim",
            "topology": "disagg",
            "gpu_budget": 8,
            "model": {"name": "synthetic-model"},
            "engines": {"common": {"system": "synthetic-system", "backend": "vllm"}},
            "autoscalers": [
                {
                    "name": "fixed",
                    "type": "static",
                    "config": {"num_prefill": 1, "num_decode": 1},
                }
            ],
        },
        "evaluations": {"workloads": ["flat"]},
        "slo_profiles": [{"name": "interactive", "ttft_ms": 200, "itl_ms": 20}],
        "publish": {
            "artifact_root": "artifacts",
            "destinations": [{"type": "json", "path": "never-publish.json"}],
        },
    }
    child = tmp_path / "configs"
    child.mkdir()
    (child / "first.json").write_text(json.dumps(config))
    second = copy.deepcopy(config)
    second["backend"]["topology"] = "agg"
    second["backend"]["autoscalers"][0]["config"].pop("num_prefill")
    second["backend"]["engines"]["common"]["tp_size"] = 2
    second["slo_profiles"][0]["ttft_ms"] = 400
    (child / "second.json").write_text(json.dumps(second))
    manifest = {
        "schema_version": 1,
        "name": "Independent cases",
        "cases": [
            {
                "id": "first",
                "config": "configs/first.json",
                "labels": {"size": "small"},
            },
            {"id": "second", "config": "configs/second.json"},
        ],
    }
    source = tmp_path / "suite.json"
    source.write_text(json.dumps(manifest))
    return source, manifest


def test_independent_configs_keep_topology_slo_and_relative_paths(suite_files):
    source, _ = suite_files
    suite = load_suite_config(source)

    first, second = suite.cases
    assert first.config.backend.topology == "disagg"
    assert second.config.backend.topology == "agg"
    assert first.config.sla_profiles[0].ttft_ms == 200
    assert second.config.sla_profiles[0].ttft_ms == 400
    assert first.config.source_path == source.parent / "configs/first.json"
    assert first.config.publish.artifact_root == source.parent / "configs/artifacts"
    assert (
        first.config_file_sha256
        == hashlib.sha256(first.config.source_path.read_bytes()).hexdigest()
    )
    assert suite.selected_cases({"second"}) == (second,)
    assert sum(case.expected_runs for case in suite.cases) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"unknown": 1}, "unknown key"),
        ({"cases": []}, "non-empty list"),
        ({"cases": [{"id": "../escape", "config": "configs/first.json"}]}, ".id"),
        ({"cases": [{"id": "x", "config": "missing.yaml"}]}, ".config"),
        (
            {"cases": [{"id": "x", "config": "configs/first.json", "run_ids": []}]},
            "non-empty list",
        ),
        (
            {
                "cases": [
                    {"id": "x", "config": "configs/first.json", "run_ids": ["bad"]}
                ]
            },
            "unknown run id",
        ),
        (
            {
                "cases": [
                    {"id": "x", "config": "configs/first.json", "labels": {"a": 1}}
                ]
            },
            "string keys and values",
        ),
    ],
)
def test_invalid_suite_fails_before_execution(suite_files, change, message):
    source, manifest = suite_files
    manifest.update(change)
    with pytest.raises(SuiteConfigError, match=message):
        parse_suite_config(manifest, source_path=source)


def test_unique_case_ids_and_exact_run_selection(suite_files):
    source, manifest = suite_files
    first_run = next(load_suite_config(source).cases[0].config.iter_runs()).run_id
    manifest["cases"][0]["run_ids"] = [first_run]
    parsed = parse_suite_config(manifest, source_path=source)
    assert parsed.cases[0].run_ids == (first_run,)
    manifest["cases"][1]["id"] = "first"
    with pytest.raises(SuiteConfigError, match="duplicate case"):
        parse_suite_config(manifest, source_path=source)
    with pytest.raises(SuiteConfigError, match="unknown case"):
        parsed.selected_cases({"missing"})
    with pytest.raises(SuiteConfigError, match="at least one"):
        parsed.selected_cases(set())


def test_duplicate_yaml_keys_are_rejected(suite_files):
    source, _ = suite_files
    source.write_text("schema_version: 1\nname: a\nname: b\ncases: []\n")
    with pytest.raises(SuiteConfigError, match="duplicate key"):
        load_suite_config(source)


def test_digest_changes_when_referenced_config_changes(suite_files):
    source, _ = suite_files
    before = suite_config_sha256(load_suite_config(source))
    path = source.parent / "configs/second.json"
    config = json.loads(path.read_text())
    config["slo_profiles"][0]["ttft_ms"] = 900
    path.write_text(json.dumps(config))
    assert suite_config_sha256(load_suite_config(source)) != before


@pytest.mark.timeout(15)
def test_cli_dry_run_never_imports_native_runtime(suite_files):
    source, _ = suite_files
    script = Path(__file__).resolve().parents[1] / "scripts/run_suite_config.py"
    guard = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'dynamo' or name.startswith(('dynamo.', 'aisimulate')):
        raise AssertionError('unexpected native import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [sys.argv[1], sys.argv[2], '--dry-run', '--case-id', 'second']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", guard, str(script), str(source)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "second:" in result.stdout
    assert "first:" not in result.stdout
    assert not (source.parent / "configs/artifacts").exists()


def _fake_match_executor(calls):
    def execute(config, *, run_ids, resume_dir, session_dir):
        calls.append((config, run_ids, resume_dir, session_dir))
        if session_dir is not None:
            session_dir.mkdir()
        execution = resume_dir or session_dir
        runs = [
            run
            for run in config.iter_runs()
            if run_ids is None or run.run_id in run_ids
        ]
        return {
            "summary": {
                "status": "ok",
                "planned_runs": len(runs),
                "succeeded_runs": len(runs),
                "failed_runs": 0,
            },
            "provenance": {"session_id": execution.name},
            "results": [
                {
                    "run_id": run.run_id,
                    "status": "ok",
                    "backend": run.backend,
                    "autoscaler": run.autoscaler,
                    "workload": run.workload,
                    "sla": run.sla,
                    "repetition": run.repetition,
                    "seed": run.seed,
                    "metrics": {"goodput_rps": config.sla_profiles[0].ttft_ms},
                }
                for run in runs
            ],
        }

    return execute


def test_suite_isolates_artifacts_and_resumes_through_match_validator(
    suite_files, monkeypatch
):
    source, _ = suite_files
    config = load_suite_config(source)
    calls = []
    monkeypatch.setattr(
        suite_runner, "execute_match_config", _fake_match_executor(calls)
    )
    monkeypatch.setattr(
        suite_runner, "render_match_report", lambda report: "<html>report</html>"
    )
    output = source.parent / "results"
    report = suite_runner.execute_suite_config(config, output_dir=output)

    assert report["summary"]["succeeded_cases"] == 2
    assert len(calls) == 2
    assert calls[0][3] == output / "cases/first/execution"
    assert calls[1][3] == output / "cases/second/execution"
    assert all(not item[0].publish.destinations for item in calls)
    assert not (source.parent / "configs/never-publish.json").exists()
    assert (output / "cases/first/results.json").is_file()
    assert (output / "cases/second/index.html").is_file()
    with (output / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in rows] == ["first", "second"]
    assert [float(row["metric.goodput_rps"]) for row in rows] == [200, 400]
    # Identical inner run IDs stay distinguishable by their suite case ID.
    assert rows[0]["run_id"] == rows[1]["run_id"]

    suite_runner.execute_suite_config(config, output_dir=output, resume=True)
    assert len(calls) == 4
    assert calls[2][2] == output / "cases/first/execution"
    assert calls[3][2] == output / "cases/second/execution"
    assert calls[2][3] is None
    with pytest.raises(FileExistsError):
        suite_runner.execute_suite_config(config, output_dir=output)
    with pytest.raises(SuiteConfigError, match="does not match"):
        suite_runner.execute_suite_config(
            config, output_dir=output, case_ids={"second"}, resume=True
        )


def test_suite_records_case_failure_and_continues(suite_files, monkeypatch):
    source, _ = suite_files
    config = load_suite_config(source)
    calls = []
    fake = _fake_match_executor(calls)

    def executor(match, **kwargs):
        if match.backend.topology == "disagg":
            raise ValueError("checkpoint input digest mismatch")
        return fake(match, **kwargs)

    monkeypatch.setattr(suite_runner, "execute_match_config", executor)
    monkeypatch.setattr(
        suite_runner, "render_match_report", lambda report: "<html>report</html>"
    )
    output = source.parent / "results"
    report = suite_runner.execute_suite_config(config, output_dir=output)

    assert report["summary"]["failed_cases"] == 1
    assert report["summary"]["succeeded_cases"] == 1
    assert (
        report["cases"]["first"]["error"]["message"]
        == "checkpoint input digest mismatch"
    )
    assert json.loads((output / "suite.json").read_text()) == report


@pytest.mark.parametrize("changed_field", ["slo", "model"])
def test_suite_resume_rejects_changed_referenced_config(
    suite_files, monkeypatch, changed_field
):
    source, _ = suite_files
    monkeypatch.setattr(suite_runner, "execute_match_config", _fake_match_executor([]))
    monkeypatch.setattr(
        suite_runner, "render_match_report", lambda report: "<html>report</html>"
    )
    output = source.parent / "results"
    suite_runner.execute_suite_config(load_suite_config(source), output_dir=output)
    path = source.parent / "configs/first.json"
    content = json.loads(path.read_text())
    if changed_field == "slo":
        content["slo_profiles"][0]["ttft_ms"] = 300
    else:
        content["backend"]["model"]["name"] = "another-synthetic-model"
    path.write_text(json.dumps(content))
    with pytest.raises(SuiteConfigError, match="does not match"):
        suite_runner.execute_suite_config(
            load_suite_config(source), output_dir=output, resume=True
        )


def test_suite_resume_rejects_changed_trace_bytes(suite_files, monkeypatch):
    source, _ = suite_files
    trace = source.parent / "configs/synthetic.jsonl"
    trace.write_text('{"timestamp":0,"input_length":8,"output_length":4}\n')
    path = source.parent / "configs/first.json"
    content = json.loads(path.read_text())
    content["evaluations"] = {
        "traces": [{"name": "external", "format": "dynamo", "paths": [trace.name]}]
    }
    path.write_text(json.dumps(content))
    monkeypatch.setattr(suite_runner, "execute_match_config", _fake_match_executor([]))
    monkeypatch.setattr(
        suite_runner, "render_match_report", lambda report: "<html>report</html>"
    )
    output = source.parent / "results"
    suite_runner.execute_suite_config(load_suite_config(source), output_dir=output)
    trace.write_text('{"timestamp":0,"input_length":16,"output_length":4}\n')
    with pytest.raises(SuiteConfigError, match="does not match"):
        suite_runner.execute_suite_config(
            load_suite_config(source), output_dir=output, resume=True
        )


def test_suite_selected_case_does_not_execute_other_cases(suite_files, monkeypatch):
    source, _ = suite_files
    calls = []
    monkeypatch.setattr(
        suite_runner, "execute_match_config", _fake_match_executor(calls)
    )
    monkeypatch.setattr(
        suite_runner, "render_match_report", lambda report: "<html>report</html>"
    )
    output = source.parent / "results"
    result = suite_runner.execute_suite_config(
        load_suite_config(source), output_dir=output, case_ids={"second"}
    )
    assert len(calls) == 1
    assert calls[0][0].backend.topology == "agg"
    assert result["summary"]["planned_cases"] == 1
    assert not (output / "cases/first").exists()


@pytest.mark.timeout(20)
def test_suite_uses_real_match_checkpoints_and_retries_only_failed_runs(
    suite_files, monkeypatch
):
    source, _ = suite_files
    config = load_suite_config(source)
    calls: dict[str, int] = {}

    def simulate(match, item, context):
        role = match.backend.topology
        calls[role] = calls.get(role, 0) + 1
        artifact = context.session_root / "runs" / item.run_id
        artifact.mkdir(parents=True, exist_ok=False)
        (artifact / "attempt.txt").write_text(str(calls[role]))
        if role == "agg" and calls[role] == 1:
            raise RuntimeError("synthetic first-attempt failure")
        return {
            **match_runner._result_identity(item),
            "status": "ok",
            "metrics": {"goodput_per_gpu": 2.0, "goodput_rps": 4.0},
            "artifacts": {"directory": str(artifact)},
        }

    monkeypatch.setattr(match_runner, "_run_sim_item", simulate)
    output = source.parent / "results"
    first = suite_runner.execute_suite_config(config, output_dir=output)
    assert first["summary"]["succeeded_cases"] == 1
    assert first["summary"]["failed_cases"] == 1
    assert calls == {"disagg": 1, "agg": 1}
    for case_id in ("first", "second"):
        session = output / "cases" / case_id / "execution"
        assert (session / "session-manifest.json").is_file()
        assert len(list((session / "checkpoints").glob("*.json"))) == 1
        assert (output / "cases" / case_id / "index.html").is_file()

    resumed = suite_runner.execute_suite_config(config, output_dir=output, resume=True)
    assert resumed["summary"]["succeeded_cases"] == 2
    assert calls == {"disagg": 1, "agg": 2}
    assert resumed["cases"]["first"]["results"][0]["execution"]["reused"] is True
    assert resumed["cases"]["second"]["results"][0]["execution"]["reused"] is False
    archived = list(
        (output / "cases/second/execution/attempts").glob("*/checkpoint.json")
    )
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["result"]["status"] == "failed"
    assert (archived[0].parent / "run/attempt.txt").read_text() == "1"

    suite_runner.execute_suite_config(config, output_dir=output, resume=True)
    assert calls == {"disagg": 1, "agg": 2}
    assert not (source.parent / "configs/never-publish.json").exists()

    # Operational budgets may be raised without invalidating completed work.
    # Both suite validation and the underlying journals must agree on this.
    path = source.parent / "configs/first.json"
    changed = json.loads(path.read_text())
    changed["execution"] = {
        "timeout_s": 30,
        "max_memory_mb": 128,
        "max_parallel_runs": 2,
    }
    path.write_text(json.dumps(changed))
    updated = suite_runner.execute_suite_config(
        load_suite_config(source), output_dir=output, resume=True
    )
    assert updated["summary"]["succeeded_cases"] == 2
    assert calls == {"disagg": 1, "agg": 2}
    history = updated["configuration_history"]
    assert len(history) == 4
    assert history[0]["config_sha256"] != history[-1]["config_sha256"]
    assert (
        history[0]["cases"]["first"]["config_file_sha256"]
        != history[-1]["cases"]["first"]["config_file_sha256"]
    )
