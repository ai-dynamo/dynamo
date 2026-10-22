# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for rendering saved Arena results without replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from autoscaling_arena.saved_report import (
    SavedReportError,
    load_saved_report,
    main,
    render_saved_result,
)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _normalized_report() -> dict[str, Any]:
    return {
        "match": {
            "name": "saved match",
            "description": "already normalized",
            "backend": "sim",
        },
        "summary": {
            "status": "ok",
            "planned_runs": 1,
            "executed_runs": 1,
            "succeeded_runs": 1,
            "failed_runs": 0,
            "skipped_runs": 0,
            "rank_by": "goodput_rps",
            "metrics": ["goodput_rps"],
        },
        "provenance": {"session_id": "saved-fixture"},
        "resolved_config": {
            "backend": {
                "type": "sim",
                "topology": "agg",
                "model": {"name": "fixture/model"},
                "engines": {
                    "aggregate": {
                        "backend": "vllm",
                        "system": "h200_sxm",
                        "num_gpus": 1,
                    }
                },
                "replay": {"concurrency": None},
            }
        },
        "results": [
            {
                "run_id": "saved-static",
                "backend": "sim",
                "autoscaler": "static",
                "workload": "flat",
                "sla": "interactive",
                "repetition": 0,
                "status": "ok",
                "metrics": {"goodput_rps": 2.5},
                "timeline": [],
                "evaluation": {
                    "sla": {"ttft_ms": 300, "itl_ms": 50, "e2e_ms": None},
                    "arrival_series": {
                        "status": "unavailable",
                        "note": "Saved fixture has no arrivals.",
                        "bucket_width_s": None,
                        "points": [],
                    },
                },
            }
        ],
    }


def _write_saved_match_config(tmp_path: Path) -> Path:
    config = tmp_path / "saved.match.yaml"
    config.write_text(
        """\
schema_version: 1
name: saved-match
backend:
  type: sim
  substrate: gpt_oss
  topology: disagg
  autoscalers:
    - name: static
      type: static
      config: {num_prefill: 1, num_decode: 1}
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
    - type: console
"""
    )
    return config


def test_load_normalized_match_report_without_replay_and_apply_overrides(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "results.json", _normalized_report())

    report = load_saved_report(
        source,
        title="Retitled",
        description="Rendered later",
        rank_by="duration_s",
    )

    assert report["match"]["name"] == "Retitled"
    assert report["match"]["description"] == "Rendered later"
    assert report["summary"]["rank_by"] == "duration_s"
    assert report["summary"]["metrics"] == ["duration_s", "goodput_rps"]
    assert report["results"][0]["metrics"] == {"goodput_rps": 2.5}


def test_render_saved_result_writes_existing_standalone_report_and_normalized_json(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "results.json", _normalized_report())
    html_path = tmp_path / "published" / "report.html"
    normalized_path = tmp_path / "published" / "normalized.json"

    written_html, written_json = render_saved_result(
        source,
        output_path=html_path,
        normalized_output_path=normalized_path,
    )

    assert written_html == html_path
    assert written_json == normalized_path
    rendered = html_path.read_text()
    assert rendered.startswith("<!doctype html>")
    assert 'id="leaderboard-body"' in rendered
    assert 'id="arena-chart"' in rendered
    assert "saved-static" in rendered
    normalized = json.loads(normalized_path.read_text())
    assert normalized["results"][0]["metrics"]["goodput_rps"] == 2.5


def test_render_saved_result_embeds_validated_portable_replay_command(
    tmp_path: Path,
) -> None:
    from autoscaling_arena.match_config import load_match_config
    from autoscaling_arena.match_runner import match_config_sha256, match_replay_sha256

    config_path = _write_saved_match_config(tmp_path)
    config = load_match_config(config_path)
    report = _normalized_report()
    report["results"][0]["run_id"] = next(config.iter_runs()).run_id
    report["provenance"]["config_sha256"] = match_config_sha256(config)
    source = _write_json(tmp_path / "results.json", report)
    output = tmp_path / "report.html"

    render_saved_result(
        source,
        output_path=output,
        match_config_path=config_path,
        replay_config_ref="configs/saved.match.yaml",
    )

    rendered = output.read_text()
    assert "configs/saved.match.yaml" in rendered
    assert f"--expect-config-sha256 {match_replay_sha256(config)}" in rendered
    assert "--run-id 0001-sim-static-flat-interactive-r0-s0" in rendered
    assert str(tmp_path) not in rendered


def test_render_saved_result_uses_safe_config_placeholder_by_default(
    tmp_path: Path,
) -> None:
    from autoscaling_arena.match_config import load_match_config
    from autoscaling_arena.match_runner import match_replay_sha256

    config_path = _write_saved_match_config(tmp_path)
    config = load_match_config(config_path)
    report = _normalized_report()
    report["results"][0]["run_id"] = next(config.iter_runs()).run_id
    # A portable replay digest takes precedence over a path-sensitive legacy
    # provenance digest after the workspace has moved.
    report["provenance"]["config_sha256"] = "0" * 64
    report["provenance"]["replay_config_sha256"] = match_replay_sha256(config)
    source = _write_json(tmp_path / "results.json", report)
    output = tmp_path / "report.html"

    render_saved_result(
        source,
        output_path=output,
        match_config_path=config_path,
    )

    rendered = output.read_text()
    assert '\\"$MATCH_CONFIG\\"' in rendered
    assert str(tmp_path) not in rendered


def test_render_saved_result_rejects_config_drift_and_absolute_replay_ref(
    tmp_path: Path,
) -> None:
    config_path = _write_saved_match_config(tmp_path)
    report = _normalized_report()
    report["provenance"]["config_sha256"] = "0" * 64
    source = _write_json(tmp_path / "results.json", report)

    with pytest.raises(SavedReportError, match="digest does not match"):
        render_saved_result(
            source,
            output_path=tmp_path / "drift.html",
            match_config_path=config_path,
        )

    from autoscaling_arena.match_config import load_match_config
    from autoscaling_arena.match_runner import match_config_sha256

    config = load_match_config(config_path)
    report["results"][0]["run_id"] = next(config.iter_runs()).run_id
    report["provenance"]["config_sha256"] = match_config_sha256(config)
    source.write_text(json.dumps(report))
    with pytest.raises(SavedReportError, match="portable relative path"):
        render_saved_result(
            source,
            output_path=tmp_path / "absolute.html",
            match_config_path=config_path,
            replay_config_ref=str(config_path),
        )


def test_cli_refuses_to_overwrite_an_existing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_json(tmp_path / "results.json", _normalized_report())
    output = tmp_path / "report.html"

    assert main([str(source), "--out", str(output)]) == 0
    original = output.read_text()
    assert main([str(source), "--out", str(output)]) == 2

    assert output.read_text() == original
    assert "output already exists" in capsys.readouterr().err


def test_unknown_saved_schema_is_rejected(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "unknown.json", {"schema": "unknown.v9"})

    with pytest.raises(SavedReportError, match="unsupported saved-results schema"):
        load_saved_report(source)


@pytest.mark.parametrize("destination_kind", ["input", "symlink"])
def test_render_saved_result_preserves_input_and_symlink_targets(
    tmp_path: Path, destination_kind: str
) -> None:
    source = _write_json(tmp_path / "results.json", _normalized_report())
    before = source.read_bytes()
    destination = source
    if destination_kind == "symlink":
        target = tmp_path / "existing.html"
        target.write_text("keep this report")
        destination = tmp_path / "linked.html"
        destination.symlink_to(target)
    with pytest.raises(SavedReportError):
        render_saved_result(source, output_path=destination, overwrite=True)
    assert source.read_bytes() == before
    if destination_kind == "symlink":
        assert target.read_text() == "keep this report"
