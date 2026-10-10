# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import compile_run
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_raw_run(
    raw_root: Path, run_id: str = "20260828T123456Z-baseline-abcdef"
) -> Path:
    run_dir = raw_root / run_id
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metadata.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "kind": "baseline",
            "control_plane": {
                "mode": "stock",
                "standalone_controller_run_id": None,
                "native_planner": None,
            },
            "started": "2026-08-28T12:34:56Z",
            "ended": "2026-08-28T12:35:06Z",
            "status": "completed",
            "exit_code": 0,
            "configuration": {"batch_size": 2},
        },
    )
    _write_jsonl(
        run_dir / "progress.jsonl",
        [
            {
                "observed_at": "2026-08-28T12:34:58Z",
                "elapsed_seconds": 2.0,
                "status": "in_progress",
                "total": 2,
                "completed": 0,
                "failed": 0,
                "remaining": 2,
                "delta_completed": 0,
                "interval_seconds": 0.0,
                "interval_completion_rate_rps": 0.0,
            },
            {
                "observed_at": "2026-08-28T12:35:06Z",
                "elapsed_seconds": 10.0,
                "status": "completed",
                "total": 2,
                "completed": 2,
                "failed": 0,
                "remaining": 0,
                "delta_completed": 2,
                "interval_seconds": 8.0,
                "interval_completion_rate_rps": 0.25,
            },
        ],
    )
    _write_jsonl(
        run_dir / "online-requests.jsonl",
        [
            {
                "request_index": 1,
                "http_status": 200,
                "ok": True,
                "latency_ms": 500.0,
                "ttft_ms": 100.0,
                "completion_tokens": 12,
            },
            {
                "request_index": 2,
                "http_status": 503,
                "ok": False,
                "latency_ms": 1500.0,
                "ttft_ms": None,
                "completion_tokens": None,
            },
        ],
    )
    return run_dir


def test_compile_run_generates_summary_tables_and_provenance(tmp_path: Path) -> None:
    raw_root = tmp_path / "results" / "raw"
    compiled_root = tmp_path / "compiled"
    run_dir = _make_raw_run(raw_root)

    output_dir = compile_run.compile_run(
        tmp_path, run_dir.name, compiled_root / "summary"
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["batch"]["completed"] == 2
    assert summary["source_control_plane"]["mode"] == "stock"
    assert summary["batch"]["duration_seconds"] == 10.0
    assert summary["online"]["sample_count"] == 2
    assert summary["online"]["successful_requests"] == 1
    assert summary["online"]["latency_ms"]["p50"] == 500.0
    assert summary["data_quality"]["issues"] == []
    assert (output_dir / "README.md").is_file()

    with (output_dir / "progress.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["completed"] for row in rows] == ["0", "2"]
    assert (output_dir / "online-requests.csv").is_file()


def test_compile_run_records_nonmonotonic_progress(tmp_path: Path) -> None:
    raw_root = tmp_path / "results" / "raw"
    compiled_root = tmp_path / "compiled"
    run_dir = _make_raw_run(raw_root)
    _write_jsonl(
        run_dir / "progress.jsonl",
        [
            {
                "observed_at": "2026-08-28T12:34:58Z",
                "elapsed_seconds": 2.0,
                "status": "in_progress",
                "total": 2,
                "completed": 2,
            },
            {
                "observed_at": "2026-08-28T12:35:06Z",
                "elapsed_seconds": 1.0,
                "status": "completed",
                "total": 2,
                "completed": 1,
            },
        ],
    )

    output_dir = compile_run.compile_run(
        tmp_path, run_dir.name, compiled_root / "summary"
    )
    issues = json.loads((output_dir / "summary.json").read_text())["data_quality"][
        "issues"
    ]

    assert any("backward in time" in issue for issue in issues)
    assert any("decreases completed count" in issue for issue in issues)


def test_compile_run_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    raw_root = tmp_path / "results" / "raw"
    compiled_root = tmp_path / "compiled"
    run_dir = _make_raw_run(raw_root)
    output = compiled_root / "summary"
    compile_run.compile_run(tmp_path, run_dir.name, output)

    with pytest.raises(FileExistsError):
        compile_run.compile_run(tmp_path, run_dir.name, output)


def test_compile_run_cleans_staging_directory_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "results" / "raw"
    run_dir = _make_raw_run(raw_root)
    output = tmp_path / "compiled" / "summary"

    def fail_write_json(_path: Path, _value: object) -> None:
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(compile_run, "write_json", fail_write_json)

    with pytest.raises(RuntimeError, match="injected write failure"):
        compile_run.compile_run(tmp_path, run_dir.name, output)

    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_iter_jsonl_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"latency_ms": NaN}\n', encoding="utf-8")

    with pytest.raises(compile_run.HarnessError, match="non-finite JSON"):
        list(compile_run.iter_jsonl(path))


def test_iter_jsonl_enforces_explicit_record_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.jsonl"
    _write_jsonl(path, [{"request_index": index} for index in range(4)])
    monkeypatch.setattr(compile_run, "MAX_JSONL_RECORDS", 3)

    with pytest.raises(compile_run.HarnessError, match="3-record compile limit"):
        list(compile_run.iter_jsonl(path))
