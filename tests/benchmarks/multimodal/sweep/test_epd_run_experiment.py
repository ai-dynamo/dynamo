# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchmarks.multimodal.sweep.experiments.epd import run_experiment
from benchmarks.multimodal.sweep.experiments.epd.run_experiment import (
    ReplayCell,
    ReplaySchedule,
    build_cells,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("backend", ["vllm", "VLLM"]),
        ("image_count", ["1", "01"]),
        ("image_token_budget", ["64", "064"]),
        ("isl", ["1", "01"]),
        ("osl", ["1", "01"]),
        ("qps", ["0.5", ".5"]),
    ],
)
def test_build_cells_rejects_duplicates_after_normalization(axis, values):
    args = Namespace(
        backend=["vllm"],
        topology=["aggregate"],
        image_count=["1"],
        image_token_budget=["-1"],
        isl=["1"],
        osl=["1"],
        qps=["0.5"],
    )
    setattr(args, axis, values)

    with pytest.raises(ValueError, match="duplicate .* after normalization"):
        build_cells(args)


@pytest.fixture
def schedule_file(tmp_path):
    path = tmp_path / "schedule.jsonl"
    rows = [
        {
            "timestamp": timestamp,
            "payload": {
                "model": "served-model",
                "stream": True,
                "messages": [{"role": "user", "content": f"Request {index}"}],
                "max_tokens": index + 1,
            },
        }
        for index, timestamp in enumerate((10, 25.5, 25.5))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("", "schedule is empty"),
        ("{truncated", "invalid JSON"),
        ('{"timestamp": 30, "payload": {}}', "streaming chat payload"),
        (
            '{"timestamp": 30, "payload": {}, "session_id": "a"}',
            "timestamp and payload",
        ),
        ('{"timestamp": true, "payload": {}}', "timestamps must be finite"),
        ('{"timestamp": NaN, "payload": {}}', "invalid JSON: nonfinite number"),
        ('{"timestamp": -1, "payload": {}}', "nonnegative"),
    ],
)
def test_replay_rejects_invalid_schedule(tmp_path, contents, match):
    schedule = tmp_path / "invalid.jsonl"
    schedule.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ReplaySchedule.read(schedule, "served-model")


def test_replay_rejects_out_of_order_rows(schedule_file):
    rows = schedule_file.read_text(encoding="utf-8").splitlines()
    schedule_file.write_text("\n".join(reversed(rows)), encoding="utf-8")
    with pytest.raises(ValueError, match=":3:.*nondecreasing"):
        ReplaySchedule.read(schedule_file, "served-model")


def test_replay_rejects_nonfinite_payload_value(schedule_file):
    rows = schedule_file.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["payload"]["temperature"] = float("inf")
    rows[0] = json.dumps(row)
    schedule_file.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=":1: invalid JSON: nonfinite number"):
        ReplaySchedule.read(schedule_file, "served-model")


def test_schedule_conflicts_with_generated_axes(schedule_file, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run_experiment.main(
            [
                "--backend",
                "vllm",
                "--topology",
                "epd",
                "--model",
                "served-model",
                "--output-dir",
                str(tmp_path / "results"),
                "--schedule-file",
                str(schedule_file),
                "--qps",
                "0.5",
                "--dry-run",
            ]
        )

    assert error.value.code == 2
    assert "--schedule-file cannot be combined with --qps" in capsys.readouterr().err


def test_generated_workload_keeps_defaults(tmp_path, capsys):
    assert (
        run_experiment.main(
            [
                "--backend",
                "vllm",
                "--topology",
                "aggregate",
                "--model",
                "served-model",
                "--output-dir",
                str(tmp_path / "results"),
                "--image-dir",
                str(tmp_path),
                "--image-count",
                "2",
                "--osl",
                "32",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "vllm/aggregate/default/img2-isl9000-osl32-qps0p5" in capsys.readouterr().out


def test_replay_passes_the_same_schedule_to_every_cell(
    schedule_file, tmp_path, monkeypatch
):
    original = schedule_file.read_bytes()
    outputs = tmp_path / "results"
    groups = []

    @contextmanager
    def fake_service(key, args, output_dir):
        groups.append(key)
        yield "http://localhost:test"

    def fake_aiperf(command, **kwargs):
        if command[1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout=run_experiment.EXPECTED_AIPERF_VERSION
            )
        assert "--fixed-schedule" in command
        arg = {
            option: command[command.index(option) + 1]
            for option in (
                "--custom-dataset-type",
                "--fixed-schedule-start-offset",
                "--request-count",
                "--export-level",
                "--input-file",
                "--artifact-dir",
            )
        }
        assert arg["--custom-dataset-type"] == "mooncake_trace"
        assert arg["--fixed-schedule-start-offset"] == "0"
        assert arg["--request-count"] == "3"
        assert arg["--export-level"] == "raw"
        assert not {"--request-rate", "--concurrency", "--warmup-request-count"} & set(
            command
        )
        assert Path(arg["--input-file"]).read_bytes() == original
        schedule_file.write_bytes(b"changed after validation")
        report = {
            "aiperf_version": run_experiment.EXPECTED_AIPERF_VERSION,
            "request_count": {"avg": 3},
            "request_throughput": {"avg": 42},
            "replay_sched_lag_p99": {"avg": 2.5},
        }
        (Path(arg["--artifact-dir"]) / "profile_export_aiperf.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_experiment.shutil, "which", lambda _: "/aiperf")
    monkeypatch.setattr(run_experiment.subprocess, "run", fake_aiperf)
    monkeypatch.setattr(run_experiment, "managed_service", fake_service)

    assert (
        run_experiment.main(
            [
                "--backend",
                "vllm",
                "sglang",
                "--topology",
                "aggregate",
                "epd",
                "--model",
                "different-model-id",
                "--served-model-name",
                "served-model",
                "--output-dir",
                str(outputs),
                "--schedule-file",
                str(schedule_file),
            ]
        )
        == 0
    )
    assert groups == [
        ("vllm", "aggregate", None),
        ("vllm", "epd", None),
        ("sglang", "aggregate", None),
        ("sglang", "epd", None),
    ]
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs.glob("*/fixed-schedule/summary.json")
    ]
    assert len(summaries) == 4
    assert len({summary["schedule_sha256"] for summary in summaries}) == 1
    for summary in summaries:
        assert summary["request_count"] == 3
        assert summary["achieved_qps"] == 42
        assert summary["schedule_lag_p99_ms"] == 2.5


@pytest.mark.parametrize(
    ("report_update", "match"),
    [
        ({"request_count": {"avg": 2}}, "request_count=2 expected=3"),
        ({"error_summary": [{"count": 1}]}, "errors=1"),
        ({"was_cancelled": True}, "was_cancelled=true"),
    ],
)
def test_replay_rejects_invalid_results(schedule_file, tmp_path, report_update, match):
    cell = ReplayCell("vllm", "epd", ReplaySchedule.read(schedule_file, "served-model"))
    path = tmp_path / "profile_export_aiperf.json"
    report = {
        "aiperf_version": run_experiment.EXPECTED_AIPERF_VERSION,
        "request_count": {"avg": 3},
        **report_update,
    }
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=match):
        run_experiment.validate_result(path, cell)
