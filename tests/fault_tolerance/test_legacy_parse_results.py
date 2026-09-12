# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.fault_tolerance.deploy import legacy_parse_results

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

FAULT_TIME = datetime(2026, 8, 13, 12, 0, 0)


def _write_ready_log(
    test_dir: Path,
    process_name: str,
    message: str,
    recovery_seconds: int,
) -> None:
    process_dir = test_dir / process_name
    process_dir.mkdir()
    timestamp = FAULT_TIME + timedelta(seconds=recovery_seconds)
    log_entry = {
        "time": f"{timestamp.isoformat()}Z",
        "message": message,
    }
    (process_dir / "replica-0.log").write_text(
        json.dumps(log_entry) + "\n", encoding="utf-8"
    )


@pytest.fixture
def current_worker_result(tmp_path: Path) -> Path:
    _write_ready_log(
        tmp_path,
        "worker",
        "worker for Qwen/Qwen3-0.6B has been initialized",
        recovery_seconds=5,
    )
    return tmp_path


@pytest.fixture
def archived_pre_rename_result(tmp_path: Path) -> Path:
    for process_name, recovery_seconds in (
        ("VllmWorker", 3),
        ("VllmDecodeWorker", 7),
        ("VllmPrefillWorker", 11),
    ):
        _write_ready_log(
            tmp_path,
            process_name,
            "VllmWorker for Qwen/Qwen3-0.6B has been initialized",
            recovery_seconds,
        )
    return tmp_path


def test_calculate_recovery_time_parses_current_aggregate_worker(
    current_worker_result: Path,
) -> None:
    recovery_time = legacy_parse_results.calculate_recovery_time(
        current_worker_result, "worker", FAULT_TIME
    )

    assert recovery_time == 5


def test_calculate_recovery_time_parses_archived_vllm_result(
    archived_pre_rename_result: Path,
) -> None:
    recovery_time = legacy_parse_results.calculate_recovery_time(
        archived_pre_rename_result, "worker", FAULT_TIME
    )

    assert recovery_time == 11


def test_calculate_recovery_time_scans_each_process_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanned_directories = []

    def record_scan(log_dir, process_name):
        scanned_directories.append((Path(log_dir).name, process_name))
        return {}

    monkeypatch.setattr(legacy_parse_results, "parse_process_log", record_scan)

    recovery_time = legacy_parse_results.calculate_recovery_time(
        tmp_path, "worker", FAULT_TIME
    )

    assert recovery_time is None
    assert scanned_directories == [
        ("Frontend", "Frontend"),
        ("worker", "worker"),
        ("decode", "decode"),
        ("prefill", "prefill"),
        ("VllmWorker", "VllmWorker"),
        ("VllmDecodeWorker", "VllmDecodeWorker"),
        ("VllmPrefillWorker", "VllmPrefillWorker"),
        ("TRTLLMWorker", "TRTLLMWorker"),
    ]
