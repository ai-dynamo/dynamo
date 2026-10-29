# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path

import pytest

from tests.utils.output_paths import prepare_test_output_dir, resolve_test_output_path

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("test[*]/log.txt", "test[%2A]/log.txt"),
        (
            'test["<>:|*?\r\n%]/log.txt',
            "test[%22%3C%3E%3A%7C%2A%3F%0D%0A%25]/log.txt",
        ),
        (
            "test[value with spaces]/log.txt",
            "test[value with spaces]/log.txt",
        ),
        ("suite*/test?/log.txt", "suite%2A/test%3F/log.txt"),
        (Path("test[*]/log.txt"), "test[%2A]/log.txt"),
        ("test[%2A]/log.txt", "test[%252A]/log.txt"),
    ],
)
def test_relative_output_paths_encode_special_characters(
    path: str | Path,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DYN_TEST_OUTPUT_PATH", str(tmp_path))
    assert resolve_test_output_path(path) == str(tmp_path / expected)


@pytest.mark.parametrize("as_path", [False, True])
def test_absolute_output_paths_are_unchanged(
    as_path: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DYN_TEST_OUTPUT_PATH", str(tmp_path / "custom-root"))
    absolute = tmp_path / 'test["<>:|*?\r\n%]/log.txt'
    path = absolute if as_path else str(absolute)

    assert resolve_test_output_path(path) == str(absolute)


def _make_stale(path: Path, session_start: float) -> None:
    os.utime(path, (session_start - 60, session_start - 60))


def test_repeat_execution_keeps_previous_attempt_output(tmp_path: Path) -> None:
    session_start = time.time()
    log_dir = tmp_path / "test_deployment[sglang-agg_embed-normal]"
    log_dir.mkdir()
    (log_dir / "stale.txt").write_text("earlier session")
    stale_attempt = tmp_path / f"{log_dir.name}.attempt1"
    stale_attempt.mkdir()
    _make_stale(log_dir, session_start)
    _make_stale(stale_attempt, session_start)

    prepare_test_output_dir(str(log_dir), session_start)

    assert list(log_dir.iterdir()) == []
    assert not stale_attempt.exists()

    (log_dir / "test.log.txt").write_text("first attempt")
    prepare_test_output_dir(str(log_dir), session_start)

    assert list(log_dir.iterdir()) == []
    assert (stale_attempt / "test.log.txt").read_text() == "first attempt"

    (log_dir / "test.log.txt").write_text("second attempt")
    prepare_test_output_dir(str(log_dir), session_start)

    assert (stale_attempt / "test.log.txt").read_text() == "first attempt"
    second_attempt = tmp_path / f"{log_dir.name}.attempt2"
    assert (second_attempt / "test.log.txt").read_text() == "second attempt"
    assert list(log_dir.iterdir()) == []


def test_keeps_attempts_written_by_another_worker(tmp_path: Path) -> None:
    # xdist workers and same-named tests in different files share one directory.
    session_start = time.time()
    log_dir = tmp_path / "test_deployment"
    other_worker_attempt = tmp_path / f"{log_dir.name}.attempt1"
    other_worker_attempt.mkdir()
    (other_worker_attempt / "test.log.txt").write_text("worker gw0")
    log_dir.mkdir()
    (log_dir / "test.log.txt").write_text("worker gw1")

    prepare_test_output_dir(str(log_dir), session_start)

    assert (other_worker_attempt / "test.log.txt").read_text() == "worker gw0"
    archived = tmp_path / f"{log_dir.name}.attempt2"
    assert (archived / "test.log.txt").read_text() == "worker gw1"
    assert list(log_dir.iterdir()) == []
