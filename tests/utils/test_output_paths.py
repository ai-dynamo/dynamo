# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def test_repeat_execution_keeps_previous_attempt_output(tmp_path: Path) -> None:
    attempts: dict[str, int] = {}
    log_dir = tmp_path / "test_deployment[sglang-agg_embed-normal]"
    log_dir.mkdir()
    (log_dir / "stale.txt").write_text("earlier session")
    stale_attempt = tmp_path / f"{log_dir.name}.attempt1"
    stale_attempt.mkdir()

    prepare_test_output_dir(str(log_dir), attempts)

    assert list(log_dir.iterdir()) == []
    assert not stale_attempt.exists()

    (log_dir / "test.log.txt").write_text("first attempt")
    prepare_test_output_dir(str(log_dir), attempts)

    assert list(log_dir.iterdir()) == []
    assert (stale_attempt / "test.log.txt").read_text() == "first attempt"

    (log_dir / "test.log.txt").write_text("second attempt")
    prepare_test_output_dir(str(log_dir), attempts)

    assert (stale_attempt / "test.log.txt").read_text() == "first attempt"
    second_attempt = tmp_path / f"{log_dir.name}.attempt2"
    assert (second_attempt / "test.log.txt").read_text() == "second attempt"
    assert list(log_dir.iterdir()) == []


def test_same_output_dir_shares_attempt_count(tmp_path: Path) -> None:
    # Same-named tests in different files resolve to one output directory.
    attempts: dict[str, int] = {}
    log_dir = tmp_path / "test_deployment"

    prepare_test_output_dir(str(log_dir), attempts)
    (log_dir / "test.log.txt").write_text("tests/a/test_x.py")
    prepare_test_output_dir(str(log_dir), attempts)

    archived = tmp_path / f"{log_dir.name}.attempt1"
    assert (archived / "test.log.txt").read_text() == "tests/a/test_x.py"
    assert list(log_dir.iterdir()) == []
