# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from benchmarks.multimodal.sweep.runner import _RESULT_MARKERS, result_exists

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def test_result_exists_all_markers_present(tmp_path: Path) -> None:
    for marker in _RESULT_MARKERS:
        (tmp_path / marker).write_text("data")
    assert result_exists(tmp_path) is True


def test_result_exists_missing_marker(tmp_path: Path) -> None:
    (tmp_path / _RESULT_MARKERS[0]).write_text("data")
    assert result_exists(tmp_path) is False


def test_result_exists_empty_marker(tmp_path: Path) -> None:
    for marker in _RESULT_MARKERS:
        (tmp_path / marker).write_text("")
    assert result_exists(tmp_path) is False


def test_result_exists_one_empty_one_present(tmp_path: Path) -> None:
    (tmp_path / _RESULT_MARKERS[0]).write_text("data")
    (tmp_path / _RESULT_MARKERS[1]).write_text("")
    assert result_exists(tmp_path) is False


def test_result_exists_nonexistent_dir() -> None:
    assert result_exists(Path("/nonexistent/dir")) is False
