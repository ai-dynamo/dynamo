# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for resolving the release in development on main."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import release_line  # noqa: E402


def test_newest_minor_branch_wins() -> None:
    branches = ["release/1.4.0", "release/1.6.0", "release/1.5.0"]

    assert release_line.newest_release_line(branches) == (1, 6)


@pytest.mark.parametrize(
    "branch",
    [
        "release/1.7.1",
        "release/1.7.0-glm-dev.1",
        "release/1.7.0.post1",
        "release/v1.7.0",
        "release/1.7",
    ],
)
def test_patch_and_non_minor_branches_do_not_move_main(branch: str) -> None:
    assert release_line.newest_release_line(["release/1.6.0", branch]) == (1, 6)


def test_a_new_major_outranks_the_last_minor() -> None:
    assert release_line.newest_release_line(["release/1.9.0", "release/2.0.0"]) == (
        2,
        0,
    )


def test_no_release_branch_resolves_to_none() -> None:
    assert release_line.newest_release_line(["release/1.6.1"]) is None


@pytest.mark.parametrize(
    ("line", "expected"),
    [((1, 6), "1.7.0"), ((1, 8), "1.9.0"), ((1, 9), "2.0.0"), ((2, 0), "2.1.0")],
)
def test_next_version_rolls_the_major_after_nine(
    line: tuple[int, int], expected: str
) -> None:
    assert release_line.next_version(*line) == expected
