#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from dynamo.replay.config import resolve_planner_profile_data

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_an_existing_npz_is_used_directly(tmp_path: Path) -> None:
    npz = tmp_path / "profile.npz"
    npz.write_bytes(b"not really an npz, only its presence matters here")

    assert resolve_planner_profile_data(npz).npz_path == npz


def test_a_missing_npz_is_rejected_rather_than_accepted_on_its_suffix(
    tmp_path: Path,
) -> None:
    """The mocker resolver is the only step that checks the path exists.

    Short-circuiting on the suffix skipped it, so a mistyped .npz was accepted
    here and rejected by the same input under dynamo.mocker.
    """
    with pytest.raises(FileNotFoundError):
        resolve_planner_profile_data(tmp_path / "typo.npz")


def test_no_planner_profile_data_stays_none() -> None:
    assert resolve_planner_profile_data(None).npz_path is None
