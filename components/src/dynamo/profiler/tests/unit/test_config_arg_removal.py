# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared argument removers in profiler/utils/config.py."""

import pytest

from dynamo.profiler.utils.config import (
    remove_all_argument_occurrences,
    remove_valued_arguments,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]

REMOVERS = (remove_all_argument_occurrences, remove_valued_arguments)


@pytest.mark.parametrize("remove", REMOVERS)
def test_key_without_a_value_does_not_consume_the_next_flag(remove) -> None:
    """Both removers used to delete the following token unconditionally, so a
    mistyped ``--tp`` swallowed the next argument and left its value behind as
    a stray positional."""
    args = ["--tp", "--model-path", "/models/m", "--trust-remote-code"]

    assert remove(list(args), "--tp") == [
        "--model-path",
        "/models/m",
        "--trust-remote-code",
    ]


@pytest.mark.parametrize("remove", REMOVERS)
def test_negative_values_are_still_consumed(remove) -> None:
    """``-1`` is a value, not the start of another argument."""
    args = ["--max-model-len", "-1", "--model-path", "/models/m"]

    assert remove(list(args), "--max-model-len") == ["--model-path", "/models/m"]


@pytest.mark.parametrize("remove", REMOVERS)
def test_ordinary_valued_argument_still_removes_both_tokens(remove) -> None:
    args = ["--model-path", "/models/m", "--tp", "2", "--trust-remote-code"]

    assert remove(list(args), "--tp") == [
        "--model-path",
        "/models/m",
        "--trust-remote-code",
    ]


@pytest.mark.parametrize("remove", REMOVERS)
def test_trailing_key_is_removed(remove) -> None:
    assert remove(["--model-path", "/models/m", "--tp"], "--tp") == [
        "--model-path",
        "/models/m",
    ]
