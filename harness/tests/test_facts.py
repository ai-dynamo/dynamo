# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`dynamo_test.facts`."""

import pytest
from dynamo_test.facts import Fact, FactNotKnown, Status


def test_known_carries_the_value_and_its_address():
    fact = Fact.known("Qwen/Qwen3-0.6B", "DGD spec.components[Worker].args[0]")
    assert fact.is_known
    assert fact.require() == "Qwen/Qwen3-0.6B"
    assert fact.source == "DGD spec.components[Worker].args[0]"


def test_absent_and_unknown_are_distinct_states():
    absent = Fact.absent("args", "scanned 12 tokens; --model is not among them")
    unknown = Fact.unknown("args", "command could not be tokenised")
    assert absent.status is Status.ABSENT
    assert unknown.status is Status.UNKNOWN
    assert absent.status is not unknown.status


def test_require_names_what_could_not_be_read():
    fact = Fact.unknown("args[0]", "unterminated single quote at offset 41")
    with pytest.raises(FactNotKnown) as exc:
        fact.require()
    assert "args[0]" in str(exc.value)
    assert "unterminated single quote" in str(exc.value)


def test_a_fact_has_no_truth_value():
    """``if not fact:`` is the exact shape of both measured false-greens.

    A scan that could not read its source and one that read it and found
    nothing are both falsy, so the bug is invisible at the call site. Raising
    makes it unwritable instead.
    """
    with pytest.raises(TypeError) as exc:
        bool(Fact.unknown("args", "unreadable"))
    assert "ABSENT from UNKNOWN" in str(exc.value)

    with pytest.raises(TypeError):
        if Fact.known("x", "src"):  # noqa: SIM103
            pass


def test_or_else_collapses_absent_and_unknown_deliberately():
    assert Fact.known("a", "src").or_else("z") == "a"
    assert Fact.absent("src", "not set").or_else("z") == "z"
    assert Fact.unknown("src", "unreadable").or_else("z") == "z"


def test_facts_are_immutable():
    fact = Fact.known("a", "src")
    with pytest.raises(Exception):
        fact.value = "b"  # type: ignore[misc]
