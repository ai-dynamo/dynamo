#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import random

import pytest

from dynamo.frontend.token_prefix import TokenPrefix


def test_delta_token_is_absent_from_the_previous_view():
    prefix = TokenPrefix()
    prefix.append([1, 2])
    previous, current = prefix.append([9])
    assert 9 not in previous
    assert 9 in current
    assert list(previous) == [1, 2]
    assert list(current) == [1, 2, 9]
    assert previous[-1] == 2
    assert current[-2] == 2


def test_saved_view_ignores_later_appends_and_reset():
    prefix = TokenPrefix()
    _, previous = prefix.append([4, 5])
    prefix.append([6])
    assert list(previous) == [4, 5]
    assert 6 not in previous
    prefix.adopt([])
    assert list(previous) == [4, 5]
    assert len(prefix) == 0


def test_adopt_of_the_full_view_does_not_copy():
    prefix = TokenPrefix()
    _, current = prefix.append([1, 2, 3])
    ids = prefix._ids
    prefix.adopt(current)
    assert prefix._ids is ids
    assert list(prefix.view()) == [1, 2, 3]


def test_matches_list_concatenation():
    rng = random.Random(0)
    prefix = TokenPrefix()
    accumulated: list[int] = []
    for _ in range(400):
        delta = [rng.randrange(50) for _ in range(rng.choice((0, 1, 1, 3)))]
        previous, current = prefix.append(delta)
        assert list(previous) == accumulated
        accumulated = accumulated + delta
        assert list(current) == accumulated
        probe = rng.randrange(60)
        before = accumulated[: len(accumulated) - len(delta)]
        assert (probe in previous) == (probe in before)
        assert (probe in current) == (probe in accumulated)
        if before:
            assert previous[-1] == before[-1]
        else:
            with pytest.raises(IndexError):
                previous[-1]


def test_unhashable_membership_matches_a_list():
    prefix = TokenPrefix()
    _, current = prefix.append([1])
    assert ([] in current) is False
