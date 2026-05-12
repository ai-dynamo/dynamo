# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CircuitBreaker using VirtualClock."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.types import CircuitState

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _cb(**kwargs):
    clock = VirtualClock()
    return clock, CircuitBreaker(clock, **kwargs)


def test_initial_state_closed_for_unknown_plugin():
    _, cb = _cb()
    assert cb.state("p1") == CircuitState.CLOSED
    assert cb.can_call("p1") is True


def test_transitions_to_open_after_threshold_failures():
    _, cb = _cb(failure_threshold=3)
    for _ in range(2):
        cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.CLOSED
    cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.OPEN
    assert cb.can_call("p1") is False


def test_success_resets_failure_count_before_opening():
    _, cb = _cb(failure_threshold=3)
    cb.record_failure("p1")
    cb.record_failure("p1")
    cb.record_success("p1")
    cb.record_failure("p1")
    cb.record_failure("p1")
    # Still below threshold after the reset.
    assert cb.state("p1") == CircuitState.CLOSED


def test_open_transitions_to_half_open_after_cooldown():
    clock, cb = _cb(failure_threshold=1, cooldown_seconds=10.0)
    cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.OPEN

    clock.advance(5.0)
    assert cb.state("p1") == CircuitState.OPEN  # still in cooldown

    clock.advance(5.0)
    # Now 10s elapsed -> HALF_OPEN via the lazy check in state().
    assert cb.state("p1") == CircuitState.HALF_OPEN
    assert cb.can_call("p1") is True


def test_half_open_success_returns_to_closed():
    clock, cb = _cb(failure_threshold=1, cooldown_seconds=10.0)
    cb.record_failure("p1")
    clock.advance(10.0)
    assert cb.state("p1") == CircuitState.HALF_OPEN

    cb.record_success("p1")
    assert cb.state("p1") == CircuitState.CLOSED


def test_half_open_failure_reopens_and_resets_cooldown():
    clock, cb = _cb(failure_threshold=1, cooldown_seconds=10.0)
    cb.record_failure("p1")
    clock.advance(10.0)
    assert cb.state("p1") == CircuitState.HALF_OPEN

    cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.OPEN

    # Cooldown starts over from the reopen moment.
    clock.advance(9.0)
    assert cb.state("p1") == CircuitState.OPEN
    clock.advance(1.0)
    assert cb.state("p1") == CircuitState.HALF_OPEN


def test_reset_clears_state_back_to_closed():
    _, cb = _cb(failure_threshold=2)
    cb.record_failure("p1")
    cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.OPEN
    cb.reset("p1")
    assert cb.state("p1") == CircuitState.CLOSED
    # Implicit entry cleared; counter starts from zero.
    cb.record_failure("p1")
    assert cb.state("p1") == CircuitState.CLOSED


def test_on_open_callback_fires_on_closed_to_open():
    _, cb = _cb(failure_threshold=2)
    opens: list[str] = []
    cb.on_open(opens.append)
    cb.record_failure("p1")
    assert opens == []
    cb.record_failure("p1")
    assert opens == ["p1"]


def test_on_open_callback_fires_on_half_open_to_open_reopen():
    clock, cb = _cb(failure_threshold=1, cooldown_seconds=5.0)
    opens: list[str] = []
    cb.on_open(opens.append)
    cb.record_failure("p1")
    assert opens == ["p1"]

    clock.advance(5.0)  # HALF_OPEN
    cb.record_failure("p1")  # reopen
    assert opens == ["p1", "p1"]


def test_multiple_plugins_tracked_independently():
    clock, cb = _cb(failure_threshold=2, cooldown_seconds=5.0)
    cb.record_failure("p1")
    cb.record_failure("p1")
    cb.record_failure("p2")
    assert cb.state("p1") == CircuitState.OPEN
    assert cb.state("p2") == CircuitState.CLOSED


def test_invalid_config_rejected():
    clock = VirtualClock()
    with pytest.raises(ValueError):
        CircuitBreaker(clock, failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(clock, cooldown_seconds=0)
