# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the program-state model.

These tests do not require Dynamo runtime / KvRouter; they validate the pure
state machine.
"""

from __future__ import annotations

import pytest

from dynamo.thunderagent_router.program_state import (
    ProgramLifecycle,
    ProgramStatus,
    ProgramTable,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_begin_request_creates_program_in_reasoning():
    table = ProgramTable()
    program = table.begin_request("p1")
    assert program.program_id == "p1"
    assert program.status == ProgramStatus.REASONING
    assert program.lifecycle == ProgramLifecycle.ACTIVE
    assert program.step_count == 1


def test_begin_request_increments_step_and_resets_acting_since():
    table = ProgramTable()
    p = table.begin_request("p1")
    table.end_request("p1", prompt_tokens=10, completion_tokens=5)
    assert p.acting_since > 0
    table.begin_request("p1")
    assert p.step_count == 2
    assert p.acting_since == 0.0
    assert p.status == ProgramStatus.REASONING


def test_end_request_records_real_token_totals_and_prefix():
    table = ProgramTable()
    table.begin_request("p1")
    p = table.end_request(
        "p1", prompt_tokens=120, completion_tokens=30, last_prefix_token_ids=[1, 2, 3]
    )
    assert p is not None
    assert p.last_prompt_tokens == 120
    assert p.last_completion_tokens == 30
    assert p.token_total == 150
    assert p.last_prefix_token_ids == [1, 2, 3]
    assert p.status == ProgramStatus.ACTING


def test_release_clears_state_and_signals_waiter():
    import asyncio

    table = ProgramTable()
    table.begin_request("p1")
    p = table.programs["p1"]
    p.waiting = asyncio.Event()
    p.lifecycle = ProgramLifecycle.PAUSED
    table.paused["p1"] = True

    released = table.release("p1")
    assert released is not None
    assert released.lifecycle == ProgramLifecycle.TERMINATED
    assert "p1" not in table.programs
    assert "p1" not in table.paused
    assert released.waiting is None


def test_counts_reflects_lifecycle_and_status():
    table = ProgramTable()
    table.begin_request("a")
    table.end_request("a", 10, 5)  # ACTING
    table.begin_request("b")  # REASONING
    table.begin_request("c")
    table.programs["c"].lifecycle = ProgramLifecycle.PAUSED

    counts = table.counts()
    assert counts == {"reasoning": 1, "acting": 1, "paused": 1}
