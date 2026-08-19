# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the program-state model."""

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


def test_end_request_records_real_token_total():
    table = ProgramTable()
    table.begin_request("p1")
    p = table.end_request("p1", prompt_tokens=120, completion_tokens=30)
    assert p is not None
    assert p.token_total == 150
    assert p.status == ProgramStatus.ACTING


def test_rollback_drops_a_program_the_attempt_created():
    table = ProgramTable()
    snapshot = table.snapshot_request("p1")
    assert snapshot is None
    table.begin_request("p1", estimated_prompt_tokens=64)
    table.paused["p1"] = None

    table.rollback_request("p1", snapshot)

    assert "p1" not in table.programs
    assert "p1" not in table.paused


def test_rollback_restores_a_pre_existing_program_instead_of_dropping_it():
    table = ProgramTable()
    table.begin_request("p1")
    table.end_request("p1", prompt_tokens=100, completion_tokens=10)
    program = table.programs["p1"]
    program.lifecycle = ProgramLifecycle.PAUSED
    table.paused["p1"] = None
    acting_since = program.acting_since

    snapshot = table.snapshot_request("p1")
    table.begin_request("p1", estimated_prompt_tokens=500)

    table.rollback_request("p1", snapshot)

    assert table.programs["p1"] is program
    assert program.step_count == 1
    assert program.token_total == 110
    assert program.status == ProgramStatus.ACTING
    assert program.lifecycle == ProgramLifecycle.PAUSED
    assert program.acting_since == acting_since
    assert "p1" in table.paused


def test_rollback_leaves_a_replacement_program_alone():
    table = ProgramTable()
    table.begin_request("p1", estimated_prompt_tokens=10)
    snapshot = table.snapshot_request("p1")
    table.release("p1")
    replacement = table.begin_request("p1", estimated_prompt_tokens=42)

    table.rollback_request("p1", snapshot)

    assert table.programs["p1"] is replacement
    assert replacement.token_total == 42
    assert replacement.step_count == 1
