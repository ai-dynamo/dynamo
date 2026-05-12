# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI: G3 behavior parity verify.

Re-runs each scenario in ``scenarios.ALL_SCENARIOS`` through the current
``PlannerStateMachine`` and asserts the output matches the locked fixture
in ``golden/<scenario_name>.jsonl`` byte-for-byte.

This test is the **fixture lock guard**:
- After the ``pre-plugin-architecture`` git tag, any change to
  ``core/state_machine.py`` / ``core/load_scaling.py`` /
  ``core/throughput_scaling.py`` that alters ``on_tick`` output will
  cause this test to fail in CI.
- During the plugin framework rollout the test ensures we don't
  accidentally break PSM behavior (the PSM modules are explicitly
  read-only until the eventual cleanup).

If a fixture diverges intentionally (e.g. PSM is removed and the
test gets rewritten to run against the orchestrator), the README
protocol documents how to regenerate fixtures.

Picked up by the existing ``planner-test`` CI job via pytest markers
``pre_merge and planner and gpu_0``.
"""

import pytest

from .dump_tool import (
    DEFAULT_OUTPUT_DIR,
    _compare_records,
    _read_fixture,
    _run_scenario,
)
from .scenarios import ALL_SCENARIOS

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_g3_fixture_parity(scenario):
    """Re-run scenario through current PSM, compare to locked fixture."""
    fixture_path = DEFAULT_OUTPUT_DIR / f"{scenario.name}.jsonl"
    assert fixture_path.exists(), (
        f"missing fixture: {fixture_path}\n"
        "Generate via: python -m planner.tests.plugins.g3_fixtures.dump_tool"
    )

    expected = _read_fixture(fixture_path)
    actual = _run_scenario(scenario)
    diffs = _compare_records(expected, actual, scenario.name)

    assert not diffs, (
        f"G3 fixture parity broken for scenario '{scenario.name}'.\n"
        "PSM behavior changed since fixture was locked at "
        "git tag 'pre-plugin-architecture'. Investigate:\n"
        "1. Was PSM intentionally modified? Check core/state_machine.py / "
        "load_scaling.py / throughput_scaling.py recent changes.\n"
        "2. If unintentional, fix the regression.\n"
        "3. If intentional (e.g. PSM removal), regenerate fixtures via "
        "the README protocol.\n\n"
        "Diffs:\n" + "\n\n".join(diffs)
    )
