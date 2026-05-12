# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario YAML loadable gate.

Every YAML in scenarios/ must:
  1. Parse without error (incl. ``extends:`` resolution).
  2. Pass Pydantic validation on every event and assertion.
  3. Reference only fields that exist on ``TickSnapshot``.
  4. Use only valid ``ref:`` prefixes (planner / counters / fleet / overlay).
  5. Have ``class: alpha`` or ``class: gamma``.

The heavy lifting lives in ``scenarios.py`` (single source of truth); this
file just iterates and reports.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def _iter_scenario_yamls() -> list[Path]:
    """Return every non-base scenario YAML in sorted order."""
    return sorted(
        p for p in _SCENARIOS_DIR.glob("*.yaml") if not p.stem.startswith("_")
    )


# ---------------------------------------------------------------------------
# Collect load-time errors at module import so failures surface during
# pytest collection rather than only when the test functions run.
# ---------------------------------------------------------------------------
_LOAD_ERRORS: list[str] = []

try:
    from dynamo.planner.tests.testbed.scenarios import load_scenario

    for _yaml_path in _iter_scenario_yamls():
        try:
            load_scenario(_yaml_path)
        except Exception as _exc:
            # Truncate huge ValidationError stacks; the first line names the
            # offending field which is what authors need.
            _LOAD_ERRORS.append(f"  {_yaml_path.stem}: {str(_exc).splitlines()[0]}")
except ImportError as _e:  # pragma: no cover — only when packaging is broken
    _LOAD_ERRORS.append(f"  Could not import scenarios module: {_e}")


def test_all_scenario_yamls_loadable() -> None:
    """Every YAML in scenarios/ parses successfully with valid field/ref names."""
    if _LOAD_ERRORS:
        msg = "\n".join(_LOAD_ERRORS)
        pytest.fail(
            f"{len(_LOAD_ERRORS)} scenario YAML error(s) detected at collection:\n{msg}"
        )


@pytest.mark.parametrize(
    "scenario_path",
    _iter_scenario_yamls(),
    ids=[p.stem for p in _iter_scenario_yamls()],
)
def test_scenario_yaml_parses(scenario_path: Path) -> None:
    """Each individual YAML parses, validates, and has a valid class label."""
    from dynamo.planner.tests.testbed.scenarios import load_scenario

    scenario = load_scenario(scenario_path)
    assert scenario.name, f"{scenario_path.stem} has no name"
    assert scenario.class_name in (
        "alpha",
        "gamma",
    ), f"{scenario_path.stem} has invalid class={scenario.class_name!r}"
