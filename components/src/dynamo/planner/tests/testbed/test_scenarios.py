"""
pytest entry point: parametrize over every scenario YAML in scenarios/.

Usage:
    # run all scenarios
    pytest components/src/dynamo/planner/tests/testbed/test_scenarios.py -v

    # run only α-class scenarios
    pytest components/src/dynamo/planner/tests/testbed/test_scenarios.py -v \
        -k "alpha"

    # run only γ-class scenarios (requires mocker trace)
    pytest components/src/dynamo/planner/tests/testbed/test_scenarios.py -v \
        -k "gamma"
"""
from __future__ import annotations

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------
_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_ALPHA_YAMLS = sorted(
    p for p in _SCENARIOS_DIR.glob("*.yaml") if not p.stem.startswith("_")
    and not p.stem.startswith("G")
)
_GAMMA_YAMLS = sorted(
    p for p in _SCENARIOS_DIR.glob("G*.yaml")
)


def _scenario_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# α-class
# ---------------------------------------------------------------------------
@pytest.mark.testbed
@pytest.mark.parametrize(
    "scenario_path",
    _ALPHA_YAMLS,
    ids=[_scenario_id(p) for p in _ALPHA_YAMLS],
)
def test_alpha(scenario_path: Path, tmp_path: Path) -> None:
    """Run one α-class scenario and assert all expectations pass."""
    from dynamo.planner.tests.testbed.scenarios import load_scenario
    from dynamo.planner.tests.testbed.runner import ScenarioRunner
    from dynamo.planner.tests.testbed.assertions import evaluate_all

    scenario = load_scenario(scenario_path)
    runner = ScenarioRunner(scenario)
    history = runner.run()

    csv_path = tmp_path / f"{scenario.name}.csv"
    history.to_csv(csv_path)

    failures = evaluate_all(history, scenario)
    if failures:
        msg = "\n".join(f"  [{i}] {f}" for i, f in enumerate(failures, 1))
        pytest.fail(
            f"Scenario {scenario.name!r}: {len(failures)} assertion(s) failed:\n{msg}"
        )


# ---------------------------------------------------------------------------
# γ-class
# ---------------------------------------------------------------------------
@pytest.mark.testbed
@pytest.mark.gamma
@pytest.mark.parametrize(
    "scenario_path",
    _GAMMA_YAMLS,
    ids=[_scenario_id(p) for p in _GAMMA_YAMLS],
)
def test_gamma(scenario_path: Path, tmp_path: Path) -> None:
    """Run one γ-class scenario and assert all expectations pass."""
    pytest.importorskip(
        "dynamo.llm",
        reason="γ-class scenarios require the dynamo.llm (mocker) package",
    )
    from dynamo.planner.tests.testbed.scenarios import load_scenario
    from dynamo.planner.tests.testbed.runner import ScenarioRunner
    from dynamo.planner.tests.testbed.assertions import evaluate_all

    scenario = load_scenario(scenario_path)
    runner = ScenarioRunner(scenario)
    history = runner.run()

    csv_path = tmp_path / f"{scenario.name}.csv"
    history.to_csv(csv_path)

    failures = evaluate_all(history, scenario)
    if failures:
        msg = "\n".join(f"  [{i}] {f}" for i, f in enumerate(failures, 1))
        pytest.fail(
            f"Scenario {scenario.name!r}: {len(failures)} assertion(s) failed:\n{msg}"
        )
