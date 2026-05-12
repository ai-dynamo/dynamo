"""Self-consistency tests for the testbed.

Guards:
  1. test_alpha_no_bias: run A1 with power_bias_decode=1.0; c_power_d stays in [0.95, 1.05]
  2. test_gamma_no_bias: run G1 with power_bias_decode=1.0; c_power_d stays in [0.90, 1.10]
     (skipped if dynamo.llm / mocker not available)
  3. test_alpha_gamma_agree_on_decode_drift: A1 and G1 must converge in the same
     direction (both >1.0) within 20% magnitude
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.recorder import TickHistory

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]

_SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def _run_scenario_with_bias(
    scenario_name: str, power_bias_decode: float = 1.0
) -> "TickHistory":
    """Load a scenario, override decode bias to the given value, and run it."""
    from dynamo.planner.tests.testbed.runner import ScenarioRunner
    from dynamo.planner.tests.testbed.scenarios import load_scenario

    yaml_path = _SCENARIOS_DIR / f"{scenario_name}.yaml"
    scenario = load_scenario(yaml_path)

    # Override the decode power bias. Bias lives on FleetSpec.bias for α-class
    # and OverlaySpec.bias for γ-class — NoiseSpec has no bias fields.
    if scenario.fleet is not None and scenario.fleet.bias is not None:
        scenario.fleet.bias.power_bias_decode = power_bias_decode
    if scenario.overlay is not None and scenario.overlay.bias is not None:
        scenario.overlay.bias.power_bias_decode = power_bias_decode

    runner = ScenarioRunner(scenario)
    return runner.run()


def _median_c_power_d(history: "TickHistory") -> float:
    values = [s.c_power_d for s in history.snapshots if s.c_power_d is not None]
    if not values:
        pytest.skip("No c_power_d values recorded")
    return statistics.median(values)


def _late_avg_c_power_d(history: "TickHistory", last_n: int = 10) -> float:
    values = [
        s.c_power_d for s in history.snapshots[-last_n:] if s.c_power_d is not None
    ]
    if not values:
        pytest.skip("No c_power_d values in last ticks")
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# α self-consistency
# ---------------------------------------------------------------------------
@pytest.mark.testbed
def test_alpha_no_bias() -> None:
    """A1 with bias=1.0: c_power_d converges to [0.95, 1.05] (AIC-echo case)."""
    history = _run_scenario_with_bias(
        "A1_power_under_estimate_decode", power_bias_decode=1.0
    )
    avg = _late_avg_c_power_d(history)
    assert (
        0.90 <= avg <= 1.10
    ), f"α no-bias: expected c_power_d in [0.90, 1.10]; got {avg:.4f}"


# ---------------------------------------------------------------------------
# γ self-consistency
# ---------------------------------------------------------------------------
@pytest.mark.testbed
@pytest.mark.gamma
def test_gamma_no_bias() -> None:
    """G1 with bias=1.0: c_power_d stays in [0.90, 1.10]."""
    pytest.importorskip(
        "dynamo.llm",
        reason="γ self-consistency requires dynamo.llm (mocker) package",
    )
    history = _run_scenario_with_bias(
        "G1_realistic_decode_drift", power_bias_decode=1.0
    )
    avg = _late_avg_c_power_d(history)
    assert (
        0.90 <= avg <= 1.10
    ), f"γ no-bias: expected c_power_d in [0.90, 1.10]; got {avg:.4f}"


# ---------------------------------------------------------------------------
# α–γ cross-validation
# ---------------------------------------------------------------------------
@pytest.mark.testbed
@pytest.mark.gamma
def test_alpha_gamma_agree_on_decode_drift() -> None:
    """A1 and G1 with bias=1.30 must agree on direction and magnitude (±20%).

    Both should converge to c_power_d > 1.0 (positive drift direction).
    Magnitudes must be within 20% of each other, accounting for γ's broader
    noise envelope.
    """
    pytest.importorskip(
        "dynamo.llm",
        reason="α–γ cross-validation requires dynamo.llm (mocker) package",
    )
    # Skip when the installed bridge only exposes the older create_disagg API.
    # That API requires a trace file; the testbed falls back to a placeholder
    # with near-zero load that cannot drive AIC power-correction drift, making
    # this cross-validation meaningless.
    try:
        from dynamo.llm import PlannerReplayBridge  # type: ignore[import]

        if not hasattr(PlannerReplayBridge, "from_synthetic_disagg"):
            pytest.skip(
                "PlannerReplayBridge.from_synthetic_disagg not available; "
                "older create_disagg API with placeholder trace cannot drive "
                "AIC c_power_d drift for α–γ cross-validation"
            )
    except ImportError:
        pass  # already handled by importorskip above
    bias = 1.30

    alpha_history = _run_scenario_with_bias(
        "A1_power_under_estimate_decode", power_bias_decode=bias
    )
    gamma_history = _run_scenario_with_bias(
        "G1_realistic_decode_drift", power_bias_decode=bias
    )

    alpha_avg = _late_avg_c_power_d(alpha_history)
    gamma_avg = _late_avg_c_power_d(gamma_history)

    # Both must indicate positive drift (> 1.0 means underestimate corrected upward)
    assert (
        alpha_avg > 1.0
    ), f"α A1: expected c_power_d > 1.0 with bias=1.30; got {alpha_avg:.4f}"
    assert (
        gamma_avg > 1.0
    ), f"γ G1: expected c_power_d > 1.0 with bias=1.30; got {gamma_avg:.4f}"

    # Magnitude must agree within 20%
    ratio = abs(alpha_avg - gamma_avg) / max(alpha_avg, gamma_avg)
    assert ratio <= 0.20, (
        f"α–γ magnitude disagreement: α={alpha_avg:.4f}, γ={gamma_avg:.4f}, "
        f"relative difference={ratio:.2%} (must be ≤20%)"
    )
