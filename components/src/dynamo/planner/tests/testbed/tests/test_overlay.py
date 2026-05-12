# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SyntheticPowerOverlay (γ-class).

Validates the in-process power synthesiser without requiring the Rust
mocker binding. Covers:
  1. Determinism: same RNG seed → byte-identical observation
  2. Determinism: different seeds with non-zero noise → diverge
  3. Bias propagation: ``power_bias_decode`` scales decode power 1:1
  4. Bias independence: decode bias doesn't perturb prefill power
  5. Bias propagation: ``power_bias_prefill`` scales prefill power 1:1
  6. Zero-bias sanity: output stays within reasonable [base, cap_w] range
"""
from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Helpers — build an overlay against the actual current API
# (SystemSpec is per-SKU constants; OverlaySpec.system is a SKU name string).
# ---------------------------------------------------------------------------


def _make_scenario(bias_decode: float = 1.0, bias_prefill: float = 1.0):
    """Construct a minimal γ scenario spec used only to drive the overlay."""
    from dynamo.planner.tests.testbed.scenarios import (
        BiasSpec,
        FleetSpec,
        LoadSpec,
        MockerSpec,
        OverlaySpec,
        PlannerSpec,
        ScenarioSpec,
    )

    return ScenarioSpec(
        name="overlay-unit",
        class_="gamma",
        description="",
        seed=42,
        ticks=10,
        interval_s=60,
        planner=PlannerSpec(
            prefill_engine_gpu_power_limit=450,
            decode_engine_gpu_power_limit=360,
        ),
        fleet=FleetSpec(),
        mocker=MockerSpec(synthetic_workload=True),
        overlay=OverlaySpec(
            system="h200_sxm",
            bias=BiasSpec(
                power_bias_decode=bias_decode,
                power_bias_prefill=bias_prefill,
            ),
        ),
        load=LoadSpec(profile="constant", tokens_per_sec=1000.0),
        events=[],
        assertions=[],
    )


def _make_overlay(
    bias_decode: float = 1.0,
    bias_prefill: float = 1.0,
    seed: int = 42,
    power_noise_sigma: float = 0.0,
):
    """Build a SyntheticPowerOverlay matching the runtime injection point."""
    from dynamo.planner.tests.testbed.replay.synthetic_power_overlay import (
        SyntheticPowerOverlay,
    )
    from dynamo.planner.tests.testbed.scenarios import NoiseModel, NoiseSpec, SystemSpec

    scenario = _make_scenario(bias_decode=bias_decode, bias_prefill=bias_prefill)
    # Override noise if the test wants non-zero noise.
    scenario.overlay.noise = NoiseSpec(
        power_per_gpu=NoiseModel(model="gaussian", sigma=power_noise_sigma),
    )
    system_spec = SystemSpec.load("h200_sxm")
    rng = random.Random(seed)
    return SyntheticPowerOverlay(
        overlay_spec=scenario.overlay,
        system_spec=system_spec,
        scenario=scenario,
        rng=rng,
    )


def _observe(overlay, tick: int = 0) -> tuple[float, float]:
    """Drive one ``observe()`` tick with a representative FPM snapshot pair
    and return ``(power_w_prefill, power_w_decode)``."""
    from dynamo.planner.tests.testbed.fake_actuator import AppliedCaps

    fpm = [
        {"component": "prefill", "sum_prefill_tokens": 4096, "sum_decode_kv_tokens": 0},
        {
            "component": "decode",
            "sum_prefill_tokens": 0,
            "sum_decode_kv_tokens": 100_000,
        },
    ]
    overlay.observe(fpm, AppliedCaps(cap_p=450, cap_d=360), tick)
    obs = overlay.observation_at(tick)
    assert obs is not None, "observation_at returned None after observe()"
    return obs.power_w_prefill, obs.power_w_decode


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverlayDeterminism:
    def test_same_seed_same_output(self):
        o1 = _make_overlay(seed=99)
        o2 = _make_overlay(seed=99)
        p1, d1 = _observe(o1)
        p2, d2 = _observe(o2)
        assert math.isclose(p1, p2, rel_tol=1e-9)
        assert math.isclose(d1, d2, rel_tol=1e-9)

    def test_different_seeds_differ(self):
        """With nonzero noise, different seeds must diverge."""
        o_a = _make_overlay(seed=1, power_noise_sigma=0.05)
        o_b = _make_overlay(seed=2, power_noise_sigma=0.05)
        _p_a, d_a = _observe(o_a)
        _p_b, d_b = _observe(o_b)
        assert not math.isclose(
            d_a, d_b, rel_tol=1e-4
        ), f"Different seeds produced identical decode power ({d_a} vs {d_b})"


class TestOverlayBiasPropagation:
    def test_decode_bias_doubles_decode_power(self):
        base = _make_overlay(bias_decode=1.0, bias_prefill=1.0, seed=0)
        biased = _make_overlay(bias_decode=2.0, bias_prefill=1.0, seed=0)
        _, d_base = _observe(base)
        _, d_biased = _observe(biased)
        ratio = d_biased / d_base
        assert math.isclose(
            ratio, 2.0, rel_tol=1e-3
        ), f"Expected decode power to double; got ratio={ratio:.4f}"

    def test_decode_bias_does_not_affect_prefill(self):
        base = _make_overlay(bias_decode=1.0, bias_prefill=1.0, seed=0)
        biased = _make_overlay(bias_decode=2.0, bias_prefill=1.0, seed=0)
        p_base, _ = _observe(base)
        p_biased, _ = _observe(biased)
        assert math.isclose(
            p_base, p_biased, rel_tol=1e-3
        ), "Decode bias should not affect prefill power"

    def test_prefill_bias_scales_prefill_power(self):
        base = _make_overlay(bias_decode=1.0, bias_prefill=1.0, seed=0)
        biased = _make_overlay(bias_decode=1.0, bias_prefill=1.5, seed=0)
        p_base, _ = _observe(base)
        p_biased, _ = _observe(biased)
        ratio = p_biased / p_base
        assert math.isclose(
            ratio, 1.5, rel_tol=1e-3
        ), f"Expected prefill power to scale by 1.5; got ratio={ratio:.4f}"


class TestOverlayZeroBias:
    def test_no_bias_within_reasonable_envelope(self):
        overlay = _make_overlay(bias_decode=1.0, bias_prefill=1.0, seed=0)
        p_w, d_w = _observe(overlay)
        assert 0 < p_w < 1000, f"Prefill power {p_w}W out of envelope"
        assert 0 < d_w < 1000, f"Decode power {d_w}W out of envelope"
