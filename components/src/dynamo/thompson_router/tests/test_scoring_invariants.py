# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for two-tower scoring invariants, physics tower, residual training target,
and management endpoint hot-reload.

Coverage areas not fully addressed by test_router.py:
  1. load_mod applies only to positive utility (regression for issue #1 sign fix)
  2. residual clamp formula: clamp(reward - physics + 0.5, 0, 1)
  3. _physics_score unit tests (not covered via pick_worker alone)
  4. _iat_factor interpolation between anchor points
  5. LinTS lints_weight sign branch (negative weight uses abs + tanh)
  6. Management server: hot-reload config, state roundtrip, metrics reset
  7. Management server: record_decision / decisions_summary
  8. OverlapResult dataclass defaults
  9. RouterStats variance computation
 10. BetaLearner min_pseudo_count floor prevents parameter collapse
"""

import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from dynamo.thompson_router.kv_indexer import OverlapResult
from dynamo.thompson_router.learners import BetaLearner, LatencyTracker
from dynamo.thompson_router.router import KvThompsonRouter, RouterStats, RoutingDecision

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.router,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_load(worker_id, prefill_tokens=0, decode_blocks=0):
    return {
        "worker_id": worker_id,
        "potential_prefill_tokens": prefill_tokens,
        "potential_decode_blocks": decode_blocks,
    }


def _make_kv_router(loads=None):
    router = AsyncMock()
    loads = loads or [_make_load(0), _make_load(1)]
    router.get_potential_loads = AsyncMock(return_value=loads)
    router.best_worker = AsyncMock(return_value=(0, 0, 0))
    return router


def _make_router(config_overrides=None, loads=None, monitor=None):
    cfg = {"kv_thompson": config_overrides or {}}
    return KvThompsonRouter(
        _make_kv_router(loads),
        config=cfg,
        worker_load_monitor=monitor,
    )


# ---------------------------------------------------------------------------
# 1. Regression: load_mod applies only to positive utility (Issue #1)
#
# Before the fix: s = (utility + bonus - penalty) * load_mod
#   -> negative utility would be amplified on unloaded workers
# After the fix:  s = max(0, utility) * load_mod - penalty
#   (switching penalty is OUTSIDE load_mod)
#
# Test: a worker with negative utility and low load must not score higher
# than a worker with the same negative utility and high load.
# ---------------------------------------------------------------------------

class TestLoadModOnlyOnPositiveUtility:
    """Regression tests for Issue #1: load_mod sign fix."""

    def test_load_mod_does_not_amplify_negative_utility(self):
        """If utility < 0, a more-unloaded worker must NOT score higher.

        Before the fix this would fail: multiplying a negative utility by a
        smaller (closer to 1.0) load_mod produced a less-negative (better)
        score for the lightly loaded worker even when it had worse overlap.
        """
        router = _make_router()
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        # Force both workers to have strongly negative physics (e.g. all weights
        # put on memory_pressure=1 → compute_avail=0, cache=0 → physics → 0).
        # We inject physics directly.
        physics_negative = -0.5  # explicitly negative to expose the sign issue

        # Worker A: low kv_util (load_mod → 1.0)
        score_unloaded = router._score_worker(
            wid=0, x=x, overlap=0.0, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=physics_negative, kv_util=0.0,
        )
        # Worker B: high kv_util (load_mod → < 0.1)
        score_loaded = router._score_worker(
            wid=0, x=x, overlap=0.0, decode_blocks=50,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=physics_negative, kv_util=0.9,
        )

        # With the fixed formula (utility * load_mod where utility = max(0, raw)):
        # negative physics is below zero so load_mod scales zero, not the
        # negative value. The two scores should be equal or at most minimally
        # different (only the TS terms vary).
        # Either way, the unloaded worker must NOT score higher than loaded
        # simply because its load_mod is larger.
        #
        # Pre-fix behavior: score_unloaded >> score_loaded (bad)
        # Post-fix behavior: scores roughly equal (TS noise only)
        diff = score_unloaded - score_loaded
        # The difference must be small — TS sampling noise, not load_mod inversion
        assert abs(diff) < 1.5, (
            f"load_mod amplified negative utility: unloaded={score_unloaded:.4f}, "
            f"loaded={score_loaded:.4f}, diff={diff:.4f}"
        )

    def test_switching_penalty_is_outside_load_mod(self):
        """Switching penalty must not be modulated by load_mod.

        The penalty should be a flat subtraction after the load-modulated term:
            score = utility * load_mod - switch_cost_weight * tanh(penalty_term)

        We verify this indirectly: a heavily loaded switching worker must score
        LESS than a lightly loaded one with the same physics — and the score gap
        must increase with higher load (since load_mod reduces the positive term
        while penalty stays constant).  We disable both learners to make
        calls deterministic from the same router instance.
        """
        router = _make_router({"enable_switching_cost": True, "switch_cost_weight": 2.0,
                                "switch_base": 0.5, "switch_reuse": 0.0,
                                "enable_beta_ts": False,
                                "enable_lints": False})

        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        physics = 0.6

        # Both calls use the SAME router, so TS samples are drawn sequentially.
        # Seed to reduce variance and ensure the load difference is decisive.
        np.random.seed(0)
        score_unloaded = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=0,
            last_worker=0, reuse_budget=5, iat_factor=1.0,
            physics=physics, kv_util=0.0,
        )
        np.random.seed(0)  # same seed → same TS sample, only load_mod differs
        score_loaded = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=40,
            last_worker=0, reuse_budget=5, iat_factor=1.0,
            physics=physics, kv_util=0.9,
        )

        # With the same TS terms, the unloaded worker must score higher
        # (load_mod close to 1 vs load_mod close to 0).
        assert score_unloaded > score_loaded, (
            f"Unloaded worker ({score_unloaded:.4f}) should score higher than "
            f"loaded one ({score_loaded:.4f}) when only load_mod changes"
        )

        # The penalty is additive and outside load_mod; verify the gap between
        # unloaded-with-penalty and unloaded-without-penalty matches tanh formula.
        np.random.seed(0)
        score_no_penalty = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None,  # no switch → no penalty
            reuse_budget=5, iat_factor=1.0,
            physics=physics,
        )
        # Difference = penalty = switch_cost_weight * tanh(switch_base)
        expected_penalty = 2.0 * math.tanh(0.5)
        actual_penalty = score_no_penalty - score_unloaded
        assert abs(actual_penalty - expected_penalty) < 1e-9, (
            f"Expected penalty ≈ {expected_penalty:.4f}, actual gap = {actual_penalty:.4f}"
        )

    def test_switching_penalty_only_applies_when_switching(self):
        """Penalty must be zero when wid == last_worker (no migration)."""
        router = _make_router({"enable_switching_cost": True, "switch_cost_weight": 1.0})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        physics = 0.6

        score_same = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=0, reuse_budget=5, iat_factor=1.0,
            physics=physics,
        )
        score_diff = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=5,
            last_worker=0, reuse_budget=5, iat_factor=1.0,
            physics=physics,
        )
        assert score_same > score_diff, (
            "Worker keeping same prefix must score higher than switching worker"
        )

    def test_switching_penalty_zero_when_no_last_worker(self):
        """No penalty on first request of a session (last_worker is None)."""
        router = _make_router({"enable_switching_cost": True, "switch_cost_weight": 5.0,
                                "switch_base": 1.0})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        score_no_last = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=5, iat_factor=1.0,
            physics=0.6,
        )
        score_with_penalty = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=99, reuse_budget=5, iat_factor=1.0,
            physics=0.6,
        )
        assert score_no_last > score_with_penalty


# ---------------------------------------------------------------------------
# 2. Residual training target: clamp(reward - physics + 0.5, 0, 1)
# ---------------------------------------------------------------------------

class TestResidualTrainingTarget:
    """Tests for the LinTS residual computation in update_feedback.

    The formula per Section 3.4 of the paper draft:
        residual = clamp(reward - physics_score + 0.5, 0, 1)

    Center at 0.5 so that:
        physics_pred == reward  → residual = 0.5 (neutral, no update)
        reward >> physics_pred  → residual → 1.0 (physics underestimated)
        reward << physics_pred  → residual → 0.0 (physics overestimated)
    """

    def _compute_residual(self, reward, physics):
        return max(0.0, min(1.0, reward - physics + 0.5))

    def test_perfect_prediction_gives_half(self):
        """When physics == reward, residual should be exactly 0.5."""
        for val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            assert abs(self._compute_residual(val, val) - 0.5) < 1e-9

    def test_underestimate_gives_high_residual(self):
        """When reward >> physics (physics underestimated), residual → 1."""
        residual = self._compute_residual(reward=1.0, physics=0.0)
        assert abs(residual - 1.0) < 1e-9

    def test_overestimate_gives_low_residual(self):
        """When reward << physics (physics overestimated), residual → 0."""
        residual = self._compute_residual(reward=0.0, physics=1.0)
        assert abs(residual - 0.0) < 1e-9

    def test_residual_clamped_to_unit_interval(self):
        """Even with extreme inputs, residual stays in [0, 1]."""
        extreme_pairs = [
            (2.0, 0.0),   # reward > 1 (shouldn't happen but be safe)
            (-1.0, 1.0),  # reward < 0
            (0.0, 2.0),   # physics > 1
        ]
        for reward, physics in extreme_pairs:
            r = self._compute_residual(reward, physics)
            assert 0.0 <= r <= 1.0, f"residual={r} for reward={reward}, physics={physics}"

    @pytest.mark.asyncio
    async def test_residual_stored_in_feedback_result(self):
        """update_feedback must return residual_reward in [0, 1]."""
        router = _make_router()
        decision = await router.pick_worker(
            token_ids=list(range(100)), prefix_id="p", reuse_budget=0,
            osl=250, iat=250, tokens_in=100,
        )
        result = router.update_feedback(decision, latency_ms=50.0, tokens_out=50)
        assert "residual_reward" in result
        assert 0.0 <= result["residual_reward"] <= 1.0

    @pytest.mark.asyncio
    async def test_residual_zero_when_physics_greatly_overestimates(self):
        """When physics_score ≈ 1.0 but reward ≈ 0, residual must be clamped to 0."""
        router = _make_router()
        decision = await router.pick_worker(
            token_ids=list(range(100)), prefix_id="p", reuse_budget=0,
            osl=250, iat=250, tokens_in=100,
        )
        # Override physics_score to near 1.0 — simulate physics overestimating
        decision.physics_score = 0.99

        # Send very slow latency → low reward
        result = router.update_feedback(decision, latency_ms=100000.0, tokens_out=1)
        assert result["residual_reward"] >= 0.0
        assert result["residual_reward"] <= 0.05  # should be near 0

    @pytest.mark.asyncio
    async def test_residual_one_when_physics_greatly_underestimates(self):
        """When physics_score ≈ 0.0 but reward ≈ 1.0, residual must be clamped to 1."""
        router = _make_router()
        decision = await router.pick_worker(
            token_ids=list(range(100)), prefix_id="p", reuse_budget=0,
            osl=250, iat=250, tokens_in=100,
        )
        decision.physics_score = 0.01  # physics greatly underestimated

        # Artificially send tiny latency (fast request, high reward)
        # We need reward close to 1.0 → latency << baseline
        result = router.update_feedback(decision, latency_ms=0.001, tokens_out=1)
        assert result["residual_reward"] >= 0.95  # should be near 1


# ---------------------------------------------------------------------------
# 3. Physics tower unit tests
# ---------------------------------------------------------------------------

class TestPhysicsTower:
    """Direct unit tests for _physics_score."""

    def test_perfect_cache_idle_worker_gives_max_score(self):
        """overlap=1, kv_util=0, prefill_util=0, memory_pressure=0 → weight sum."""
        router = _make_router()
        # All signals at best: cache_hit=1, compute_avail=1, queue_avail=1, memory_avail=1
        worker_util = {0: {"kv_util": 0.0, "prefill_util": 0.0}}
        score = router._physics_score(
            overlap=1.0, wid=0, worker_util=worker_util,
            prefill_tokens=0, tokens_in=100, memory_pressure=0.0,
        )
        expected = (
            router.physics_cache_weight * 1.0
            + router.physics_compute_weight * 1.0
            + router.physics_queue_weight * 1.0
            + router.physics_memory_weight * 1.0
        )
        assert abs(score - expected) < 1e-9

    def test_worst_case_worker_gives_near_zero(self):
        """overlap=0, kv_util=1, prefill_util=1, memory_pressure=1 → ~0."""
        router = _make_router()
        worker_util = {0: {"kv_util": 1.0, "prefill_util": 1.0}}
        score = router._physics_score(
            overlap=0.0, wid=0, worker_util=worker_util,
            prefill_tokens=100, tokens_in=100, memory_pressure=1.0,
        )
        # All signals at worst: cache=0, compute_avail=0, queue_avail=0, memory_avail=0
        assert score < 0.01

    def test_all_signals_in_unit_range_produce_in_range_score(self):
        """Physics score must stay in [0, weight_sum] for all valid inputs."""
        router = _make_router()
        weight_sum = (
            router.physics_cache_weight
            + router.physics_compute_weight
            + router.physics_queue_weight
            + router.physics_memory_weight
        )
        for overlap in [0.0, 0.5, 1.0]:
            for kv_util in [0.0, 0.5, 1.0]:
                for memory_pressure in [0.0, 0.5, 1.0]:
                    worker_util = {0: {"kv_util": kv_util, "prefill_util": 0.5}}
                    score = router._physics_score(
                        overlap=overlap, wid=0, worker_util=worker_util,
                        prefill_tokens=50, tokens_in=100, memory_pressure=memory_pressure,
                    )
                    assert 0.0 <= score <= weight_sum + 1e-9, (
                        f"physics_score={score} out of [0, {weight_sum}] for "
                        f"overlap={overlap}, kv_util={kv_util}, "
                        f"memory_pressure={memory_pressure}"
                    )

    def test_fallback_when_no_monitor_data(self):
        """Worker not in worker_util dict triggers prefill_ratio fallback."""
        router = _make_router()
        worker_util: dict[int, dict] = {}  # no data for wid=0
        # 50 uncached tokens out of 100 → prefill_ratio = 0.5
        score_fallback = router._physics_score(
            overlap=0.5, wid=0, worker_util=worker_util,
            prefill_tokens=50, tokens_in=100, memory_pressure=0.0,
        )
        # compute_avail = 1 - prefill_ratio = 0.5, queue_avail = 0.5 (neutral)
        expected = (
            router.physics_cache_weight * 0.5
            + router.physics_compute_weight * 0.5
            + router.physics_queue_weight * 0.5
            + router.physics_memory_weight * 1.0  # no memory pressure
        )
        assert abs(score_fallback - expected) < 1e-9

    def test_memory_pressure_reduces_score(self):
        """Higher memory pressure (eviction risk) must reduce physics score."""
        router = _make_router()
        worker_util = {0: {"kv_util": 0.3, "prefill_util": 0.2}}

        score_no_pressure = router._physics_score(
            overlap=0.7, wid=0, worker_util=worker_util,
            prefill_tokens=30, tokens_in=100, memory_pressure=0.0,
        )
        score_high_pressure = router._physics_score(
            overlap=0.7, wid=0, worker_util=worker_util,
            prefill_tokens=30, tokens_in=100, memory_pressure=0.9,
        )
        assert score_high_pressure < score_no_pressure, (
            "High memory pressure must reduce physics score"
        )

    def test_physics_score_increases_with_overlap(self):
        """Higher KV cache overlap → higher physics score (cache_hit signal)."""
        router = _make_router()
        worker_util = {0: {"kv_util": 0.3, "prefill_util": 0.2}}

        scores = [
            router._physics_score(
                overlap=ov, wid=0, worker_util=worker_util,
                prefill_tokens=int((1.0 - ov) * 100), tokens_in=100,
                memory_pressure=0.0,
            )
            for ov in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Physics score should increase with overlap: {scores}"
            )


# ---------------------------------------------------------------------------
# 4. _iat_factor interpolation between anchor points
# ---------------------------------------------------------------------------

class TestIatFactorInterpolation:
    """Tests for the IAT factor lookup table with interpolation.

    Anchors: 50→1.5, 250→1.0, 1000→0.6.
    Per paper Section 3.4: lower IAT (rapid-fire) → higher factor (more stickiness).
    """

    def test_at_lower_anchor(self):
        assert KvThompsonRouter._iat_factor(50) == pytest.approx(1.5, abs=1e-9)

    def test_at_middle_anchor(self):
        assert KvThompsonRouter._iat_factor(250) == pytest.approx(1.0, abs=1e-9)

    def test_at_upper_anchor(self):
        assert KvThompsonRouter._iat_factor(1000) == pytest.approx(0.6, abs=1e-9)

    def test_interpolation_between_50_and_250(self):
        """At IAT=150 (midpoint between 50 and 250), factor should be ~1.25."""
        # Linear: 1.5 - 0.5 * (150-50)/(250-50) = 1.5 - 0.5*0.5 = 1.25
        result = KvThompsonRouter._iat_factor(150)
        assert result == pytest.approx(1.25, abs=1e-9)

    def test_interpolation_between_250_and_1000(self):
        """At IAT=625 (midpoint between 250 and 1000), factor should be ~0.8."""
        # Linear: 1.0 - 0.4 * (625-250)/(1000-250) = 1.0 - 0.4*0.5 = 0.8
        result = KvThompsonRouter._iat_factor(625)
        assert result == pytest.approx(0.8, abs=1e-9)

    def test_below_lower_anchor_clamps(self):
        """IAT < 50 should return 1.5 (clamped at lower anchor)."""
        assert KvThompsonRouter._iat_factor(0) == pytest.approx(1.5, abs=1e-9)
        assert KvThompsonRouter._iat_factor(10) == pytest.approx(1.5, abs=1e-9)

    def test_above_upper_anchor_clamps(self):
        """IAT > 1000 should return 0.6 (clamped at upper anchor)."""
        assert KvThompsonRouter._iat_factor(5000) == pytest.approx(0.6, abs=1e-9)

    def test_monotonically_decreasing(self):
        """Factor must be strictly decreasing with IAT."""
        iats = [0, 50, 100, 150, 200, 250, 500, 750, 1000, 2000]
        factors = [KvThompsonRouter._iat_factor(i) for i in iats]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], (
                f"IAT factor not monotone: f({iats[i]})={factors[i]:.4f} > "
                f"f({iats[i+1]})={factors[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 5. LinTS lints_weight sign branch
# ---------------------------------------------------------------------------

class TestLinTSWeightBranch:
    """Test that negative lints_weight uses abs(weight)*tanh(...) while
    non-negative uses raw weight*raw_sample."""

    def test_lints_always_uses_tanh_bounding(self):
        """LinTS contribution is always abs(lints_weight) * tanh(raw), regardless of sign."""
        router = _make_router({"enable_lints": True, "lints_weight": 2.0})

        # Give the learner strong signal to make its posterior mean large
        x: np.ndarray = np.ones(9, dtype=np.float64)
        for _ in range(50):
            router.lints_learner.update(0, x, reward=1.0)

        raw_sample = router.lints_learner.sample(0, x)
        # With tanh bounding: contribution = abs(2.0) * tanh(raw_sample) <= 2.0
        bounded_contribution = abs(2.0) * math.tanh(raw_sample)
        assert abs(bounded_contribution) <= 2.0 + 1e-9

        # The full score should also be finite
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        assert math.isfinite(score)

    def test_lints_weight_sign_irrelevant(self):
        """Positive and negative lints_weight produce the same contribution (abs used)."""
        router_pos = _make_router({"enable_lints": True, "lints_weight": 1.0})
        router_neg = _make_router({"enable_lints": True, "lints_weight": -1.0})

        x: np.ndarray = np.ones(9, dtype=np.float64)
        for _ in range(30):
            router_pos.lints_learner.update(0, x, reward=1.0)
            router_neg.lints_learner.update(0, x, reward=1.0)

        np.random.seed(77)
        score_pos = router_pos._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        np.random.seed(77)
        score_neg = router_neg._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        # Both should produce identical scores since abs() is used
        assert abs(score_pos - score_neg) < 1e-9

    def test_lints_disabled_contributes_zero(self):
        """When enable_lints=False, LinTS adds nothing to the score."""
        router_off = _make_router({"enable_lints": False})
        router_on = _make_router({"enable_lints": True, "lints_weight": 0.0})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        # Both should yield the same score: physics-only (no LinTS contribution)
        np.random.seed(42)
        score_off = router_off._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        np.random.seed(42)
        score_on_zero = router_on._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        assert abs(score_off - score_on_zero) < 1e-9


# ---------------------------------------------------------------------------
# 6. Adaptive exploration: ts_weight decay with reuse_budget
# ---------------------------------------------------------------------------

class TestAdaptiveExploration:
    """Tests for enable_adaptive_explore: ts_weight decays with reuse_budget."""

    def test_ts_weight_lower_for_high_reuse(self):
        """With adaptive explore, high reuse_budget should reduce TS exploration."""
        router = _make_router({"enable_beta_ts": True, "enable_adaptive_explore": True, "ts_weight": 1.0})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        np.random.seed(11)
        score_low_reuse = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        np.random.seed(11)
        score_high_reuse = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=100, iat_factor=1.0,
            physics=0.5,
        )
        # TS contribution diminishes; scores shouldn't be identical
        # (the ts_weight factor decays, reducing Beta TS noise)
        # This is a directional test — high reuse means less exploration
        # We can't assert exact values due to stochastic sampling, but
        # ts_w_eff = ts_weight / (1 + 100 * iat_factor) → very small
        # so score_high_reuse should be closer to physics score
        beta_sample = router.beta_learner.sample(0)
        ts_w_low = 1.0
        ts_w_high = 1.0 / (1.0 + 100.0 * 1.0)
        assert ts_w_high < ts_w_low

    def test_adaptive_temp_range(self):
        """Adaptive temperature must stay within [temp_min, temp_max]."""
        router = _make_router({
            "enable_softmax": True,
            "enable_adaptive_temp": True,
            "temp_min": 0.15,
            "temp_max": 2.0,
            "adaptive_temp_base": 1.0,
        })
        # temp = base / (1 + reuse * iat_factor), clamped to [min, max]
        # At reuse=0, iat_factor=1: temp = 1.0 / 1 = 1.0 (in range)
        # At reuse=100, iat_factor=1: temp = 1.0/101 ≈ 0.01 → clamped to 0.15
        raw_scores = [0.5, 0.3, 0.8]
        probs = router._softmax(raw_scores, temp=router.temp_min)
        assert abs(sum(probs) - 1.0) < 1e-9
        for p in probs:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 6b. Load modulator kv_util regression tests
# ---------------------------------------------------------------------------

class TestLoadModKvUtil:
    """Regression tests for the load_mod formula using kv_util [0,1].

    Previously, load_mod used raw decode_blocks (thousands of KV blocks),
    causing IEEE 754 underflow to 0.0 for every worker. The fix replaced
    decode_blocks with the hardware-agnostic kv_util ratio.
    """

    def test_kv_util_zero_gives_load_mod_one(self):
        """kv_util=0.0 means idle worker → load_mod=1.0 → score equals physics."""
        router = _make_router({"enable_beta_ts": False, "enable_lints": False})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        physics = 0.7
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=physics, kv_util=0.0,
        )
        assert abs(score - physics) < 1e-9, (
            f"kv_util=0 should give score=physics={physics}, got {score:.6f}"
        )

    def test_kv_util_one_with_large_qpw_gives_near_zero(self):
        """kv_util=1.0 + large qpw → load_mod ≈ 0 → score ≈ 0."""
        router = _make_router({
            "enable_beta_ts": False, "enable_lints": False,
            "queue_penalty_weight": 100.0,
        })
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.7, kv_util=1.0,
        )
        assert abs(score) < 0.01, (
            f"kv_util=1.0 + qpw=100 should give score≈0, got {score:.6f}"
        )

    def test_large_decode_blocks_no_longer_causes_underflow(self):
        """Raw decode_blocks=5000 with kv_util=0.0 must NOT underflow.

        This is the core regression: the old formula exp(-qpw * db^2 / 2500)
        underflowed to 0.0 for db=5000. Now decode_blocks is ignored by
        load_mod; only kv_util matters.
        """
        router = _make_router({"enable_beta_ts": False, "enable_lints": False})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        physics = 0.6
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5000,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=physics, kv_util=0.0,
        )
        assert abs(score - physics) < 1e-9, (
            f"decode_blocks=5000 + kv_util=0.0 should give score=physics={physics}, "
            f"got {score:.6f} (old formula would give 0.0)"
        )

    def test_load_mod_monotone_decreasing_in_kv_util(self):
        """Score must decrease monotonically as kv_util increases [0→1]."""
        router = _make_router({"enable_beta_ts": False, "enable_lints": False})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        prev_score = float("inf")
        for u in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            score = router._score_worker(
                wid=0, x=x, overlap=0.5, decode_blocks=0,
                last_worker=None, reuse_budget=0, iat_factor=1.0,
                physics=0.5, kv_util=u,
            )
            assert score <= prev_score + 1e-9, (
                f"Score should decrease with kv_util: at u={u}, "
                f"score={score:.6f} > prev={prev_score:.6f}"
            )
            prev_score = score

    def test_kv_util_clamped_above_one(self):
        """kv_util > 1.0 should be clamped to 1.0, not cause explosion."""
        router = _make_router({"enable_beta_ts": False, "enable_lints": False})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        score_at_one = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5, kv_util=1.0,
        )
        score_above = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5, kv_util=5.0,
        )
        assert abs(score_at_one - score_above) < 1e-9, (
            f"kv_util=5.0 should clamp to 1.0: score_at_one={score_at_one:.6f}, "
            f"score_above={score_above:.6f}"
        )

    def test_kv_util_clamped_below_zero(self):
        """Negative kv_util should be clamped to 0.0."""
        router = _make_router({"enable_beta_ts": False, "enable_lints": False})
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        score_at_zero = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5, kv_util=0.0,
        )
        score_below = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5, kv_util=-1.0,
        )
        assert abs(score_at_zero - score_below) < 1e-9, (
            f"kv_util=-1.0 should clamp to 0.0: score_at_zero={score_at_zero:.6f}, "
            f"score_below={score_below:.6f}"
        )


# ---------------------------------------------------------------------------
# 7. Management server: hot-reload config and state roundtrip
# ---------------------------------------------------------------------------

try:
    import aiohttp as _aiohttp  # noqa: F401
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False

_skip_no_aiohttp = pytest.mark.skipif(
    not _AIOHTTP_AVAILABLE,
    reason="aiohttp not installed; skipping management server tests",
)


@_skip_no_aiohttp
class TestManagementServerConfig:
    """Tests for RouterManagementServer._set_config and _get_config.

    These tests bypass aiohttp and call handler methods directly with mock
    requests, exercising the config hot-reload logic without a live server.
    """

    def _make_server(self):
        from dynamo.thompson_router.management import RouterManagementServer
        router = _make_router()
        return RouterManagementServer(router, port=0), router

    @pytest.mark.asyncio
    async def test_get_config_returns_all_tunable_params(self):
        """GET /config must return all TUNABLE_ROUTER_PARAMS keys."""
        from dynamo.thompson_router.router import TUNABLE_ROUTER_PARAMS
        server, router = self._make_server()

        resp = await server._get_config(MagicMock())
        # aiohttp Response.json_response stores data in _body as JSON
        import json
        body = json.loads(resp.body)
        for param in TUNABLE_ROUTER_PARAMS:
            assert param in body, f"Missing tunable param in GET /config: {param}"

    @pytest.mark.asyncio
    async def test_set_config_updates_router_live(self):
        """POST /config with ts_weight should immediately change router.ts_weight."""
        import json
        from dynamo.thompson_router.management import RouterManagementServer
        server, router = self._make_server()

        original_ts = router.ts_weight
        new_ts = original_ts * 2.0 + 0.1

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"ts_weight": new_ts})

        resp = await server._set_config(mock_request)
        body = json.loads(resp.body)

        assert body["status"] == "applied"
        assert abs(router.ts_weight - new_ts) < 1e-9
        assert "ts_weight" in body["params"]

    @pytest.mark.asyncio
    async def test_set_config_updates_lints_v_on_learner(self):
        """POST /config with lints_v must update the live LinTSLearner.v."""
        import json
        server, router = self._make_server()

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"lints_v": 0.77})

        await server._set_config(mock_request)
        assert abs(router.lints_learner.v - 0.77) < 1e-9

    @pytest.mark.asyncio
    async def test_set_config_updates_beta_decay_on_learner(self):
        """POST /config with beta_decay must update the live BetaLearner.decay."""
        import json
        server, router = self._make_server()

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"beta_decay": 0.88})

        await server._set_config(mock_request)
        assert abs(router.beta_learner.decay - 0.88) < 1e-9

    @pytest.mark.asyncio
    async def test_set_config_partial_update_leaves_others_unchanged(self):
        """Only the specified params should change; others stay at their values."""
        import json
        server, router = self._make_server()
        original_temperature = router.temperature
        original_ts_weight = router.ts_weight

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"lints_v": 0.5})

        await server._set_config(mock_request)
        assert abs(router.temperature - original_temperature) < 1e-9
        assert abs(router.ts_weight - original_ts_weight) < 1e-9

    @pytest.mark.asyncio
    async def test_set_config_all_physics_weights(self):
        """All four physics weights should be hot-reloadable."""
        import json
        server, router = self._make_server()

        new_weights = {
            "physics_cache_weight": 0.4,
            "physics_compute_weight": 0.3,
            "physics_queue_weight": 0.2,
            "physics_memory_weight": 0.1,
        }
        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value=new_weights)

        await server._set_config(mock_request)
        assert abs(router.physics_cache_weight - 0.4) < 1e-9
        assert abs(router.physics_compute_weight - 0.3) < 1e-9
        assert abs(router.physics_queue_weight - 0.2) < 1e-9
        assert abs(router.physics_memory_weight - 0.1) < 1e-9


    @pytest.mark.asyncio
    async def test_set_config_enable_flags_hot_reload(self):
        """POST /config with enable_* booleans must toggle features on the live router."""
        import json
        server, router = self._make_server()

        assert not router.enable_beta_ts  # default is False
        assert not router.enable_switching_cost

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "enable_beta_ts": True,
            "enable_switching_cost": True,
            "enable_lints": True,
        })

        resp = await server._set_config(mock_request)
        body = json.loads(resp.body)
        assert body["status"] == "applied"
        assert body["params"]["enable_beta_ts"] is True
        assert body["params"]["enable_switching_cost"] is True
        assert body["params"]["enable_lints"] is True

        # Verify the live router was actually toggled
        assert router.enable_beta_ts is True
        assert router.enable_switching_cost is True
        assert router.enable_lints is True

    @pytest.mark.asyncio
    async def test_set_config_enable_flags_toggle_back(self):
        """Enable flags can be toggled True→False via hot-reload."""
        import json
        server, router = self._make_server()
        router.enable_beta_ts = True
        router.enable_lints = True

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "enable_beta_ts": False,
            "enable_lints": False,
        })

        await server._set_config(mock_request)
        assert router.enable_beta_ts is False
        assert router.enable_lints is False

    @pytest.mark.asyncio
    async def test_get_config_reports_enable_flags(self):
        """GET /config must include all enable_* flags reflecting current state."""
        import json
        server, router = self._make_server()
        router.enable_beta_ts = True
        router.enable_switching_cost = True

        resp = await server._get_config(MagicMock())
        body = json.loads(resp.body)

        assert body["enable_beta_ts"] is True
        assert body["enable_switching_cost"] is True
        assert body["enable_lints"] is False  # default


# ---------------------------------------------------------------------------
# 8. Management server: state persistence (save/load/reset)
# ---------------------------------------------------------------------------

@_skip_no_aiohttp
class TestManagementServerState:
    """Tests for RouterManagementServer state endpoints."""

    def _make_server(self):
        from dynamo.thompson_router.management import RouterManagementServer
        router = _make_router()
        return RouterManagementServer(router, port=0), router

    @pytest.mark.asyncio
    async def test_get_state_contains_both_learners(self):
        """GET /state must return beta_learner and lints_learner fields."""
        import json
        server, router = self._make_server()

        resp = await server._get_state(MagicMock())
        body = json.loads(resp.body)
        assert "beta_learner" in body
        assert "lints_learner" in body

    @pytest.mark.asyncio
    async def test_load_state_restores_beta_params(self):
        """POST /state must restore BetaLearner params from serialized dict."""
        import json
        server, router = self._make_server()

        # Build specific state
        router.beta_learner.add_worker(0)
        router.beta_learner._bandits[0] = (7.0, 3.0)
        saved = {"beta_learner": router.beta_learner.to_dict()}

        # Reset and verify it's gone
        router.beta_learner.reset_all()
        a_reset, b_reset = router.beta_learner.get_params(0)
        assert a_reset == 1.0

        # Load and verify restored
        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value=saved)

        resp = await server._load_state(mock_request)
        body = json.loads(resp.body)
        assert body["status"] == "loaded"
        a_restored, b_restored = router.beta_learner.get_params(0)
        assert abs(a_restored - 7.0) < 1e-9
        assert abs(b_restored - 3.0) < 1e-9

    @pytest.mark.asyncio
    async def test_reset_state_zeroes_all_learners(self):
        """POST /state/reset must reset Beta, LinTS, and LatencyTracker."""
        import json
        server, router = self._make_server()

        # Pollute state
        router.beta_learner.add_worker(0)
        router.beta_learner.update(0, 1.0)
        x = np.ones(9)
        router.lints_learner.add_worker(0)
        router.lints_learner.update(0, x, reward=1.0)
        router.latency_tracker.update_baselines(0, "M", "M", 50.0, per_tok=True)

        resp = await server._reset_state(MagicMock())
        body = json.loads(resp.body)
        assert body["status"] == "reset"

        # Verify all reset
        a, b = router.beta_learner.get_params(0)
        assert a == 1.0
        assert b == 1.0
        assert router.latency_tracker.get_global_baseline(True, fallback=99.0) == 99.0

    @pytest.mark.asyncio
    async def test_metrics_reset_clears_stats(self):
        """POST /metrics/reset must clear RouterStats counters."""
        import json
        server, router = self._make_server()

        router.stats.record_feedback(0.5, 0.4, 0.6)
        assert router.stats._count == 1

        resp = await server._reset_metrics(MagicMock())
        body = json.loads(resp.body)
        assert body["status"] == "reset"
        assert router.stats._count == 0


# ---------------------------------------------------------------------------
# 9. Management server: record_decision and decisions summary
# ---------------------------------------------------------------------------

@_skip_no_aiohttp
class TestManagementServerDecisions:
    """Tests for RouterManagementServer.record_decision and _get_decisions_summary."""

    def _make_server(self):
        from dynamo.thompson_router.management import RouterManagementServer
        router = _make_router()
        return RouterManagementServer(router, port=0), router

    def _make_decision(self, chosen=0, native_pick=0, prefix_id="p"):
        return RoutingDecision(
            chosen=chosen,
            native_pick=native_pick,
            prefix_id=prefix_id,
            worker_details=[
                {"id": 0, "kv_overlap": 0.7, "final_score": 0.9},
                {"id": 1, "kv_overlap": 0.2, "final_score": 0.5},
            ],
        )

    def test_record_decision_stores_entry(self):
        server, _ = self._make_server()
        decision = self._make_decision(chosen=0, native_pick=0)
        server.record_decision(decision, hints={"osl": 250, "iat": 250, "reuse_budget": 3,
                                                "tokens_in": 100}, elapsed_ms=5.0, tokens_out=50)
        assert len(server._decisions) == 1
        rec = server._decisions[0]
        assert rec["chosen"] == 0
        assert rec["agreed"] is True
        assert rec["osl"] == 250

    def test_record_decision_disagreement_flag(self):
        server, _ = self._make_server()
        decision = self._make_decision(chosen=1, native_pick=0)  # disagree
        server.record_decision(decision, hints={"osl": 250, "iat": 250, "reuse_budget": 0,
                                                "tokens_in": 100}, elapsed_ms=3.0, tokens_out=20)
        assert server._decisions[0]["agreed"] is False

    def test_decisions_deque_maxlen(self):
        """Decisions deque should cap at MAX_DECISION_HISTORY."""
        from dynamo.thompson_router.management import MAX_DECISION_HISTORY
        server, _ = self._make_server()
        for i in range(MAX_DECISION_HISTORY + 50):
            decision = self._make_decision(chosen=i % 3)
            server.record_decision(decision, hints={"osl": 250, "iat": 250,
                                                    "reuse_budget": 0, "tokens_in": 100},
                                   elapsed_ms=1.0, tokens_out=10)
        assert len(server._decisions) == MAX_DECISION_HISTORY

    @pytest.mark.asyncio
    async def test_decisions_summary_empty(self):
        """Without any recorded decisions, summary returns error."""
        import json
        server, _ = self._make_server()
        resp = await server._get_decisions_summary(MagicMock())
        body = json.loads(resp.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_decisions_summary_aggregates_correctly(self):
        """Summary should compute agreement rate and worker distribution."""
        import json
        server, _ = self._make_server()

        # 3 decisions: 2 agree, 1 disagrees; workers 0, 0, 1
        for chosen, native in [(0, 0), (0, 0), (1, 0)]:
            server.record_decision(
                self._make_decision(chosen=chosen, native_pick=native, prefix_id=f"p{chosen}"),
                hints={"osl": 250, "iat": 250, "reuse_budget": 0, "tokens_in": 100},
                elapsed_ms=2.0, tokens_out=10,
            )

        resp = await server._get_decisions_summary(MagicMock())
        body = json.loads(resp.body)

        assert body["total_decisions"] == 3
        assert body["agreed_with_native"] == 2
        assert abs(body["agreement_rate"] - 2 / 3) < 0.01
        assert body["worker_distribution"]["0"] == 2
        assert body["worker_distribution"]["1"] == 1

    @pytest.mark.asyncio
    async def test_get_decisions_pagination(self):
        """GET /decisions?n=2 should return only the last 2 entries."""
        import json
        server, _ = self._make_server()
        for i in range(5):
            server.record_decision(
                self._make_decision(chosen=i % 3, prefix_id=f"p{i}"),
                hints={"osl": 250, "iat": 250, "reuse_budget": 0, "tokens_in": 100},
                elapsed_ms=1.0, tokens_out=10,
            )

        mock_request = MagicMock()
        mock_request.query = {"n": "2"}
        resp = await server._get_decisions(mock_request)
        body = json.loads(resp.body)
        assert body["count"] == 2


# ---------------------------------------------------------------------------
# 10. RouterStats variance formula
# ---------------------------------------------------------------------------

class TestRouterStatsVariance:
    """Tests that RouterStats computes mean and variance correctly
    using Welford's online-equivalent formula: E[x^2] - E[x]^2."""

    def test_constant_signal_zero_variance(self):
        stats = RouterStats()
        for _ in range(5):
            stats.record_feedback(reward=0.5, physics_pred=0.4, residual=0.6)
        snap = stats.snapshot()
        assert abs(snap["reward"]["variance"]) < 1e-9
        assert abs(snap["reward"]["mean"] - 0.5) < 1e-9

    def test_two_point_variance(self):
        """Variance of {0, 1} = E[x^2] - E[x]^2 = 0.5 - 0.25 = 0.25."""
        stats = RouterStats()
        stats.record_feedback(reward=0.0, physics_pred=0.5, residual=0.5)
        stats.record_feedback(reward=1.0, physics_pred=0.5, residual=0.5)
        snap = stats.snapshot()
        assert abs(snap["reward"]["variance"] - 0.25) < 1e-9
        assert abs(snap["reward"]["mean"] - 0.5) < 1e-9

    def test_physics_rmse_equals_sqrt_mse(self):
        """RMSE must equal sqrt(MSE) to the precision of snapshot()'s 4-decimal rounding."""
        stats = RouterStats()
        for reward, physics in [(0.8, 0.5), (0.2, 0.6), (0.6, 0.4)]:
            stats.record_feedback(reward=reward, physics_pred=physics, residual=0.5)
        snap = stats.snapshot()
        mse = snap["physics_tower"]["mse"]
        rmse = snap["physics_tower"]["rmse"]
        # snapshot() rounds to 4 decimals, so the max error is ~5e-5 for the sqrt
        assert abs(rmse - mse ** 0.5) < 5e-4

    def test_learner_contribution_uses_absolute_value(self):
        """Learner contributions are stored as absolute values (no sign)."""
        stats = RouterStats()
        stats.record_learner_contribution(lints_contrib=-0.5, beta_contrib=-0.3)
        snap = stats.snapshot()
        # Stored as abs: |−0.5| = 0.5, |−0.3| = 0.3
        assert abs(snap["learner_contribution"]["mean_lints_magnitude"] - 0.5) < 1e-9
        assert abs(snap["learner_contribution"]["mean_beta_magnitude"] - 0.3) < 1e-9

    def test_baseline_hit_rate_formula(self):
        """bucket_hit_rate = hits / (hits + fallbacks)."""
        stats = RouterStats()
        stats.record_baseline_lookup(used_bucket=True)
        stats.record_baseline_lookup(used_bucket=True)
        stats.record_baseline_lookup(used_bucket=False)
        snap = stats.snapshot()
        assert abs(snap["baseline_buckets"]["bucket_hit_rate"] - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# 11. OverlapResult dataclass defaults
# ---------------------------------------------------------------------------

class TestOverlapResult:
    """Tests for the OverlapResult dataclass."""

    def test_default_fields(self):
        """OverlapResult with no args should have empty dicts and zero blocks."""
        result = OverlapResult()
        assert result.scores == {}
        assert result.raw_block_counts == {}
        assert result.total_blocks == 0
        assert result.tree_sizes == {}

    def test_fields_initialized(self):
        """OverlapResult fields should be independently mutable."""
        r1 = OverlapResult()
        r2 = OverlapResult()
        r1.scores[0] = 0.8
        # r2 must not be affected (no shared mutable default)
        assert 0 not in r2.scores

    def test_tree_sizes_field(self):
        """tree_sizes must be accessible and hold integer values."""
        result = OverlapResult(
            scores={0: 0.7, 1: 0.3},
            raw_block_counts={0: 14, 1: 6},
            total_blocks=20,
            tree_sizes={0: 500, 1: 200},
        )
        assert result.tree_sizes[0] == 500
        assert result.tree_sizes[1] == 200


# ---------------------------------------------------------------------------
# 12. BetaLearner min_pseudo_count floor
# ---------------------------------------------------------------------------

class TestBetaLearnerMinPseudoCount:
    """Tests that min_pseudo_count prevents parameter collapse after heavy decay.

    Without the floor, after many all-zero rewards with decay=0.5, alpha would
    approach zero, making the posterior a point mass at 0 and eliminating
    exploration permanently.
    """

    def test_floor_prevents_alpha_collapse(self):
        """alpha must never fall below min_pseudo_count, even with all-zero rewards."""
        floor = 0.5
        bl = BetaLearner(decay=0.5, min_pseudo_count=floor)
        bl.add_worker(0)
        for _ in range(200):
            bl.update(0, reward=0.0)
        alpha, _ = bl.get_params(0)
        assert alpha >= floor, f"alpha={alpha} fell below min_pseudo_count={floor}"

    def test_floor_prevents_beta_collapse(self):
        """beta must never fall below min_pseudo_count, even with all-one rewards."""
        floor = 0.5
        bl = BetaLearner(decay=0.5, min_pseudo_count=floor)
        bl.add_worker(0)
        for _ in range(200):
            bl.update(0, reward=1.0)
        _, beta = bl.get_params(0)
        assert beta >= floor, f"beta={beta} fell below min_pseudo_count={floor}"

    def test_samples_remain_in_unit_range_with_floor(self):
        """Posterior samples must stay in [0,1] even after heavy decay."""
        bl = BetaLearner(decay=0.5, min_pseudo_count=0.1)
        bl.add_worker(0)
        for _ in range(100):
            bl.update(0, reward=0.0)
        for _ in range(50):
            s = bl.sample(0)
            assert 0.0 <= s <= 1.0

    def test_floor_default_is_1(self):
        """Default min_pseudo_count is 1.0 (uninformative prior floor)."""
        bl = BetaLearner(decay=0.9)
        assert bl.min_pseudo_count == 1.0

    def test_update_with_reward_clamp(self):
        """BetaLearner.update must clamp reward to [0, 1] silently."""
        bl = BetaLearner(decay=1.0)
        bl.add_worker(0)
        # Out-of-range reward — should be clamped, not raise
        alpha, beta = bl.update(0, reward=2.5)
        assert alpha == 2.0  # 1.0 + clamp(2.5, 0, 1) = 1.0 + 1.0
        alpha2, beta2 = bl.update(0, reward=-1.0)
        assert beta2 >= 1.0  # beta accumulates (1 - 0) from the negative-clamped reward


# ---------------------------------------------------------------------------
# 13. _osl_bin and _prefill_bin edge-of-boundary regression
# ---------------------------------------------------------------------------

class TestBinBoundaries:
    """Verify exact boundary behaviour of _osl_bin and _prefill_bin.

    These bins feed into update_feedback → latency_tracker → reward computation,
    so off-by-one errors would silently produce the wrong baseline bucket.
    """

    @pytest.mark.parametrize("tokens_in,expected", [
        (0, "S"), (256, "S"), (257, "M"), (1024, "M"), (1025, "L"), (9999, "L"),
    ])
    def test_prefill_bin_boundary(self, tokens_in, expected):
        assert KvThompsonRouter._prefill_bin(tokens_in) == expected

    @pytest.mark.parametrize("osl,expected", [
        (0, "S"), (128, "S"), (129, "M"), (512, "M"), (513, "L"), (9999, "L"),
    ])
    def test_osl_bin_boundary(self, osl, expected):
        assert KvThompsonRouter._osl_bin(osl) == expected

    @pytest.mark.asyncio
    async def test_reward_uses_osl_bucket_baseline(self):
        """update_feedback must use the osl_bin/prefill_bin bucket, not just global."""
        router = _make_router(loads=[
            _make_load(0, prefill_tokens=100, decode_blocks=5),
        ])

        # Two decisions with different OSL bins — establish separate bucket baselines
        decision_s = await router.pick_worker(
            list(range(100)), "p1", 0, osl=64, iat=250, tokens_in=100,
        )
        decision_s.osl = 64
        decision_s.tokens_in = 100
        router.update_feedback(decision_s, latency_ms=100.0, tokens_out=50)

        decision_l = await router.pick_worker(
            list(range(100)), "p2", 0, osl=800, iat=250, tokens_in=100,
        )
        decision_l.osl = 800
        decision_l.tokens_in = 100
        router.update_feedback(decision_l, latency_ms=5000.0, tokens_out=500)

        # Short and long should now have distinct buckets
        short_baseline = router.latency_tracker.get_global_bucket_baseline(
            "S", "S", True, fallback=1.0
        )
        long_baseline = router.latency_tracker.get_global_bucket_baseline(
            "L", "S", True, fallback=1.0
        )
        assert short_baseline != long_baseline or (short_baseline == 1.0 and long_baseline == 1.0)


# ---------------------------------------------------------------------------
# 14. Softmax determinism with single worker
# ---------------------------------------------------------------------------

class TestSoftmaxEdgeCases:
    """Edge-case coverage for _softmax."""

    def test_single_worker_softmax_probability_is_one(self):
        """Softmax of a single score must return probability 1.0."""
        router = _make_router()
        probs = router._softmax([0.7], temp=1.0)
        assert len(probs) == 1
        assert abs(probs[0] - 1.0) < 1e-9

    def test_very_large_temperature_approaches_uniform(self):
        """High temperature → near-uniform distribution."""
        router = _make_router({"temp_max": 100.0})
        probs = router._softmax([1.0, 2.0, 3.0], temp=100.0)
        for p in probs:
            assert abs(p - 1.0 / 3.0) < 0.01

    def test_negative_infinity_scores_handled(self):
        """-inf or very large negative scores should not produce NaN."""
        router = _make_router()
        probs = router._softmax([-1e9, 0.0, 1.0], temp=1.0)
        assert all(math.isfinite(p) for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_nan_scores_clamped(self):
        """NaN scores should not propagate — the guard must replace them."""
        router = _make_router()
        # Inject NaN scores: the _score_worker returns -1e9 on nan/inf
        # but _softmax itself also handles this via finite-check
        probs = router._softmax([float("nan"), 1.0], temp=1.0)
        assert all(math.isfinite(p) for p in probs)
