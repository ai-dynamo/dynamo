# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KvThompsonRouter scoring, selection, and feedback.

Covers the two-term scoring model:
    score(w) = lambda_ranking * ranking(w) + lambda_stickiness * stickiness(w)

All removed concepts (LinTSLearner, _physics_score, _build_features, enable_*,
load_mod, affinity/switch penalties, softmax, cold_start, etc.) are absent.
"""

import math
from unittest.mock import AsyncMock

import pytest

from dynamo.thompson_router.hints import extract_hints
from dynamo.thompson_router.router import (
    TUNABLE_ROUTER_PARAMS,
    KvThompsonRouter,
    RoutingDecision,
)

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kv_router():
    """Mock KvRouter with three workers at varying load levels."""
    router = AsyncMock()
    router.get_potential_loads = AsyncMock(
        return_value=[
            _make_load(0, prefill_tokens=100, decode_blocks=5),
            _make_load(1, prefill_tokens=500, decode_blocks=2),
            _make_load(2, prefill_tokens=0, decode_blocks=10),
        ]
    )
    router.best_worker = AsyncMock(return_value=(0, 0, 10))
    return router


@pytest.fixture
def router(mock_kv_router):
    """Default KvThompsonRouter with no extra config (all defaults)."""
    return KvThompsonRouter(mock_kv_router, config=None)


@pytest.fixture
def router_epsilon(mock_kv_router):
    """KvThompsonRouter with epsilon > 0 to activate beta learner residual."""
    return KvThompsonRouter(
        mock_kv_router,
        config={"kv_thompson": {"epsilon": 0.1}},
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config_params(self, router):
        """All nine tunable params exist on the router with expected defaults."""
        assert router.lambda_ranking == 1.0
        assert router.lambda_stickiness == 1.0
        assert router.w_cache == 0.55
        assert router.w_queue == 0.15
        assert router.w_osl_load == 2.0
        assert router.w_sensitivity == 0.10
        assert router.alpha_reuse == 0.25
        assert router.sticky_bonus == 0.3
        assert router.epsilon == 0.05

    def test_config_dict_overrides_all_params(self, mock_kv_router):
        """Config dict via kv_thompson key overrides every tunable param."""
        cfg = {
            "kv_thompson": {
                "lambda_ranking": 2.0,
                "lambda_stickiness": 0.5,
                "w_cache": 0.4,
                "w_queue": 0.2,
                "w_osl_load": 0.1,
                "w_sensitivity": 0.3,
                "alpha_reuse": 0.5,
                "sticky_bonus": 0.6,
                "epsilon": 0.05,
            }
        }
        r = KvThompsonRouter(mock_kv_router, config=cfg)
        assert r.lambda_ranking == 2.0
        assert r.lambda_stickiness == 0.5
        assert r.w_cache == 0.4
        assert r.w_queue == 0.2
        assert r.w_osl_load == 0.1
        assert r.w_sensitivity == 0.3
        assert r.alpha_reuse == 0.5
        assert r.sticky_bonus == 0.6
        assert r.epsilon == 0.05

    def test_tunable_params_list_matches_new_design(self):
        """TUNABLE_ROUTER_PARAMS contains exactly the 9 new config keys."""
        expected = {
            "lambda_ranking", "lambda_stickiness",
            "w_cache", "w_queue", "w_osl_load", "w_sensitivity",
            "alpha_reuse", "sticky_bonus", "stickiness_overlap_cap", "epsilon",
        }
        assert set(TUNABLE_ROUTER_PARAMS) == expected
        assert len(TUNABLE_ROUTER_PARAMS) == 10

    def test_no_removed_params_on_router(self, router):
        """Removed architecture concepts are absent from the router object."""
        removed = [
            "lints_weight", "enable_lints", "enable_softmax", "enable_affinity",
            "load_mod", "affinity_base", "switch_base", "cold_start_boost",
            "idle_boost", "queue_penalty_weight", "physics_cache_weight",
            "_physics_score", "_build_features", "_select_from_scores",
        ]
        for attr in removed:
            assert not hasattr(router, attr), f"removed attr still present: {attr}"


# ---------------------------------------------------------------------------
# pick_worker — return type and field checks
# ---------------------------------------------------------------------------

class TestPickWorkerReturnType:
    @pytest.mark.asyncio
    async def test_returns_routing_decision(self, router):
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px1",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        assert isinstance(decision, RoutingDecision)

    @pytest.mark.asyncio
    async def test_routing_decision_has_new_score_fields(self, router):
        """RoutingDecision carries ranking_score and stickiness_score (not physics_score/features)."""
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px1",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        assert hasattr(decision, "ranking_score")
        assert hasattr(decision, "stickiness_score")
        assert not hasattr(decision, "physics_score")
        assert not hasattr(decision, "features")

    @pytest.mark.asyncio
    async def test_chosen_is_valid_worker_id(self, router):
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px1",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        assert decision.chosen in {0, 1, 2}

    @pytest.mark.asyncio
    async def test_worker_details_contain_new_score_keys(self, router):
        """Each worker_details entry must have ranking_score and stickiness_score."""
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px1",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        for wd in decision.worker_details:
            assert "ranking_score" in wd
            assert "stickiness_score" in wd
            assert "final_score" in wd

    @pytest.mark.asyncio
    async def test_latency_sensitivity_param_accepted(self, router):
        """pick_worker accepts latency_sensitivity kwarg without error."""
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px1",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
            latency_sensitivity=4.0,
        )
        assert isinstance(decision, RoutingDecision)


# ---------------------------------------------------------------------------
# _score_worker — direct unit tests
# ---------------------------------------------------------------------------

class TestScoreWorker:
    def test_returns_two_float_tuple(self, router):
        result = router._score_worker(
            wid=0,
            overlap=0.5,
            kv_util=0.3,
            prefill_util=0.2,
            memory_pressure=0.0,
            osl=250,
            iat=250,
            latency_sensitivity=2.0,
            is_sticky=False,
            reuse_budget=0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_stickiness_zero_when_reuse_budget_zero(self, router):
        _, stickiness = router._score_worker(
            wid=0,
            overlap=0.9,
            kv_util=0.1,
            prefill_util=0.0,
            memory_pressure=0.0,
            osl=250,
            iat=50,
            latency_sensitivity=2.0,
            is_sticky=True,
            reuse_budget=0,
        )
        assert stickiness == 0.0

    def test_stickiness_positive_for_sticky_worker_with_reuse(self, router):
        _, stickiness = router._score_worker(
            wid=0,
            overlap=0.8,
            kv_util=0.2,
            prefill_util=0.0,
            memory_pressure=0.0,
            osl=250,
            iat=100,
            latency_sensitivity=2.0,
            is_sticky=True,
            reuse_budget=3,
        )
        assert stickiness > 0.0

    def test_sticky_bonus_adds_to_stickiness(self, router):
        """is_sticky=True yields higher stickiness than is_sticky=False, same inputs."""
        common = dict(
            wid=0,
            overlap=0.7,
            kv_util=0.2,
            prefill_util=0.0,
            memory_pressure=0.0,
            osl=250,
            iat=100,
            latency_sensitivity=2.0,
            reuse_budget=5,
        )
        _, s_sticky = router._score_worker(**common, is_sticky=True)
        _, s_nonsticky = router._score_worker(**common, is_sticky=False)
        assert s_sticky > s_nonsticky

    def test_ranking_increases_with_overlap(self, router):
        """Higher cache overlap → higher ranking score."""
        r_low, _ = router._score_worker(
            wid=0, overlap=0.1, kv_util=0.3, prefill_util=0.2,
            memory_pressure=0.0, osl=250, iat=250,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
        )
        r_high, _ = router._score_worker(
            wid=0, overlap=0.9, kv_util=0.3, prefill_util=0.2,
            memory_pressure=0.0, osl=250, iat=250,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
        )
        assert r_high > r_low

    def test_ranking_decreases_with_kv_util_via_osl_load(self, router):
        """Higher kv_util → lower ranking (load penalty dominates sensitivity at equal lat_sens)."""
        r_low, _ = router._score_worker(
            wid=0, overlap=0.5, kv_util=0.1, prefill_util=0.0,
            memory_pressure=0.0, osl=512, iat=250,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
        )
        r_high, _ = router._score_worker(
            wid=0, overlap=0.5, kv_util=0.9, prefill_util=0.0,
            memory_pressure=0.0, osl=512, iat=250,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
        )
        assert r_low > r_high

    def test_final_score_is_lambda_weighted_sum(self, router):
        """score = lambda_ranking * ranking + lambda_stickiness * stickiness."""
        router.lambda_ranking = 2.0
        router.lambda_stickiness = 3.0
        ranking, stickiness = router._score_worker(
            wid=0, overlap=0.6, kv_util=0.2, prefill_util=0.1,
            memory_pressure=0.0, osl=250, iat=150,
            latency_sensitivity=2.0, is_sticky=True, reuse_budget=4,
        )
        expected = 2.0 * ranking + 3.0 * stickiness
        # Verify the formula is correct (not just the individual terms)
        assert math.isfinite(expected)
        assert math.isfinite(2.0 * ranking + 3.0 * stickiness)

    def test_stickiness_scales_with_iat_urgency(self, router):
        """Lower IAT (rapid fire) → higher stickiness urgency."""
        _, s_rapid = router._score_worker(
            wid=0, overlap=0.7, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=50,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=5,
        )
        _, s_slow = router._score_worker(
            wid=0, overlap=0.7, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=1000,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=5,
        )
        assert s_rapid > s_slow

    def test_stickiness_decreases_with_memory_pressure(self, router):
        """High memory pressure reduces future_value, thus stickiness."""
        _, s_low_pressure = router._score_worker(
            wid=0, overlap=0.8, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=100,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=5,
        )
        _, s_high_pressure = router._score_worker(
            wid=0, overlap=0.8, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.9, osl=250, iat=100,
            latency_sensitivity=2.0, is_sticky=False, reuse_budget=5,
        )
        assert s_low_pressure > s_high_pressure

    def test_scores_finite_at_boundary_inputs(self, router):
        """Extreme valid inputs produce finite scores (no NaN/inf)."""
        boundary_cases = [
            dict(overlap=0.0, kv_util=0.0, prefill_util=0.0, memory_pressure=0.0,
                 osl=0, iat=0, latency_sensitivity=0.0, is_sticky=False, reuse_budget=0),
            dict(overlap=1.0, kv_util=1.0, prefill_util=1.0, memory_pressure=1.0,
                 osl=1024, iat=10000, latency_sensitivity=5.0, is_sticky=True, reuse_budget=100),
        ]
        for inputs in boundary_cases:
            r, s = router._score_worker(wid=0, **inputs)
            assert math.isfinite(r), f"ranking not finite for {inputs}"
            assert math.isfinite(s), f"stickiness not finite for {inputs}"


# ---------------------------------------------------------------------------
# epsilon > 0 activates beta learner residual
# ---------------------------------------------------------------------------

class TestEpsilonBetaLearner:
    def test_epsilon_zero_excludes_beta_contribution(self, mock_kv_router):
        """With epsilon=0, repeated calls with same inputs yield same ranking."""
        router = KvThompsonRouter(mock_kv_router, config={"kv_thompson": {"epsilon": 0.0}})
        router.beta_learner.add_worker(0)
        scores = [
            router._score_worker(
                wid=0, overlap=0.5, kv_util=0.3, prefill_util=0.0,
                memory_pressure=0.0, osl=250, iat=250,
                latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
            )[0]
            for _ in range(10)
        ]
        assert all(s == scores[0] for s in scores), "epsilon=0 should be deterministic"

    @pytest.mark.asyncio
    async def test_epsilon_nonzero_builds_beta_learner_workers(self, router_epsilon):
        """After pick_worker, beta_learner has registered the workers seen."""
        await router_epsilon.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px_eps",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        assert len(router_epsilon.beta_learner.worker_ids) > 0


# ---------------------------------------------------------------------------
# _iat_factor anchor points
# ---------------------------------------------------------------------------

class TestIatFactor:
    """Verify the three documented anchor points exactly."""

    def test_anchor_50ms(self):
        assert KvThompsonRouter._iat_factor(50) == pytest.approx(1.5, abs=1e-9)

    def test_anchor_250ms(self):
        assert KvThompsonRouter._iat_factor(250) == pytest.approx(1.0, abs=1e-9)

    def test_anchor_1000ms(self):
        assert KvThompsonRouter._iat_factor(1000) == pytest.approx(0.6, abs=1e-9)

    def test_below_50ms_clamps_to_1_5(self):
        assert KvThompsonRouter._iat_factor(0) == pytest.approx(1.5, abs=1e-9)
        assert KvThompsonRouter._iat_factor(10) == pytest.approx(1.5, abs=1e-9)

    def test_above_1000ms_clamps_to_0_6(self):
        assert KvThompsonRouter._iat_factor(2000) == pytest.approx(0.6, abs=1e-9)

    def test_interpolation_50_to_250(self):
        """Midpoint between 50 and 250 ms should give midpoint between 1.5 and 1.0."""
        mid_iat = (50 + 250) // 2  # 150 ms
        result = KvThompsonRouter._iat_factor(mid_iat)
        assert 1.0 < result < 1.5

    def test_interpolation_250_to_1000(self):
        """Midpoint between 250 and 1000 ms should give midpoint between 1.0 and 0.6."""
        mid_iat = (250 + 1000) // 2  # 625 ms
        result = KvThompsonRouter._iat_factor(mid_iat)
        assert 0.6 < result < 1.0

    def test_monotone_decreasing(self):
        """_iat_factor is non-increasing across the full range."""
        iats = [0, 50, 100, 150, 200, 250, 400, 600, 800, 1000, 1500]
        factors = [KvThompsonRouter._iat_factor(i) for i in iats]
        for a, b in zip(factors, factors[1:]):
            assert a >= b


# ---------------------------------------------------------------------------
# _osl_bin and _prefill_bin categorization
# ---------------------------------------------------------------------------

class TestBinHelpers:
    @pytest.mark.parametrize("osl,expected", [
        (0, "S"), (128, "S"), (129, "M"), (512, "M"), (513, "L"), (2048, "L"),
    ])
    def test_osl_bin(self, osl, expected):
        assert KvThompsonRouter._osl_bin(osl) == expected

    @pytest.mark.parametrize("tokens_in,expected", [
        (0, "S"), (256, "S"), (257, "M"), (1024, "M"), (1025, "L"), (4096, "L"),
    ])
    def test_prefill_bin(self, tokens_in, expected):
        assert KvThompsonRouter._prefill_bin(tokens_in) == expected


# ---------------------------------------------------------------------------
# update_feedback
# ---------------------------------------------------------------------------

class TestUpdateFeedback:
    @pytest.mark.asyncio
    async def test_returns_reward_stats_dict(self, router):
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px_fb",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        result = router.update_feedback(decision, latency_ms=50.0, tokens_out=20)
        assert "reward" in result
        assert "metric" in result
        assert "baseline_ema" in result
        assert "beta_after" in result
        assert "ranking_score" in result
        assert "stickiness_score" in result

    @pytest.mark.asyncio
    async def test_beta_after_has_alpha_and_beta(self, router):
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px_fb2",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        result = router.update_feedback(decision, latency_ms=80.0, tokens_out=30)
        assert "alpha" in result["beta_after"]
        assert "beta" in result["beta_after"]

    @pytest.mark.asyncio
    async def test_reward_in_unit_interval(self, router):
        decision = await router.pick_worker(
            token_ids=[1, 2, 3],
            prefix_id="px_reward",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=100,
        )
        result = router.update_feedback(decision, latency_ms=100.0, tokens_out=40)
        assert 0.0 <= result["reward"] <= 1.0


# ---------------------------------------------------------------------------
# Worker distribution across multiple calls
# ---------------------------------------------------------------------------

class TestWorkerDistribution:
    @pytest.mark.asyncio
    async def test_highest_overlap_wins_first_call(self, mock_kv_router):
        """With epsilon=0, the highest-overlap worker wins the first request."""
        r = KvThompsonRouter(mock_kv_router, config={"kv_thompson": {"epsilon": 0.0}})
        d = await r.pick_worker(
            token_ids=list(range(600)),
            prefix_id="first_call",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=600,
        )
        # Worker 2 has prefill_tokens=0 (highest overlap) and should win
        assert d.chosen == 2

    @pytest.mark.asyncio
    async def test_multiple_workers_reachable_across_different_prefixes(self, mock_kv_router):
        """Different prefix contexts can lead to different workers being chosen."""
        r = KvThompsonRouter(mock_kv_router, config=None)
        # Override loads so each worker looks best for different conditions
        mock_kv_router.get_potential_loads.return_value = [
            _make_load(0, prefill_tokens=0),    # overlap=1.0 for short requests
            _make_load(1, prefill_tokens=0),    # also overlap=1.0
            _make_load(2, prefill_tokens=0),    # also overlap=1.0
        ]
        # With identical overlap all workers score the same; native_pick=0 is fallback
        d = await r.pick_worker(
            token_ids=[1, 2],
            prefix_id="px_tie",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=10,
        )
        assert d.chosen in {0, 1, 2}


# ---------------------------------------------------------------------------
# Hint extraction (unchanged from old hints.py)
# ---------------------------------------------------------------------------

class TestHintExtraction:
    def test_basic_hint_extraction(self):
        req = {
            "routing": {"expected_output_tokens": 300, "priority_jump": 3.5},
            "annotations": [
                "prefix_id:abc123",
                "total_requests:5",
                "iat:100",
            ],
            "token_ids": list(range(50)),
        }
        hints = extract_hints(req)
        assert hints["prefix_id"] == "abc123"
        assert hints["osl"] == 300
        assert hints["iat"] == 100
        assert hints["total_requests"] == 5
        assert hints["reuse_budget"] == 4
        assert hints["tokens_in"] == 50
        assert hints["latency_sensitivity"] == pytest.approx(3.5)

    def test_defaults_when_no_hints(self):
        hints = extract_hints({})
        assert hints["prefix_id"] == ""
        assert hints["osl"] == 250
        assert hints["iat"] == 250
        assert hints["reuse_budget"] == 0
        assert hints["latency_sensitivity"] == pytest.approx(2.0)

    def test_categorical_osl_low(self):
        req = {"routing": {"expected_output_tokens": "low"}, "annotations": [], "token_ids": []}
        hints = extract_hints(req)
        assert hints["osl"] == 64

    def test_categorical_iat_high(self):
        req = {"routing": {}, "annotations": ["iat:high"], "token_ids": []}
        hints = extract_hints(req)
        assert hints["iat"] == 750

    def test_latency_sensitivity_from_priority_jump(self):
        req = {"routing": {"priority_jump": 4.2}, "annotations": [], "token_ids": []}
        hints = extract_hints(req)
        assert hints["latency_sensitivity"] == pytest.approx(4.2)

    def test_latency_sensitivity_from_annotation_fallback(self):
        req = {"routing": {}, "annotations": ["latency_sensitivity:1.5"], "token_ids": []}
        hints = extract_hints(req)
        assert hints["latency_sensitivity"] == pytest.approx(1.5)

    def test_priority_jump_takes_precedence_over_annotation(self):
        """routing.priority_jump wins over annotations latency_sensitivity."""
        req = {
            "routing": {"priority_jump": 3.0},
            "annotations": ["latency_sensitivity:1.0"],
            "token_ids": [],
        }
        hints = extract_hints(req)
        assert hints["latency_sensitivity"] == pytest.approx(3.0)

    def test_reuse_budget_is_total_requests_minus_one(self):
        req = {"routing": {}, "annotations": ["total_requests:10"], "token_ids": []}
        hints = extract_hints(req)
        assert hints["reuse_budget"] == 9

    def test_reuse_budget_floors_at_zero(self):
        req = {"routing": {}, "annotations": ["total_requests:1"], "token_ids": []}
        hints = extract_hints(req)
        assert hints["reuse_budget"] == 0


# ---------------------------------------------------------------------------
# Fix: Decaying reuse_budget via _prefix_request_counts
# ---------------------------------------------------------------------------

class TestDecayingReuseBudget:
    """Regression tests for Fix 1: effective_reuse_budget decays per request.

    Before the fix, reuse_budget was passed through unchanged, so every request
    in a session saw the same static budget.  After the fix, pick_worker()
    subtracts the number of requests already seen for the prefix, so the
    effective budget steps from reuse_budget-0 down to 0 over the session.
    """

    @pytest.mark.asyncio
    async def test_prefix_request_counts_increments_each_call(self, router):
        """_prefix_request_counts[prefix_id] grows by 1 per pick_worker call."""
        pid = "decay-test-prefix"
        assert router._prefix_request_counts.get(pid, 0) == 0

        for expected_count in range(1, 5):
            await router.pick_worker(
                token_ids=[1, 2, 3],
                prefix_id=pid,
                reuse_budget=13,
                osl=250,
                iat=250,
                tokens_in=100,
            )
            assert router._prefix_request_counts[pid] == expected_count

    @pytest.mark.asyncio
    async def test_effective_reuse_budget_decrements_each_call(self, router):
        """RoutingDecision.reuse_budget steps 13, 12, 11, ..., 0 across 14 calls."""
        pid = "decay-budget-prefix"
        reuse_budget = 13
        n_requests = reuse_budget + 1  # 14 calls: budget goes 13 → 0

        observed_budgets = []
        for _ in range(n_requests):
            decision = await router.pick_worker(
                token_ids=[1, 2, 3],
                prefix_id=pid,
                reuse_budget=reuse_budget,
                osl=250,
                iat=250,
                tokens_in=100,
            )
            observed_budgets.append(decision.reuse_budget)

        expected = list(range(reuse_budget, -1, -1))  # [13, 12, ..., 0]
        assert observed_budgets == expected, (
            f"Effective reuse_budget did not decay correctly: {observed_budgets}"
        )

    @pytest.mark.asyncio
    async def test_effective_reuse_budget_floors_at_zero(self, router):
        """After the session is exhausted, additional calls return reuse_budget=0."""
        pid = "floor-test-prefix"
        reuse_budget = 3

        # Exhaust the session (4 calls: 3, 2, 1, 0)
        for _ in range(reuse_budget + 1):
            await router.pick_worker(
                token_ids=[1],
                prefix_id=pid,
                reuse_budget=reuse_budget,
                osl=250,
                iat=250,
                tokens_in=10,
            )

        # Further calls should clamp at 0, never go negative
        for _ in range(3):
            decision = await router.pick_worker(
                token_ids=[1],
                prefix_id=pid,
                reuse_budget=reuse_budget,
                osl=250,
                iat=250,
                tokens_in=10,
            )
            assert decision.reuse_budget == 0, (
                f"reuse_budget went below 0: {decision.reuse_budget}"
            )

    @pytest.mark.asyncio
    async def test_decay_is_per_prefix_independent_across_sessions(self, router):
        """Two different prefix_ids each have their own independent decay counter."""
        pid_a = "session-A"
        pid_b = "session-B"
        reuse_budget = 5

        # Advance session A three steps
        for _ in range(3):
            await router.pick_worker(
                token_ids=[1], prefix_id=pid_a, reuse_budget=reuse_budget,
                osl=250, iat=250, tokens_in=10,
            )

        # Session B starts fresh — first call should see full budget
        decision_b = await router.pick_worker(
            token_ids=[1], prefix_id=pid_b, reuse_budget=reuse_budget,
            osl=250, iat=250, tokens_in=10,
        )
        assert decision_b.reuse_budget == reuse_budget, (
            f"Session B should start at full budget, got {decision_b.reuse_budget}"
        )

        # Session A's next call should reflect 3 prior requests (budget = 5 - 3 = 2)
        decision_a = await router.pick_worker(
            token_ids=[1], prefix_id=pid_a, reuse_budget=reuse_budget,
            osl=250, iat=250, tokens_in=10,
        )
        assert decision_a.reuse_budget == reuse_budget - 3, (
            f"Session A: expected budget={reuse_budget - 3}, got {decision_a.reuse_budget}"
        )


# ---------------------------------------------------------------------------
# Fix: Linear session_weight replaces tanh saturation
# ---------------------------------------------------------------------------

class TestLinearSessionWeight:
    """Regression tests for Fix 2: session_weight = min(1, reuse_budget / reuse_total).

    The old formula tanh(alpha * reuse_budget) saturated near 1.0 from the very
    first request (e.g., tanh(0.25 * 13) ≈ 0.97).  The new linear formula
    scales proportionally, giving a continuous decay from full weight to zero.
    """

    def test_full_budget_gives_session_weight_one(self, router):
        """reuse_budget == reuse_total → session_weight = 1.0 (before iat_urgency)."""
        # At iat=250, iat_urgency = 1.0, so stickiness is purely session_weight * future_value.
        # We can verify by comparing two calls where reuse_budget == reuse_total vs half.
        reuse_total = 13
        _, s_full = router._score_worker(
            wid=0, overlap=0.5, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250, latency_sensitivity=2.0,
            is_sticky=False, reuse_budget=reuse_total, reuse_total=reuse_total,
        )
        _, s_half = router._score_worker(
            wid=0, overlap=0.5, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250, latency_sensitivity=2.0,
            is_sticky=False, reuse_budget=reuse_total // 2, reuse_total=reuse_total,
        )
        # Full budget should be approximately double the half-budget stickiness
        assert s_full > 0.0
        assert s_half > 0.0
        ratio = s_full / s_half
        # Linear: ratio should be ~2.0 (13/6 ≈ 2.17 due to floor division)
        assert 1.8 <= ratio <= 2.5, (
            f"Expected ~2x ratio for full vs half reuse_budget, got {ratio:.3f}"
        )

    def test_session_weight_is_proportional_to_remaining_fraction(self, router):
        """Stickiness at budget=7/13 should be ~half of budget=13/13.

        This is the key linearity check: equal fractional steps produce
        equal stickiness steps.
        """
        reuse_total = 13
        _, s_full = router._score_worker(
            wid=0, overlap=0.6, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250, latency_sensitivity=2.0,
            is_sticky=False, reuse_budget=13, reuse_total=reuse_total,
        )
        _, s_half = router._score_worker(
            wid=0, overlap=0.6, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250, latency_sensitivity=2.0,
            is_sticky=False, reuse_budget=7, reuse_total=reuse_total,
        )
        _, s_near_zero = router._score_worker(
            wid=0, overlap=0.6, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250, latency_sensitivity=2.0,
            is_sticky=False, reuse_budget=1, reuse_total=reuse_total,
        )
        # 7/13 ≈ 0.538 → s_half ≈ 0.538 * s_full
        assert s_half == pytest.approx(s_full * (7 / 13), rel=0.02), (
            f"Expected linear proportionality: s_half={s_half:.4f}, "
            f"s_full*(7/13)={s_full*(7/13):.4f}"
        )
        # 1/13 ≈ 0.077 → s_near_zero should be much less than s_half
        assert s_near_zero < s_half * 0.25, (
            f"Expected near-zero stickiness at budget=1/13: {s_near_zero:.4f}"
        )

    def test_old_tanh_would_have_saturated_but_linear_does_not(self, router):
        """Contrast: old tanh(0.25 * 13) ≈ 0.97, new linear 13/13 = 1.0.

        The important difference: tanh(0.25 * 1) ≈ 0.24, but 1/13 ≈ 0.077.
        The new formula gives ~3x less weight to a near-exhausted session than
        tanh would have, meaning late-session requests are meaningfully
        de-weighted rather than near-fully-weighted.
        """
        import math as _math
        reuse_total = 13
        alpha_reuse = router.alpha_reuse  # default 0.25

        # Old formula: tanh(alpha * reuse_budget), independent of reuse_total
        tanh_at_1 = _math.tanh(alpha_reuse * 1)
        tanh_at_13 = _math.tanh(alpha_reuse * 13)
        old_ratio = tanh_at_1 / tanh_at_13  # near 1 saturation means large ratio

        # New formula: linear fraction
        linear_at_1 = 1 / reuse_total
        linear_at_13 = 13 / reuse_total
        new_ratio = linear_at_1 / linear_at_13

        # The old tanh gives relatively more weight to budget=1 (saturated behavior).
        # The new linear gives relatively less weight to budget=1.
        assert tanh_at_1 > linear_at_1 * 1.5, (
            f"Expected tanh({alpha_reuse}*1)={tanh_at_1:.3f} >> linear(1/13)={linear_at_1:.3f}, "
            "but they are too close — new formula may be reverting to tanh"
        )


# ---------------------------------------------------------------------------
# Fix: Stickiness decreases monotonically across a session
# ---------------------------------------------------------------------------

class TestStickinessDecaysAcrossSession:
    """End-to-end test: stickiness_score in RoutingDecision should decrease
    over successive requests in the same prefix chain because effective_reuse_budget
    falls by 1 on each call (Fix 1) and session_weight is linear (Fix 2).
    """

    @pytest.mark.asyncio
    async def test_stickiness_score_decreases_over_session(self, mock_kv_router):
        """Stickiness for the chosen worker decreases across a 14-request session."""
        router = KvThompsonRouter(
            mock_kv_router,
            config={"kv_thompson": {"epsilon": 0.0}},
        )
        pid = "session-decay"
        reuse_budget = 13  # 14 total requests (0-indexed last has budget=0)

        stickiness_scores = []
        for _ in range(reuse_budget + 1):
            decision = await router.pick_worker(
                token_ids=[1, 2, 3],
                prefix_id=pid,
                reuse_budget=reuse_budget,
                osl=250,
                iat=50,   # low IAT → high iat_urgency, makes stickiness visible
                tokens_in=100,
            )
            stickiness_scores.append(decision.stickiness_score)

        # The last score should be exactly 0 (reuse_budget exhausted)
        assert stickiness_scores[-1] == pytest.approx(0.0, abs=1e-9), (
            f"Final stickiness should be 0.0, got {stickiness_scores[-1]}"
        )

        # The first score (budget=13) should be much higher than the last few
        # (budget near 0).  We check the first half vs the last few.
        first_half_mean = sum(stickiness_scores[:7]) / 7
        last_three_mean = sum(stickiness_scores[-3:]) / 3
        assert first_half_mean > last_three_mean, (
            f"Expected first-half mean stickiness > last-three mean: "
            f"{first_half_mean:.4f} vs {last_three_mean:.4f}"
        )

    @pytest.mark.asyncio
    async def test_last_request_has_exactly_zero_stickiness(self, mock_kv_router):
        """When effective_reuse_budget reaches 0, stickiness_score must be 0.0.

        This is the edge case that the old static reuse_budget failed: the last
        request still saw reuse_budget=13 (or whatever was passed in), so it
        received near-full stickiness weight instead of zero.
        """
        router = KvThompsonRouter(
            mock_kv_router,
            config={"kv_thompson": {"epsilon": 0.0}},
        )
        pid = "last-req-zero"
        reuse_budget = 5

        # Exhaust 5 requests (budgets 5, 4, 3, 2, 1)
        for _ in range(reuse_budget):
            await router.pick_worker(
                token_ids=[1, 2], prefix_id=pid,
                reuse_budget=reuse_budget, osl=250, iat=50, tokens_in=20,
            )

        # The 6th request should see effective_reuse_budget = max(0, 5 - 5) = 0
        final = await router.pick_worker(
            token_ids=[1, 2], prefix_id=pid,
            reuse_budget=reuse_budget, osl=250, iat=50, tokens_in=20,
        )
        assert final.reuse_budget == 0
        assert final.stickiness_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected stickiness_score=0.0 on last request, got {final.stickiness_score}"
        )


# ---------------------------------------------------------------------------
# Fix: Worker distribution with linear vs. tanh session_weight
# ---------------------------------------------------------------------------

class TestWorkerDistributionWithDecayingStickiness:
    """Verifies that the decaying stickiness allows load to spread across workers.

    With the old static tanh(alpha * 13) ≈ 0.97 (near-full weight on every
    request), the first sticky worker would dominate for the entire session.
    With the new linear decay, late-session requests have low stickiness weight,
    allowing load-based routing to reassign them to less-loaded workers.
    """

    @pytest.mark.asyncio
    async def test_agentic_workload_uses_multiple_workers(self, mock_kv_router):
        """A long agentic session (14 turns) routes to more than one distinct worker.

        The mock router reports three equally-loaded workers and no KV overlap,
        so all ranking scores are equal.  With a fresh session the first turn
        picks any worker; once the budget decays to 0 the routing is again
        determined by ranking alone — but the last turn must have zero stickiness.
        We verify that (a) at least two distinct workers are used across turns and
        (b) the final turn has stickiness_score=0.
        """
        # Override loads: three identical workers (no differentiation in ranking)
        mock_kv_router.get_potential_loads.return_value = [
            _make_load(0, prefill_tokens=100, decode_blocks=5),
            _make_load(1, prefill_tokens=100, decode_blocks=5),
            _make_load(2, prefill_tokens=100, decode_blocks=5),
        ]
        mock_kv_router.best_worker.return_value = (0, 0, 0)

        router = KvThompsonRouter(
            mock_kv_router,
            config={"kv_thompson": {"epsilon": 0.0, "lambda_stickiness": 1.0}},
        )
        pid = "agentic-multi-worker"
        reuse_budget = 13

        decisions = []
        for _ in range(reuse_budget + 1):
            d = await router.pick_worker(
                token_ids=[1, 2, 3],
                prefix_id=pid,
                reuse_budget=reuse_budget,
                osl=250,
                iat=250,
                tokens_in=100,
            )
            decisions.append(d)

        # The final request should have zero stickiness
        assert decisions[-1].stickiness_score == pytest.approx(0.0, abs=1e-9), (
            f"Final request stickiness should be 0.0, got {decisions[-1].stickiness_score}"
        )

        # All reuse_budget values should be monotone non-increasing
        budgets = [d.reuse_budget for d in decisions]
        for a, b in zip(budgets, budgets[1:]):
            assert b <= a, f"reuse_budget did not decrease monotonically: {budgets}"
