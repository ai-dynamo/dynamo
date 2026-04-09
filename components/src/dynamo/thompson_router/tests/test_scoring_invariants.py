# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Scoring invariants for the two-term Thompson router model.

    score(w) = lambda_ranking * ranking(w) + lambda_stickiness * stickiness(w)

Coverage:
  1. Ranking monotonicity: overlap ↑ → ranking ↑
  2. Ranking monotonicity: kv_util ↑ → ranking ↓  (via osl_load interaction)
  3. Stickiness is zero for one-shot sessions (reuse_budget = 0)
  4. Stickiness is positive for sticky worker with reuse_budget > 0
  5. Stickiness increases with reuse_budget (saturates via tanh)
  6. Stickiness decreases with high memory_pressure (eviction risk)
  7. Stickiness increases with low IAT (rapid-fire = more urgency)
  8. Lambda scaling: doubling lambda_stickiness doubles stickiness contribution
  9. All valid inputs produce finite scores
 10. Management server: config hot-reload with new param names
 11. Management server: state / reset endpoints
 12. RouterStats variance computation
 13. BetaLearner min_pseudo_count floor prevents parameter collapse
"""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest

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


def _make_router(mock_kv_router, **overrides):
    """Build a KvThompsonRouter with optional kv_thompson config overrides.

    Defaults epsilon=0 for deterministic scoring invariants unless overridden.
    """
    overrides.setdefault("epsilon", 0.0)
    cfg = {"kv_thompson": overrides}
    return KvThompsonRouter(mock_kv_router, config=cfg)


def _score(router, **kwargs):
    """Shortcut: call _score_worker and return (ranking, stickiness)."""
    defaults = dict(
        wid=0, overlap=0.5, kv_util=0.3, prefill_util=0.1,
        memory_pressure=0.0, osl=250, iat=250,
        latency_sensitivity=2.0, is_sticky=False, reuse_budget=0,
    )
    defaults.update(kwargs)
    return router._score_worker(**defaults)


@pytest.fixture
def mock_kv_router():
    router = AsyncMock()
    router.get_potential_loads = AsyncMock(
        return_value=[
            _make_load(0, prefill_tokens=100, decode_blocks=5),
            _make_load(1, prefill_tokens=500, decode_blocks=2),
        ]
    )
    router.best_worker = AsyncMock(return_value=(0, 0, 10))
    return router


@pytest.fixture
def router(mock_kv_router):
    # epsilon=0 for deterministic scoring invariants
    return KvThompsonRouter(mock_kv_router, config={"kv_thompson": {"epsilon": 0.0}})


# ---------------------------------------------------------------------------
# 1. Ranking monotonicity: overlap
# ---------------------------------------------------------------------------

class TestRankingMonotonicity:
    def test_higher_overlap_yields_higher_ranking(self, router):
        """Per model: ranking += w_cache * overlap; overlap↑ → ranking↑."""
        overlaps = [0.0, 0.2, 0.5, 0.8, 1.0]
        rankings = [_score(router, overlap=o)[0] for o in overlaps]
        for a, b in zip(rankings, rankings[1:]):
            assert b > a, f"ranking did not increase: {rankings}"

    def test_higher_kv_util_yields_lower_ranking(self, router):
        """Per model: ranking -= w_osl_load * (osl_norm * kv_util); kv_util↑ → ranking↓."""
        kv_utils = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # Use a large osl to make osl_norm non-trivial
        rankings = [_score(router, kv_util=u, osl=512)[0] for u in kv_utils]
        for a, b in zip(rankings, rankings[1:]):
            assert b < a, f"ranking did not decrease with kv_util: {rankings}"

    def test_higher_prefill_util_yields_lower_ranking(self, router):
        """Per model: ranking += w_queue * (1 - prefill_util); prefill_util↑ → ranking↓."""
        utils = [0.0, 0.3, 0.6, 1.0]
        rankings = [_score(router, prefill_util=u)[0] for u in utils]
        for a, b in zip(rankings, rankings[1:]):
            assert b < a, f"ranking did not decrease with prefill_util: {rankings}"


# ---------------------------------------------------------------------------
# 2. Stickiness: zero for one-shot sessions
# ---------------------------------------------------------------------------

class TestStickinessOneShot:
    def test_stickiness_zero_reuse_budget_zero(self, router):
        """reuse_budget=0 → stickiness=0 regardless of other inputs."""
        cases = [
            dict(overlap=1.0, is_sticky=True, iat=50, memory_pressure=0.0),
            dict(overlap=0.5, is_sticky=False, iat=1000, memory_pressure=0.9),
        ]
        for kw in cases:
            _, s = _score(router, reuse_budget=0, **kw)
            assert s == 0.0, f"expected stickiness=0 for {kw}, got {s}"


# ---------------------------------------------------------------------------
# 3 & 4. Stickiness: positive and monotone in reuse_budget
# ---------------------------------------------------------------------------

class TestStickinessBudget:
    def test_stickiness_positive_with_nonzero_reuse(self, router):
        _, s = _score(router, reuse_budget=1, overlap=0.8, is_sticky=False, iat=250)
        assert s > 0.0

    def test_stickiness_increases_with_reuse_budget(self, router):
        """Linear decay: session_weight = reuse_budget / reuse_total → stickiness
        increases as reuse_budget approaches reuse_total.

        We pass matching reuse_total so the fraction varies in (0, 1].
        """
        reuse_total = 50
        budgets = [1, 5, 10, 20, 40, 50]
        stickiness_vals = [
            router._score_worker(
                wid=0, overlap=0.7, kv_util=0.1, prefill_util=0.0,
                memory_pressure=0.0, osl=250, iat=200,
                latency_sensitivity=2.0, is_sticky=False,
                reuse_budget=b, reuse_total=reuse_total,
            )[1]
            for b in budgets
        ]
        for a, b in zip(stickiness_vals, stickiness_vals[1:]):
            assert b >= a, f"stickiness not non-decreasing: {stickiness_vals}"

    def test_stickiness_is_linear_not_saturating(self, router):
        """session_weight = min(1, reuse_budget / reuse_total) is linear through the origin.

        With the old tanh(alpha * reuse_budget) formula, large budgets converged to
        the same value and even small budgets produced near-full weight.  With the
        new linear formula, session_weight is strictly proportional to the remaining
        fraction, so doubling the remaining budget doubles the stickiness.
        """
        reuse_total = 100
        _, s_quarter = router._score_worker(
            wid=0, overlap=0.7, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250,
            latency_sensitivity=2.0, is_sticky=False,
            reuse_budget=25, reuse_total=reuse_total,
        )
        _, s_half = router._score_worker(
            wid=0, overlap=0.7, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250,
            latency_sensitivity=2.0, is_sticky=False,
            reuse_budget=50, reuse_total=reuse_total,
        )
        _, s_full = router._score_worker(
            wid=0, overlap=0.7, kv_util=0.0, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=250,
            latency_sensitivity=2.0, is_sticky=False,
            reuse_budget=100, reuse_total=reuse_total,
        )
        # Linear through origin: s at fraction f == f * s_full.
        # Allow 2% relative tolerance for floating-point rounding.
        assert s_half == pytest.approx(s_full * 0.5, rel=0.02), (
            f"s_half={s_half:.4f} should be s_full*0.5={s_full*0.5:.4f}"
        )
        assert s_quarter == pytest.approx(s_full * 0.25, rel=0.02), (
            f"s_quarter={s_quarter:.4f} should be s_full*0.25={s_full*0.25:.4f}"
        )
        # Also confirm it did NOT saturate: half should be meaningfully less than full
        assert s_half < s_full * 0.6, (
            f"Expected s_half < 60% of s_full (no saturation), got {s_half:.4f} vs {s_full:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Stickiness decreases with memory_pressure
# ---------------------------------------------------------------------------

class TestStickinessMemoryPressure:
    def test_high_memory_pressure_reduces_stickiness(self, router):
        """future_value = overlap * (1 - memory_pressure); pressure↑ → stickiness↓."""
        pressures = [0.0, 0.3, 0.6, 0.9]
        stickiness_vals = [
            _score(router, memory_pressure=p, overlap=0.8,
                   reuse_budget=5, is_sticky=False, iat=200)[1]
            for p in pressures
        ]
        for a, b in zip(stickiness_vals, stickiness_vals[1:]):
            assert b < a, f"stickiness did not decrease with pressure: {stickiness_vals}"

    def test_full_memory_pressure_kills_non_sticky_stickiness(self, router):
        """At memory_pressure=1.0 and is_sticky=False, future_value=0 → stickiness=0."""
        _, s = _score(
            router, memory_pressure=1.0, overlap=0.9,
            reuse_budget=10, is_sticky=False, iat=50,
        )
        assert s == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Stickiness increases with low IAT
# ---------------------------------------------------------------------------

class TestStickinessIAT:
    def test_low_iat_yields_higher_stickiness_than_high_iat(self, router):
        _, s_low = _score(router, iat=50, reuse_budget=5, overlap=0.7, is_sticky=False)
        _, s_high = _score(router, iat=1000, reuse_budget=5, overlap=0.7, is_sticky=False)
        assert s_low > s_high

    def test_stickiness_monotone_decreasing_with_iat(self, router):
        iats = [50, 100, 200, 400, 700, 1000, 2000]
        vals = [_score(router, iat=i, reuse_budget=5, overlap=0.7, is_sticky=False)[1]
                for i in iats]
        for a, b in zip(vals, vals[1:]):
            assert a >= b, f"stickiness not decreasing with IAT: {vals}"


# ---------------------------------------------------------------------------
# 7. Lambda scaling: doubling lambda_stickiness doubles contribution
# ---------------------------------------------------------------------------

class TestLambdaScaling:
    def test_doubling_lambda_stickiness_doubles_stickiness_term(self, mock_kv_router):
        """score = λ₁*ranking + λ₂*stickiness; doubling λ₂ doubles stickiness part."""
        r1 = _make_router(mock_kv_router, lambda_ranking=1.0, lambda_stickiness=1.0)
        r2 = _make_router(mock_kv_router, lambda_ranking=1.0, lambda_stickiness=2.0)

        kw = dict(
            wid=0, overlap=0.7, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=100,
            latency_sensitivity=2.0, is_sticky=True, reuse_budget=4,
        )
        rank1, stick1 = r1._score_worker(**kw)
        rank2, stick2 = r2._score_worker(**kw)

        # ranking is independent of lambda values
        assert rank1 == pytest.approx(rank2, abs=1e-9)
        # stickiness raw value also independent of lambda
        assert stick1 == pytest.approx(stick2, abs=1e-9)

        score1 = r1.lambda_ranking * rank1 + r1.lambda_stickiness * stick1
        score2 = r2.lambda_ranking * rank2 + r2.lambda_stickiness * stick2
        extra = (r2.lambda_stickiness - r1.lambda_stickiness) * stick1
        assert score2 == pytest.approx(score1 + extra, abs=1e-9)

    def test_zero_lambda_stickiness_means_only_ranking_counts(self, mock_kv_router):
        r = _make_router(mock_kv_router, lambda_ranking=1.0, lambda_stickiness=0.0)
        kw = dict(
            wid=0, overlap=0.7, kv_util=0.2, prefill_util=0.0,
            memory_pressure=0.0, osl=250, iat=50,
            latency_sensitivity=2.0, is_sticky=True, reuse_budget=10,
        )
        ranking, stickiness = r._score_worker(**kw)
        effective_score = r.lambda_ranking * ranking + r.lambda_stickiness * stickiness
        assert effective_score == pytest.approx(ranking, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. All valid inputs → finite scores
# ---------------------------------------------------------------------------

class TestFiniteScores:
    @pytest.mark.parametrize("overlap,kv_util,prefill_util,memory_pressure,osl,iat,lat_sens,is_sticky,reuse", [
        (0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, False, 0),
        (1.0, 1.0, 1.0, 1.0, 1024, 10000, 5.0, True, 100),
        (0.5, 0.5, 0.5, 0.5, 512, 500, 2.0, False, 5),
        (0.0, 1.0, 0.0, 1.0, 0, 1000, 1.0, True, 0),
        (1.0, 0.0, 1.0, 0.0, 1024, 50, 5.0, False, 50),
    ])
    def test_score_is_finite(self, router, overlap, kv_util, prefill_util,
                             memory_pressure, osl, iat, lat_sens, is_sticky, reuse):
        r, s = router._score_worker(
            wid=0, overlap=overlap, kv_util=kv_util, prefill_util=prefill_util,
            memory_pressure=memory_pressure, osl=osl, iat=iat,
            latency_sensitivity=lat_sens, is_sticky=is_sticky, reuse_budget=reuse,
        )
        assert math.isfinite(r), f"ranking not finite: overlap={overlap}"
        assert math.isfinite(s), f"stickiness not finite: overlap={overlap}"


# ---------------------------------------------------------------------------
# Guard: skip management tests if aiohttp is not installed
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


def _make_mock_request(payload: dict) -> MagicMock:
    """Build an AsyncMock aiohttp request whose .json() returns payload."""
    req = AsyncMock()
    req.json = AsyncMock(return_value=payload)
    return req


def _parse_response(resp) -> dict:
    """Decode a web.Response body produced by web.json_response."""
    import json
    return json.loads(resp.body)


def _mgmt_server(mock_kv_router, **router_kwargs):
    """Create a (management_server, router) pair for direct-call testing."""
    from dynamo.thompson_router.management import RouterManagementServer
    r = KvThompsonRouter(mock_kv_router, config=None)
    return RouterManagementServer(r, port=0), r


# ---------------------------------------------------------------------------
# 9. Management server: config hot-reload
# ---------------------------------------------------------------------------

@_skip_no_aiohttp
class TestManagementHotReload:
    @pytest.mark.asyncio
    async def test_hot_reload_updates_all_nine_params(self, mock_kv_router):
        """POST /config applies all 9 tunable params directly to the live router."""
        mgmt, r = _mgmt_server(mock_kv_router)

        new_params = {
            "lambda_ranking": 2.5,
            "lambda_stickiness": 0.8,
            "w_cache": 0.6,
            "w_queue": 0.1,
            "w_osl_load": 0.15,
            "w_sensitivity": 0.15,
            "alpha_reuse": 0.5,
            "sticky_bonus": 0.4,
            "epsilon": 0.07,
        }
        resp = await mgmt._set_config(_make_mock_request(new_params))
        body = _parse_response(resp)

        assert body["status"] == "applied"
        assert r.lambda_ranking == pytest.approx(2.5)
        assert r.lambda_stickiness == pytest.approx(0.8)
        assert r.w_cache == pytest.approx(0.6)
        assert r.w_queue == pytest.approx(0.1)
        assert r.w_osl_load == pytest.approx(0.15)
        assert r.w_sensitivity == pytest.approx(0.15)
        assert r.alpha_reuse == pytest.approx(0.5)
        assert r.sticky_bonus == pytest.approx(0.4)
        assert r.epsilon == pytest.approx(0.07)

    @pytest.mark.asyncio
    async def test_hot_reload_partial_update_leaves_other_params_unchanged(self, mock_kv_router):
        """Sending a subset of params only mutates those params."""
        mgmt, r = _mgmt_server(mock_kv_router)
        original_w_cache = r.w_cache

        resp = await mgmt._set_config(_make_mock_request({"lambda_ranking": 3.0}))
        body = _parse_response(resp)

        assert body["status"] == "applied"
        assert r.lambda_ranking == pytest.approx(3.0)
        assert r.w_cache == pytest.approx(original_w_cache)

    @pytest.mark.asyncio
    async def test_get_config_returns_new_param_names(self, mock_kv_router):
        """GET /config response contains all 9 new param names (no old names)."""
        mgmt, _ = _mgmt_server(mock_kv_router)
        resp = await mgmt._get_config(MagicMock())
        body = _parse_response(resp)

        new_keys = {
            "lambda_ranking", "lambda_stickiness",
            "w_cache", "w_queue", "w_osl_load", "w_sensitivity",
            "alpha_reuse", "sticky_bonus", "epsilon",
        }
        assert new_keys.issubset(set(body.keys()))

        old_keys = {
            "lints_weight", "enable_lints", "enable_softmax", "physics_cache_weight",
            "affinity_base", "switch_base", "idle_boost",
        }
        for k in old_keys:
            assert k not in body, f"removed key still present in /config: {k}"

    @pytest.mark.asyncio
    async def test_beta_decay_hot_reload(self, mock_kv_router):
        """beta_decay should also be hot-reloadable via POST /config."""
        mgmt, r = _mgmt_server(mock_kv_router)
        resp = await mgmt._set_config(_make_mock_request({"beta_decay": 0.88}))
        body = _parse_response(resp)
        assert body["status"] == "applied"
        assert r.beta_learner.decay == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# 10. Management server: state and reset endpoints
# ---------------------------------------------------------------------------

@_skip_no_aiohttp
class TestManagementStateEndpoints:
    @pytest.mark.asyncio
    async def test_state_roundtrip(self, mock_kv_router):
        """GET /state → POST /state restores learner params in-place."""
        mgmt, r = _mgmt_server(mock_kv_router)
        r.beta_learner.add_worker(0)
        r.beta_learner.update(0, 0.9)   # push alpha up
        original_params = r.beta_learner.get_params(0)

        # Capture state
        get_resp = await mgmt._get_state(MagicMock())
        state = _parse_response(get_resp)

        # Reset then restore
        r.beta_learner.reset_all()
        assert r.beta_learner.get_params(0) == (1.0, 1.0)

        load_resp = await mgmt._load_state(_make_mock_request(state))
        load_body = _parse_response(load_resp)
        assert load_body["status"] == "loaded"

        restored = r.beta_learner.get_params(0)
        assert restored[0] == pytest.approx(original_params[0], abs=1e-6)
        assert restored[1] == pytest.approx(original_params[1], abs=1e-6)

    @pytest.mark.asyncio
    async def test_reset_state_clears_learners(self, mock_kv_router):
        """POST /state/reset returns beta_learner to (1, 1) for all workers."""
        mgmt, r = _mgmt_server(mock_kv_router)
        r.beta_learner.add_worker(7)
        r.beta_learner.update(7, 1.0)  # alpha grows above 1.0

        resp = await mgmt._reset_state(MagicMock())
        body = _parse_response(resp)
        assert body["status"] == "reset"
        assert r.beta_learner.get_params(7) == (1.0, 1.0)

    @pytest.mark.asyncio
    async def test_metrics_reset_clears_stats(self, mock_kv_router):
        """POST /metrics/reset zeros rolling stats counters."""
        mgmt, r = _mgmt_server(mock_kv_router)
        r.stats.record_feedback(0.8)
        r.stats.record_feedback(0.5)
        assert r.stats._count == 2

        resp = await mgmt._reset_metrics(MagicMock())
        body = _parse_response(resp)
        assert body["status"] == "reset"
        assert r.stats._count == 0

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_kv_router):
        """GET /health returns status ok and router_type kv_thompson."""
        mgmt, _ = _mgmt_server(mock_kv_router)
        resp = await mgmt._health(MagicMock())
        body = _parse_response(resp)
        assert body["status"] == "ok"
        assert body["router_type"] == "kv_thompson"


# ---------------------------------------------------------------------------
# 11. RouterStats variance computation
# ---------------------------------------------------------------------------

class TestRouterStats:
    def test_reward_variance_zero_for_constant_stream(self):
        stats = RouterStats()
        for _ in range(10):
            stats.record_feedback(0.7)
        snap = stats.snapshot()
        assert snap["reward"]["mean"] == pytest.approx(0.7, abs=1e-4)
        assert snap["reward"]["variance"] == pytest.approx(0.0, abs=1e-6)

    def test_reward_variance_nonzero_for_mixed_stream(self):
        stats = RouterStats()
        for v in [0.0, 1.0] * 10:
            stats.record_feedback(v)
        snap = stats.snapshot()
        assert snap["reward"]["variance"] > 0.0

    def test_decision_scores_recorded(self):
        stats = RouterStats()
        stats.record_decision_scores(ranking=0.5, stickiness=0.2)
        stats.record_decision_scores(ranking=0.7, stickiness=0.4)
        snap = stats.snapshot()
        assert snap["ranking"]["mean"] > 0.0
        assert snap["stickiness"]["mean"] > 0.0

    def test_reset_clears_all_counters(self):
        stats = RouterStats()
        stats.record_feedback(0.9)
        stats.record_decision_scores(0.8, 0.3)
        stats.reset()
        snap = stats.snapshot()
        assert snap["total_observations"] == 0
        assert snap["reward"]["mean"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 12. BetaLearner min_pseudo_count prevents collapse
# ---------------------------------------------------------------------------

class TestBetaLearnerFloor:
    def test_heavy_decay_does_not_collapse_params(self):
        """With aggressive decay, alpha and beta should stay >= min_pseudo_count."""
        learner = BetaLearner(decay=0.5, min_pseudo_count=1.0)
        learner.add_worker(0)
        for _ in range(50):
            learner.update(0, 1.0)  # always success → alpha should dominate
        alpha, beta = learner.get_params(0)
        assert alpha >= 1.0
        assert beta >= 1.0

    def test_alpha_beta_always_positive(self):
        """Posterior params must always be strictly positive regardless of reward history."""
        learner = BetaLearner(decay=0.9, min_pseudo_count=0.5)
        learner.add_worker(0)
        for reward in [0.0] * 100:
            learner.update(0, reward)
        alpha, beta = learner.get_params(0)
        assert alpha > 0.0
        assert beta > 0.0
