# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Thompson router learner components."""

import time

import numpy as np
import pytest

from dynamo.thompson_router.learners import (
    BetaLearner,
    LatencyTracker,
    LinTSLearner,
    PendingDecisions,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.router,
]


# ---------------------------------------------------------------------------
# BetaLearner
# ---------------------------------------------------------------------------
class TestBetaLearner:
    def test_default_prior(self):
        bl = BetaLearner()
        alpha, beta = bl.get_params(999)
        assert alpha == 1.0
        assert beta == 1.0

    def test_add_worker_idempotent(self):
        bl = BetaLearner()
        bl.add_worker(1, alpha=5.0, beta=3.0)
        bl.add_worker(1, alpha=99.0, beta=99.0)
        alpha, beta = bl.get_params(1)
        assert alpha == 5.0
        assert beta == 3.0

    def test_sample_in_range(self):
        bl = BetaLearner()
        bl.add_worker(0)
        for _ in range(100):
            s = bl.sample(0)
            assert 0.0 <= s <= 1.0

    def test_update_shifts_posterior(self):
        bl = BetaLearner(decay=1.0)
        bl.add_worker(0)
        for _ in range(50):
            bl.update(0, reward=1.0)
        assert bl.mean(0) > 0.8

        bl2 = BetaLearner(decay=1.0)
        bl2.add_worker(0)
        for _ in range(50):
            bl2.update(0, reward=0.0)
        assert bl2.mean(0) < 0.2

    def test_decay_forgets(self):
        bl = BetaLearner(decay=0.9)
        bl.add_worker(0)
        for _ in range(100):
            bl.update(0, reward=1.0)
        alpha_high, _ = bl.get_params(0)

        for _ in range(100):
            bl.update(0, reward=0.0)
        alpha_after, _ = bl.get_params(0)
        assert alpha_after < alpha_high

    def test_effective_window(self):
        bl = BetaLearner(decay=0.99)
        assert abs(bl.effective_window - 100.0) < 1.0

    def test_half_life(self):
        bl = BetaLearner(decay=0.995)
        assert 130.0 < bl.half_life < 145.0

    def test_no_decay_infinite_window(self):
        bl = BetaLearner(decay=1.0)
        assert bl.effective_window == float("inf")
        assert bl.half_life == float("inf")

    def test_remove_worker(self):
        bl = BetaLearner()
        bl.add_worker(1)
        bl.remove_worker(1)
        assert 1 not in bl.worker_ids

    def test_reset_all(self):
        bl = BetaLearner(decay=1.0)
        bl.add_worker(0)
        bl.add_worker(1)
        bl.update(0, 1.0)
        bl.update(1, 0.0)
        bl.reset_all()
        for wid in [0, 1]:
            a, b = bl.get_params(wid)
            assert a == 1.0
            assert b == 1.0

    def test_serialization_roundtrip(self):
        bl = BetaLearner(decay=0.99)
        bl.add_worker(0)
        bl.add_worker(1)
        bl.update(0, 0.8)
        bl.update(1, 0.3)

        state = bl.to_dict()
        bl2 = BetaLearner.from_dict(state)

        for wid in [0, 1]:
            a1, b1 = bl.get_params(wid)
            a2, b2 = bl2.get_params(wid)
            assert abs(a1 - a2) < 1e-10
            assert abs(b1 - b2) < 1e-10

    def test_load_state_in_place(self):
        bl = BetaLearner()
        bl.add_worker(0)
        bl.update(0, 1.0)
        state = bl.to_dict()

        bl2 = BetaLearner()
        bl2.load_state(state)
        a1, b1 = bl.get_params(0)
        a2, b2 = bl2.get_params(0)
        assert abs(a1 - a2) < 1e-10

    def test_effective_sample_size(self):
        bl = BetaLearner(decay=1.0, min_pseudo_count=1.0)
        bl.add_worker(0)
        assert bl.effective_sample_size(0) == 0.0
        bl.update(0, 0.5)
        assert bl.effective_sample_size(0) == 1.0


# ---------------------------------------------------------------------------
# LinTSLearner
# ---------------------------------------------------------------------------
class TestLinTSLearner:
    def test_default_init(self):
        lt = LinTSLearner(feature_dim=7)
        lt.add_worker(0)
        A, b = lt.get_params(0)
        assert A.shape == (7, 7)
        assert b.shape == (7,)
        np.testing.assert_allclose(b, 0.0)

    def test_add_worker_idempotent(self):
        lt = LinTSLearner(feature_dim=3)
        lt.add_worker(0)
        lt.update(0, np.array([1.0, 0.0, 0.0]), 1.0)
        A_before, _ = lt.get_params(0)
        lt.add_worker(0)
        A_after, _ = lt.get_params(0)
        np.testing.assert_array_equal(A_before, A_after)

    def test_sample_returns_scalar(self):
        lt = LinTSLearner(feature_dim=4)
        lt.add_worker(0)
        x = np.array([1.0, 0.5, 0.3, 0.1])
        s = lt.sample(0, x)
        assert isinstance(s, float)

    def test_update_changes_state(self):
        lt = LinTSLearner(feature_dim=3, forget_rate=0.999)
        lt.add_worker(0)
        A_before, b_before = lt.get_params(0)

        x = np.array([1.0, 0.5, 0.2])
        lt.update(0, x, reward=0.9)

        A_after, b_after = lt.get_params(0)
        assert not np.allclose(A_before, A_after)
        assert not np.allclose(b_before, b_after)

    def test_posterior_mean_shifts_with_data(self):
        lt = LinTSLearner(feature_dim=2, lambda_=1.0, forget_rate=0.999)
        lt.add_worker(0)

        x = np.array([1.0, 0.0])
        for _ in range(50):
            lt.update(0, x, reward=1.0)

        mean = lt.posterior_mean(0)
        assert mean[0] > 0.3

    def test_remove_worker(self):
        lt = LinTSLearner(feature_dim=3)
        lt.add_worker(0)
        lt.remove_worker(0)
        assert 0 not in lt.worker_ids

    def test_reset_all(self):
        lt = LinTSLearner(feature_dim=3, lambda_=2.0)
        lt.add_worker(0)
        lt.update(0, np.array([1.0, 0.0, 0.0]), 1.0)
        lt.reset_all()
        A, b = lt.get_params(0)
        np.testing.assert_allclose(A, 2.0 * np.eye(3))
        np.testing.assert_allclose(b, 0.0)

    def test_serialization_roundtrip(self):
        lt = LinTSLearner(feature_dim=3, v=0.5, forget_rate=0.99)
        lt.add_worker(0)
        lt.update(0, np.array([1.0, 0.5, 0.2]), 0.8)

        state = lt.to_dict()
        lt2 = LinTSLearner.from_dict(state)

        A1, b1 = lt.get_params(0)
        A2, b2 = lt2.get_params(0)
        np.testing.assert_allclose(A1, A2)
        np.testing.assert_allclose(b1, b2)

    def test_cholesky_fallback_with_degenerate_matrix(self):
        lt = LinTSLearner(feature_dim=2, lambda_=0.0, jitter_max=1e-12)
        lt.add_worker(0)
        lt._A[0] = np.zeros((2, 2))
        lt._b[0] = np.array([1.0, 0.0])
        s = lt.sample(0, np.array([1.0, 0.0]))
        assert np.isfinite(s)


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------
class TestLatencyTracker:
    def test_global_baseline_fallback(self):
        lt = LatencyTracker(ema_alpha=0.5)
        assert lt.get_global_baseline(per_tok=True, fallback=10.0) == 10.0
        assert lt.get_global_baseline(per_tok=False, fallback=5.0) == 5.0

    def test_update_and_retrieve(self):
        lt = LatencyTracker(ema_alpha=1.0)
        lt.update_baselines(wid=0, osl="M", prefill_bin="L", metric=100.0, per_tok=True)
        assert lt.get_global_baseline(per_tok=True, fallback=0.0) == 100.0

    def test_ema_smoothing(self):
        lt = LatencyTracker(ema_alpha=0.5)
        lt.update_baselines(wid=0, osl="M", prefill_bin="L", metric=100.0, per_tok=True)
        lt.update_baselines(wid=0, osl="M", prefill_bin="L", metric=200.0, per_tok=True)
        baseline = lt.get_global_baseline(per_tok=True, fallback=0.0)
        assert abs(baseline - 150.0) < 1.0

    def test_hierarchical_lookup(self):
        lt = LatencyTracker(ema_alpha=1.0)
        lt.update_baselines(wid=0, osl="M", prefill_bin="L", metric=50.0, per_tok=False)
        val = lt.get_baseline(wid=0, osl="M", prefill_bin="L", per_tok=False, fallback=999.0)
        assert val == 50.0

        val2 = lt.get_baseline(wid=0, osl="X", prefill_bin="Y", per_tok=False, fallback=999.0)
        assert val2 == 50.0

    def test_reset(self):
        lt = LatencyTracker()
        lt.update_baselines(wid=0, osl="M", prefill_bin="L", metric=100.0, per_tok=True)
        lt.reset()
        assert lt.get_global_baseline(per_tok=True, fallback=42.0) == 42.0

    def test_latency_metric_per_token(self):
        metric, per_tok = LatencyTracker.latency_metric(1000.0, 100)
        assert per_tok is True
        assert abs(metric - 10.0) < 1e-6

    def test_latency_metric_absolute(self):
        metric, per_tok = LatencyTracker.latency_metric(500.0, 0)
        assert per_tok is False
        assert metric == 500.0

    def test_compute_reward_fast(self):
        reward = LatencyTracker.compute_reward(metric=10.0, baseline=100.0, success=True)
        assert reward > 0.8

    def test_compute_reward_slow(self):
        reward = LatencyTracker.compute_reward(metric=1000.0, baseline=100.0, success=True)
        assert reward < 0.15

    def test_compute_reward_failure(self):
        assert LatencyTracker.compute_reward(metric=10.0, baseline=100.0, success=False) == 0.0

    def test_compute_reward_equal(self):
        reward = LatencyTracker.compute_reward(metric=100.0, baseline=100.0, success=True)
        assert abs(reward - 0.5) < 1e-6

    def test_global_bucket_baseline_separates_osl(self):
        """Long-decode and short-decode requests should have separate baselines."""
        lt = LatencyTracker(ema_alpha=1.0)
        # Short-decode requests: fast (10 ms/tok)
        lt.update_baselines(wid=0, osl="S", prefill_bin="M", metric=10.0, per_tok=True)
        # Long-decode requests: slower (50 ms/tok)
        lt.update_baselines(wid=0, osl="L", prefill_bin="M", metric=50.0, per_tok=True)

        short_baseline = lt.get_global_bucket_baseline("S", "M", True, fallback=1.0)
        long_baseline = lt.get_global_bucket_baseline("L", "M", True, fallback=1.0)

        assert abs(short_baseline - 10.0) < 1e-6
        assert abs(long_baseline - 50.0) < 1e-6

    def test_global_bucket_baseline_shared_across_workers(self):
        """Bucket baselines are global, not per-worker."""
        lt = LatencyTracker(ema_alpha=0.5)
        lt.update_baselines(wid=0, osl="M", prefill_bin="M", metric=100.0, per_tok=True)
        lt.update_baselines(wid=1, osl="M", prefill_bin="M", metric=200.0, per_tok=True)

        # Both workers contribute to the same global bucket
        baseline = lt.get_global_bucket_baseline("M", "M", True, fallback=1.0)
        assert abs(baseline - 150.0) < 1e-6  # EMA: 0.5*100=100, then 0.5*200+0.5*100=150

    def test_global_bucket_baseline_falls_back_to_global(self):
        """Unseen bucket falls through to global baseline."""
        lt = LatencyTracker(ema_alpha=1.0)
        lt.update_baselines(wid=0, osl="M", prefill_bin="M", metric=100.0, per_tok=True)

        # Unseen bucket (osl="L") should fall back to global
        baseline = lt.get_global_bucket_baseline("L", "M", True, fallback=999.0)
        assert abs(baseline - 100.0) < 1e-6

    def test_global_bucket_separates_per_tok(self):
        """ms/token and raw ms buckets stay separate even for same osl/prefill."""
        lt = LatencyTracker(ema_alpha=1.0)
        lt.update_baselines(wid=0, osl="M", prefill_bin="M", metric=10.0, per_tok=True)
        lt.update_baselines(wid=0, osl="M", prefill_bin="M", metric=500.0, per_tok=False)

        assert abs(lt.get_global_bucket_baseline("M", "M", True, 1.0) - 10.0) < 1e-6
        assert abs(lt.get_global_bucket_baseline("M", "M", False, 1.0) - 500.0) < 1e-6

    def test_mixed_osl_reward_fairness(self):
        """A long-decode request that is fast for its type should get high reward."""
        lt = LatencyTracker(ema_alpha=0.5)
        # Establish baselines: short=10 ms/tok, long=50 ms/tok
        for _ in range(5):
            lt.update_baselines(wid=0, osl="S", prefill_bin="M", metric=10.0, per_tok=True)
            lt.update_baselines(wid=1, osl="L", prefill_bin="M", metric=50.0, per_tok=True)

        # A long-decode request at 30 ms/tok is fast for its type
        long_baseline = lt.get_global_bucket_baseline("L", "M", True, fallback=1.0)
        reward_long = LatencyTracker.compute_reward(30.0, long_baseline, True)

        # Without bucketing (global baseline ≈ mix of 10 and 50), 30 ms/tok looks bad
        global_baseline = lt.get_global_baseline(True, fallback=1.0)
        reward_global = LatencyTracker.compute_reward(30.0, global_baseline, True)

        # Bucketed reward should be higher (30 < 50 baseline vs 30 ≈ mixed baseline)
        assert reward_long > reward_global


# ---------------------------------------------------------------------------
# PendingDecisions
# ---------------------------------------------------------------------------
class TestPendingDecisions:
    def test_add_and_pop(self):
        pd = PendingDecisions()
        pd.add("req1", {"wid": 0, "start_ts": time.time()})
        rec = pd.pop("req1")
        assert rec is not None
        assert rec["wid"] == 0
        assert pd.pop("req1") is None

    def test_count(self):
        pd = PendingDecisions()
        assert pd.count() == 0
        pd.add("a", {"wid": 0})
        pd.add("b", {"wid": 1})
        assert pd.count() == 2

    def test_per_worker_counts(self):
        pd = PendingDecisions()
        pd.add("a", {"wid": 0})
        pd.add("b", {"wid": 0})
        pd.add("c", {"wid": 1})
        counts = pd.per_worker_counts()
        assert counts[0] == 2
        assert counts[1] == 1

    def test_sweep_expires_old(self):
        pd = PendingDecisions(timeout_seconds=1.0, sweep_interval_seconds=0.0)
        old_ts = time.time() - 5.0
        pd.add("old", {"wid": 0, "start_ts": old_ts})
        pd.add("new", {"wid": 1, "start_ts": time.time()})

        expired = pd.sweep(time.time())
        assert len(expired) == 1
        assert expired[0][0] == "old"
        assert pd.count() == 1

    def test_sweep_respects_interval(self):
        pd = PendingDecisions(timeout_seconds=0.0, sweep_interval_seconds=999.0)
        now = time.time()
        pd._last_sweep = now
        pd.add("x", {"wid": 0, "start_ts": 0.0})
        expired = pd.sweep(now + 1.0)
        assert len(expired) == 0

    def test_pop_unknown_returns_none(self):
        pd = PendingDecisions()
        assert pd.pop("nonexistent") is None
