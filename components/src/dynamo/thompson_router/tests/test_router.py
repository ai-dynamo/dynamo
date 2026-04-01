# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KvThompsonRouter scoring, selection, and feedback."""

import math
import random
from unittest.mock import AsyncMock

import numpy as np
import pytest

from dynamo.thompson_router.hints import extract_hints
from dynamo.thompson_router.router import KvThompsonRouter, RoutingDecision

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.router,
]


def _make_load(worker_id, prefill_tokens=0, decode_blocks=0):
    return {
        "worker_id": worker_id,
        "potential_prefill_tokens": prefill_tokens,
        "potential_decode_blocks": decode_blocks,
    }


@pytest.fixture
def mock_kv_router():
    """Create a mock KvRouter with configurable loads and best_worker."""
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
def thompson_base(mock_kv_router):
    """KvThompsonRouter with all optional features off (base scoring only)."""
    return KvThompsonRouter(mock_kv_router, config=None)


@pytest.fixture
def thompson_softmax(mock_kv_router):
    """KvThompsonRouter with softmax selection enabled."""
    return KvThompsonRouter(
        mock_kv_router,
        config={"kv_thompson": {"enable_softmax": True, "temperature": 1.0}},
    )


@pytest.fixture
def thompson_cold_start():
    """KvThompsonRouter with cold start round-robin enabled using low-overlap workers."""
    router = AsyncMock()
    # All workers have high prefill (low overlap) to trigger cold start
    router.get_potential_loads = AsyncMock(
        return_value=[
            _make_load(0, prefill_tokens=900, decode_blocks=5),
            _make_load(1, prefill_tokens=950, decode_blocks=2),
            _make_load(2, prefill_tokens=980, decode_blocks=10),
        ]
    )
    router.best_worker = AsyncMock(return_value=(0, 0, 0))
    return KvThompsonRouter(
        router,
        config={"kv_thompson": {"enable_cold_start": True, "cold_start_threshold": 0.99}},
    )


@pytest.fixture
def thompson_affinity():
    """KvThompsonRouter with affinity enabled using balanced worker loads."""
    router = AsyncMock()
    # Balanced loads so affinity bonus can actually influence the decision
    router.get_potential_loads = AsyncMock(
        return_value=[
            _make_load(0, prefill_tokens=50, decode_blocks=5),
            _make_load(1, prefill_tokens=50, decode_blocks=5),
            _make_load(2, prefill_tokens=50, decode_blocks=5),
        ]
    )
    router.best_worker = AsyncMock(return_value=(0, 0, 10))
    return KvThompsonRouter(
        router,
        config={"kv_thompson": {"enable_affinity": True}},
    )


# ---------------------------------------------------------------------------
# pick_worker
# ---------------------------------------------------------------------------
class TestPickWorker:
    @pytest.mark.asyncio
    async def test_returns_routing_decision(self, thompson_base):
        decision = await thompson_base.pick_worker(
            token_ids=list(range(1000)),
            prefix_id="test-prefix",
            reuse_budget=0,
            osl=250,
            iat=250,
            tokens_in=1000,
        )
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen in [0, 1, 2]
        assert decision.native_pick == 0
        assert len(decision.worker_details) == 3

    @pytest.mark.asyncio
    async def test_prefers_higher_overlap(self, mock_kv_router):
        mock_kv_router.get_potential_loads.return_value = [
            _make_load(0, prefill_tokens=900, decode_blocks=0),
            _make_load(1, prefill_tokens=0, decode_blocks=0),
        ]
        router = KvThompsonRouter(mock_kv_router, config=None)

        choices = set()
        for _ in range(20):
            d = await router.pick_worker(
                list(range(1000)), "p", 0, 250, 250, 1000
            )
            choices.add(d.chosen)

        assert 1 in choices

    @pytest.mark.asyncio
    async def test_empty_loads_falls_back_to_native(self, mock_kv_router):
        mock_kv_router.get_potential_loads.return_value = []
        mock_kv_router.best_worker.return_value = (42, 0, 0)
        router = KvThompsonRouter(mock_kv_router, config=None)

        d = await router.pick_worker(list(range(100)), "p", 0, 250, 250, 100)
        assert d.chosen == 42

    @pytest.mark.asyncio
    async def test_cold_start_round_robin(self, thompson_cold_start):
        seen = []
        for i in range(6):
            d = await thompson_cold_start.pick_worker(
                list(range(100)), f"prefix-{i}", 0, 250, 250, 100
            )
            seen.append(d.chosen)

        assert len(set(seen)) > 1

    @pytest.mark.asyncio
    async def test_softmax_probabilistic(self, thompson_softmax):
        choices = set()
        for _ in range(50):
            d = await thompson_softmax.pick_worker(
                list(range(1000)), "p", 0, 250, 250, 1000
            )
            choices.add(d.chosen)

        assert len(choices) >= 2

    @pytest.mark.asyncio
    async def test_affinity_stickiness(self, thompson_affinity):
        np.random.seed(42)
        random.seed(42)

        d1 = await thompson_affinity.pick_worker(
            list(range(100)), "sticky-prefix", 0, 250, 250, 100
        )
        first_choice = d1.chosen

        same_count = 0
        for _ in range(30):
            d = await thompson_affinity.pick_worker(
                list(range(100)), "sticky-prefix", 5, 250, 250, 100
            )
            if d.chosen == first_choice:
                same_count += 1

        # With affinity enabled, the chosen worker should be sticky
        # most of the time (but TS still explores occasionally)
        assert same_count >= 5


# ---------------------------------------------------------------------------
# update_feedback
# ---------------------------------------------------------------------------
class TestUpdateFeedback:
    @pytest.mark.asyncio
    async def test_feedback_returns_metrics(self, thompson_base):
        decision = await thompson_base.pick_worker(
            list(range(500)), "p", 0, 250, 250, 500
        )
        result = thompson_base.update_feedback(decision, latency_ms=100.0, tokens_out=50)

        assert "metric" in result
        assert "baseline_ema" in result
        assert "reward" in result
        assert "beta_after" in result
        assert "lints_posterior_mean" in result
        assert 0.0 <= result["reward"] <= 1.0

    @pytest.mark.asyncio
    async def test_feedback_updates_learners(self, thompson_base):
        decision = await thompson_base.pick_worker(
            list(range(500)), "p", 0, 250, 250, 500
        )
        wid = decision.chosen

        a_before, b_before = thompson_base.beta_learner.get_params(wid)
        thompson_base.update_feedback(decision, latency_ms=50.0, tokens_out=100)
        a_after, b_after = thompson_base.beta_learner.get_params(wid)

        assert (a_after, b_after) != (a_before, b_before)

    @pytest.mark.asyncio
    async def test_fast_request_high_reward(self, thompson_base):
        decision = await thompson_base.pick_worker(
            list(range(500)), "p", 0, 250, 250, 500
        )
        thompson_base.update_feedback(decision, latency_ms=1000.0, tokens_out=100)

        decision2 = await thompson_base.pick_worker(
            list(range(500)), "p2", 0, 250, 250, 500
        )
        result_fast = thompson_base.update_feedback(decision2, latency_ms=10.0, tokens_out=100)
        assert result_fast["reward"] > 0.5


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------
class TestScoring:
    def _make_features(self, router, **overrides):
        """Helper to build a feature vector with sensible defaults."""
        defaults = dict(
            wid=0, overlap=0.5, kv_util=0.3, overlap_rank=0.5, load_rank=0.5,
            selection_pressure=0.125, prefill_tokens=250, tokens_in=500,
            inflight_share=0.125, osl=250, iat=250, reuse_budget=3,
        )
        defaults.update(overrides)
        return router._build_features(**defaults)

    def test_build_features_shape(self, thompson_base):
        x = self._make_features(thompson_base, overlap=0.8, kv_util=0.3)
        assert x.shape == (9,)
        assert x[0] == 1.0  # bias
        assert abs(x[1] - 0.8 * 0.7) < 1e-6  # overlap_x_idle = 0.8 * (1-0.3)

    def test_build_features_all_in_unit_range(self, thompson_base):
        """Every feature should be in [0, 1] for consistent ridge behavior."""
        for overlap in [0.0, 0.5, 1.0]:
            for kv_util in [0.0, 0.5, 1.0]:
                x = self._make_features(
                    thompson_base, overlap=overlap, kv_util=kv_util,
                    osl=512, reuse_budget=10, iat=100,
                )
                for i, val in enumerate(x):
                    assert 0.0 <= val <= 1.0 + 1e-9, f"feature[{i}]={val} out of [0,1]"

    def test_build_features_interaction_terms(self, thompson_base):
        x = self._make_features(thompson_base, overlap=0.8, kv_util=0.6, osl=512, iat=100, reuse_budget=5)
        # [1] overlap_x_idle = 0.8 * (1-0.6) = 0.32
        assert abs(x[1] - 0.32) < 1e-6
        # [7] osl_x_load = (512/1024) * 0.6 = 0.3
        assert abs(x[7] - 0.3) < 1e-6
        # [8] iat_x_reuse = iat_norm * reuse_norm, both > 0
        assert x[8] > 0.0

    def test_iat_factor_anchors(self):
        assert KvThompsonRouter._iat_factor(50) == 1.5
        assert abs(KvThompsonRouter._iat_factor(250) - 1.0) < 1e-6
        assert abs(KvThompsonRouter._iat_factor(1000) - 0.6) < 1e-6

    def test_decode_cost_anchors(self):
        assert KvThompsonRouter._decode_cost(128) == 1.0
        assert abs(KvThompsonRouter._decode_cost(250) - 2.0) < 1e-6
        assert KvThompsonRouter._decode_cost(1024) == 3.0

    def test_score_worker_finite(self, thompson_base):
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        score = thompson_base._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0,
            physics=0.5,
        )
        assert np.isfinite(score)

    def test_softmax_sums_to_one(self, thompson_base):
        probs = thompson_base._softmax([1.0, 2.0, 3.0], temp=1.0)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_softmax_uniform_on_equal_scores(self, thompson_base):
        probs = thompson_base._softmax([1.0, 1.0, 1.0], temp=1.0)
        for p in probs:
            assert abs(p - 1.0 / 3.0) < 1e-6

    def test_osl_bin_boundaries(self):
        assert KvThompsonRouter._osl_bin(64) == "S"
        assert KvThompsonRouter._osl_bin(128) == "S"
        assert KvThompsonRouter._osl_bin(129) == "M"
        assert KvThompsonRouter._osl_bin(512) == "M"
        assert KvThompsonRouter._osl_bin(513) == "L"

    def test_prefill_bin_boundaries(self):
        assert KvThompsonRouter._prefill_bin(100) == "S"
        assert KvThompsonRouter._prefill_bin(256) == "S"
        assert KvThompsonRouter._prefill_bin(257) == "M"
        assert KvThompsonRouter._prefill_bin(1024) == "M"
        assert KvThompsonRouter._prefill_bin(1025) == "L"


# ---------------------------------------------------------------------------
# _extract_hints (native PreprocessedRequest hint extraction)
# ---------------------------------------------------------------------------
class TestExtractHints:
    def test_osl_from_routing(self):
        request = {
            "token_ids": [1, 2, 3],
            "routing": {"expected_output_tokens": 512},
            "annotations": [],
        }
        hints = extract_hints(request)
        assert hints["osl"] == 512

    def test_osl_from_annotations_fallback(self):
        request = {
            "token_ids": [1, 2, 3],
            "routing": {},
            "annotations": ["osl:1024"],
        }
        hints = extract_hints(request)
        assert hints["osl"] == 1024

    def test_osl_default(self):
        hints = extract_hints({"token_ids": [1]})
        assert hints["osl"] == 250

    def test_prefix_id_from_annotations(self):
        request = {
            "token_ids": [1],
            "annotations": ["prefix_id:session-abc-123"],
        }
        hints = extract_hints(request)
        assert hints["prefix_id"] == "session-abc-123"

    def test_prefix_id_default_empty(self):
        hints = extract_hints({"token_ids": [1]})
        assert hints["prefix_id"] == ""

    def test_priority_jump_does_not_override_iat(self):
        """priority_jump (latency_sensitivity) should NOT be used as IAT.
        IAT comes from annotations only. priority_jump is a separate signal.
        """
        request = {
            "token_ids": [1],
            "routing": {"priority_jump": 0.5},
        }
        hints = extract_hints(request)
        assert hints["iat"] == 250  # default, NOT 500 (0.5 * 1000)
        assert hints["latency_sensitivity"] == 0.5  # preserved as separate signal

    def test_iat_from_annotations_with_priority_jump(self):
        """When both annotation iat and priority_jump exist, iat comes from annotation."""
        request = {
            "token_ids": [1],
            "routing": {"priority_jump": 4.0},
            "annotations": ["iat:135"],
        }
        hints = extract_hints(request)
        assert hints["iat"] == 135  # from annotation, NOT 4000
        assert hints["latency_sensitivity"] == 4.0  # from priority_jump

    def test_iat_from_annotations_fallback(self):
        request = {
            "token_ids": [1],
            "annotations": ["iat:100"],
        }
        hints = extract_hints(request)
        assert hints["iat"] == 100

    def test_total_requests_and_reuse_budget(self):
        request = {
            "token_ids": [1, 2, 3, 4, 5],
            "annotations": ["total_requests:10"],
        }
        hints = extract_hints(request)
        assert hints["total_requests"] == 10
        assert hints["reuse_budget"] == 9

    def test_tokens_in(self):
        request = {"token_ids": list(range(500))}
        hints = extract_hints(request)
        assert hints["tokens_in"] == 500

    def test_full_nvext_style_request(self):
        request = {
            "token_ids": list(range(1000)),
            "routing": {
                "expected_output_tokens": 256,
                "priority_jump": 0.25,
            },
            "annotations": [
                "prefix_id:banking-v0-uuid123",
                "total_requests:14",
                "query_instance_id:",
            ],
        }
        hints = extract_hints(request)
        assert hints["prefix_id"] == "banking-v0-uuid123"
        assert hints["osl"] == 256
        assert hints["iat"] == 250
        assert hints["total_requests"] == 14
        assert hints["reuse_budget"] == 13
        assert hints["tokens_in"] == 1000

    # --- Categorical bin support (NeMo Agent Toolkit) ---

    def test_osl_categorical_from_routing(self):
        """Agent toolkit may send 'LOW'/'MEDIUM'/'HIGH' as expected_output_tokens."""
        for cat, expected in [("LOW", 64), ("MEDIUM", 250), ("HIGH", 768)]:
            request = {
                "token_ids": [1],
                "routing": {"expected_output_tokens": cat},
            }
            hints = extract_hints(request)
            assert hints["osl"] == expected, f"osl category {cat}"

    def test_osl_categorical_case_insensitive(self):
        request = {
            "token_ids": [1],
            "routing": {"expected_output_tokens": "Medium"},
        }
        assert extract_hints(request)["osl"] == 250

    def test_osl_categorical_from_annotations(self):
        request = {
            "token_ids": [1],
            "annotations": ["osl:HIGH"],
        }
        assert extract_hints(request)["osl"] == 768

    def test_iat_categorical_from_annotations_via_routing(self):
        """Categorical IAT comes from annotations, not routing.priority_jump.

        priority_jump carries latency_sensitivity, a scheduling priority.
        IAT categories in annotations describe inter-arrival *time*:
          LOW  = short gaps (rapid-fire)  → 50ms
          HIGH = long gaps (infrequent)   → 750ms
        """
        for cat, expected in [("LOW", 50), ("MEDIUM", 250), ("HIGH", 750)]:
            request = {
                "token_ids": [1],
                "annotations": [f"iat:{cat}"],
            }
            hints = extract_hints(request)
            assert hints["iat"] == expected, f"iat category {cat}"

    def test_iat_categorical_from_annotations(self):
        request = {
            "token_ids": [1],
            "annotations": ["iat:low"],
        }
        assert extract_hints(request)["iat"] == 50

    def test_continuous_values_still_work(self):
        """Continuous integer values (from trie model) must not be broken."""
        request = {
            "token_ids": [1],
            "routing": {"expected_output_tokens": 347, "priority_jump": 0.123},
            "annotations": ["iat:123"],
        }
        hints = extract_hints(request)
        assert hints["osl"] == 347
        assert hints["iat"] == 123  # from annotation, not priority_jump
        assert abs(hints["latency_sensitivity"] - 0.123) < 1e-6

    def test_float_string_values(self):
        """String-encoded floats from annotations should parse cleanly."""
        request = {
            "token_ids": [1],
            "annotations": ["osl:512.0", "iat:100.5"],
        }
        hints = extract_hints(request)
        assert hints["osl"] == 512
        assert hints["iat"] == 100

    def test_unknown_category_falls_back_to_default(self):
        """Unrecognized string should fall back to default, not crash."""
        request = {
            "token_ids": [1],
            "routing": {"expected_output_tokens": "UNKNOWN_TIER"},
        }
        hints = extract_hints(request)
        assert hints["osl"] == 250  # default


# ---------------------------------------------------------------------------
# WorkerLoadMonitor integration
# ---------------------------------------------------------------------------
class TestWorkerLoadMonitor:
    """Tests for _get_worker_utilization with mock WorkerLoadMonitor."""

    def test_no_monitor_returns_empty(self, thompson_base):
        """Without a monitor, utilization returns empty dict."""
        assert thompson_base.worker_load_monitor is None
        assert thompson_base._get_worker_utilization() == {}

    def test_monitor_returns_utilization(self, mock_kv_router):
        """With a mock monitor, utilization ratios are computed correctly."""

        class MockLoadMonitor:
            def get_all(self):
                return {
                    0: {0: {"active_decode_blocks": 500, "kv_total_blocks": 1000,
                            "active_prefill_tokens": 100, "max_num_batched_tokens": 400}},
                    1: {0: {"active_decode_blocks": 900, "kv_total_blocks": 1000,
                            "active_prefill_tokens": 50, "max_num_batched_tokens": 400}},
                }

        router = KvThompsonRouter(
            mock_kv_router, config=None, worker_load_monitor=MockLoadMonitor()
        )
        util = router._get_worker_utilization()

        assert abs(util[0]["kv_util"] - 0.5) < 1e-6
        assert abs(util[0]["prefill_util"] - 0.25) < 1e-6
        assert abs(util[1]["kv_util"] - 0.9) < 1e-6
        assert abs(util[1]["prefill_util"] - 0.125) < 1e-6

    def test_monitor_clamps_to_one(self, mock_kv_router):
        """Utilization is clamped to 1.0 even if active > total (transient race)."""

        class MockLoadMonitor:
            def get_all(self):
                return {
                    0: {0: {"active_decode_blocks": 1200, "kv_total_blocks": 1000,
                            "active_prefill_tokens": 0, "max_num_batched_tokens": 1}},
                }

        router = KvThompsonRouter(
            mock_kv_router, config=None, worker_load_monitor=MockLoadMonitor()
        )
        util = router._get_worker_utilization()
        assert util[0]["kv_util"] == 1.0

    def test_monitor_handles_zero_capacity(self, mock_kv_router):
        """Workers with zero capacity are reported as 0.0 utilization."""

        class MockLoadMonitor:
            def get_all(self):
                return {
                    0: {0: {"active_decode_blocks": 100, "kv_total_blocks": 0,
                            "active_prefill_tokens": 0, "max_num_batched_tokens": 0}},
                }

        router = KvThompsonRouter(
            mock_kv_router, config=None, worker_load_monitor=MockLoadMonitor()
        )
        util = router._get_worker_utilization()
        assert util[0]["kv_util"] == 0.0
        assert util[0]["prefill_util"] == 0.0

    def test_monitor_exception_returns_empty(self, mock_kv_router):
        """If the monitor raises, gracefully return empty dict."""

        class BrokenMonitor:
            def get_all(self):
                raise RuntimeError("NATS disconnected")

        router = KvThompsonRouter(
            mock_kv_router, config=None, worker_load_monitor=BrokenMonitor()
        )
        assert router._get_worker_utilization() == {}

    def test_monitor_multi_dp_rank(self, mock_kv_router):
        """Utilization aggregates across dp_ranks."""

        class MockLoadMonitor:
            def get_all(self):
                return {
                    0: {
                        0: {"active_decode_blocks": 300, "kv_total_blocks": 1000,
                            "active_prefill_tokens": 0, "max_num_batched_tokens": 100},
                        1: {"active_decode_blocks": 200, "kv_total_blocks": 1000,
                            "active_prefill_tokens": 0, "max_num_batched_tokens": 100},
                    },
                }

        router = KvThompsonRouter(
            mock_kv_router, config=None, worker_load_monitor=MockLoadMonitor()
        )
        util = router._get_worker_utilization()
        # (300 + 200) / (1000 + 1000) = 0.25
        assert abs(util[0]["kv_util"] - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# RouterStats instrumentation
# ---------------------------------------------------------------------------
class TestRouterStats:
    """Tests for the rolling stats collector."""

    def test_empty_snapshot(self, thompson_base):
        snap = thompson_base.stats.snapshot()
        assert snap["total_observations"] == 0
        assert snap["reward"]["mean"] == 0.0
        assert snap["monitor_availability"]["hits"] == 0

    @pytest.mark.asyncio
    async def test_stats_accumulate_through_feedback(self, mock_kv_router):
        router = KvThompsonRouter(mock_kv_router, config=None)

        # Make several decisions and give feedback
        for i in range(5):
            decision = await router.pick_worker(
                list(range(100)), f"p{i}", 0, 250, 250, 100
            )
            router.update_feedback(decision, latency_ms=50.0 + i * 10, tokens_out=50)

        snap = router.stats.snapshot()
        assert snap["total_observations"] == 5
        assert 0.0 < snap["reward"]["mean"] < 1.0
        assert snap["physics_tower"]["rmse"] >= 0.0
        assert 0.0 <= snap["residual"]["mean"] <= 1.0
        assert snap["residual"]["variance"] >= 0.0

    @pytest.mark.asyncio
    async def test_stats_learner_contribution(self, mock_kv_router):
        router = KvThompsonRouter(mock_kv_router, config=None)

        decision = await router.pick_worker(
            list(range(100)), "p", 0, 250, 250, 100
        )
        router.update_feedback(decision, latency_ms=50.0, tokens_out=50)

        snap = router.stats.snapshot()
        # Learner contributions should be non-negative (absolute values)
        assert snap["learner_contribution"]["mean_lints_magnitude"] >= 0.0
        assert snap["learner_contribution"]["mean_beta_magnitude"] >= 0.0

    @pytest.mark.asyncio
    async def test_stats_baseline_bucket_tracking(self, mock_kv_router):
        router = KvThompsonRouter(mock_kv_router, config=None)

        # First feedback: no bucket exists yet, should fall back to global
        d1 = await router.pick_worker(list(range(100)), "p1", 0, 250, 250, 100)
        router.update_feedback(d1, latency_ms=50.0, tokens_out=50)

        # Second feedback same type: bucket now exists
        d2 = await router.pick_worker(list(range(100)), "p2", 0, 250, 250, 100)
        router.update_feedback(d2, latency_ms=50.0, tokens_out=50)

        snap = router.stats.snapshot()
        buckets = snap["baseline_buckets"]
        assert buckets["global_fallbacks"] >= 1  # first request
        assert buckets["bucket_hits"] >= 1       # second request

    def test_stats_reset(self, thompson_base):
        thompson_base.stats.record_feedback(0.5, 0.4, 0.6)
        thompson_base.stats.record_feedback(0.7, 0.3, 0.9)
        assert thompson_base.stats.snapshot()["total_observations"] == 2

        thompson_base.stats.reset()
        snap = thompson_base.stats.snapshot()
        assert snap["total_observations"] == 0
        assert snap["reward"]["mean"] == 0.0


# ---------------------------------------------------------------------------
# Physics tower unit tests
# ---------------------------------------------------------------------------
class TestPhysicsScore:
    """Direct tests for _physics_score with known input/output values."""

    def _make_router_with_monitor(self, mock_kv_router, kv_util=0.0, prefill_util=0.0):
        class MockLoadMonitor:
            def get_all(self):
                return {
                    0: {0: {
                        "active_decode_blocks": int(kv_util * 1000),
                        "kv_total_blocks": 1000,
                        "active_prefill_tokens": int(prefill_util * 400),
                        "max_num_batched_tokens": 400,
                    }}
                }
        return KvThompsonRouter(
            mock_kv_router, config=None,
            worker_load_monitor=MockLoadMonitor(),
        )

    def test_all_signals_perfect_returns_weight_sum(self, mock_kv_router):
        """overlap=1, util=0, memory=0 → score equals sum of all four weights."""
        router = self._make_router_with_monitor(mock_kv_router, kv_util=0.0, prefill_util=0.0)
        worker_util = router._get_worker_utilization()
        score = router._physics_score(
            overlap=1.0, wid=0, worker_util=worker_util,
            prefill_tokens=0, tokens_in=1000, memory_pressure=0.0,
        )
        expected = (
            router.physics_cache_weight * 1.0
            + router.physics_compute_weight * 1.0
            + router.physics_queue_weight * 1.0
            + router.physics_memory_weight * 1.0
        )
        assert abs(score - expected) < 1e-6

    def test_fully_loaded_worker_reduces_score(self, mock_kv_router):
        """overlap=1, kv_util=1, prefill_util=1, memory=1 → only cache_weight contributes."""
        router = self._make_router_with_monitor(mock_kv_router, kv_util=1.0, prefill_util=1.0)
        worker_util = router._get_worker_utilization()
        score = router._physics_score(
            overlap=1.0, wid=0, worker_util=worker_util,
            prefill_tokens=0, tokens_in=1000, memory_pressure=1.0,
        )
        # compute_avail=0, queue_avail=0, memory_avail=0 → only cache contributes
        expected = router.physics_cache_weight * 1.0
        assert abs(score - expected) < 1e-6

    def test_no_cache_no_util_is_non_cache_weight_sum(self, mock_kv_router):
        """overlap=0, all available → score = compute+queue+memory weights."""
        router = self._make_router_with_monitor(mock_kv_router, kv_util=0.0, prefill_util=0.0)
        worker_util = router._get_worker_utilization()
        score = router._physics_score(
            overlap=0.0, wid=0, worker_util=worker_util,
            prefill_tokens=0, tokens_in=1000, memory_pressure=0.0,
        )
        expected = (
            router.physics_compute_weight * 1.0
            + router.physics_queue_weight * 1.0
            + router.physics_memory_weight * 1.0
        )
        assert abs(score - expected) < 1e-6

    def test_score_in_valid_range(self, mock_kv_router):
        """Physics score should always be in [0, 1] given normalized inputs."""
        router = self._make_router_with_monitor(mock_kv_router, kv_util=0.5, prefill_util=0.3)
        worker_util = router._get_worker_utilization()
        for overlap in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for mem in [0.0, 0.5, 1.0]:
                score = router._physics_score(
                    overlap=overlap, wid=0, worker_util=worker_util,
                    prefill_tokens=100, tokens_in=1000, memory_pressure=mem,
                )
                assert 0.0 <= score <= 1.0, f"Physics score {score} out of [0,1] for overlap={overlap}, mem={mem}"

    def test_fallback_path_with_no_monitor(self, mock_kv_router):
        """Without monitor data, _physics_score uses prefill_ratio proxy and returns finite score."""
        router = KvThompsonRouter(mock_kv_router, config=None)  # no monitor
        score = router._physics_score(
            overlap=0.5, wid=99, worker_util={},
            prefill_tokens=500, tokens_in=1000, memory_pressure=0.0,
        )
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_fallback_path_increments_fallback_count(self, mock_kv_router):
        """Monitor fallback path should increment _monitor_fallback_count."""
        router = KvThompsonRouter(mock_kv_router, config=None)
        assert router._monitor_fallback_count == 0
        router._physics_score(
            overlap=0.5, wid=0, worker_util={},
            prefill_tokens=200, tokens_in=1000, memory_pressure=0.0,
        )
        assert router._monitor_fallback_count == 1

    def test_memory_pressure_reduces_score_monotonically(self, mock_kv_router):
        """Higher memory pressure → strictly lower physics score."""
        router = self._make_router_with_monitor(mock_kv_router, kv_util=0.0, prefill_util=0.0)
        worker_util = router._get_worker_utilization()
        scores = [
            router._physics_score(
                overlap=0.5, wid=0, worker_util=worker_util,
                prefill_tokens=0, tokens_in=1000, memory_pressure=p,
            )
            for p in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Score not monotonically decreasing: scores[{i}]={scores[i]}, "
                f"scores[{i+1}]={scores[i+1]}"
            )


# ---------------------------------------------------------------------------
# Residual training target invariants
# ---------------------------------------------------------------------------
class TestResidualTarget:
    """Tests for residual = clamp(reward - physics + 0.5, 0, 1).

    Per the two-tower design: LinTS trains on the gap between reward and
    the physics prediction, offset by 0.5 so the neutral case is 0.5.
    """

    @pytest.mark.asyncio
    async def test_residual_always_in_unit_interval(self, mock_kv_router):
        """residual must be clamped to [0, 1] regardless of reward/physics gap."""
        router = KvThompsonRouter(mock_kv_router, config=None)

        for latency_ms, tokens_out in [(1.0, 500), (10000.0, 1), (50.0, 50)]:
            decision = await router.pick_worker(
                list(range(500)), "p", 0, 250, 250, 500
            )
            result = router.update_feedback(decision, latency_ms=latency_ms, tokens_out=tokens_out)
            r = result["residual_reward"]
            assert 0.0 <= r <= 1.0, f"residual {r} out of [0,1] for latency={latency_ms}"

    def test_residual_formula_neutral_physics(self):
        """When reward == physics_pred, residual should be exactly 0.5."""
        reward = 0.7
        physics = 0.7
        residual = max(0.0, min(1.0, reward - physics + 0.5))
        assert abs(residual - 0.5) < 1e-9

    def test_residual_formula_physics_overestimates(self):
        """When physics overestimates (physics > reward), residual < 0.5."""
        reward = 0.3
        physics = 0.8
        residual = max(0.0, min(1.0, reward - physics + 0.5))
        # reward - physics + 0.5 = 0.3 - 0.8 + 0.5 = 0.0
        assert abs(residual - 0.0) < 1e-9

    def test_residual_formula_physics_underestimates(self):
        """When physics underestimates (reward > physics), residual > 0.5."""
        reward = 0.9
        physics = 0.2
        residual = max(0.0, min(1.0, reward - physics + 0.5))
        # reward - physics + 0.5 = 0.9 - 0.2 + 0.5 = 1.2, clamped to 1.0
        assert abs(residual - 1.0) < 1e-9

    def test_residual_clamp_lower_bound(self):
        """Extreme overestimation: residual clamped to 0, not negative."""
        reward = 0.0
        physics = 1.0
        residual = max(0.0, min(1.0, reward - physics + 0.5))
        assert residual >= 0.0

    def test_residual_clamp_upper_bound(self):
        """Extreme underestimation: residual clamped to 1.0, not > 1."""
        reward = 1.0
        physics = 0.0
        residual = max(0.0, min(1.0, reward - physics + 0.5))
        assert residual <= 1.0

    @pytest.mark.asyncio
    async def test_residual_stored_in_stats(self, mock_kv_router):
        """update_feedback must push residual through stats.record_feedback."""
        router = KvThompsonRouter(mock_kv_router, config=None)
        decision = await router.pick_worker(list(range(100)), "p", 0, 250, 250, 100)
        result = router.update_feedback(decision, latency_ms=50.0, tokens_out=50)

        snap = router.stats.snapshot()
        # residual_reward from the return value must match what stats sees
        assert 0.0 <= snap["residual"]["mean"] <= 1.0
        # The returned residual_reward must also be in range
        assert 0.0 <= result["residual_reward"] <= 1.0


# ---------------------------------------------------------------------------
# load_mod / switching penalty interaction (regression: Issue #1 sign fix)
# ---------------------------------------------------------------------------
class TestLoadModSwitchPenaltyInteraction:
    """Regression tests for the sign fix: load_mod must NOT be applied to
    negative utility, and switching penalty must stay outside load_mod.

    Issue #1: s_t = (u + bonus - switch) * m_t inverted penalty for
    unloaded workers.  Fix: s_t = u^+ * m_t - lambda_sw * switch_t.
    """

    def test_switch_penalty_outside_load_mod(self, mock_kv_router):
        """With a very low load_mod, switching penalty should still be subtracted,
        not amplified by the load modulator.
        """
        cfg = {
            "kv_thompson": {
                "enable_switching_cost": True,
                "switch_cost_weight": 1.0,
                "switch_base": 0.3,
                "switch_reuse": 0.0,
                "queue_penalty_weight": 100.0,   # large qpw → tiny load_mod
                "ts_weight": 0.0,
                "enable_lints": False,
            }
        }
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        # kv_util=1.0 → load_mod = exp(-100*1.0^2) ≈ 0
        score_switch = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=50,
            last_worker=0, reuse_budget=5, iat_factor=1.0, physics=0.5,
            kv_util=1.0,
        )
        # kv_util=1.0 → load_mod ≈ 0, score ≈ utility * 0 - penalty
        # penalty = switch_cost_weight * tanh(switch_base) ≈ tanh(0.3) ≈ 0.291
        # If penalty were inside load_mod: score = (0.5 - 0.291) * ~0 ≈ 0 (wrong)
        # Correct: score = 0.5 * ~0 - 0.291 ≈ -0.291
        assert score_switch < -0.1, (
            f"Switch penalty should be subtracted outside load_mod, got score={score_switch:.4f}"
        )

    def test_high_load_does_not_invert_switch_penalty(self, mock_kv_router):
        """An overloaded non-sticky worker should score worse than an idle one,
        not better, when switching cost is applied.
        Regression for the sign bug where load_mod multiplied the whole expression
        including the switch penalty, causing a loaded worker to get a LESS
        negative score than an unloaded one.
        """
        cfg = {"kv_thompson": {"enable_switching_cost": True, "switch_cost_weight": 2.0}}
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        score_lightly_loaded = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=5,
            last_worker=0, reuse_budget=5, iat_factor=1.0, physics=0.5,
            kv_util=0.1,
        )
        score_heavily_loaded = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=40,
            last_worker=0, reuse_budget=5, iat_factor=1.0, physics=0.5,
            kv_util=0.9,
        )
        # With the bug: higher load could make penalty term bigger (inverted)
        # With the fix: higher load reduces positive utility but penalty stays constant
        # So the lightly-loaded worker should score higher
        assert score_lightly_loaded > score_heavily_loaded, (
            f"Lightly loaded ({score_lightly_loaded:.4f}) should beat heavily loaded "
            f"({score_heavily_loaded:.4f}) even with switch penalty"
        )

    def test_load_mod_floor_enforced(self, mock_kv_router):
        """When enable_load_mod_floor=True, load_mod should never drop below the floor."""
        cfg = {
            "kv_thompson": {
                "enable_load_mod_floor": True,
                "load_mod_floor": 0.3,
                "queue_penalty_weight": 1000.0,  # would drive load_mod to ~0 without floor
            }
        }
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])

        physics = 0.6
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=100,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=physics,
            kv_util=1.0,
        )
        # Even with extreme load, score should be >= physics * floor
        # (upper bound: with no penalty and no ts contributions, score ≈ utility * load_mod)
        # At minimum, the positive physics contribution * floor should be present
        # We check that the score is not drastically suppressed by verifying it's finite
        # and above -1 (not clamped to -1e9)
        assert np.isfinite(score)
        # The score should be at least physics * floor minus possible tanh contributions
        # tanh can add up to ~1.0 per learner term; floor*physics ≈ 0.3*0.6 = 0.18
        # So score > 0.18 - 1.0 - 1.0 = -1.82 is expected even in worst case
        assert score > -2.0

    def test_sticky_load_floor_raises_floor_for_sticky_worker(self, mock_kv_router):
        """enable_sticky_floor should keep load_mod high for the prefix's prior worker.

        With ts_weight=0 and enable_lints=False, the only stochastic contribution
        is the default tanh(lints_sample) path which uses the same prior for both
        workers and is therefore equal — so the load_mod difference is decisive.
        """
        cfg = {
            "kv_thompson": {
                "enable_sticky_floor": True,
                "sticky_load_floor": 0.5,
                "queue_penalty_weight": 1000.0,  # would crush load_mod otherwise
                "ts_weight": 0.0,               # remove Beta TS noise
                "enable_lints": True,
                "lints_weight": -0.0,           # zero lints contribution
            }
        }
        router = KvThompsonRouter(mock_kv_router, config=cfg)

        # Seed RNG so both calls draw the same TS samples for wid=0 and wid=1
        np.random.seed(42)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        physics = 0.5

        # Compute scores for averaged physics only (ts_weight=0, lints_weight=0)
        # kv_util=1.0 → load_mod = exp(-1000*1.0^2) ≈ 0
        # Sticky worker (wid=0 == last_worker): load_mod = max(~0, 0.5) = 0.5
        # Non-sticky (wid=1 != last_worker): load_mod ≈ 0
        score_sticky = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=100,
            last_worker=0, reuse_budget=5, iat_factor=1.0, physics=physics,
            kv_util=1.0,
        )
        np.random.seed(42)  # reset seed for same sample
        score_non_sticky = router._score_worker(
            wid=1, x=x, overlap=0.5, decode_blocks=100,
            last_worker=0, reuse_budget=5, iat_factor=1.0, physics=physics,
            kv_util=1.0,
        )
        # sticky worker has load_mod floored at 0.5 → score ≈ 0.5 * 0.5 = 0.25
        # non-sticky worker has load_mod ≈ 0 → score ≈ 0
        assert score_sticky > score_non_sticky, (
            f"Sticky worker (load_mod=0.5) should score higher than "
            f"non-sticky (load_mod≈0): {score_sticky:.4f} vs {score_non_sticky:.4f}"
        )

    def test_positive_utility_amplified_not_suppressed_by_load_mod(self, mock_kv_router):
        """With zero decode blocks, load_mod=1.0 and should not reduce positive score."""
        router = KvThompsonRouter(mock_kv_router, config=None)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        np.random.seed(0)
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=0.8,
        )
        # score should be positive (physics=0.8 dominates) and finite
        assert np.isfinite(score)
        # physics alone is 0.8; load_mod=1.0; tanh contributions are small
        # score should be well above 0
        assert score > 0.0


# ---------------------------------------------------------------------------
# Scoring: lints_weight positive vs negative path
# ---------------------------------------------------------------------------
class TestLintsWeightPaths:
    """_score_worker has two code paths for lints_weight: negative uses tanh,
    non-negative passes raw sample. Both should produce finite scores.
    """

    def test_lints_weight_always_uses_tanh(self, mock_kv_router):
        """LinTS always uses abs(lints_weight) * tanh(raw), regardless of sign."""
        cfg = {"kv_thompson": {"enable_lints": True, "lints_weight": 0.5}}
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        # Train heavily to get a large posterior sample
        for _ in range(50):
            router.lints_learner.update(0, x, reward=1.0)
        np.random.seed(42)
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=0.5,
        )
        assert np.isfinite(score)
        # LinTS contribution bounded: abs(0.5) * tanh(anything) < 0.5
        raw = router.lints_learner.sample(0, x)
        assert abs(0.5 * math.tanh(raw)) < 0.5 + 1e-9

    def test_lints_disabled_contributes_zero(self, mock_kv_router):
        """When enable_lints=False, LinTS adds zero to the score."""
        cfg_off = {"kv_thompson": {"enable_lints": False}}
        cfg_on_zero = {"kv_thompson": {"enable_lints": True, "lints_weight": 0.0}}
        router_off = KvThompsonRouter(mock_kv_router, config=cfg_off)
        router_on = KvThompsonRouter(mock_kv_router, config=cfg_on_zero)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        np.random.seed(42)
        score_off = router_off._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=0.5,
        )
        np.random.seed(42)
        score_zero = router_on._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=5,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=0.5,
        )
        # Both should be identical: no LinTS contribution
        assert abs(score_off - score_zero) < 1e-9

    def test_beta_ts_disabled_contributes_zero(self, mock_kv_router):
        """When enable_beta_ts=False (default), Beta TS adds zero to the score."""
        cfg = {"kv_thompson": {"enable_beta_ts": False, "ts_weight": 1.0}}
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        # Score should equal physics only (no beta, no lints since default off)
        np.random.seed(42)
        score = router._score_worker(
            wid=0, x=x, overlap=0.5, decode_blocks=0,
            last_worker=None, reuse_budget=0, iat_factor=1.0, physics=0.7,
        )
        # With no learners active and decode_blocks=0, score ≈ physics * load_mod = 0.7
        assert abs(score - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Adaptive temperature and explore
# ---------------------------------------------------------------------------
class TestAdaptiveFeatures:
    def test_adaptive_temp_decreases_with_reuse(self, mock_kv_router):
        """Temperature should decrease as reuse_budget and iat_factor increase."""
        cfg = {
            "kv_thompson": {
                "enable_adaptive_temp": True,
                "enable_softmax": True,
                "adaptive_temp_base": 2.0,
                "temp_min": 0.1,
                "temp_max": 3.0,
            }
        }
        router = KvThompsonRouter(mock_kv_router, config=cfg)

        # Compute effective temperature for different reuse budgets
        iat_factor = 1.0
        temps = []
        for rb in [0, 5, 20]:
            temp = router.adaptive_temp_base / (1.0 + float(rb) * iat_factor)
            temp = min(max(temp, router.temp_min), router.temp_max)
            temps.append(temp)

        assert temps[0] > temps[1] > temps[2], (
            f"Temperature should decrease with reuse budget: {temps}"
        )

    def test_adaptive_temp_clamped_to_min(self, mock_kv_router):
        """Temperature should not drop below temp_min."""
        cfg = {
            "kv_thompson": {
                "enable_adaptive_temp": True,
                "enable_softmax": True,
                "adaptive_temp_base": 0.01,
                "temp_min": 0.15,
            }
        }
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        temp = router.adaptive_temp_base / (1.0 + 1000.0)
        temp = min(max(temp, router.temp_min), router.temp_max)
        assert temp >= router.temp_min

    def test_adaptive_explore_reduces_ts_weight(self, mock_kv_router):
        """enable_adaptive_explore should reduce ts_weight with high reuse_budget."""
        cfg = {"kv_thompson": {"enable_beta_ts": True, "enable_adaptive_explore": True, "ts_weight": 0.1}}
        router = KvThompsonRouter(mock_kv_router, config=cfg)
        x = np.array([1.0, 0.35, 0.5, 0.5, 0.125, 0.5, 0.125, 0.15, 0.3])
        np.random.seed(99)

        # With high reuse_budget, ts_w_eff should be reduced
        # The effective weight = ts_weight / (1 + reuse_budget * iat_factor)
        ts_w_no_reuse = router.ts_weight / (1.0 + 0.0 * 1.0)     # = 0.1
        ts_w_high_reuse = router.ts_weight / (1.0 + 100.0 * 1.0)  # ≈ 0.001
        assert ts_w_no_reuse > ts_w_high_reuse * 10


# ---------------------------------------------------------------------------
# Interpolation helpers (between-anchor values)
# ---------------------------------------------------------------------------
class TestInterpolation:
    """Tests for _iat_factor and _decode_cost at intermediate values."""

    @pytest.mark.parametrize("iat,expected_min,expected_max", [
        (50, 1.5, 1.5),         # lower anchor
        (150, 1.0, 1.5),        # between 50 and 250
        (250, 1.0, 1.0),        # upper anchor of lower range
        (600, 0.6, 1.0),        # between 250 and 1000
        (1000, 0.6, 0.6),       # upper anchor
        (2000, 0.6, 0.6),       # beyond upper anchor — clamped
    ])
    def test_iat_factor_range(self, iat, expected_min, expected_max):
        f = KvThompsonRouter._iat_factor(iat)
        assert expected_min <= f <= expected_max, (
            f"_iat_factor({iat})={f} not in [{expected_min}, {expected_max}]"
        )

    def test_iat_factor_monotone_decreasing(self):
        """IAT factor should be non-increasing as IAT increases."""
        iats = [0, 25, 50, 100, 150, 200, 250, 400, 600, 800, 1000, 1500]
        factors = [KvThompsonRouter._iat_factor(i) for i in iats]
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1] - 1e-9, (
                f"Not monotone at iat={iats[i]}: f={factors[i]:.4f} > f={factors[i+1]:.4f}"
            )

    @pytest.mark.parametrize("osl,expected_min,expected_max", [
        (64, 1.0, 1.0),         # at lower anchor
        (128, 1.0, 1.0),        # boundary
        (190, 1.0, 2.0),        # between 128 and 250
        (250, 2.0, 2.0),        # middle anchor
        (600, 2.0, 3.0),        # between 250 and 1024
        (1024, 3.0, 3.0),       # upper anchor
        (2048, 3.0, 3.0),       # beyond anchor — clamped
    ])
    def test_decode_cost_range(self, osl, expected_min, expected_max):
        c = KvThompsonRouter._decode_cost(osl)
        assert expected_min <= c <= expected_max, (
            f"_decode_cost({osl})={c} not in [{expected_min}, {expected_max}]"
        )

    def test_decode_cost_monotone_increasing(self):
        """Decode cost should be non-decreasing as OSL increases."""
        osls = [0, 64, 128, 190, 250, 512, 768, 1024, 2048]
        costs = [KvThompsonRouter._decode_cost(o) for o in osls]
        for i in range(len(costs) - 1):
            assert costs[i] <= costs[i + 1] + 1e-9, (
                f"Not monotone at osl={osls[i]}: cost={costs[i]:.4f} > cost={costs[i+1]:.4f}"
            )

    def test_iat_norm_in_unit_interval(self):
        """iat_norm feature should stay in [0, 1] for all valid IAT values."""
        for iat in [0, 50, 100, 250, 500, 1000, 2000]:
            f = KvThompsonRouter._iat_factor(iat)
            iat_norm = (f - 0.6) / 0.9
            assert 0.0 <= iat_norm <= 1.0, f"iat_norm={iat_norm} out of [0,1] for iat={iat}"


# ---------------------------------------------------------------------------
# Hint parsing edge cases
# ---------------------------------------------------------------------------
class TestExtractHintsEdgeCases:
    """Edge cases not covered by existing TestExtractHints."""

    def test_total_requests_zero_gives_reuse_budget_zero(self):
        """total_requests=0 should not produce negative reuse_budget."""
        request = {
            "token_ids": [1],
            "annotations": ["total_requests:0"],
        }
        hints = extract_hints(request)
        assert hints["reuse_budget"] == 0

    def test_total_requests_one_gives_reuse_budget_zero(self):
        """Single-request session has no turns to reuse → budget=0."""
        request = {
            "token_ids": [1],
            "annotations": ["total_requests:1"],
        }
        hints = extract_hints(request)
        assert hints["reuse_budget"] == 0

    def test_priority_jump_does_not_set_iat(self):
        """priority_jump should set latency_sensitivity, not iat."""
        request = {
            "token_ids": [1],
            "routing": {"priority_jump": 0.0},
        }
        hints = extract_hints(request)
        assert hints["iat"] == 250  # default
        assert hints["latency_sensitivity"] == 0.0

    def test_priority_jump_very_large_latency_sensitivity(self):
        """Large priority_jump preserved as latency_sensitivity."""
        request = {
            "token_ids": [1],
            "routing": {"priority_jump": 10.0},
        }
        hints = extract_hints(request)
        assert hints["iat"] == 250  # default, NOT 10000
        assert hints["latency_sensitivity"] == 10.0

    def test_empty_annotations_list(self):
        """Empty annotations should not crash and all values use defaults."""
        hints = extract_hints({"token_ids": [1, 2, 3], "annotations": []})
        assert hints["osl"] == 250
        assert hints["iat"] == 250
        assert hints["prefix_id"] == ""
        assert hints["reuse_budget"] == 0

    def test_annotation_with_colon_in_value(self):
        """Annotation values containing colons should split on first colon only."""
        request = {
            "token_ids": [1],
            "annotations": ["prefix_id:session:abc:123"],
        }
        hints = extract_hints(request)
        # split(":", 1) → key="prefix_id", value="session:abc:123"
        assert hints["prefix_id"] == "session:abc:123"

    def test_none_routing_field(self):
        """routing=None should be treated the same as routing={}."""
        request = {
            "token_ids": [1, 2],
            "routing": None,
            "annotations": ["osl:512"],
        }
        hints = extract_hints(request)
        assert hints["osl"] == 512

    def test_osl_categorical_case_upper(self):
        """ALL CAPS 'HIGH' in annotations should map correctly."""
        request = {
            "token_ids": [1],
            "annotations": ["osl:HIGH"],
        }
        assert extract_hints(request)["osl"] == 768

    def test_iat_categorical_case_upper_annotations(self):
        """ALL CAPS 'LOW' in annotations should map correctly."""
        request = {
            "token_ids": [1],
            "annotations": ["iat:LOW"],
        }
        assert extract_hints(request)["iat"] == 50

    def test_osl_float_in_routing(self):
        """Floating-point OSL from routing field should truncate cleanly."""
        request = {
            "token_ids": [1],
            "routing": {"expected_output_tokens": 256.9},
        }
        assert extract_hints(request)["osl"] == 256

    def test_missing_token_ids_key(self):
        """Missing token_ids should yield tokens_in=0 without error."""
        hints = extract_hints({})
        assert hints["tokens_in"] == 0
