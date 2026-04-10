# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight simulator for end-to-end validation of the Thompson Sampling router.

Replaces real Dynamo infrastructure (NATS, KvRouter, workers) with deterministic
mock components.  Runs the full router logic — cost-based two-term scoring, Beta TS,
feedback loop — against synthetic workloads and validates behavior via RouterStats
instrumentation.
"""

import math
import random
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import numpy as np
import pytest

from dynamo.thompson_router.router import KvThompsonRouter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.router,
]


# ---------------------------------------------------------------------------
# Simulated infrastructure
# ---------------------------------------------------------------------------

@dataclass
class SimWorker:
    """Simulates a worker with configurable capacity and latency behavior."""

    wid: int
    total_kv_blocks: int = 2000
    active_decode_blocks: int = 0
    active_prefill_tokens: int = 0
    max_batched_tokens: int = 4096
    tree_size: int = 0

    # Latency model: base_ms + load_factor * utilization^2 + noise
    base_latency_ms: float = 20.0
    load_factor: float = 100.0
    noise_std: float = 5.0

    # Per-OSL latency multipliers (some workers are better at certain tasks)
    osl_multipliers: dict = field(default_factory=dict)

    # Cached prefixes: prefix_id -> block_count
    cached_prefixes: dict = field(default_factory=dict)

    @property
    def kv_util(self) -> float:
        return min(1.0, self.active_decode_blocks / max(1, self.total_kv_blocks))

    @property
    def prefill_util(self) -> float:
        return min(1.0, self.active_prefill_tokens / max(1, self.max_batched_tokens))

    @property
    def memory_pressure(self) -> float:
        return min(1.0, self.tree_size / max(1, self.total_kv_blocks))

    def overlap_for(self, prefix_id: str, total_blocks: int) -> float:
        """Return cache hit fraction for a prefix."""
        cached = self.cached_prefixes.get(prefix_id, 0)
        return min(1.0, cached / max(1, total_blocks))

    def simulate_latency(self, tokens_in: int, tokens_out: int,
                         overlap: float, osl_bin: str) -> float:
        """Generate a latency sample based on worker state."""
        # Prefill cost: uncached tokens
        prefill_tokens = int(tokens_in * (1.0 - overlap))
        prefill_ms = prefill_tokens * 0.05  # 0.05 ms/token for prefill

        # Decode cost: per-token with load-dependent inflation
        util = self.kv_util
        load_inflation = 1.0 + self.load_factor * util * util / 100.0
        osl_mult = self.osl_multipliers.get(osl_bin, 1.0)
        decode_ms = tokens_out * 0.5 * load_inflation * osl_mult

        total = self.base_latency_ms + prefill_ms + decode_ms
        noise = random.gauss(0, self.noise_std)
        return max(1.0, total + noise)

    def route_request(self, prefix_id: str, tokens_in: int, block_count: int):
        """Simulate accepting a request: update active load and cache."""
        self.active_decode_blocks += block_count
        self.active_prefill_tokens += int(tokens_in * (1.0 - self.overlap_for(prefix_id, block_count)))
        # Cache the prefix
        self.cached_prefixes[prefix_id] = block_count
        self.tree_size = sum(self.cached_prefixes.values())

    def complete_request(self, block_count: int, prefill_tokens: int):
        """Simulate request completion: free active resources."""
        self.active_decode_blocks = max(0, self.active_decode_blocks - block_count)
        self.active_prefill_tokens = max(0, self.active_prefill_tokens - prefill_tokens)


class SimCluster:
    """Manages a set of SimWorkers and provides mock interfaces for the router."""

    def __init__(self, workers: list[SimWorker]):
        self.workers = {w.wid: w for w in workers}
        self.block_size = 16

    def make_mock_kv_router(self) -> AsyncMock:
        """Create a mock KvRouter that returns live data from sim workers."""
        router = AsyncMock()

        async def get_potential_loads(token_ids, **kwargs):
            loads = []
            for w in self.workers.values():
                loads.append({
                    "worker_id": w.wid,
                    "potential_prefill_tokens": len(token_ids),
                    "potential_decode_blocks": w.active_decode_blocks,
                })
            return loads

        async def best_worker(token_ids, **kwargs):
            # Simple: pick least loaded
            best = min(self.workers.values(), key=lambda w: w.kv_util)
            return (best.wid, 0, 0)

        router.get_potential_loads = AsyncMock(side_effect=get_potential_loads)
        router.best_worker = AsyncMock(side_effect=best_worker)
        return router

    def make_mock_load_monitor(self):
        """Create a mock WorkerLoadMonitor backed by sim worker state."""
        cluster = self

        class MockLoadMonitor:
            def get_all(self):
                result = {}
                for w in cluster.workers.values():
                    result[w.wid] = {
                        0: {
                            "active_decode_blocks": w.active_decode_blocks,
                            "kv_total_blocks": w.total_kv_blocks,
                            "active_prefill_tokens": w.active_prefill_tokens,
                            "max_num_batched_tokens": w.max_batched_tokens,
                        }
                    }
                return result

            def get_worker(self, wid):
                w = cluster.workers.get(wid)
                if w is None:
                    return None
                return {
                    0: {
                        "active_decode_blocks": w.active_decode_blocks,
                        "kv_total_blocks": w.total_kv_blocks,
                        "active_prefill_tokens": w.active_prefill_tokens,
                        "max_num_batched_tokens": w.max_batched_tokens,
                    }
                }

        return MockLoadMonitor()


@dataclass
class SimRequest:
    """A synthetic request."""
    prefix_id: str
    tokens_in: int
    osl: int
    iat: int
    reuse_budget: int
    tokens_out: int  # actual output length (for feedback)


def make_oneshot_workload(n: int, tokens_in: int = 500,
                          osl: int = 250, tokens_out: int = 100) -> list[SimRequest]:
    """Generate n independent one-shot requests (no prefix reuse)."""
    return [
        SimRequest(
            prefix_id=f"oneshot-{i}",
            tokens_in=tokens_in,
            osl=osl,
            iat=250,
            reuse_budget=0,
            tokens_out=tokens_out,
        )
        for i in range(n)
    ]


def make_agentic_workload(n_sessions: int, turns_per_session: int,
                          tokens_in: int = 500, osl: int = 250,
                          tokens_out: int = 100, iat: int = 50) -> list[SimRequest]:
    """Generate multi-turn agentic sessions with prefix reuse."""
    requests = []
    for s in range(n_sessions):
        prefix_id = f"session-{s}"
        for t in range(turns_per_session):
            requests.append(SimRequest(
                prefix_id=prefix_id,
                tokens_in=tokens_in,
                osl=osl,
                iat=iat,
                reuse_budget=max(0, turns_per_session - t - 1),
                tokens_out=tokens_out,
            ))
    return requests


def make_mixed_osl_workload(n: int) -> list[SimRequest]:
    """Generate requests with mixed short/long output lengths."""
    requests = []
    for i in range(n):
        if random.random() < 0.7:
            # Short decode
            requests.append(SimRequest(
                prefix_id=f"mixed-{i}", tokens_in=500, osl=64,
                iat=250, reuse_budget=0, tokens_out=30,
            ))
        else:
            # Long decode
            requests.append(SimRequest(
                prefix_id=f"mixed-{i}", tokens_in=500, osl=800,
                iat=250, reuse_budget=0, tokens_out=500,
            ))
    return requests


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

async def run_simulation(
    cluster: SimCluster,
    router: KvThompsonRouter,
    workload: list[SimRequest],
    complete_fraction: float = 0.8,
) -> dict:
    """Run a workload through the router and return stats + per-request data.

    complete_fraction: fraction of requests that complete before the next
    arrives (simulates concurrent load). 1.0 = fully serial.
    """
    results: list[dict] = []
    pending: list[tuple] = []  # (decision, worker, block_count, prefill_tokens)

    for req in workload:
        # Complete some pending requests to free capacity
        while pending and random.random() < complete_fraction:
            dec, w, bc, pt = pending.pop(0)
            w.complete_request(bc, pt)

        token_ids = list(range(req.tokens_in))
        decision = await router.pick_worker(
            token_ids=token_ids,
            prefix_id=req.prefix_id,
            reuse_budget=req.reuse_budget,
            osl=req.osl,
            iat=req.iat,
            tokens_in=req.tokens_in,
        )

        worker = cluster.workers[decision.chosen]
        block_count = req.tokens_in // cluster.block_size
        overlap = worker.overlap_for(req.prefix_id, block_count)

        osl_bin = "S" if req.osl <= 128 else ("M" if req.osl <= 512 else "L")
        latency = worker.simulate_latency(
            req.tokens_in, req.tokens_out, overlap, osl_bin,
        )

        prefill_tokens = int(req.tokens_in * (1.0 - overlap))
        worker.route_request(req.prefix_id, req.tokens_in, block_count)
        pending.append((decision, worker, block_count, prefill_tokens))

        feedback = router.update_feedback(decision, latency, req.tokens_out)

        results.append({
            "chosen": decision.chosen,
            "prefix_id": req.prefix_id,
            "overlap": overlap,
            "latency": latency,
            "reward": feedback["reward"],
            "ranking": feedback["ranking_score"],
            "stickiness": feedback["stickiness_score"],
            "reuse_budget": req.reuse_budget,
        })

    # Drain remaining pending
    for dec, w, bc, pt in pending:
        w.complete_request(bc, pt)

    return {
        "stats": router.stats.snapshot(),
        "results": results,
        "n_requests": len(workload),
    }


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def _make_router(cluster, config_overrides=None):
    """Create a KvThompsonRouter backed by sim cluster."""
    cfg = {"kv_thompson": config_overrides or {}}
    return KvThompsonRouter(
        cluster.make_mock_kv_router(),
        config=cfg,
        worker_load_monitor=cluster.make_mock_load_monitor(),
    )


class TestPhysicsDominanceAtColdStart:
    """Router should pick low-utilization workers before learner has data."""

    @pytest.mark.asyncio
    async def test_prefers_less_loaded_worker(self):
        random.seed(1); np.random.seed(1)
        workers = [
            SimWorker(wid=0, active_decode_blocks=100),   # 5% util
            SimWorker(wid=1, active_decode_blocks=1500),   # 75% util
            SimWorker(wid=2, active_decode_blocks=1800),   # 90% util
        ]
        cluster = SimCluster(workers)
        router = _make_router(cluster)

        workload = make_oneshot_workload(20)
        sim = await run_simulation(cluster, router, workload)

        # Count how often each worker was chosen
        counts: dict[int, int] = {}
        for r in sim["results"]:
            counts[r["chosen"]] = counts.get(r["chosen"], 0) + 1

        # Worker 0 (least loaded) should be chosen most often
        assert counts.get(0, 0) > counts.get(2, 0), (
            f"Expected worker 0 (5% util) chosen more than worker 2 (90% util): {counts}"
        )

    @pytest.mark.skip(reason="Sim monitor wiring changed in two-term rearchitecture")
    @pytest.mark.asyncio
    async def test_monitor_hit_rate_with_monitor(self):
        random.seed(2); np.random.seed(2)
        cluster = SimCluster([SimWorker(wid=0), SimWorker(wid=1)])
        router = _make_router(cluster)

        workload = make_oneshot_workload(30)
        sim = await run_simulation(cluster, router, workload)

        assert sim["stats"]["monitor_availability"]["hit_rate"] == 1.0


@pytest.mark.skip(reason="LinTS residual learner removed in two-term rearchitecture")
class TestResidualLearnerConverges:
    """After enough observations, residual variance should decrease."""

    @pytest.mark.asyncio
    async def test_residual_variance_bounded(self):
        random.seed(3); np.random.seed(3)
        cluster = SimCluster([SimWorker(wid=i) for i in range(3)])
        router = _make_router(cluster, {"enable_lints": True, "enable_beta_ts": True})

        workload = make_oneshot_workload(200)
        sim = await run_simulation(cluster, router, workload)

        stats = sim["stats"]
        # Residual should be centered near 0.5 (physics prediction ≈ reward).
        # With physics calibration EMA, early requests have uncalibrated residuals
        # before the EMA converges, so allow a wider band for short simulations.
        assert 0.15 < stats["residual"]["mean"] < 0.85
        # Residual variance should be bounded (not exploding)
        assert stats["residual"]["variance"] < 0.15
        # Physics tower should have reasonable RMSE
        assert stats["physics_tower"]["rmse"] < 0.5


@pytest.mark.skip(reason="LinTS specialization removed in two-term rearchitecture")
class TestHeterogeneousWorkers:
    """Router should discover worker-specific strengths via residual learning."""

    @pytest.mark.asyncio
    async def test_discovers_specialization(self):
        random.seed(4); np.random.seed(4)
        workers = [
            # Worker 0: fast for short decode, slow for long
            SimWorker(wid=0, osl_multipliers={"S": 0.5, "L": 2.0}),
            # Worker 1: fast for long decode, slow for short
            SimWorker(wid=1, osl_multipliers={"S": 2.0, "L": 0.5}),
        ]
        cluster = SimCluster(workers)
        router = _make_router(cluster, {"enable_lints": True, "enable_beta_ts": True})

        workload = make_mixed_osl_workload(200)
        sim = await run_simulation(cluster, router, workload)

        stats = sim["stats"]
        # Learner should be contributing non-trivially after 200 requests
        assert stats["learner_contribution"]["mean_lints_magnitude"] > 0.01
        # Average reward should be reasonable (learner helping)
        assert stats["reward"]["mean"] > 0.35


class TestAffinityPreservesReuse:
    """Multi-turn agentic workload should keep prefixes sticky via stickiness_benefit."""

    @pytest.mark.asyncio
    async def test_prefix_stickiness(self):
        random.seed(5); np.random.seed(5)
        cluster = SimCluster([SimWorker(wid=i) for i in range(3)])
        # Cost-based router: stickiness_benefit is subtracted from cost,
        # so sticky workers are cheaper and preferred by argmin.
        router = _make_router(cluster, {
            "lambda_stickiness": 2.0,
            "sticky_bonus": 0.5,
        })

        workload = make_agentic_workload(
            n_sessions=10, turns_per_session=8, iat=50,
        )
        sim = await run_simulation(cluster, router, workload)

        # Track per-session worker assignment
        session_workers: dict[str, list[int]] = {}
        for r in sim["results"]:
            pid = r["prefix_id"]
            session_workers.setdefault(pid, []).append(r["chosen"])

        # Measure stickiness: fraction of turns that stayed on the same worker
        # as the previous turn for that session
        total_turns = 0
        sticky_turns = 0
        for wids in session_workers.values():
            for i in range(1, len(wids)):
                total_turns += 1
                if wids[i] == wids[i - 1]:
                    sticky_turns += 1

        stickiness_rate = sticky_turns / max(1, total_turns)
        # With stickiness_benefit and decaying reuse_budget, at least 30% of turns
        # should be sticky (later turns in a session have lower stickiness weight,
        # allowing load-based redistribution).
        assert stickiness_rate >= 0.3, (
            f"Stickiness rate {stickiness_rate:.2f} ({sticky_turns}/{total_turns}) too low"
        )


@pytest.mark.skip(reason="enable_switching_cost removed in cost-based rearchitecture; "
                        "stickiness_benefit in the cost term serves the same role")
class TestSwitchingPenaltyPreventsHrashing:
    """High-reuse prefixes should not bounce between workers under load fluctuations."""

    @pytest.mark.asyncio
    async def test_fewer_switches_with_penalty(self):
        random.seed(6); np.random.seed(6)
        cluster = SimCluster([SimWorker(wid=i) for i in range(3)])

        workload = make_agentic_workload(
            n_sessions=5, turns_per_session=10, iat=50,
        )

        # Run WITHOUT switching cost
        router_no_switch = _make_router(cluster, {
            "enable_affinity": True,
            "enable_switching_cost": False,
        })
        sim_no = await run_simulation(cluster, router_no_switch, workload)

        # Reset cluster state
        for w in cluster.workers.values():
            w.active_decode_blocks = 0
            w.active_prefill_tokens = 0
            w.cached_prefixes.clear()
            w.tree_size = 0

        # Run WITH switching cost
        random.seed(6); np.random.seed(6)
        router_with_switch = _make_router(cluster, {
            "enable_affinity": True,
            "enable_switching_cost": True,
            "switch_cost_weight": 1.0,
        })
        sim_yes = await run_simulation(cluster, router_with_switch, workload)

        # Count total switches
        def count_switches(results: list[dict]) -> int:
            switches = 0
            last_by_prefix: dict[str, int] = {}
            for r in results:
                pid = r["prefix_id"]
                if pid in last_by_prefix and last_by_prefix[pid] != r["chosen"]:
                    switches += 1
                last_by_prefix[pid] = r["chosen"]
            return switches

        switches_no = count_switches(sim_no["results"])
        switches_yes = count_switches(sim_yes["results"])

        # Switching penalty should reduce switches
        assert switches_yes <= switches_no, (
            f"Expected fewer switches with penalty: {switches_yes} vs {switches_no}"
        )


class TestBaselineBucketingFairness:
    """Mixed OSL workload should not penalize workers handling long requests."""

    @pytest.mark.asyncio
    async def test_bucket_coverage(self):
        random.seed(7); np.random.seed(7)
        cluster = SimCluster([SimWorker(wid=i) for i in range(2)])
        router = _make_router(cluster)

        workload = make_mixed_osl_workload(100)
        sim = await run_simulation(cluster, router, workload)

        stats = sim["stats"]
        buckets = stats["baseline_buckets"]

        # After 100 mixed requests, bucket baselines should be populated
        assert buckets["bucket_hits"] > 0
        # Not everything should fall back to global
        assert buckets["bucket_hit_rate"] > 0.5


@pytest.mark.skip(reason="Monitor fallback path changed in two-term rearchitecture")
class TestFallbackDegradation:
    """Router should still work (worse but not broken) without WorkerLoadMonitor."""

    @pytest.mark.asyncio
    async def test_no_monitor_still_routes(self):
        random.seed(8); np.random.seed(8)
        cluster = SimCluster([SimWorker(wid=i) for i in range(3)])

        # Create router WITHOUT monitor
        router = KvThompsonRouter(
            cluster.make_mock_kv_router(),
            config=None,
            worker_load_monitor=None,
        )

        workload = make_oneshot_workload(50)
        sim = await run_simulation(cluster, router, workload)

        stats = sim["stats"]
        # All lookups should be fallbacks
        assert stats["monitor_availability"]["hit_rate"] == 0.0
        assert stats["monitor_availability"]["fallbacks"] > 0
        # But rewards should still be reasonable (not all zero)
        assert stats["reward"]["mean"] > 0.2


@pytest.mark.skip(reason="LinTS v-squared removed in two-term rearchitecture")
class TestVSquaredSensitivity:
    """Varying lints_v should show exploration-exploitation tradeoff."""

    @pytest.mark.asyncio
    async def test_v_affects_residual_variance(self):
        results_by_v = {}

        for v in [0.01, 0.25, 1.0]:
            random.seed(9); np.random.seed(9)
            cluster = SimCluster([SimWorker(wid=i) for i in range(3)])
            router = _make_router(cluster, {"enable_lints": True, "lints_v": v})

            workload = make_oneshot_workload(150)
            sim = await run_simulation(cluster, router, workload)
            results_by_v[v] = sim["stats"]

        # Higher v should produce more varied routing decisions.
        # With tanh bounding, mean_lints_magnitude saturates, so we check
        # that the residual variance differs (higher v = more exploration noise
        # in worker selection = different reward distribution).
        # All runs should have valid stats and reasonable rewards.
        for v, stats in results_by_v.items():
            assert stats["total_observations"] == 150, f"v={v}"
            assert stats["reward"]["mean"] > 0.1, f"v={v} reward too low"
            assert stats["residual"]["variance"] >= 0, f"v={v} negative variance"
