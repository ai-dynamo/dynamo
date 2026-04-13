# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KvThompsonRouter -- Cost-based scoring router using native KvRouter (pyo3).

Uses Dynamo's native KvRouter for KV cache state (overlap scores, load signals)
and scores workers with a cost-based two-term model (lower is better):

    cost(w) = λ₁ × ranking_cost(w) - λ₂ × stickiness_benefit(w)

  ranking_cost(w):       "cost to serve THIS request on worker w" — prefill cost,
                         decode load, memory pressure, optional bandit residual.
  stickiness_benefit(w): "future value of keeping this prefix here" — session
                         context (reuse budget, IAT urgency) × per-worker cache value.

Load signals come from the native KvRouter's in-process tracking
(potential_decode_blocks, potential_prefill_tokens) rather than the async
WorkerLoadMonitor, ensuring accurate per-worker differentiation.

Selection uses argmin (temperature=0) or negated-softmax with temperature,
matching the native KV-aware router's convention.

The router is instantiated by a processor or standalone handler and called
in-process -- no NATS RPC.
"""

import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynamo.thompson_router.learners import BetaLearner, LatencyTracker

logger = logging.getLogger(__name__)

TUNABLE_ROUTER_PARAMS = [
    "lambda_ranking",
    "lambda_stickiness",
    "w_prefill",
    "w_decode",
    "w_mem",
    "alpha_reuse",
    "sticky_bonus",
    "stickiness_overlap_cap",
    "epsilon",
    "temperature",
]


class RouterStats:
    """Rolling statistics for router instrumentation."""

    def __init__(self, window: int = 1000) -> None:
        self.window = window
        self._count: int = 0

        # Ranking score distribution
        self._ranking_sum: float = 0.0
        self._ranking_sq_sum: float = 0.0

        # Stickiness score distribution
        self._stickiness_sum: float = 0.0
        self._stickiness_sq_sum: float = 0.0

        # Raw reward statistics
        self._reward_sum: float = 0.0
        self._reward_sq_sum: float = 0.0

        # Baseline bucket stats
        self._bucket_hits: int = 0
        self._global_fallbacks: int = 0

        # Monitor availability
        self._monitor_hits: int = 0
        self._monitor_fallbacks: int = 0

    def record_feedback(self, reward: float) -> None:
        """Record one feedback observation."""
        self._count += 1
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward

    def record_decision_scores(
        self, ranking: float, stickiness: float
    ) -> None:
        """Record per-decision ranking and stickiness scores."""
        self._ranking_sum += ranking
        self._ranking_sq_sum += ranking * ranking
        self._stickiness_sum += stickiness
        self._stickiness_sq_sum += stickiness * stickiness

    def record_baseline_lookup(self, used_bucket: bool) -> None:
        if used_bucket:
            self._bucket_hits += 1
        else:
            self._global_fallbacks += 1

    def record_monitor_lookup(self, hit: bool) -> None:
        if hit:
            self._monitor_hits += 1
        else:
            self._monitor_fallbacks += 1

    def snapshot(self) -> dict:
        """Return current stats as a JSON-serializable dict."""
        n = max(1, self._count)
        nd = max(1, self._ranking_sum != 0 and self._count or 1)
        return {
            "total_observations": self._count,
            "reward": {
                "mean": round(self._reward_sum / n, 4),
                "variance": round(
                    self._reward_sq_sum / n - (self._reward_sum / n) ** 2, 6
                ),
            },
            "ranking": {
                "mean": round(self._ranking_sum / max(1, self._count), 4),
                "variance": round(
                    self._ranking_sq_sum / max(1, self._count)
                    - (self._ranking_sum / max(1, self._count)) ** 2, 6
                ),
            },
            "stickiness": {
                "mean": round(self._stickiness_sum / max(1, self._count), 4),
                "variance": round(
                    self._stickiness_sq_sum / max(1, self._count)
                    - (self._stickiness_sum / max(1, self._count)) ** 2, 6
                ),
            },
            "baseline_buckets": {
                "bucket_hits": self._bucket_hits,
                "global_fallbacks": self._global_fallbacks,
                "bucket_hit_rate": round(
                    self._bucket_hits / max(1, self._bucket_hits + self._global_fallbacks), 4
                ),
            },
            "monitor_availability": {
                "hits": self._monitor_hits,
                "fallbacks": self._monitor_fallbacks,
                "hit_rate": round(
                    self._monitor_hits / max(1, self._monitor_hits + self._monitor_fallbacks), 4
                ),
            },
        }

    def reset(self) -> None:
        self.__init__(self.window)


@dataclass
class RoutingDecision:
    """Result of a pick_worker() call, passed back to update_feedback()."""

    chosen: int
    native_pick: int
    request_id: str = ""
    loads_by_wid: dict[int, dict] = field(default_factory=dict)
    worker_details: list[dict] = field(default_factory=list)
    prefix_id: str = ""
    osl: int = 250
    iat: int = 250
    reuse_budget: int = 0
    tokens_in: int = 0
    last_worker: int | None = None
    ranking_score: float = 0.0
    stickiness_score: float = 0.0


class KvThompsonRouter:
    """Two-term scoring router backed by native KvRouter (pyo3).

    score(w) = λ₁ × ranking(w) + λ₂ × stickiness(w)

    ranking(w):
        w_cache × overlap + w_queue × (1 - prefill_util)
        - w_osl_load × (osl_norm × kv_util)
        + w_sensitivity × (lat_sens_norm × (1 - kv_util))
        + ε × tanh(BetaSample)

    stickiness(w):
        session_weight × (future_value + is_sticky × bonus)
        where session_weight = tanh(α × reuse_budget) × iat_urgency
              future_value = overlap × (1 - memory_pressure)
    """

    def __init__(self, kv_router, config: dict | None = None, kv_indexer=None,
                 kv_block_size: int = 16, worker_load_monitor=None):
        self.kv_router = kv_router
        self.kv_indexer = kv_indexer
        self.kv_router_block_size = kv_block_size
        self.worker_load_monitor = worker_load_monitor
        cfg = config or {}

        kt = cfg.get("kv_thompson", {})

        # --- Two-term scoring weights ---
        self.lambda_ranking = float(kt.get("lambda_ranking", 1.0))
        self.lambda_stickiness = float(kt.get("lambda_stickiness", 1.0))

        # --- Ranking cost weights ---
        self.w_prefill = float(kt.get("w_prefill", 0.55))
        self.w_decode = float(kt.get("w_decode", 0.30))
        self.w_mem = float(kt.get("w_mem", 0.15))

        # --- Stickiness term params ---
        self.alpha_reuse = float(kt.get("alpha_reuse", 0.25))
        self.sticky_bonus = float(kt.get("sticky_bonus", 0.3))
        self.stickiness_overlap_cap = float(kt.get("stickiness_overlap_cap", 0.5))

        # --- Bandit residual (disabled by default) ---
        self.epsilon = float(kt.get("epsilon", 0.05))
        self.temperature = float(kt.get("temperature", 0.0))
        beta_decay = float(kt.get("beta_decay", 0.995))
        latency_ema_alpha = float(kt.get("latency_ema_alpha", 0.2))

        self.beta_learner = BetaLearner(decay=beta_decay)
        self.latency_tracker = LatencyTracker(ema_alpha=latency_ema_alpha)

        self._prefix_workers: dict[str, int] = {}
        self._prefix_request_counts: dict[str, int] = {}

        # Per-request in-flight tracking
        self._inflight_lock = threading.Lock()
        self._inflight: dict[str, dict] = {}
        self._active_prefill: dict[int, int] = {}
        self._active_blocks: dict[int, int] = {}

        # Fallback counters for WorkerLoadMonitor availability
        self._monitor_fallback_count: int = 0
        self._monitor_hit_count: int = 0

        # Rolling instrumentation stats
        self.stats = RouterStats()

        logger.info(
            "KvThompsonRouter initialized (λ_ranking=%.2f, λ_stickiness=%.2f, "
            "w_prefill=%.2f, w_decode=%.2f, w_mem=%.2f, "
            "alpha_reuse=%.2f, sticky_bonus=%.2f, epsilon=%.3f, temperature=%.3f, "
            "worker_load_monitor=%s)",
            self.lambda_ranking, self.lambda_stickiness,
            self.w_prefill, self.w_decode, self.w_mem,
            self.alpha_reuse, self.sticky_bonus, self.epsilon, self.temperature,
            "enabled" if self.worker_load_monitor is not None else "disabled",
        )

    async def pick_worker(
        self,
        token_ids: list[int],
        prefix_id: str,
        reuse_budget: int,
        osl: int,
        iat: int,
        tokens_in: int,
        latency_sensitivity: float = 2.0,
    ) -> RoutingDecision:
        """Score workers and pick the best one."""
        loads = await self.kv_router.get_potential_loads(token_ids)
        native_pick, _, _ = await self.kv_router.best_worker(token_ids)

        # Get clean per-request overlap from KvIndexer if available
        indexer_overlap = None
        if self.kv_indexer is not None:
            try:
                indexer_overlap = await self.kv_indexer.find_matches_for_request(
                    token_ids
                )
            except Exception:
                logger.debug("KvIndexer.find_matches_for_request failed, using fallback")

        worker_ids: list[int] = []
        raw_scores: list[float] = []
        loads_by_wid: dict[int, dict] = {}
        last_worker = self._prefix_workers.get(prefix_id)
        worker_details: list[dict] = []

        # Get kv_total_blocks for memory_pressure normalization
        kv_total_blocks = self._get_kv_total_blocks()
        block_size = self.kv_router_block_size or 16

        # Decay reuse_budget: subtract requests already seen for this prefix
        requests_seen = self._prefix_request_counts.get(prefix_id, 0)
        effective_reuse_budget = max(0, reuse_budget - requests_seen)
        # Track this request
        self._prefix_request_counts[prefix_id] = requests_seen + 1

        # --- Future workload aware w_prefill modulation ---
        # If this prefix will be reused, cache overlap matters more — boost w_prefill.
        # remaining_fraction decays from 1.0 (first request) to 0.0 (last request).
        # At alpha_reuse=0.5: first request gets w_prefill*1.5, last gets w_prefill*1.0.
        remaining_fraction = (
            effective_reuse_budget / max(1, reuse_budget)
            if reuse_budget > 0
            else 0.0
        )
        effective_w_prefill = self.w_prefill * (1.0 + self.alpha_reuse * remaining_fraction)

        # --- Pass 1: Compute raw kv_native_score + per-worker signals ---
        # Need all scores before normalization.
        per_worker: list[dict] = []
        for load_info in loads:
            wid = load_info["worker_id"]
            worker_ids.append(wid)
            loads_by_wid[wid] = load_info
            self.beta_learner.add_worker(wid)

            # KV-native score: matches native router formula in block units
            # potential_prefill_tokens includes active_tokens(decay) + new_tokens()
            # potential_decode_blocks includes active_blocks() + new_blocks()
            # w_prefill is modulated by reuse_budget: high-reuse prefixes weight cache more
            potential_prefill = load_info.get("potential_prefill_tokens", 0)
            potential_decode = load_info.get("potential_decode_blocks", 0)
            prefill_blocks = potential_prefill / max(1, block_size)
            kv_native_score = effective_w_prefill * prefill_blocks + self.w_decode * potential_decode

            # Overlap (for stickiness and worker_details)
            if indexer_overlap is not None:
                overlap = indexer_overlap.scores.get(wid, 0.0)
                cached_blocks = indexer_overlap.raw_block_counts.get(wid, 0)
                prefill_tokens = max(
                    0, tokens_in - cached_blocks * self.kv_router_block_size
                )
            else:
                prefill_tokens = min(potential_prefill, max(1, tokens_in))
                overlap = 1.0 - prefill_tokens / max(1, tokens_in)

            # Memory pressure: tree_size / total_kv_blocks (eviction risk)
            memory_pressure = 0.0
            if indexer_overlap is not None and kv_total_blocks > 0:
                tree_size = indexer_overlap.tree_sizes.get(wid, 0)
                if tree_size > 0:
                    memory_pressure = min(1.0, tree_size / kv_total_blocks)

            per_worker.append({
                "wid": wid,
                "kv_native_score": kv_native_score,
                "overlap": overlap,
                "prefill_tokens": prefill_tokens,
                "prefill_blocks": prefill_blocks,
                "potential_decode_blocks": potential_decode,
                "memory_pressure": memory_pressure,
            })

        # --- Normalize kv_native_score to [0, 1] across workers ---
        # This makes additive terms (epsilon, stickiness) meaningful:
        # epsilon=0.1 means "explore up to 10% of best-vs-worst gap."
        if per_worker:
            raw_scores_list = [pw["kv_native_score"] for pw in per_worker]
            score_min = min(raw_scores_list)
            score_max = max(raw_scores_list)
            score_range = score_max - score_min
            for pw in per_worker:
                pw["norm_score"] = (
                    (pw["kv_native_score"] - score_min) / score_range
                    if score_range > 0
                    else 0.0
                )

        # --- Pass 2: Score workers with normalized signals ---
        for pw in per_worker:
            wid = pw["wid"]
            is_sticky = (last_worker is not None and wid == last_worker)

            ranking_cost, stickiness_benefit = self._score_worker(
                wid=wid,
                kv_native_score=pw["norm_score"],
                memory_pressure=pw["memory_pressure"],
                osl=osl,
                iat=iat,
                is_sticky=is_sticky,
                prefill_cost=1.0 - pw["overlap"],
                reuse_budget=effective_reuse_budget,
                reuse_total=reuse_budget,
            )
            cost = self.lambda_ranking * ranking_cost - self.lambda_stickiness * stickiness_benefit
            raw_scores.append(cost)

            worker_details.append(
                {
                    "id": wid,
                    "kv_overlap": round(pw["overlap"], 4),
                    "prefill_tokens": pw["prefill_tokens"],
                    "prefill_blocks": round(pw["prefill_blocks"], 2),
                    "potential_decode_blocks": pw["potential_decode_blocks"],
                    "kv_native_score": round(pw["kv_native_score"], 2),
                    "norm_score": round(pw["norm_score"], 4),
                    "memory_pressure": round(pw["memory_pressure"], 4),
                    "is_sticky": is_sticky,
                    "ranking_score": round(ranking_cost, 4),
                    "stickiness_score": round(stickiness_benefit, 4),
                    "final_score": round(cost, 4),
                }
            )

        if not worker_ids:
            chosen = native_pick
        elif self.temperature == 0.0:
            best_idx = int(np.argmin(raw_scores))
            chosen = worker_ids[best_idx]
        else:
            # Negated softmax: lower cost → higher selection probability
            neg_scores = [-s / self.temperature for s in raw_scores]
            max_neg = max(neg_scores)
            exp_scores = [math.exp(s - max_neg) for s in neg_scores]
            total = sum(exp_scores)
            probs = [e / total for e in exp_scores]
            best_idx = int(np.random.choice(len(worker_ids), p=probs))
            chosen = worker_ids[best_idx]

        self._prefix_workers[prefix_id] = chosen

        # Record decision scores for the chosen worker
        for wd in worker_details:
            if wd["id"] == chosen:
                self.stats.record_decision_scores(
                    wd["ranking_score"], wd["stickiness_score"]
                )
                break

        # Record routing decision in KvIndexer for predict-from-decision mode
        if self.kv_indexer is not None:
            self.kv_indexer.record_routing_decision(chosen, token_ids)

        request_id = uuid.uuid4().hex[:12]

        # Per-decision logging
        logger.debug(
            "DECISION: prefix=%s osl=%d iat=%dms reuse=%d/%d tokens_in=%d "
            "chosen=%d native=%d agreed=%s",
            prefix_id[:16], osl, iat, effective_reuse_budget, reuse_budget, tokens_in,
            chosen, native_pick, chosen == native_pick,
        )

        chosen_detail = next((wd for wd in worker_details if wd["id"] == chosen), {})

        return RoutingDecision(
            chosen=chosen,
            native_pick=native_pick,
            request_id=request_id,
            loads_by_wid=loads_by_wid,
            worker_details=worker_details,
            prefix_id=prefix_id,
            osl=osl,
            iat=iat,
            reuse_budget=effective_reuse_budget,
            tokens_in=tokens_in,
            last_worker=last_worker,
            ranking_score=chosen_detail.get("ranking_score", 0.0),
            stickiness_score=chosen_detail.get("stickiness_score", 0.0),
        )

    # -------------------- Scoring -------------------- #

    def _score_worker(
        self,
        wid: int,
        kv_native_score: float,
        memory_pressure: float,
        osl: int,
        iat: int,
        is_sticky: bool,
        prefill_cost: float,
        reuse_budget: int,
        reuse_total: int = 0,
    ) -> tuple[float, float]:
        """Cost-based two-term scoring (lower is better).

        Returns (ranking_cost, stickiness_benefit) as separate values so the
        caller can compose: cost = λ₁ × ranking_cost - λ₂ × stickiness_benefit.

        The ranking cost is anchored by the KV-native router's exact formula
        (overlap_weight × prefill_blocks + decode_blocks), extended with
        memory pressure and bandit exploration.
        """
        # --- Ranking cost: KV-native score + memory pressure ---
        # kv_native_score = w_prefill × (potential_prefill_tokens / block_size)
        #                 + potential_decode_blocks
        # Both terms in block units, exactly matching the native KV router.
        # memory_pressure is in [0,1] so w_mem scales it to block-comparable units.
        ranking_cost = kv_native_score + self.w_mem * memory_pressure

        # --- Bandit residual: bounded exploration term ---
        # Subtracted from cost so positive samples reduce cost (favor worker).
        if self.epsilon > 0:
            ranking_cost -= self.epsilon * math.tanh(self.beta_learner.sample(wid))

        # --- Stickiness benefit: future value of keeping this prefix here ---
        # Positive values reduce cost when subtracted in the score composition.
        # prefill_cost is in [0,1] (1 - overlap), used here for cache affinity.
        if reuse_budget <= 0:
            stickiness_benefit = 0.0
        else:
            iat_urgency = self._iat_factor(iat)
            # Cap so a 99%-overlap worker isn't disproportionately stickier.
            capped_overlap = min(1.0 - prefill_cost, self.stickiness_overlap_cap)
            future_value = capped_overlap * (1.0 - memory_pressure)
            # Linear decay: fraction of remaining requests in the session.
            session_weight = min(1.0, reuse_budget / max(1, reuse_total)) * iat_urgency
            stickiness_benefit = session_weight * (
                future_value + (self.sticky_bonus if is_sticky else 0.0)
            )

        if math.isnan(ranking_cost) or math.isinf(ranking_cost):
            ranking_cost = 1e9
        if math.isnan(stickiness_benefit) or math.isinf(stickiness_benefit):
            stickiness_benefit = 0.0

        return (ranking_cost, stickiness_benefit)

    # -------------------- Worker Load Utilities -------------------- #

    def _get_kv_total_blocks(self) -> int:
        """Get kv_total_blocks from WorkerLoadMonitor for decode_load normalization.

        Returns the total KV block capacity for the first worker found
        (all workers in a homogeneous cluster have the same capacity).
        Falls back to a large default if the monitor is unavailable.
        """
        if self.worker_load_monitor is None:
            return 34_000  # reasonable default for 8B model on 80GB GPU

        try:
            all_states = self.worker_load_monitor.get_all()
            for _wid, dp_map in all_states.items():
                for _dp_rank, metrics in dp_map.items():
                    capacity = metrics.get("kv_total_blocks", 0)
                    if capacity > 0:
                        return capacity
        except Exception:
            pass

        return 34_000

    # -------------------- Static Helpers -------------------- #

    @staticmethod
    def _iat_factor(iat: int) -> float:
        """Interpolate IAT urgency from continuous IAT (ms).

        LOW IAT = rapid-fire arrivals = high urgency to stick (cache is warm).
        HIGH IAT = infrequent arrivals = low urgency (cache may be evicted).

        Anchor points: 50ms→1.5, 250ms→1.0, 1000ms→0.6.
        """
        if iat <= 50:
            return 1.5
        if iat <= 250:
            return 1.5 - 0.5 * (iat - 50) / (250 - 50)
        if iat >= 1000:
            return 0.6
        return 1.0 - 0.4 * (iat - 250) / (1000 - 250)

    @staticmethod
    def _osl_bin(osl: int) -> str:
        """Quantize expected output sequence length into a coarse bin."""
        if osl <= 128:
            return "S"
        if osl <= 512:
            return "M"
        return "L"

    @staticmethod
    def _prefill_bin(tokens_in: int) -> str:
        """Quantize input prompt length into a coarse bin."""
        if tokens_in <= 256:
            return "S"
        if tokens_in <= 1024:
            return "M"
        return "L"

    # -------------------- Feedback -------------------- #

    def update_feedback(
        self,
        decision: RoutingDecision,
        latency_ms: float,
        tokens_out: int,
    ) -> dict[str, Any]:
        """Update learners with observed latency reward."""
        metric, per_tok = LatencyTracker.latency_metric(latency_ms, tokens_out)
        osl_bin = self._osl_bin(decision.osl)
        prefill_bin = self._prefill_bin(decision.tokens_in)

        baseline = self.latency_tracker.get_global_bucket_baseline(
            osl_bin, prefill_bin, per_tok, fallback=metric,
        )
        used_bucket = (osl_bin, prefill_bin, per_tok) in self.latency_tracker._global_bucket
        self.stats.record_baseline_lookup(used_bucket)

        reward = LatencyTracker.compute_reward(metric, baseline, True)

        # Beta learner: raw reward (used when epsilon > 0)
        self.beta_learner.update(decision.chosen, reward)

        # Record stats
        self.stats.record_feedback(reward)

        # Update baselines
        self.latency_tracker.update_baselines(
            decision.chosen, osl_bin, prefill_bin, metric, per_tok
        )

        beta_alpha, beta_beta = self.beta_learner.get_params(decision.chosen)

        logger.debug(
            "Feedback: wid=%s metric=%.2f baseline=%.2f reward=%.3f "
            "ranking=%.3f stickiness=%.3f tokens_out=%d",
            decision.chosen,
            metric,
            baseline,
            reward,
            decision.ranking_score,
            decision.stickiness_score,
            tokens_out,
        )

        return {
            "metric": metric,
            "baseline_ema": baseline,
            "reward": reward,
            "ranking_score": decision.ranking_score,
            "stickiness_score": decision.stickiness_score,
            "beta_after": {
                "alpha": round(beta_alpha, 4),
                "beta": round(beta_beta, 4),
            },
        }
