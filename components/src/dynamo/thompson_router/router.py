# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KvThompsonRouter -- Two-term scoring router using native KvRouter (pyo3).

Uses Dynamo's native KvRouter for KV cache state (overlap scores, load signals)
and scores workers with a clean two-term model:

    score(w) = λ₁ × ranking(w) + λ₂ × stickiness(w)

  ranking(w):    "best worker for THIS request" — cache overlap, queue availability,
                 load-aware interaction terms, optional bandit residual.
  stickiness(w): "future value of keeping this prefix here" — session context
                 (reuse budget, IAT urgency) × per-worker future cache value.

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
    "w_cache",
    "w_queue",
    "w_osl_load",
    "w_sensitivity",
    "alpha_reuse",
    "sticky_bonus",
    "stickiness_overlap_cap",
    "epsilon",
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

        # --- Ranking term weights ---
        self.w_cache = float(kt.get("w_cache", 0.55))
        self.w_queue = float(kt.get("w_queue", 0.15))
        self.w_osl_load = float(kt.get("w_osl_load", 2.0))
        self.w_sensitivity = float(kt.get("w_sensitivity", 0.10))

        # --- Stickiness term params ---
        self.alpha_reuse = float(kt.get("alpha_reuse", 0.25))
        self.sticky_bonus = float(kt.get("sticky_bonus", 0.3))
        self.stickiness_overlap_cap = float(kt.get("stickiness_overlap_cap", 0.5))

        # --- Bandit residual (disabled by default) ---
        self.epsilon = float(kt.get("epsilon", 0.05))
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
            "w_cache=%.2f, w_queue=%.2f, w_osl_load=%.2f, w_sensitivity=%.2f, "
            "alpha_reuse=%.2f, sticky_bonus=%.2f, epsilon=%.3f, "
            "worker_load_monitor=%s)",
            self.lambda_ranking, self.lambda_stickiness,
            self.w_cache, self.w_queue, self.w_osl_load, self.w_sensitivity,
            self.alpha_reuse, self.sticky_bonus, self.epsilon,
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
        worker_util = self._get_worker_utilization()

        # Decay reuse_budget: subtract requests already seen for this prefix
        requests_seen = self._prefix_request_counts.get(prefix_id, 0)
        effective_reuse_budget = max(0, reuse_budget - requests_seen)
        # Track this request
        self._prefix_request_counts[prefix_id] = requests_seen + 1

        for load_info in loads:
            wid = load_info["worker_id"]
            worker_ids.append(wid)
            loads_by_wid[wid] = load_info
            self.beta_learner.add_worker(wid)

            if indexer_overlap is not None:
                overlap = indexer_overlap.scores.get(wid, 0.0)
                cached_blocks = indexer_overlap.raw_block_counts.get(wid, 0)
                prefill_tokens = max(
                    0, tokens_in - cached_blocks * self.kv_router_block_size
                )
            else:
                raw_prefill = load_info.get("potential_prefill_tokens", 0)
                prefill_tokens = min(raw_prefill, max(1, tokens_in))
                overlap = 1.0 - prefill_tokens / max(1, tokens_in)

            util = worker_util.get(wid, {})
            kv_util = util.get("kv_util", 0.0)
            prefill_util = util.get("prefill_util", 0.0)

            # Fallback: when WorkerLoadMonitor has no useful kv_util data,
            # use the number of prefixes assigned to this worker as a load
            # proxy.  This is always accurate (we track it ourselves) and
            # provides per-worker differentiation that decode_blocks cannot
            # (decode_blocks from get_potential_loads is the same for all
            # workers — it's a per-request estimate, not per-worker state).
            if kv_util == 0.0 and prefill_util == 0.0:
                n_workers = max(1, len(loads))
                # Count how many prefixes are assigned to this worker
                prefixes_on_worker = sum(
                    1 for pw in self._prefix_workers.values() if pw == wid
                )
                # Normalize: if this worker has more than its fair share, kv_util > 0
                fair_share = max(1, len(self._prefix_workers)) / n_workers
                if fair_share > 0 and prefixes_on_worker > 0:
                    kv_util = min(1.0, prefixes_on_worker / (2.0 * fair_share))
                    self._monitor_fallback_count += 1
                    if self._monitor_fallback_count <= 5:
                        logger.warning(
                            "LOAD_FALLBACK: wid=%d no monitor data, using "
                            "prefix_count=%d (fair_share=%.1f) as kv_util=%.3f.",
                            wid, prefixes_on_worker, fair_share, kv_util,
                        )

            # Memory pressure: tree_size / total_kv_blocks (eviction risk)
            memory_pressure = 0.0
            if indexer_overlap is not None and self.worker_load_monitor is not None:
                tree_size = indexer_overlap.tree_sizes.get(wid, 0)
                if tree_size > 0:
                    try:
                        wid_state = self.worker_load_monitor.get_worker(wid)
                        if wid_state is not None:
                            total_cap = sum(
                                m.get("kv_total_blocks", 0)
                                for m in wid_state.values()
                            )
                            if total_cap > 0:
                                memory_pressure = min(1.0, tree_size / total_cap)
                    except Exception:
                        pass

            is_sticky = (last_worker is not None and wid == last_worker)

            ranking, stickiness = self._score_worker(
                wid=wid,
                overlap=overlap,
                kv_util=kv_util,
                prefill_util=prefill_util,
                memory_pressure=memory_pressure,
                osl=osl,
                iat=iat,
                latency_sensitivity=latency_sensitivity,
                is_sticky=is_sticky,
                reuse_budget=effective_reuse_budget,
                reuse_total=reuse_budget,
            )
            score = self.lambda_ranking * ranking + self.lambda_stickiness * stickiness
            raw_scores.append(score)

            worker_details.append(
                {
                    "id": wid,
                    "kv_overlap": round(overlap, 4),
                    "prefill_tokens": prefill_tokens,
                    "kv_util": round(kv_util, 4),
                    "prefill_util": round(prefill_util, 4),
                    "memory_pressure": round(memory_pressure, 4),
                    "is_sticky": is_sticky,
                    "ranking_score": round(ranking, 4),
                    "stickiness_score": round(stickiness, 4),
                    "final_score": round(score, 4),
                }
            )

        if not worker_ids:
            chosen = native_pick
        else:
            best_idx = int(np.argmax(raw_scores))
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
            "lat_sens=%.1f chosen=%d native=%d agreed=%s",
            prefix_id[:16], osl, iat, effective_reuse_budget, reuse_budget, tokens_in,
            latency_sensitivity, chosen, native_pick, chosen == native_pick,
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
        overlap: float,
        kv_util: float,
        prefill_util: float,
        memory_pressure: float,
        osl: int,
        iat: int,
        latency_sensitivity: float,
        is_sticky: bool,
        reuse_budget: int,
        reuse_total: int = 0,
    ) -> tuple[float, float]:
        """Two-term scoring: ranking (this request) + stickiness (future value).

        Returns (ranking_score, stickiness_score) as separate values so the
        caller can apply λ₁ and λ₂.
        """
        # --- Ranking term: best worker for THIS request --- #
        osl_norm = min(osl, 1024) / 1024.0
        lat_sens_norm = min(latency_sensitivity, 5.0) / 5.0

        # Base ranking from cache and queue signals
        base_ranking = (
            self.w_cache * overlap
            + self.w_queue * (1.0 - prefill_util)
            + self.w_sensitivity * (lat_sens_norm * (1.0 - kv_util))
        )

        # Load discount: exponential gate that suppresses the ranking score
        # for loaded workers.  Multiplicative so it can overcome any cache
        # advantage — a fully loaded worker with perfect overlap still gets
        # a near-zero ranking.
        #
        # osl_norm modulates: long-output requests are penalized more harshly
        # because they occupy the worker for longer.
        #
        # w_osl_load controls steepness:
        #   w_osl_load=1.0: gentle (discount=0.47 at kv_util=0.5)
        #   w_osl_load=2.0: moderate (discount=0.22 at kv_util=0.5)
        #   w_osl_load=3.0: aggressive (discount=0.11 at kv_util=0.5)
        load_discount = math.exp(
            -self.w_osl_load * kv_util * (1.0 + osl_norm)
        )
        ranking = base_ranking * load_discount

        if self.epsilon > 0:
            ranking += self.epsilon * math.tanh(self.beta_learner.sample(wid)) * load_discount

        # --- Stickiness term: future value of this prefix on this worker --- #
        # Discounted by the same load_discount as ranking, so a heavily loaded
        # worker loses both its cache advantage AND its stickiness advantage.
        # This allows prefixes to migrate away from overloaded workers.
        if reuse_budget <= 0:
            stickiness = 0.0
        else:
            iat_urgency = self._iat_factor(iat)
            # Cap the overlap contribution to stickiness so a 99%-overlap
            # worker isn't 3.7x stickier than a 27%-overlap worker.
            # The ranking term still uses raw overlap for THIS request's
            # cache benefit, but the FUTURE value is bounded — a worker
            # above the cap has "enough" cache to be worth sticking to.
            capped_overlap = min(overlap, self.stickiness_overlap_cap)
            future_value = capped_overlap * (1.0 - memory_pressure)
            # Linear decay: fraction of remaining requests in the session.
            # reuse_budget decays from reuse_total down to 0 as requests arrive.
            # This replaces tanh(α × reuse_budget) which saturated at ~1.0
            # from the very first request.
            session_weight = min(1.0, reuse_budget / max(1, reuse_total)) * iat_urgency
            stickiness = session_weight * (
                future_value + (self.sticky_bonus if is_sticky else 0.0)
            ) * load_discount

        if math.isnan(ranking) or math.isinf(ranking):
            ranking = -1e9
        if math.isnan(stickiness) or math.isinf(stickiness):
            stickiness = 0.0

        return (ranking, stickiness)

    # -------------------- Worker Load Utilities -------------------- #

    def _get_worker_utilization(self) -> dict[int, dict]:
        """Get live utilization ratios from WorkerLoadMonitor.

        Returns { worker_id: { "kv_util": float, "prefill_util": float } }
        where both values are in [0, 1].
        """
        if self.worker_load_monitor is None:
            if self._monitor_fallback_count == 0:
                logger.info(
                    "MONITOR_DISABLED: WorkerLoadMonitor is None — "
                    "ranking scores will use fallback values."
                )
            return {}

        result: dict[int, dict] = {}
        try:
            all_states = self.worker_load_monitor.get_all()
            for wid, dp_map in all_states.items():
                total_active_blocks = 0
                total_capacity_blocks = 0
                total_active_prefill = 0
                total_max_batched = 0

                for _dp_rank, metrics in dp_map.items():
                    active = metrics.get("active_decode_blocks", 0)
                    capacity = metrics.get("kv_total_blocks", 0)
                    prefill = metrics.get("active_prefill_tokens", 0)
                    max_batch = metrics.get("max_num_batched_tokens", 0)

                    total_active_blocks += active
                    total_capacity_blocks += capacity
                    total_active_prefill += prefill
                    total_max_batched += max_batch

                kv_util = (
                    total_active_blocks / total_capacity_blocks
                    if total_capacity_blocks > 0
                    else 0.0
                )
                prefill_util = (
                    total_active_prefill / total_max_batched
                    if total_max_batched > 0
                    else 0.0
                )
                result[wid] = {
                    "kv_util": min(1.0, kv_util),
                    "prefill_util": min(1.0, prefill_util),
                }
        except Exception as e:
            logger.warning(
                "MONITOR_ERROR: WorkerLoadMonitor.get_all() raised %s — "
                "all workers will use fallback values this round.",
                e,
            )

        return result

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
