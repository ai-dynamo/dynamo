# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
KvThompsonRouter -- In-process Thompson Sampling router using native KvRouter (pyo3).

Uses Dynamo's native KvRouter for KV cache state (overlap scores, load signals)
and applies Thompson Sampling (Beta bandits + LinTS contextual bandits) on top
for learning-based worker selection.

All scoring features are independently togglable via config.
The router is instantiated by a processor or standalone handler and called
in-process -- no NATS RPC.
"""

import logging
import math
import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynamo.thompson_router.learners import BetaLearner, LatencyTracker, LinTSLearner

logger = logging.getLogger(__name__)

TUNABLE_ROUTER_PARAMS = [
    "ts_weight",
    "temperature",
    "cold_start_threshold",
    "idle_boost",
    "beta_decay",
    "lints_v",
    "lints_forget_rate",
    "queue_penalty_weight",
    "lints_weight",
    "switch_cost_weight",
    "physics_cache_weight",
    "physics_compute_weight",
    "physics_queue_weight",
    "physics_memory_weight",
]


class RouterStats:
    """Rolling statistics for router instrumentation.

    Tracks physics tower accuracy, residual distribution, learner contribution,
    fallback rates, and baseline bucket coverage. All updates are O(1) using
    Welford's online algorithm for mean/variance.
    """

    def __init__(self, window: int = 1000) -> None:
        self.window = window
        self._count: int = 0

        # Physics tower accuracy: tracks (reward, physics_pred) pairs
        self._physics_err_sum: float = 0.0    # sum of (reward - physics)
        self._physics_err_sq_sum: float = 0.0  # sum of (reward - physics)^2

        # Residual statistics: what LinTS trains on
        self._residual_sum: float = 0.0
        self._residual_sq_sum: float = 0.0

        # Raw reward statistics
        self._reward_sum: float = 0.0
        self._reward_sq_sum: float = 0.0

        # Learner contribution: |lints_score| per decision
        self._lints_contrib_sum: float = 0.0
        self._beta_contrib_sum: float = 0.0

        # Physics score distribution
        self._physics_sum: float = 0.0
        self._physics_sq_sum: float = 0.0

        # Baseline bucket stats
        self._bucket_hits: int = 0
        self._global_fallbacks: int = 0

        # Monitor availability
        self._monitor_hits: int = 0
        self._monitor_fallbacks: int = 0

    def record_feedback(
        self, reward: float, physics_pred: float, residual: float
    ) -> None:
        """Record one feedback observation."""
        self._count += 1
        err = reward - physics_pred
        self._physics_err_sum += err
        self._physics_err_sq_sum += err * err
        self._residual_sum += residual
        self._residual_sq_sum += residual * residual
        self._reward_sum += reward
        self._reward_sq_sum += reward * reward
        self._physics_sum += physics_pred
        self._physics_sq_sum += physics_pred * physics_pred

    def record_learner_contribution(
        self, lints_contrib: float, beta_contrib: float
    ) -> None:
        """Record per-decision learner score magnitudes."""
        self._lints_contrib_sum += abs(lints_contrib)
        self._beta_contrib_sum += abs(beta_contrib)

    def record_baseline_lookup(self, used_bucket: bool) -> None:
        """Record whether reward used a bucket baseline or fell back to global."""
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
        return {
            "total_observations": self._count,
            "reward": {
                "mean": round(self._reward_sum / n, 4),
                "variance": round(
                    self._reward_sq_sum / n - (self._reward_sum / n) ** 2, 6
                ),
            },
            "physics_tower": {
                "mean_score": round(self._physics_sum / n, 4),
                "mean_error": round(self._physics_err_sum / n, 4),
                "mse": round(self._physics_err_sq_sum / n, 6),
                "rmse": round((self._physics_err_sq_sum / n) ** 0.5, 4),
            },
            "residual": {
                "mean": round(self._residual_sum / n, 4),
                "variance": round(
                    self._residual_sq_sum / n - (self._residual_sum / n) ** 2, 6
                ),
            },
            "learner_contribution": {
                "mean_lints_magnitude": round(self._lints_contrib_sum / n, 4),
                "mean_beta_magnitude": round(self._beta_contrib_sum / n, 4),
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
    features: np.ndarray | None = None
    loads_by_wid: dict[int, dict] = field(default_factory=dict)
    worker_details: list[dict] = field(default_factory=list)
    prefix_id: str = ""
    osl: int = 250
    iat: int = 250
    reuse_budget: int = 0
    tokens_in: int = 0
    last_worker: int | None = None
    physics_score: float = 0.5  # Physics tower prediction at decision time


class KvThompsonRouter:
    """In-process Thompson Sampling router backed by native KvRouter (pyo3).

    Modular scoring with independently togglable features:

      Base scoring (always on):
        effective_overlap = max(overlap, idle_boost)     [if enable_idle_boost]
        load_mod = exp(-qpw * decode_blocks^2 / 2500)   [exponential queue proxy]
        score = physics_score * load_mod

      Optional features (each independently togglable):
        enable_beta_ts         -- Beta-TS global exploration (context-free worker quality)
        enable_lints           -- LinTS contextual bandit (7-dim feature-aware)
        enable_affinity        -- prefix stickiness for multi-turn sessions
        enable_switching_cost  -- penalty for migrating prefix to different worker
        enable_adaptive_temp   -- temperature decays with session depth
        enable_adaptive_explore -- Beta-TS weight decays with session depth
        enable_sticky_floor    -- protect sticky worker's load_mod minimum

      Selection:
        enable_softmax=true  -> softmax(scores, temperature) -> probabilistic pick
        enable_softmax=false -> deterministic argmax

      Cold start:
        enable_cold_start=true -> round-robin when max overlap < threshold

    Feature vector (7 dims, all in [0,1]):
        [bias, cache_hit, compute_avail, affinity, osl_norm, reuse_norm,
         iat_norm]
    """

    def __init__(self, kv_router, config: dict | None = None, kv_indexer=None,
                 kv_block_size: int = 16, worker_load_monitor=None):
        self.kv_router = kv_router
        self.kv_indexer = kv_indexer  # Optional KvIndexer for clean per-request overlap
        self.kv_router_block_size = kv_block_size
        self.worker_load_monitor = worker_load_monitor  # Optional WorkerLoadMonitor for live utilization
        cfg = config or {}

        kt = cfg.get("kv_thompson", {})

        beta_decay = float(kt.get("beta_decay", 0.995))
        lints_lambda = float(kt.get("lints_lambda", 1.0))
        lints_v = float(kt.get("lints_v", 0.25))
        lints_forget = float(kt.get("lints_forget_rate", 0.995))
        latency_ema_alpha = float(kt.get("latency_ema_alpha", 0.2))

        self.ts_weight = float(kt.get("ts_weight", 0.05))
        self.idle_boost = float(kt.get("idle_boost", 0.135))
        self.temperature = float(kt.get("temperature", 1.70))
        self.cold_start_threshold = float(kt.get("cold_start_threshold", 0.37))
        self.queue_penalty_weight = float(kt.get("queue_penalty_weight", 2.5))
        self.load_mod_floor = float(kt.get("load_mod_floor", 0.3))

        self.temp_min = float(kt.get("temp_min", 0.15))
        self.temp_max = float(kt.get("temp_max", 2.0))

        self.enable_softmax = bool(kt.get("enable_softmax", False))
        self.enable_cold_start = bool(kt.get("enable_cold_start", False))
        self.enable_idle_boost = bool(kt.get("enable_idle_boost", False))
        self.enable_load_mod_floor = bool(kt.get("enable_load_mod_floor", False))
        self.enable_lints = bool(kt.get("enable_lints", False))
        self.enable_affinity = bool(kt.get("enable_affinity", False))
        self.enable_switching_cost = bool(kt.get("enable_switching_cost", False))
        self.enable_adaptive_temp = bool(kt.get("enable_adaptive_temp", False))
        self.enable_adaptive_explore = bool(kt.get("enable_adaptive_explore", False))
        self.enable_sticky_floor = bool(kt.get("enable_sticky_floor", False))
        self.enable_beta_ts = bool(kt.get("enable_beta_ts", False))
        self.enable_adaptive_v = bool(kt.get("enable_adaptive_v", False))

        # Adaptive v: EMA tracker for residual variance.
        # When enable_adaptive_v is True, lints_learner.v is updated on each
        # feedback to sqrt(residual_ema_var), clamped to [v_min, v_max].
        self._residual_ema_mean: float = 0.5
        self._residual_ema_var: float = 0.0625  # initial guess (0.25^2)
        self._adaptive_v_alpha: float = float(kt.get("adaptive_v_alpha", 0.05))
        self._adaptive_v_min: float = float(kt.get("adaptive_v_min", 0.02))
        self._adaptive_v_max: float = float(kt.get("adaptive_v_max", 0.5))

        # Physics calibration EMA: tracks running mean and range of physics
        # scores so the residual computation can center around 0.5.
        # Without this, when utilization signals are near-constant (low
        # concurrency), physics scores are inflated and the residual is
        # systematically biased, causing LinTS to learn anti-caching.
        self._physics_ema_mean: float = 0.5
        self._physics_ema_sq: float = 0.25  # for variance
        self._physics_cal_alpha: float = float(kt.get("physics_cal_alpha", 0.02))

        self.lints_weight = float(kt.get("lints_weight", 1.0))
        self.affinity_base = float(kt.get("affinity_base", 0.5))
        self.affinity_reuse_weight = float(kt.get("affinity_reuse_weight", 0.13))
        self.switch_base = float(kt.get("switch_base", 0.2))
        self.switch_reuse = float(kt.get("switch_reuse", 0.12))
        self.switch_cost_weight = float(kt.get("switch_cost_weight", 1.0))
        self.sticky_load_floor = float(kt.get("sticky_load_floor", 0.01))
        self.adaptive_temp_base = float(kt.get("adaptive_temp_base", 1.0))

        # Physics tower weights (all signals in [0,1])
        self.physics_cache_weight = float(kt.get("physics_cache_weight", 0.35))
        self.physics_compute_weight = float(kt.get("physics_compute_weight", 0.30))
        self.physics_queue_weight = float(kt.get("physics_queue_weight", 0.20))
        self.physics_memory_weight = float(kt.get("physics_memory_weight", 0.15))

        self.feature_dim = 9
        self.beta_learner = BetaLearner(decay=beta_decay)
        self.lints_learner = LinTSLearner(
            feature_dim=self.feature_dim,
            lambda_=lints_lambda,
            v=lints_v,
            forget_rate=lints_forget,
        )
        self.latency_tracker = LatencyTracker(ema_alpha=latency_ema_alpha)

        self._prefix_workers: dict[str, int] = {}
        self._cold_start_rr: int = 0

        # Per-worker selection pressure: EMA of how often each worker is picked.
        # High pressure = worker picked too often → congestion risk (herding).
        self._selection_ema: dict[int, float] = {}
        self._selection_ema_alpha: float = 0.05  # ~20-request window

        # Per-request in-flight tracking for isolating per-request load signals.
        # Dynamo's get_potential_loads() returns cumulative values (this request +
        # all active requests). We track what we've sent to each worker so we can
        # subtract the active baseline and recover per-request values.
        self._inflight_lock = threading.Lock()
        self._inflight: dict[str, dict] = {}  # request_id -> {wid, prefill, blocks}
        self._active_prefill: dict[int, int] = {}  # wid -> total active prefill tokens
        self._active_blocks: dict[int, int] = {}  # wid -> total active decode blocks

        # Fallback counters for monitoring WorkerLoadMonitor availability
        self._monitor_fallback_count: int = 0
        self._monitor_hit_count: int = 0

        # Rolling instrumentation stats
        self.stats = RouterStats()

        features_on = [
            name
            for name, enabled in [
                ("softmax", self.enable_softmax),
                ("cold_start", self.enable_cold_start),
                ("idle_boost", self.enable_idle_boost),
                ("load_mod_floor", self.enable_load_mod_floor),
                ("lints", self.enable_lints),
                ("affinity", self.enable_affinity),
                ("switching_cost", self.enable_switching_cost),
                ("adaptive_temp", self.enable_adaptive_temp),
                ("adaptive_explore", self.enable_adaptive_explore),
                ("sticky_floor", self.enable_sticky_floor),
                ("beta_ts", self.enable_beta_ts),
                ("adaptive_v", self.enable_adaptive_v),
            ]
            if enabled
        ]
        logger.info(
            "KvThompsonRouter initialized (feature_dim=%d, beta_decay=%.3f, "
            "lints_v=%.3f, ts_weight=%.3f, features=[%s], "
            "worker_load_monitor=%s)",
            self.feature_dim,
            beta_decay,
            lints_v,
            self.ts_weight,
            ", ".join(features_on) if features_on else "base only",
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
    ) -> RoutingDecision:
        """Score workers and pick the best one.

        When a KvIndexer is available, uses its RadixTree for clean per-request
        overlap (no cumulative active-sequence noise). Falls back to
        get_potential_loads() otherwise.
        """
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
        all_overlaps: dict[int, float] = {}
        loads_by_wid: dict[int, dict] = {}
        features_by_wid: dict[int, np.ndarray] = {}
        physics_by_wid: dict[int, float] = {}
        last_worker = self._prefix_workers.get(prefix_id)
        worker_details: list[dict] = []
        worker_util = self._get_worker_utilization()

        iat_factor = self._iat_factor(iat)

        # --- Pass 1: collect per-worker raw signals --- #
        per_worker: dict[int, dict] = {}
        for load_info in loads:
            wid = load_info["worker_id"]
            worker_ids.append(wid)
            loads_by_wid[wid] = load_info
            self.beta_learner.add_worker(wid)
            self.lints_learner.add_worker(wid)

            if indexer_overlap is not None:
                overlap = indexer_overlap.scores.get(wid, 0.0)
                cached_blocks = indexer_overlap.raw_block_counts.get(wid, 0)
                prefill_tokens = max(
                    0, tokens_in - cached_blocks * self.kv_router_block_size
                )
                decode_blocks = load_info.get("potential_decode_blocks", 0)
            else:
                raw_prefill = load_info.get("potential_prefill_tokens", 0)
                prefill_tokens = min(raw_prefill, max(1, tokens_in))
                decode_blocks = load_info.get("potential_decode_blocks", 0)
                overlap = 1.0 - prefill_tokens / max(1, tokens_in)

            all_overlaps[wid] = overlap

            util = worker_util.get(wid, {})
            wid_kv_util = util.get("kv_util", 0.0)
            wid_prefill_util = util.get("prefill_util", 0.0)

            wid_memory_pressure = 0.0
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
                                wid_memory_pressure = min(
                                    1.0, tree_size / total_cap
                                )
                    except Exception:
                        pass

            per_worker[wid] = {
                "overlap": overlap,
                "prefill_tokens": prefill_tokens,
                "decode_blocks": decode_blocks,
                "kv_util": wid_kv_util,
                "prefill_util": wid_prefill_util,
                "memory_pressure": wid_memory_pressure,
            }

        # --- Compute ranks and shared signals for feature vector --- #
        n_workers = max(1, len(worker_ids))
        rank_denom = max(1, n_workers - 1)

        # Overlap ranks: higher overlap = higher rank (better)
        sorted_by_overlap = sorted(worker_ids, key=lambda w: per_worker[w]["overlap"])
        overlap_ranks = {w: i / rank_denom for i, w in enumerate(sorted_by_overlap)}

        # Load ranks: higher kv_util = higher rank (worse / more loaded)
        sorted_by_load = sorted(worker_ids, key=lambda w: per_worker[w]["kv_util"])
        load_ranks = {w: i / rank_denom for i, w in enumerate(sorted_by_load)}

        # Inflight share: count inflight requests per worker from our tracking
        with self._inflight_lock:
            inflight_counts: dict[int, int] = {}
            for req_info in self._inflight.values():
                req_wid = req_info.get("wid")
                if req_wid is not None:
                    inflight_counts[req_wid] = inflight_counts.get(req_wid, 0) + 1
        total_inflight = max(1, sum(inflight_counts.values()))

        # --- Pass 2: build features, score, and record details --- #
        for wid in worker_ids:
            pw = per_worker[wid]
            overlap = pw["overlap"]
            wid_kv_util = pw["kv_util"]
            prefill_tokens = pw["prefill_tokens"]
            decode_blocks = pw["decode_blocks"]

            # Selection pressure: EMA of how often this worker is picked
            sel_pressure = self._selection_ema.get(wid, 1.0 / n_workers)
            inflight_share = inflight_counts.get(wid, 0) / total_inflight

            x = self._build_features(
                wid=wid,
                overlap=overlap,
                kv_util=wid_kv_util,
                overlap_rank=overlap_ranks[wid],
                load_rank=load_ranks[wid],
                selection_pressure=sel_pressure,
                prefill_tokens=prefill_tokens,
                tokens_in=tokens_in,
                inflight_share=inflight_share,
                osl=osl,
                iat=iat,
                reuse_budget=reuse_budget,
            )
            features_by_wid[wid] = x

            physics = self._physics_score(
                overlap, wid, worker_util, prefill_tokens, tokens_in,
                pw["memory_pressure"],
            )
            physics_by_wid[wid] = physics

            score = self._score_worker(
                wid,
                x,
                overlap,
                decode_blocks,
                last_worker,
                reuse_budget,
                iat_factor,
                physics,
                wid_kv_util,
            )
            raw_scores.append(score)

            worker_details.append(
                {
                    "id": wid,
                    "kv_overlap": round(overlap, 4),
                    "prefill_tokens": prefill_tokens,
                    "decode_blocks": decode_blocks,
                    "kv_util": round(wid_kv_util, 4),
                    "prefill_util": round(pw["prefill_util"], 4),
                    "memory_pressure": round(pw["memory_pressure"], 4),
                    "overlap_rank": round(overlap_ranks[wid], 3),
                    "load_rank": round(load_ranks[wid], 3),
                    "selection_pressure": round(sel_pressure, 4),
                    "inflight_share": round(inflight_share, 4),
                    "physics_score": round(physics, 4),
                    "beta_sample": round(self.beta_learner.sample(wid), 4),
                    "lints_sample": round(
                        math.tanh(self.lints_learner.sample(wid, x)), 4
                    ),
                    "final_score": round(score, 4),
                }
            )

        if not worker_ids:
            chosen = native_pick
        elif self.enable_cold_start:
            best_overlap = max(all_overlaps.values()) if all_overlaps else 0.0
            if best_overlap < self.cold_start_threshold:
                idx = self._cold_start_rr % len(worker_ids)
                self._cold_start_rr += 1
                chosen = worker_ids[idx]
                logger.info(
                    "COLD_START: prefix=%s chosen=%s best_ov=%.4f "
                    "threshold=%.4f rr_idx=%d/%d",
                    prefix_id,
                    chosen,
                    best_overlap,
                    self.cold_start_threshold,
                    idx,
                    len(worker_ids),
                )
            else:
                chosen = self._select_from_scores(
                    worker_ids, raw_scores, reuse_budget, iat_factor
                )
        else:
            chosen = self._select_from_scores(
                worker_ids, raw_scores, reuse_budget, iat_factor
            )

        self._prefix_workers[prefix_id] = chosen

        # Update selection pressure EMA: chosen worker gets a bump toward 1,
        # all others decay toward 0.  Converges to empirical selection frequency.
        alpha = self._selection_ema_alpha
        for wid in worker_ids:
            old = self._selection_ema.get(wid, 1.0 / n_workers)
            self._selection_ema[wid] = old * (1 - alpha) + (alpha if wid == chosen else 0.0)

        # Record learner contribution stats for the chosen worker
        for wd in worker_details:
            if wd["id"] == chosen:
                self.stats.record_learner_contribution(
                    wd.get("lints_sample", 0.0),
                    wd.get("beta_sample", 0.0),
                )
                break

        # Record routing decision in KvIndexer for predict-from-decision mode
        if self.kv_indexer is not None:
            self.kv_indexer.record_routing_decision(chosen, token_ids)

        request_id = uuid.uuid4().hex[:12]

        # Log hints so we can verify trie overrides are reaching the router
        logger.debug(
            "HINTS: prefix=%s osl=%d iat=%dms reuse=%d tokens_in=%d "
            "chosen=%d native=%d agreed=%s",
            prefix_id[:16], osl, iat, reuse_budget, tokens_in,
            chosen, native_pick, chosen == native_pick,
        )

        # Periodic feature summary (every 100 decisions) for validation
        total_obs = self.stats._count
        if total_obs > 0 and total_obs % 100 == 0:
            chosen_x = features_by_wid.get(chosen)
            feat_names = [
                "bias", "overlap_x_idle", "load_rank", "overlap_rank",
                "sel_pressure", "prefill_frac", "inflight_share",
                "osl_x_load", "iat_x_reuse",
            ]
            if chosen_x is not None:
                feat_str = ", ".join(
                    f"{n}={v:.3f}" for n, v in zip(feat_names, chosen_x)
                )
                logger.info(
                    "FEATURE_SUMMARY[%d]: %s | v=%.4f sel_ema=[%s]",
                    total_obs,
                    feat_str,
                    self.lints_learner.v,
                    ", ".join(
                        f"{self._selection_ema.get(w, 0):.3f}"
                        for w in sorted(worker_ids)[:4]
                    ),
                )

        return RoutingDecision(
            chosen=chosen,
            native_pick=native_pick,
            request_id=request_id,
            features=features_by_wid.get(chosen),
            loads_by_wid=loads_by_wid,
            worker_details=worker_details,
            prefix_id=prefix_id,
            osl=osl,
            iat=iat,
            reuse_budget=reuse_budget,
            tokens_in=tokens_in,
            last_worker=last_worker,
            physics_score=physics_by_wid.get(chosen, 0.5),
        )

    def _select_from_scores(
        self,
        worker_ids: list[int],
        raw_scores: list[float],
        reuse_budget: int,
        iat_factor: float,
    ) -> int:
        if self.enable_softmax:
            if self.enable_adaptive_temp:
                temp = self.adaptive_temp_base / (
                    1.0 + float(reuse_budget) * iat_factor
                )
                temp = min(max(temp, self.temp_min), self.temp_max)
            else:
                temp = self.temperature
            probs = self._softmax(raw_scores, temp)
            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    return worker_ids[i]
            return worker_ids[-1]
        else:
            best_idx = int(np.argmax(raw_scores))
            return worker_ids[best_idx]

    def _softmax(self, scores: list[float], temp: float) -> list[float]:
        t = float(min(max(temp, self.temp_min), self.temp_max))
        arr = np.array(scores)
        m = float(np.max(arr))
        exps = np.exp((arr - m) / max(1e-6, t))
        s = float(np.sum(exps))
        if s <= 0.0 or not np.isfinite(s):
            return [1.0 / len(scores)] * len(scores)
        return list((exps / s).astype(float))

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

    def update_feedback(
        self,
        decision: RoutingDecision,
        latency_ms: float,
        tokens_out: int,
    ) -> dict[str, Any]:
        """Update learners with observed latency reward.

        Two-tower residual update:
          - Beta learner receives the raw reward (context-free worker quality).
          - LinTS learner receives the *residual* reward after subtracting
            the physics tower's prediction, so it only learns what the
            physics model gets wrong (interactions, non-linearities, etc.).
        """
        metric, per_tok = LatencyTracker.latency_metric(latency_ms, tokens_out)
        osl_bin = self._osl_bin(decision.osl)
        prefill_bin = self._prefill_bin(decision.tokens_in)
        # Use global-bucket baseline (osl_bin, prefill_bin, per_tok) — shared
        # across all workers.  This normalizes for request type (long-decode
        # vs short-decode) so the reward reflects worker quality, not request
        # difficulty.  Falls back to global baseline for sparse buckets.
        baseline = self.latency_tracker.get_global_bucket_baseline(
            osl_bin, prefill_bin, per_tok, fallback=metric,
        )
        used_bucket = (osl_bin, prefill_bin, per_tok) in self.latency_tracker._global_bucket
        self.stats.record_baseline_lookup(used_bucket)

        reward = LatencyTracker.compute_reward(metric, baseline, True)

        # Beta learner: raw reward (context-free worker quality)
        self.beta_learner.update(decision.chosen, reward)

        # LinTS learner: residual reward (what physics model missed)
        #
        # Calibrated residual: instead of using raw physics score (which has
        # a systematic bias when utilization signals are near-constant), we
        # center the physics prediction using a running EMA so the residual
        # mean converges to 0.5 regardless of the physics score's absolute
        # scale.  This prevents LinTS from learning spurious corrections
        # (e.g. anti-caching) due to the physics tower's inflated floor.
        #
        # calibrated_physics = 0.5 + (physics - physics_ema) / (2 * physics_std)
        # residual = clamp(reward - calibrated_physics + 0.5, 0, 1)
        physics = decision.physics_score
        cal_alpha = self._physics_cal_alpha
        self._physics_ema_mean = (1 - cal_alpha) * self._physics_ema_mean + cal_alpha * physics
        self._physics_ema_sq = (1 - cal_alpha) * self._physics_ema_sq + cal_alpha * physics * physics
        physics_var = max(1e-6, self._physics_ema_sq - self._physics_ema_mean ** 2)
        physics_std = physics_var ** 0.5

        # Map physics score to [~0, ~1] centered on 0.5 at the EMA mean.
        # The 2*std denominator maps ±2σ to [0, 1].
        calibrated = 0.5 + (physics - self._physics_ema_mean) / max(1e-6, 2.0 * physics_std)
        calibrated = max(0.0, min(1.0, calibrated))

        residual = max(0.0, min(1.0, reward - calibrated + 0.5))

        # Record instrumentation (use raw physics for stats, calibrated for LinTS)
        self.stats.record_feedback(reward, physics, residual)

        # Adaptive v: update EMA of residual variance and set lints_learner.v
        if self.enable_adaptive_v:
            alpha = self._adaptive_v_alpha
            self._residual_ema_mean = (1 - alpha) * self._residual_ema_mean + alpha * residual
            diff = residual - self._residual_ema_mean
            self._residual_ema_var = (1 - alpha) * self._residual_ema_var + alpha * diff * diff
            new_v = max(self._adaptive_v_min, min(self._adaptive_v_max, self._residual_ema_var ** 0.5))
            self.lints_learner.v = new_v

        x_chosen = decision.features
        if x_chosen is None:
            # Fallback: reconstruct features with neutral utilization (no monitor
            # data available at feedback time).  overlap approximated from stored
            # prefill/tokens_in.
            logger.debug(
                "FEATURE_FALLBACK: wid=%d features not stored at decision time, "
                "reconstructing with neutral utilization.",
                decision.chosen,
            )
            chosen_load = decision.loads_by_wid.get(decision.chosen, {})
            prefill = chosen_load.get("potential_prefill_tokens", 0)
            fb_overlap = 1.0 - prefill / max(1, decision.tokens_in)
            n_workers = max(1, len(decision.loads_by_wid))
            x_chosen = self._build_features(
                wid=decision.chosen,
                overlap=fb_overlap,
                kv_util=0.0,            # unknown at feedback time
                overlap_rank=0.5,       # neutral
                load_rank=0.5,          # neutral
                selection_pressure=self._selection_ema.get(decision.chosen, 1.0 / n_workers),
                prefill_tokens=prefill,
                tokens_in=decision.tokens_in,
                inflight_share=1.0 / n_workers,  # neutral
                osl=decision.osl,
                iat=decision.iat,
                reuse_budget=decision.reuse_budget,
            )
        self.lints_learner.update(decision.chosen, x_chosen, residual)
        self.latency_tracker.update_baselines(
            decision.chosen, osl_bin, prefill_bin, metric, per_tok
        )

        beta_alpha, beta_beta = self.beta_learner.get_params(decision.chosen)
        lints_mean = self.lints_learner.posterior_mean(decision.chosen).tolist()

        logger.debug(
            "Feedback: wid=%s metric=%.2f baseline=%.2f reward=%.3f "
            "physics=%.3f residual=%.3f tokens_out=%d",
            decision.chosen,
            metric,
            baseline,
            reward,
            decision.physics_score,
            residual,
            tokens_out,
        )

        return {
            "metric": metric,
            "baseline_ema": baseline,
            "reward": reward,
            "physics_score": decision.physics_score,
            "residual_reward": residual,
            "beta_after": {
                "alpha": round(beta_alpha, 4),
                "beta": round(beta_beta, 4),
            },
            "lints_posterior_mean": [round(v, 6) for v in lints_mean],
        }

    # -------------------- Scoring -------------------- #

    def _score_worker(
        self,
        wid: int,
        x: np.ndarray,
        overlap: float,
        decode_blocks: int,
        last_worker: int | None,
        reuse_budget: int,
        iat_factor: float,
        physics: float,
        kv_util: float = 0.0,
    ) -> float:
        """Two-tower scoring: physics (known) + learned residual + heuristics.

        Tower 1 (physics):  Directly observed signals — cache hit, utilization.
                            No learning. Stable anchor.  Range [0, ~1].
        Tower 2 (learned):  LinTS + Beta TS. Captures non-linearities,
                            interactions, and worker-specific effects that the
                            physics model misses. Bounded via tanh.
        Heuristics:         Affinity bonus and switching penalty (saturated).
        """
        # --- Tower 1: physics (passed in, already computed) --- #
        utility = physics

        # --- Tower 2: learned residual (bounded via tanh) --- #
        if self.enable_beta_ts:
            if self.enable_adaptive_explore:
                ts_w_eff = self.ts_weight / (1.0 + float(reuse_budget) * iat_factor)
            else:
                ts_w_eff = self.ts_weight
            beta_raw = self.beta_learner.sample(wid)
            utility += ts_w_eff * math.tanh(beta_raw)

        if self.enable_lints:
            raw_lints = self.lints_learner.sample(wid, x)
            utility += abs(self.lints_weight) * math.tanh(raw_lints)

        # --- Heuristic adjustments (saturated) --- #
        if self.enable_affinity and last_worker == wid and reuse_budget > 0:
            raw_affinity = (
                self.affinity_base + self.affinity_reuse_weight * float(reuse_budget)
            ) * (0.5 + 0.5 * overlap)
            utility += math.tanh(raw_affinity)

        # --- Load modulator applied to positive utility only --- #
        # Use hardware-agnostic kv_util [0,1] instead of raw decode_blocks
        # to avoid IEEE 754 underflow (decode_blocks can be thousands,
        # causing exp(-qpw * db^2 / 2500) to collapse to 0.0).
        qpw = self.queue_penalty_weight
        u = min(1.0, max(0.0, kv_util))
        load_mod = math.exp(-qpw * u * u)

        if self.enable_sticky_floor and last_worker == wid and reuse_budget > 0:
            load_mod = max(load_mod, self.sticky_load_floor)
        if self.enable_load_mod_floor and self.load_mod_floor > 0.0:
            load_mod = max(load_mod, self.load_mod_floor)

        score = utility * load_mod

        # --- Switching penalty (outside load modulator) --- #
        if (
            self.enable_switching_cost
            and last_worker is not None
            and wid != last_worker
            and reuse_budget > 0
        ):
            raw_penalty = self.switch_base + self.switch_reuse * float(reuse_budget)
            score -= self.switch_cost_weight * math.tanh(raw_penalty)

        if np.isnan(score) or np.isinf(score):
            score = -1e9

        return float(score)

    # -------------------- Worker Load Utilities -------------------- #

    def _get_worker_utilization(self) -> dict[int, dict]:
        """Get live utilization ratios from WorkerLoadMonitor.

        Returns { worker_id: { "kv_util": float, "prefill_util": float } }
        where both values are in [0, 1].  Falls back to empty dict if monitor
        is unavailable or has no data.
        """
        if self.worker_load_monitor is None:
            if self._monitor_fallback_count == 0:
                logger.info(
                    "MONITOR_DISABLED: WorkerLoadMonitor is None — all physics "
                    "scores will use prefill_ratio fallback."
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
                "all workers will use physics fallback this round.",
                e,
            )

        return result

    # -------------------- Physics Tower -------------------- #

    def _physics_score(
        self,
        overlap: float,
        wid: int,
        worker_util: dict[int, dict],
        prefill_tokens: int,
        tokens_in: int,
        memory_pressure: float,
    ) -> float:
        """Compute physics-based utility from directly observable signals.

        All inputs are in [0, 1] and have known causal relationships to latency.
        No learning — this is the stable anchor of the two-tower design.

        Signals:
            overlap         — cache hit (immediate prefill savings)
            kv_util         — compute contention (active decode load)
            prefill_util    — queue saturation (prefill backlog)
            memory_pressure — eviction risk (total cached blocks / capacity)

        Returns a score in [0, ~1] (exact max depends on weight sum).
        """
        # Cache benefit: fraction of prefill tokens already cached
        cache_hit = overlap  # [0, 1]

        # Compute load: fraction of KV capacity in active decode
        util = worker_util.get(wid)
        if util is not None:
            compute_avail = 1.0 - util["kv_util"]       # [0, 1], higher = less loaded
            queue_avail = 1.0 - util["prefill_util"]     # [0, 1], higher = less queued
            self._monitor_hit_count += 1
            self.stats.record_monitor_lookup(hit=True)
        else:
            # Fallback: no utilization data for this worker
            prefill_ratio = prefill_tokens / max(1, tokens_in)
            compute_avail = 1.0 - prefill_ratio  # rough proxy
            queue_avail = 0.5  # no information, assume neutral
            self._monitor_fallback_count += 1
            self.stats.record_monitor_lookup(hit=False)
            total = self._monitor_hit_count + self._monitor_fallback_count
            if self._monitor_fallback_count <= 5 or total % 100 == 0:
                logger.warning(
                    "PHYSICS_FALLBACK: wid=%d no utilization data "
                    "(fallback=%d/%d total, %.1f%%). Using prefill_ratio proxy.",
                    wid,
                    self._monitor_fallback_count,
                    total,
                    100.0 * self._monitor_fallback_count / max(1, total),
                )

        # Memory pressure: eviction risk (high = cache nearly full, future hits at risk)
        memory_avail = 1.0 - memory_pressure  # [0, 1], higher = less eviction risk

        return (
            self.physics_cache_weight * cache_hit
            + self.physics_compute_weight * compute_avail
            + self.physics_queue_weight * queue_avail
            + self.physics_memory_weight * memory_avail
        )

    # -------------------- Feature Vector -------------------- #

    @staticmethod
    def _decode_cost(osl: int) -> float:
        """Interpolate decode cost from continuous OSL (tokens).

        Anchor points: 128->1.0, 250->2.0, 1024->3.0.
        """
        if osl <= 128:
            return 1.0
        if osl <= 250:
            return 1.0 + (osl - 128) / (250 - 128)
        if osl >= 1024:
            return 3.0
        return 2.0 + (osl - 250) / (1024 - 250)

    @staticmethod
    def _iat_factor(iat: int) -> float:
        """Interpolate IAT factor from continuous IAT (ms).

        Anchor points: 50->1.5, 250->1.0, 1000->0.6.
        """
        if iat <= 50:
            return 1.5
        if iat <= 250:
            return 1.5 - 0.5 * (iat - 50) / (250 - 50)
        if iat >= 1000:
            return 0.6
        return 1.0 - 0.4 * (iat - 250) / (1000 - 250)

    def _build_features(
        self,
        wid: int,
        overlap: float,
        kv_util: float,
        overlap_rank: float,
        load_rank: float,
        selection_pressure: float,
        prefill_tokens: int,
        tokens_in: int,
        inflight_share: float,
        osl: int,
        iat: int,
        reuse_budget: int,
    ) -> np.ndarray:
        """Build the LinTS feature vector from non-redundant, worker-differentiating signals.

        All features are in [0, 1] and capture interactions or relative positions
        that the linear physics tower cannot represent.  No feature duplicates a
        physics tower signal in its raw form.

        Features (9 dims):
            [0] bias               = 1.0                          (per-worker intercept)
            [1] overlap_x_idle     = overlap * (1 - kv_util)      (cache value discounted by load)
            [2] load_rank          = rank(kv_util) / (n-1)        (relative load position)
            [3] overlap_rank       = rank(overlap) / (n-1)        (relative cache position)
            [4] selection_pressure = EMA of picks / total          (herding detection)
            [5] prefill_fraction   = prefill_tokens / tokens_in   (actual uncached fraction)
            [6] inflight_share     = inflight[wid] / total_inflight (direct queue depth)
            [7] osl_x_load         = osl_norm * kv_util           (long output + loaded = bad)
            [8] iat_x_reuse        = iat_norm * reuse_norm        (rapid reuse = sticky)
        """
        overlap_x_idle = overlap * (1.0 - kv_util)
        prefill_fraction = prefill_tokens / max(1, tokens_in)
        osl_norm = min(osl, 1024) / 1024.0
        reuse_norm = math.tanh(0.25 * max(reuse_budget, 0))
        iat_norm = (self._iat_factor(iat) - 0.6) / 0.9

        return np.array(
            [
                1.0,                         # [0] bias
                overlap_x_idle,              # [1] interaction: cache value × idle
                load_rank,                   # [2] relative load position
                overlap_rank,                # [3] relative cache position
                selection_pressure,          # [4] herding detection
                prefill_fraction,            # [5] actual uncached fraction
                inflight_share,              # [6] direct queue depth
                osl_norm * kv_util,          # [7] osl × load interaction
                iat_norm * reuse_norm,       # [8] iat × reuse interaction
            ],
            dtype=np.float64,
        )
