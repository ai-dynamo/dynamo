# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Modular learner components for Thompson Sampling routing.

Each class is self-contained and thread-safe, suitable for unit testing
without requiring NATS, Dynamo runtime, or backend workers.

Classes:
    BetaLearner      — Per-worker Beta-Thompson Sampling bandit
    LinTSLearner     — Per-worker Linear Thompson Sampling contextual bandit
    LatencyTracker   — Hierarchical EMA latency baselines + reward computation
    PendingDecisions — In-flight decision tracking with timeout sweep
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BetaLearner
# ---------------------------------------------------------------------------
class BetaLearner:
    """Per-worker Beta-Thompson Sampling bandit with optional exponential decay.

    Each worker maintains (alpha, beta) parameters.  On each decision,
    ``sample(wid)`` draws from Beta(alpha, beta).  After observing a
    reward in [0, 1], ``update(wid, reward)`` shifts the posterior:

        alpha = decay * alpha + reward
        beta  = decay * beta  + (1 - reward)

    When ``decay < 1.0``, old observations are exponentially forgotten,
    creating an effective sliding window.  The effective window size is
    approximately ``1 / (1 - decay)`` observations.

    Examples:
        decay=1.000 -> no forgetting (classical Bayesian, default)
        decay=0.995 -> window ~ 200 observations, half-life ~ 138
        decay=0.990 -> window ~ 100 observations, half-life ~ 69
        decay=0.980 -> window ~ 50 observations, half-life ~ 34

    A ``min_pseudo_count`` floor prevents the posterior from collapsing
    to a point mass after heavy decay (keeps exploration alive).

    Thread-safe: all mutations go through ``_lock``.
    """

    def __init__(
        self,
        decay: float = 1.0,
        min_pseudo_count: float = 1.0,
    ) -> None:
        self.decay = float(max(0.0, min(decay, 1.0)))
        self.min_pseudo_count = float(max(0.01, min_pseudo_count))
        self._lock = threading.Lock()
        self._bandits: dict[int, tuple[float, float]] = {}

    def add_worker(self, wid: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        with self._lock:
            self._bandits.setdefault(wid, (float(alpha), float(beta)))

    def remove_worker(self, wid: int) -> None:
        with self._lock:
            self._bandits.pop(wid, None)

    @property
    def worker_ids(self) -> list[int]:
        with self._lock:
            return list(self._bandits.keys())

    def sample(self, wid: int) -> float:
        """Draw a single sample from Beta(alpha, beta) for *wid*."""
        with self._lock:
            alpha, beta = self._bandits.get(wid, (1.0, 1.0))
        return float(np.random.beta(alpha, beta))

    def update(self, wid: int, reward: float) -> tuple[float, float]:
        """Bayesian update with decay and return new (alpha, beta)."""
        r = float(max(0.0, min(1.0, reward)))
        with self._lock:
            alpha, beta = self._bandits.get(wid, (1.0, 1.0))
            new_alpha = max(self.min_pseudo_count, self.decay * alpha + r)
            new_beta = max(self.min_pseudo_count, self.decay * beta + (1.0 - r))
            self._bandits[wid] = (new_alpha, new_beta)
        return new_alpha, new_beta

    def get_params(self, wid: int) -> tuple[float, float]:
        with self._lock:
            return self._bandits.get(wid, (1.0, 1.0))

    def mean(self, wid: int) -> float:
        """Posterior mean = alpha / (alpha + beta)."""
        alpha, beta = self.get_params(wid)
        return alpha / (alpha + beta)

    def effective_sample_size(self, wid: int) -> float:
        """Effective number of observations = alpha + beta - 2 * min_pseudo_count."""
        alpha, beta = self.get_params(wid)
        return alpha + beta - 2.0 * self.min_pseudo_count

    def reset(self, wid: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        with self._lock:
            self._bandits[wid] = (float(alpha), float(beta))

    @property
    def half_life(self) -> float:
        """Number of observations for an observation's weight to halve."""
        if self.decay >= 1.0:
            return float("inf")
        return math.log(0.5) / math.log(self.decay)

    @property
    def effective_window(self) -> float:
        """Approximate number of recent observations that dominate the posterior."""
        if self.decay >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - self.decay)

    def reset_all(self) -> None:
        """Reset all known workers to the uninformative prior (alpha=1, beta=1)."""
        with self._lock:
            for wid in list(self._bandits):
                self._bandits[wid] = (1.0, 1.0)

    def to_dict(self) -> dict:
        """Serialize learner state to a JSON-compatible dict."""
        with self._lock:
            bandits = {str(wid): list(ab) for wid, ab in self._bandits.items()}
        return {
            "type": "BetaLearner",
            "decay": self.decay,
            "min_pseudo_count": self.min_pseudo_count,
            "bandits": bandits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BetaLearner:
        """Restore a BetaLearner from a dict produced by ``to_dict``."""
        learner = cls(
            decay=data.get("decay", 1.0),
            min_pseudo_count=data.get("min_pseudo_count", 1.0),
        )
        for wid_str, ab in data.get("bandits", {}).items():
            learner._bandits[int(wid_str)] = (float(ab[0]), float(ab[1]))
        return learner

    def load_state(self, data: dict) -> None:
        """Load bandit state in-place from a dict (preserves object identity)."""
        with self._lock:
            self.decay = float(data.get("decay", self.decay))
            self.min_pseudo_count = float(
                data.get("min_pseudo_count", self.min_pseudo_count)
            )
            self._bandits.clear()
            for wid_str, ab in data.get("bandits", {}).items():
                self._bandits[int(wid_str)] = (float(ab[0]), float(ab[1]))


# ---------------------------------------------------------------------------
# LinTSLearner
# ---------------------------------------------------------------------------
class LinTSLearner:
    """Per-worker Linear Thompson Sampling contextual bandit.

    State per worker:
        A  -- (d x d) precision matrix, initialised to lambda * I
        b  -- (d,)  reward-weighted feature accumulator

    Posterior mean:  theta_hat = A^-1 b
    Posterior sample: theta ~ N(theta_hat, v^2 A^-1)

    On ``update(wid, x, reward)``:
        A *= forget_rate;  b *= forget_rate
        A += x x^T;        b += x * reward
        (+ ridge re-regularisation)
    """

    def __init__(
        self,
        feature_dim: int = 9,
        lambda_: float = 1.0,
        v: float = 0.25,
        forget_rate: float = 0.995,
        *,
        jitter_base: float = 1e-9,
        jitter_mult: float = 10.0,
        jitter_max: float = 1e-3,
        eig_floor: float = 1e-10,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.lambda_ = float(lambda_)
        self.v = float(v)
        self.forget_rate = float(max(1e-6, min(forget_rate, 0.999999)))
        self._jt_base = float(jitter_base)
        self._jt_mult = float(jitter_mult)
        self._jt_max = float(jitter_max)
        self._eig_floor = float(eig_floor)

        self._lock = threading.Lock()
        self._A: dict[int, np.ndarray] = {}
        self._b: dict[int, np.ndarray] = {}

    def add_worker(self, wid: int) -> None:
        with self._lock:
            if wid not in self._A:
                self._A[wid] = self.lambda_ * np.eye(
                    self.feature_dim, dtype=np.float64
                )
                self._b[wid] = np.zeros(self.feature_dim, dtype=np.float64)

    def remove_worker(self, wid: int) -> None:
        with self._lock:
            self._A.pop(wid, None)
            self._b.pop(wid, None)

    @property
    def worker_ids(self) -> list[int]:
        with self._lock:
            return list(self._A.keys())

    def _ensure_worker(self, wid: int) -> None:
        if wid not in self._A:
            with self._lock:
                if wid not in self._A:
                    self._A[wid] = self.lambda_ * np.eye(
                        self.feature_dim, dtype=np.float64
                    )
                    self._b[wid] = np.zeros(self.feature_dim, dtype=np.float64)

    def sample(self, wid: int, x: np.ndarray) -> float:
        """Posterior-sample score for worker *wid* given feature vector *x*."""
        self._ensure_worker(wid)
        with self._lock:
            A = np.array(self._A[wid], dtype=np.float64, copy=True)
            b = np.array(self._b[wid], dtype=np.float64, copy=True)

        A = 0.5 * (A + A.T)
        eye = np.eye(self.feature_dim, dtype=np.float64)
        jitter = self._jt_base
        while True:
            try:
                L = np.linalg.cholesky(A + jitter * eye)
                break
            except np.linalg.LinAlgError:
                jitter = jitter * self._jt_mult if jitter > 0 else self._jt_base
                if jitter > self._jt_max:
                    vals, vecs = np.linalg.eigh(A)
                    vals = np.maximum(vals, self._eig_floor)
                    A_inv = vecs @ (np.diag(1.0 / vals)) @ vecs.T
                    mu = A_inv @ b
                    z = np.random.normal(size=self.feature_dim)
                    noise = vecs @ (z / np.sqrt(vals))
                    theta = mu + (self.v * noise)
                    return float(theta @ x)

        y = np.linalg.solve(L, b)
        mu = np.linalg.solve(L.T, y)
        z = np.random.normal(size=self.feature_dim)
        noise = np.linalg.solve(L.T, z)
        theta = mu + (self.v * noise)
        return float(theta @ x)

    def update(self, wid: int, x: np.ndarray, reward: float) -> None:
        """Rank-1 Bayesian update with exponential forgetting."""
        self._ensure_worker(wid)
        r = float(max(0.0, min(1.0, reward)))
        with self._lock:
            A = self._A[wid]
            b = self._b[wid]
            A *= self.forget_rate
            b *= self.forget_rate
            A += np.outer(x, x)
            ridge = (1.0 - self.forget_rate) * self.lambda_
            if ridge > 0.0:
                A += ridge * np.eye(self.feature_dim, dtype=np.float64)
            self._A[wid] = 0.5 * (A + A.T)
            self._b[wid] = b + x * r

    def get_params(self, wid: int) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of (A, b) for *wid*."""
        self._ensure_worker(wid)
        with self._lock:
            return (
                np.array(self._A[wid], copy=True),
                np.array(self._b[wid], copy=True),
            )

    def posterior_mean(self, wid: int) -> np.ndarray:
        """Return theta_hat = A^-1 b (the MAP estimate)."""
        A, b = self.get_params(wid)
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.feature_dim, dtype=np.float64)

    def reset_all(self) -> None:
        """Reset all known workers to the uninformative prior (A=lambda*I, b=0)."""
        with self._lock:
            for wid in list(self._A):
                self._A[wid] = self.lambda_ * np.eye(
                    self.feature_dim, dtype=np.float64
                )
                self._b[wid] = np.zeros(self.feature_dim, dtype=np.float64)

    def to_dict(self) -> dict:
        """Serialize learner state to a JSON-compatible dict."""
        with self._lock:
            workers = {}
            for wid in self._A:
                workers[str(wid)] = {
                    "A": self._A[wid].tolist(),
                    "b": self._b[wid].tolist(),
                }
        return {
            "type": "LinTSLearner",
            "feature_dim": self.feature_dim,
            "lambda": self.lambda_,
            "v": self.v,
            "forget_rate": self.forget_rate,
            "workers": workers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LinTSLearner:
        """Restore a LinTSLearner from a dict produced by ``to_dict``."""
        learner = cls(
            feature_dim=data.get("feature_dim", 9),
            lambda_=data.get("lambda", 1.0),
            v=data.get("v", 0.25),
            forget_rate=data.get("forget_rate", 0.995),
        )
        for wid_str, state in data.get("workers", {}).items():
            wid = int(wid_str)
            learner._A[wid] = np.array(state["A"], dtype=np.float64)
            learner._b[wid] = np.array(state["b"], dtype=np.float64)
        return learner

    def load_state(self, data: dict) -> None:
        """Load learner state in-place from a dict (preserves object identity)."""
        with self._lock:
            self.v = float(data.get("v", self.v))
            self.forget_rate = float(data.get("forget_rate", self.forget_rate))
            self._A.clear()
            self._b.clear()
            for wid_str, state in data.get("workers", {}).items():
                wid = int(wid_str)
                self._A[wid] = np.array(state["A"], dtype=np.float64)
                self._b[wid] = np.array(state["b"], dtype=np.float64)


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Hierarchical EMA latency baselines and reward computation.

    Baselines are maintained at three levels (most specific to least):
        1. Per-bucket:  (worker, osl_bin, prefill_bin, per_tok)
        2. Per-worker:  (worker, per_tok)
        3. Global:      (per_tok)

    ``get_baseline`` falls through the hierarchy; ``update_baselines``
    refreshes all three.  ``compute_reward`` converts an observed latency
    metric into a [0, 1] reward relative to the baseline.
    """

    def __init__(self, ema_alpha: float = 0.2) -> None:
        self.ema_alpha = float(ema_alpha)
        self._global: dict[bool, float | None] = {False: None, True: None}
        self._global_bucket: dict[tuple[str, str, bool], float] = {}
        self._worker: dict[tuple[int, bool], float] = {}
        self._bucket: dict[tuple[int, str, str, bool], float] = {}

    def _ema(self, old: float | None, new: float) -> float:
        a = self.ema_alpha
        return new if old is None else (a * new + (1.0 - a) * old)

    def get_baseline(
        self,
        wid: int,
        osl: str,
        prefill_bin: str,
        per_tok: bool,
        fallback: float,
    ) -> float:
        """Hierarchical lookup: bucket -> worker -> global -> fallback."""
        key_b = (wid, osl, prefill_bin, per_tok)
        if key_b in self._bucket:
            return self._bucket[key_b]
        key_w = (wid, per_tok)
        if key_w in self._worker:
            return self._worker[key_w]
        if self._global[per_tok] is not None:
            return self._global[per_tok]  # type: ignore[return-value]
        return max(1.0, float(fallback))

    def get_global_baseline(self, per_tok: bool, fallback: float) -> float:
        """Return only the global-level baseline (shared across all workers).

        Use this for reward computation so that fast workers consistently
        receive higher rewards than slow workers.  The per-worker and
        per-bucket EMAs are still tracked by ``update_baselines`` for
        diagnostics but should not be used for the reward signal.
        """
        if self._global[per_tok] is not None:
            return self._global[per_tok]  # type: ignore[return-value]
        return max(1.0, float(fallback))

    def get_global_bucket_baseline(
        self,
        osl: str,
        prefill_bin: str,
        per_tok: bool,
        fallback: float,
    ) -> float:
        """Return a request-type-normalized baseline shared across all workers.

        Looks up a global bucket keyed by (osl, prefill_bin, per_tok) first,
        then falls back to the global baseline.  This compares like-for-like
        (long-decode vs long-decode) so that request difficulty does not
        confound the reward signal, while still keeping all workers on a
        common scale (so fast workers get higher rewards than slow ones).
        """
        key = (osl, prefill_bin, per_tok)
        if key in self._global_bucket:
            return self._global_bucket[key]
        if self._global[per_tok] is not None:
            return self._global[per_tok]  # type: ignore[return-value]
        return max(1.0, float(fallback))

    def update_baselines(
        self,
        wid: int,
        osl: str,
        prefill_bin: str,
        metric: float,
        per_tok: bool,
    ) -> float:
        """Update all four EMA levels and return the per-worker bucket value."""
        self._global[per_tok] = self._ema(self._global[per_tok], metric)
        key_gb = (osl, prefill_bin, per_tok)
        self._global_bucket[key_gb] = self._ema(
            self._global_bucket.get(key_gb), metric
        )
        key_w = (wid, per_tok)
        self._worker[key_w] = self._ema(self._worker.get(key_w), metric)
        key_b = (wid, osl, prefill_bin, per_tok)
        self._bucket[key_b] = self._ema(self._bucket.get(key_b), metric)
        return self._bucket[key_b]

    def reset(self) -> None:
        """Clear all EMA baselines back to initial state."""
        self._global = {False: None, True: None}
        self._global_bucket.clear()
        self._worker.clear()
        self._bucket.clear()

    @staticmethod
    def latency_metric(
        latency_ms: float, tokens_out: int | None
    ) -> tuple[float, bool]:
        """Convert raw latency into a metric.  Returns (metric, per_tok)."""
        if tokens_out is not None and int(tokens_out) > 0:
            return float(latency_ms) / float(max(1, int(tokens_out))), True
        return float(latency_ms), False

    @staticmethod
    def compute_reward(metric: float, baseline: float, success: bool) -> float:
        """Map observed metric to [0, 1] reward.

        reward = 1 / (1 + metric / baseline)
        Fast requests (metric << baseline) -> reward ~ 1
        Slow requests (metric >> baseline) -> reward ~ 0
        """
        if not success:
            return 0.0
        denom = max(1e-3, baseline)
        ratio = metric / denom
        return float(1.0 / (1.0 + ratio))


# ---------------------------------------------------------------------------
# PendingDecisions
# ---------------------------------------------------------------------------
class PendingDecisions:
    """Tracks in-flight routing decisions awaiting feedback.

    ``add(decision_id, record)`` stores a decision.
    ``pop(decision_id)`` returns and removes it (or None if unknown).
    ``sweep(now)`` expires decisions older than ``timeout_seconds`` and
    returns them so the caller can apply timeout penalties.
    """

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        sweep_interval_seconds: float = 5.0,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.sweep_interval_seconds = float(sweep_interval_seconds)
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._last_sweep: float = 0.0

    def add(self, decision_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._pending[decision_id] = record

    def pop(self, decision_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._pending.pop(decision_id, None)

    def sweep(self, now: float) -> list[tuple[str, dict[str, Any]]]:
        """Return expired decisions if the sweep interval has elapsed.

        Returns a list of (decision_id, record) tuples that exceeded
        ``timeout_seconds``.  Automatically respects the sweep interval
        to avoid scanning on every call.
        """
        if now - self._last_sweep < self.sweep_interval_seconds:
            return []
        self._last_sweep = now
        expired: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            for did, rec in list(self._pending.items()):
                if now - float(rec.get("start_ts", now)) >= self.timeout_seconds:
                    expired.append((did, rec))
                    self._pending.pop(did, None)
        return expired

    def count(self) -> int:
        with self._lock:
            return len(self._pending)

    def per_worker_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        with self._lock:
            for rec in self._pending.values():
                w = int(rec.get("wid", -1))
                counts[w] = counts.get(w, 0) + 1
        return counts
