# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability for GMS shadow-engine failover.

Emits Prometheus metrics for the failover state machine driven by
``worker_factory._maybe_wait_for_failover_lock``. Uses the same
``prometheus_client`` + ``register_prometheus_expfmt_callback`` bridge as
``register_embedding_cache_metrics`` (see ``common/utils/prometheus.py``), so the
metrics land on the shared ``DYN_SYSTEM_PORT/metrics`` surface — no extra port,
no standalone client.

Design notes
------------
* **Event-driven, not polled.** Metric values are set at the state transitions
  (in ``worker_factory``); the scrape callback only encodes the current registry.
* **States:** ``init -> standby -> waking -> active``. A dead engine cannot
  self-report — absence of the series (Prometheus staleness) plus a k8s restart
  is the "dead" signal. ``active_engines``/``shadow_ready`` are derived downstream
  as ``count(state=="active")`` / ``count(state=="standby")``.
* **Switch counters are write-through persisted** to a per-engine file in the
  shared failover dir (the GMS emptyDir). A counter increment that precedes a
  process death survives on disk and is re-exposed after the container restarts
  reload it — so a lost scrape doesn't lose the event. Failures are derived
  downstream as ``attempts - success``; we never ask a dying process to report
  its own failure.
* Only real failovers (a shadow that won a *contended* lock) increment the switch
  counters — an initial bootup acquires the lock immediately and is not a switch.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Metric name prefix. Mirrors the `dynamo_component_` convention (name_prefix.COMPONENT)
# used by the Rust registry and the existing Python engine metrics.
_PREFIX = "dynamo_component_engine_failover"

# The failover lifecycle states (1-hot in the state gauge). "dead" is intentionally
# absent: a dead engine can't emit — its series goes stale and k8s reports the restart.
STATES = ("init", "standby", "waking", "active")


class FailoverMetrics:
    """Per-engine failover metrics, exposed via the engine's system /metrics."""

    def __init__(
        self,
        engine_id: str,
        model_name: str,
        component_name: str,
        persist_dir: str,
    ) -> None:
        # Lazy import: prometheus_client must be imported after the vLLM multiproc
        # dir is configured (mirrors register_embedding_cache_metrics). We use a
        # dedicated CollectorRegistry, independent of the vLLM multiproc REGISTRY.
        from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

        self._generate_latest = generate_latest
        self._registry = CollectorRegistry()
        # RLock: transition hooks and the scrape callback are mutually exclusive;
        # re-entrant so a hook may call another under the same lock.
        self._lock = threading.RLock()
        self._engine_id = str(engine_id)
        self._lv = {
            "engine_id": self._engine_id,
            "model": model_name or "",
            "dynamo_component": component_name or "",
        }
        base = ["engine_id", "model", "dynamo_component"]

        self._state = Gauge(
            f"{_PREFIX}_state",
            "Current failover state (1 for the active state, 0 otherwise).",
            base + ["state"],
            registry=self._registry,
        )
        self._state_entered = Gauge(
            f"{_PREFIX}_state_entered_timestamp_seconds",
            "Unix timestamp when the engine entered its current failover state.",
            base,
            registry=self._registry,
        )
        self._transitions = Counter(
            f"{_PREFIX}_transitions_total",
            "Total failover state transitions.",
            base + ["from_state", "to_state"],
            registry=self._registry,
        )
        self._attempts = Counter(
            f"{_PREFIX}_switch_attempts_total",
            "Total failover promotions attempted (a shadow won a contended lock).",
            base,
            registry=self._registry,
        )
        self._successes = Counter(
            f"{_PREFIX}_switch_success_total",
            "Total failover promotions that completed and began serving.",
            base,
            registry=self._registry,
        )
        self._last_state_duration = Gauge(
            f"{_PREFIX}_last_state_duration_seconds",
            "Duration of the most recent completed occupancy of each state.",
            base + ["state"],
            registry=self._registry,
        )

        # Export zeros from the first scrape (Prometheus best practice: zeros, not absent).
        for s in STATES:
            self._state.labels(state=s, **self._lv).set(0)
        self._attempts.labels(**self._lv)
        self._successes.labels(**self._lv)

        self._cur_state: str | None = None
        self._cur_entered: float | None = None
        self._attempts_val = 0
        self._success_val = 0
        # True between a recorded failover attempt and its success, so that
        # success only ever counts a promotion that was itself counted as an
        # attempt (keeps derived failures = attempts - success >= 0 and excludes
        # the initial bootup, which is not a contended promotion).
        self._promotion_pending = False

        self._persist_path = os.path.join(
            persist_dir, f"failover_metrics_engine-{self._engine_id}.json"
        )
        self._restore()

    # ------------------------------------------------------------------ #
    # Durability: write-through the switch counters to the shared dir so a
    # pre-death increment survives and is re-exposed after a container restart.
    # ------------------------------------------------------------------ #
    def _restore(self) -> None:
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError, OSError):
            return
        attempts = int(data.get("attempts", 0) or 0)
        success = int(data.get("success", 0) or 0)
        if attempts:
            self._attempts_val = attempts
            self._attempts.labels(**self._lv).inc(attempts)
        if success:
            self._success_val = success
            self._successes.labels(**self._lv).inc(success)
        if attempts or success:
            logger.info(
                "[Shadow] Restored failover counters engine=%s attempts=%d success=%d",
                self._engine_id,
                attempts,
                success,
            )

    def _persist(self) -> None:
        try:
            tmp = f"{self._persist_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {"attempts": self._attempts_val, "success": self._success_val}, f
                )
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, self._persist_path)  # atomic replace
        except OSError as e:
            logger.warning("[Shadow] Failed to persist failover counters: %s", e)

    # ------------------------------------------------------------------ #
    # Transition hooks — called from worker_factory at each state boundary.
    # ------------------------------------------------------------------ #
    def set_state(self, new_state: str) -> None:
        if new_state not in STATES:
            logger.warning("[Shadow] Ignoring unknown failover state %r", new_state)
            return
        with self._lock:
            now = time.time()
            if self._cur_state is not None and self._cur_state != new_state:
                # Record how long we sat in the state we're leaving.
                self._last_state_duration.labels(state=self._cur_state, **self._lv).set(
                    max(0.0, now - (self._cur_entered or now))
                )
                self._transitions.labels(
                    from_state=self._cur_state, to_state=new_state, **self._lv
                ).inc()
                self._state.labels(state=self._cur_state, **self._lv).set(0)
            self._state.labels(state=new_state, **self._lv).set(1)
            self._state_entered.labels(**self._lv).set(now)
            self._cur_state = new_state
            self._cur_entered = now
        logger.info(
            "[Shadow] failover_state engine=%s -> %s", self._engine_id, new_state
        )

    def record_switch_attempt(self) -> None:
        """A real failover was triggered (shadow won a contended lock)."""
        with self._lock:
            self._attempts_val += 1
            self._promotion_pending = True
            self._attempts.labels(**self._lv).inc()
            self._persist()

    def record_switch_success(self) -> None:
        """A real failover completed and the engine began serving.

        No-op unless a failover attempt is pending, so the initial bootup
        (which is not a contended promotion and records no attempt) is not
        miscounted as a successful switch.
        """
        with self._lock:
            if not self._promotion_pending:
                return
            self._promotion_pending = False
            self._success_val += 1
            self._successes.labels(**self._lv).inc()
            self._persist()

    # ------------------------------------------------------------------ #
    # Scrape bridge.
    # ------------------------------------------------------------------ #
    def _collect(self) -> str:
        with self._lock:
            return self._generate_latest(self._registry).decode("utf-8")

    def register(self, endpoint) -> None:
        endpoint.metrics.register_prometheus_expfmt_callback(self._collect)
        logger.info(
            "[Shadow] Registered failover metrics (engine=%s, persist=%s)",
            self._engine_id,
            self._persist_path,
        )


def create_failover_metrics(
    endpoint,
    engine_id: str,
    model_name: str,
    component_name: str,
    persist_dir: str,
) -> FailoverMetrics:
    """Build, register, and return a FailoverMetrics for this engine."""
    fm = FailoverMetrics(engine_id, model_name, component_name, persist_dir)
    fm.register(endpoint)
    return fm
