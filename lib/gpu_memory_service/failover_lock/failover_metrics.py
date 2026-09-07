# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics for GMS shadow-engine failover.

Engine-agnostic: an engine integration constructs one :class:`FailoverMetrics`
per engine and drives it from its failover lifecycle. Values are set at state
transitions; the scrape callback only encodes the current registry.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# `dynamo_component_` matches the convention used by the other Dynamo engine
# metrics (see common/utils/prometheus.py); k8s/DGD/pod identity is added by the
# scrape relabeling, not baked in here.
_PREFIX = "dynamo_component_engine_failover"

STATES = ("init", "standby", "waking", "active")


class FailoverMetrics:
    """Per-engine failover metrics, exposed on the engine's system /metrics."""

    def __init__(
        self,
        engine_id: str,
        model_name: str,
        component_name: str,
        persist_dir: str,
    ) -> None:
        # Dedicated registry, independent of any engine multiproc registry.
        from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

        self._generate_latest = generate_latest
        self._registry = CollectorRegistry()
        # Guards the multi-step updates below (flipping the 1-hot gauge is
        # several set() calls; persist is read-then-write) so a concurrent
        # scrape can't observe a half-updated state. prometheus_client already
        # makes a single metric op thread-safe; this is for the compound ops.
        self._lock = threading.Lock()
        self._engine_id = str(engine_id)
        self._lv = {
            "engine_id": self._engine_id,
            "model": model_name or "",
            "dynamo_component": component_name or "",
        }
        base = ["engine_id", "model", "dynamo_component"]

        self._state = Gauge(
            f"{_PREFIX}_state",
            "Current failover state, 1-hot over the state label "
            "(init|standby|waking|active).",
            base + ["state"],
            registry=self._registry,
        )
        self._state_entered = Gauge(
            f"{_PREFIX}_state_entered_timestamp_seconds",
            "Unix time the engine entered its current failover state.",
            base,
            registry=self._registry,
        )
        self._last_state_duration = Gauge(
            f"{_PREFIX}_last_state_duration_seconds",
            "Duration of the most recent completed occupancy of each state "
            "({state=waking} is the wake/switch time).",
            base + ["state"],
            registry=self._registry,
        )
        self._transitions = Counter(
            f"{_PREFIX}_transitions_total",
            "Failover state transitions, by from_state/to_state.",
            base + ["from_state", "to_state"],
            registry=self._registry,
        )
        # Real failovers only (a shadow that won a *contended* lock); the initial
        # bootup acquires the lock immediately and is not counted. Derived
        # failures = attempts - success. These two are write-through persisted
        # (see _persist) so an increment made just before the process dies isn't
        # lost with an unscraped in-memory value; it is reloaded and re-exposed
        # on restart. This is a lightweight hedge against the death-before-scrape
        # gap -- a span/event is the fuller answer (future).
        self._attempts = Counter(
            f"{_PREFIX}_switch_attempts_total",
            "Failover promotions attempted (contended-lock wins).",
            base,
            registry=self._registry,
        )
        self._successes = Counter(
            f"{_PREFIX}_switch_success_total",
            "Failover promotions that completed and began serving.",
            base,
            registry=self._registry,
        )

        # Export zeros from the first scrape (zeros, not absent).
        for s in STATES:
            self._state.labels(state=s, **self._lv).set(0)
        self._attempts.labels(**self._lv)
        self._successes.labels(**self._lv)

        self._cur_state: str | None = None
        self._cur_entered: float | None = None
        self._attempts_val = 0
        self._success_val = 0

        self._persist_path = os.path.join(
            persist_dir, f"failover_metrics_engine-{self._engine_id}.json"
        )
        self._restore()

    # -- write-through durability for the switch counters ------------------- #
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
                "[Shadow] restored failover counters engine=%s attempts=%d success=%d",
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
            logger.warning("[Shadow] failed to persist failover counters: %s", e)

    # -- transition hooks, called from the engine at each state boundary ---- #
    def set_state(self, new_state: str) -> None:
        if new_state not in STATES:
            logger.warning("[Shadow] ignoring unknown failover state %r", new_state)
            return
        with self._lock:
            now = time.time()
            if self._cur_state is not None and self._cur_state != new_state:
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
        """Count a real failover attempt. Caller gates this on a contended lock."""
        with self._lock:
            self._attempts_val += 1
            self._attempts.labels(**self._lv).inc()
            self._persist()

    def record_switch_success(self) -> None:
        """Count a completed failover. Caller gates this on the same contended lock,
        so it always pairs with an attempt (keeps derived failures >= 0)."""
        with self._lock:
            self._success_val += 1
            self._successes.labels(**self._lv).inc()
            self._persist()

    # -- scrape bridge ------------------------------------------------------ #
    def _collect(self) -> str:
        with self._lock:
            return self._generate_latest(self._registry).decode("utf-8")

    def register(self, endpoint) -> None:
        endpoint.metrics.register_prometheus_expfmt_callback(self._collect)
        logger.info(
            "[Shadow] registered failover metrics (engine=%s, persist=%s)",
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
