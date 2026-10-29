# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dynamo_planner_retune_*`` Prometheus metrics of live retune.

``RetuneMetrics`` implements ``contracts.RetuneMetricsSink``: decision,
failure and skip counters, the supervisor's state, the scores of the latest
decision, router convergence, PATCH errors and trace health (design §4.10).
Every method is cheap and never raises, so callers on the event loop, in the
sweep thread or in the actuator need no guard.

The planner serves ``prometheus_client.REGISTRY`` on
``metric_reporting_prometheus_port``, so the default registry is the one to
use. The collectors are created once per registry and shared: building a
second ``RetuneMetrics`` on the same registry (a controller rebuild, a test)
does not fail with a duplicate registration.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from typing import Optional

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge

from dynamo.planner.core.types import RETUNE_OUTCOMES

logger = logging.getLogger(__name__)

PREFIX = "dynamo_planner_retune"
# Gate reasons (``contracts.GateReason``) and decision outcomes are known up
# front; their series start at 0 so ``increase()`` sees the first event.
_SKIP_REASONS = ("not_leader", "not_ready", "already_decided")
# Live outcomes recorded in the state history besides the controller's.
_LIVE_OUTCOMES = ("promoted", "rolled_back")


class _Collectors:
    """The metric objects of one registry."""

    def __init__(self, registry: CollectorRegistry) -> None:
        kw = {"registry": registry}
        self.decisions_total = Counter(
            f"{PREFIX}_decisions_total",
            "Retune decisions, counted once the state write persisted them.",
            labelnames=["outcome"],
            **kw,
        )
        self.failures_total = Counter(
            f"{PREFIX}_failures_total",
            "Failed retune windows and supervisor failures, by reason.",
            labelnames=["reason"],
            **kw,
        )
        self.skipped_total = Counter(
            f"{PREFIX}_skipped_total",
            "Due windows the launch gate held back, once per window and reason.",
            labelnames=["reason"],
            **kw,
        )
        self.leader = Gauge(
            f"{PREFIX}_leader",
            "1 while this planner holds the retune leadership Lease.",
            **kw,
        )
        self.ready = Gauge(
            f"{PREFIX}_ready",
            "1 while every spec and health premise holds, so windows launch.",
            **kw,
        )
        self.initialized = Gauge(
            f"{PREFIX}_initialized",
            "1 once the leader finished initializing from the persisted state.",
            **kw,
        )
        self.rollback_pending = Gauge(
            f"{PREFIX}_rollback_pending",
            "1 while routers are being reset to their startup parameters.",
            **kw,
        )
        self.controller_generation = Gauge(
            f"{PREFIX}_controller_generation",
            "Generation of the installed retune controller; it grows on each rebuild.",
            **kw,
        )
        self.baseline_score = Gauge(
            f"{PREFIX}_baseline_score",
            "Objective score of the current parameters on the latest decided "
            "window (NaN when not scored).",
            **kw,
        )
        self.best_score = Gauge(
            f"{PREFIX}_best_score",
            "Best candidate score on the latest decided window (NaN when none).",
            **kw,
        )
        self.sweep_seconds = Gauge(
            f"{PREFIX}_sweep_seconds",
            "Wall-clock duration of the latest sweep (NaN when it did not run).",
            **kw,
        )
        self.missing_candidates = Gauge(
            f"{PREFIX}_missing_candidates",
            "Candidates of the latest sweep without a score.",
            **kw,
        )
        self.router_converged = Gauge(
            f"{PREFIX}_router_converged",
            "1 when every frontend router holds the target policy parameters.",
            **kw,
        )
        self.router_patch_errors_total = Counter(
            f"{PREFIX}_router_patch_errors_total",
            "Failed /router_policy PATCH or reset calls, by HTTP status or kind.",
            labelnames=["code"],
            **kw,
        )
        self.trace_sink_missing = Gauge(
            f"{PREFIX}_trace_sink_missing",
            "1 for a Ready frontend pod without its own trace segment directory.",
            labelnames=["pod"],
            **kw,
        )
        self.trace_dropped_members_total = Counter(
            f"{PREFIX}_trace_dropped_members_total",
            "Trace gzip members dropped by window cuts for a bad CRC, size or "
            "truncation.",
            **kw,
        )
        for outcome in (*RETUNE_OUTCOMES, *_LIVE_OUTCOMES):
            self.decisions_total.labels(outcome=outcome)
        for reason in _SKIP_REASONS:
            self.skipped_total.labels(reason=reason)
        for gauge in (self.baseline_score, self.best_score, self.sweep_seconds):
            gauge.set(math.nan)


_LOCK = threading.Lock()
_BY_REGISTRY: weakref.WeakKeyDictionary[
    CollectorRegistry, _Collectors
] = weakref.WeakKeyDictionary()


def _collectors(registry: CollectorRegistry) -> _Collectors:
    with _LOCK:
        collectors = _BY_REGISTRY.get(registry)
        if collectors is None:
            collectors = _Collectors(registry)
            _BY_REGISTRY[registry] = collectors
        return collectors


def _label(value: object) -> str:
    text = str(value)
    return text if text else "unknown"


def _number(value: Optional[float]) -> float:
    return math.nan if value is None else float(value)


class RetuneMetrics:
    """``RetuneMetricsSink`` over Prometheus collectors on ``registry``.

    ``registry`` defaults to ``prometheus_client.REGISTRY``, which the
    planner's metrics server exposes. Instances on the same registry share
    their collectors.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._m = _collectors(REGISTRY if registry is None else registry)

    def decision(self, outcome: str) -> None:
        try:
            self._m.decisions_total.labels(outcome=_label(outcome)).inc()
        except Exception:
            logger.debug("retune decision metric failed", exc_info=True)

    def failure(self, reason: str) -> None:
        try:
            self._m.failures_total.labels(reason=_label(reason)).inc()
        except Exception:
            logger.debug("retune failure metric failed", exc_info=True)

    def skipped(self, reason: str) -> None:
        try:
            self._m.skipped_total.labels(reason=_label(reason)).inc()
        except Exception:
            logger.debug("retune skipped metric failed", exc_info=True)

    def supervisor_state(
        self,
        *,
        leader: bool,
        ready: bool,
        initialized: bool,
        rollback_pending: bool,
        generation: int,
    ) -> None:
        try:
            self._m.leader.set(1 if leader else 0)
            self._m.ready.set(1 if ready else 0)
            self._m.initialized.set(1 if initialized else 0)
            self._m.rollback_pending.set(1 if rollback_pending else 0)
            self._m.controller_generation.set(generation)
        except Exception:
            logger.debug("retune supervisor metrics failed", exc_info=True)

    def scores(
        self,
        *,
        baseline: Optional[float],
        best: Optional[float],
        sweep_seconds: Optional[float],
        missing_candidates: int,
    ) -> None:
        try:
            self._m.baseline_score.set(_number(baseline))
            self._m.best_score.set(_number(best))
            self._m.sweep_seconds.set(_number(sweep_seconds))
            self._m.missing_candidates.set(missing_candidates)
        except Exception:
            logger.debug("retune score metrics failed", exc_info=True)

    def router_converged(self, converged: bool) -> None:
        try:
            self._m.router_converged.set(1 if converged else 0)
        except Exception:
            logger.debug("retune router_converged metric failed", exc_info=True)

    def patch_error(self, code: str) -> None:
        try:
            self._m.router_patch_errors_total.labels(code=_label(code)).inc()
        except Exception:
            logger.debug("retune patch error metric failed", exc_info=True)

    def trace_sink_missing(self, pod: str, missing: bool) -> None:
        """Set the pod's series to 1, or drop it once the sink is back.

        Dropping instead of writing 0 keeps no series for pods that are gone.
        """

        try:
            if missing:
                self._m.trace_sink_missing.labels(pod=_label(pod)).set(1)
            else:
                try:
                    self._m.trace_sink_missing.remove(_label(pod))
                except KeyError:
                    pass
        except Exception:
            logger.debug("retune trace_sink_missing metric failed", exc_info=True)

    def dropped_members(self, count: int) -> None:
        try:
            if count > 0:
                self._m.trace_dropped_members_total.inc(count)
        except Exception:
            logger.debug("retune dropped members metric failed", exc_info=True)
