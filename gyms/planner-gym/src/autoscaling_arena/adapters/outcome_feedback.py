# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded completion-window feedback, available before each replay decision."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping

from autoscaling_arena.worker_limits import resident_workers


class OutcomeFeedback:
    """Exact past counters; completions may belong to earlier arrival intervals.

    This is online feedback, not the final arrival-cohort leaderboard score.
    GPU time is integrated by native replay, including starts and drains.
    """

    def __init__(self, *, slo, short_output_itl="skip", window_s=60.0):
        if short_output_itl not in {"fail", "skip"}:
            raise ValueError("invalid short-output ITL policy")
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("feedback window must be positive and finite")
        self.slo = {k: slo.get(k) for k in ("ttft_ms", "itl_ms", "e2e_ms")}
        self.short_output_itl = short_output_itl
        self.window_s = window_s
        self.latest = None
        self._counters = None
        self._windows = deque()

    def observe(self, sample):
        if isinstance(sample, (str, bytes, bytearray)):
            sample = json.loads(sample)
        if not isinstance(sample, Mapping):
            raise TypeError("telemetry must be a mapping")
        feedback = sample.get("completion_feedback")
        if not isinstance(feedback, Mapping) or feedback.get("schema_version") != 1:
            raise ValueError(
                "MPC and Cognitive Agent require completion feedback v1, which this "
                "replay runtime's telemetry does not carry; see docs/mpc.md"
            )
        if feedback["sla"] != self.slo:
            raise ValueError("completion-feedback SLO differs from controller SLO")
        at = sample["sampled_at_ms"] / 1000
        if not math.isfinite(at) or at < 0:
            raise ValueError("invalid telemetry time")
        if self.latest is not None:
            previous_at = self.latest["sampled_at_ms"] / 1000
            if at == previous_at and sample.get("kind") == "final":
                return
            if at <= previous_at:
                raise ValueError("feedback observations must advance in time")
        else:
            previous_at = sample["interval_start_ms"] / 1000
            if previous_at != 0 and sample.get("kind") != "baseline":
                raise ValueError("completion feedback requires the initial baseline")
        key = (
            "good_requests"
            if self.short_output_itl == "skip"
            else "good_requests_strict_itl"
        )
        counters = {
            k: feedback[k]
            for k in ("terminal_requests", "completed_requests", key, "gpu_seconds")
        }
        if any(
            not isinstance(v, (int, float))
            or isinstance(v, bool)
            or not math.isfinite(v)
            or v < 0
            for v in counters.values()
        ):
            raise ValueError("invalid completion counters")
        if any(type(counters[k]) is not int for k in counters if k != "gpu_seconds"):
            raise ValueError("request counters must be integers")
        if (
            not counters[key]
            <= counters["completed_requests"]
            <= counters["terminal_requests"]
        ):
            raise ValueError("inconsistent completion counters")
        before = self._counters or {k: 0 for k in counters}
        delta = {k: v - before[k] for k, v in counters.items()}
        if any(v < 0 for v in delta.values()):
            raise ValueError("completion counters went backwards")
        if not delta[key] <= delta["completed_requests"] <= delta["terminal_requests"]:
            raise ValueError("inconsistent completion deltas")
        pairs = feedback["latency_samples"]
        if len(pairs) > 64 or len(pairs) > feedback["latency_population"]:
            raise ValueError("invalid bounded latency sample")
        for pair in pairs:
            if (
                len(pair) != 2
                or any(
                    v is not None
                    and (
                        not isinstance(v, (int, float))
                        or isinstance(v, bool)
                        or not math.isfinite(v)
                        or v < 0
                    )
                    for v in pair
                )
                or pair[0] is None
            ):
                raise ValueError("invalid latency pair")
        for pool in ("prefill", "decode"):
            resident_workers(sample, pool)
        self._counters = counters
        self.latest = dict(sample)
        if at > previous_at:
            self._windows.append(
                {
                    "start_s": previous_at,
                    "end_s": at,
                    "terminal": delta["terminal_requests"],
                    "completed": delta["completed_requests"],
                    "good": delta[key],
                    "gpu_seconds": delta["gpu_seconds"],
                    "ttft_sum": sample["traffic"]["avg_ttft_ms"]
                    * sample["traffic"]["ttft_count"],
                    "ttft_count": sample["traffic"]["ttft_count"],
                    "itl_sum": sample["traffic"]["avg_itl_ms"]
                    * sample["traffic"]["itl_count"],
                    "itl_count": sample["traffic"]["itl_count"],
                }
            )
        while self._windows and self._windows[0]["end_s"] <= at - self.window_s:
            self._windows.popleft()

    def view(self, now_s):
        if self.latest is None:
            return None
        at = self.latest["sampled_at_ms"] / 1000
        if at > now_s + 1e-6:
            raise ValueError("future telemetry must not enter a decision")
        if now_s - at > 10 + 1e-6:
            raise ValueError("completion feedback is stale")
        if not self._windows:
            return None
        sums = {
            key: sum(w[key] for w in self._windows)
            for key in (
                "terminal",
                "completed",
                "good",
                "gpu_seconds",
                "ttft_sum",
                "ttft_count",
                "itl_sum",
                "itl_count",
            )
        }
        duration = at - self._windows[0]["start_s"]
        return {
            "basis": "past_completion_window",
            "start_s": self._windows[0]["start_s"],
            "end_s": at,
            "terminal_requests": sums["terminal"],
            "completed_requests": sums["completed"],
            "good_requests": sums["good"],
            "gpu_seconds": sums["gpu_seconds"],
            "good_rate": sums["good"] / sums["terminal"] if sums["terminal"] else None,
            "goodput_rps": sums["good"] / duration,
            "goodput_per_gpu": (
                sums["good"] / sums["gpu_seconds"] if sums["gpu_seconds"] else None
            ),
            "average_gpus": sums["gpu_seconds"] / duration,
            "mean_ttft_ms": (
                sums["ttft_sum"] / sums["ttft_count"] if sums["ttft_count"] else None
            ),
            "mean_itl_ms": (
                sums["itl_sum"] / sums["itl_count"] if sums["itl_count"] else None
            ),
            "short_output_itl": self.short_output_itl,
        }


class LatencyCalibration:
    """Bounded paired residuals learned only from stable, completed intervals."""

    def __init__(self, ttft_slo, itl_slo, *, short_output_itl="skip"):
        self.ttft_slo, self.itl_slo = ttft_slo, itl_slo
        self.short_output_itl = short_output_itl
        self.residuals = deque(maxlen=256)
        self.last_sample_s = None

    def observe(self, sample, prediction, fleet):
        at = sample["sampled_at_ms"] / 1000
        if self.last_sample_s == at:
            return
        self.last_sample_s = at
        while self.residuals and self.residuals[0][2] < at - 60:
            self.residuals.popleft()
        if prediction is None:
            return
        # A prediction belongs to the previous interval's fleet. Never attribute
        # a mixed startup/drain interval's completions to the new action alone.
        actual_fleet = tuple(
            len(sample["active_" + role + "_ids"]) for role in ("prefill", "decode")
        )
        if (
            prediction[0] != fleet
            or actual_fleet != fleet
            or sample["interval_start_ms"] / 1000 < prediction[3]
            or any(
                sample[state + "_" + role + "_ids"]
                for state in ("starting", "draining")
                for role in ("prefill", "decode")
            )
        ):
            return
        pt, pi = prediction[1:3]
        if pt <= 0 or pi <= 0:
            return
        for ttft, itl in sample["completion_feedback"]["latency_samples"]:
            self.residuals.append((ttft / pt, None if itl is None else itl / pi, at))

    def fraction(self, ttft, itl, prior):
        if not self.residuals:
            return prior
        empirical = sum(
            ttft * rt <= self.ttft_slo
            and (
                (self.short_output_itl == "skip")
                if ri is None
                else itl * ri <= self.itl_slo
            )
            for rt, ri, _ in self.residuals
        ) / len(self.residuals)
        weight = min(1.0, len(self.residuals) / 32)
        return weight * empirical + (1 - weight) * prior

    def mean_itl(self, prediction):
        ratios = [ri for _, ri, _ in self.residuals if ri is not None]
        return prediction * sum(ratios) / len(ratios) if ratios else prediction
