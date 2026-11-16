# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured efficiency feedback and whole-worker choices for KnowLine."""

from autoscaling_arena.adapters.outcome_feedback import OutcomeFeedback
from autoscaling_arena.worker_limits import WorkerLimits, resident_workers


class KnowLinePolicy:
    def __init__(self, mode, context, bounds, step):
        self.mode = mode
        self.step = step
        self.budget = context["gpu_budget"]
        engines = context["engines"]
        self.costs = (
            engines["prefill"]["gpus_per_worker"] if mode == "disagg" else 0,
            engines["decode" if mode == "disagg" else "aggregate"]["gpus_per_worker"],
        )
        if (
            type(self.budget) is not int
            or self.budget <= 0
            or any(
                type(c) is not int or c <= 0
                for c in (self.costs if mode == "disagg" else self.costs[1:])
            )
        ):
            raise ValueError("KnowLine requires a positive GPU budget and worker costs")
        limits = context.get("worker_limits")
        limits = WorkerLimits(**limits) if limits else WorkerLimits(1, self.budget)
        self.targets = limits.targets(
            topology=mode,
            gpu_budget=self.budget,
            prefill_gpus=self.costs[0],
            decode_gpus=self.costs[1],
            min_prefill=bounds["prefill"][0],
            max_prefill=bounds["prefill"][1],
            min_decode=bounds["decode"][0],
            max_decode=bounds["decode"][1],
        )
        if not self.targets:
            raise ValueError("KnowLine has no feasible fleet")
        self.feedback = OutcomeFeedback(
            slo=context["slo"], short_output_itl=context.get("short_output_itl", "skip")
        )

    def decision_inputs(self, snapshot, current, history):
        now = snapshot["now_s"]
        outcomes = self.feedback.view(now)
        latest = self.feedback.latest
        if latest is None:
            raise ValueError(
                "KnowLine requires native fleet telemetry before its first decision"
            )
        pair = (current.get("prefill", 0), current["decode"])
        resident = tuple(
            resident_workers(latest, role) for role in ("prefill", "decode")
        )
        requested = tuple(
            len(latest["active_" + role + "_ids"])
            + len(latest["starting_" + role + "_ids"])
            for role in ("prefill", "decode")
        )
        allocated = sum(n * c for n, c in zip(resident, self.costs))
        # A stale fleet snapshot must not create fictitious headroom. Replay
        # publishes telemetry before the tick at the same simulated timestamp.
        if requested != pair or pair not in self.targets or allocated > self.budget:
            raise ValueError(
                "KnowLine fleet telemetry disagrees with the current feasible fleet"
            )
        snapshot["outcomes"] = outcomes
        snapshot["allocation"] = {
            "resident_workers": {"prefill": resident[0], "decode": resident[1]},
            "resident_gpus": allocated,
            "free_gpus": self.budget - allocated,
            "includes": ["active", "starting", "draining"],
        }
        snapshot["previous_action"] = None
        if history:
            previous = history[-1]
            snapshot["previous_action"] = {
                "requested_targets": previous["targets"],
                "elapsed_s": now - previous["now_s"],
                "outcomes_before": previous.get("outcomes"),
                "outcomes_now": outcomes,
                "interpretation": "Observed outcomes can lag actions; this comparison is not a causal attribution.",
            }
        criteria = {}
        for p, d in self.targets:
            target = (p, d)
            if any(abs(n - old) > self.step for n, old in zip(target, pair)):
                continue
            added = sum(
                max(0, n - old) * c for n, old, c in zip(target, pair, self.costs)
            )
            if added > self.budget - allocated:
                continue
            criteria[f"p={p},d={d}"] = {
                "prefill_workers": p,
                "decode_workers": d,
                "target_gpus": sum(n * c for n, c in zip(target, self.costs)),
                "additional_gpus_now": added,
            }
        if not criteria:
            raise ValueError("KnowLine has no immediately admissible fleet")
        return {
            "fleet": {
                "type": "choice",
                "instructions": (
                    "Choose ONE joint fleet to maximize requests meeting ALL configured SLOs per GPU-second. "
                    "Use current.outcomes.goodput_per_gpu, good_rate, mean_ttft_ms, mean_itl_ms, "
                    "the current queue and traffic, and previous_action/history to judge efficiency. "
                    "Higher SLO success alone is not the objective: extra capacity must earn its GPU cost. "
                    "Options are absolute WHOLE worker counts; worker GPU costs differ by role. "
                    "All options satisfy the shared budget and worker bounds. Active, starting and draining GPUs count. "
                    "Removed workers retain GPUs while draining, so reductions cannot immediately fund additions. "
                    "Expected workers include starts. Allow startup time and avoid repeated actions before outcomes arrive. "
                    "Outcome metrics cover past completions, including failures, not the final arrival-cohort score; "
                    "long requests can complete after the action that served them. Null means unknown. "
                    "Prefer holding when evidence is insufficient."
                ),
                "criteria": criteria,
            }
        }
