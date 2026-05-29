# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Workload-shape verification check.

Used by the H1 (radix-tree retention) discriminator pair:
- ``no_prefix`` (L1) must produce many unique prompts (~6800) → max tree growth
- ``same_prefix`` (L2) must collapse to a single shared prompt → min tree growth

If the workload shape does NOT match what the YAML promised, any downstream
slope comparison is meaningless. This check is intentionally FATAL — it
raises AssertionError on mismatch so the test surfaces an invalid run before
anyone reads MB/min numbers off of it.

AIPerf 0.7.x ``profile_export.jsonl`` schema (per-request records):
  - ``metadata.conversation_id`` — stable per-prompt identifier
    (e.g. ``session_001309``). This is the "distinct prompt" key —
    AIPerf reuses conversation_ids when the dataset is exhausted /
    shared, and the field is present even though the raw prompt text
    is not exported.
  - ``metrics.input_sequence_length.value`` — ISL in tokens per request.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from glob import escape as _glob_escape
from glob import glob as _glob
from typing import Optional

from tests.fault_tolerance.deploy.checks import Check


def _load_dirs_matching(ctx, load_name: Optional[str]) -> list:
    """Return load-*/ subdirs under ctx.log_dir whose dirname contains
    ``load_name``. Mirrors the convention used by LoadApplied (which
    filters by basename substring).
    """
    log_dir = getattr(ctx, "log_dir", None)
    if not log_dir:
        return []
    base = os.path.join(_glob_escape(log_dir), "load", "load-*")
    out = sorted(_glob(base))
    out = [p for p in out if os.path.isdir(p)]
    if load_name is None:
        return out
    return [p for p in out if load_name in os.path.basename(p)]


def _iter_profile_records(load_dir: str):
    """Yield parsed per-request JSON records from profile_export.jsonl."""
    path = os.path.join(load_dir, "profile_export.jsonl")
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class WorkloadShapeVerified(Check):
    """Verify that the load harness actually produced the workload shape
    the YAML promised — checked by counting distinct ``conversation_id``s
    and computing the ISL distribution from AIPerf's per-request JSONL.

    Parameters (all keyword):
      - ``load_name``: which rung to validate (matched against load-* dir
        basename).
      - ``min_unique_prompts``: fail if observed distinct conversation_ids
        is below this floor. Use on ``no_prefix`` to assert "random
        prompts actually were random."
      - ``max_unique_prompts``: fail if observed distinct conversation_ids
        is above this ceiling. Use on ``same_prefix`` to assert "the
        shared system prompt actually collapsed prompt variety."
      - ``isl_mean_min`` / ``isl_mean_max``: sanity-check that the ISL
        distribution didn't drift from cycle-6 baseline. The
        no_prefix-vs-same_prefix experiment ONLY varies prefix uniqueness;
        if ISL also changed, the slope delta is confounded.

    Behaviour: logs observed unique-prompt count, ISL mean, ISL p99 at
    INFO. Raises AssertionError when any active bound is violated.
    """

    load_name: str = "steady"
    min_unique_prompts: Optional[int] = None
    max_unique_prompts: Optional[int] = None
    isl_mean_min: Optional[float] = None
    isl_mean_max: Optional[float] = None

    def validate(self, ctx) -> None:
        load_dirs = _load_dirs_matching(ctx, self.load_name)
        assert load_dirs, (
            f"WorkloadShapeVerified: no load-*/ dir matched "
            f"load_name={self.load_name!r} under {getattr(ctx, 'log_dir', None)!r}"
        )

        conv_ids: set = set()
        isl_values: list = []
        total_records = 0
        for ldir in load_dirs:
            for rec in _iter_profile_records(ldir):
                total_records += 1
                meta = rec.get("metadata") or {}
                cid = meta.get("conversation_id")
                if cid is not None:
                    conv_ids.add(cid)
                metrics = rec.get("metrics") or {}
                isl_node = metrics.get("input_sequence_length") or {}
                isl_val = isl_node.get("value")
                if isl_val is not None:
                    try:
                        isl_values.append(float(isl_val))
                    except (TypeError, ValueError):
                        continue

        assert total_records > 0, (
            f"WorkloadShapeVerified: profile_export.jsonl had 0 parseable "
            f"records under load_name={self.load_name!r} (dirs={load_dirs})"
        )

        unique_prompts = len(conv_ids)
        if isl_values:
            isl_values_sorted = sorted(isl_values)
            isl_mean = statistics.fmean(isl_values_sorted)
            # p99 via nearest-rank — fmean+sort is enough for a sanity gate.
            p99_idx = max(0, int(round(0.99 * len(isl_values_sorted))) - 1)
            isl_p99 = isl_values_sorted[p99_idx]
        else:
            isl_mean = float("nan")
            isl_p99 = float("nan")

        ctx.logger.info(
            f"WorkloadShapeVerified[{self.load_name}]: "
            f"records={total_records} unique_prompts={unique_prompts} "
            f"isl_mean={isl_mean:.1f} isl_p99={isl_p99:.1f} "
            f"(bounds: min_unique={self.min_unique_prompts} "
            f"max_unique={self.max_unique_prompts} "
            f"isl_mean_min={self.isl_mean_min} isl_mean_max={self.isl_mean_max})"
        )

        if self.min_unique_prompts is not None:
            assert unique_prompts >= self.min_unique_prompts, (
                f"WorkloadShapeVerified: observed {unique_prompts} unique "
                f"conversation_ids on load {self.load_name!r}; expected "
                f">= {self.min_unique_prompts}. Workload-shape contract "
                f"BROKEN — downstream slope numbers are invalid."
            )
        if self.max_unique_prompts is not None:
            assert unique_prompts <= self.max_unique_prompts, (
                f"WorkloadShapeVerified: observed {unique_prompts} unique "
                f"conversation_ids on load {self.load_name!r}; expected "
                f"<= {self.max_unique_prompts}. Workload-shape contract "
                f"BROKEN — downstream slope numbers are invalid."
            )
        if self.isl_mean_min is not None:
            assert isl_mean >= self.isl_mean_min, (
                f"WorkloadShapeVerified: observed ISL mean={isl_mean:.1f} "
                f"on load {self.load_name!r}; expected >= {self.isl_mean_min}. "
                f"ISL distribution drifted from baseline — slope delta is "
                f"confounded by length, not just prefix uniqueness."
            )
        if self.isl_mean_max is not None:
            assert isl_mean <= self.isl_mean_max, (
                f"WorkloadShapeVerified: observed ISL mean={isl_mean:.1f} "
                f"on load {self.load_name!r}; expected <= {self.isl_mean_max}. "
                f"ISL distribution drifted from baseline — slope delta is "
                f"confounded by length, not just prefix uniqueness."
            )

    @property
    def description(self) -> str:
        bounds = []
        if self.min_unique_prompts is not None:
            bounds.append(f"unique >= {self.min_unique_prompts}")
        if self.max_unique_prompts is not None:
            bounds.append(f"unique <= {self.max_unique_prompts}")
        if self.isl_mean_min is not None:
            bounds.append(f"isl_mean >= {self.isl_mean_min}")
        if self.isl_mean_max is not None:
            bounds.append(f"isl_mean <= {self.isl_mean_max}")
        joined = "; ".join(bounds) if bounds else "no active bounds"
        return f"Workload shape verified on load {self.load_name!r} ({joined})"
