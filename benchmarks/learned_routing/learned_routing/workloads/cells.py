# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate cells and the split manifest (PLAN "Splits", CONTRACT "Cell spec").

Usage::

    python -m learned_routing.workloads.cells --campaign-root CR

Writes ``CR/cells/{train,val,test}.candidates.jsonl`` and ``CR/cells/SPLIT_MANIFEST.json`` and
materializes every derived trace the cells reference under ``CR/traces/derived``. ``load.value`` and
the SLA are left null for calibration, which also converts ``measure_trace`` (trace time) into the
harness ``measure`` window (replay time) once it has chosen the load.

**Unit of independence.** Every cell carries a ``segment``: a Mooncake time window, an AgentX play
subset, a synthetic-session generator seed, or a FAST25-synthetic window. Cells sharing a segment are
one workload. FAST25 conversation shares Mooncake's arrival and length skeleton, so its cells reuse
the segment id of the Mooncake window they align with.

**Split design** (the plan is an explicit table, so it can be audited line by line):

- Mooncake: six disjoint, self-contained 10-minute slices ``[10k, 10k + 10)`` minutes, each a
  4-minute warm-up plus a 6-minute measurement window ``[10k + 4, 10k + 10)``. Windows w0, w2, w4
  train; w1 validates; w3, w5 test. No trace row is in two slices, so no split replays another
  split's rows, even as warm-up (build audit S1: the earlier 12-minute slices every 8 minutes made
  each window's warm-up the previous window's measurement, across splits).
- AgentX: the 82 complete 128K plays are dealt, stratified by explicit-subagent presence, into
  three train subsets of 11, three validation subsets of 7 and four test subsets of 7. Every
  AgentX trace caps idle gaps at 300 s (``think_cap_s``).
- Synthetic sessions: generator seeds 0-3 train, 4-5 validate, 6-9 test.
- FAST25 conversation (aligned to w3, w5) and FAST25 synthetic (two disjoint windows, each with
  its own 3-minute warm-up) are test-only held-out families. Toolagent is excluded (a relabeled
  Mooncake copy).
- Closed-loop cells (Mooncake, FAST25, synthetic sessions) exclude their warm-up by identity:
  ``measure.warmup_trace_ms`` (see :mod:`learned_routing.goodput`) never scores a request of a
  session that arrives in the slice's warm-up, whatever the closed-loop dispatch order.
- Every completion-basis cell (closed loop and AgentX lanes) ends its window at full occupancy
  (``measure.end: full_occupancy``): the last instant at which all ``load.value`` sessions or
  lanes were occupied. Lanes run out of plays at very different times, so the former end, the
  last dispatch, scored mostly drain (build audit r1 F1). Lanes cells still need a replay-time
  ``measure.warmup_ms`` from calibration.
- Worker counts: train {4, 8}; validation {4, 6, 8}; test {2, 6, 16, 32} plus held-out cells at
  {4, 8}. N = 6 is in validation and test, so test N = 6 cells carry ``selection_exposed``.
  AgentX has no N = 16 or 32 cells: 82 unique plays cannot load 16-32 workers at one or more lanes
  per worker with lanes below plays, and recycling plays would replay the same workload.
- Transform ranges: train isl_unique/isl_prefix in {1, 1.5}, osl in {0.5, 1, 2}, prefix_root in
  {1, 2}, think in {0.5, 1, 2}; validation interpolates (1.25, 1.5); test extrapolates (2.5, 4.0,
  0.25, root 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .common import (
    MAX_MODEL_LEN,
    campaign_root,
    canonical_json,
    sha256_file,
    write_atomic,
    write_json,
)
from .transform import Derived, TransformSpec, materialize

CELLS_VERSION = "lr-cells-v3"
MINUTE_MS = 60_000
KV_TOKENS_PER_WORKER = 301_808
CONV_TIME_SCALE = (
    3_536_999 / 3_600_000
)  # FAST25 conversation/toolagent grid vs Mooncake (audit r1 F1)
AGENTX_THINK_CAP_S = 300.0
AGENTX_SPLIT_SEED = "agentx-split-v1"
WORKERS = {"train": (4, 8), "val": (4, 6, 8), "test": (2, 4, 6, 8, 16, 32)}

# Replay cost in milliseconds per request, in-process, from the build-workloads smoke
# (runs/build_workloads/smoke/cost.json), rounded up: model = 0.3 + rows * coeff / 1000. With a
# per-cell replay check (--validation-json), expected_cost_s = max(model, 1.5 * measured wall).
COST_MS_PER_REQUEST = {
    "mooncake": 0.6,
    "fast25_conversation": 0.6,
    "fast25_synthetic": 0.6,
    "synthetic_sessions": 0.7,
    "agentx": 6.0,
}
MEASURED_COST_MARGIN = 1.5


@dataclass(frozen=True)
class Source:
    family: str
    path: str  # relative to CR
    trace_format: str
    block_size: int


SOURCES = {
    "mooncake": Source(
        "mooncake", "traces/mooncake/mooncake_trace.jsonl", "mooncake", 512
    ),
    "fast25_conversation": Source(
        "fast25_conversation", "traces/fast25/conversation_trace.jsonl", "mooncake", 512
    ),
    "fast25_synthetic": Source(
        "fast25_synthetic", "traces/fast25/synthetic_trace.jsonl", "mooncake", 512
    ),
    "agentx": Source("agentx", "traces/agentx/plays", "weka", 64),
}


MOONCAKE_SPLITS = ("train", "val", "train", "test", "train", "test")
WINDOWED_FAMILIES = ("mooncake", "fast25_conversation", "fast25_synthetic")


def mooncake_window(k: int) -> dict:
    """Window ``wk``: slice ``[10k, 10k + 10)`` minutes, measurement ``[10k + 4, 10k + 10)``."""
    return {
        "slice": (10 * k * MINUTE_MS, (10 + 10 * k) * MINUTE_MS),
        "measure": ((4 + 10 * k) * MINUTE_MS, (10 + 10 * k) * MINUTE_MS),
    }


def segments() -> dict:
    """Segment id -> definition, split and family."""
    out = {}
    for k, split in enumerate(MOONCAKE_SPLITS):
        out[f"mooncake:w{k}"] = {
            "family": "mooncake",
            "split": split,
            **mooncake_window(k),
        }
    for k in (3, 5):
        window = mooncake_window(k)
        out[f"conv:w{k}"] = {
            "family": "fast25_conversation",
            "split": "test",
            "slice": tuple(t * CONV_TIME_SCALE for t in window["slice"]),
            "measure": tuple(t * CONV_TIME_SCALE for t in window["measure"]),
            "segment": f"mooncake:w{k}",
        }
    for k, (lo, hi) in enumerate(((0.0, 8.5), (8.5, 17.0))):
        out[f"fast25_synthetic:x{k}"] = {
            "family": "fast25_synthetic",
            "split": "test",
            "slice": (int(lo * MINUTE_MS), int(hi * MINUTE_MS)),
            "measure": (int((lo + 3) * MINUTE_MS), int(hi * MINUTE_MS)),
        }
    for seed in range(10):
        split = "train" if seed < 4 else "val" if seed < 6 else "test"
        out[f"sessions:s{seed}"] = {
            "family": "synthetic_sessions",
            "split": split,
            "path": f"traces/synthetic_sessions/sessions_seed{seed}.jsonl",
            "measure": (180_000, 720_000),
        }
    return out


AGENTX_SEGMENTS = (
    ("A1", "train", 11),
    ("A2", "train", 11),
    ("A3", "train", 11),
    ("V1", "val", 7),
    ("V2", "val", 7),
    ("V3", "val", 7),
    ("T1", "test", 7),
    ("T2", "test", 7),
    ("T3", "test", 7),
    ("T4", "test", 7),
)


def allocate_plays(
    plays: dict[str, bool], seed: str = AGENTX_SPLIT_SEED
) -> dict[str, list[str]]:
    """Deal plays into AGENTX_SEGMENTS, stratified by explicit-subagent presence.

    ``plays`` maps file name -> has explicit subagents. Plays are ordered by a seeded hash within each
    stratum (subagent plays first), then dealt round-robin over the segments that still have room.
    """
    capacity = {name: size for name, _, size in AGENTX_SEGMENTS}
    if sum(capacity.values()) != len(plays):
        raise ValueError(
            f"segments hold {sum(capacity.values())} plays, corpus has {len(plays)}"
        )

    def key(name: str) -> str:
        return hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()

    order = sorted((n for n, sub in plays.items() if sub), key=key) + sorted(
        (n for n, sub in plays.items() if not sub), key=key
    )
    out: dict[str, list[str]] = {name: [] for name in capacity}
    names = [name for name, _, _ in AGENTX_SEGMENTS]
    cursor = 0
    for play in order:
        while (
            len(out[names[cursor % len(names)]]) >= capacity[names[cursor % len(names)]]
        ):
            cursor += 1
        out[names[cursor % len(names)]].append(play)
        cursor += 1
    return {name: sorted(members) for name, members in out.items()}


@dataclass(frozen=True)
class Plan:
    """One candidate cell: segment, transform tag, worker count, load mode and level, axis."""

    segment: str
    tag: str
    n: int
    mode: str  # open | closed | lanes
    level: str
    axis: str | None = None
    extra_axes: tuple[str, ...] = ()


# Transform tags -> TransformSpec overrides (AgentX also gets think_cap_s).
TRANSFORMS = {
    "base": {},
    "islu1.25": {"isl_unique_mult": 1.25},
    "islu1.5": {"isl_unique_mult": 1.5},
    "islu2.5": {"isl_unique_mult": 2.5},
    "islp1.5": {"isl_prefix_mult": 1.5},
    "islp2.5": {"isl_prefix_mult": 2.5},
    "osl0.25": {"osl_mult": 0.25},
    "osl0.5": {"osl_mult": 0.5},
    "osl1.5": {"osl_mult": 1.5},
    "osl2.0": {"osl_mult": 2.0},
    "osl2.5": {"osl_mult": 2.5},
    "osl4.0": {"osl_mult": 4.0},
    "root2": {"prefix_root_mult": 2},
    "root4": {"prefix_root_mult": 4},
    "think0.25": {"think_mult": 0.25},
    "think0.5": {"think_mult": 0.5},
    "think1.5": {"think_mult": 1.5},
    "think2.0": {"think_mult": 2.0},
    "think4.0": {"think_mult": 4.0},
}

# Ranges seen in training, per transform parameter (identity included); test extrapolation tags
# must fall outside them.
TRAIN_RANGES = {
    "isl_unique_mult": (1.0, 1.5),
    "isl_prefix_mult": (1.0, 1.5),
    "osl_mult": (0.5, 2.0),
    "prefix_root_mult": (1, 2),
    "think_mult": (0.5, 2.0),
}

TRAIN = [
    Plan("mooncake:w0", "base", 4, "open", "L1"),
    Plan("mooncake:w0", "base", 4, "open", "L3"),
    Plan("mooncake:w0", "base", 8, "open", "L2"),
    Plan("mooncake:w0", "base", 8, "closed", "L2"),
    Plan("mooncake:w2", "base", 4, "open", "L2"),
    Plan("mooncake:w2", "base", 8, "open", "L3"),
    Plan("mooncake:w2", "base", 4, "closed", "L3"),
    Plan("mooncake:w2", "base", 8, "closed", "L1"),
    Plan("mooncake:w4", "base", 8, "open", "L1"),
    Plan("mooncake:w4", "base", 4, "closed", "L2"),
    Plan("mooncake:w0", "islu1.5", 4, "open", "L2"),
    Plan("mooncake:w2", "islu1.5", 8, "closed", "L2"),
    Plan("mooncake:w4", "islp1.5", 8, "open", "L2"),
    Plan("mooncake:w0", "islp1.5", 4, "closed", "L2"),
    Plan("mooncake:w2", "root2", 4, "open", "L2"),
    Plan("mooncake:w4", "root2", 8, "open", "L3"),
    Plan("mooncake:w4", "osl2.0", 4, "open", "L2"),
    Plan("mooncake:w0", "osl0.5", 8, "open", "L2"),
    Plan("sessions:s0", "base", 4, "open", "L2"),
    Plan("sessions:s0", "base", 8, "closed", "L2"),
    Plan("sessions:s1", "base", 8, "open", "L1"),
    Plan("sessions:s1", "base", 4, "open", "L3"),
    Plan("sessions:s2", "think0.5", 8, "open", "L2"),
    Plan("sessions:s2", "think2.0", 4, "closed", "L2"),
    Plan("sessions:s3", "root2", 4, "open", "L2"),
    Plan("sessions:s3", "islu1.5", 8, "open", "L3"),
    Plan("agentx:A1", "base", 4, "lanes", "L1"),
    Plan("agentx:A1", "base", 4, "lanes", "L3"),
    Plan("agentx:A2", "base", 4, "lanes", "L2"),
    Plan("agentx:A3", "base", 8, "lanes", "L2"),
    Plan("agentx:A1+A2", "base", 8, "lanes", "L3"),
    Plan("agentx:A2", "think0.5", 4, "lanes", "L2"),
    Plan("agentx:A3", "think2.0", 8, "lanes", "L1"),
    Plan("agentx:A3", "osl1.5", 4, "lanes", "L2"),
]

VAL = [
    Plan("mooncake:w1", "base", 4, "open", "L2"),
    Plan("mooncake:w1", "base", 8, "closed", "L2"),
    Plan("mooncake:w1", "base", 6, "open", "L2"),
    Plan("mooncake:w1", "base", 8, "open", "L3"),
    Plan("mooncake:w1", "islu1.25", 6, "closed", "L2"),
    Plan("sessions:s5", "base", 4, "closed", "L1"),
    Plan("sessions:s4", "base", 4, "open", "L2"),
    Plan("sessions:s4", "base", 8, "closed", "L2"),
    Plan("sessions:s5", "think1.5", 6, "open", "L2"),
    Plan("sessions:s5", "base", 8, "open", "L3"),
    Plan("agentx:V1", "base", 4, "lanes", "L2"),
    Plan("agentx:V2", "base", 6, "lanes", "L2"),
    Plan("agentx:V3", "base", 8, "lanes", "L1"),
    Plan("agentx:V2", "base", 4, "lanes", "L3"),
]

TEST = [
    # Held-out Mooncake time windows at the training worker counts.
    Plan("mooncake:w3", "base", 4, "open", "L2", "time_window"),
    Plan("mooncake:w3", "base", 8, "closed", "L2", "time_window"),
    Plan("mooncake:w5", "base", 8, "open", "L2", "time_window"),
    Plan("mooncake:w5", "base", 4, "open", "L3", "time_window"),
    Plan("mooncake:w3", "base", 8, "open", "L1", "time_window"),
    Plan("mooncake:w5", "base", 4, "closed", "L3", "time_window"),
    # Transform values outside the training range (on held-out segments).
    Plan(
        "mooncake:w3",
        "islu2.5",
        8,
        "open",
        "L2",
        "transform_extrapolation",
        ("time_window",),
    ),
    Plan(
        "mooncake:w5",
        "islp2.5",
        4,
        "open",
        "L2",
        "transform_extrapolation",
        ("time_window",),
    ),
    Plan(
        "mooncake:w3",
        "osl4.0",
        4,
        "open",
        "L2",
        "transform_extrapolation",
        ("time_window",),
    ),
    Plan(
        "mooncake:w5",
        "root4",
        8,
        "open",
        "L2",
        "transform_extrapolation",
        ("time_window",),
    ),
    Plan(
        "mooncake:w5",
        "osl0.25",
        8,
        "closed",
        "L2",
        "transform_extrapolation",
        ("time_window",),
    ),
    Plan(
        "sessions:s6",
        "think4.0",
        4,
        "open",
        "L2",
        "transform_extrapolation",
        ("workload_draw",),
    ),
    Plan(
        "sessions:s7",
        "think0.25",
        8,
        "open",
        "L2",
        "transform_extrapolation",
        ("workload_draw",),
    ),
    Plan(
        "sessions:s8",
        "islu2.5",
        8,
        "closed",
        "L2",
        "transform_extrapolation",
        ("workload_draw",),
    ),
    Plan(
        "agentx:T1",
        "think4.0",
        4,
        "lanes",
        "L2",
        "transform_extrapolation",
        ("agentx_plays",),
    ),
    Plan(
        "agentx:T2",
        "osl2.5",
        8,
        "lanes",
        "L2",
        "transform_extrapolation",
        ("agentx_plays",),
    ),
    # Held-out plays, generator draws and families.
    Plan("agentx:T1", "base", 4, "lanes", "L2", "agentx_plays"),
    Plan("agentx:T2", "base", 8, "lanes", "L1", "agentx_plays"),
    Plan("agentx:T3", "base", 4, "lanes", "L3", "agentx_plays"),
    Plan("agentx:T4", "base", 8, "lanes", "L2", "agentx_plays"),
    Plan("sessions:s6", "base", 4, "open", "L2", "workload_draw"),
    Plan("sessions:s7", "base", 8, "closed", "L2", "workload_draw"),
    Plan("sessions:s8", "base", 8, "open", "L3", "workload_draw"),
    Plan("sessions:s9", "base", 4, "open", "L1", "workload_draw"),
    Plan("conv:w3", "base", 4, "open", "L2", "family"),
    Plan("conv:w5", "base", 8, "open", "L2", "family"),
    Plan("conv:w3", "base", 8, "closed", "L2", "family"),
    Plan("fast25_synthetic:x0", "base", 4, "open", "L2", "family"),
    Plan("fast25_synthetic:x1", "base", 8, "open", "L2", "family"),
    Plan("fast25_synthetic:x0", "base", 8, "closed", "L2", "family"),
]
for _n in (2, 6, 16, 32):
    TEST += [
        Plan("mooncake:w3", "base", _n, "open", "L2", "worker_count", ("time_window",)),
        Plan(
            "mooncake:w5", "base", _n, "closed", "L2", "worker_count", ("time_window",)
        ),
        Plan("mooncake:w5", "base", _n, "open", "L3", "worker_count", ("time_window",)),
        Plan(
            "sessions:s6", "base", _n, "open", "L2", "worker_count", ("workload_draw",)
        ),
        Plan(
            "sessions:s9",
            "base",
            _n,
            "closed",
            "L2",
            "worker_count",
            ("workload_draw",),
        ),
    ]
    if _n in (2, 32):
        TEST.append(
            Plan(
                "mooncake:w3",
                "base",
                _n,
                "open",
                "L1",
                "worker_count",
                ("time_window",),
            )
        )
    if _n == 2:
        TEST += [
            Plan(
                "agentx:T3", "base", 2, "lanes", "L2", "worker_count", ("agentx_plays",)
            ),
            Plan(
                "agentx:T4", "base", 2, "lanes", "L3", "worker_count", ("agentx_plays",)
            ),
        ]
    if _n == 6:
        TEST += [
            Plan(
                "agentx:T3", "base", 6, "lanes", "L2", "worker_count", ("agentx_plays",)
            ),
            Plan(
                "agentx:T1", "base", 6, "lanes", "L3", "worker_count", ("agentx_plays",)
            ),
        ]

PLANS = {"train": TRAIN, "val": VAL, "test": TEST}
MODE = {
    "open": "open_speedup",
    "closed": "closed_concurrency",
    "lanes": "agentic_lanes",
}


def measure_rule(mode: str) -> dict:
    """The load-independent part of a cell's harness ``measure`` for a plan mode."""
    if mode == "open":
        return {"basis": "arrival"}
    return {"basis": "completion", "end": "full_occupancy"}


@dataclass
class Builder:
    cr: Path
    plays: dict[str, list[str]] = field(default_factory=dict)
    derived: dict[str, Derived] = field(default_factory=dict)
    measured: dict[str, dict] = field(default_factory=dict)

    def rel(self, path: Path) -> str:
        return "CR/" + str(Path(path).resolve().relative_to(self.cr))

    def segment_def(self, segment: str) -> dict:
        if segment.startswith("agentx:"):
            names = [
                n
                for part in segment.split(":", 1)[1].split("+")
                for n in self.plays[part]
            ]
            split = {
                s
                for name, s, _ in AGENTX_SEGMENTS
                for part in segment.split(":", 1)[1].split("+")
                if name == part
            }
            if len(split) != 1:
                raise ValueError(f"{segment}: play subsets from several splits")
            return {"family": "agentx", "split": split.pop(), "plays": names}
        return segments()[segment]

    def spec_for(self, segment: str, tag: str) -> tuple[Source, TransformSpec]:
        definition = self.segment_def(segment)
        family = definition["family"]
        overrides = dict(TRANSFORMS[tag])
        if family == "agentx":
            source = SOURCES["agentx"]
            spec = {
                "plays": definition["plays"],
                "think_cap_s": AGENTX_THINK_CAP_S,
                **overrides,
            }
        elif family == "synthetic_sessions":
            source = Source(family, definition["path"], "mooncake", 64)
            spec = overrides
        else:
            source = SOURCES[family]
            spec = {"window": list(definition["slice"]), **overrides}
        return source, TransformSpec.from_dict(spec)

    def derive(self, segment: str, tag: str) -> tuple[Source, Derived]:
        source, spec = self.spec_for(segment, tag)
        key = canonical_json({"source": source.path, "spec": spec.to_dict()})
        if key not in self.derived:
            self.derived[key] = materialize(
                self.cr / source.path,
                source.trace_format,
                spec,
                source.block_size,
                self.cr / "traces" / "derived",
            )
        return source, self.derived[key]

    def measure_trace(self, segment: str, derived: Derived, mode: str) -> dict:
        definition = self.segment_def(segment)
        realized = derived.meta["realized"]
        if definition["family"] == "agentx":
            return {
                "basis": "completion",
                "plays": realized["plays"],
                "requests": realized["requests"],
                "max_lanes": realized["plays"],
                "lanes_below_plays_max": max(1, realized["plays"] - 1),
                "note": "lanes run plays sequentially (play_index % lanes); lanes >= plays equals synchronized-start "
                "timestamp mode (audit r1 F3), so use lanes < plays; AISim adds primers and 10 warm-up requests per lane. "
                "The window ends at full occupancy (measure.end, the first lane out of plays); calibration sets "
                "measure.warmup_ms in replay time",
            }
        path = derived.path
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        first_arrivals = [float(r["timestamp"]) for r in rows if "timestamp" in r]
        origin = min(first_arrivals)
        if definition["family"] == "synthetic_sessions":
            start, end = definition["measure"]
            rebase = 0.0
        else:
            start, end = definition["measure"]
            rebase = derived.meta["transform_stats"]["extra"]["window_rebase_ms"]
        warmup = start - rebase - origin
        window = end - start
        in_warmup = sum(1 for t in first_arrivals if t - origin < warmup)
        in_window = sum(
            1 for t in first_arrivals if warmup <= t - origin < warmup + window
        )
        return {
            "basis": "arrival" if mode == "open" else "completion",
            "time_base": "trace milliseconds from the trace's first arrival, before arrival_speedup_ratio",
            "warmup_ms": warmup,
            "window_ms": window,
            "sessions_in_warmup": in_warmup,
            "sessions_in_window": in_window,
            "sessions_after_window": len(first_arrivals) - in_warmup - in_window,
            "rows": len(rows),
            "note": "open loop: set measure.warmup_ms = warmup_ms / speedup and measure.window_ms = window_ms / speedup; "
            "closed loop: measure.warmup_trace_ms = warmup_ms excludes the warm-up sessions by identity",
        }

    def expected_cost(
        self, cell_id: str, trace_sha: str, family: str, rows: int
    ) -> float:
        model = 0.3 + rows * COST_MS_PER_REQUEST[family] / 1000.0
        got = self.measured.get(cell_id)
        if got and got.get("trace_sha256") == trace_sha and not got.get("error"):
            model = max(model, MEASURED_COST_MARGIN * float(got["wall_s"]))
        return round(model, 2)

    def cell(self, split: str, plan: Plan) -> dict:
        source, derived = self.derive(plan.segment, plan.tag)
        definition = self.segment_def(plan.segment)
        family = definition["family"]
        segment = definition.get("segment", plan.segment)
        mode = MODE[plan.mode]
        cell_id = f"{plan.segment.replace(':', '-').replace('+', '_')}-{plan.tag}-n{plan.n}-{plan.mode}-{plan.level}"
        rows = derived.meta["realized"].get(
            "rows", derived.meta["realized"].get("requests")
        )
        measure_trace = self.measure_trace(plan.segment, derived, plan.mode)
        cell = {
            "cell_id": cell_id,
            "split": split,
            "family": family,
            "segment": segment,
            "trace_files": [self.rel(derived.path)],
            "trace_sha256": [derived.sha256],
            "trace_format": derived.meta["trace_format"],
            "trace_block_size": derived.meta["trace_block_size"],
            "derived_meta": self.rel(derived.meta_path),
            "transform": derived.meta["spec"],
            "transform_tag": plan.tag,
            "load": {"mode": mode, "value": None, "level": plan.level},
            "num_workers": plan.n,
            "sla": {"ttft_ms": None, "itl_ms": None, "e2e_slowdown": None},
            "measure": measure_rule(plan.mode),
            "measure_trace": measure_trace,
            "holdout_axis": plan.axis,
            "holdout_axes": [a for a in (plan.axis, *plan.extra_axes) if a],
            "selection_exposed": split == "test" and plan.n == 6,
            "engine_ref": "CR/config/engine.json",
            "expected_cost_s": self.expected_cost(
                cell_id, derived.sha256, family, rows
            ),
            "trace_rows": rows,
        }
        if plan.mode == "closed":
            # Load-independent, so fixed here; open-loop windows depend on the calibrated speedup.
            cell["measure"]["warmup_trace_ms"] = measure_trace["warmup_ms"]
        table = derived.meta["realized"].get("distinct_block_tokens_per_trace_window_s")
        if table:
            cell["cache_pressure_ref"] = {
                "distinct_block_tokens_per_trace_window_s": table,
                "kv_tokens_per_worker": KV_TOKENS_PER_WORKER,
                "formula": "pressure = U(100 * speedup) / (num_workers * kv_tokens_per_worker), U interpolated in the table",
            }
        return cell


def overlapping_slices() -> list[tuple[str, str]]:
    """Pairs of windowed segments of one family whose trace slices share time (must be none)."""
    windowed = sorted(
        (d["family"], d["slice"], name)
        for name, d in segments().items()
        if d["family"] in WINDOWED_FAMILIES
    )
    return [
        (a, b)
        for i, (fam_a, (lo_a, hi_a), a) in enumerate(windowed)
        for fam_b, (lo_b, hi_b), b in windowed[i + 1 :]
        if fam_a == fam_b and lo_b < hi_a and lo_a < hi_b
    ]


def validate(cells: dict[str, list[dict]], plays: dict[str, list[str]]) -> dict:
    """Check the split invariants; returns a summary. Raises on any violation."""
    overlaps = overlapping_slices()
    if overlaps:
        raise ValueError(f"windowed segments share trace rows: {overlaps}")
    ids = [c["cell_id"] for split in cells.values() for c in split]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate cell ids {dupes}")
    segment_splits = defaultdict(set)
    for split, members in cells.items():
        for cell in members:
            if cell["num_workers"] not in WORKERS[split]:
                raise ValueError(
                    f"{cell['cell_id']}: N={cell['num_workers']} not allowed in {split}"
                )
            if (
                split == "test"
                and cell["num_workers"] in (4, 8)
                and cell["holdout_axis"] in (None, "worker_count")
            ):
                raise ValueError(
                    f"{cell['cell_id']}: a test cell at N in (4, 8) needs a non-worker holdout axis"
                )
            if (
                split == "test"
                and cell["num_workers"] not in (4, 8)
                and cell["holdout_axis"] != "worker_count"
            ):
                raise ValueError(
                    f"{cell['cell_id']}: test cell at a held-out N must say worker_count"
                )
            for part in (
                cell["segment"].split(":", 1)[1].split("+")
                if cell["family"] == "agentx"
                else [cell["segment"]]
            ):
                segment_splits[
                    part if cell["family"] != "agentx" else f"agentx:{part}"
                ].add(split)
            if cell["holdout_axis"] == "transform_extrapolation":
                spec = cell["transform"]
                outside = [
                    k for k, (lo, hi) in TRAIN_RANGES.items() if not lo <= spec[k] <= hi
                ]
                if not outside:
                    raise ValueError(
                        f"{cell['cell_id']}: no transform value outside the training range"
                    )
            if split in ("train", "val"):
                spec = cell["transform"]
                inside = all(
                    lo <= spec[k] <= hi for k, (lo, hi) in TRAIN_RANGES.items()
                )
                if not inside:
                    raise ValueError(
                        f"{cell['cell_id']}: {split} transform outside the training range"
                    )
    leaks = {s: sorted(v) for s, v in segment_splits.items() if len(v) > 1}
    if leaks:
        raise ValueError(f"segments used by several splits: {leaks}")
    all_plays = [p for members in plays.values() for p in members]
    if len(all_plays) != len(set(all_plays)):
        raise ValueError("an AgentX play is in two segments")
    return {
        "segments_by_split": {
            split: sorted({s for s, v in segment_splits.items() if split in v})
            for split in cells
        }
    }


def summarize_validation(path: Path, cells: dict[str, list[dict]]) -> dict:
    """Fold a per-cell replay check (cell id -> {trace_sha256, error, wall_s, completed, ...}) into the manifest."""
    results = json.loads(Path(path).read_text())
    rows, missing, stale = [], [], []
    for split, members in cells.items():
        for cell in members:
            got = results.get(cell["cell_id"])
            if got is None:
                missing.append(cell["cell_id"])
            elif got.get("trace_sha256") != cell["trace_sha256"][0]:
                stale.append(cell["cell_id"])
            else:
                rows.append((cell, got))
    by_family: dict[str, list[float]] = defaultdict(list)
    for cell, got in rows:
        by_family[cell["family"]].append(got["wall_s"])
    return {
        "source": str(path),
        "cells_checked": len(rows),
        "missing": missing,
        "stale": stale,
        "errors": sorted(c["cell_id"] for c, g in rows if g.get("error")),
        "incomplete": sorted(
            c["cell_id"] for c, g in rows if g.get("completed") != c["trace_rows"]
        ),
        "wall_s_by_family": {
            fam: {
                "n": len(v),
                "min": min(v),
                "mean": round(sum(v) / len(v), 3),
                "max": max(v),
            }
            for fam, v in sorted(by_family.items())
        },
        "over_expected_cost": sorted(
            c["cell_id"] for c, g in rows if g["wall_s"] > c["expected_cost_s"]
        ),
    }


def build(cr: Path, validation_json: Path | None = None) -> dict:
    manifest = json.loads((cr / "traces" / "MANIFEST.json").read_text())
    play_entries = {
        Path(f["path"]).name: f for f in manifest["files"] if f["family"] == "agentx"
    }
    plays = allocate_plays(
        {
            name: e["stats"]["explicit_subagent_groups"] > 0
            for name, e in play_entries.items()
        }
    )
    measured = (
        json.loads(Path(validation_json).read_text())
        if validation_json is not None
        else {}
    )
    builder = Builder(cr=cr, plays=plays, measured=measured)
    cells = {
        split: [builder.cell(split, plan) for plan in plans]
        for split, plans in PLANS.items()
    }
    summary = validate(cells, plays)

    out_dir = cr / "cells"
    files = {}
    for split, members in cells.items():
        path = out_dir / f"{split}.candidates.jsonl"
        write_atomic(
            path,
            "".join(json.dumps(c, sort_keys=True) + "\n" for c in members).encode(),
        )
        files[split] = {
            "path": str(path.relative_to(cr)),
            "sha256": sha256_file(path),
            "cells": len(members),
        }

    def tally(split: str, key) -> dict:
        return dict(sorted(Counter(key(c) for c in cells[split]).items()))

    split_manifest = {
        "schema": "learned-routing.split-manifest.v1",
        "cells_version": CELLS_VERSION,
        "status": "candidates: load.value, sla and measure windows are null until calibration; test is frozen at calibration",
        "traces_manifest_sha256": sha256_file(cr / "traces" / "MANIFEST.json"),
        "files": files,
        "families": {
            "mooncake": "train/val/test by disjoint, self-contained time windows (no row in two slices)",
            "synthetic_sessions": "train/val/test by generator seed",
            "agentx": "train/val/test by disjoint play subsets; agentic_lanes only",
            "fast25_conversation": "test only (held-out family); segments shared with the aligned Mooncake windows",
            "fast25_synthetic": "test only (held-out family); two disjoint windows",
            "toolagent": "excluded: relabeled copy of mooncake_trace (audit setup/determinism-cost-context r1 F1)",
        },
        "segments": {
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
            for name, d in segments().items()
        },
        "agentx_play_subsets": {
            name: {
                "split": split,
                "plays": plays[name],
                "requests": sum(play_entries[p]["rows"] for p in plays[name]),
                "subagent_plays": sum(
                    play_entries[p]["stats"]["explicit_subagent_groups"] > 0
                    for p in plays[name]
                ),
            }
            for name, split, _ in AGENTX_SEGMENTS
        },
        "agentx_play_allocation": f"seeded ({AGENTX_SPLIT_SEED}) round-robin deal, subagent plays first",
        "agentx_think_cap_s": AGENTX_THINK_CAP_S,
        "segments_by_split": summary["segments_by_split"],
        "holdout_axes": {
            "time_window": "held-out Mooncake windows w3, w5 at N in {4, 8}; every slice is disjoint from every other",
            "transform_extrapolation": "transform values outside TRAIN_RANGES, on held-out segments",
            "agentx_plays": "AgentX play subsets T1-T4, disjoint from train and val plays",
            "workload_draw": "synthetic-session generator seeds 6-9, disjoint from train and val seeds",
            "family": "FAST25 conversation (correlated with Mooncake w3/w5) and FAST25 synthetic (two disjoint windows), never trained on",
            "worker_count": "N in {2, 6, 16, 32}, on held-out segments; N = 6 also appears in validation (selection_exposed)",
        },
        "train_ranges": {k: list(v) for k, v in TRAIN_RANGES.items()},
        "transform_tags": TRANSFORMS,
        "worker_counts": {k: list(v) for k, v in WORKERS.items()},
        "load_modes": {
            "open_speedup": "Mooncake-format traces: arrival_speedup_ratio (for session traces it divides inter-turn delays too)",
            "closed_concurrency": "replay_concurrency",
            "agentic_lanes": "AgentX only; trace_timestamps mode is a synchronized start of every play (audit r1 F3) and is not used",
        },
        "counts": {
            split: {
                "cells": len(cells[split]),
                "by_family": tally(split, lambda c: c["family"]),
                "by_num_workers": tally(split, lambda c: c["num_workers"]),
                "by_load_mode": tally(split, lambda c: c["load"]["mode"]),
                "by_level": tally(split, lambda c: c["load"]["level"]),
                "by_holdout_axis": tally(split, lambda c: c["holdout_axis"] or "none"),
                "expected_cost_s_total": round(
                    sum(c["expected_cost_s"] for c in cells[split]), 1
                ),
                "expected_cost_s_max": max(c["expected_cost_s"] for c in cells[split]),
            }
            for split in cells
        },
        "cost_model": {
            "ms_per_request_in_process": COST_MS_PER_REQUEST,
            "formula": "expected_cost_s = max(0.3 + trace_rows * ms_per_request / 1000, 1.5 * measured wall of the per-cell "
            "replay check at a provisional load); one replay, in-process, before load calibration",
            "evidence": [
                "runs/build_workloads/smoke/cost.json",
                "runs/build_workloads/smoke/cells_validation.json",
            ],
        },
        "cell_fields_added": {
            "segment": "unit of independence (LR-02, LR-11)",
            "trace_sha256": "SHA-256 of the derived trace file (checked by the harness)",
            "derived_meta": "the derived trace's meta JSON (spec, source SHA, realized stats)",
            "transform_tag": "short name of the transform",
            "measure": "harness measurement rule: basis; on completion-basis cells (closed loop, lanes) end full_occupancy "
            "(window ends when fewer than load.value sessions or lanes stay occupied); warmup_trace_ms (identity warm-up) on "
            "closed-loop cells. Calibration adds open-loop warmup_ms and window_ms and lanes warmup_ms, in replay time",
            "measure_trace": "warm-up and window in trace time (open loop) or play counts (lanes)",
            "holdout_axes": "every axis a test cell is held out on; holdout_axis is the primary one",
            "selection_exposed": "true for test cells at N = 6, a worker count validation also uses (LR-10)",
            "expected_cost_s": "replay cost estimate from the build smoke",
            "trace_rows": "requests in the derived trace",
            "cache_pressure_ref": "LR-12 numerator table: distinct block tokens per trace-time window",
        },
        "max_model_len": MAX_MODEL_LEN,
        "derived_traces": sorted(
            (
                {
                    "sha256": d.sha256,
                    "path": builder.rel(d.path),
                    "meta": builder.rel(d.meta_path),
                    "source": d.meta["source"]["path"].replace(str(cr) + "/", "CR/"),
                    "rows": d.meta["realized"].get(
                        "rows", d.meta["realized"].get("requests")
                    ),
                    "cells": sorted(
                        c["cell_id"]
                        for split in cells.values()
                        for c in split
                        if c["trace_sha256"][0] == d.sha256
                    ),
                }
                for d in builder.derived.values()
            ),
            key=lambda e: e["path"],
        ),
    }
    if validation_json is not None:
        split_manifest["replay_check"] = summarize_validation(validation_json, cells)
    write_json(out_dir / "SPLIT_MANIFEST.json", split_manifest)
    return split_manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write candidate cells and the split manifest."
    )
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=None,
        help="per-cell replay check to summarize into the manifest (cell id -> result)",
    )
    args = parser.parse_args(argv)
    manifest = build(campaign_root(args.campaign_root), args.validation_json)
    print(
        json.dumps(
            {"files": manifest["files"], "counts": manifest["counts"]},
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
