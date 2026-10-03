# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, seeded trace transforms that materialize ``CR/traces/derived/<sha>.jsonl``.

Usage::

    python -m learned_routing.workloads.transform --campaign-root CR \\
        --source CR/traces/mooncake/mooncake_trace.jsonl --format mooncake \\
        --spec '{"window": [180000, 540000], "isl_unique_mult": 1.5, "seed": 0}'

The output file is named by the SHA-256 of its bytes; ``<sha>.meta.json`` next to it records the
canonical spec, the sources, the realized statistics and the transform version. Equal inputs give
equal bytes.

Transforms run in this fixed order (a parameter at its identity value is a no-op):

1. ``window`` ``[t0_ms, t1_ms)`` (Mooncake format): keep sessions whose first turn arrives in the
   window (a single-turn row is a one-row session), and rebase timestamps by ``-t0``.
   ``plays`` (Weka format): keep the named plays, in the given order.
2. ``prefix_root_mult`` ``k`` (integer): ``k`` disjoint hash-space copies. A *root* is the set of
   requests sharing a first block, and its *root segment* is the longest prefix common to all of
   them (a shared system prompt). A request's *family key* is its first block after the root
   segment (or its last block if it has none), so requests that share anything beyond the root
   segment share a key. Each family goes to copy ``hash(seed, key) mod k`` and every block id ``h``
   in it becomes ``(copy, h)``. Families stay intact; only root segments are duplicated. A family
   key at depth 0 (the literal "hash of the first block") would be a no-op on traces with a few
   roots: Mooncake has 4 first blocks and FAST25 conversation has 1.
3. ``isl_prefix_mult`` / ``isl_unique_mult``: a block is *shared* if it occurs in two or more
   requests, otherwise *unique*. The shared blocks are exactly the union of every request's
   "blocks shared with an earlier request" prefix; this is the only per-hash classification that
   keeps hash identity consistent, because a block that is new in request A and reused by B must
   expand the same way in both. Each shared block ``h`` expands to ``floor(m) + [u(h) < frac(m)]``
   copies, with ``u(h)`` a seeded per-hash draw, so every request containing ``h`` gets the same
   copies. Under hash chaining the unique blocks of a request form its suffix; that suffix of length
   ``n`` becomes ``floor(n * m + u(row))`` fresh blocks. The partial tail of the last block is kept:
   ``input_length' = block_size * (blocks' - 1) + tail``. Requests keep at least one block.
4. ``osl_mult``: ``output_length' = max(1, round(output_length * m))`` (Weka keeps authored zeros).
5. ``think_cap_s`` then ``think_mult``. The cap bounds idle time: Mooncake session rows cap
   ``delay`` at ``think_cap_s``; a Weka play compresses every idle gap between its busy periods
   (the union of its requests' recorded ``[t, t + api_time]`` intervals) that is longer than the cap
   linearly down to the cap, a strictly monotone time warp that keeps every busy period, request
   duration and ordering. AgentX plays contain idle gaps of hours to days (user away), which would
   leave an agentic lane idle for most of a replay; the AgentX scenario likewise caps idle delays.
   ``think_mult``: Mooncake session rows scale ``delay`` (completion-relative inter-turn delay).
   Weka plays are dilated in time about their first request: every ``t``, ``api_time``,
   ``think_time``, ``ttft`` and subagent ``duration_ms`` is scaled, so every recorded dependency delay
   (sequence, spawn, join) scales by ``m`` and every ordering is preserved. A trace with no
   inter-turn delays rejects ``think_mult != 1`` instead of ignoring it.
6. Context cap (``max_model_len``): Mooncake rows with ``input_length >= max_model_len`` are dropped
   (replay rejects them for every policy); outputs are clamped so ``in + out <= max_model_len``
   (replay would truncate them silently). Weka plays clamp outputs only and fail if an input
   reaches the cap, because dropping a request would break the play graph. Counts are recorded.
7. Hash ids are renumbered densely in order of first appearance, so they stay within ``u32``.

``isl_*_mult`` and ``prefix_root_mult`` are Mooncake-format only: Weka plays use a local hash
scope, so plays never share prefixes and a root multiplier has nothing to split.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agentx import iter_requests
from .common import (
    MAX_MODEL_LEN,
    campaign_root,
    canonical_json,
    index_draw,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
    summarize,
    unit_draw,
    write_atomic,
    write_json,
)

TRANSFORM_VERSION = "lr-transform-v1"
FORMATS = ("mooncake", "weka")
PRESSURE_WINDOWS_S = (25, 50, 100, 200, 400, 800)


@dataclass(frozen=True)
class TransformSpec:
    window: tuple[float, float] | None = None
    plays: tuple[str, ...] | None = None
    isl_unique_mult: float = 1.0
    isl_prefix_mult: float = 1.0
    osl_mult: float = 1.0
    prefix_root_mult: int = 1
    think_mult: float = 1.0
    think_cap_s: float | None = None
    seed: int = 0
    max_model_len: int = MAX_MODEL_LEN

    @classmethod
    def from_dict(cls, data: dict) -> TransformSpec:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown transform keys {unknown}; known {sorted(known)}")
        values = dict(data)
        if values.get("window") is not None:
            t0, t1 = values["window"]
            values["window"] = (float(t0), float(t1))
        if values.get("plays") is not None:
            values["plays"] = tuple(str(p) for p in values["plays"])
        for key in ("isl_unique_mult", "isl_prefix_mult", "osl_mult", "think_mult"):
            if key in values:
                values[key] = float(values[key])
        if values.get("think_cap_s") is not None:
            values["think_cap_s"] = float(values["think_cap_s"])
        for key in ("prefix_root_mult", "seed", "max_model_len"):
            if key in values:
                if float(values[key]) != int(values[key]):
                    raise ValueError(f"{key} must be an integer, got {values[key]!r}")
                values[key] = int(values[key])
        spec = cls(**values)
        spec.validate()
        return spec

    def validate(self) -> None:
        for key in ("isl_unique_mult", "isl_prefix_mult", "osl_mult", "think_mult"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and > 0, got {value}")
        if self.think_cap_s is not None and not (
            math.isfinite(self.think_cap_s) and self.think_cap_s >= 0
        ):
            raise ValueError(
                f"think_cap_s must be finite and >= 0, got {self.think_cap_s}"
            )
        if self.prefix_root_mult < 1:
            raise ValueError(
                f"prefix_root_mult must be >= 1, got {self.prefix_root_mult}"
            )
        if self.window is not None and not self.window[0] < self.window[1]:
            raise ValueError(f"window must satisfy t0 < t1, got {self.window}")
        if self.plays is not None and not self.plays:
            raise ValueError("plays must be non-empty when given")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["window"] = list(self.window) if self.window is not None else None
        data["plays"] = list(self.plays) if self.plays is not None else None
        return data


@dataclass
class TransformStats:
    rows_in: int = 0
    rows_out: int = 0
    sessions_out: int = 0
    plays_out: int = 0
    dropped_isl_at_cap: int = 0
    clamped_osl_at_cap: int = 0
    prefix_root_families: int = 0
    prefix_root_roots: int = 0
    shared_blocks_in: int = 0
    unique_blocks_in: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Mooncake format


def _sessions(rows: Sequence[dict]) -> list[list[int]]:
    """Row indices grouped into sessions in first-appearance order (single-turn rows alone)."""
    order: list[list[int]] = []
    by_id: dict = {}
    for index, row in enumerate(rows):
        session_id = row.get("session_id")
        if session_id is None:
            order.append([index])
            continue
        if session_id not in by_id:
            by_id[session_id] = len(order)
            order.append([])
        order[by_id[session_id]].append(index)
    return order


def _first_arrival(rows: Sequence[dict], session: list[int]) -> float:
    first = rows[session[0]]
    if "delay" in first and first.get("timestamp") is None:
        raise ValueError(
            "a session's first row must carry a timestamp, not only a delay"
        )
    timestamp = first.get("timestamp", first.get("created_time"))
    return 0.0 if timestamp is None else float(timestamp)


def _apply_window(
    rows: list[dict], window: tuple[float, float], stats: TransformStats
) -> list[dict]:
    """Keep sessions arriving in ``[t0, t1)``; rebase so the first kept arrival is at 0.

    Replay rebases a trace to its first arrival anyway; doing it here makes derived-trace time
    explicit. ``stats.extra["window_rebase_ms"]`` is the offset subtracted (``>= t0``).
    """
    t0, t1 = window
    chosen = [s for s in _sessions(rows) if t0 <= _first_arrival(rows, s) < t1]
    if not chosen:
        raise ValueError(f"window {window} selects no rows")
    rebase = min(_first_arrival(rows, s) for s in chosen)
    stats.extra["window_rebase_ms"] = rebase
    kept: list[dict] = []
    for session in chosen:
        arrival = _first_arrival(rows, session)
        for position, index in enumerate(session):
            row = dict(rows[index])
            if position == 0 or "delay" not in row:
                stamp = float(row.get("timestamp", arrival)) - rebase
                row["timestamp"] = int(stamp) if stamp == int(stamp) else stamp
            kept.append(row)
    return kept


def _root_segments(rows: Sequence[dict]) -> dict:
    """First block -> length of the prefix common to every request with that first block."""
    common: dict = {}
    for row in rows:
        ids = row["hash_ids"]
        if not ids:
            continue
        root = ids[0]
        if root not in common:
            common[root] = list(ids)
            continue
        prefix = common[root]
        limit = min(len(prefix), len(ids))
        k = 0
        while k < limit and prefix[k] == ids[k]:
            k += 1
        del prefix[k:]
    return {root: len(prefix) for root, prefix in common.items()}


def _apply_prefix_root_mult(
    rows: list[dict], k: int, seed: int, stats: TransformStats
) -> list[dict]:
    segments = _root_segments(rows)
    stats.prefix_root_roots = len(segments)
    copy_by_session: dict = {}
    families = set()
    out = []
    for session in _sessions(rows):
        first_ids = rows[session[0]]["hash_ids"]
        if first_ids:
            depth = min(segments[first_ids[0]], len(first_ids) - 1)
            key = first_ids[depth]
        else:
            key = ("empty", session[0])
        families.add(key)
        copy = index_draw(seed, "prefix_root", key, k)
        for index in session:
            row = dict(rows[index])
            row["hash_ids"] = [(copy, h) for h in row["hash_ids"]]
            copy_by_session[session[0]] = copy
            out.append(row)
    stats.prefix_root_families = len(families)
    stats.extra["prefix_root_copy_sizes"] = dict(
        sorted(Counter(copy_by_session.values()).items())
    )
    return out


def _scaled_count(n: int, mult: float, draw: float) -> int:
    return int(math.floor(n * mult + draw))


def _apply_isl_mults(
    rows: list[dict],
    prefix_mult: float,
    unique_mult: float,
    seed: int,
    block_size: int,
    stats: TransformStats,
) -> list[dict]:
    occurrences = Counter()
    for row in rows:
        occurrences.update(set(row["hash_ids"]))
    shared = {h for h, count in occurrences.items() if count >= 2}
    stats.shared_blocks_in = len(shared)
    stats.unique_blocks_in = len(occurrences) - len(shared)
    if prefix_mult == 1.0 and unique_mult == 1.0:
        return rows
    whole, frac = divmod(prefix_mult, 1.0)
    copies: dict = {}
    out = []
    for row_index, row in enumerate(rows):
        ids = row["hash_ids"]
        split = 0
        while split < len(ids) and ids[split] in shared:
            split += 1
        if any(h in shared for h in ids[split:]):
            raise ValueError(
                f"row {row_index}: a shared block follows a unique one; hash ids are not prefix-chained"
            )
        new_ids = []
        for h in ids[:split]:
            count = copies.get(h)
            if count is None:
                count = int(whole) + (unit_draw(seed, "isl_prefix", repr(h)) < frac)
                copies[h] = count
            new_ids.extend(("p", h, j) for j in range(count))
        unique_count = _scaled_count(
            len(ids) - split, unique_mult, unit_draw(seed, "isl_unique", row_index)
        )
        new_ids.extend(("u", row_index, j) for j in range(unique_count))
        if not new_ids:
            new_ids = [("u", row_index, 0)]
        tail = (
            row["input_length"] - block_size * (len(ids) - 1)
            if ids
            else row["input_length"]
        )
        tail = min(max(tail, 1), block_size)
        new_row = dict(row)
        new_row["hash_ids"] = new_ids
        new_row["input_length"] = block_size * (len(new_ids) - 1) + tail
        out.append(new_row)
    return out


def _has_delays(rows: Sequence[dict]) -> bool:
    return any("delay" in row for row in rows)


def transform_mooncake(
    rows: list[dict], spec: TransformSpec, block_size: int
) -> tuple[list[dict], TransformStats]:
    stats = TransformStats(rows_in=len(rows))
    if spec.plays is not None:
        raise ValueError("plays applies to weka traces only")
    if any("wait_for" in row for row in rows):
        raise ValueError("rows with wait_for dependencies are not supported")
    if spec.window is not None:
        rows = _apply_window(rows, spec.window, stats)
    if spec.prefix_root_mult > 1:
        rows = _apply_prefix_root_mult(rows, spec.prefix_root_mult, spec.seed, stats)
    rows = _apply_isl_mults(
        rows, spec.isl_prefix_mult, spec.isl_unique_mult, spec.seed, block_size, stats
    )
    if (spec.think_mult != 1.0 or spec.think_cap_s is not None) and not _has_delays(
        rows
    ):
        raise ValueError(
            "think_mult/think_cap_s need multi-turn sessions with inter-turn delays; this trace has none"
        )

    sessions = _sessions(rows)
    out: list[dict] = []
    for session in sessions:
        session_rows = [dict(rows[i]) for i in session]
        if len(session_rows) > 1 and any(
            r["input_length"] >= spec.max_model_len for r in session_rows
        ):
            raise ValueError(
                f"session {session_rows[0].get('session_id')!r} reaches max_model_len; "
                "lower the ISL multipliers instead of dropping turns"
            )
        for row in session_rows:
            if spec.osl_mult != 1.0:
                row["output_length"] = max(
                    1, int(round(row["output_length"] * spec.osl_mult))
                )
            if (
                "delay" in row
                and spec.think_cap_s is not None
                and float(row["delay"]) > spec.think_cap_s * 1000.0
            ):
                row["delay"] = spec.think_cap_s * 1000.0
                stats.extra["capped_delays"] = stats.extra.get("capped_delays", 0) + 1
            if spec.think_mult != 1.0 and "delay" in row:
                row["delay"] = float(row["delay"]) * spec.think_mult
            if row["input_length"] >= spec.max_model_len:
                stats.dropped_isl_at_cap += 1
                continue
            if row["input_length"] + row["output_length"] > spec.max_model_len:
                row["output_length"] = max(1, spec.max_model_len - row["input_length"])
                stats.clamped_osl_at_cap += 1
            out.append(row)

    renumber: dict = {}
    for row in out:
        row["hash_ids"] = [
            renumber.setdefault(h, len(renumber)) for h in row["hash_ids"]
        ]
    stats.rows_out = len(out)
    stats.sessions_out = len(_sessions(out))
    if not out:
        raise ValueError("transform produced an empty trace")
    return out, stats


# ---------------------------------------------------------------------------------------------
# Weka format


def nested_timestamps_relative(play: dict) -> bool:
    """True if some subagent request starts before its marker (the importer's relative witness)."""
    for entry in play["requests"]:
        if entry["type"] != "subagent":
            continue
        if any(float(r["t"]) < float(entry["t"]) - 1e-6 for r in entry["requests"]):
            return True
    return False


def _warp_play_times(play: dict, warp) -> dict:
    """Apply a monotone time map to every timestamp of a play (absolute nested timestamps)."""

    def warp_request(request: dict) -> dict:
        request = dict(request)
        request["t"] = warp(float(request["t"]))
        return request

    entries = []
    for entry in play["requests"]:
        if entry["type"] == "subagent":
            entry = dict(entry)
            start = float(entry["t"])
            entry["t"] = warp(start)
            if entry.get("duration_ms") is not None:
                end = warp(start + entry["duration_ms"] / 1000.0)
                entry["duration_ms"] = int(round((end - entry["t"]) * 1000.0))
            entry["requests"] = [warp_request(r) for r in entry["requests"]]
            entries.append(entry)
        else:
            entries.append(warp_request(entry))
    play = dict(play)
    play["requests"] = entries
    return play


def busy_periods(play: dict) -> list[tuple[float, float]]:
    """Union of the play's recorded request intervals ``[t, t + api_time]``, in time order."""
    intervals = sorted(
        (float(r["t"]), float(r["t"]) + float(r.get("api_time") or 0.0))
        for r in iter_requests(play["requests"])
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _cap_idle(play: dict, cap_s: float, stats: TransformStats) -> dict:
    if nested_timestamps_relative(play):
        raise ValueError(
            f"play {play.get('id')}: relative nested timestamps; idle capping assumes absolute"
        )
    busy = busy_periods(play)
    # Knots of a piecewise-linear, strictly increasing map: slope 1 on busy periods, and each idle
    # gap longer than the cap is compressed linearly to exactly the cap.
    xs, ys = [busy[0][0]], [busy[0][0]]
    shift = 0.0
    for (_, end), (start, _) in zip(busy, busy[1:]):
        xs.append(end)
        ys.append(end - shift)
        gap = start - end
        if gap > cap_s:
            shift += gap - cap_s
            stats.extra["capped_idle_gaps"] = stats.extra.get("capped_idle_gaps", 0) + 1
        xs.append(start)
        ys.append(start - shift)
    xs.append(busy[-1][1])
    ys.append(busy[-1][1] - shift)
    stats.extra["idle_seconds_removed"] = (
        stats.extra.get("idle_seconds_removed", 0.0) + shift
    )
    if shift == 0.0:
        return play

    def warp(t: float) -> float:
        if t <= xs[0]:
            return t
        if t >= xs[-1]:
            return t - shift
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= t:
                lo = mid
            else:
                hi = mid
        if xs[hi] == xs[lo]:
            return ys[lo]
        return ys[lo] + (t - xs[lo]) * (ys[hi] - ys[lo]) / (xs[hi] - xs[lo])

    return _warp_play_times(play, warp)


def _dilate_play(play: dict, mult: float) -> dict:
    if nested_timestamps_relative(play):
        raise ValueError(
            f"play {play.get('id')}: relative nested timestamps; time dilation assumes absolute"
        )
    requests = list(iter_requests(play["requests"]))
    origin = min(float(r["t"]) for r in requests)

    def scale_time(value: float) -> float:
        return origin + (float(value) - origin) * mult

    def scale_request(request: dict) -> dict:
        request = dict(request)
        request["t"] = scale_time(request["t"])
        for key in ("api_time", "think_time", "ttft"):
            if request.get(key) is not None:
                request[key] = float(request[key]) * mult
        return request

    entries = []
    for entry in play["requests"]:
        if entry["type"] == "subagent":
            entry = dict(entry)
            entry["t"] = scale_time(entry["t"])
            if entry.get("duration_ms") is not None:
                entry["duration_ms"] = int(round(entry["duration_ms"] * mult))
            entry["requests"] = [scale_request(r) for r in entry["requests"]]
            entries.append(entry)
        else:
            entries.append(scale_request(entry))
    play = dict(play)
    play["requests"] = entries
    return play


def _map_requests(play: dict, fn) -> dict:
    entries = []
    for entry in play["requests"]:
        if entry["type"] == "subagent":
            entry = dict(entry)
            entry["requests"] = [fn(dict(r)) for r in entry["requests"]]
            entries.append(entry)
        else:
            entries.append(fn(dict(entry)))
    play = dict(play)
    play["requests"] = entries
    return play


def transform_weka(
    plays: list[tuple[str, dict]], spec: TransformSpec
) -> tuple[list[dict], TransformStats]:
    for key in ("isl_unique_mult", "isl_prefix_mult"):
        if getattr(spec, key) != 1.0:
            raise ValueError(
                f"{key} is Mooncake-format only (Weka hash ids are play-local)"
            )
    if spec.prefix_root_mult != 1:
        raise ValueError(
            "prefix_root_mult is Mooncake-format only (Weka plays never share prefixes)"
        )
    if spec.window is not None:
        raise ValueError(
            "window is Mooncake-format only; subset Weka traces with plays"
        )
    stats = TransformStats(
        rows_in=sum(len(list(iter_requests(p["requests"]))) for _, p in plays)
    )
    out = []
    for name, play in plays:
        if spec.think_cap_s is not None:
            play = _cap_idle(play, spec.think_cap_s, stats)
        if spec.think_mult != 1.0:
            play = _dilate_play(play, spec.think_mult)

        def adjust(request: dict) -> dict:
            if request["in"] >= spec.max_model_len:
                raise ValueError(
                    f"play {name}: input {request['in']} reaches max_model_len"
                )
            if spec.osl_mult != 1.0 and request["out"] > 0:
                request["out"] = max(1, int(round(request["out"] * spec.osl_mult)))
            if request["in"] + max(request["out"], 1) > spec.max_model_len:
                request["out"] = spec.max_model_len - request["in"]
                stats.clamped_osl_at_cap += 1
            return request

        out.append(_map_requests(play, adjust))
    stats.plays_out = len(out)
    stats.rows_out = sum(len(list(iter_requests(p["requests"]))) for p in out)
    return out, stats


# ---------------------------------------------------------------------------------------------
# Statistics and materialization


def mooncake_stats(rows: Sequence[dict], block_size: int) -> dict:
    seen = set()
    reuse = []
    for row in rows:
        ids = row["hash_ids"]
        k = 0
        while k < len(ids) and ids[k] in seen:
            k += 1
        reuse.append(k / max(len(ids), 1))
        seen.update(ids)
    sessions = _sessions(rows)
    arrivals = [_first_arrival(rows, s) for s in sessions]
    tie_units = Counter(arrivals)
    span_ms = max(arrivals) - min(arrivals) if arrivals else 0.0
    stats = {
        "rows": len(rows),
        "sessions": len(sessions),
        "multi_turn_sessions": sum(len(s) > 1 for s in sessions),
        "turns_per_session": summarize([len(s) for s in sessions]),
        "isl": summarize([r["input_length"] for r in rows]),
        "osl": summarize([r["output_length"] for r in rows]),
        "max_isl_plus_osl": max(r["input_length"] + r["output_length"] for r in rows),
        "delay_ms": summarize([float(r["delay"]) for r in rows if "delay" in r]),
        "first_arrival_span_ms": span_ms,
        "session_arrival_rate_per_s": (len(sessions) - 1) / (span_ms / 1000.0)
        if span_ms > 0
        else None,
        "request_rate_per_s_if_single_turn": (len(rows) - 1) / (span_ms / 1000.0)
        if span_ms > 0
        else None,
        "distinct_first_arrivals": len(tie_units),
        "crn_units_in_ties": sum(c for c in tie_units.values() if c > 1),
        "distinct_blocks": len(seen),
        "distinct_roots": len({r["hash_ids"][0] for r in rows if r["hash_ids"]}),
        "mean_prefix_reuse_frac": sum(reuse) / len(reuse),
        "input_tokens": sum(r["input_length"] for r in rows),
    }
    # LR-12 cache pressure numerator: distinct block tokens touched per trace-time window. The
    # pressure of a cell is U(window_s * speedup) / (N * per-worker KV tokens), so calibration can
    # evaluate it at any speedup. Later session turns are placed at first arrival plus cumulative
    # delays (service time ignored), so session traces get an approximate table.
    if span_ms > 0:
        times = [0.0] * len(rows)
        for session in sessions:
            clock = _first_arrival(rows, session)
            for position, index in enumerate(session):
                row = rows[index]
                if position and "delay" in row:
                    clock += float(row["delay"])
                elif position:
                    clock = float(row.get("timestamp", clock))
                times[index] = clock
        stats["pressure_times_approximate"] = len(sessions) != len(rows)
        table = {}
        for window_s in PRESSURE_WINDOWS_S:
            width = window_s * 1000.0
            buckets: dict[int, set] = defaultdict(set)
            for t, row in zip(times, rows):
                buckets[int(t // width)].update(row["hash_ids"])
            full = [
                len(b) for i, b in buckets.items() if (i + 1) * width <= span_ms + 1e-9
            ]
            if full:
                table[str(window_s)] = sum(full) / len(full) * block_size
        stats["distinct_block_tokens_per_trace_window_s"] = table
    return stats


def weka_stats(plays: Sequence[dict]) -> dict:
    requests = [r for p in plays for r in iter_requests(p["requests"])]
    contexts = [r["in"] + max(r["out"], 1) for r in requests]
    return {
        "plays": len(plays),
        "requests": len(requests),
        "explicit_subagent_groups": sum(
            e["type"] == "subagent" for p in plays for e in p["requests"]
        ),
        "isl": summarize([r["in"] for r in requests]),
        "osl": summarize([r["out"] for r in requests]),
        "max_context": max(contexts),
        "input_tokens": sum(r["in"] for r in requests),
        "distinct_block_tokens_per_play": summarize(
            [
                len(
                    {
                        h
                        for r in iter_requests(p["requests"])
                        for h in r.get("hash_ids") or []
                    }
                )
                * int(p["block_size"])
                for p in plays
            ]
        ),
        "recorded_span_s_per_play": summarize(
            [
                max(
                    float(r["t"]) + float(r.get("api_time") or 0.0)
                    for r in iter_requests(p["requests"])
                )
                - min(float(r["t"]) for r in iter_requests(p["requests"]))
                for p in plays
            ]
        ),
    }


def _load_weka_plays(
    source: Path, names: Sequence[str] | None
) -> list[tuple[str, dict]]:
    if source.is_file():
        plays = [
            (f"{source.name}#{i:06d}", json.loads(line))
            for i, line in enumerate(source.read_text().splitlines())
            if line.strip()
        ]
        if names is not None:
            by_name = dict(plays)
            missing = [n for n in names if n not in by_name]
            if missing:
                raise ValueError(f"plays not found in {source}: {missing[:5]}")
            plays = [(n, by_name[n]) for n in names]
        return plays
    available = sorted(p.name for p in source.iterdir() if p.suffix == ".json")
    chosen = list(names) if names is not None else available
    missing = [n for n in chosen if n not in available]
    if missing:
        raise ValueError(f"plays not found in {source}: {missing[:5]}")
    return [(n, json.loads((source / n).read_text())) for n in chosen]


@dataclass(frozen=True)
class Derived:
    path: Path
    meta_path: Path
    sha256: str
    spec_key: str
    meta: dict


def source_key(source: Path) -> dict:
    source = Path(source)
    if source.is_file():
        return {"path": str(source), "sha256": sha256_file(source)}
    files = sorted(p for p in source.iterdir() if p.suffix == ".json")
    manifest = "".join(f"{p.name}\0{sha256_file(p)}\n" for p in files)
    return {
        "path": str(source),
        "dir_manifest_sha256": sha256_text(manifest),
        "files": len(files),
    }


def materialize(
    source: Path,
    trace_format: str,
    spec: TransformSpec,
    block_size: int,
    out_dir: Path,
    source_info: dict | None = None,
) -> Derived:
    """Apply ``spec`` to ``source`` and write ``out_dir/<sha>.jsonl`` plus ``<sha>.meta.json``."""
    source = Path(source)
    if trace_format not in FORMATS:
        raise ValueError(
            f"unsupported trace_format {trace_format!r}; expected one of {FORMATS}"
        )
    source_info = source_info or source_key(source)
    if trace_format == "mooncake":
        rows, stats = transform_mooncake(read_jsonl(source), spec, block_size)
        content = "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        ).encode()
        realized = mooncake_stats(rows, block_size)
    else:
        plays, stats = transform_weka(_load_weka_plays(source, spec.plays), spec)
        content = "".join(
            json.dumps(play, separators=(",", ":")) + "\n" for play in plays
        ).encode()
        realized = weka_stats(plays)
    sha = sha256_bytes(content)
    key_obj = {
        "version": TRANSFORM_VERSION,
        "format": trace_format,
        "block_size": block_size,
        "source": {k: v for k, v in source_info.items() if k != "path"},
        "spec": spec.to_dict(),
    }
    spec_key = sha256_text(canonical_json(key_obj))
    meta = {
        "schema": "learned-routing.derived-trace.v1",
        "transform_version": TRANSFORM_VERSION,
        "sha256": sha,
        "spec_key": spec_key,
        "trace_format": trace_format,
        "trace_block_size": block_size,
        "source": source_info,
        "spec": spec.to_dict(),
        "transform_stats": asdict(stats),
        "realized": realized,
    }
    out_dir = Path(out_dir)
    path = out_dir / f"{sha}.jsonl"
    meta_path = out_dir / f"{sha}.meta.json"
    if not path.exists() or sha256_file(path) != sha:
        write_atomic(path, content)
    write_json(meta_path, meta)
    return Derived(
        path=path, meta_path=meta_path, sha256=sha, spec_key=spec_key, meta=meta
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a derived trace under CR/traces/derived."
    )
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Mooncake JSONL, Weka play dir or play JSONL",
    )
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument(
        "--block-size", type=int, default=None, help="default: 512 mooncake, 64 weka"
    )
    parser.add_argument(
        "--spec", default="{}", help="JSON object of TransformSpec fields, or @file"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: CR/traces/derived"
    )
    args = parser.parse_args(argv)
    text = Path(args.spec[1:]).read_text() if args.spec.startswith("@") else args.spec
    spec = TransformSpec.from_dict(json.loads(text))
    block_size = args.block_size or (512 if args.format == "mooncake" else 64)
    out_dir = args.out_dir or campaign_root(args.campaign_root) / "traces" / "derived"
    derived = materialize(args.source, args.format, spec, block_size, out_dir)
    print(
        json.dumps(
            {
                "path": str(derived.path),
                "sha256": derived.sha256,
                "spec_key": derived.spec_key,
                "transform_stats": derived.meta["transform_stats"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
