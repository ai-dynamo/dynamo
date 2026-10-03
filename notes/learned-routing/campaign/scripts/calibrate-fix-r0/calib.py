"""Calibration helpers (stage 2): sweep cells, evaluation through the harness, offline rescoring.

Fixer r0 copy (audits/calibration/fix-r0.md, finding F1): identical to runs/calibrate/scripts/calib.py
except that open-loop session cells use a lifetime-derived warm-up (``session_warmup_s``) instead of
the fixed 180 s, and new artifacts go under runs/calibrate-fix-r0 (derived-trace index and scored
items are read through from runs/calibrate).

Run with WT/.venv/bin/python, PYTHONDONTWRITEBYTECODE=1, LR_ROOT=CR. Every replay goes through
learned_routing.evaluate.Evaluator (slot pool + result cache); rescoring reads the cached
per_request rows and never replays.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from learned_routing import goodput
from learned_routing.cells import Cell, resolve_replicate
from learned_routing.e0 import E0Table
from learned_routing.evaluate import Evaluator, Task
from learned_routing.paths import Layout
from learned_routing.policy import load_specs
from learned_routing.workloads import agentx_lowered as ax

CR = Path("<campaign-root>")
LAYOUT = Layout.resolve(CR)
CAL = CR / "runs" / "calibrate"
FIX = CR / "runs" / "calibrate-fix-r0"
SPLITS = ("train", "val", "test")
COPIES_PER_LANE = 8
NULL_SLA = {"ttft_ms": None, "itl_ms": None, "e2e_slowdown": None}


def candidates(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (CR / "cells" / f"{split}.candidates.jsonl").read_text().splitlines()
        if line.strip()
    ]


def resolve(path: str) -> Path:
    return LAYOUT.resolve_path(path)


@lru_cache(maxsize=None)
def _trace_rows(path: str) -> tuple:
    return tuple(json.loads(x) for x in resolve(path).read_text().splitlines() if x.strip())


def token_rate(cell: dict) -> dict:
    """Offered input tokens per trace-second over the cell's measured window (speedup 1).

    Mooncake-format flat rows: rows whose timestamp lies in [t0 + warmup, t0 + warmup + window).
    Session traces: every turn of each session whose first turn arrives in that interval.
    """
    rows = _trace_rows(cell["trace_files"][0])
    mt = cell["measure_trace"]
    stamps = [r["timestamp"] for r in rows if "timestamp" in r]
    t0 = min(stamps)
    lo, hi = t0 + mt["warmup_ms"], t0 + mt["warmup_ms"] + mt["window_ms"]
    tokens, requests, sessions = 0, 0, 0
    current_in = False
    for r in rows:
        if "timestamp" in r:
            current_in = lo <= r["timestamp"] < hi
            sessions += current_in
        if current_in:
            tokens += r["input_length"]
            requests += 1
    window_s = mt["window_ms"] / 1000.0
    return {
        "tokens_per_s": tokens / window_s,
        "requests_per_s": requests / window_s,
        "sessions_per_s": sessions / window_s,
        "window_s": window_s,
    }


def base_template(segment: str, split: str = "train", transform_tag: str = "base",
                  mode: str | None = None) -> dict:
    for cell in candidates(split):
        if cell["segment"] != segment or cell["transform_tag"] != transform_tag:
            continue
        if mode is None or cell["load"]["mode"] == mode:
            return cell
    raise KeyError((segment, split, transform_tag, mode))


# Arrival logging quantum per family (median gap between consecutive distinct first-arrival
# timestamps: Mooncake raw trace 3,053 ms, FAST25 conversation 3,000 ms, generated sessions 1,000 ms
# grid; FAST25 synthetic and AgentX are not slot-quantized). Replicates spread each session's first
# arrival uniformly inside its slot (protocol crn-spread-v1).
ARRIVAL_SPREAD_MS = {
    "mooncake": 3053.0,
    "fast25_conversation": 3000.0,
    "synthetic_sessions": 1000.0,
}


def _common(template: dict, cell_id: str, num_workers: int, split: str) -> dict:
    raw = copy.deepcopy(template)
    raw["cell_id"] = cell_id
    raw["num_workers"] = int(num_workers)
    raw["split"] = split
    raw["holdout_axes"] = template.get("holdout_axes", [])
    raw["sla"] = dict(NULL_SLA)
    raw.pop("notes", None)
    spread = ARRIVAL_SPREAD_MS.get(template["family"])
    if spread:
        raw["arrival_spread_ms"] = spread
    else:
        raw.pop("arrival_spread_ms", None)
    return raw


def open_cell(template: dict, cell_id: str, num_workers: int, u_tok_per_worker: float,
              split: str = "calib", level: str | None = None, rate: float | None = None) -> dict:
    """Open loop at offered input-token rate u per worker: speedup = u * N / TR(window).

    ``rate`` overrides TR (a transformed cell uses its untransformed base window's rate)."""
    raw = _common(template, cell_id, num_workers, split)
    rate = rate if rate is not None else token_rate(template)["tokens_per_s"]
    speedup = float(f"{u_tok_per_worker * num_workers / rate:.6g}")
    mt = template["measure_trace"]
    raw["load"] = {
        "mode": "open_speedup",
        "value": speedup,
        "level": level,
        "per_worker": u_tok_per_worker,
        "per_worker_unit": "input_tokens_per_s_per_worker",
    }
    raw["measure"] = {
        "basis": "arrival",
        "warmup_ms": float(f"{mt['warmup_ms'] / speedup:.9g}"),
        "window_ms": float(f"{mt['window_ms'] / speedup:.9g}"),
    }
    return raw


def closed_cell(template: dict, cell_id: str, num_workers: int, c_per_worker: float,
                split: str = "calib", level: str | None = None) -> dict:
    raw = _common(template, cell_id, num_workers, split)
    total = max(1, int(round(c_per_worker * num_workers)))
    raw["load"] = {
        "mode": "closed_concurrency",
        "value": total,
        "level": level,
        "per_worker": c_per_worker,
        "per_worker_unit": "sessions_per_worker",
    }
    raw["measure"] = {
        "basis": "completion",
        "end": "full_occupancy",
        "warmup_trace_ms": template["measure_trace"]["warmup_ms"],
    }
    return raw


def agentx_trace(plays: list[str], split: str, transform: dict, num_copies: int,
                 seed: int = 0) -> dict:
    t = ax.PlayTransform.from_cell(transform)
    spec = ax.GenSpec(
        split=split,
        mode="closed",
        num_copies=num_copies,
        seed=seed,
        think_cap_s=t.think_cap_s,
        think_mult=t.think_mult,
        osl_mult=t.osl_mult,
        plays=tuple(sorted(plays)),
    )
    index = _load_index()
    key = _index_key(Path("agentx_lowered"), {"spec": ax.asdict(spec)}, 64)
    entry = index.get(key)
    if entry is None or not Path(entry["path"]).exists():
        header, rows, meta = ax.generate(spec, CR)
        data = ax.agentic_bytes(header, rows)
        sha = ax.sha256_bytes(data)
        out = ax.lowered_dir(CR) / "gen" / f"{sha}.jsonl"
        if not out.exists():
            ax.write_atomic(out, data)
            meta["path"] = str(out)
            meta["sha256"] = sha
            ax.write_json(out.with_name(out.name + ".meta.json"), meta)
        entry = {"path": str(out), "sha256": sha, "rows": len(rows)}
        index[key] = entry
        save_index()
    return {"path": Path(entry["path"]), "sha256": entry["sha256"], "rows": entry["rows"],
            "spec": ax.asdict(spec)}


def copy_span_list(meta_path: Path) -> list[float]:
    meta = json.loads(Path(meta_path).read_text())
    manifest = json.loads(Path(meta["base_manifest"]).read_text())["plays"]
    return [manifest[c["play"]]["span_ms"] for c in sorted(meta["copies"], key=lambda c: c["ordinal"])]


def min_lane_totals(gen: dict, lanes: int, ks) -> list[float]:
    """Per replicate k: min over lanes of the recorded spans of the copies the lane runs.

    Trace-intrinsic (no replay): replicate k permutes the copies (crn-order-v1 for
    agentic_mooncake) and replay deals copy position j to lane j % lanes.
    """
    from learned_routing import replicates

    spans = copy_span_list(Path(str(gen["path"]) + ".meta.json"))
    out = []
    for k in ks:
        seed = replicates.replicate_seed(gen["sha256"], k)
        order = replicates._shuffle(list(range(len(spans))), seed, "copies")
        totals = [0.0] * lanes
        for j, src in enumerate(order):
            totals[j % lanes] += spans[src]
        out.append(min(totals))
    return out


AX_WARMUP_Q90_MULT = 1.0
AX_COPIES_PER_LANE = (8, 10, 12, 16, 24)
AX_REPLICATES_CHECKED = 16


def lanes_cell(template: dict, cell_id: str, num_workers: int, l_per_worker: float,
               warmup_ms: float | None, split: str = "calib", level: str | None = None,
               trace_split: str | None = None, copies_per_lane: int | None = None) -> dict:
    """Lanes cell; with warmup_ms None, the frozen rule picks the warm-up and copies per lane.

    Rule (calibration): warm-up = AX_WARMUP_Q90_MULT x p90 recorded span over the cell's copies;
    copies per lane = the smallest of AX_COPIES_PER_LANE for which, in every replicate
    k < AX_REPLICATES_CHECKED, the lane with the least recorded work still holds warm-up + 2 x
    the mean copy span (so the full-occupancy window cannot be empty by construction).
    """
    raw = _common(template, cell_id, num_workers, split)
    lanes = max(1, int(round(l_per_worker * num_workers)))
    plays = template["transform"]["plays"]
    choices = (copies_per_lane,) if copies_per_lane else (
        AX_COPIES_PER_LANE if warmup_ms is None else (COPIES_PER_LANE,))
    for cpl in choices:
        gen = agentx_trace(plays, trace_split or template["split"], template["transform"],
                           cpl * lanes)
        if warmup_ms is not None:
            break
        spans = np.array(copy_span_list(Path(str(gen["path"]) + ".meta.json")))
        warm = float(round(AX_WARMUP_Q90_MULT * np.percentile(spans, 90) / 1000.0) * 1000.0)
        need = warm + 2.0 * float(spans.mean())
        mins = min_lane_totals(gen, lanes, range(AX_REPLICATES_CHECKED))
        if min(mins) >= need:
            break
    COPIES_USED[cell_id] = {"copies_per_lane": cpl}
    if warmup_ms is None:
        COPIES_USED[cell_id].update(warmup_ms=warm, min_lane_recorded_ms=min(mins), need_ms=need,
                                    ok=bool(min(mins) >= need))
    rel = "CR/" + str(gen["path"].relative_to(CR))
    raw.update(
        trace_format="agentic_mooncake",
        trace_files=[rel],
        trace_sha256=[gen["sha256"]],
        trace_block_size=64,
        derived_meta=rel + ".meta.json",
        trace_rows=gen["rows"],
        expected_cost_s=round(5.0 + 0.006 * gen["rows"], 1),
        lowered={
            "generator": ax.VERSION,
            "spec": {k: v for k, v in gen["spec"].items()},
            "copies_per_lane": cpl,
        },
    )
    raw.pop("measure_trace", None)
    raw["load"] = {
        "mode": "agentic_lanes",
        "value": lanes,
        "level": level,
        "per_worker": l_per_worker,
        "per_worker_unit": "lanes_per_worker",
    }
    raw["measure"] = {"basis": "completion", "end": "full_occupancy",
                      "warmup_ms": warm if warmup_ms is None else warmup_ms}
    return raw


COPIES_USED: dict = {}


def write_cells(cells: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cells))
    return path


def evaluate(cells: list[dict], policies: list[str], repeats: int, out: Path,
             deadline_s: float | None = None, offset: int = 0, slots: int = 20) -> dict:
    import time

    specs = load_specs(policies)
    tasks = [
        Task(spec, Cell(raw=c, layout=LAYOUT), k)
        for c in cells
        for k in range(offset, offset + repeats)
        for spec in specs
    ]
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    started = time.monotonic()
    with Evaluator(LAYOUT, concurrency=slots, results_path=out, log_dir=out.parent / "logs") as ev:
        records = ev.evaluate(tasks, deadline=deadline)
        counts = dict(ev.counts)
    return {
        "tasks": len(tasks),
        "skipped": sum(r is None for r in records),
        "errors": sum(1 for r in records if r is not None and r.get("error")),
        "counts": counts,
        "wall_s": round(time.monotonic() - started, 1),
    }


# --------------------------------------------------------------------------------------------
# Offline rescoring from cached per_request rows

_E0 = None


def e0_table() -> E0Table:
    global _E0
    if _E0 is None:
        _E0 = E0Table(json.loads(LAYOUT.engine_json.read_text()), LAYOUT.e0_dir)
    return _E0


def load_rows(record: dict) -> list[dict]:
    with gzip.open(record["per_request_path"], "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scored_items(record: dict, raw: dict, measure: dict | None = None) -> dict:
    """In-window scored requests of one record as arrays (A2 goodput, harness window rules).

    Returns completed flag, mean ITL (nan when OSL <= 1), slowdown e2e/E0 (nan if incomplete),
    the window length, and the number of unscored-but-counted missing rows (arrival basis).
    """
    cell = Cell(raw=raw, layout=LAYOUT)
    measure = dict(cell.measure if measure is None else measure)
    rows = load_rows(record)
    warmup_ids = None
    if measure.get("warmup_trace_ms") is not None:
        rep = resolve_replicate(cell, int(record["repeat"]))
        warmup_ids = goodput.warmup_ids_from_trace(
            Path(rep.trace_path).read_text().splitlines(), float(measure["warmup_trace_ms"])
        )
    excluded = goodput.warmup_mask(rows, measure, warmup_ids)
    window = goodput.measurement_window(
        rows,
        open_loop=cell.is_open_loop,
        measure=measure,
        makespan_ms=float(record["duration_ms"]),
        excluded=excluded,
        occupancy_cap=None if cell.is_open_loop else int(cell.load["value"]),
    )
    e0 = e0_table()
    start, end = window["start_ms"], window["end_ms"]
    idx = []
    for i, row in enumerate(rows):
        if window["basis"] == "arrival":
            if window["fallback"] or start <= row["arrival_time_ms"] <= end:
                idx.append(i)
        elif not excluded[i] and goodput._completes_inside(row, window):
            idx.append(i)
    missing = 0
    if window["basis"] == "arrival":
        missing = max(int(record["num_requests"]) - len(rows), 0)
    done = np.array([goodput.completed(rows[i]) for i in idx], dtype=bool)
    itl = np.array(
        [
            (goodput.mean_itl_ms(rows[i]) if done[j] else np.nan) or np.nan
            for j, i in enumerate(idx)
        ],
        dtype=float,
    )
    # mean_itl_ms returns None for OSL <= 1: the ITL check is skipped there.
    itl_skip = np.array(
        [done[j] and rows[i]["output_length"] <= 1 for j, i in enumerate(idx)], dtype=bool
    )
    slow = np.array(
        [
            rows[i]["e2e_latency_ms"] / e0(rows[i]["input_length"], rows[i]["output_length"])
            if done[j]
            else np.nan
            for j, i in enumerate(idx)
        ],
        dtype=float,
    )
    e0.persist()
    ttft = np.array([rows[i]["ttft_ms"] if done[j] else np.nan for j, i in enumerate(idx)])
    isl = np.array([rows[i]["input_length"] for i in idx])
    arrival = np.array([rows[i]["arrival_time_ms"] for i in idx], dtype=float)
    order = np.argsort(arrival, kind="stable")
    return {
        "arrival": arrival[order],
        "osl": np.array([rows[i]["output_length"] for i in idx])[order],
        "done": done[order],
        "itl": itl[order],
        "itl_skip": itl_skip[order],
        "slow": slow[order],
        "ttft": ttft[order],
        "isl": isl[order],
        "missing": missing,
        "window_s": (end - start) / 1000.0,
        "window": {k: window.get(k) for k in ("start_ms", "end_ms", "basis", "warmup_ms")},
        "occupancy": {k: window.get(k) for k in goodput.OCCUPANCY_FIELDS},
    }


def good_mask(items: dict, itl_ms: float | None, slowdown: float | None) -> np.ndarray:
    ok = items["done"].copy()
    if itl_ms is not None:
        ok &= items["itl_skip"] | (np.nan_to_num(items["itl"], nan=np.inf) <= itl_ms)
    if slowdown is not None:
        ok &= np.nan_to_num(items["slow"], nan=np.inf) <= slowdown * (1.0 + goodput.E2E_REL_TOL)
    return ok


def gf(items: dict, itl_ms, slowdown) -> tuple[float, float]:
    """(good_frac_window, goodput_rps_window) at (I, S)."""
    good = int(good_mask(items, itl_ms, slowdown).sum())
    n = len(items["done"]) + items["missing"]
    return (good / n if n else math.nan, good / items["window_s"] if items["window_s"] else math.nan)


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# --------------------------------------------------------------------------------------------
# Base-trace token rate (open-loop load unit)


def base_rate_cell(cell: dict) -> dict:
    """The untransformed cell whose trace window defines the open-loop token rate of ``cell``.

    Transforms reuse their base window's speedup (they change requests, not arrivals).
    Mooncake-format windowed families: the candidate (any split) with the same family, segment
    and transform_tag base. Session traces: the unsliced source seed file (identity transform).
    """
    for split in SPLITS:
        for other in candidates(split):
            if (
                other["family"] == cell["family"]
                and other["segment"] == cell["segment"]
                and other["transform_tag"] == "base"
                and other["trace_format"] == "mooncake"
            ):
                return other
    meta = json.loads(resolve(cell["derived_meta"]).read_text())
    if cell["family"] != "synthetic_sessions" or meta["spec"].get("window") is not None:
        raise KeyError(f"no base window for {cell['cell_id']}")
    source = Path(meta["source"]["path"])
    base = copy.deepcopy(cell)
    base["trace_files"] = ["CR/" + str(source.relative_to(CR))]
    base["transform_tag"] = "base"
    return base


def base_token_rate(cell: dict) -> float:
    return token_rate(base_rate_cell(cell))["tokens_per_s"]


# --------------------------------------------------------------------------------------------
# Sessions with think-time-invariant load (long generated traces)

SESSION_LONG_DURATION_S = 48000.0
SESS_WARMUP_S_R0 = 180.0  # calibration r0's fixed warm-up (superseded, audit F1)
SESS_WINDOW_S = 540.0  # replay-time measurement window
SESS_TAIL_S = 60.0  # replay-time session arrivals kept after the window

# Fixer r0 (audit calibration F1): the open-loop warm-up must outlast the session population's
# relaxation. Session lifetime envelope of session i in replay time at think multiplier m:
#     D_i = m * sum_j delay_ij + KAPPA * sum_j E0(ISL_ij, OSL_ij)
# (think time is load-invariant; KAPPA bounds the contended service stretch e2e / E0: measured at
# steady state, mean 1.7-2.0 for default and 2.3-2.7 for round robin, out/late_lifetimes.json).
# Warm-up W = max(180 s, ceil to 60 s of the QUANTILE of D over the train segments' sessions
# (seeds 0-3, sessions starting in [0, LIFETIME_SPAN_S) s) with the cell's transform applied).
# Trace-intrinsic and policy-free; computed on train segments only, applied to every split.
SESS_WARMUP_KAPPA = 3.0
SESS_WARMUP_QUANTILE = 99.0
SESS_WARMUP_MIN_S = 180.0
# Round 2: the warm-up is TWO envelope lifetimes. With one (W0 = 600 s at think 1), default's good
# fraction at N = 8 and 0.45 sessions/s/worker was still 13% above its 2400 s-history value; with
# two it is within 1-3% up to 0.45 (out/conv_check analysis, runs/calibrate-fix-r0/conv_check).
SESS_WARMUP_LIFETIMES = 2
LIFETIME_SEEDS = (0, 1, 2, 3)
LIFETIME_SPAN_S = 2000.0
_WARMUP_MEMO: dict = {}


def _lifetime_spec(template: dict) -> dict:
    t = dict(template["transform"])
    return {
        "window": [0.0, LIFETIME_SPAN_S * 1000.0],
        "isl_unique_mult": t.get("isl_unique_mult", 1.0),
        "isl_prefix_mult": t.get("isl_prefix_mult", 1.0),
        "osl_mult": t.get("osl_mult", 1.0),
        "prefix_root_mult": t.get("prefix_root_mult", 1),
        "think_mult": float(t.get("think_mult", 1.0)),
        "seed": t.get("seed", 0),
        "max_model_len": t.get("max_model_len", 131072),
    }


def session_lifetimes(template: dict, kappa: float = SESS_WARMUP_KAPPA) -> np.ndarray:
    """Envelope lifetimes (s) of the train segments' sessions under ``template``'s transform."""
    from collections import defaultdict as _dd

    from learned_routing.workloads.common import read_jsonl
    from learned_routing.workloads.transform import TransformSpec, transform_mooncake

    spec = TransformSpec.from_dict(_lifetime_spec(template))
    e0 = e0_table()
    out = []
    for seed in LIFETIME_SEEDS:
        rows, _ = transform_mooncake(read_jsonl(long_session_source(seed)), spec, 64)
        think, serv = _dd(float), _dd(float)
        for r in rows:
            think[r["session_id"]] += float(r.get("delay", 0.0))
            serv[r["session_id"]] += e0(r["input_length"], r["output_length"])
        out.extend((think[s] + kappa * serv[s]) / 1000.0 for s in think)
    e0.persist()
    return np.array(out)


def warmup_key(template: dict) -> str:
    return json.dumps(_lifetime_spec(template), sort_keys=True)


def session_warmup_s(template: dict) -> float:
    key = warmup_key(template)
    if key not in _WARMUP_MEMO:
        d = session_lifetimes(template)
        q = float(np.percentile(d, SESS_WARMUP_QUANTILE))
        w0 = max(SESS_WARMUP_MIN_S, math.ceil(q / 60.0) * 60.0)
        w = SESS_WARMUP_LIFETIMES * w0
        fill = float(np.minimum(d, w0).mean() / d.mean())
        fill_r0 = float(np.minimum(d, SESS_WARMUP_S_R0).mean() / d.mean())
        _WARMUP_MEMO[key] = {"warmup_s": w, "envelope_lifetime_s": w0, "lifetimes": SESS_WARMUP_LIFETIMES,
                             "quantile_s": q, "sessions": int(len(d)),
                             "mean_s": float(d.mean()), "fill_at_one_lifetime": fill,
                             "fill_at_180s": fill_r0, "kappa": SESS_WARMUP_KAPPA,
                             "quantile": SESS_WARMUP_QUANTILE}
    return _WARMUP_MEMO[key]["warmup_s"]
CLOSED_WARMUP_SLOTS = 2  # closed loop: warm-up sessions = 2 x concurrency
CLOSED_WINDOW_SLOTS = 10  # closed loop: measured sessions = 10 x concurrency


def long_session_source(seed: int) -> Path:
    return CR / "traces" / "synthetic_sessions" / "long" / f"sessions_seed{seed}_d{int(SESSION_LONG_DURATION_S)}.jsonl"


def session_seed(cell: dict) -> int:
    return int(cell["segment"].split(":s")[1])


class DerivedRef:
    """What a cell needs from a materialized derived trace (path, sha, meta)."""

    def __init__(self, path, meta_path, sha256, meta):
        self.path, self.meta_path, self.sha256, self.meta = Path(path), Path(meta_path), sha256, meta


INDEX_R0 = CAL / "derived_index.json"
INDEX = FIX / "derived_index.json"
_index: dict | None = None


def _index_key(source: Path, spec: dict, block_size: int) -> str:
    import hashlib

    text = json.dumps({"source": str(source), "spec": spec, "bs": block_size}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _load_index() -> dict:
    global _index
    if _index is None:
        _index = json.loads(INDEX_R0.read_text()) if INDEX_R0.exists() else {}
        if INDEX.exists():
            _index.update(json.loads(INDEX.read_text()))
    return _index


def save_index() -> None:
    from learned_routing.canon import atomic_write_text

    atomic_write_text(INDEX, json.dumps(_load_index(), indent=0, sort_keys=True))


def materialize_spec(args) -> tuple[str, dict]:
    """Worker body: materialize one derived trace; returns (index key, entry)."""
    from learned_routing.workloads.transform import TransformSpec, materialize

    source, spec, block_size = args
    d = materialize(Path(source), "mooncake", TransformSpec.from_dict(spec), block_size,
                    CR / "traces" / "derived")
    entry = {"path": str(d.path), "meta_path": str(d.meta_path), "sha256": d.sha256,
             "rows": d.meta["realized"]["rows"]}
    return _index_key(Path(source), spec, block_size), entry


def derive(source: Path, spec: dict, block_size: int = 64) -> DerivedRef:
    index = _load_index()
    key = _index_key(source, spec, block_size)
    entry = index.get(key)
    if entry is None or not Path(entry["path"]).exists():
        key, entry = materialize_spec((str(source), spec, block_size))
        index[key] = entry
        save_index()
    meta = {"realized": {"rows": entry["rows"]}}
    return DerivedRef(entry["path"], entry["meta_path"], entry["sha256"], meta)


def derive_many(jobs: list[tuple[Path, dict]], processes: int = 8, block_size: int = 64) -> None:
    """Materialize missing derived traces in parallel (index updated once at the end)."""
    from concurrent.futures import ProcessPoolExecutor

    index = _load_index()
    todo, seen = [], set()
    for source, spec in jobs:
        key = _index_key(source, spec, block_size)
        if key in seen or (key in index and Path(index[key]["path"]).exists()):
            continue
        seen.add(key)
        todo.append((str(source), spec, block_size))
    if not todo:
        return
    with ProcessPoolExecutor(processes) as pool:
        for key, entry in pool.map(materialize_spec, todo):
            index[key] = entry
    save_index()


def _session_spec(template: dict, window_ms: float, think_mult: float) -> dict:
    t = dict(template["transform"])
    spec = {
        "window": [0.0, float(window_ms)],
        "isl_unique_mult": t.get("isl_unique_mult", 1.0),
        "isl_prefix_mult": t.get("isl_prefix_mult", 1.0),
        "osl_mult": t.get("osl_mult", 1.0),
        "prefix_root_mult": t.get("prefix_root_mult", 1),
        "think_mult": float(think_mult),
        "seed": t.get("seed", 0),
        "max_model_len": t.get("max_model_len", 131072),
    }
    return spec


def _attach_trace(raw: dict, derived) -> None:
    rel = "CR/" + str(Path(derived.path).relative_to(CR))
    raw["trace_files"] = [rel]
    raw["trace_sha256"] = [derived.sha256]
    raw["derived_meta"] = "CR/" + str(Path(derived.meta_path).relative_to(CR))
    raw["trace_rows"] = derived.meta["realized"]["rows"]
    raw["expected_cost_s"] = round(2.0 + 1e-3 * raw["trace_rows"], 2)
    raw.pop("cache_pressure_ref", None)


def session_open_job(template: dict, num_workers: int, rho: float) -> tuple[Path, dict]:
    speedup = float(f"{rho * num_workers:.6g}")
    think = float(template["transform"].get("think_mult", 1.0))
    window_ms = (session_warmup_s(template) + SESS_WINDOW_S + SESS_TAIL_S) * 1000.0 * speedup
    return long_session_source(session_seed(template)), _session_spec(template, window_ms, think * speedup)


def session_closed_job(template: dict, num_workers: int, c_per_worker: float) -> tuple[Path, dict]:
    total = max(1, int(round(c_per_worker * num_workers)))
    warm = max(CLOSED_WARMUP_SLOTS * total, 60)
    sessions = warm + CLOSED_WINDOW_SLOTS * total
    think = float(template["transform"].get("think_mult", 1.0))
    return long_session_source(session_seed(template)), _session_spec(template, sessions * 1000.0, think)


def session_open_cell(template: dict, cell_id: str, num_workers: int, rho: float,
                      split: str = "calib", level: str | None = None) -> dict:
    """Open-loop sessions at rho sessions/s/worker with think times invariant to the speedup.

    The generator emits 1 session/s, so speedup s = rho * N; the derived trace keeps sessions
    starting in [0, (warm-up + window + tail) * s) and pre-multiplies think delays by s (replay
    divides them by s again), times the cell's own think_mult.
    """
    raw = _common(template, cell_id, num_workers, split)
    speedup = float(f"{rho * num_workers:.6g}")
    think = float(template["transform"].get("think_mult", 1.0))
    warmup_s = session_warmup_s(template)
    window_ms = (warmup_s + SESS_WINDOW_S + SESS_TAIL_S) * 1000.0 * speedup
    derived = derive(long_session_source(session_seed(template)),
                     _session_spec(template, window_ms, think * speedup))
    _attach_trace(raw, derived)
    raw["load"] = {
        "mode": "open_speedup",
        "value": speedup,
        "level": level,
        "per_worker": rho,
        "per_worker_unit": "sessions_per_s_per_worker",
    }
    raw["measure"] = {
        "basis": "arrival",
        "warmup_ms": warmup_s * 1000.0,
        "window_ms": SESS_WINDOW_S * 1000.0,
    }
    info = _WARMUP_MEMO[warmup_key(template)]
    raw["measure_trace"] = {
        "note": "think-invariant sessions: trace-time window = replay window x speedup; warm-up = 2 x "
                "max(180 s, ceil60 of the p99 session-lifetime envelope m x sum(think) + 3 x sum(E0) "
                "over train segments s0-s3 with this transform) (calibration fixer r0, audit F1)",
        "trace_window_ms": window_ms,
        "think_mult_in_trace": think * speedup,
        "warmup_rule": "session-lifetime-p99-kappa3-x2-v1",
        "lifetime_envelope_p99_s": round(info["quantile_s"], 3),
        "lifetime_envelope_rounded_s": info["envelope_lifetime_s"],
    }
    return raw


def session_closed_cell(template: dict, cell_id: str, num_workers: int, c_per_worker: float,
                        split: str = "calib", level: str | None = None) -> dict:
    raw = _common(template, cell_id, num_workers, split)
    total = max(1, int(round(c_per_worker * num_workers)))
    warm = max(CLOSED_WARMUP_SLOTS * total, 60)
    sessions = warm + CLOSED_WINDOW_SLOTS * total
    think = float(template["transform"].get("think_mult", 1.0))
    derived = derive(long_session_source(session_seed(template)),
                     _session_spec(template, sessions * 1000.0, think))
    _attach_trace(raw, derived)
    raw["load"] = {
        "mode": "closed_concurrency",
        "value": total,
        "level": level,
        "per_worker": c_per_worker,
        "per_worker_unit": "sessions_per_worker",
    }
    raw["measure"] = {
        "basis": "completion",
        "end": "full_occupancy",
        "warmup_trace_ms": float(warm * 1000.0),
    }
    raw["measure_trace"] = {
        "note": "closed sessions: generator emits 1 session/s; warm-up = first 2C sessions by identity, then 10C measured sessions",
        "warmup_sessions_nominal": warm,
        "sessions_nominal": sessions,
    }
    return raw
