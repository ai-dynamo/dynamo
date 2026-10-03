"""Freeze (stage 2 step 3): fill load, SLA and measurement rules into every cell.

usage: build_final.py DECISIONS.json OUT_DIR
Writes OUT_DIR/{train,val,test}.jsonl (cells) and OUT_DIR/build_report.json. The caller copies
them into CR/cells after review. Rules: facts/calibration.json ("rules").
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import calib

ADDED_TEST_CELLS = [
    # (new cell id, template cell id, segment subset, N, level); Amendment A3: lowered AgentX
    # reaches every N of the grid, so the worker-count hold-out gains AgentX at N = 16 and 32,
    # mirroring the N = 2 and N = 6 AgentX cells (one L2 and one L3 cell per N).
    ("agentx-T3-base-n16-lanes-L2", "agentx-T3-base-n2-lanes-L2", "T3", 16, "L2"),
    ("agentx-T1-base-n16-lanes-L3", "agentx-T1-base-n6-lanes-L3", "T1", 16, "L3"),
    ("agentx-T3-base-n32-lanes-L2", "agentx-T3-base-n6-lanes-L2", "T3", 32, "L2"),
    ("agentx-T4-base-n32-lanes-L3", "agentx-T4-base-n2-lanes-L3", "T4", 32, "L3"),
]

# Held-out FAST25 families have no train cells: they inherit Mooncake's SLA and per-worker levels
# (open loop: offered input tokens per second per worker over their own base window).
FAMILY_SOURCE = {
    "mooncake": "mooncake",
    "fast25_conversation": "mooncake",
    "fast25_synthetic": "mooncake",
    "synthetic_sessions": "synthetic_sessions",
    "agentx": "agentx",
}


def per_worker(dec: dict, family: str, mode: str, level: str, n: int) -> float:
    value = dec["levels"][FAMILY_SOURCE[family]][mode].get(str(n), {}).get(level)
    if value is None:
        raise MissingLevel(f"{family}/{mode}/N{n}/{level}")
    return float(value)


def finalize(cell: dict, dec: dict) -> dict:
    fam, mode, level, n = cell["family"], cell["load"]["mode"], cell["load"]["level"], int(cell["num_workers"])
    tag = cell.get("transform_tag", "base")
    if tag != "base":
        # transformed cells: the same band rule, calibrated on train segments with the transform
        value = dec["transform_levels"].get(f"{fam}|{tag}|{mode}|{n}", {}).get(level)
        if value is None:
            raise MissingLevel(f"{fam}/{tag}/{mode}/N{n}/{level}")
        v = float(value)
    else:
        v = per_worker(dec, fam, mode, level, n)
    sla = dec["sla"][FAMILY_SOURCE[fam]]
    split = cell["split"]
    if fam == "synthetic_sessions":
        if mode == "open_speedup":
            out = calib.session_open_cell(cell, cell["cell_id"], n, v, split=split, level=level)
        else:
            out = calib.session_closed_cell(cell, cell["cell_id"], n, v, split=split, level=level)
    elif fam == "agentx":
        out = calib.lanes_cell(cell, cell["cell_id"], n, v, None, split=split, level=level,
                               trace_split=split if split in calib.SPLITS else "train")
        used = calib.COPIES_USED[cell["cell_id"]]
        if not used["ok"]:
            raise ValueError(f"{cell['cell_id']}: no copies-per-lane choice keeps a window: {used}")
        out["measure_trace"] = {
            "note": "lanes warm-up = 1.0 x p90 recorded span over the cell's copies; copies per "
                    "lane = smallest of (8, 10, 12, 16, 24) whose least-loaded lane holds warm-up "
                    "+ 2 x mean copy span in every replicate k < 16 (trace-intrinsic check)",
            **used,
        }
    elif mode == "open_speedup":
        template = copy.deepcopy(cell)
        base = calib.base_rate_cell(cell)
        rate = calib.token_rate(base)["tokens_per_s"]
        speedup = float(f"{v * n / rate:.6g}")
        out = calib._common(template, cell["cell_id"], n, split)
        mt = cell["measure_trace"]
        out["load"] = {"mode": "open_speedup", "value": speedup, "level": level,
                       "per_worker": v, "per_worker_unit": "input_tokens_per_s_per_worker",
                       "base_window_tokens_per_s": rate}
        out["measure"] = {"basis": "arrival",
                          "warmup_ms": float(f"{mt['warmup_ms'] / speedup:.9g}"),
                          "window_ms": float(f"{mt['window_ms'] / speedup:.9g}")}
    else:
        out = calib.closed_cell(cell, cell["cell_id"], n, v, split=split, level=level)
    out["sla"] = {"ttft_ms": None, "itl_ms": sla["itl_ms"], "e2e_slowdown": sla["e2e_slowdown"]}
    out["load"]["rule"] = dec["rule_id"]
    out["load"]["level"] = level
    out["split"] = split
    for key in ("holdout_axis", "holdout_axes", "selection_exposed", "segment", "transform_tag"):
        if key in cell:
            out[key] = cell[key]
    out["calibration"] = dec["calibration_id"]
    return out


def calib_spans(cell: dict) -> list[float]:
    meta = json.loads(calib.resolve(cell["derived_meta"]).read_text())
    manifest = json.loads(Path(meta["base_manifest"]).read_text())["plays"]
    return [manifest[c["play"]]["span_ms"] for c in meta["copies"]]


def added_cells() -> list[dict]:
    test = {c["cell_id"]: c for c in calib.candidates("test")}
    subsets = json.loads((calib.CR / "cells" / "SPLIT_MANIFEST.json").read_text())["agentx_play_subsets"]
    out = []
    for cid, tid, subset, n, level in ADDED_TEST_CELLS:
        cell = copy.deepcopy(test[tid])
        cell["cell_id"] = cid
        cell["num_workers"] = n
        cell["segment"] = f"agentx:{subset}"
        cell["load"] = {"mode": "agentic_lanes", "value": None, "level": level}
        cell["transform"]["plays"] = sorted(subsets[subset]["plays"])
        cell["holdout_axes"] = ["worker_count", "agentx_plays"]
        cell["holdout_axis"] = cell.get("holdout_axis")
        cell["selection_exposed"] = False
        cell["notes"] = "added at calibration (Amendment A3: lowered AgentX at every N)"
        out.append(cell)
    return out


NOISE_TEMPLATES = [
    # (family tag, train candidate used as template, mode)
    ("mooncake-w0", "mooncake-w0-base-n8-open-L2"),
    ("mooncake-w2", "mooncake-w2-base-n8-closed-L1"),
    ("sessions-s0", "sessions-s0-base-n4-open-L2"),
    ("sessions-s0", "sessions-s0-base-n8-closed-L2"),
    ("agentx-A1", "agentx-A1-base-n4-lanes-L1"),
]
NOISE_N = (2, 16, 32)


def noise_cells(dec: dict) -> list[dict]:
    """Calibration-only cells on train segments at N = 2, 16, 32 (L2): noise per family and N."""
    train = {c["cell_id"]: c for c in calib.candidates("train")}
    out = []
    for tag, tid in NOISE_TEMPLATES:
        for n in NOISE_N:
            cell = copy.deepcopy(train[tid])
            mode = cell["load"]["mode"].split("_")[0]
            cell["cell_id"] = f"noise-{tag}-base-n{n}-{mode}-L2"
            cell["num_workers"] = n
            cell["load"] = {"mode": cell["load"]["mode"], "value": None, "level": "L2"}
            cell["split"] = "calib"
            cell["notes"] = "calibration-only noise cell (train segment), never a split cell"
            try:
                out.append(finalize(cell, dec))
            except MissingLevel:
                if not PARTIAL[0]:
                    raise
    return out


PARTIAL = [False]


class MissingLevel(KeyError):
    pass


def main():
    dec = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(sys.argv[2])
    partial = "--partial" in sys.argv[3:]
    PARTIAL[0] = partial
    splits = [a for a in sys.argv[3:] if a in calib.SPLITS] or list(calib.SPLITS)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"cells": {}}
    report["skipped"] = []
    for split in splits:
        cells = calib.candidates(split)
        if split == "test":
            cells += added_cells()
        final = []
        for c in cells:
            try:
                final.append(finalize(c, dec))
            except MissingLevel as exc:
                if not partial:
                    raise
                report["skipped"].append(str(exc))
        calib.write_cells(final, out_dir / f"{split}.jsonl")
        report["cells"][split] = len(final)
    noise = noise_cells(dec)
    calib.write_cells(noise, out_dir / "noise.jsonl")
    report["cells"]["noise"] = len(noise)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=1))
    calib.save_index()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
