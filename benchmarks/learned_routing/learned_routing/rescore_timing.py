# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score timing-perturbed records against both the nominal and the perturbation-consistent E0.

``lr-eval`` scores every record against the nominal E0 (:mod:`learned_routing.e0`): a cell's
``engine_overrides`` (``speedup_ratio``, ``decode_speedup_ratio``; LR-14 timing robustness) do not
change E0. A slower engine is therefore also a tighter E2E-slowdown SLA. On AgentX (S = 1.16) at
speedup 0.8 no request without prefix reuse can be good, so the cell degenerates into a
cache-hit metric (phase2-mid sim-exploitation audit F2).

The consistent reference re-anchors the SLO to the perturbed engine:

    E0'(ISL, OSL) = prefill(ISL) / s + decode(ISL, OSL) / (s * d)

with ``s = speedup_ratio`` and ``d = decode_speedup_ratio``: aisimulate-core divides a prefill
pass by ``s`` and a decode pass by ``s * d``, and E0 is a batch-1 request alone (pure prefill
chunks, then pure decode steps), so E0' is that request's uncontended latency on the perturbed
engine. Nominal cells (``s = d = 1``) get E0' = E0.

Each output row carries both scorings (``nominal_e0`` = the record's own metrics, recomputed and
asserted equal to the record's ``goodput_rps_window``; ``consistent_e0``), each with
``goodput_rps_window``, ``good_frac_window``, ``slowdown_atom_frac`` and the SLO-scale
``rescore``. No replay is run; rows come from each record's ``per_request_path``.

    python -m learned_routing.rescore_timing --results R.jsonl [R.jsonl ...] \
        --cells C.jsonl [C.jsonl ...] --out OUT.jsonl [--root CR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

from learned_routing import goodput
from learned_routing.cells import Cell, load_cells, resolve_replicate
from learned_routing.e0 import E0Table
from learned_routing.paths import Layout

TIMING_KEYS = ("speedup_ratio", "decode_speedup_ratio")
CONSISTENT_METHOD = "prefill/s + decode/(s*d) over ais-chunked-estimator-v2"
KEPT = ("goodput_rps_window", "good_frac_window", "slowdown_atom_frac", "rescore")
IDENTITY = (
    "policy_sha",
    "policy_name",
    "cell_id",
    "cell_sha",
    "repeat",
    "family",
    "split",
    "segment",
    "load_level",
    "load_mode",
    "num_workers",
)


class RescoreError(RuntimeError):
    pass


def timing_factors(overrides: dict | None) -> tuple[float, float]:
    """(s, d) from a cell's ``engine_overrides``; 1.0 for an absent knob."""
    overrides = overrides or {}
    s = float(overrides.get("speedup_ratio", 1.0))
    d = float(overrides.get("decode_speedup_ratio", 1.0))
    if s <= 0 or d <= 0:
        raise RescoreError(f"timing factors must be positive, got s={s}, d={d}")
    return s, d


class ConsistentE0:
    """E0' = prefill / s + decode / (s d), from a nominal :class:`E0Table`."""

    def __init__(
        self, table: E0Table, speedup_ratio: float, decode_speedup_ratio: float
    ):
        self.table = table
        self.s = float(speedup_ratio)
        self.d = float(decode_speedup_ratio)

    def __call__(self, isl: int, osl: int) -> float:
        full = self.table(isl, osl)
        if self.s == 1.0 and self.d == 1.0:
            return full
        prefill = self.table.prefill_ms(isl)
        return prefill / self.s + (full - prefill) / (self.s * self.d)


def _rows(record: dict) -> list[dict]:
    path = record.get("per_request_path")
    if not path or not Path(path).exists():
        raise RescoreError(
            f"{record.get('cell_id')} k{record.get('repeat')}: no per-request rows "
            f"(per_request_path {path!r}); evaluate without --no-per-request"
        )
    return [
        json.loads(x) for x in gzip.decompress(Path(path).read_bytes()).splitlines()
    ]


def _metrics(rows, record: dict, cell: Cell, e0, warmup_ids) -> dict:
    summary = {
        "num_requests": record["num_requests"],
        "duration_ms": record["duration_ms"],
        "completed_requests": record["completed"],
    }
    return goodput.compute_metrics(
        rows,
        summary,
        sla=cell.sla,
        measure=cell.measure,
        open_loop=cell.is_open_loop,
        num_workers=cell.num_workers,
        e0=e0,
        warmup_ids=warmup_ids,
        occupancy_cap=None if cell.is_open_loop else int(cell.load["value"]),
    )


def rescore_record(
    record: dict, cell: Cell, table: E0Table, warmup_cache: dict | None = None
) -> dict:
    """Both scorings of one ``lr-eval`` record (see the module docstring)."""
    if record.get("error"):
        raise RescoreError(f"record {record.get('cache_key')} has an error")
    if record.get("cell_sha") and record["cell_sha"] != cell.content_sha():
        raise RescoreError(
            f"{record['cell_id']}: record cell_sha {record['cell_sha'][:12]} "
            f"!= cell content {cell.content_sha()[:12]}"
        )
    s, d = timing_factors(cell.raw.get("engine_overrides"))
    rows = _rows(record)
    warmup_ids = None
    warmup_trace_ms = cell.measure.get("warmup_trace_ms")
    if warmup_trace_ms is not None:
        trace = str(resolve_replicate(cell, int(record["repeat"])).trace_path)
        cache = warmup_cache if warmup_cache is not None else {}
        if trace not in cache:
            cache[trace] = goodput.warmup_ids_from_trace(
                Path(trace).read_text().splitlines(), float(warmup_trace_ms)
            )
        warmup_ids = cache[trace]
    nominal = _metrics(rows, record, cell, table, warmup_ids)
    if nominal["goodput_rps_window"] != record["goodput_rps_window"]:
        raise RescoreError(
            f"{record['cell_id']} k{record['repeat']}: nominal recompute "
            f"{nominal['goodput_rps_window']!r} != record {record['goodput_rps_window']!r}"
        )
    consistent = _metrics(rows, record, cell, ConsistentE0(table, s, d), warmup_ids)
    out = {k: record.get(k) for k in IDENTITY}
    out.update(
        speedup_ratio=s,
        decode_speedup_ratio=d,
        engine_overrides=dict(cell.raw.get("engine_overrides") or {}),
        consistent_e0_method=CONSISTENT_METHOD,
        nominal_e0={k: nominal.get(k) for k in KEPT},
        consistent_e0={k: consistent.get(k) for k in KEPT},
    )
    return out


def rescore_files(
    results: list[Path], cells: list[Path], out: Path, layout: Layout
) -> dict:
    by_content: dict[tuple[str, str], Cell] = {}
    for path in cells:
        for cell in load_cells(path, layout):
            by_content[(cell.cell_id, cell.content_sha())] = cell
    engine = json.loads(layout.engine_json.read_text())
    table = E0Table(engine, layout.e0_dir)
    warmup_cache: dict = {}
    rows_out, skipped = [], 0
    for path in results:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                skipped += 1
                continue
            cell = by_content.get((record["cell_id"], record.get("cell_sha")))
            if cell is None:
                raise RescoreError(
                    f"no cell with id {record['cell_id']} and content "
                    f"{str(record.get('cell_sha'))[:12]} in {[str(c) for c in cells]}"
                )
            rows_out.append(rescore_record(record, cell, table, warmup_cache))
    table.persist()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_out))
    return {"rescored": len(rows_out), "skipped_errors": skipped, "out": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--cells", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    layout = Layout.resolve(args.root)
    print(json.dumps(rescore_files(args.results, args.cells, args.out, layout)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
