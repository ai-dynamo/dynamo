# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Window-basis LR-13 guards from a record's per-request rows (phase2-mid gaming audit F1/F2/F4).

The record's own ``guards`` (:func:`learned_routing.goodput.guard_metrics`) count every row,
warm-up and drain included, and have no per-worker good fraction or long-request view, so they
could not see the phase-2 M1 failure: one worker per replicate holding only requests of 60K+
tokens, none of them good. This module recomputes, over exactly the requests the record's
goodput window scores:

- ``share_max_window`` against the cap ``min(2/N, 1/N + 0.25)`` (``violates_cap``);
- ``worker_good_frac_min``, the hot (max-share) worker's and the other workers' good fractions;
- good fractions by ISL: ``>= 32K`` and ``>= 64K`` tokens, and the top ISL decile (cut at the
  90th percentile of the record's ISLs, a property of the workload replicate);
- ``nmi_worker_islq``: normalized mutual information I(worker; ISL quartile) / H(ISL quartile)
  (segregation by request size);
- ``dump_workers``: workers holding at least one request of ``>= 60K`` tokens whose in-window
  requests are at least 80% such requests, and the good fraction of those long requests.

Definitions follow the phase2-mid gaming audit (runs/audits/phase2-mid-gaming/scripts/analyze.py,
giant.py), and the window, good and warm-up rules are the harness's own (:mod:`goodput`); the
recomputed in-window good count must equal the record's ``window_good``.

    python -m learned_routing.window_guards --results R.jsonl [...] --cells C.jsonl [...] \
        --out OUT.json [--root CR]

writes ``per_record`` (keyed ``policy_name|cell_id|repeat``) and ``by_policy_cell`` (means over
replicates).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from learned_routing import goodput
from learned_routing.cells import Cell, load_cells, resolve_replicate
from learned_routing.e0 import E0Table
from learned_routing.paths import Layout

GIANT_ISL = 60_000
DUMP_FRAC = 0.8
ISL_32K = 32_768
ISL_64K = 65_536


class GuardError(RuntimeError):
    pass


def cap_for(num_workers: int) -> float:
    return min(2.0 / num_workers, 1.0 / num_workers + 0.25)


def _pct(values, q: float):
    if not values:
        return None
    s = sorted(values)
    r = q / 100 * (len(s) - 1)
    lo = math.floor(r)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (r - lo)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return (
        -sum(c / total * math.log(c / total) for c in counter.values() if c)
        if total
        else 0.0
    )


def nmi(pairs) -> float | None:
    """I(W; X) / H(X) over (w, x) pairs."""
    if not pairs:
        return None
    hw = _entropy(Counter(w for w, _ in pairs))
    hx = _entropy(Counter(x for _, x in pairs))
    hj = _entropy(Counter(pairs))
    return (hw + hx - hj) / hx if hx > 0 else None


def in_window_flags(
    rows, record: dict, cell: Cell, e0, warmup_ids
) -> list[tuple[bool, bool]]:
    """(in_window, good) per row, by the harness's window and A2 rules."""
    excluded = goodput.warmup_mask(rows, cell.measure, warmup_ids)
    start, end = record["window_start_ms"], record["window_end_ms"]
    out = []
    for row, skip in zip(rows, excluded):
        if record["window_basis"] == "arrival":
            inside = bool(record.get("window_fallback")) or (
                start <= row["arrival_time_ms"] <= end
            )
        else:
            inside = (
                not skip
                and goodput.completed(row)
                and start <= row["terminal_time_ms"] < end
            )
        out.append((inside, goodput.a2_good(row, cell.sla, e0)))
    return out


def record_guards(rows, record: dict, cell: Cell, e0, warmup_ids=None) -> dict:
    n_workers = int(cell.num_workers)
    flags = in_window_flags(rows, record, cell, e0, warmup_ids)
    win = [i for i, (inside, _) in enumerate(flags) if inside]
    good_in = sum(flags[i][1] for i in win)
    if good_in != record["window_good"]:
        raise GuardError(
            f"{record.get('policy_name')} {record['cell_id']} k{record['repeat']}: in-window good "
            f"{good_in} != record window_good {record['window_good']}"
        )
    worker = [goodput.worker_of(r) for r in rows]
    wc = Counter(worker[i] for i in win)
    total = sum(wc.values())
    hot, hot_n = max(wc.items(), key=lambda kv: (kv[1], -kv[0])) if wc else (None, 0)
    isls = [r["input_length"] for r in rows]
    p90, p75, p50, p25 = (_pct(isls, q) for q in (90, 75, 50, 25))

    def quartile(isl):
        if isl >= p75:
            return "q4"
        if isl >= p50:
            return "q3"
        if isl >= p25:
            return "q2"
        return "q1"

    per_worker_good = defaultdict(list)
    for i in win:
        per_worker_good[worker[i]].append(flags[i][1])

    def good_where(pred):
        sel = [flags[i][1] for i in win if pred(rows[i]["input_length"])]
        return (sum(sel) / len(sel) if sel else None), len(sel)

    g32, n32 = good_where(lambda isl: isl >= ISL_32K)
    g64, n64 = good_where(lambda isl: isl >= ISL_64K)
    gtop, ntop = good_where(lambda isl: isl >= p90)
    giants = Counter()
    for i in win:
        giants[worker[i]] += rows[i]["input_length"] >= GIANT_ISL
    dumps = sorted(w for w, n in wc.items() if giants[w] and giants[w] >= DUMP_FRAC * n)
    giant_flags = [flags[i][1] for i in win if rows[i]["input_length"] >= GIANT_ISL]
    share = hot_n / total if total else None
    cap = cap_for(n_workers)
    return {
        "n_window": total,
        "cap": cap,
        "share_max_window": share,
        "violates_cap": share is not None and share > cap,
        "hot_worker": hot,
        "good_frac_window": good_in / total if total else None,
        "hot_good_frac": _mean(per_worker_good[hot]) if hot is not None else None,
        "others_good_frac": _mean([flags[i][1] for i in win if worker[i] != hot]),
        "worker_good_frac_min": min(_mean(v) for v in per_worker_good.values())
        if per_worker_good
        else None,
        "good_isl_ge_32k": g32,
        "n_isl_ge_32k": n32,
        "good_isl_ge_64k": g64,
        "n_isl_ge_64k": n64,
        "good_top_isl_decile": gtop,
        "n_top_isl_decile": ntop,
        "isl_cuts_p90_p75_p50_p25": [p90, p75, p50, p25],
        "nmi_worker_islq": nmi(
            [(worker[i], quartile(rows[i]["input_length"])) for i in win]
        ),
        "giants_in_window": len(giant_flags),
        "good_giants": sum(giant_flags) / len(giant_flags) if giant_flags else None,
        "dump_workers": len(dumps),
    }


def _rows(record: dict) -> list[dict]:
    path = record.get("per_request_path")
    if not path or not Path(path).exists():
        raise GuardError(
            f"{record['cell_id']} k{record['repeat']}: no per-request rows ({path!r})"
        )
    return [
        json.loads(x) for x in gzip.decompress(Path(path).read_bytes()).splitlines()
    ]


def guards_files(results: list[Path], cells: list[Path], layout: Layout) -> dict:
    by_content: dict[tuple[str, str], Cell] = {}
    for path in cells:
        for cell in load_cells(path, layout):
            by_content[(cell.cell_id, cell.content_sha())] = cell
    table = E0Table(json.loads(layout.engine_json.read_text()), layout.e0_dir)
    warm: dict[str, frozenset] = {}
    per_record = {}
    for path in results:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error"):
                continue
            cell = by_content.get((record["cell_id"], record.get("cell_sha")))
            if cell is None:
                raise GuardError(
                    f"no cell {record['cell_id']} with the record's content sha"
                )
            warmup_ids = None
            if cell.measure.get("warmup_trace_ms") is not None:
                trace = str(resolve_replicate(cell, int(record["repeat"])).trace_path)
                if trace not in warm:
                    warm[trace] = goodput.warmup_ids_from_trace(
                        Path(trace).read_text().splitlines(),
                        float(cell.measure["warmup_trace_ms"]),
                    )
                warmup_ids = warm[trace]
            key = f"{record.get('policy_name')}|{record['cell_id']}|{record['repeat']}"
            per_record[key] = record_guards(
                _rows(record), record, cell, table, warmup_ids
            )
    table.persist()
    sums: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for key, m in per_record.items():
        policy, cell_id, _ = key.rsplit("|", 2)
        for field, value in m.items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                sums[f"{policy}|{cell_id}"][field].append(value)
    by_policy_cell = {
        pc: {f: sum(v) / len(v) for f, v in fields.items()} | {"k": len(fields["cap"])}
        for pc, fields in sums.items()
    }
    return {"per_record": per_record, "by_policy_cell": by_policy_cell}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+", required=True, type=Path)
    ap.add_argument("--cells", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    out = guards_files(args.results, args.cells, Layout.resolve(args.root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"records": len(out["per_record"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
