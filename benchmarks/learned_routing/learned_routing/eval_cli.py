# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``lr-eval``: evaluate policies on cells through the slot pool and the result cache.

Examples::

    lr-eval --policy-spec default round_robin --cells CR/cells/val.jsonl \\
        --out CR/runs/pilot/r1/results.jsonl --repeats 3 --max-wall-seconds 540
    lr-eval --policy-spec specs.jsonl --cells test.jsonl --repeats 3 --bundle-out /tmp/shard
    lr-eval --ingest /tmp/shard --out CR/runs/test/remote/results.jsonl
    lr-eval --slots-status

Repeats are CRN workload replicates k = offset .. offset + K - 1 (Amendment A1). Exit status:
0 all tasks evaluated without errors; 2 all evaluated, some errored; 3 stopped early by
``--max-wall-seconds`` (rerun to continue; finished results are cached); 1 usage or setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from learned_routing.cells import load_cells
from learned_routing.evaluate import Evaluator, Task
from learned_routing.paths import NUM_SLOTS, Layout
from learned_routing.policy import load_specs
from learned_routing.slots import SlotPool

EXIT_ERRORS = 2
EXIT_INCOMPLETE = 3


def build_tasks(specs, cells, repeats: int, offset: int, shard: tuple[int, int] | None):
    tasks = [
        Task(spec, cell, k)
        for cell in cells
        for k in range(offset, offset + repeats)
        for spec in specs
    ]
    if shard is None:
        return tasks
    index, count = shard

    def owner(task: Task) -> int:
        # Shard by (cell, k) so every policy of a CRN pair lands on the same node.
        digest = hashlib.sha256(f"{task.cell.cell_id}|{task.k}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % count

    return [task for task in tasks if owner(task) == index]


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if not match or not 0 <= int(match[1]) < int(match[2]):
        raise argparse.ArgumentTypeError(
            f"--shard must be i/n with 0 <= i < n, got {text!r}"
        )
    return int(match[1]), int(match[2])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lr-eval", description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--root", help="campaign root (default: LR_ROOT or the campaign directory)"
    )
    p.add_argument(
        "--policy-spec",
        nargs="+",
        default=[],
        help="spec files (JSON object/list, JSONL, YAML) or builtin names: default, round_robin",
    )
    p.add_argument("--cells", nargs="+", default=[], help="cells JSONL file(s)")
    p.add_argument("--cell-filter", help="regex on cell_id")
    p.add_argument("--out", help="results JSONL to append to")
    p.add_argument(
        "--repeats", type=int, default=1, help="CRN replicates per (policy, cell)"
    )
    p.add_argument(
        "--repeat-offset", type=int, default=0, help="first replicate index k"
    )
    p.add_argument(
        "--slots", type=int, default=NUM_SLOTS, help="max concurrent replays here"
    )
    p.add_argument(
        "--num-slots",
        type=int,
        default=NUM_SLOTS,
        help="size of the machine-wide slot pool",
    )
    p.add_argument("--slots-dir", help="slot pool directory (default ROOT/slots)")
    p.add_argument(
        "--max-wall-seconds", type=float, help="stop launching replays after this"
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        help="fixed per-replay timeout (default 20x cost, >=120 s)",
    )
    p.add_argument(
        "--no-per-request", action="store_true", help="do not keep per-request rows"
    )
    p.add_argument("--refresh", action="store_true", help="ignore cached results")
    p.add_argument(
        "--shard", type=parse_shard, help="evaluate shard i/n of the (cell, k) set"
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--bundle-out",
        help="write a self-contained remote bundle instead of evaluating",
    )
    p.add_argument(
        "--ingest", help="merge a remote bundle's results into the local cache"
    )
    p.add_argument(
        "--ingest-any-build",
        action="store_true",
        help="accept remote results whose bindings build differs from the local one",
    )
    p.add_argument(
        "--slots-status", action="store_true", help="print slot holders and exit"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    layout = Layout.resolve(args.root)
    if args.slots_status:
        slots_dir = Path(args.slots_dir) if args.slots_dir else layout.slots_dir
        for row in SlotPool(slots_dir, args.num_slots).status():
            print(json.dumps(row))
        return 0
    if args.ingest:
        from learned_routing.bundle import ingest

        summary = ingest(
            Path(args.ingest), layout, out=args.out, any_build=args.ingest_any_build
        )
        print(json.dumps(summary, sort_keys=True))
        return 0 if not summary["rejected"] else EXIT_ERRORS
    if not args.policy_spec or not args.cells:
        print("lr-eval: --policy-spec and --cells are required", file=sys.stderr)
        return 1
    specs = load_specs(args.policy_spec)
    cells = [cell for path in args.cells for cell in load_cells(path, layout)]
    if args.cell_filter:
        pattern = re.compile(args.cell_filter)
        cells = [cell for cell in cells if pattern.search(cell.cell_id)]
    if not cells:
        print("lr-eval: no cells selected", file=sys.stderr)
        return 1
    if args.bundle_out:
        from learned_routing.bundle import build_bundle

        manifest = build_bundle(
            Path(args.bundle_out),
            layout,
            specs=specs,
            cells=cells,
            repeats=args.repeats,
            repeat_offset=args.repeat_offset,
        )
        print(
            json.dumps(
                {k: manifest[k] for k in ("bundle", "tasks", "build_id", "bytes")}
            )
        )
        return 0
    tasks = build_tasks(specs, cells, args.repeats, args.repeat_offset, args.shard)
    deadline = None
    if args.max_wall_seconds:
        deadline = time.monotonic() + args.max_wall_seconds
    started = time.monotonic()
    with Evaluator(
        layout,
        concurrency=min(args.slots, args.num_slots),
        num_slots=args.num_slots,
        keep_per_request=not args.no_per_request,
        timeout_s=args.timeout_s,
        results_path=Path(args.out) if args.out else None,
        log_dir=(Path(args.out).parent / "logs") if args.out else None,
        verbose=not args.quiet,
        refresh=args.refresh,
        slots_dir=Path(args.slots_dir) if args.slots_dir else None,
    ) as evaluator:
        records = evaluator.evaluate(tasks, deadline=deadline)
        counts = dict(evaluator.counts)
        build = evaluator.build["build_id"]
    skipped = sum(record is None for record in records)
    errors = sum(1 for record in records if record is not None and record.get("error"))
    summary = {
        "tasks": len(tasks),
        "evaluated": len(records) - skipped,
        "skipped": skipped,
        "errors": errors,
        "counts": counts,
        "wall_s": round(time.monotonic() - started, 2),
        "build_id": build,
        "out": args.out,
    }
    print(json.dumps(summary, sort_keys=True))
    if skipped:
        return EXIT_INCOMPLETE
    return EXIT_ERRORS if errors else 0


def main_exit() -> None:
    sys.exit(main())


if __name__ == "__main__":
    main_exit()
