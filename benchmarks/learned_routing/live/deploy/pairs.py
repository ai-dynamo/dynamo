# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a batch of (label, policy, input) pairs for one live job (job_node.sh pairs mode).

A pairs file lists the measured runs of one allocation, in run order, one per line::

    # label        policy slug (plan)                 input run (dir under LR_SMOKE_INPUTS)   est_s
    default_a      default_defaults                   conv-w3-base-n4-open-L2__default_a      780
    m1v2           m1v2_p2-m1v2-s1-g20-best_so_far    conv-w3-base-n4-open-L2__m1v2           775

``est_s`` is the expected AIPerf wall time in seconds (job_node.sh skips a pair that no longer
fits before the job's end). ``#`` starts a comment. ``check`` refuses the batch unless every pair
names a planned policy and a generated input whose replicate is the plan's, whose worker count is
the job's, and whose salt no other pair uses (each live run needs its own salt so no two runs
share a KV block). Labels must be unique, so every pair gets its own output directory. Standard
library only: job_node.sh runs it with the head node's host python3 before any setup cost.

    pairs.py check --pairs FILE --plan-dir PLAN --inputs-root DIR --num-workers 4 [--table-out TSV]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Pair:
    index: int
    label: str
    slug: str
    run: str
    est_s: int


class PairsError(ValueError):
    pass


def parse(text: str) -> list[Pair]:
    pairs: list[Pair] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise PairsError(
                f"line {lineno}: want 'label slug run est_s', got {len(fields)} fields"
            )
        label, slug, run, est = fields
        for what, value in (("label", label), ("slug", slug), ("run", run)):
            if not NAME.match(value):
                raise PairsError(
                    f"line {lineno}: {what} {value!r} is not [A-Za-z0-9._-]+"
                )
        if not est.isdigit() or int(est) <= 0:
            raise PairsError(f"line {lineno}: est_s {est!r} is not a positive integer")
        pairs.append(Pair(len(pairs) + 1, label, slug, run, int(est)))
    if not pairs:
        raise PairsError("no pairs")
    return pairs


def check(
    pairs: list[Pair], plan_dir: Path, inputs_root: Path, num_workers: int
) -> list[str]:
    """Problems with the batch; empty when every pair can run in this job."""
    problems: list[str] = []
    replicate = json.loads((plan_dir / "PLAN.json").read_text())["replicate"]
    labels: dict[str, int] = {}
    salts: dict[str, int] = {}
    for pair in pairs:
        where = f"pair {pair.index} ({pair.label})"
        if pair.label in labels:
            problems.append(f"{where}: label repeats pair {labels[pair.label]}")
        labels.setdefault(pair.label, pair.index)
        if not (plan_dir / "policies" / pair.slug / "policy_plan.json").is_file():
            problems.append(f"{where}: policy {pair.slug!r} is not in the plan")
        manifest_path = inputs_root / pair.run / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{where}: unreadable {manifest_path}: {exc}")
            continue
        if not (inputs_root / pair.run / "aiperf_input.jsonl").is_file():
            problems.append(f"{where}: {pair.run} has no aiperf_input.jsonl")
        # Idle calibration inputs carry no cell: no replicate or worker count to match.
        if manifest.get("cell_id") is not None:
            if manifest.get("k") != replicate:
                problems.append(
                    f"{where}: input replicate k={manifest.get('k')} but the plan is k={replicate}"
                )
            if manifest.get("num_workers") != num_workers:
                problems.append(
                    f"{where}: input is for {manifest.get('num_workers')} workers, job has {num_workers}"
                )
        salt = manifest.get("salt")
        if not salt:
            problems.append(f"{where}: input manifest has no salt")
        elif salt in salts:
            problems.append(f"{where}: salt {salt!r} repeats pair {salts[salt]}")
        else:
            salts[salt] = pair.index
    return problems


def table(pairs: list[Pair]) -> str:
    return "".join(
        f"{p.index:02d}\t{p.label}\t{p.slug}\t{p.run}\t{p.est_s}\n" for p in pairs
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="validate a pairs file for one job")
    p_check.add_argument("--pairs", type=Path, required=True)
    p_check.add_argument("--plan-dir", type=Path, required=True)
    p_check.add_argument("--inputs-root", type=Path, required=True)
    p_check.add_argument("--num-workers", type=int, required=True)
    p_check.add_argument("--table-out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        pairs = parse(args.pairs.read_text())
    except PairsError as exc:
        print(f"error: {args.pairs}: {exc}", file=sys.stderr)
        return 1
    problems = check(pairs, args.plan_dir, args.inputs_root, args.num_workers)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    if args.table_out is not None:
        args.table_out.write_text(table(pairs))
    print(
        json.dumps(
            {"pairs": len(pairs), "est_s": sum(p.est_s for p in pairs), "ok": True}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
