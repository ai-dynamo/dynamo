# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``lr-report``: Markdown tables, ``summary.json`` and figures from ``results.jsonl`` files.

Example::

    lr-report --results CR/runs/test/*/results.jsonl --out-dir CR/report/test \\
        --metric goodput_rps_window --cluster segment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from learned_routing.report import (
    GROUP_KEYS,
    MixedContentError,
    Report,
    load_records,
    write_report,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lr-report", description=__doc__.split("\n\n")[0])
    p.add_argument("--results", nargs="+", required=True, help="results.jsonl files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metric", default="goodput_rps_window")
    p.add_argument(
        "--reference",
        help="reference policy name or sha prefix (default: default@defaults if present)",
    )
    p.add_argument(
        "--cluster",
        choices=("cell", "segment"),
        default="cell",
        help="bootstrap cluster unit",
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-by", default=",".join(GROUP_KEYS))
    p.add_argument(
        "--policy-filter",
        help="substring; keep only matching policy names (plus the reference)",
    )
    p.add_argument("--split", help="keep only records of this split")
    p.add_argument(
        "--build-id",
        help="keep only records whose bindings build ID starts with this prefix",
    )
    p.add_argument(
        "--mixed-contents",
        choices=("error", "split"),
        default="error",
        help="a cell_id under several contents (cell SHA, build, protocol, harness version): "
        "fail (default) or report each content as its own cell",
    )
    p.add_argument("--title", default="Learned-routing results")
    p.add_argument("--no-figures", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = load_records(args.results)
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    if args.policy_filter:
        keep = args.policy_filter
        records = [
            r
            for r in records
            if keep in (r.get("policy_name") or "")
            or r.get("policy_name") == (args.reference or "default@defaults")
        ]
    if not records:
        print("lr-report: no records", file=sys.stderr)
        return 1
    try:
        report = Report(
            records,
            metric=args.metric,
            reference=args.reference,
            cluster=args.cluster,
            n_boot=args.bootstrap,
            seed=args.seed,
            group_keys=[k for k in args.group_by.split(",") if k],
            build_id=args.build_id,
            contents=args.mixed_contents,
        )
    except MixedContentError as exc:
        print(f"lr-report: {exc}", file=sys.stderr)
        return 2
    payload = write_report(
        report, Path(args.out_dir), title=args.title, make_figures=not args.no_figures
    )
    print(
        json.dumps(
            {
                "out_dir": args.out_dir,
                "records": payload["records"],
                "errors": payload["errors"],
                "figures": payload["figures"],
            }
        )
    )
    return 0


def main_exit() -> None:
    sys.exit(main())


if __name__ == "__main__":
    main_exit()
