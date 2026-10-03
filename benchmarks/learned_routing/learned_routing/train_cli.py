# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``lr-train``: resumable CMA-ES over a policy space, scored by normalized replay goodput.

Example::

    lr-train --space space.yaml --cells CR/cells/train.jsonl --val CR/cells/val.jsonl \\
        --run-dir CR/runs/pilot/m1-s1 --budget-evals 400 --popsize 16 --seed 1 \\
        --max-wall-seconds 540

Call it again with the same arguments to continue: it resumes from ``DIR/checkpoint.pkl`` and
refuses to resume if the space, cells, reference, objective or seeds changed. Exit status: 0 done
(budget spent or CMA-ES stopped); 3 paused by ``--max-wall-seconds`` (call again); 1 error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from learned_routing.cells import Cell, load_cells
from learned_routing.paths import NUM_SLOTS, Layout
from learned_routing.policy import default_reference, load_specs
from learned_routing.space import Space
from learned_routing.train import (
    OBJECTIVES,
    TrainConfig,
    Trainer,
    TrainError,
    fake_quadratic_evaluator,
)

EXIT_PAUSED = 3
SLIM_FIELDS = (
    "cache_key",
    "policy_sha",
    "policy_name",
    "cell_id",
    "repeat",
    "cell_sha",
    "build_id",
    "replicate_protocol",
    "harness_version",
    "trace_sha256",
    "policy_seed",
    "cached",
    "error",
    "goodput_rps_window",
    "good_frac_window",
    "goodput_rps",
    "good_frac",
    "goodput_rps_report",
    "wall_s",
    "run_id",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lr-train", description=__doc__.split("\n\n")[0])
    p.add_argument("--root")
    p.add_argument("--space", required=True)
    p.add_argument("--cells", required=True, nargs="+", help="train cells JSONL")
    p.add_argument("--val", nargs="*", default=[], help="validation cells JSONL")
    p.add_argument("--cell-filter", help="regex on train cell_id")
    p.add_argument("--val-filter", help="regex on val cell_id")
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--budget-evals", type=int, required=True, help="CMA-ES candidate evaluations"
    )
    p.add_argument("--popsize", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-wall-seconds", type=float, required=True)
    p.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="CRN replicates per evaluation (A1: >= 2)",
    )
    p.add_argument("--replicate-pool", type=int, default=8)
    p.add_argument("--cells-per-gen", type=int)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-replicates", type=int, default=3)
    p.add_argument("--objective", choices=OBJECTIVES, default="ratio")
    p.add_argument("--metric", default="goodput_rps_window")
    p.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="smoothing added to both sides of each ratio",
    )
    p.add_argument(
        "--reference", default="default", help="reference spec file or builtin name"
    )
    p.add_argument(
        "--reeval-frac",
        type=float,
        default=0.0,
        help="UH-CMA-ES rank-change diagnostic",
    )
    p.add_argument("--slots", type=int, default=NUM_SLOTS)
    p.add_argument("--num-slots", type=int, default=NUM_SLOTS)
    p.add_argument("--slots-dir", help="slot pool directory (default ROOT/slots)")
    p.add_argument("--keep-per-request", action="store_true")
    p.add_argument(
        "--fake-objective",
        help="testing only: PATH=VALUE[,PATH=VALUE] quadratic objective, no replays",
    )
    return p


def _cells(paths, layout, pattern) -> list[Cell]:
    cells = [cell for path in paths for cell in load_cells(path, layout)]
    if pattern:
        regex = re.compile(pattern)
        cells = [cell for cell in cells if regex.search(cell.cell_id)]
    return cells


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    layout = Layout.resolve(args.root)
    space = Space.load(args.space)
    train_cells = _cells(args.cells, layout, args.cell_filter)
    val_cells = _cells(args.val, layout, args.val_filter)
    references = load_specs([args.reference])
    if len(references) != 1:
        print("lr-train: --reference must name exactly one spec", file=sys.stderr)
        return 1
    config = TrainConfig(
        budget_evals=args.budget_evals,
        popsize=args.popsize,
        seed=args.seed,
        replicates=args.replicates,
        replicate_pool=args.replicate_pool,
        cells_per_gen=args.cells_per_gen,
        val_every=args.val_every,
        val_replicates=args.val_replicates,
        objective=args.objective,
        metric=args.metric,
        eps=args.eps,
        max_wall_seconds=args.max_wall_seconds,
        reeval_frac=args.reeval_frac,
    )
    run_dir = Path(args.run_dir)
    evaluator = None
    if args.fake_objective:
        target = {}
        for item in args.fake_objective.split(","):
            path, value = item.split("=", 1)
            target[path.strip()] = float(value)
        evaluate = fake_quadratic_evaluator(target)
    else:
        from learned_routing.evaluate import Evaluator

        evaluator = Evaluator(
            layout,
            concurrency=min(args.slots, args.num_slots),
            num_slots=args.num_slots,
            keep_per_request=args.keep_per_request,
            results_path=run_dir / "results.jsonl",
            results_fields=SLIM_FIELDS,
            log_dir=run_dir / "logs",
            verbose=False,
            slots_dir=Path(args.slots_dir) if args.slots_dir else None,
        )

        def evaluate(tasks, deadline):
            return evaluator.evaluate(tasks, deadline=deadline)

    reference = references[0] if args.reference != "default" else default_reference()
    try:
        trainer = Trainer(
            space, train_cells, val_cells, run_dir, evaluate, reference, config
        )
        status = trainer.run()
    except TrainError as exc:
        print(json.dumps({"phase": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        if evaluator is not None:
            evaluator.close()
    print(json.dumps(status, sort_keys=True))
    return EXIT_PAUSED if status["phase"] == "paused" else 0


def main_exit() -> None:
    sys.exit(main())


if __name__ == "__main__":
    main_exit()
