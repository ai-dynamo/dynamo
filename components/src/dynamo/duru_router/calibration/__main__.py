# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan graph-aware calibration points or measure them through a serving adapter."""

import argparse
import importlib
import json
from pathlib import Path

from .measure import measure
from .plan import atomic_json, design


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--cap", type=int, required=True)
    plan.add_argument("--alignment", type=int, required=True)
    plan.add_argument("--max-prefill", type=int, default=8192)
    plan.add_argument("--capture-sizes", type=int, nargs="*", default=[])
    plan.add_argument(
        "--prefill-graphs",
        action="store_true",
        help="Resolved engine mode uses these graph sizes for prefill, not only decode",
    )
    plan.add_argument("--prefix-points", type=int, default=8)
    plan.add_argument("--decode-points", type=int, default=12)
    plan.add_argument("--repeats", type=int, default=10)
    plan.add_argument("--warmups", type=int, default=3)
    plan.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("measure")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument(
        "--adapter",
        required=True,
        help="module:factory returning a context-managed serving adapter",
    )
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        options = vars(args).copy()
        options.pop("command")
        output = options.pop("out")
        result = design(**options)
        atomic_json(output, result)
        print(
            f"{len(result['P'])} prefill lengths x {len(result['D'])} prefix depths + {len(result['K'])} decode contexts = {len(result['cases'])} points"
        )
    else:
        module, name = args.adapter.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        contract = read(args.contract)
        with factory(contract) as adapter:
            measure(
                args.out,
                contract,
                read(args.plan),
                adapter.submit,
                adapter.drain,
                adapter.ledger,
                adapter.reset,
                resume=args.resume,
                vocab_size=adapter.vocab_size,
            )


if __name__ == "__main__":
    main()
