#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate parity tables for parser stages from one common entrypoint.

Examples:
    python3 tests/parity/generate_parity_table.py parser --html > tests/parity/parser/PARITY.html
    python3 tests/parity/generate_parity_table.py parser --mode stream > tests/parity/parser/PARITY.stream.md
    python3 tests/parity/generate_parity_table.py reasoning --html > tests/parity/reasoning/PARITY.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dynamo parity tables.",
    )
    parser.add_argument(
        "stage",
        choices=("parser", "reasoning"),
        help="Parity stage to render.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.stage == "parser":
        from tests.parity.parser import table
    else:
        from tests.parity.reasoning import table

    table.main(rest)


if __name__ == "__main__":
    main()
