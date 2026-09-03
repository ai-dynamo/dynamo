#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adjust source-only relative paths after pages/ is copied into a snapshot root."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CODE_EXAMPLE_SRC = re.compile(r"""(src=["'])(?:\.\./)((?:\.\./)+examples/)""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    changed = 0
    for path in args.snapshot.rglob("*.mdx"):
        text = path.read_text(encoding="utf-8")
        rewritten = CODE_EXAMPLE_SRC.sub(r"\1\2", text)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1
    print(f"rewrite_snapshot_paths: updated {changed} file(s) under {args.snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
