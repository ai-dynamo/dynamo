#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mirror docs/fern/main.css into CustomFooter.tsx's SITE_CSS block.

main.css is the canonical stylesheet, served via docs.yml `css:` (the
server-rendered / no-JS baseline). The footer's SITE_CSS <style> block is the
copy that survives the NVIDIA global theme, which replaces the project `css:`
stylesheet at publish (#11952). This script regenerates the block between the
`sync-site-css:begin/end` markers; `--check` exits 1 if the mirror is stale.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "main.css"
FOOTER = ROOT / "components" / "CustomFooter.tsx"
BLOCK = re.compile(
    r"(// sync-site-css:begin \(generated from \.\./main\.css\)\nconst SITE_CSS = `\n)"
    r".*?"
    r"(\n`;\n// sync-site-css:end)",
    re.DOTALL,
)


def main() -> int:
    check = "--check" in sys.argv[1:]
    css = CSS.read_text()
    for banned in ("`", "${"):
        if banned in css:
            print(
                f"sync_site_css: main.css contains {banned!r}, unsafe inside a "
                "template literal; escape or restructure it",
                file=sys.stderr,
            )
            return 1
    footer = FOOTER.read_text()
    if not BLOCK.search(footer):
        print(
            "sync_site_css: SITE_CSS markers not found in CustomFooter.tsx",
            file=sys.stderr,
        )
        return 1
    updated = BLOCK.sub(lambda m: m.group(1) + css.rstrip("\n") + m.group(2), footer)
    if updated == footer:
        print("sync_site_css: SITE_CSS is in sync")
        return 0
    if check:
        print(
            "sync_site_css: SITE_CSS is stale; run "
            "`python3 docs/fern/scripts/sync_site_css.py`",
            file=sys.stderr,
        )
        return 1
    FOOTER.write_text(updated)
    print("sync_site_css: SITE_CSS regenerated from main.css")
    return 0


if __name__ == "__main__":
    sys.exit(main())
