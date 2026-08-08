#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the release-pinned Dynamo Rust API reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rust_api_discovery import RustReference
from rust_api_discovery import discover_rust_reference as _discover_rust_reference
from rust_api_rendering import render_page

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_FERN_ROOT = SCRIPT_DIR.parent
DEFAULT_RELEASES_DATA = DEFAULT_FERN_ROOT / "components" / "releases.data.ts"


def _outputs(fern_root: Path, reference: RustReference) -> dict[Path, str]:
    """Build every generated Rust output in one pass."""
    return {
        fern_root
        / "pages"
        / "reference"
        / "api"
        / "rust"
        / "README.mdx": render_page(reference),
    }


def _apply_outputs(outputs: dict[Path, str], *, check: bool) -> int:
    """Write outputs or report stale files in check mode."""
    stale = []
    for path, rendered in outputs.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == rendered:
            print(f"{path.name}: unchanged")
            continue
        stale.append(path.name)
        if check:
            print(f"{path.name}: STALE (regeneration would change it)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"{path.name}: wrote {len(rendered.encode('utf-8'))} bytes")
    if check and stale:
        print(f"check failed: {len(stale)} Rust output(s) stale", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Generate Rust API outputs or verify that they are current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fern-root", type=Path, default=DEFAULT_FERN_ROOT)
    args = parser.parse_args(argv)
    reference = discover_rust_reference()
    return _apply_outputs(_outputs(args.fern_root, reference), check=args.check)


def discover_rust_reference() -> RustReference:
    """Discover against the repository and canonical release data."""
    return _discover_rust_reference(REPO_ROOT, DEFAULT_RELEASES_DATA)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
