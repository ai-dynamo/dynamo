#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the Dynamo Python API reference source-of-truth data.

Thin CLI/orchestrator that composes :mod:`api_discovery` (griffe-driven
static discovery of the eleven curated Dynamo Python packages) with
:mod:`api_rendering` (deterministic TypeScript + MDX serialization) to
emit three outputs from one parse:

  * ``docs/fern/components/api-reference.data.ts`` -- a typed TypeScript
    data module the React ``ApiPythonIndex`` and ``ApiSurfaceBrowser``
    components read from. Includes every class's public methods so the
    compact-grouped page layout can expand into per-symbol detail.
  * ``docs/fern/reference/api/python/README.mdx`` -- the Python language
    landing page, a compact grouped index of every curated module.
  * ``docs/fern/reference/api/python/<slug>.mdx`` -- one page per curated
    module. The MDX file is generated end-to-end (no manual stubs); the
    generated span contains an ``<llms-only>`` Markdown fallback so agent
    exports still see the full symbol table when the React components are
    stripped.

Usage (from any cwd; paths resolve relative to this file)::

    python3 gen_python_api.py            # write / refresh every output
    python3 gen_python_api.py --check    # exit 1 if any output is stale

Isolated invocation (bypasses the repo's Python resolution, which is
unrelated to this generator)::

    uv run --no-project --python 3.13 --with 'griffe==2.1.0' \\
        python3 docs/fern/scripts/gen_python_api.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from api_discovery import Module, discover_all_modules
from api_rendering import render_landing_page, render_module_page, render_ts_data

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FERN_ROOT = SCRIPT_DIR.parent


def _module_page_path(fern_root: Path, module: Module) -> Path:
    return fern_root / "reference" / "api" / "python" / f"{module.slug}.mdx"


def _landing_page_path(fern_root: Path) -> Path:
    return fern_root / "reference" / "api" / "python" / "README.mdx"


def _data_ts_path(fern_root: Path) -> Path:
    return fern_root / "components" / "api-reference.data.ts"


def _rendered_outputs(fern_root: Path, modules: list[Module]) -> dict[Path, str]:
    """Compute every output path -> new text mapping in one deterministic pass."""
    outputs: dict[Path, str] = {
        _data_ts_path(fern_root): render_ts_data(modules),
        _landing_page_path(fern_root): render_landing_page(modules),
    }
    for module in modules:
        outputs[_module_page_path(fern_root, module)] = render_module_page(module)
    return outputs


def _apply_outputs(outputs: dict[Path, str], *, check: bool) -> int:
    """Write outputs (or diff them in ``--check`` mode) and report drift."""
    stale: list[str] = []
    for path, new_text in outputs.items():
        old_text = path.read_text(encoding="utf-8") if path.is_file() else None
        if new_text == old_text:
            print(f"{path.name}: unchanged")
            continue
        stale.append(path.name)
        if check:
            print(f"{path.name}: STALE (regeneration would change it)")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text, encoding="utf-8")
            print(f"{path.name}: wrote {len(new_text.encode('utf-8'))} bytes")
    if check and stale:
        print(
            f"check failed: {len(stale)} output(s) stale -- run gen_python_api.py",
            file=sys.stderr,
        )
        return 1
    return 0


def _orphaned_module_pages(fern_root: Path, outputs: dict[Path, str]) -> list[Path]:
    """Generated module pages on disk that no current module owns."""
    page_dir = _landing_page_path(fern_root).parent
    if not page_dir.is_dir():
        return []
    expected = set(outputs)
    return sorted(path for path in page_dir.glob("*.mdx") if path not in expected)


def _apply_orphans(orphans: list[Path], *, check: bool) -> int:
    """Report orphaned pages in check mode or delete them in write mode."""
    for path in orphans:
        if check:
            print(f"{path.name}: STALE (orphaned generated page)")
        else:
            path.unlink()
            print(f"{path.name}: removed orphaned generated page")
    return 1 if check and orphans else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; see the module docstring for the two modes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if regeneration would change any output; write nothing",
    )
    parser.add_argument(
        "--fern-root",
        type=Path,
        default=DEFAULT_FERN_ROOT,
        help="docs/fern root (override for hermetic tests; defaults to sibling)",
    )
    args = parser.parse_args(argv)
    modules = discover_all_modules()
    outputs = _rendered_outputs(args.fern_root, modules)
    output_status = _apply_outputs(outputs, check=args.check)
    orphan_status = _apply_orphans(
        _orphaned_module_pages(args.fern_root, outputs),
        check=args.check,
    )
    return max(output_status, orphan_status)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
