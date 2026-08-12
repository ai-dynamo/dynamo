#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the Dynamo Python API reference source-of-truth data.

Thin CLI/orchestrator that composes :mod:`api_discovery` (griffe-driven
static discovery of the eleven curated Dynamo Python packages) with
:mod:`api_rendering` (deterministic MDX serialization) to emit two kinds of
output from one parse:

  * ``docs/fern/pages/reference/api/python/README.mdx`` -- the Python
    language landing page, a ``<CardGroup>`` indexing every curated module.
  * ``docs/fern/pages/reference/api/python/<slug>.mdx`` -- one page per
    curated module, generated end-to-end (no manual stubs), with an anchored
    ``<Accordion>`` per symbol carrying its import statement, signature,
    and public methods.

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

import api_freshness
from api_discovery import SEARCH_PATH_PARTS, Module, discover_all_modules
from api_rendering import render_landing_page, render_module_page

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FERN_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def _module_page_path(fern_root: Path, module: Module) -> Path:
    return fern_root / "pages" / "reference" / "api" / "python" / f"{module.slug}.mdx"


def _landing_page_path(fern_root: Path) -> Path:
    return fern_root / "pages" / "reference" / "api" / "python" / "README.mdx"


def _rendered_outputs(fern_root: Path, modules: list[Module]) -> dict[Path, str]:
    """Compute every output path -> new text mapping in one deterministic pass."""
    outputs: dict[Path, str] = {
        _landing_page_path(fern_root): render_landing_page(modules),
    }
    for module in modules:
        outputs[_module_page_path(fern_root, module)] = render_module_page(module)
    return outputs


def _apply_outputs(
    outputs: dict[Path, str], *, check: bool, blame_branch: bool = True
) -> int:
    """Write outputs (or diff them in ``--check`` mode) and report drift.

    ``blame_branch`` is False when ``--since`` showed this branch touched
    nothing feeding the generator, which means main moved underneath it. The
    drift is still printed; it just does not fail a check the branch cannot fix.
    """
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
        if not blame_branch:
            print(
                f"{len(stale)} output(s) stale, but this branch changed no API "
                f"source, generated page, or generator script. The drift came "
                f"from main and regenerating here would only hold until the "
                f"next unrelated merge. Not failing.",
                file=sys.stderr,
            )
            return 0
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
    parser.add_argument(
        "--since",
        metavar="REF",
        help=(
            "with --check, only fail when this branch changed something feeding "
            "the generator; drift inherited from REF is reported, not failed"
        ),
    )
    args = parser.parse_args(argv)
    modules = discover_all_modules()
    outputs = _rendered_outputs(args.fern_root, modules)
    blame_branch = True
    if args.check and args.since:
        blame_branch = api_freshness.blames_branch(
            args.since,
            repo_root=REPO_ROOT,
            sources=tuple("/".join(parts) for parts in SEARCH_PATH_PARTS),
            outputs=outputs,
            script_dir=SCRIPT_DIR,
        )
    output_status = _apply_outputs(outputs, check=args.check, blame_branch=blame_branch)
    orphan_status = _apply_orphans(
        _orphaned_module_pages(args.fern_root, outputs),
        check=args.check,
    )
    return max(output_status, orphan_status)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
