# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for dynamo-attributions."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from enum import IntEnum
from pathlib import Path

from .extractor import extract_transitive
from .licenses import fetch_all_licenses
from .types import Ecosystem

logger = logging.getLogger(__name__)


class _ExitCodes(IntEnum):
    SUCCESS = 0
    FAILURE = 2
    BAD_INPUT = 3


def _resolve_ecosystem(value: str) -> Ecosystem | None:
    if value == "all":
        return None
    return Ecosystem(value)


def _check_prerequisites(dynamo_path: str) -> str | None:
    """Validate git and repo path. Returns error message or None."""
    if not shutil.which("git"):
        return "git is required but not found in PATH."
    p = Path(dynamo_path)
    if not (p / ".git").exists():
        return f"{p.resolve()} is not a git repository (no .git found)."
    return None


def _extract_python_from_image(
    image: str,
    dynamo_path: str,
) -> list[dict[str, str]]:
    """Extract Python packages from a container image using v1's extractor.

    Calls container/compliance/extractors/python_pkgs.extract_python() which
    mounts python_helper.py into the container and runs importlib.metadata
    to get package names, versions, and SPDX-normalized licenses.
    """
    repo_root = str(Path(dynamo_path).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from container.compliance.extractors.python_pkgs import extract_python
    except ImportError:
        logger.error(
            "Could not import container.compliance.extractors. "
            "Ensure --dynamo-path points to the dynamo repo root."
        )
        return []

    print(f"Extracting Python packages from {image} via importlib.metadata...")
    return extract_python(image)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="dynamo-attributions",
        description="Generate ATTRIBUTIONS-*.md from lock files and registry license APIs",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch/ref to read lock files from (default: main)",
    )
    parser.add_argument(
        "--ecosystem",
        choices=["rust", "python", "go", "all"],
        default="all",
        help="Ecosystem to generate attributions for (default: all)",
    )
    parser.add_argument(
        "--image",
        help="Container image to extract Python packages from (required for Python)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated attribution files (default: .)",
    )
    parser.add_argument(
        "--license-cache",
        help="Path to license cache JSON (default: ~/.dynamo_license_cache.json)",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub token for Go module license lookups (or set GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--dynamo-path",
        default=".",
        help="Path to dynamo repo (default: .)",
    )

    args = parser.parse_args()
    eco = _resolve_ecosystem(args.ecosystem)

    prereq_err = _check_prerequisites(args.dynamo_path)
    if prereq_err:
        print(f"Error: {prereq_err}", file=sys.stderr)
        return _ExitCodes.FAILURE

    # --- Python: requires --image ---
    python_from_image: list[dict[str, str]] | None = None
    wants_python = eco is None or eco == Ecosystem.PYTHON

    if args.image and wants_python:
        python_from_image = _extract_python_from_image(args.image, args.dynamo_path)
    elif eco == Ecosystem.PYTHON:
        print(
            "Error: --image is required for the Python ecosystem.\n"
            "  dynamo-attributions --ecosystem python --image IMAGE",
            file=sys.stderr,
        )
        return _ExitCodes.BAD_INPUT

    # --- Build transitive dep tree (Rust + Go only) ---
    tree = extract_transitive(
        dynamo_path=args.dynamo_path,
        branch=args.branch,
        ecosystem=eco,
    )
    packages = tree.all_packages()

    if not packages and not python_from_image:
        print("No packages found. Diagnostics:", file=sys.stderr)
        if eco in (None, Ecosystem.RUST):
            print(
                f"  Rust: could not read Cargo.lock from branch '{args.branch}'",
                file=sys.stderr,
            )
        if eco in (None, Ecosystem.GO):
            print(
                f"  Go: could not read deploy/operator/go.mod from branch '{args.branch}'",
                file=sys.stderr,
            )
        if eco in (None, Ecosystem.PYTHON) and not python_from_image:
            print("  Python: --image not provided", file=sys.stderr)
        print(
            f"\nVerify branch '{args.branch}' exists and --dynamo-path "
            f"'{args.dynamo_path}' is correct.",
            file=sys.stderr,
        )
        return _ExitCodes.BAD_INPUT

    # --- Fetch licenses for Rust/Go ---
    eco_packages: dict[str, list[dict]] = {"cargo": [], "pypi": [], "golang": []}

    if packages:
        github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
        cache_path = Path(args.license_cache) if args.license_cache else None

        print(f"Fetching licenses for {len(packages)} packages...")

        def _progress(current: int, total: int, name: str) -> None:
            if current % 25 == 0 or current == total or current == 1:
                print(f"  [{current}/{total}] {name}")

        license_results = fetch_all_licenses(
            packages,
            cache_path=cache_path,
            github_token=github_token,
            on_progress=_progress,
        )

        for lic in license_results:
            eco_packages[lic.ecosystem].append(lic.to_resolver_dict())

        errors = [r for r in license_results if r.error]
        if errors:
            print(
                f"\n{len(errors)} packages had license lookup errors:",
                file=sys.stderr,
            )
            for e in errors[:10]:
                print(
                    f"  {e.ecosystem}:{e.name}:{e.version} - {e.error}",
                    file=sys.stderr,
                )
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more", file=sys.stderr)

    # --- Add Python packages from --image (already have SPDX licenses) ---
    if python_from_image:
        print(
            f"Adding {len(python_from_image)} Python packages from container image..."
        )
        for r in python_from_image:
            eco_packages["pypi"].append(
                {
                    "name": r["package_name"],
                    "version": r["version"],
                    "ecosystem": "pypi",
                    "license_expression": r["spdx_license"],
                    "repository": "",
                    "homepage": "",
                }
            )

    # --- Render ---
    from .renderer import _generate_go, _generate_python, _generate_rust, _write_file

    output_dir = Path(args.output_dir)
    license_texts: dict[str, str] = {}
    wrote_any = False

    if eco_packages.get("cargo") and (eco is None or eco == Ecosystem.RUST):
        out = output_dir / "ATTRIBUTIONS-Rust.md"
        _write_file(out, _generate_rust(eco_packages["cargo"], license_texts))
        print(f"Wrote {len(eco_packages['cargo'])} Rust attributions to {out}")
        wrote_any = True

    if eco_packages.get("pypi") and (eco is None or eco == Ecosystem.PYTHON):
        out = output_dir / "ATTRIBUTIONS-Python.md"
        _write_file(out, _generate_python(eco_packages["pypi"], license_texts))
        print(f"Wrote {len(eco_packages['pypi'])} Python attributions to {out}")
        wrote_any = True

    if eco_packages.get("golang") and (eco is None or eco == Ecosystem.GO):
        out = output_dir / "ATTRIBUTIONS-Go.md"
        _write_file(out, _generate_go(eco_packages["golang"], license_texts))
        print(f"Wrote {len(eco_packages['golang'])} Go attributions to {out}")
        wrote_any = True

    if not wrote_any:
        print("No attribution files generated.")

    return _ExitCodes.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
