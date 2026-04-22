# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for dynamo-attributions."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from enum import IntEnum
from pathlib import Path

from .extractor import extract_transitive
from .licenses import fetch_all_licenses
from .renderer import _generate_go, _generate_python, _generate_rust, _write_file
from .types import Ecosystem

logger = logging.getLogger(__name__)


class _ExitCodes(IntEnum):
    SUCCESS = 0
    FAILURE = 2
    BAD_INPUT = 3


# Map Ecosystem -> (purl-key used in eco_packages dict, output filename)
_ECO_OUTPUT: dict[Ecosystem, tuple[str, str]] = {
    Ecosystem.RUST: ("cargo", "ATTRIBUTIONS-Rust.md"),
    Ecosystem.PYTHON: ("pypi", "ATTRIBUTIONS-Python.md"),
    Ecosystem.GO: ("golang", "ATTRIBUTIONS-Go.md"),
}


def _resolve_ecosystem(value: str) -> Ecosystem | None:
    """Coerce the --ecosystem CLI value to an Ecosystem (or None for 'all').

    Raises ValueError with a helpful message rather than the bare KeyError that
    Ecosystem(value) would produce, so argparse failures and library callers
    both get the same diagnostic.
    """
    if value == "all":
        return None
    try:
        return Ecosystem(value)
    except ValueError as exc:
        valid = ", ".join(sorted(e.value for e in Ecosystem))
        raise ValueError(f"unknown ecosystem '{value}' (valid: {valid}, all)") from exc


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


def _build_parser() -> argparse.ArgumentParser:
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
        choices=[*sorted(e.value for e in Ecosystem), "all"],
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
    return parser


def main() -> int:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        eco = _resolve_ecosystem(args.ecosystem)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return _ExitCodes.BAD_INPUT

    prereq_err = _check_prerequisites(args.dynamo_path)
    if prereq_err:
        print(f"Error: {prereq_err}", file=sys.stderr)
        return _ExitCodes.FAILURE

    # --- Python: requires --image ---
    python_from_image: list[dict[str, str]] | None = None
    image_provided = bool(args.image)
    wants_python = eco is None or eco == Ecosystem.PYTHON

    if image_provided and wants_python:
        python_from_image = _extract_python_from_image(args.image, args.dynamo_path)
        # Distinguish "image scanned, found nothing" from "no image given".
        # The former is a real signal — usually a mismounted helper or a
        # python-less image — and deserves an error exit, not a silent fallthrough.
        if not python_from_image:
            print(
                f"Error: Python package extraction returned no results from image '{args.image}'.\n"
                "  Verify the image is reachable and contains Python packages.\n"
                "  Ensure container/compliance/extractors/python_helper.py is accessible.",
                file=sys.stderr,
            )
            return _ExitCodes.BAD_INPUT
    elif eco == Ecosystem.PYTHON:
        print(
            "Error: --image is required for the Python ecosystem.\n"
            "  python3 container/compliance/generate_root_attributions.py --ecosystem python --image IMAGE",
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
                f"  Go: no go.mod files found on branch '{args.branch}'",
                file=sys.stderr,
            )
        if eco in (None, Ecosystem.PYTHON) and not python_from_image:
            if image_provided:
                print(
                    f"  Python: --image {args.image!r} returned 0 packages",
                    file=sys.stderr,
                )
            else:
                print("  Python: --image not provided", file=sys.stderr)
        print(
            f"\nVerify branch '{args.branch}' exists and --dynamo-path "
            f"'{args.dynamo_path}' is correct.",
            file=sys.stderr,
        )
        return _ExitCodes.BAD_INPUT

    # --- Fetch licenses for Rust/Go ---
    eco_packages: dict[str, list[dict]] = {key: [] for key, _ in _ECO_OUTPUT.values()}

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
            eco_packages[_ECO_OUTPUT[Ecosystem.PYTHON][0]].append(
                {
                    "name": r["package_name"],
                    "version": r["version"],
                    "ecosystem": _ECO_OUTPUT[Ecosystem.PYTHON][0],
                    "license_expression": r["spdx_license"],
                    "repository": "",
                    "homepage": "",
                }
            )

    # --- Render ---
    output_dir = Path(args.output_dir)
    license_texts: dict[str, str] = {}
    wrote_any = False

    _generators = {
        Ecosystem.RUST: _generate_rust,
        Ecosystem.PYTHON: _generate_python,
        Ecosystem.GO: _generate_go,
    }

    for ecosystem, (purl_key, filename) in _ECO_OUTPUT.items():
        if not eco_packages.get(purl_key):
            continue
        if eco is not None and eco != ecosystem:
            continue
        out = output_dir / filename
        _write_file(out, _generators[ecosystem](eco_packages[purl_key], license_texts))
        print(
            f"Wrote {len(eco_packages[purl_key])} {ecosystem.value} attributions to {out}"
        )
        wrote_any = True

    if not wrote_any:
        print("No attribution files generated.")

    return _ExitCodes.SUCCESS


if __name__ == "__main__":
    sys.exit(main())
