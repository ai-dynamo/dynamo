# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestrator for per-ecosystem NOTICES generation.

Invoked from each container's `licenses` Dockerfile stage:

    python3 -m compliance.generators \\
        --ecosystem rust,python,dpkg,go,native \\
        --venv /opt/dynamo/venv \\
        --output-dir /legal \\
        --go-sbom /tmp/sbom-go.cdx.json \\
        --native-yaml /opt/compliance/native_packages.yaml \\
        --native-image dynamo-runtime

Per-ecosystem flags are only consulted when that ecosystem is requested.
Unknown ecosystem names error out (typo-safe).

Exit codes:
  0  every requested generator ran cleanly (whether or not it found any
     components — empty NOTICES files are valid for an image that doesn't
     ship that ecosystem)
  1  one or more generators raised
  2  argument validation error (handled by argparse)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import common

logger = logging.getLogger("compliance.generators")

_ALL_ECOSYSTEMS = ("rust", "python", "dpkg", "go", "native")


def _parse_ecosystems(raw: str) -> list[str]:
    if raw == "all":
        return list(_ALL_ECOSYSTEMS)
    parsed = [e.strip() for e in raw.split(",") if e.strip()]
    bad = [e for e in parsed if e not in _ALL_ECOSYSTEMS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown ecosystem(s): {bad}. Valid: {list(_ALL_ECOSYSTEMS)}"
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compliance.generators",
        description="Generate per-ecosystem NOTICES-*.txt + *-deps.csv into /legal/",
    )
    parser.add_argument(
        "--ecosystem",
        type=_parse_ecosystems,
        default=list(_ALL_ECOSYSTEMS),
        help='Comma-separated list (or "all"). Default: all five.',
    )
    parser.add_argument(
        "--venv",
        type=Path,
        help=(
            "Runtime venv root for python/rust generators. The generators glob "
            "<venv>/lib/python*/site-packages/. Use this when the runtime image "
            "uses a uv/python venv (e.g. dynamo-runtime style)."
        ),
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to a site-packages directory. Repeatable. Use instead of (or "
            "in addition to) --venv when the runtime image uses pip "
            "--break-system-packages into the system Python and there's no venv "
            "(e.g. lmsysorg/sglang upstream base)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where /legal/<ecosystem>/ subdirs are written",
    )
    parser.add_argument(
        "--go-sbom",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to a cyclonedx-gomod output (required for go). Repeatable when "
            "an image combines multiple Go binaries — e.g. frontend pulls in /epp "
            "from the EPP image and consumes EPP's emitted SBOM alongside any "
            "Go SBOM produced inside the frontend build itself."
        ),
    )
    parser.add_argument(
        "--native-yaml",
        type=Path,
        help="Path to native_packages.yaml (required for native)",
    )
    parser.add_argument(
        "--native-image",
        help="Image name to filter native_packages entries (required for native)",
    )
    parser.add_argument(
        "--subtract-sbom",
        type=Path,
        default=None,
        help=(
            "Path to a baseline CycloneDX SBOM (typically a file from "
            "container/compliance/base_sboms/). Components matching its "
            "(name, version) set are filtered from generator output so "
            "NOTICES attributes only what's installed above the baseline. "
            "Pass this from the runtime's licenses stage; omit it for "
            "from-image == baseline cases (no subtraction needed)."
        ),
    )
    parser.add_argument(
        "--rust-licenses-dir",
        type=Path,
        default=None,
        help=(
            "Directory of real crate LICENSE files harvested from the cargo "
            "registry in wheel_builder (subdir per '<name>-<version>'). When "
            "present, the rust generator prefers these over canonical SPDX text."
        ),
    )
    parser.add_argument(
        "--go-licenses-dir",
        type=Path,
        default=None,
        help=(
            "Directory of real module LICENSE files harvested from the go "
            "module cache in the go-builder (mirrors the escaped module path "
            "'<path>@<version>'). When present, the go generator prefers these "
            "over canonical SPDX text."
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=(
            "Path to policy/licenses.toml. When given, the unified osrb-deps.csv "
            "Notes column is populated with the matching [[exceptions]] `reason` "
            "(the justification for allowing a non-permissive package)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    # Build the search-path list shared by python + rust. Both --venv and
    # --site-packages can be supplied (additive); at least one is required.
    search_paths: list[Path] = []
    if args.venv is not None:
        search_paths.append(args.venv)
    search_paths.extend(args.site_packages)

    # Load the baseline subtraction set once, share across ecosystems.
    subtract: set[tuple[str, str]] | None = None
    if args.subtract_sbom is not None:
        subtract = common.load_subtract_keys(args.subtract_sbom)

    # Each generator writes its NOTICES-<Eco>.txt FLAT at --output-dir (no
    # per-ecosystem subdir) and returns its (already baseline-subtracted)
    # components, which we accumulate for the unified CSV + CycloneDX below.
    failures: list[str] = []
    all_components: list = []
    for eco in args.ecosystem:
        comps = None
        try:
            if eco == "rust":
                if not search_paths:
                    failures.append(
                        "rust: at least one of --venv or --site-packages is required"
                    )
                    continue
                from . import rust as gen

                comps = gen.generate(
                    search_paths,
                    args.output_dir,
                    subtract=subtract,
                    licenses_dir=args.rust_licenses_dir,
                )
            elif eco == "python":
                if not search_paths:
                    failures.append(
                        "python: at least one of --venv or --site-packages is required"
                    )
                    continue
                from . import python as gen  # type: ignore[no-redef]

                comps = gen.generate(search_paths, args.output_dir, subtract=subtract)
            elif eco == "dpkg":
                from . import dpkg as gen  # type: ignore[no-redef]

                comps = gen.generate(args.output_dir, subtract=subtract)
            elif eco == "go":
                if not args.go_sbom:
                    failures.append("go: at least one --go-sbom is required")
                    continue
                from . import go as gen  # type: ignore[no-redef]

                comps = gen.generate(
                    args.go_sbom,
                    args.output_dir,
                    subtract=subtract,
                    licenses_dir=args.go_licenses_dir,
                )
            elif eco == "native":
                if args.native_yaml is None or args.native_image is None:
                    failures.append(
                        "native: --native-yaml and --native-image are required"
                    )
                    continue
                from . import native as gen  # type: ignore[no-redef]

                comps = gen.generate(
                    args.native_yaml,
                    args.output_dir,
                    image_filter=args.native_image,
                    subtract=subtract,
                )
            else:  # pragma: no cover - guarded by argparse type
                failures.append(f"unknown ecosystem {eco}")
                continue
        except NotImplementedError as exc:
            logger.warning("%s generator not yet implemented: %s", eco, exc)
            # Stub generators are deliberate; don't fail the build until
            # the implementation lands. The runtime image still gets the
            # ecosystems that ARE implemented (rust today).
        except Exception as exc:
            logger.exception("Generator for %s raised: %s", eco, exc)
            failures.append(f"{eco}: {exc}")
        else:
            if comps:
                all_components.extend(comps)

    # Fail fast on any generator failure BEFORE writing merged outputs, so a
    # partial osrb-deps.csv / osrb.cdx.json is never produced from an incomplete
    # component set (the full traceback was already logged above).
    if failures:
        logger.error("Generator failures: %s", failures)
        return 1

    # Unified OSRB outputs (delta-only — the accumulated components are already
    # baseline-subtracted): one merged CSV with a Notes column + one CycloneDX
    # SBOM. The merged CSV supersedes the per-ecosystem <eco>-deps.csv the
    # generators wrote flat, so drop those (validation runs on the merged CSV).
    for eco in _ALL_ECOSYSTEMS:
        (args.output_dir / f"{eco}-deps.csv").unlink(missing_ok=True)

    exception_reasons: dict[tuple[str, str], str] = {}
    if args.policy is not None:
        try:
            from ..policy.validate import load_policy

            policy = load_policy(args.policy)
            for exc in policy.exceptions:
                etype, ename = exc.get("type"), exc.get("name")
                if etype and ename:
                    exception_reasons.setdefault(
                        (etype, ename), exc.get("reason") or ""
                    )
        except Exception as exc:  # don't fail generation over the Notes column
            logger.warning("could not load policy for Notes column: %s", exc)

    common.write_merged_csv(
        all_components, exception_reasons, args.output_dir / "osrb-deps.csv"
    )
    common.write_cyclonedx(all_components, args.output_dir / "osrb.cdx.json")
    logger.info(
        "Wrote unified osrb-deps.csv + osrb.cdx.json (%d components) to %s",
        len(all_components),
        args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
