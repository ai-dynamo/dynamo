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
        help="Runtime venv root (required for python/rust)",
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
        help="Path to cyclonedx-gomod output (required for go)",
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
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    failures: list[str] = []
    for eco in args.ecosystem:
        eco_out = args.output_dir / eco
        try:
            if eco == "rust":
                if args.venv is None:
                    failures.append("rust: --venv is required")
                    continue
                from . import rust as gen
                gen.generate(args.venv, eco_out)
            elif eco == "python":
                if args.venv is None:
                    failures.append("python: --venv is required")
                    continue
                from . import python as gen  # type: ignore[no-redef]
                gen.generate(args.venv, eco_out)
            elif eco == "dpkg":
                from . import dpkg as gen  # type: ignore[no-redef]
                gen.generate(eco_out)
            elif eco == "go":
                if args.go_sbom is None:
                    failures.append("go: --go-sbom is required")
                    continue
                from . import go as gen  # type: ignore[no-redef]
                gen.generate(args.go_sbom, eco_out)
            elif eco == "native":
                if args.native_yaml is None or args.native_image is None:
                    failures.append("native: --native-yaml and --native-image are required")
                    continue
                from . import native as gen  # type: ignore[no-redef]
                gen.generate(args.native_yaml, eco_out, image_filter=args.native_image)
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

    if failures:
        logger.error("Generator failures: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
