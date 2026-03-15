#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate attribution CSV files for container images.

Extracts dpkg and Python package information from a container image either by:
- (default) running helper scripts inside the container via `docker run`, or
- (with --builder) using `docker buildx build` to mount the image filesystem
  without running the container (no Docker-in-Docker required).

Optionally computes a diff against a base image to show only added/changed packages.

Usage:
    python generate_attributions.py <image:tag> [--output out.csv] [--base-image base:tag]
    python generate_attributions.py <image:tag> --framework vllm --cuda-version 12.9
    python generate_attributions.py <image:tag> --builder my-builder --platform linux/amd64
"""

import argparse
import csv
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running as a script from any directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from extractors.dpkg import extract_dpkg  # noqa: E402
from extractors.python_pkgs import extract_python  # noqa: E402

log = logging.getLogger(__name__)

VALID_TYPES = {"dpkg", "python"}


def resolve_base_image(
    framework: str,
    target: str,
    cuda_version: str,
    context_yaml_path: Path,
) -> str:
    """Resolve the base image from context.yaml for a given framework/target/cuda combo."""
    try:
        import yaml
    except ImportError:
        log.error(
            "PyYAML is required for --framework/--cuda-version base image resolution. "
            "Install it with: pip install pyyaml"
        )
        sys.exit(1)

    if not context_yaml_path.is_file():
        log.error("context.yaml not found at %s", context_yaml_path)
        sys.exit(1)

    with open(context_yaml_path, "r") as f:
        context = yaml.safe_load(f)

    if target == "frontend":
        frontend_image = context.get("dynamo", {}).get("frontend_image")
        if not frontend_image:
            log.error("frontend_image not found in context.yaml dynamo section")
            sys.exit(1)
        return frontend_image

    # Runtime target: look up runtime_image and runtime_image_tag
    fw_config = context.get(framework, {})
    cuda_key = f"cuda{cuda_version}"
    cuda_config = fw_config.get(cuda_key, {})

    runtime_image = cuda_config.get("runtime_image")
    runtime_image_tag = cuda_config.get("runtime_image_tag")

    if not runtime_image or not runtime_image_tag:
        log.error(
            "Could not resolve base image for framework=%s cuda=%s target=%s. "
            "Keys runtime_image/runtime_image_tag not found under %s.%s in context.yaml",
            framework,
            cuda_version,
            target,
            framework,
            cuda_key,
        )
        sys.exit(1)

    return f"{runtime_image}:{runtime_image_tag}"


def compute_diff(
    target_packages: list[dict[str, str]],
    base_packages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Compute packages in target that are new or have different versions vs base.

    Returns packages present in target but not in base, or with a different version.
    """
    base_lookup = {}
    for pkg in base_packages:
        key = (pkg["package_name"], pkg["type"])
        base_lookup[key] = pkg["version"]

    diff = []
    for pkg in target_packages:
        key = (pkg["package_name"], pkg["type"])
        base_version = base_lookup.get(key)
        if base_version is None or base_version != pkg["version"]:
            diff.append(pkg)

    return diff


def write_csv(packages: list[dict[str, str]], output_path: str | None) -> None:
    """Write packages to CSV, sorted by (type, package_name)."""
    sorted_packages = sorted(packages, key=lambda p: (p["type"], p["package_name"]))
    fieldnames = ["package_name", "version", "type", "spdx_license"]

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_packages)
        log.info("Wrote %d entries to %s", len(sorted_packages), output_path)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_packages)


def extract_all(
    image: str,
    types: set[str],
    docker_cmd: str,
    verbose: bool,
) -> list[dict[str, str]]:
    """Run all requested extractors against an image."""
    packages = []

    if "dpkg" in types:
        log.info("Extracting dpkg packages from %s ...", image)
        packages.extend(extract_dpkg(image, docker_cmd=docker_cmd, verbose=verbose))

    if "python" in types:
        log.info("Extracting Python packages from %s ...", image)
        packages.extend(extract_python(image, docker_cmd=docker_cmd, verbose=verbose))

    return packages


_EXTRACT_DOCKERFILE = _SCRIPT_DIR / "Dockerfile.extract"


def _parse_tsv(tsv_path: Path, pkg_type: str) -> list[dict[str, str]]:
    """Parse a tab-separated extraction output file into package dicts."""
    packages = []
    if not tsv_path.is_file():
        return packages
    for line in tsv_path.read_text(errors="replace").strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        pkg_name, version, spdx_license = parts
        packages.append(
            {
                "package_name": pkg_name,
                "version": version,
                "type": pkg_type,
                "spdx_license": spdx_license,
            }
        )
    return packages


def extract_all_buildx(
    image: str,
    types: set[str],
    builder: str,
    platform: str,
    verbose: bool,
) -> list[dict[str, str]]:
    """Extract packages using docker buildx build — no docker run of target image.

    Mounts the target image filesystem via BuildKit's bind-from-stage mechanism,
    runs helper scripts against it, and exports TSV output to a local temp directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--builder",
            builder,
            "--platform",
            platform,
            "--build-arg",
            f"TARGET_IMAGE={image}",
            "--output",
            f"type=local,dest={tmpdir}",
            "--pull",  # always re-resolve target image digest from registry
            "-f",
            str(_EXTRACT_DOCKERFILE),
            str(_SCRIPT_DIR),
        ]
        if verbose:
            cmd.append("--progress=plain")
            log.info("Running: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            stderr = result.stderr if not verbose else ""
            raise RuntimeError(
                f"buildx extraction failed for {image} (exit {result.returncode}): {stderr}"
            )

        packages = []
        if "dpkg" in types:
            pkgs = _parse_tsv(Path(tmpdir) / "dpkg.tsv", "dpkg")
            if verbose:
                log.info("Extracted %d dpkg packages via buildx", len(pkgs))
            packages.extend(pkgs)
        if "python" in types:
            pkgs = _parse_tsv(Path(tmpdir) / "python.tsv", "python")
            if verbose:
                log.info("Extracted %d Python packages via buildx", len(pkgs))
            packages.extend(pkgs)

    return packages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate attribution CSV files for container images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my-registry/dynamo:vllm-runtime -o vllm.csv
  %(prog)s my-registry/dynamo:vllm-runtime --framework vllm --cuda-version 12.9 -o vllm.csv
  %(prog)s my-registry/dynamo:vllm-runtime --base-image nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04 -o vllm.csv
  %(prog)s my-registry/dynamo:frontend --framework dynamo --target frontend -o frontend.csv
        """,
    )
    parser.add_argument(
        "image", help="Container image to scan (e.g., my-registry/dynamo:latest)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV file path (default: stdout)",
    )
    parser.add_argument(
        "--base-image",
        help="Base image for diff calculation (explicit, overrides --framework auto-resolve)",
    )
    parser.add_argument(
        "--framework",
        choices=["vllm", "sglang", "trtllm", "dynamo"],
        help="Framework name for auto-resolving base image from context.yaml",
    )
    parser.add_argument(
        "--target",
        default="runtime",
        choices=["runtime", "frontend"],
        help="Build target for base image resolution (default: runtime)",
    )
    parser.add_argument(
        "--cuda-version",
        choices=["12.9", "13.0", "13.1"],
        help="CUDA version for base image resolution",
    )
    parser.add_argument(
        "--context-yaml",
        default=str(_REPO_ROOT / "container" / "context.yaml"),
        help="Path to context.yaml (default: container/context.yaml in repo root)",
    )
    parser.add_argument(
        "--types",
        default="dpkg,python",
        help="Comma-separated extraction types (default: dpkg,python)",
    )
    parser.add_argument(
        "--docker-cmd",
        default="docker",
        help="Docker command to use (default: docker)",
    )
    parser.add_argument(
        "--builder",
        default="",
        help=(
            "docker buildx builder name. When set, uses BuildKit filesystem extraction "
            "(no docker run) instead of docker run. Use --platform to target a specific "
            "architecture (defaults to linux/amd64)."
        ),
    )
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="Target platform for buildx extraction, e.g. linux/amd64 or linux/arm64 (default: linux/amd64)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    types = set(args.types.split(","))
    invalid = types - VALID_TYPES
    if invalid:
        log.error("Invalid types: %s (valid: %s)", invalid, VALID_TYPES)
        sys.exit(1)

    # Resolve base image if needed
    base_image = args.base_image
    if not base_image and args.framework:
        if args.target != "frontend" and not args.cuda_version:
            log.error(
                "--cuda-version is required when using --framework for runtime targets"
            )
            sys.exit(1)
        base_image = resolve_base_image(
            framework=args.framework,
            target=args.target,
            cuda_version=args.cuda_version or "",
            context_yaml_path=Path(args.context_yaml),
        )
        log.info("Auto-resolved base image: %s", base_image)

    # Extract from target image
    if args.builder:
        log.info(
            "Using BuildKit extraction (builder=%s, platform=%s)",
            args.builder,
            args.platform,
        )
        target_packages = extract_all_buildx(
            args.image, types, args.builder, args.platform, args.verbose
        )
    else:
        target_packages = extract_all(args.image, types, args.docker_cmd, args.verbose)
    log.info("Total packages extracted from target: %d", len(target_packages))

    # Write full CSV
    write_csv(target_packages, args.output)

    # Compute and write diff if base image is available
    if base_image:
        log.info("Extracting packages from base image for diff: %s", base_image)
        if args.builder:
            base_packages = extract_all_buildx(
                base_image, types, args.builder, args.platform, args.verbose
            )
        else:
            base_packages = extract_all(
                base_image, types, args.docker_cmd, args.verbose
            )
        log.info("Total packages extracted from base: %d", len(base_packages))

        diff_packages = compute_diff(target_packages, base_packages)
        log.info("Diff: %d new/changed packages", len(diff_packages))

        if args.output:
            # Insert _diff before the file extension
            output_path = Path(args.output)
            diff_path = str(output_path.with_stem(output_path.stem + "_diff"))
            write_csv(diff_packages, diff_path)
        else:
            # Write diff to stdout with a separator
            print("\n# --- DIFF (new/changed packages vs base) ---", file=sys.stderr)
            write_csv(diff_packages, None)


if __name__ == "__main__":
    main()
