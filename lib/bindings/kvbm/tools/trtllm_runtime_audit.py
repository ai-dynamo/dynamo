#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import glob
import json
import os
from importlib import metadata
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Optional


DEFAULT_PINNED_CHECKOUT = Path("/tmp/trtllm-latest/tensorrt_llm")
DEFAULT_LIBRARY_DIR_PATTERNS = (
    "/usr/local/cuda*/targets/*/lib",
    "/usr/local/cuda*/targets/*/lib/stubs",
    "/usr/local/cuda*/lib64",
    "/usr/local/cuda*/lib",
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/opt/*/lib",
    "/opt/*/lib64",
)


def _parse_expected_cuda_major(requirements: Iterable[str]) -> Optional[int]:
    for requirement in requirements:
        match = re.search(r"nvidia-nccl-cu(\d+)", requirement)
        if match is not None:
            return int(match.group(1))
    for requirement in requirements:
        match = re.search(r"cuda-python\s*>=\s*(\d+)", requirement)
        if match is not None:
            return int(match.group(1))
    return None


def _package_surface(package_root: Path) -> dict[str, Any]:
    return {
        "package_root": str(package_root),
        "has_pyexecutor": (package_root / "_torch" / "pyexecutor").is_dir(),
        "has_disaggregation": (package_root / "_torch" / "disaggregation").is_dir(),
    }


def _resolve_distribution(name: str) -> Any:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _distribution_summary(distribution: Any) -> dict[str, Any]:
    if distribution is None:
        return {
            "installed": False,
            "version": None,
            "expected_cuda_major": None,
            "package_root": None,
            "has_pyexecutor": False,
            "has_disaggregation": False,
            "requirements": [],
        }

    requirements = list(distribution.requires or [])
    package_root = Path(distribution.locate_file("tensorrt_llm"))
    summary = _package_surface(package_root)
    summary.update(
        {
            "installed": True,
            "version": distribution.version,
            "expected_cuda_major": _parse_expected_cuda_major(requirements),
            "requirements": requirements,
        }
    )
    return summary


def _expand_library_dirs(patterns: Iterable[str]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if not path.is_dir() or path in seen:
                continue
            seen.add(path)
            dirs.append(path)
    return dirs


def _extract_soname_major(path: Path) -> Optional[int]:
    match = re.search(r"\.so\.(\d+)(?:$|\.)", path.name)
    if match is not None:
        return int(match.group(1))
    real_path = Path(os.path.realpath(path))
    match = re.search(r"\.so\.(\d+)(?:$|\.)", real_path.name)
    if match is not None:
        return int(match.group(1))
    return None


def _library_summary(library_dirs: Iterable[Path]) -> dict[str, Any]:
    entries = []
    majors = set()
    for directory in library_dirs:
        for path in sorted(directory.glob("libcublasLt.so*")):
            major = _extract_soname_major(path)
            if major is not None:
                majors.add(major)
            entries.append(
                {
                    "path": str(path),
                    "resolved_path": os.path.realpath(path),
                    "major": major,
                }
            )
    return {
        "search_dirs": [str(path) for path in library_dirs],
        "libcublaslt": entries,
        "available_majors": sorted(majors),
    }


def build_runtime_report(
    *,
    distribution: Any = None,
    pinned_checkout: Path = DEFAULT_PINNED_CHECKOUT,
    library_dirs: Optional[Iterable[Path]] = None,
) -> dict[str, Any]:
    distribution = _resolve_distribution("tensorrt_llm") if distribution is None else distribution
    installed = _distribution_summary(distribution)
    checkout = {
        "path": str(pinned_checkout),
        "exists": pinned_checkout.is_dir(),
        "has_pyexecutor": False,
        "has_disaggregation": False,
    }
    if pinned_checkout.is_dir():
        checkout.update(_package_surface(pinned_checkout))

    resolved_library_dirs = (
        list(library_dirs)
        if library_dirs is not None
        else _expand_library_dirs(DEFAULT_LIBRARY_DIR_PATTERNS)
    )
    libraries = _library_summary(resolved_library_dirs)

    findings = []
    if installed["installed"] and checkout["has_disaggregation"] and not installed["has_disaggregation"]:
        findings.append(
            "installed tensorrt_llm package does not expose _torch.disaggregation, "
            "but the pinned checkout does"
        )

    expected_cuda_major = installed["expected_cuda_major"]
    if expected_cuda_major is not None and expected_cuda_major not in libraries["available_majors"]:
        findings.append(
            "installed tensorrt_llm package expects CUDA major "
            f"{expected_cuda_major}, but libcublasLt majors "
            f"{libraries['available_majors'] or '[]'} were found"
        )

    return {
        "python_executable": sys.executable,
        "installed_tensorrt_llm": installed,
        "pinned_checkout": checkout,
        "libraries": libraries,
        "findings": findings,
        "status": "blocked" if findings else "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the installed TRT-LLM runtime without importing tensorrt_llm. "
            "This helps detect package-surface and CUDA-library mismatches before "
            "running the KVBM TRT-LLM smoke path."
        )
    )
    parser.add_argument(
        "--pinned-checkout",
        type=Path,
        default=DEFAULT_PINNED_CHECKOUT,
        help="Pinned TRT-LLM checkout package root",
    )
    parser.add_argument(
        "--library-dir",
        action="append",
        type=Path,
        help="Additional library directory to scan for libcublasLt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON",
    )
    args = parser.parse_args()

    report = build_runtime_report(
        pinned_checkout=args.pinned_checkout,
        library_dirs=args.library_dir,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"python: {report['python_executable']}")
        installed = report["installed_tensorrt_llm"]
        print(
            "installed tensorrt_llm: "
            f"{installed['version'] or 'missing'} "
            f"(disaggregation={installed['has_disaggregation']}, "
            f"pyexecutor={installed['has_pyexecutor']}, "
            f"expected_cuda_major={installed['expected_cuda_major']})"
        )
        checkout = report["pinned_checkout"]
        print(
            "pinned checkout: "
            f"{checkout['path']} "
            f"(exists={checkout['exists']}, "
            f"disaggregation={checkout['has_disaggregation']}, "
            f"pyexecutor={checkout['has_pyexecutor']})"
        )
        available = report["libraries"]["available_majors"]
        print(f"libcublasLt majors: {available or '[]'}")
        if report["findings"]:
            print("findings:")
            for finding in report["findings"]:
                print(f"- {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
