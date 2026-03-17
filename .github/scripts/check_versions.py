#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dynamo Version Consistency Checker.

Validates that backend versions (vLLM, SGLang, TRT-LLM, NIXL) are consistent
across all configuration files. Run in pre-merge CI to catch version drift.

Usage:
    python3 .github/scripts/check_versions.py
    python3 .github/scripts/check_versions.py --repo-root /path/to/dynamo

Exit codes:
    0 - All versions consistent
    1 - Version inconsistencies found
    2 - Configuration error (missing files, parse errors)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(2)


def normalize_version(ver: str) -> str:
    """Normalize version string to a comparable format.

    Handles various formats:
    - v0.17.1 -> 0.17.1 (strip v prefix)
    - tensorrt-llm==1.3.0rc7 -> 1.3.0rc7 (extract from pip spec)
    - nixl[cu12]<=0.10.1 -> 0.10.1 (extract from pip spec with extras)
    - vllm[flashinfer,runai]==0.17.1 -> 0.17.1 (extract from pip spec with extras)
    - v0.5.9-runtime -> 0.5.9 (extract from SGLang runtime tag)
    - v0.5.9-cu130-runtime -> 0.5.9 (extract from SGLang CUDA runtime tag)
    """
    ver = ver.strip()

    # Handle SGLang runtime image tags: v0.5.9-runtime or v0.5.9-cu130-runtime
    if "-runtime" in ver:
        m = re.match(r"v?(\d+\.\d+\.\d+)", ver)
        if m:
            return m.group(1)

    # Handle pip specs: package==version, package<=version, package[extras]==version
    pip_match = re.search(r"[<>=]+(\d+\.\d+\.\d+(?:rc\d+)?(?:\.post\d+)?)", ver)
    if pip_match:
        return pip_match.group(1)

    # Strip leading 'v' prefix
    if ver.startswith("v"):
        return ver[1:]

    return ver


def parse_context_yaml(repo: Path) -> dict[str, str]:
    """Parse container/context.yaml and extract backend versions.

    Returns dict with keys: vllm, sglang, trtllm, nixl
    Values are normalized version strings.
    """
    ctx_path = repo / "container" / "context.yaml"
    if not ctx_path.exists():
        return {}

    with open(ctx_path) as f:
        data = yaml.safe_load(f)

    versions: dict[str, str] = {}

    # NIXL from dynamo.nixl_ref
    if "dynamo" in data and "nixl_ref" in data["dynamo"]:
        versions["nixl"] = normalize_version(data["dynamo"]["nixl_ref"])

    # vLLM from vllm.cuda12.9.vllm_ref (prefer CUDA 12.9 as canonical)
    if "vllm" in data:
        for cuda_key in ["cuda12.9", "cuda13.0"]:
            if cuda_key in data["vllm"] and "vllm_ref" in data["vllm"][cuda_key]:
                versions["vllm"] = normalize_version(data["vllm"][cuda_key]["vllm_ref"])
                break

    # SGLang from sglang.cuda12.9.runtime_image_tag
    if "sglang" in data:
        for cuda_key in ["cuda12.9", "cuda13.0"]:
            if (
                cuda_key in data["sglang"]
                and "runtime_image_tag" in data["sglang"][cuda_key]
            ):
                versions["sglang"] = normalize_version(
                    data["sglang"][cuda_key]["runtime_image_tag"]
                )
                break

    # TRT-LLM from trtllm.pip_wheel
    if "trtllm" in data and "pip_wheel" in data["trtllm"]:
        versions["trtllm"] = normalize_version(data["trtllm"]["pip_wheel"])

    return versions


def _extract_version_from_file(filepath: Path, pattern: str) -> str | None:
    """Extract version from file using regex pattern."""
    if not filepath.exists():
        return None
    content = filepath.read_text()
    m = re.search(pattern, content)
    return normalize_version(m.group(1)) if m else None


def check_vllm_consistency(repo: Path, expected_ver: str) -> list[str]:
    """Check vLLM version consistency across files."""
    errors: list[str] = []

    # Check pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        ver = _extract_version_from_file(pyproject, r"vllm\[[^\]]*\]==([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(f"vLLM: pyproject.toml has {ver}, expected {expected_ver}")

    # Check install_vllm.sh
    install_script = repo / "container" / "deps" / "vllm" / "install_vllm.sh"
    if install_script.exists():
        ver = _extract_version_from_file(install_script, r"VLLM_VER=([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(
                f"vLLM: install_vllm.sh has VLLM_VER={ver}, expected {expected_ver}"
            )

    return errors


def check_sglang_consistency(repo: Path, expected_ver: str) -> list[str]:
    """Check SGLang version consistency across files."""
    errors: list[str] = []

    # Check pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        ver = _extract_version_from_file(pyproject, r"sglang\[[^\]]*\]==([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(f"SGLang: pyproject.toml has {ver}, expected {expected_ver}")

    return errors


def check_trtllm_consistency(repo: Path, expected_ver: str) -> list[str]:
    """Check TensorRT-LLM version consistency across files."""
    errors: list[str] = []

    # Check pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        ver = _extract_version_from_file(pyproject, r"tensorrt-llm==([0-9a-z.]+)")
        if ver and ver != expected_ver:
            errors.append(f"TRT-LLM: pyproject.toml has {ver}, expected {expected_ver}")

    return errors


def check_trtllm_internal_consistency(repo: Path) -> list[str]:
    """Check TRT-LLM internal consistency between pip_wheel and github_trtllm_commit."""
    errors: list[str] = []
    ctx_path = repo / "container" / "context.yaml"
    if not ctx_path.exists():
        return errors

    with open(ctx_path) as f:
        data = yaml.safe_load(f)

    if "trtllm" not in data:
        return errors

    pip_wheel = data["trtllm"].get("pip_wheel", "")
    github_commit = data["trtllm"].get("github_trtllm_commit", "")

    pip_ver = normalize_version(pip_wheel) if pip_wheel else None
    github_ver = normalize_version(github_commit) if github_commit else None

    if pip_ver and github_ver and pip_ver != github_ver:
        errors.append(
            f"TRT-LLM: context.yaml pip_wheel ({pip_ver}) != "
            f"github_trtllm_commit ({github_ver})"
        )

    return errors


def check_nixl_consistency(repo: Path, expected_ver: str) -> list[str]:
    """Check NIXL version consistency across files."""
    errors: list[str] = []

    # Check pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        ver = _extract_version_from_file(pyproject, r"nixl\[[^\]]*\][<>=]+([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(f"NIXL: pyproject.toml has {ver}, expected {expected_ver}")

    # Check lib/llm/Cargo.toml
    cargo = repo / "lib" / "llm" / "Cargo.toml"
    if cargo.exists():
        ver = _extract_version_from_file(
            cargo, r'nixl-sys\s*=\s*\{\s*version\s*=\s*"=([0-9.]+)"'
        )
        if ver and ver != expected_ver:
            errors.append(
                f"NIXL: lib/llm/Cargo.toml has {ver}, expected {expected_ver}"
            )

    # Check lib/bindings/kvbm/pyproject.toml
    kvbm = repo / "lib" / "bindings" / "kvbm" / "pyproject.toml"
    if kvbm.exists():
        ver = _extract_version_from_file(kvbm, r"nixl\[[^\]]*\]==([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(
                f"NIXL: lib/bindings/kvbm/pyproject.toml has {ver}, expected {expected_ver}"
            )

    # Check container/deps/trtllm/install_nixl.sh
    install_script = repo / "container" / "deps" / "trtllm" / "install_nixl.sh"
    if install_script.exists():
        ver = _extract_version_from_file(install_script, r"NIXL_COMMIT=([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(
                f"NIXL: install_nixl.sh has NIXL_COMMIT={ver}, expected {expected_ver}"
            )

    # Check deploy/pre-deployment/nixl/build_and_deploy.sh
    deploy_script = repo / "deploy" / "pre-deployment" / "nixl" / "build_and_deploy.sh"
    if deploy_script.exists():
        ver = _extract_version_from_file(deploy_script, r"NIXL_VERSION=([0-9.]+)")
        if ver and ver != expected_ver:
            errors.append(
                f"NIXL: build_and_deploy.sh has NIXL_VERSION={ver}, expected {expected_ver}"
            )

    return errors


def check_backend_consistency(repo: Path) -> list[str]:
    """Check all backend version consistency.

    Uses container/context.yaml as the source of truth.
    """
    versions = parse_context_yaml(repo)
    errors: list[str] = []

    if "vllm" in versions:
        errors.extend(check_vllm_consistency(repo, versions["vllm"]))

    if "sglang" in versions:
        errors.extend(check_sglang_consistency(repo, versions["sglang"]))

    if "trtllm" in versions:
        errors.extend(check_trtllm_consistency(repo, versions["trtllm"]))

    errors.extend(check_trtllm_internal_consistency(repo))

    if "nixl" in versions:
        errors.extend(check_nixl_consistency(repo, versions["nixl"]))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Dynamo backend version consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current directory)",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()

    if not (repo / "container" / "context.yaml").exists():
        print(f"WARNING: No container/context.yaml found in {repo}", file=sys.stderr)
        print("Skipping backend version consistency check.")
        sys.exit(0)

    print("Checking backend version consistency...")
    print(f"Repository: {repo}")
    print()

    versions = parse_context_yaml(repo)
    print("Source of truth (container/context.yaml):")
    for backend, ver in sorted(versions.items()):
        print(f"  {backend}: {ver}")
    print()

    errors = check_backend_consistency(repo)

    if errors:
        print("Version inconsistencies found:")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\nFound {len(errors)} inconsistency(ies).")
        sys.exit(1)
    else:
        print("All backend versions are consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
