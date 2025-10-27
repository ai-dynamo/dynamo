#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Generate framework_versions.md from dependency extraction data.

This creates a simple, focused document showing key framework versions,
matching the structure from PR #3572 but auto-generated from dependency CSV.

Improvements over manual approach:
- Auto-generated nightly (always current)
- Shows latest + release comparison
- Extracts CUDA versions from tags
- Quick reference table at top
- Groups by component
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load dependencies from CSV."""
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def find_dependency(
    deps: List[Dict], name_pattern: str, component: str = None
) -> Optional[Dict]:
    """Find a dependency by name pattern and optional component."""
    name_lower = name_pattern.lower()
    for dep in deps:
        dep_name_lower = dep["Dependency Name"].lower()
        if name_lower in dep_name_lower or dep_name_lower in name_lower:
            if component is None or dep["Component"] == component:
                return dep
    return None


def extract_cuda_version(tag: str) -> str:
    """Extract CUDA version from image tag like '12.8.1-runtime-ubuntu24.04'."""
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", tag)
    if match:
        return match.group(1)
    return "N/A"


def generate_framework_versions(
    csv_path: Path, release_csv_path: Path = None
) -> str:
    """Generate the framework_versions.md content."""
    # Load dependencies
    latest_deps = load_csv(csv_path)
    release_deps = (
        load_csv(release_csv_path)
        if release_csv_path and release_csv_path.exists()
        else None
    )

    # Extract release version from filename if available
    release_version = None
    if release_csv_path:
        match = re.search(r"v(\d+\.\d+\.\d+)", str(release_csv_path))
        if match:
            release_version = match.group(1)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    lines = []

    # Header
    lines.extend(
        [
            "<!--",
            "SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "SPDX-License-Identifier: Apache-2.0",
            "",
            'Licensed under the Apache License, Version 2.0 (the "License");',
            "you may not use this file except in compliance with the License.",
            "You may obtain a copy of the License at",
            "",
            "http://www.apache.org/licenses/LICENSE-2.0",
            "",
            "Unless required by applicable law or agreed to in writing, software",
            'distributed under the License is distributed on an "AS IS" BASIS,',
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
            "See the License for the specific language governing permissions and",
            "limitations under the License.",
            "-->",
            "",
            "# Dynamo Framework Versions",
            "",
            f"> **⚠️ AUTO-GENERATED** - Last updated: {timestamp}",
            "> ",
            "> This document is automatically generated from [dependency extraction](.github/reports/dependency_versions_latest.csv).",
            "> To update, run: `python3 .github/scripts/dependency-extraction/generate_framework_versions.py`",
            "",
            "This document tracks the major framework dependencies and versions used in NVIDIA Dynamo.",
            "",
        ]
    )

    # Quick Reference Table
    lines.extend(["## Quick Reference", ""])

    # Build table data
    frameworks = [
        ("vLLM", "vllm", "vllm"),
        ("TensorRT-LLM", "tensorrt-llm", "trtllm"),
        ("SGLang", "sglang", "sglang"),
        ("FlashInfer", "flashinfer", "vllm"),
        ("NIXL", "nixl", "trtllm"),
    ]

    # Find CUDA base image
    cuda_base = find_dependency(latest_deps, "base image", "shared")
    cuda_version = "N/A"
    if cuda_base and "cuda" in cuda_base["Dependency Name"].lower():
        cuda_version = extract_cuda_version(cuda_base["Version"])

    # Create table
    lines.append("| Component | Latest (main) | Release |")
    lines.append("|-----------|---------------|---------|")

    for fw_name, fw_pattern, fw_component in frameworks:
        fw_dep = find_dependency(latest_deps, fw_pattern, fw_component)
        if fw_dep:
            latest_ver = fw_dep["Version"]
            release_ver = "N/A"
            if release_deps:
                rel_fw = find_dependency(release_deps, fw_pattern, fw_component)
                if rel_fw:
                    release_ver = rel_fw["Version"]
                    if release_version:
                        release_ver = f"{release_ver}"
            lines.append(f"| {fw_name} | `{latest_ver}` | `{release_ver}` |")

    # Add CUDA
    lines.append(f"| CUDA (base) | `{cuda_version}` | N/A |")

    # Add Python
    python_dep = find_dependency(latest_deps, "python", "shared")
    if python_dep and python_dep["Category"] == "Language":
        lines.append(f"| Python | `{python_dep['Version']}` | N/A |")

    lines.append("")

    # Core Framework Dependencies (detailed)
    lines.extend(["## Core Framework Dependencies", ""])

    # vLLM
    vllm_dep = find_dependency(latest_deps, "vllm", "vllm")
    if vllm_dep:
        lines.extend(
            [
                "### vLLM",
                f"- **Version**: `{vllm_dep['Version']}`",
                "- **Component**: `vllm`",
            ]
        )
        flashinfer = find_dependency(latest_deps, "flashinfer", "vllm")
        if flashinfer:
            lines.append(
                f"- **FlashInfer**: `{flashinfer['Version']}` (high-performance attention kernels)"
            )
        lines.append("")

    # TensorRT-LLM
    trtllm_dep = find_dependency(latest_deps, "tensorrt-llm", "trtllm")
    if trtllm_dep:
        lines.extend(
            [
                "### TensorRT-LLM",
                f"- **Version**: `{trtllm_dep['Version']}`",
                "- **Component**: `trtllm`",
            ]
        )
        nixl = find_dependency(latest_deps, "nixl", "trtllm")
        if nixl:
            lines.append(
                f"- **NIXL**: `{nixl['Version']}` (distributed inference networking)"
            )
        ucx = find_dependency(latest_deps, "ucx", "trtllm")
        if ucx:
            lines.append(f"- **UCX**: `{ucx['Version']}` (communication framework)")
        lines.append("")

    # SGLang
    sglang_dep = find_dependency(latest_deps, "sglang", "sglang")
    if sglang_dep:
        lines.extend(
            [
                "### SGLang",
                f"- **Version**: `{sglang_dep['Version']}`",
                "- **Component**: `sglang`",
                "",
            ]
        )

    # Base Images
    lines.extend(["## Base Images", ""])

    # CUDA Dev Image
    if cuda_base and "cuda" in cuda_base["Dependency Name"].lower():
        cuda_ver = extract_cuda_version(cuda_base["Version"])
        lines.extend(
            [
                "### CUDA Development Image",
                f"- **Version**: CUDA {cuda_ver}",
                f"- **Base Image Tag**: `{cuda_base['Version']}`",
                "- **Description**: NVIDIA CUDA development environment",
                f"- **Source**: `{cuda_base['Source File']}`",
                "",
            ]
        )

    # Runtime Images by component
    lines.extend(
        [
            "### CUDA Runtime Images",
            "- **Description**: NVIDIA CUDA runtime environment for production deployments",
            "",
        ]
    )

    for comp in ["shared", "trtllm", "vllm", "sglang"]:
        runtime = find_dependency(latest_deps, "runtime", comp)
        if runtime and runtime["Category"] == "Runtime Image":
            cuda_ver = extract_cuda_version(runtime["Version"])
            lines.append(f"- **{comp.upper()}**: `{runtime['Version']}` (CUDA {cuda_ver})")

    lines.append("")

    # Framework-Specific Configurations
    lines.extend(["## Framework-Specific Configurations", ""])

    configs = {
        "vllm": {
            "title": "vLLM",
            "build": "`container/deps/vllm/install_vllm.sh`",
            "dockerfile": "`container/Dockerfile.vllm`",
        },
        "trtllm": {
            "title": "TensorRT-LLM",
            "build": "`container/Dockerfile.trtllm`",
            "wheel": "`container/build_trtllm_wheel.sh`",
        },
        "sglang": {
            "title": "SGLang",
            "build": "`container/Dockerfile.sglang`",
        },
    }

    for comp_key, config in configs.items():
        lines.append(f"### {config['title']} Configuration")
        lines.append(f"- **Build Location**: {config['build']}")
        if "wheel" in config:
            lines.append(f"- **Wheel Builder**: {config['wheel']}")
        if "dockerfile" in config:
            lines.append(f"- **Dockerfile**: {config['dockerfile']}")
        lines.append("")

    # Dependency Management
    lines.extend(
        [
            "## Dependency Management",
            "",
            "### Build Scripts",
            "- **Main Build Script**: `container/build.sh`",
            "- **vLLM Installation**: `container/deps/vllm/install_vllm.sh`",
            "- **TensorRT-LLM Wheel**: `container/build_trtllm_wheel.sh`",
            "- **NIXL Installation**: `container/deps/trtllm/install_nixl.sh`",
            "",
            "### Python Dependencies",
            "- **Requirements File**: `container/deps/requirements.txt`",
            "- **Standard Requirements**: `container/deps/requirements.standard.txt`",
            "- **Test Requirements**: `container/deps/requirements.test.txt`",
            "",
        ]
    )

    # Notes
    lines.extend(
        [
            "## Notes",
            "",
            "- FlashInfer is only used when building vLLM from source or for ARM64 builds",
            "- Different frameworks may use slightly different CUDA versions for runtime images",
            "- NIXL and UCX are primarily used for distributed inference scenarios",
            "- The dependency versions are centrally managed through Docker build arguments and shell script variables",
            "",
        ]
    )

    # See Also section
    lines.extend(
        [
            "## See Also",
            "",
            "- [Support Matrix](docs/reference/support-matrix.md) - Supported platforms and versions",
            "- [Dependency Reports](.github/reports/README.md) - Full dependency tracking (262 total dependencies)",
            "- [Dependency Extraction System](.github/scripts/dependency-extraction/README.md) - How this doc is generated",
            "- [Container README](container/README.md) - Container build and usage details",
            "",
        ]
    )

    # Footer
    lines.extend(
        [
            "---",
            "",
            "_This document is automatically generated. Do not edit manually._",
            f"_Last generated: {timestamp}_",
            "",
            "**Update Instructions:**",
            "```bash",
            "python3 .github/scripts/dependency-extraction/generate_framework_versions.py",
            "```",
        ]
    )

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate framework_versions.md from dependency CSV"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(".github/reports/dependency_versions_latest.csv"),
        help="Path to latest dependency CSV",
    )
    parser.add_argument(
        "--release-csv",
        type=Path,
        default=None,
        help="Optional: Path to release CSV for comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("framework_versions.md"),
        help="Output path",
    )

    args = parser.parse_args()

    # Generate content
    content = generate_framework_versions(args.csv, args.release_csv)

    # Write to file
    args.output.write_text(content)

    print(f"✅ Generated {args.output}")


if __name__ == "__main__":
    main()