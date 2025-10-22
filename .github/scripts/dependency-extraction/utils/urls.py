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
URL generation utilities for dependencies.

This module contains functions for generating package source URLs
and GitHub file URLs.
"""

import urllib.parse


def generate_github_file_url(
    file_path: str, github_repo: str, github_branch: str, line_number: str = None
) -> str:
    """Generate a GitHub URL for a file."""
    url = f"https://github.com/{github_repo}/blob/{github_branch}/{file_path}"

    # Add line number if available
    if line_number and line_number.isdigit():
        url += f"#L{line_number}"

    return url


def generate_package_source_url(dep_name: str, category: str, source_file: str) -> str:
    """
    Generate a URL to the package's source (PyPI, NGC, Docker Hub, etc.).

    Args:
        dep_name: Dependency name
        category: Dependency category
        source_file: Path to the source file

    Returns:
        URL to the package's home page or N/A
    """
    dep_lower = dep_name.lower()

    # Docker images
    if category in ("Base Image", "Docker Compose Service"):
        dep_str = dep_name.lower()
        if "nvcr.io" in dep_str or "nvidia" in dep_str:
            # Extract image name for NGC
            image_slug = dep_name.split("/")[-1].lower()
            return f"https://catalog.ngc.nvidia.com/orgs/nvidia/containers/{image_slug}"
        elif "/" in dep_name:
            # Docker Hub
            return f"https://hub.docker.com/r/{dep_name}"

    # Helm Charts
    if category == "Helm Chart Dependency":
        # OCI registries
        if "nvcr.io" in dep_name:
            chart_slug = dep_name.split("/")[-1]
            return (
                f"https://catalog.ngc.nvidia.com/orgs/nvidia/helm-charts/{chart_slug}"
            )
        # Artifact Hub
        if not dep_name.startswith("file://"):
            chart_name = dep_name.split("/")[-1] if "/" in dep_name else dep_name
            return f"https://artifacthub.io/packages/search?ts_query_web={urllib.parse.quote(chart_name)}"

    # Python packages
    if "Python" in category or "pyproject.toml" in source_file:
        # Special handling for Git dependencies
        if (
            "git+" in dep_name
            or dep_name.startswith("http://")
            or dep_name.startswith("https://")
        ):
            # Return the Git URL directly
            return dep_name

        # Standard PyPI package
        package_name = dep_name.split("[")[0] if "[" in dep_name else dep_name
        return f"https://pypi.org/project/{package_name}/"

    # Go modules
    if category in ("Go Module", "Go Dependency"):
        # Use pkg.go.dev for Go module documentation
        return f"https://pkg.go.dev/{dep_name}"

    # Rust crates
    if category == "Rust Crate":
        return f"https://crates.io/crates/{dep_name}"

    # Rust toolchain
    if category == "Rust Toolchain":
        return "https://rust-lang.github.io/rustup/concepts/toolchains.html"

    # Language versions
    if category == "Language":
        if "python" in dep_lower:
            return "https://www.python.org/downloads/"
        elif "go" in dep_lower:
            return "https://go.dev/dl/"
        elif "rust" in dep_lower:
            return "https://www.rust-lang.org/tools/install"

    # CUDA
    if "cuda" in dep_lower:
        return "https://developer.nvidia.com/cuda-downloads"

    # Default: return N/A
    return "N/A"
