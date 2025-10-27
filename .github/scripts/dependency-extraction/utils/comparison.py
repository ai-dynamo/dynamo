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
Version comparison utilities for dependency tracking.

This module contains functions for comparing dependency versions and
detecting discrepancies across the repository.
"""

from typing import Dict, List

from .formatting import normalize_dependency_name, normalize_version_for_comparison


def detect_version_discrepancies(
    dependencies: List[Dict[str, str]], known_discrepancies: List[Dict[str, str]] = None
) -> List[Dict[str, any]]:
    """
    Detect dependencies that appear multiple times with different versions.

    Args:
        dependencies: List of dependency dictionaries
        known_discrepancies: Optional list of known/intentional discrepancies
            Format: [{"dependency": "PyTorch", "reason": "..."}, ...]

    Returns:
        List of dictionaries containing discrepancy information:
        - normalized_name: The normalized dependency name
        - versions: List of original version strings found
        - normalized_versions: List of normalized versions
        - instances: List of {version, source_file, component} for each occurrence
        - is_critical: Whether any instance is critical
        - is_known: Whether this is a documented known discrepancy
        - known_reason: Reason for known discrepancy (if applicable)

    Note: This intentionally filters out some categories to reduce false positives:
    - Base/Runtime Images (intentionally different per component)
    - Go indirect dependencies (transitive, expected to vary)
    - Pinning style differences (e.g., "0.6.0" vs "<=0.6.0" are considered the same)
    """
    # Categories to skip (expected to vary by component)
    skip_categories = {
        "Base Image",
        "Runtime Image",
        "Docker Compose Service",  # Services use different base images
    }

    # Dependency names to skip (even if they have different categories)
    skip_names = {
        "base image",
        "runtime image",
        "base",  # Often refers to base images
    }

    # Create a map of known discrepancies for quick lookup
    known_discrepancy_map = {}
    if known_discrepancies:
        for kd in known_discrepancies:
            dep_name = kd.get("dependency", "").lower()
            if dep_name:
                known_discrepancy_map[dep_name] = kd.get("reason", "")

    # Group dependencies by normalized name
    dependency_groups = {}

    for dep in dependencies:
        category = dep["Category"]
        normalized_name = normalize_dependency_name(dep["Dependency Name"], category)

        # Skip unversioned dependencies for discrepancy detection
        if dep["Version"] in ["unspecified", "N/A", "", "latest"]:
            continue

        # Skip categories that are expected to vary
        if category in skip_categories:
            continue

        # Skip dependency names that are expected to vary
        if normalized_name in skip_names:
            continue

        # Skip Go indirect dependencies (transitive dependencies)
        if category == "Go Dependency" and "indirect" in dep.get("Notes", "").lower():
            continue

        if normalized_name not in dependency_groups:
            dependency_groups[normalized_name] = []

        dependency_groups[normalized_name].append(
            {
                "original_name": dep["Dependency Name"],
                "version": dep["Version"],
                "source_file": dep["Source File"],
                "component": dep["Component"],
                "category": dep["Category"],
                "critical": dep["Critical"] == "Yes",
            }
        )

    # Detect discrepancies: same normalized name with different versions
    # Use normalized versions to ignore pinning style differences
    discrepancies = []

    for normalized_name, instances in dependency_groups.items():
        # Get unique normalized versions (ignoring pinning operators)
        normalized_versions = set(
            normalize_version_for_comparison(inst["version"]) for inst in instances
        )

        # If multiple normalized versions exist, it's a real discrepancy
        if len(normalized_versions) > 1:
            # Get the original versions for display
            original_versions = sorted(set(inst["version"] for inst in instances))

            # Check if this is a known discrepancy
            is_known = normalized_name in known_discrepancy_map
            known_reason = (
                known_discrepancy_map.get(normalized_name, "") if is_known else None
            )

            discrepancies.append(
                {
                    "normalized_name": normalized_name,
                    "versions": original_versions,
                    "normalized_versions": sorted(normalized_versions),
                    "instances": instances,
                    "is_critical": any(inst["critical"] for inst in instances),
                    "is_known": is_known,
                    "known_reason": known_reason,
                }
            )

    return discrepancies


def output_github_warnings(discrepancies: List[Dict[str, any]]) -> None:
    """
    Output GitHub Actions warning annotations for version discrepancies.

    This uses the GitHub Actions workflow command format:
    ::warning file={file},line={line}::{message}

    See: https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions
    """
    for disc in discrepancies:
        normalized_name = disc["normalized_name"]
        versions = disc["versions"]
        is_critical = disc["is_critical"]
        is_known = disc.get("is_known", False)
        known_reason = disc.get("known_reason", "")
        instances = disc["instances"]

        # Create a concise message for the annotation
        critical_prefix = "[CRITICAL] " if is_critical else ""
        known_prefix = "[KNOWN] " if is_known else ""
        versions_str = ", ".join(versions)

        # Output a warning for each source file where the dependency appears
        for inst in instances:
            message = (
                f"{critical_prefix}{known_prefix}Version discrepancy detected for '{normalized_name}': "
                f"found {inst['version']} here, but also appears as {versions_str} elsewhere"
            )

            if is_known and known_reason:
                message += f" (Known issue: {known_reason})"

            # Output GitHub Actions warning annotation
            # Format: ::warning file={name}::{message}
            print(f"::warning file={inst['source_file']}::{message}")