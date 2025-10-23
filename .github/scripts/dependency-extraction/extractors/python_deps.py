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
Python dependency extractor.

Extracts dependencies from requirements.txt and pyproject.toml files.
"""

import re
from pathlib import Path
from typing import Dict, List

import toml

from .base import BaseExtractor


class PythonDependencyExtractor(BaseExtractor):
    """Extracts Python dependencies from requirements files and pyproject.toml."""

    def extract_requirements(self, file_path: Path, category: str = "Python Package") -> List[Dict[str, str]]:
        """
        Extract dependencies from a requirements.txt file.

        Args:
            file_path: Path to requirements.txt
            category: Category override (e.g., "Python Package (Test)")

        Returns:
            List of dependency dictionaries
        """
        if not self._file_exists(file_path):
            return []

        contents = self._read_file(file_path)
        if not contents:
            return []

        dependencies = []
        source_file = self.get_relative_path(file_path)

        for line_num, line in enumerate(contents.splitlines(), 1):
            stripped = line.strip()

            # Skip comments and empty lines
            if not stripped or stripped.startswith("#"):
                continue

            # Skip requirements file references
            if stripped.startswith("-r ") or stripped.startswith("--requirement"):
                continue

            # Skip index/find-links options
            if stripped.startswith(("-i ", "--index-url", "-f ", "--find-links")):
                continue

            # Parse dependency spec
            # Handle: package==version, package>=version, package[extras]>=version
            # Also handle git+ URLs
            dep_name, version, notes = self._parse_requirement_line(stripped)

            if dep_name:
                dependencies.append(
                    self._create_dependency(
                        name=dep_name,
                        version=version,
                        category=category,
                        source_file=source_file,
                        notes=notes,
                        line_number=line_num,
                    )
                )

        return dependencies

    def extract_pyproject_toml(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Extract dependencies from pyproject.toml.

        Args:
            file_path: Path to pyproject.toml

        Returns:
            List of dependency dictionaries
        """
        if not self._file_exists(file_path):
            return []

        try:
            data = toml.load(file_path)
        except Exception as e:
            self.errors.append(f"Error parsing {file_path}: {e}")
            return []

        dependencies = []
        source_file = self.get_relative_path(file_path)

        # Extract main dependencies
        project_deps = data.get("project", {}).get("dependencies", [])
        for dep_spec in project_deps:
            dep_name, version = self._parse_pyproject_dependency(dep_spec)
            if dep_name:
                dependencies.append(
                    self._create_dependency(
                        name=dep_name,
                        version=version,
                        category="Python Package",
                        source_file=source_file,
                        notes="From pyproject.toml [project.dependencies]",
                    )
                )

        # Extract optional dependencies
        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        for group_name, deps in optional_deps.items():
            for dep_spec in deps:
                dep_name, version = self._parse_pyproject_dependency(dep_spec)
                if dep_name:
                    dependencies.append(
                        self._create_dependency(
                            name=dep_name,
                            version=version,
                            category=f"Python Package ({group_name})",
                            source_file=source_file,
                            notes=f"Optional dependency group: {group_name}",
                        )
                    )

        return dependencies

    def _parse_requirement_line(self, line: str) -> tuple:
        """
        Parse a single requirements.txt line.

        Returns:
            Tuple of (dep_name, version, notes)
        """
        # Handle Git URLs
        if line.startswith("git+"):
            match = re.search(r"git\+https?://[^/]+/([^/]+)/([^/@#]+)", line)
            if match:
                org = match.group(1)
                repo = match.group(2).replace(".git", "")
                return f"git+{org}/{repo}", "from Git", f"Git dependency: {line[:80]}"
            return line[:50], "from Git", "Git repository dependency"

        # Handle URL installs
        if line.startswith(("http://", "https://")):
            return line[:50], "from URL", "Installed from URL"

        # Standard package with version specifiers
        # Match: package[extras]>=version or package==version
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)(\[[^\]]+\])?([<>=!~]+)?(.*)$", line)
        if match:
            package_name = match.group(1)
            extras = match.group(2) or ""
            operator = match.group(3) or ""
            version_part = match.group(4).strip() if match.group(4) else ""

            # Build full name with extras
            full_name = package_name + extras if extras else package_name

            # Determine version
            if operator and version_part:
                # Remove any trailing comments or options
                version_part = version_part.split("#")[0].split(";")[0].strip()
                version = f"{operator}{version_part}" if version_part else "unspecified"
            else:
                version = "unspecified"

            return full_name, version, ""

        # Fallback: return the line as-is
        return line.split("==")[0].split(">=")[0].split("<=")[0].strip(), "unspecified", ""

    def _parse_pyproject_dependency(self, dep_spec: str) -> tuple:
        """
        Parse a pyproject.toml dependency specification.

        Returns:
            Tuple of (dep_name, version)
        """
        # Match: package[extras]>=version or package==version
        match = re.match(r"^([a-zA-Z0-9_\-]+)(\[[^\]]+\])?([<>=!~@]+)?(.*)$", dep_spec)
        if match:
            package_name = match.group(1)
            extras = match.group(2) or ""
            operator = match.group(3) or ""
            version_part = match.group(4) if match.group(4) else ""

            full_name = package_name + extras if extras else package_name

            if operator == "@":
                # URL dependency
                version = "from URL" if ("git+" in version_part or "http" in version_part) else f"@{version_part[:30]}"
            elif operator and version_part:
                version = f"{operator}{version_part}"
            else:
                version = "unspecified"

            return full_name, version

        return dep_spec, "unspecified"

