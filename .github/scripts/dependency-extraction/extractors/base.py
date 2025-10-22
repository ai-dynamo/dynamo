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
Base extractor class for dependency extraction.

All specific extractors (Dockerfile, Python, Go, etc.) inherit from this base class.
"""

from pathlib import Path
from typing import Dict, List, Optional


class BaseExtractor:
    """Base class for all dependency extractors."""

    def __init__(
        self,
        repo_root: Path,
        component: str,
        github_repo: str = "ai-dynamo/dynamo",
        github_branch: str = "main",
    ):
        """
        Initialize the base extractor.

        Args:
            repo_root: Path to repository root
            component: Component name (e.g., "trtllm", "vllm", "shared")
            github_repo: GitHub repository in format "owner/repo"
            github_branch: Git branch for GitHub URLs
        """
        self.repo_root = repo_root
        self.component = component
        self.github_repo = github_repo
        self.github_branch = github_branch
        self.dependencies: List[Dict[str, str]] = []
        self.errors: List[str] = []

    def extract(self, file_path: Path, **kwargs) -> List[Dict[str, str]]:
        """
        Extract dependencies from a file.

        Args:
            file_path: Path to the file to extract from
            **kwargs: Additional extractor-specific arguments

        Returns:
            List of dependency dictionaries with keys:
            - Dependency Name
            - Version
            - Category
            - Source File
            - Notes
            - Line Number (optional)
        """
        raise NotImplementedError("Subclasses must implement extract()")

    def _create_dependency(
        self,
        name: str,
        version: str,
        category: str,
        source_file: str,
        notes: str = "",
        line_number: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Create a dependency dictionary with standard keys.

        Args:
            name: Dependency name
            version: Version string
            category: Dependency category
            source_file: Path to source file (relative to repo root)
            notes: Additional notes
            line_number: Optional line number in source file

        Returns:
            Dictionary with dependency information
        """
        return {
            "Dependency Name": name,
            "Version": version,
            "Category": category,
            "Source File": source_file,
            "Notes": notes,
            "Line Number": str(line_number) if line_number else "",
        }

    def _file_exists(self, file_path: Path) -> bool:
        """Check if file exists and log error if not."""
        if not file_path.exists():
            self.errors.append(f"File not found: {file_path}")
            return False
        return True

    def _read_file(self, file_path: Path) -> Optional[str]:
        """
        Read file contents and handle errors.

        Returns:
            File contents as string, or None if error
        """
        try:
            return file_path.read_text()
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return None

    def get_relative_path(self, file_path: Path) -> str:
        """Get path relative to repo root."""
        try:
            return str(file_path.relative_to(self.repo_root))
        except ValueError:
            # If path is not relative to repo_root, return as-is
            return str(file_path)

