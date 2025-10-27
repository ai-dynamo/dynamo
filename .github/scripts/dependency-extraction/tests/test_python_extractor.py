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
Unit tests for Python dependency extractor.

Run with: pytest .github/scripts/dependency-extraction/tests/test_python_extractor.py
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.python_deps import PythonDependencyExtractor


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestPythonDependencyExtractor:
    """Tests for PythonDependencyExtractor."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a temporary extractor instance."""
        return PythonDependencyExtractor(
            repo_root=tmp_path,
            component="test",
            github_repo="test/repo",
            github_branch="main",
        )

    def test_parse_simple_requirement(self, extractor):
        """Test parsing a simple requirement line."""
        dep_name, version, notes = extractor._parse_requirement_line("pytest==7.0.0")
        assert dep_name == "pytest"
        assert version == "==7.0.0"

    def test_parse_requirement_with_extras(self, extractor):
        """Test parsing requirement with extras."""
        dep_name, version, notes = extractor._parse_requirement_line("package[extra]>=1.0.0")
        assert dep_name == "package[extra]"
        assert version == ">=1.0.0"

    def test_parse_git_requirement(self, extractor):
        """Test parsing Git URL requirement."""
        git_url = "git+https://github.com/org/repo.git"
        dep_name, version, notes = extractor._parse_requirement_line(git_url)
        assert "git" in dep_name.lower() or "git" in version.lower()

    def test_parse_unversioned_requirement(self, extractor):
        """Test parsing unversioned requirement."""
        dep_name, version, notes = extractor._parse_requirement_line("some-package")
        assert dep_name == "some-package"
        assert version == "unspecified"

    def test_extract_requirements_file(self, extractor, tmp_path):
        """Test extracting dependencies from requirements.txt."""
        # Create a temporary requirements.txt
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
# Test requirements file
pytest==7.0.0
numpy>=1.20.0
pandas[excel]>=1.3.0

# Comment line
fastapi

git+https://github.com/org/repo.git
"""
        )

        deps = extractor.extract_requirements(req_file)

        assert len(deps) >= 4  # At least pytest, numpy, pandas, fastapi
        dep_names = [d["Dependency Name"] for d in deps]
        assert "pytest" in dep_names
        assert "numpy" in dep_names

    def test_parse_pyproject_dependency(self, extractor):
        """Test parsing pyproject.toml dependency spec."""
        dep_name, version = extractor._parse_pyproject_dependency("pytest==7.0.0")
        assert dep_name == "pytest"
        assert version == "==7.0.0"

    def test_parse_pyproject_with_extras(self, extractor):
        """Test parsing pyproject.toml dependency with extras."""
        dep_name, version = extractor._parse_pyproject_dependency(
            "package[extra]>=1.0.0"
        )
        assert dep_name == "package[extra]"
        assert version == ">=1.0.0"

    def test_nonexistent_file(self, extractor, tmp_path):
        """Test handling of nonexistent file."""
        fake_file = tmp_path / "nonexistent.txt"
        deps = extractor.extract_requirements(fake_file)
        assert len(deps) == 0
        assert len(extractor.errors) > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])