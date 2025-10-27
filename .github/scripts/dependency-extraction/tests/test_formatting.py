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
Unit tests for formatting utilities.

Run with: pytest .github/scripts/dependency-extraction/tests/test_formatting.py
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.formatting import (
    format_dependency_name,
    format_package_name,
    normalize_dependency_name,
    normalize_version_for_comparison,
    strip_version_suffixes,
)


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestFormatPackageName:
    """Tests for format_package_name function."""

    def test_special_cases(self):
        """Test well-known package name formatting."""
        assert format_package_name("pytorch", "") == "PyTorch"
        assert format_package_name("numpy", "") == "NumPy"
        assert format_package_name("fastapi", "") == "FastAPI"
        assert format_package_name("tensorflow", "") == "TensorFlow"

    def test_hyphenated_names(self):
        """Test hyphen-separated name formatting."""
        assert format_package_name("some-package", "") == "Some Package"
        assert format_package_name("my-cool-lib", "") == "My Cool Lib"

    def test_underscore_names(self):
        """Test underscore-separated name formatting."""
        assert format_package_name("some_package", "") == "Some Package"
        assert format_package_name("my_cool_lib", "") == "My Cool Lib"

    def test_camel_case(self):
        """Test camelCase name formatting."""
        assert format_package_name("SomePackage", "") == "Some Package"
        assert format_package_name("MyCoolLib", "") == "My Cool Lib"

    def test_simple_names(self):
        """Test simple single-word names."""
        assert format_package_name("redis", "") == "Redis"
        assert format_package_name("celery", "") == "Celery"


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestStripVersionSuffixes:
    """Tests for strip_version_suffixes function."""

    def test_strip_ver(self):
        """Test stripping ' Ver' suffix."""
        assert strip_version_suffixes("PyTorch Ver") == "PyTorch"

    def test_strip_version(self):
        """Test stripping ' Version' suffix."""
        assert strip_version_suffixes("CUDA Version") == "CUDA"

    def test_strip_ref(self):
        """Test stripping ' Ref' suffix."""
        assert strip_version_suffixes("Git Ref") == "Git"

    def test_strip_tag(self):
        """Test stripping ' Tag' suffix."""
        assert strip_version_suffixes("Image Tag") == "Image"

    def test_no_suffix(self):
        """Test names without suffixes remain unchanged."""
        assert strip_version_suffixes("PyTorch") == "PyTorch"
        assert strip_version_suffixes("CUDA") == "CUDA"


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestNormalizeDependencyName:
    """Tests for normalize_dependency_name function."""

    def test_pytorch_normalization(self):
        """Test PyTorch name variations."""
        assert normalize_dependency_name("torch", "Python Package") == "pytorch"
        assert normalize_dependency_name("pytorch", "Python Package") == "pytorch"
        assert normalize_dependency_name("PyTorch", "Python Package") == "pytorch"

    def test_tensorrt_normalization(self):
        """Test TensorRT-LLM name variations."""
        assert normalize_dependency_name("trtllm", "") == "tensorrt-llm"
        assert normalize_dependency_name("tensorrt-llm", "") == "tensorrt-llm"
        assert normalize_dependency_name("TensorRT-LLM", "") == "tensorrt-llm"

    def test_pytorch_exceptions(self):
        """Test that PyTorch Triton is not normalized to pytorch."""
        result = normalize_dependency_name("pytorch triton", "Python Package")
        assert result != "pytorch"
        assert "triton" in result.lower()

    def test_go_module_no_normalization(self):
        """Test that Go modules are not normalized."""
        go_module = "github.com/pkg/errors"
        assert normalize_dependency_name(go_module, "Go Module") == go_module

    def test_unknown_dependency(self):
        """Test unknown dependencies are lowercased but not normalized."""
        assert normalize_dependency_name("UnknownPackage", "") == "unknownpackage"


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestNormalizeVersionForComparison:
    """Tests for normalize_version_for_comparison function."""

    def test_remove_equality(self):
        """Test removing == operator."""
        assert normalize_version_for_comparison("==1.2.3") == "1.2.3"

    def test_remove_greater_equal(self):
        """Test removing >= operator."""
        assert normalize_version_for_comparison(">=1.2.3") == "1.2.3"

    def test_remove_less_equal(self):
        """Test removing <= operator."""
        assert normalize_version_for_comparison("<=1.2.3") == "1.2.3"

    def test_remove_tilde(self):
        """Test removing ~= operator."""
        assert normalize_version_for_comparison("~=1.2.3") == "1.2.3"

    def test_compound_version(self):
        """Test compound version specs."""
        assert normalize_version_for_comparison(">=1.2.3,<2.0.0") == "1.2.3"

    def test_version_with_build(self):
        """Test versions with build metadata."""
        assert normalize_version_for_comparison("2.7.1+cu128") == "2.7.1+cu128"

    def test_plain_version(self):
        """Test plain versions remain unchanged."""
        assert normalize_version_for_comparison("1.2.3") == "1.2.3"


@pytest.mark.unit
@pytest.mark.weekly
@pytest.mark.gpu_0
class TestFormatDependencyName:
    """Tests for format_dependency_name function."""

    def test_git_url(self):
        """Test Git URL dependency naming."""
        result = format_dependency_name("git+https://github.com/org/repo.git", "Python Package", "")
        assert "Repo" in result or "repo" in result.lower()

    def test_package_with_extras(self):
        """Test package with extras."""
        result = format_dependency_name("package[extra1,extra2]", "Python Package", "")
        assert "[extra1,extra2]" in result

    def test_go_module(self):
        """Test Go module names remain unchanged."""
        go_module = "github.com/pkg/errors"
        result = format_dependency_name(go_module, "Go Module", "")
        assert result == go_module

    def test_docker_image(self):
        """Test Docker base image formatting."""
        result = format_dependency_name("nvcr.io/nvidia/pytorch", "Base Image", "")
        assert "NVIDIA" in result
        assert "PyTorch" in result

    def test_regular_package(self):
        """Test regular package formatting."""
        result = format_dependency_name("pytorch", "Python Package", "2.0.1")
        assert result == "PyTorch"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
