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

"""Tests for .github/scripts/check_versions.py.

TDD tests for the version consistency checker. Run with:
    pytest tests/ci/test_check_versions.py -v -m "unit and pre_merge"
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class TestNormalizeVersion:
    """Test version normalization logic."""

    def test_strips_v_prefix(self):
        from check_versions import normalize_version

        assert normalize_version("v0.17.1") == "0.17.1"

    def test_strips_v_prefix_with_rc(self):
        from check_versions import normalize_version

        assert normalize_version("v1.3.0rc7") == "1.3.0rc7"

    def test_extracts_from_pip_equality_spec(self):
        from check_versions import normalize_version

        assert normalize_version("tensorrt-llm==1.3.0rc7") == "1.3.0rc7"

    def test_extracts_from_pip_lte_spec(self):
        from check_versions import normalize_version

        assert normalize_version("nixl[cu12]<=0.10.1") == "0.10.1"

    def test_extracts_from_pip_spec_with_extras(self):
        from check_versions import normalize_version

        assert normalize_version("vllm[flashinfer,runai]==0.17.1") == "0.17.1"
        assert normalize_version("sglang[diffusion]==0.5.9") == "0.5.9"

    def test_extracts_from_sglang_runtime_tag(self):
        from check_versions import normalize_version

        assert normalize_version("v0.5.9-runtime") == "0.5.9"

    def test_extracts_from_sglang_cuda_runtime_tag(self):
        from check_versions import normalize_version

        assert normalize_version("v0.5.9-cu130-runtime") == "0.5.9"

    def test_handles_plain_version(self):
        from check_versions import normalize_version

        assert normalize_version("0.10.1") == "0.10.1"

    def test_handles_rc_version(self):
        from check_versions import normalize_version

        assert normalize_version("1.3.0rc7") == "1.3.0rc7"


class TestParseContextYaml:
    """Test container/context.yaml parsing."""

    def test_extracts_vllm_version(self, tmp_path: Path):
        from check_versions import parse_context_yaml

        ctx = tmp_path / "container" / "context.yaml"
        ctx.parent.mkdir(parents=True)
        ctx.write_text(
            """
vllm:
  cuda12.9:
    vllm_ref: v0.17.1
"""
        )
        versions = parse_context_yaml(tmp_path)
        assert versions["vllm"] == "0.17.1"

    def test_extracts_sglang_version(self, tmp_path: Path):
        from check_versions import parse_context_yaml

        ctx = tmp_path / "container" / "context.yaml"
        ctx.parent.mkdir(parents=True)
        ctx.write_text(
            """
sglang:
  cuda12.9:
    runtime_image_tag: v0.5.9-runtime
"""
        )
        versions = parse_context_yaml(tmp_path)
        assert versions["sglang"] == "0.5.9"

    def test_extracts_trtllm_version(self, tmp_path: Path):
        from check_versions import parse_context_yaml

        ctx = tmp_path / "container" / "context.yaml"
        ctx.parent.mkdir(parents=True)
        ctx.write_text(
            """
trtllm:
  pip_wheel: tensorrt-llm==1.3.0rc7
"""
        )
        versions = parse_context_yaml(tmp_path)
        assert versions["trtllm"] == "1.3.0rc7"

    def test_extracts_nixl_version(self, tmp_path: Path):
        from check_versions import parse_context_yaml

        ctx = tmp_path / "container" / "context.yaml"
        ctx.parent.mkdir(parents=True)
        ctx.write_text(
            """
dynamo:
  nixl_ref: 0.10.1
"""
        )
        versions = parse_context_yaml(tmp_path)
        assert versions["nixl"] == "0.10.1"

    def test_returns_all_backend_versions(self, full_mock_repo: Path):
        from check_versions import parse_context_yaml

        versions = parse_context_yaml(full_mock_repo)
        assert versions["vllm"] == "0.17.1"
        assert versions["sglang"] == "0.5.9"
        assert versions["trtllm"] == "1.3.0rc7"
        assert versions["nixl"] == "0.10.1"


class TestCheckVllmConsistency:
    """Test vLLM version consistency checks."""

    def test_detects_pyproject_mismatch(self, tmp_path: Path):
        from check_versions import check_vllm_consistency

        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
vllm:
  cuda12.9:
    vllm_ref: v0.17.1
"""
        )
        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.16.0"]
"""
        )
        errors = check_vllm_consistency(tmp_path, "0.17.1")
        assert len(errors) >= 1
        assert any("pyproject.toml" in e and "0.16.0" in e for e in errors)

    def test_detects_install_script_mismatch(self, tmp_path: Path):
        from check_versions import check_vllm_consistency

        (tmp_path / "container" / "deps" / "vllm").mkdir(parents=True)
        (tmp_path / "container" / "deps" / "vllm" / "install_vllm.sh").write_text(
            """#!/bin/bash
VLLM_VER=0.16.0
"""
        )
        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.17.1"]
"""
        )
        errors = check_vllm_consistency(tmp_path, "0.17.1")
        assert len(errors) >= 1
        assert any("install_vllm.sh" in e and "0.16.0" in e for e in errors)

    def test_passes_when_consistent(self, tmp_path: Path):
        from check_versions import check_vllm_consistency

        (tmp_path / "container" / "deps" / "vllm").mkdir(parents=True)
        (tmp_path / "container" / "deps" / "vllm" / "install_vllm.sh").write_text(
            """#!/bin/bash
VLLM_VER=0.17.1
"""
        )
        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.17.1"]
"""
        )
        errors = check_vllm_consistency(tmp_path, "0.17.1")
        assert errors == []


class TestCheckSglangConsistency:
    """Test SGLang version consistency checks."""

    def test_detects_pyproject_mismatch(self, tmp_path: Path):
        from check_versions import check_sglang_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
sglang = ["sglang[diffusion]==0.5.8"]
"""
        )
        errors = check_sglang_consistency(tmp_path, "0.5.9")
        assert len(errors) == 1
        assert "pyproject.toml" in errors[0]
        assert "0.5.8" in errors[0]

    def test_passes_when_consistent(self, tmp_path: Path):
        from check_versions import check_sglang_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
sglang = ["sglang[diffusion]==0.5.9"]
"""
        )
        errors = check_sglang_consistency(tmp_path, "0.5.9")
        assert errors == []


class TestCheckTrtllmConsistency:
    """Test TensorRT-LLM version consistency checks."""

    def test_detects_pyproject_mismatch(self, tmp_path: Path):
        from check_versions import check_trtllm_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
trtllm = ["tensorrt-llm==1.2.0"]
"""
        )
        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
trtllm:
  pip_wheel: tensorrt-llm==1.3.0rc7
  github_trtllm_commit: v1.3.0rc7
"""
        )
        errors = check_trtllm_consistency(tmp_path, "1.3.0rc7")
        assert len(errors) >= 1
        assert any("pyproject.toml" in e and "1.2.0" in e for e in errors)

    def test_detects_internal_mismatch(self, tmp_path: Path):
        from check_versions import check_trtllm_internal_consistency

        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
trtllm:
  pip_wheel: tensorrt-llm==1.3.0rc7
  github_trtllm_commit: v1.2.0
"""
        )
        errors = check_trtllm_internal_consistency(tmp_path)
        assert len(errors) == 1
        assert "pip_wheel" in errors[0] and "github_trtllm_commit" in errors[0]

    def test_passes_when_consistent(self, tmp_path: Path):
        from check_versions import check_trtllm_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
trtllm = ["tensorrt-llm==1.3.0rc7"]
"""
        )
        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
trtllm:
  pip_wheel: tensorrt-llm==1.3.0rc7
  github_trtllm_commit: v1.3.0rc7
"""
        )
        errors = check_trtllm_consistency(tmp_path, "1.3.0rc7")
        assert errors == []


class TestCheckNixlConsistency:
    """Test NIXL version consistency checks."""

    def test_detects_pyproject_mismatch(self, tmp_path: Path):
        from check_versions import check_nixl_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.dependencies]
nixl = ["nixl[cu12]<=0.9.0"]
"""
        )
        errors = check_nixl_consistency(tmp_path, "0.10.1")
        assert len(errors) >= 1
        assert any("pyproject.toml" in e and "0.9.0" in e for e in errors)

    def test_detects_cargo_mismatch(self, tmp_path: Path):
        from check_versions import check_nixl_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.dependencies]
nixl = ["nixl[cu12]<=0.10.1"]
"""
        )
        (tmp_path / "lib" / "llm").mkdir(parents=True)
        (tmp_path / "lib" / "llm" / "Cargo.toml").write_text(
            """
[dependencies]
nixl-sys = { version = "=0.9.0", optional = true }
"""
        )
        errors = check_nixl_consistency(tmp_path, "0.10.1")
        assert len(errors) >= 1
        assert any("Cargo.toml" in e and "0.9.0" in e for e in errors)

    def test_detects_install_script_mismatch(self, tmp_path: Path):
        from check_versions import check_nixl_consistency

        (tmp_path / "pyproject.toml").write_text(
            """
[project.dependencies]
nixl = ["nixl[cu12]<=0.10.1"]
"""
        )
        (tmp_path / "container" / "deps" / "trtllm").mkdir(parents=True)
        (tmp_path / "container" / "deps" / "trtllm" / "install_nixl.sh").write_text(
            """#!/bin/bash
NIXL_COMMIT=0.9.0
"""
        )
        errors = check_nixl_consistency(tmp_path, "0.10.1")
        assert len(errors) >= 1
        assert any("install_nixl.sh" in e and "0.9.0" in e for e in errors)

    def test_passes_when_consistent(self, full_mock_repo: Path):
        from check_versions import check_nixl_consistency

        errors = check_nixl_consistency(full_mock_repo, "0.10.1")
        assert errors == []


class TestCheckBackendConsistency:
    """Test combined backend consistency check."""

    def test_detects_multiple_mismatches(self, tmp_path: Path):
        from check_versions import check_backend_consistency

        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
dynamo:
  nixl_ref: 0.10.1
vllm:
  cuda12.9:
    vllm_ref: v0.17.1
sglang:
  cuda12.9:
    runtime_image_tag: v0.5.9-runtime
trtllm:
  pip_wheel: tensorrt-llm==1.3.0rc7
  github_trtllm_commit: v1.3.0rc7
"""
        )
        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.16.0"]
sglang = ["sglang[diffusion]==0.5.8"]
trtllm = ["tensorrt-llm==1.2.0"]

[project.dependencies]
nixl = ["nixl[cu12]<=0.9.0"]
"""
        )
        errors = check_backend_consistency(tmp_path)
        assert len(errors) >= 4

    def test_passes_when_all_consistent(self, full_mock_repo: Path):
        from check_versions import check_backend_consistency

        errors = check_backend_consistency(full_mock_repo)
        assert errors == []


class TestMain:
    """Test main entry point."""

    def test_exits_zero_when_consistent(self, full_mock_repo: Path, monkeypatch):
        import check_versions

        monkeypatch.setattr(
            "sys.argv", ["check_versions.py", "--repo-root", str(full_mock_repo)]
        )
        with pytest.raises(SystemExit) as exc_info:
            check_versions.main()
        assert exc_info.value.code == 0

    def test_exits_nonzero_when_inconsistent(self, tmp_path: Path, monkeypatch):
        import check_versions

        (tmp_path / "container").mkdir()
        (tmp_path / "container" / "context.yaml").write_text(
            """
vllm:
  cuda12.9:
    vllm_ref: v0.17.1
"""
        )
        (tmp_path / "pyproject.toml").write_text(
            """
[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.16.0"]
"""
        )
        monkeypatch.setattr(
            "sys.argv", ["check_versions.py", "--repo-root", str(tmp_path)]
        )
        with pytest.raises(SystemExit) as exc_info:
            check_versions.main()
        assert exc_info.value.code == 1
