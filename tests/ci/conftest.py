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

"""Shared fixtures for CI script tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def mock_repo(tmp_path: Path) -> Path:
    """Create a minimal mock repo structure for testing."""
    (tmp_path / "container").mkdir()
    (tmp_path / "pyproject.toml").write_text('version = "1.0.0"\n')
    return tmp_path


@pytest.fixture
def full_mock_repo(tmp_path: Path) -> Path:
    """Create a complete mock repo with all version-related files."""
    (tmp_path / "container").mkdir()
    (tmp_path / "container" / "deps" / "vllm").mkdir(parents=True)
    (tmp_path / "container" / "deps" / "trtllm").mkdir(parents=True)
    (tmp_path / "deploy" / "pre-deployment" / "nixl").mkdir(parents=True)
    (tmp_path / "lib" / "llm").mkdir(parents=True)
    (tmp_path / "lib" / "bindings" / "kvbm").mkdir(parents=True)

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
[project]
version = "1.0.0"

[project.optional-dependencies]
vllm = ["vllm[flashinfer,runai]==0.17.1"]
sglang = ["sglang[diffusion]==0.5.9"]
trtllm = ["tensorrt-llm==1.3.0rc7"]

[project.dependencies]
nixl = ["nixl[cu12]<=0.10.1"]
"""
    )

    (tmp_path / "container" / "deps" / "vllm" / "install_vllm.sh").write_text(
        """#!/bin/bash
VLLM_VER=0.17.1
"""
    )

    (tmp_path / "container" / "deps" / "trtllm" / "install_nixl.sh").write_text(
        """#!/bin/bash
NIXL_COMMIT=0.10.1
"""
    )

    (
        tmp_path / "deploy" / "pre-deployment" / "nixl" / "build_and_deploy.sh"
    ).write_text(
        """#!/bin/bash
NIXL_VERSION=0.10.1
"""
    )

    (tmp_path / "lib" / "llm" / "Cargo.toml").write_text(
        """
[package]
name = "llm"

[dependencies]
nixl-sys = { version = "=0.10.1", optional = true }
"""
    )

    (tmp_path / "lib" / "bindings" / "kvbm" / "pyproject.toml").write_text(
        """
[project.optional-dependencies]
cu12 = ["nixl[cu12]==0.10.1"]
cu13 = ["nixl[cu13]==0.10.1"]
"""
    )

    return tmp_path
