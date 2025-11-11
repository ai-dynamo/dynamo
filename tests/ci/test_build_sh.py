# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_wheel_dir():
    """Create a temporary directory for wheel files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy wheel file so the directory is valid
        wheel_file = Path(tmpdir) / "tensorrt_llm-1.0.0-py3-none-any.whl"
        wheel_file.touch()
        yield tmpdir


@pytest.fixture
def build_script_path():
    """Get the path to the build.sh script"""
    script_dir = Path(__file__).parent.parent.parent / "container"
    build_sh = script_dir / "build.sh"
    assert build_sh.exists(), f"build.sh not found at {build_sh}"
    return str(build_sh)


def run_build_script(build_script_path, args, expect_failure=False):
    """
    Run build.sh with specified arguments and return the result.

    Args:
        build_script_path: Path to build.sh
        args: List of arguments to pass to build.sh
        expect_failure: If True, expect non-zero exit code

    Returns:
        tuple: (exit_code, stdout, stderr)
    """
    cmd = ["bash", build_script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if not expect_failure and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")

    return result.returncode, result.stdout, result.stderr


class TestBuildShTRTLLMDownload:
    """Test download intention scenarios for TRTLLM"""

    def test_default_behavior_downloads(self, build_script_path):
        """Test that default behavior (no TRTLLM flags) defaults to download"""
        args = ["--framework", "TRTLLM", "--dry-run"]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: true" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: false" in stdout
        assert (
            "Inferring download because both TENSORRTLLM_PIP_WHEEL and TENSORRTLLM_INDEX_URL are not set"
            in stdout
        )

    def test_download_with_pip_wheel_only(self, build_script_path):
        """Test download with --tensorrtllm-pip-wheel flag only"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel",
            "tensorrt-llm==1.2.0",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: true" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: false" in stdout
        assert (
            "Installing tensorrt-llm==1.2.0 trtllm version from default pip index"
            in stdout
        )

    def test_download_with_pip_wheel_and_index_url(self, build_script_path):
        """Test download with both --tensorrtllm-pip-wheel and --tensorrtllm-index-url"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel",
            "tensorrt-llm==1.2.0",
            "--tensorrtllm-index-url",
            "https://custom.pypi.org/simple",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: true" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: false" in stdout
        assert (
            "Installing tensorrt-llm==1.2.0 trtllm version from index: https://custom.pypi.org/simple"
            in stdout
        )

    def test_download_with_commit(self, build_script_path):
        """Test download with --tensorrtllm-pip-wheel and --tensorrtllm-commit"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel",
            "tensorrt-llm==1.2.0",
            "--tensorrtllm-commit",
            "abc123def456",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: true" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: false" in stdout


class TestBuildShTRTLLMInstall:
    """Test install from pre-built wheel directory scenarios"""

    def test_install_with_wheel_dir_and_commit(self, build_script_path, temp_wheel_dir):
        """Test install with --tensorrtllm-pip-wheel-dir and --tensorrtllm-commit"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel-dir",
            temp_wheel_dir,
            "--tensorrtllm-commit",
            "abc123def456",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: true" in stdout
        assert "Intent to Build TRTLLM: false" in stdout


class TestBuildShTRTLLMBuild:
    """Test build from source scenarios"""

    def test_build_with_git_url_and_commit(self, build_script_path):
        """Test build with --tensorrtllm-git-url and --tensorrtllm-commit"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--tensorrtllm-commit",
            "abc123def456",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: true" in stdout

    def test_build_with_git_url_and_wheel_dir(self, build_script_path, temp_wheel_dir):
        """Test build with --tensorrtllm-git-url and --tensorrtllm-pip-wheel-dir"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--tensorrtllm-pip-wheel-dir",
            temp_wheel_dir,
            "--tensorrtllm-commit",
            "abc123def456",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: true" in stdout


class TestBuildShTRTLLMInvalidCombinations:
    """Test invalid/conflicting flag combinations"""

    def test_build_with_git_url_requires_commit(self, build_script_path):
        """Test build with --tensorrtllm-git-url flag requires commit"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(
            build_script_path, args, expect_failure=True
        )

        # Git URL alone requires commit to be specified
        assert exit_code != 0, "Script should fail without commit"
        combined_output = stdout + stderr
        assert (
            "[ERROR] TRTLLM framework was set as a target but the TRTLLM_COMMIT variable was not set"
            in combined_output
        )

    def test_install_with_wheel_dir(self, build_script_path, temp_wheel_dir):
        """Test install with --tensorrtllm-pip-wheel-dir flag"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel-dir",
            temp_wheel_dir,
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code != 0, f"Script failed with exit code {exit_code}"
        combined_output = stdout + stderr
        assert (
            "[ERROR] TRTLLM framework was set as a target but the TRTLLM_COMMIT variable was not set."
            in combined_output
        )

    def test_conflicting_all_three_intentions(self, build_script_path, temp_wheel_dir):
        """Test that all three intentions together causes an error"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel",
            "tensorrt-llm==1.2.0",
            "--tensorrtllm-index-url",
            "https://custom.pypi.org/simple",
            "--tensorrtllm-pip-wheel-dir",
            temp_wheel_dir,
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(
            build_script_path, args, expect_failure=True
        )

        assert exit_code != 0, "Script should have failed with conflicting flags"
        combined_output = stdout + stderr
        assert (
            "[ERROR] Could not figure out the trtllm installation intent"
            in combined_output
        )

    def test_wheel_dir_with_git_url_builds(self, build_script_path, temp_wheel_dir):
        """Test that --tensorrtllm-git-url takes precedence over --tensorrtllm-pip-wheel-dir"""
        # Note: Based on the code, git-url takes precedence and triggers build intention
        # The wheel-dir is used as output directory for the build
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-pip-wheel-dir",
            temp_wheel_dir,
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--tensorrtllm-commit",
            "abc123def456",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        # According to the code logic (line 678), git-url sets build=true
        # And wheel-dir only sets install=true if git-url is NOT set (line 670)
        # So git-url takes precedence and this should succeed with build intention
        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: true" in stdout

    def test_index_url_without_pip_wheel_fails(self, build_script_path):
        """Test that --tensorrtllm-index-url alone without pip-wheel fails"""
        # According to the code, index-url alone doesn't trigger download
        # Only when both index-url AND pip-wheel are set (line 686)
        # So this should fail with an error about unclear intention
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-index-url",
            "https://custom.pypi.org/simple",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(
            build_script_path, args, expect_failure=True
        )

        # Should fail because no clear intention can be determined
        assert exit_code != 0, "Script should fail with unclear intention"
        combined_output = stdout + stderr
        assert (
            "[ERROR] Could not figure out the trtllm installation intent"
            in combined_output
        )


class TestBuildShTRTLLMFlagValidation:
    """Test individual flag parsing and validation"""

    def test_tensorrtllm_commit_flag_parsed(self, build_script_path):
        """Test that --tensorrtllm-commit flag is properly parsed"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-git-url",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "--tensorrtllm-commit",
            "test-commit-hash",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: true" in stdout

    def test_tensorrtllm_git_url_flag_parsed(self, build_script_path):
        """Test that --tensorrtllm-git-url flag is properly parsed"""
        args = [
            "--framework",
            "TRTLLM",
            "--tensorrtllm-git-url",
            "https://custom-git-url.example.com/TensorRT-LLM",
            "--tensorrtllm-commit",
            "test-commit",
            "--dry-run",
        ]
        exit_code, stdout, stderr = run_build_script(build_script_path, args)

        assert exit_code == 0, f"Script failed with exit code {exit_code}"
        assert "Intent to Download TRTLLM: false" in stdout
        assert "Intent to Install TRTLLM: false" in stdout
        assert "Intent to Build TRTLLM: true" in stdout
