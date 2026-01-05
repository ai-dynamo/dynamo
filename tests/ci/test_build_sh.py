# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture(scope="module")
def dynamo_repo_dir():
    """
    Create a temporary copy of the dynamo repository for isolated testing.

    Attempts to download the repository from GitHub at the current HEAD commit.
    Falls back to creating a local archive if download fails.
    """
    # Get the repository root (3 levels up from this test file)
    repo_root = Path(__file__).parent.parent.parent

    # Get commit hash from environment variable or git
    commit_hash = os.environ.get("DYNAMO_COMMIT_SHA")
    if commit_hash:
        print(f"Using commit from DYNAMO_COMMIT_SHA: {commit_hash}")
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            commit_hash = result.stdout.strip()
            print(f"Using commit from git HEAD: {commit_hash}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Failed to get git HEAD commit: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "dynamo.tar.gz"

        # Try to download from GitHub
        github_url = f"https://github.com/ai-dynamo/dynamo/archive/{commit_hash}.tar.gz"
        downloaded = False

        try:
            print(f"Attempting to download repository from {github_url}")
            with urllib.request.urlopen(github_url, timeout=30) as response:
                with open(tar_path, "wb") as f:
                    shutil.copyfileobj(response, f)
            downloaded = True
            print("Successfully downloaded repository archive from GitHub")
        except Exception as e:
            print(f"Failed to download from GitHub: {e}")
            print("Falling back to local git archive")

        # Fallback to local git archive
        if not downloaded:
            try:
                subprocess.run(
                    [
                        "git",
                        "archive",
                        "HEAD",
                        "--format=tar.gz",
                        f"--output={tar_path}",
                    ],
                    cwd=repo_root,
                    check=True,
                    timeout=60,
                )
                print("Successfully created local git archive")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                raise RuntimeError(f"Failed to create local git archive: {e}")

        # Extract the archive
        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")

        # GitHub archives extract to dynamo-{commit}/ subdirectory
        # Local git archives extract directly
        extracted_contents = list(extract_dir.iterdir())
        if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
            # GitHub format: single subdirectory
            repo_dir = extracted_contents[0]
        else:
            # Local git archive format: files directly in extract_dir
            repo_dir = extract_dir

        print(f"Repository extracted to: {repo_dir}")
        yield str(repo_dir)


@pytest.fixture
def temp_wheel_dir():
    """Create a temporary directory for wheel files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy wheel file so the directory is valid
        wheel_file = Path(tmpdir) / "tensorrt_llm-1.0.0-py3-none-any.whl"
        wheel_file.touch()
        yield tmpdir


@pytest.fixture
def build_script_path(dynamo_repo_dir):
    """Get the path to the build.sh script from the temporary repository"""
    build_sh = Path(dynamo_repo_dir) / "container" / "build.sh"
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

    # Add placeholder environment variables required by build.sh
    env = os.environ.copy()
    env["commit_id"] = "commit_id"
    env["current_tag"] = "current_tag"
    env["latest_tag"] = "latest_tag"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)

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
        assert "TRTLLM pip wheel output directory is: /tmp/trtllm_wheel/"

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
