# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests to verify KVBM package and wheels are properly installed."""

import subprocess

import pytest


# Helper functions for KVBM verification
def _check_kvbm_wheel_exists():
    """Helper to verify KVBM wheel file exists in expected location."""
    result = subprocess.run(
        ["bash", "-c", "ls /opt/dynamo/wheelhouse/kvbm*.whl"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"KVBM wheel not found in /opt/dynamo/wheelhouse/\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert (
        "kvbm" in result.stdout
    ), f"Expected kvbm wheel in output, got: {result.stdout}"


def _check_kvbm_imports():
    """Helper to verify KVBM package and core classes can be imported."""
    try:
        import kvbm
        from kvbm import BlockManager, KvbmLeader, KvbmWorker

        assert kvbm is not None, "kvbm module is None"
        assert BlockManager is not None, "BlockManager class not available"
        assert KvbmLeader is not None, "KvbmLeader class not available"
        assert KvbmWorker is not None, "KvbmWorker class not available"
    except ImportError as e:
        pytest.fail(f"Failed to import KVBM package or core classes: {e}")


# Base tests (no framework markers) - run in main job with --framework none --enable-kvbm
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_kvbm_wheel_exists():
    """Verify KVBM wheel file exists in expected location."""
    _check_kvbm_wheel_exists()


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_kvbm_imports():
    """Verify KVBM package and core classes can be imported."""
    _check_kvbm_imports()


# vLLM-specific tests - run in vLLM job (vLLM auto-enables KVBM)
@pytest.mark.pre_merge
@pytest.mark.vllm
@pytest.mark.unit
@pytest.mark.gpu_0
def test_kvbm_wheel_exists_vllm():
    """Verify KVBM wheel exists in vLLM image."""
    _check_kvbm_wheel_exists()


@pytest.mark.pre_merge
@pytest.mark.vllm
@pytest.mark.unit
@pytest.mark.gpu_0
def test_kvbm_imports_vllm():
    """Verify KVBM package and core classes can be imported in vLLM image."""
    _check_kvbm_imports()


# TRT-LLM-specific tests - run in TRT-LLM job (TRT-LLM auto-enables KVBM)
@pytest.mark.pre_merge
@pytest.mark.trtllm
@pytest.mark.unit
@pytest.mark.gpu_0
def test_kvbm_wheel_exists_trtllm():
    """Verify KVBM wheel exists in TRT-LLM image."""
    _check_kvbm_wheel_exists()


@pytest.mark.pre_merge
@pytest.mark.trtllm
@pytest.mark.unit
@pytest.mark.gpu_0
def test_kvbm_imports_trtllm():
    """Verify KVBM package and core classes can be imported in TRT-LLM image."""
    _check_kvbm_imports()
