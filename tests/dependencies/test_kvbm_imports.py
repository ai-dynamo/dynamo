# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests to verify KVBM package and wheels are properly installed."""

import subprocess

import pytest


@pytest.mark.pre_merge
def test_kvbm_wheel_exists():
    """Verify KVBM wheel file exists in expected location."""
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
    assert "kvbm" in result.stdout, f"Expected kvbm wheel in output, got: {result.stdout}"


@pytest.mark.pre_merge
def test_kvbm_imports():
    """Verify KVBM package can be imported."""
    try:
        import kvbm

        assert kvbm is not None
    except ImportError as e:
        pytest.fail(f"Failed to import kvbm: {e}")


@pytest.mark.pre_merge
def test_kvbm_core_classes():
    """Verify KVBM core classes are available."""
    try:
        from kvbm import BlockManager, KvbmLeader, KvbmWorker

        assert BlockManager is not None, "BlockManager class not available"
        assert KvbmLeader is not None, "KvbmLeader class not available"
        assert KvbmWorker is not None, "KvbmWorker class not available"
    except ImportError as e:
        pytest.fail(f"Failed to import KVBM core classes: {e}")
