# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the --max-vram-gib test selection in tests/conftest.py."""

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]
pytest_plugins = ["pytester"]


def test_max_vram_gib_keeps_only_gpu_1_tests(pytester, monkeypatch):
    """The GPU-parallel orchestrator gives each test one GPU."""
    pytester.makeconftest((Path(__file__).parent / "conftest.py").read_text())
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.gpu_1
        @pytest.mark.profiled_vram_gib(4)
        def test_one_gpu():
            pass

        @pytest.mark.gpu_2
        @pytest.mark.profiled_vram_gib(4)
        def test_two_gpus():
            pass
        """
    )
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    # --max-vram-gib checks CUDA_VISIBLE_DEVICES against the detected GPUs and
    # writes the orchestrator's test metadata to the temp dir.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(pytester.path))

    result = pytester.runpytest("-o", "addopts=", "-v", "--max-vram-gib=10")

    result.assert_outcomes(passed=1, deselected=1)
    result.stdout.fnmatch_lines(["*::test_one_gpu PASSED*"])
