# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.hf_cache import (
    _MODELS_DIR_ENV_KEYS,
    _apply_models_dir_env,
    _restore_models_dir_env,
)
from tests.serve.lora_utils import MinioLoraConfig, MinioService


@pytest.mark.pre_merge
@pytest.mark.unit
def test_apply_bare_cache_layout(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    try:
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path)
        assert "HF_HOME" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["DYNAMO_MODELS_DIR"] == str(tmp_path)
        assert "TRANSFORMERS_CACHE" not in os.environ
    finally:
        _restore_models_dir_env(orig, tmp_cache)


@pytest.mark.pre_merge
@pytest.mark.unit
def test_apply_hf_home_layout(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "hub").mkdir()
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    try:
        assert os.environ["HF_HOME"] == str(tmp_path)
        assert "HF_HUB_CACHE" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["DYNAMO_MODELS_DIR"] == str(tmp_path)
        assert "TRANSFORMERS_CACHE" not in os.environ
    finally:
        _restore_models_dir_env(orig, tmp_cache)


@pytest.mark.pre_merge
@pytest.mark.unit
def test_restore_clears_vars_that_were_absent(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig, tmp_cache)
    for k in _MODELS_DIR_ENV_KEYS:
        assert k not in os.environ


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.parametrize("use_hf_home", [False, True])
def test_restore_preserves_preexisting_values(tmp_path, monkeypatch, use_hf_home):
    if use_hf_home:
        (tmp_path / "hub").mkdir()
    sentinel = {k: f"preexisting_{k}" for k in _MODELS_DIR_ENV_KEYS}
    for k, v in sentinel.items():
        monkeypatch.setenv(k, v)
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig, tmp_cache)
    for k, v in sentinel.items():
        assert os.environ[k] == v


@pytest.mark.pre_merge
@pytest.mark.unit
def test_models_dir_nonexistent_exits_with_code_2(tmp_path):
    missing = tmp_path / "no_such_dir"
    # Run from the project root so conftest.py is discovered and --models-dir
    # is registered before pytest_configure fires.
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"--models-dir={missing}",
            "--collect-only",
            "tests/test_models_dir_flag.py",
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr + result.stdout


@pytest.mark.pre_merge
@pytest.mark.unit
def test_restore_handles_missing_tmp_cache(tmp_path, monkeypatch):
    """_restore_models_dir_env logs a warning but does not raise when tmp_cache_dir is gone."""
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    orig, _ = _apply_models_dir_env(str(tmp_path))
    nonexistent = str(tmp_path / "already_deleted")
    _restore_models_dir_env(orig, nonexistent)  # must not raise
    for k in _MODELS_DIR_ENV_KEYS:
        assert k not in os.environ


@pytest.mark.pre_merge
@pytest.mark.unit
def test_download_lora_skips_in_models_dir_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMO_MODELS_DIR", str(tmp_path))
    service = MinioService(MinioLoraConfig())
    with pytest.raises(pytest.skip.Exception, match="read-only cache mode"):
        service.download_lora()
