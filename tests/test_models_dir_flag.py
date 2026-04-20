# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _MODELS_DIR_ENV_KEYS,
    _apply_models_dir_env,
    _restore_models_dir_env,
)
from tests.serve.lora_utils import MinioLoraConfig, MinioService


@pytest.mark.pre_merge
@pytest.mark.gpu_0
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
        cache = os.environ.get("TRANSFORMERS_CACHE")
        assert cache is not None and cache != str(tmp_path)
        assert Path(cache).is_dir() and os.access(cache, os.W_OK)
    finally:
        _restore_models_dir_env(orig, tmp_cache)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
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
        cache = os.environ.get("TRANSFORMERS_CACHE")
        assert cache is not None and cache != str(tmp_path)
        assert Path(cache).is_dir() and os.access(cache, os.W_OK)
    finally:
        _restore_models_dir_env(orig, tmp_cache)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_restore_clears_vars_that_were_absent(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig, tmp_cache)
    for k in _MODELS_DIR_ENV_KEYS:
        assert k not in os.environ


@pytest.mark.pre_merge
@pytest.mark.gpu_0
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
@pytest.mark.gpu_0
@pytest.mark.unit
def test_apply_sets_writable_transformers_cache(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    orig, tmp_cache = _apply_models_dir_env(str(tmp_path))
    try:
        cache = os.environ.get("TRANSFORMERS_CACHE")
        assert cache is not None
        assert cache != str(tmp_path)
        assert Path(cache).is_dir()
        assert os.access(cache, os.W_OK)
    finally:
        _restore_models_dir_env(orig, tmp_cache)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_models_dir_nonexistent_exits_with_code_2():
    # Run from the project root so conftest.py is discovered and --models-dir
    # is registered before pytest_configure fires.
    project_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--models-dir=/nonexistent_path_xyz_dynamo_8362",
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
@pytest.mark.gpu_0
@pytest.mark.unit
def test_download_lora_skips_in_models_dir_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMO_MODELS_DIR", str(tmp_path))
    service = MinioService(MinioLoraConfig())
    with pytest.raises(pytest.skip.Exception, match="read-only cache mode"):
        service.download_lora()
