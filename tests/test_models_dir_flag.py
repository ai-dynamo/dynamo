# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from tests.conftest import _MODELS_DIR_ENV_KEYS, _apply_models_dir_env, _restore_models_dir_env


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_apply_bare_cache_layout(tmp_path):
    orig = _apply_models_dir_env(str(tmp_path))
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path)
    assert os.environ.get("HF_HOME") != str(tmp_path)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["DYNAMO_MODELS_DIR"] == str(tmp_path)
    _restore_models_dir_env(orig)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_apply_hf_home_layout(tmp_path):
    (tmp_path / "hub").mkdir()
    orig = _apply_models_dir_env(str(tmp_path))
    assert os.environ["HF_HOME"] == str(tmp_path)
    assert os.environ.get("HF_HUB_CACHE") != str(tmp_path)
    _restore_models_dir_env(orig)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_restore_clears_vars_that_were_absent(tmp_path):
    for k in _MODELS_DIR_ENV_KEYS:
        os.environ.pop(k, None)
    orig = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig)
    for k in _MODELS_DIR_ENV_KEYS:
        assert k not in os.environ


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_restore_preserves_preexisting_values(tmp_path):
    sentinel = {k: f"preexisting_{k}" for k in _MODELS_DIR_ENV_KEYS}
    for k, v in sentinel.items():
        os.environ[k] = v
    orig = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig)
    for k, v in sentinel.items():
        assert os.environ[k] == v


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_apply_sets_writable_transformers_cache(tmp_path):
    orig = _apply_models_dir_env(str(tmp_path))
    cache = os.environ.get("TRANSFORMERS_CACHE")
    assert cache is not None
    assert cache != str(tmp_path)
    assert Path(cache).is_dir()
    assert os.access(cache, os.W_OK)
    _restore_models_dir_env(orig)
