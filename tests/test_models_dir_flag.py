# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tests.hf_cache as hf_cache
from tests.hf_cache import (
    _MODELS_DIR_ENV_KEYS,
    _TRANSFORMERS_CACHE_OVERRIDE_KEYS,
    _apply_models_dir_env,
    _disable_offline_with_mistral_patch,
    _enable_offline_with_mistral_patch,
    _restore_models_dir_env,
)
from tests.serve.lora_utils import MinioLoraConfig, MinioService


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_apply_bare_cache_layout(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    orig = _apply_models_dir_env(str(tmp_path))
    try:
        assert os.environ["HF_HUB_CACHE"] == str(tmp_path)
        assert "HF_HOME" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["DYNAMO_MODELS_DIR"] == str(tmp_path)
        for k in _TRANSFORMERS_CACHE_OVERRIDE_KEYS:
            assert k not in os.environ
    finally:
        _restore_models_dir_env(orig)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_apply_hf_home_layout(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    (tmp_path / "hub").mkdir()
    orig = _apply_models_dir_env(str(tmp_path))
    try:
        assert os.environ["HF_HOME"] == str(tmp_path)
        assert "HF_HUB_CACHE" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["DYNAMO_MODELS_DIR"] == str(tmp_path)
        for k in _TRANSFORMERS_CACHE_OVERRIDE_KEYS:
            assert k not in os.environ
    finally:
        _restore_models_dir_env(orig)


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_restore_clears_vars_that_were_absent(tmp_path, monkeypatch):
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    orig = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig)
    for k in _MODELS_DIR_ENV_KEYS:
        assert k not in os.environ
    assert "PYTHONPATH" not in os.environ


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("use_hf_home", [False, True])
def test_restore_preserves_preexisting_values(tmp_path, monkeypatch, use_hf_home):
    if use_hf_home:
        (tmp_path / "hub").mkdir()
    sentinel = {k: f"preexisting_{k}" for k in _MODELS_DIR_ENV_KEYS}
    for k, v in sentinel.items():
        monkeypatch.setenv(k, v)
    orig = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig)
    for k, v in sentinel.items():
        assert os.environ[k] == v


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.timeout(60)
def test_models_dir_nonexistent_exits_with_code_2(tmp_path):
    missing = tmp_path / "no_such_dir"
    # Run from the project root so conftest.py is discovered and --models-dir
    # is registered before pytest_configure fires.
    # Note: the child pytest process collects from this file itself — keep
    # module-level imports here side-effect-free to avoid spurious child failures.
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
        timeout=30,
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr + result.stdout


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_download_lora_skips_in_models_dir_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMO_MODELS_DIR", str(tmp_path))
    service = MinioService(MinioLoraConfig())
    with pytest.raises(pytest.skip.Exception, match="read-only cache mode"):
        service.download_lora()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_disable_removes_patch_dir(monkeypatch):
    """_disable_offline_with_mistral_patch cleans up the sitecustomize patch directory."""
    import tempfile

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hf_cache, "_mistral_patch_applied", False)

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    patch_dir = os.path.join(tempfile.gettempdir(), f"dynamo_test_hf_patch_{worker_id}")

    os.makedirs(patch_dir, exist_ok=True)
    (Path(patch_dir) / "sitecustomize.py").write_text("# stub")
    monkeypatch.setenv("PYTHONPATH", patch_dir)

    _disable_offline_with_mistral_patch()

    assert not Path(patch_dir).exists()
    assert "PYTHONPATH" not in os.environ


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_enable_normalizes_pythonpath_empty_components(monkeypatch):
    """_enable_offline_with_mistral_patch filters empty components from PYTHONPATH."""
    monkeypatch.setenv("PYTHONPATH", ":some:existing:path:")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hf_cache, "_mistral_patch_applied", False)
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        monkeypatch.setattr(
            PreTrainedTokenizerBase,
            "_patch_mistral_regex",
            classmethod(lambda cls, t, *a, **kw: t),
            raising=False,
        )
    except ImportError:
        pytest.skip("transformers not installed")

    _enable_offline_with_mistral_patch()
    pythonpath = os.environ.get("PYTHONPATH", "")
    assert "" not in pythonpath.split(
        ":"
    ), f"Empty component in PYTHONPATH: {pythonpath!r}"

    _disable_offline_with_mistral_patch()


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_pythonpath_restored_after_apply_restore(tmp_path, monkeypatch):
    original = "some:existing:path"
    monkeypatch.setenv("PYTHONPATH", original)
    for k in _MODELS_DIR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(hf_cache, "_mistral_patch_applied", False)
    orig = _apply_models_dir_env(str(tmp_path))
    _restore_models_dir_env(orig)
    assert os.environ["PYTHONPATH"] == original


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_enable_disable_enable_cycle(monkeypatch):
    """_enable/_disable is safe to call in sequence; PYTHONPATH and HF_HUB_OFFLINE are correct after each call."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(hf_cache, "_mistral_patch_applied", False)

    # Inject a no-op _patch_mistral_regex so the test always exercises the full
    # patching code path, regardless of the installed transformers version.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "_patch_mistral_regex"):

            @classmethod  # type: ignore[misc]
            def _noop_patch(cls, tokenizer, *args, **kwargs):
                return tokenizer

            monkeypatch.setattr(
                PreTrainedTokenizerBase,
                "_patch_mistral_regex",
                _noop_patch,
                raising=False,
            )
    except ImportError:
        pytest.skip("transformers not installed")

    _enable_offline_with_mistral_patch()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert hf_cache._mistral_patch_applied is True
    pythonpath_after_enable = os.environ.get("PYTHONPATH")

    _disable_offline_with_mistral_patch()
    assert "HF_HUB_OFFLINE" not in os.environ
    assert hf_cache._mistral_patch_applied is False
    assert os.environ.get("PYTHONPATH") is None

    _enable_offline_with_mistral_patch()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert hf_cache._mistral_patch_applied is True
    assert os.environ.get("PYTHONPATH") == pythonpath_after_enable

    _disable_offline_with_mistral_patch()
    assert hf_cache._mistral_patch_applied is False
