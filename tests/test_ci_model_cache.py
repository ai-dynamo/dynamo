# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CI model cache linking in conftest.py.

Validates that _link_ci_model_to_hf_cache creates correct HF-compatible
cache entries so that both Python huggingface_hub and Rust hf_hub::Cache
resolve models through symlinks to CI-cached directories.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import _link_ci_model_to_hf_cache

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]

MODEL_ID = "FakeOrg/FakeModel"


@pytest.fixture()
def ci_model_dir(tmp_path):
    """Create a fake CI model directory with minimal valid content."""
    model_dir = tmp_path / "ci_models" / "FakeOrg" / "FakeModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "fake"}')
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
    return tmp_path / "ci_models"


@pytest.fixture()
def hf_cache_dir(tmp_path, monkeypatch):
    """Point HF_HUB_CACHE to a temp directory."""
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_dir))
    return cache_dir


class TestLinkCiModelToHfCache:
    """Tests for _link_ci_model_to_hf_cache."""

    def test_creates_valid_hf_cache_structure(self, ci_model_dir, hf_cache_dir):
        """Core test: linking produces the refs/main + snapshots/ci-local layout."""
        result = _link_ci_model_to_hf_cache(MODEL_ID, str(ci_model_dir))
        assert result is True

        repo_dir = hf_cache_dir / "models--FakeOrg--FakeModel"

        # refs/main must contain the revision hash
        refs_file = repo_dir / "refs" / "main"
        assert refs_file.exists()
        assert refs_file.read_text() == "ci-local"

        # snapshots/ci-local must be a symlink to the CI model path
        snapshot_link = repo_dir / "snapshots" / "ci-local"
        assert snapshot_link.is_symlink()
        assert snapshot_link.resolve() == (ci_model_dir / "FakeOrg" / "FakeModel").resolve()

    def test_model_files_accessible_through_symlink(self, ci_model_dir, hf_cache_dir):
        """Verify files in the CI dir are readable through the HF cache symlink."""
        _link_ci_model_to_hf_cache(MODEL_ID, str(ci_model_dir))

        snapshot = hf_cache_dir / "models--FakeOrg--FakeModel" / "snapshots" / "ci-local"
        assert (snapshot / "config.json").read_text() == '{"model_type": "fake"}'
        assert (snapshot / "tokenizer.json").exists()
        assert (snapshot / "model.safetensors").exists()

    def test_hf_hub_resolves_linked_model(self, ci_model_dir, hf_cache_dir):
        """huggingface_hub's snapshot_download(local_files_only=True) finds the model."""
        from huggingface_hub import snapshot_download

        _link_ci_model_to_hf_cache(MODEL_ID, str(ci_model_dir))

        # This is the real validation: HF's own resolver finds our fake cache entry
        resolved_path = snapshot_download(
            MODEL_ID,
            local_files_only=True,
            cache_dir=str(hf_cache_dir),
        )
        assert Path(resolved_path).resolve() == (ci_model_dir / "FakeOrg" / "FakeModel").resolve()

    def test_idempotent_relink(self, ci_model_dir, hf_cache_dir):
        """Calling twice doesn't fail — symlink gets replaced cleanly."""
        assert _link_ci_model_to_hf_cache(MODEL_ID, str(ci_model_dir)) is True
        assert _link_ci_model_to_hf_cache(MODEL_ID, str(ci_model_dir)) is True

        snapshot_link = hf_cache_dir / "models--FakeOrg--FakeModel" / "snapshots" / "ci-local"
        assert snapshot_link.is_symlink()

    def test_returns_false_when_ci_dir_missing(self, hf_cache_dir, tmp_path):
        """No CI directory → returns False, no crash."""
        result = _link_ci_model_to_hf_cache(MODEL_ID, str(tmp_path / "nonexistent"))
        assert result is False

    def test_returns_false_when_config_json_missing(self, hf_cache_dir, tmp_path):
        """CI directory exists but has no config.json → returns False."""
        model_dir = tmp_path / "ci_models" / "FakeOrg" / "FakeModel"
        model_dir.mkdir(parents=True)
        # Intentionally no config.json
        result = _link_ci_model_to_hf_cache(MODEL_ID, str(tmp_path / "ci_models"))
        assert result is False

    def test_returns_false_when_model_subdir_missing(self, hf_cache_dir, tmp_path):
        """CI root exists but specific model subdir doesn't → returns False."""
        ci_root = tmp_path / "ci_models"
        ci_root.mkdir()
        result = _link_ci_model_to_hf_cache(MODEL_ID, str(ci_root))
        assert result is False

    def test_does_not_pollute_hf_cache_on_failure(self, hf_cache_dir, tmp_path):
        """Failed link attempts don't leave partial cache entries."""
        _link_ci_model_to_hf_cache(MODEL_ID, str(tmp_path / "nonexistent"))
        repo_dir = hf_cache_dir / "models--FakeOrg--FakeModel"
        assert not repo_dir.exists()
