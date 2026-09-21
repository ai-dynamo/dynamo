# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-argument resolution for the TensorRT-LLM worker.

The fake caches below reproduce the shapes seen in the failing CI job: a repo
whose ``refs/main`` is empty, which made ``snapshot_download`` hand the engine
``<repo>/snapshots`` and crash the worker with "Unrecognized model".
"""

import os

import pytest

from dynamo.trtllm.utils.model_path import resolve_model_path

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

MODEL = "Qwen/Qwen3-0.6B"
COMMIT = "c1899de289a04d12100db370d81485cdf75e47ca"


def _hub(tmp_path, monkeypatch):
    """Point Hugging Face's cache lookup at an empty directory under tmp_path."""
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    hub = hf_home / "hub"
    hub.mkdir(parents=True)
    return hub


def _repo(hub):
    repo = hub / "models--Qwen--Qwen3-0.6B"
    (repo / "refs").mkdir(parents=True)
    (repo / "snapshots").mkdir(parents=True)
    return repo


def _snapshot(repo, commit, complete=True):
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir()
    if complete:
        (snapshot / "config.json").write_text('{"model_type": "qwen3"}')
    return snapshot


def test_empty_ref_falls_back_to_the_cached_snapshot(tmp_path, monkeypatch):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    snapshot = _snapshot(repo, COMMIT)

    resolved = resolve_model_path(MODEL)

    assert resolved == str(snapshot)
    assert os.path.isfile(os.path.join(resolved, "config.json"))


def test_valid_ref_keeps_the_repository_id(tmp_path, monkeypatch):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text(COMMIT)
    _snapshot(repo, COMMIT)

    assert resolve_model_path(MODEL) == MODEL


def test_no_local_cache_keeps_the_repository_id(tmp_path, monkeypatch):
    _hub(tmp_path, monkeypatch)

    assert resolve_model_path(MODEL) == MODEL


def test_unusable_ref_without_a_complete_snapshot_keeps_the_repository_id(
    tmp_path, monkeypatch
):
    repo = _repo(_hub(tmp_path, monkeypatch))
    (repo / "refs" / "main").write_text("")
    _snapshot(repo, COMMIT, complete=False)

    assert resolve_model_path(MODEL) == MODEL
