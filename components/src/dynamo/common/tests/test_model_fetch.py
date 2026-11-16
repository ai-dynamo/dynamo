# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for fetching models and identifying sources engines cannot resolve themselves."""

from pathlib import Path

import pytest

from dynamo.common.model_fetch import fetch_model, needs_local_model_path
from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ngc://nvstaging/nim/nemotron-3-ultra:hf-rl-052726-nvfp4-e9744cc", True),
        ("Qwen/Qwen3-0.6B", False),
        ("/data/llms/Qwen3-0.6B", False),
    ],
)
def test_needs_local_model_path(model: str, expected: bool) -> None:
    assert needs_local_model_path(model) is expected


MAIN_COMMIT = "1" * 40
PINNED_COMMIT = "2" * 40


def _hf_cache_snapshot(cache: Path, model: str, commit: str, ref: str | None) -> Path:
    """Add a tokenizer-only snapshot of ``model`` at ``commit`` to a Hugging Face cache."""
    repo = cache / f"models--{model.replace('/', '--')}"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}")
    if ref is not None:
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / ref).write_text(commit)
    return snapshot


# The snapshot path fetches in a spawned process that imports the bindings.
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "snapshot_mode", [False, True], ids=["in-process", "subprocess"]
)
async def test_fetch_model_returns_the_requested_revision(
    monkeypatch, tmp_path, snapshot_mode
):
    model = "test-org/pinned-model"
    cache = tmp_path / "hub"
    main = _hf_cache_snapshot(cache, model, MAIN_COMMIT, "main")
    pinned = _hf_cache_snapshot(cache, model, PINNED_COMMIT, None)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if snapshot_mode:
        monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, str(tmp_path / "snapshot"))
    else:
        monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)

    at_commit = await fetch_model(model, ignore_weights=True, revision=PINNED_COMMIT)
    default = await fetch_model(model, ignore_weights=True)

    assert Path(at_commit) == pinned
    assert Path(default) == main
