# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the model argument handed to the TensorRT-LLM engine.

For a bare repository id TensorRT-LLM resolves the model itself through
``huggingface_hub.snapshot_download``. When the Hub API cannot be reached that
function falls back to the local cache and reads ``refs/<revision>`` without
validating it, so an empty ref file yields ``<repo>/snapshots`` -- a directory
that exists but holds no ``config.json``. The engine then aborts with
"Unrecognized model ... Should have a `model_type` key in its config.json" and
the worker exits. Handing the engine an existing directory skips that
resolution entirely.

This module only reads the cache; it never writes, repairs, or deletes anything
in it, because several workers and concurrent CI jobs share one cache directory.
It imports nothing that needs CUDA, so it stays testable without a GPU.
"""

import logging
import os
import re
from typing import Optional

from huggingface_hub.constants import DEFAULT_REVISION, HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")

_CONFIG_FILE = "config.json"


def _hub_cache_dir() -> str:
    """Hugging Face cache root, in the library's own order of precedence.

    Read per call rather than through the module constant, which freezes the
    environment as it was at import time.
    """
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return cache
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return HF_HUB_CACHE


def _is_complete_snapshot(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _CONFIG_FILE))


def _referenced_commit(repo_dir: str, revision: str) -> Optional[str]:
    """Commit the cache names for this revision, or None when unusable."""
    if _COMMIT_HASH.match(revision):
        return revision
    try:
        with open(os.path.join(repo_dir, "refs", revision), encoding="utf-8") as ref:
            commit = ref.read().strip()
    except OSError:
        return None
    return commit if _COMMIT_HASH.match(commit) else None


def _sole_complete_snapshot(repo_dir: str) -> Optional[str]:
    """The one locally complete snapshot, or None when there is not exactly one."""
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    try:
        entries = sorted(os.listdir(snapshots_dir))
    except OSError:
        return None
    complete = [
        os.path.join(snapshots_dir, entry)
        for entry in entries
        if _is_complete_snapshot(os.path.join(snapshots_dir, entry))
    ]
    return complete[0] if len(complete) == 1 else None


def resolve_model_path(model: str, revision: Optional[str] = None) -> str:
    """Return the model argument to give the engine.

    The repository id is returned unchanged unless the local cache reference is
    unusable and exactly one complete snapshot is present, in which case that
    snapshot directory is returned so the engine never repeats the broken
    lookup. A machine with no local copy keeps downloading exactly as before.
    """
    if os.path.exists(model):
        return model

    revision = revision or DEFAULT_REVISION
    repo_dir = os.path.join(
        _hub_cache_dir(), repo_folder_name(repo_id=model, repo_type="model")
    )

    commit = _referenced_commit(repo_dir, revision)
    if commit is not None and _is_complete_snapshot(
        os.path.join(repo_dir, "snapshots", commit)
    ):
        return model

    if _COMMIT_HASH.match(revision):
        # A pinned commit is missing locally; never substitute another snapshot.
        return model

    snapshot = _sole_complete_snapshot(repo_dir)
    if snapshot is None:
        return model

    logging.info(
        "Cache reference %s for %s is unusable; using the local snapshot %s",
        os.path.join(repo_dir, "refs", revision),
        model,
        snapshot,
    )
    return snapshot
