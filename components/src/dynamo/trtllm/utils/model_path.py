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
    environment as it was at import time. The order, the legacy variable, and
    the expansion all come from ``huggingface_hub.constants``; reading a
    different directory than the engine would make this module inert.
    """
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return os.path.expandvars(os.path.expanduser(cache))
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.expandvars(os.path.expanduser(os.path.join(home, "hub")))
    return HF_HUB_CACHE


def _is_complete_snapshot(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _CONFIG_FILE))


def _names_a_commit(ref_file: str) -> bool:
    try:
        with open(ref_file, encoding="utf-8") as ref:
            commit = ref.read().strip()
    except OSError:
        return False
    return bool(_COMMIT_HASH.match(commit))


def _sole_complete_snapshot(repo_dir: str) -> Optional[str]:
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

    The repository id is returned unchanged unless no revision was requested,
    the cache reference for the default revision is unusable, and exactly one
    complete snapshot is present -- only then is that snapshot directory
    returned, so the engine never repeats the broken lookup. A machine with no
    local copy keeps downloading exactly as before.

    Every other branch returns the input, because substituting a snapshot the
    cache does not name for the requested revision would load different weights
    without saying so, which is worse than the crash this module prevents.
    """
    if os.path.exists(model):
        return model

    if revision is not None:
        # An explicit revision names which weights to load. An unusable
        # reference is no evidence that a cached snapshot holds them, and a
        # cache carrying only some other revision must not stand in for it.
        return model

    repo_dir = os.path.join(
        _hub_cache_dir(), repo_folder_name(repo_id=model, repo_type="model")
    )
    ref_file = os.path.join(repo_dir, "refs", DEFAULT_REVISION)

    if _names_a_commit(ref_file):
        # The reference names the commit to load. If that snapshot is missing
        # or half-written, it is the engine's to fetch or finish; another
        # commit's snapshot is not a substitute for it.
        return model

    snapshot = _sole_complete_snapshot(repo_dir)
    if snapshot is None:
        return model

    logging.info(
        "Cache reference %s for %s is unusable; using the local snapshot %s",
        ref_file,
        model,
        snapshot,
    )
    return snapshot
