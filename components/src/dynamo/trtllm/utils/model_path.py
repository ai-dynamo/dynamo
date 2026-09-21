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
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.constants import (
    DEFAULT_REVISION,
    ENV_VARS_TRUE_VALUES,
    HF_HUB_CACHE,
)
from huggingface_hub.errors import (
    DisabledRepoError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.file_download import repo_folder_name

_CONFIG_FILE = "config.json"

_HUB_PROBE_TIMEOUT = 10.0

# A refusal that names this repository: the Hub answered, so it is reachable.
_HUB_ANSWERED_ERRORS = (
    DisabledRepoError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# A failure to get an answer at all. ``huggingface_hub``'s own HTTP and offline
# errors derive from ``OSError``, but the transport underneath does not: on 1.x
# an unreachable endpoint surfaces as ``httpx.ConnectError``, which is a plain
# ``Exception``. Catching only ``OSError`` would leave the outage this module
# exists for uncaught; 0.x uses ``requests``, whose errors are ``OSError``s.
_UNREACHABLE_ERRORS: tuple[type[BaseException], ...] = (OSError,)
try:
    from httpx import HTTPError as _HttpxHTTPError
except ImportError:
    pass
else:
    _UNREACHABLE_ERRORS = (OSError, _HttpxHTTPError)


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
    """A config, and no entry still waiting on its blob.

    ``huggingface_hub`` links a file into the snapshot only once that blob has
    finished downloading, so a link that resolves to nothing marks a download
    another process is still doing, not a snapshot the engine can load.
    """
    if not os.path.isfile(os.path.join(path, _CONFIG_FILE)):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return all(os.path.exists(os.path.join(path, entry)) for entry in entries)


def _ref_is_empty(ref_file: str) -> bool:
    """The one reference shape that makes the engine's own lookup go wrong.

    An empty reference is joined onto ``<repo>/snapshots`` as an empty commit,
    yielding a directory that exists and holds no config. A reference that is
    missing, unreadable, or names something else makes the lookup raise
    instead, which is a loud failure this module has no better answer for.
    """
    try:
        with open(ref_file, encoding="utf-8") as ref:
            return ref.read().strip() == ""
    except OSError:
        return False


def _offline_mode() -> bool:
    """``huggingface_hub``'s own reading of the offline variables.

    Only ``1``, ``ON``, ``YES`` and ``TRUE`` mean offline, case-insensitively;
    anything else, ``false`` and ``no`` included, means online. Reading these
    any other way would stop the probe on a machine the Hub is reachable from.
    Read per call, since the library's constant is fixed at import time.
    """
    value = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    return value is not None and value.upper() in ENV_VARS_TRUE_VALUES


def _hub_is_unreachable(model: str) -> bool:
    """Whether the engine's own lookup can still ask the Hub.

    A reachable Hub resolves the revision itself, which beats anything guessed
    from a cache whose reference no longer describes it. Only once that has
    failed is the local snapshot the better answer. Asked only on the broken
    reference path, so a healthy start makes no request.

    Only a missing answer counts as unreachable. A refusal naming the
    repository proves the Hub is reachable, and any other exception is a defect
    in this call rather than an outage: it propagates, because silently reading
    it as an outage would substitute a snapshot on the strength of our own bug.
    """
    if _offline_mode():
        return True
    try:
        HfApi().model_info(model, timeout=_HUB_PROBE_TIMEOUT)
    except _HUB_ANSWERED_ERRORS:
        return False
    except _UNREACHABLE_ERRORS as error:
        logging.info("Hub lookup for %s failed: %s", model, error)
        return True
    return False


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

    The repository id comes back unchanged unless all four hold: no revision
    was requested, the cache reference for the default revision is empty,
    exactly one complete snapshot is present, and the Hub can no longer be
    asked. Only then is that snapshot directory returned, which is the one
    situation where the engine's own lookup is certain to produce the hashless
    path and crash.

    Every other branch returns the input, because substituting a snapshot that
    nothing ties to the requested revision would load different weights without
    saying so, which is worse than the crash this module prevents.
    """
    if os.path.exists(model):
        return model

    if revision is not None:
        return model

    repo_dir = os.path.join(
        _hub_cache_dir(), repo_folder_name(repo_id=model, repo_type="model")
    )
    ref_file = os.path.join(repo_dir, "refs", DEFAULT_REVISION)

    if not _ref_is_empty(ref_file):
        return model

    snapshot = _sole_complete_snapshot(repo_dir)
    if snapshot is None:
        return model

    if not _hub_is_unreachable(model):
        return model

    logging.info(
        "Cache reference %s for %s is empty and the Hub is unreachable; "
        "using the local snapshot %s",
        ref_file,
        model,
        snapshot,
    )
    return snapshot
