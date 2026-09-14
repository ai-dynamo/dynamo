# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton QA model repositories, downloaded from Artifactory.

Each repository is published per build as
``triton/models/<MODEL_TYPE>/<version>/<repo>/<repo>-<version>-<build>.tar``,
where ``MODEL_TYPE`` is the ``<GPU>-<host>-<sm>`` folder and ``<version>`` pairs
a Triton release with its core version (``26.07-2.71.0``), so the release alone
selects it. The repositories hold GPU-architecture-specific TensorRT plans, so
``MODEL_TYPE`` must match the GPU the models will run on: it comes from the live
GPU, and a GPU with no published set is an error rather than a guess. The newest
tar in each repository folder is the one taken.

Every tar carries a top-level ``<repo>/`` directory, so extracting them all into
one root produces the layout ``qa/L0_infer/test.sh`` expects under ``DATADIR``.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

ARTIFACTORY_URL = "https://artifactory.nvidia.com/artifactory"
ARTIFACTORY_REPO = "sw-dl-triton-generic-local"
MODEL_REPOS = ("qa_model_repository", "qa_ensemble_model_repository")

# GPU name fragment -> (folder prefix, compute capability). L4, L40 and L40S are
# all compute capability 8.9 and share the set Triton CI builds on its L40
# runners, so the "L4" fragment covers all three.
_GPU_MODEL_TYPES = (
    ("A100", "A100", "8.0"),
    ("H100", "H100", "9.0"),
    ("H200", "H100", "9.0"),
    ("L4", "L40", "8.9"),
    ("B200", "B200", "10.0"),
)

_HOST_ARCHES = {"x86_64": "x86", "aarch64": "sbsa"}


def resolve_model_type() -> str:
    """Map GPU 0 and the host arch to the ``<GPU>-<x86|sbsa>-<sm>`` folder name.

    The repositories hold TensorRT plans built for one GPU architecture, so a GPU
    with no published set is an error: the plans of another architecture cannot
    serve.
    """
    host = _HOST_ARCHES.get(platform.machine())
    if host is None:
        raise ValueError(f"No QA model sets for host arch {platform.machine()}")

    try:
        completed = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Could not ask nvidia-smi for GPU 0: {error}") from None

    lines = completed.stdout.strip().splitlines()
    name = lines[0] if lines else ""
    for fragment, gpu, compute_capability in _GPU_MODEL_TYPES:
        if fragment in name:
            model_type = f"{gpu}-{host}-{compute_capability}"
            logger.info("Detected GPU MODEL_TYPE=%s", model_type)
            return model_type

    raise ValueError(
        f"No QA model set for GPU {name!r}; published sets cover "
        f"{sorted({gpu for _, gpu, _ in _GPU_MODEL_TYPES})}"
    )


def download_qa_models(
    *,
    upstream_version: str,
    output_root: Path,
    user: str,
    token: str,
    model_type: str,
) -> Path:
    """Extract the newest tar of every model repository into ``output_root``.

    ``upstream_version`` is the Triton release the models were built for
    (``"26.07"``), which selects their version folder. Returns ``output_root``
    for use as ``DATADIR``.
    """
    _reject_unsafe_root(output_root)

    with _artifactory_session(user, token) as session:
        version = _version_folder(session, model_type, upstream_version)
        logger.info("Using version %s (MODEL_TYPE=%s)", version, model_type)

        # Rebuilt from scratch so a stale repository cannot stand in for a
        # download that produced nothing.
        shutil.rmtree(output_root, ignore_errors=True)
        output_root.mkdir(parents=True)

        for repo in MODEL_REPOS:
            folder = f"triton/models/{model_type}/{version}/{repo}"
            tars = [
                name
                for name, is_folder in _list_children(session, folder)
                if not is_folder and name.endswith(".tar")
            ]
            # test.sh reads every repository, so a missing one surfaces here
            # rather than hundreds of models into the run.
            if not tars:
                raise RuntimeError(f"No .tar under {folder}")
            tar_name = max(tars, key=_natural_sort_key)
            url = f"{ARTIFACTORY_URL}/{ARTIFACTORY_REPO}/{folder}/{tar_name}"
            _download_and_extract(session, url, output_root)

    # Each tar carries a top-level <repo>/, so this also catches one that
    # extracted nothing.
    missing = [repo for repo in MODEL_REPOS if not (output_root / repo).is_dir()]
    if missing:
        raise RuntimeError(f"{output_root} is missing model repositories: {missing}")

    logger.info(
        "QA models ready under %s: %s",
        output_root,
        sorted(path.name for path in output_root.iterdir()),
    )
    return output_root


def _artifactory_session(user: str, token: str) -> requests.Session:
    session = requests.Session()
    session.auth = (user, token)
    # The redirect header streams tars through Artifactory instead of a
    # CloudFront 302 that the CI runner egress cannot reach.
    session.headers["X-JFrog-Download-Redirect-To"] = "None"
    retry = Retry(
        total=3,
        backoff_factor=5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _list_children(session: requests.Session, folder: str) -> list[tuple[str, bool]]:
    """``(name, is_folder)`` for each child of an Artifactory folder."""
    response = session.get(
        f"{ARTIFACTORY_URL}/api/storage/{ARTIFACTORY_REPO}/{folder}/",
        timeout=60,
    )
    if not response.ok:
        # An empty folder and a failed request (auth, network, rate limit) both
        # yield no names, so the failure is called out here rather than surfacing
        # later as a missing version or tar.
        logger.warning(
            "Failed to list '%s' (HTTP %s); treating as empty",
            folder,
            response.status_code,
        )
        return []

    children = []
    for child in response.json().get("children") or []:
        name = child.get("uri", "").lstrip("/")
        if name:
            children.append((name, bool(child.get("folder"))))
    return children


def _version_folder(
    session: requests.Session, model_type: str, upstream_version: str
) -> str:
    """Folder holding a release's models, named ``<release>-<core version>``.

    A release resolves to one folder (``26.07`` -> ``26.07-2.71.0``). A respin
    adds another, and the newest published (non-dev) one wins.
    """
    prefix = f"{upstream_version}-"
    versions = [
        name
        for name, is_folder in _list_children(session, f"triton/models/{model_type}")
        if is_folder and name.startswith(prefix) and not name.endswith("dev")
    ]
    if not versions:
        raise RuntimeError(
            f"No published version for {model_type}/{upstream_version}-*"
        )
    return max(versions, key=_natural_sort_key)


def _natural_sort_key(name: str) -> list[tuple[int, int, str]]:
    """Sort key ordering embedded numbers numerically, like ``sort -V``."""
    return [
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in re.split(r"(\d+)", name)
    ]


def _download_and_extract(
    session: requests.Session, url: str, output_root: Path
) -> None:
    logger.info("Downloading %s", url)
    started = time.monotonic()
    # Staged inside output_root so the multi-GB tar lands on the same (usually
    # larger) filesystem as the models rather than in the default temp dir.
    staged_fd, staged_name = tempfile.mkstemp(dir=output_root, suffix=".tar")
    staged_path = Path(staged_name)
    try:
        # open() takes ownership of the descriptor and closes it.
        with open(staged_fd, "wb") as staged:
            # Connect timeout, then the silence between chunks that means the
            # stream is dead rather than slow.
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                written = 0
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    written += staged.write(chunk)

        logger.info(
            "Fetched %.1f GiB in %.1f min; extracting",
            written / 1024**3,
            (time.monotonic() - started) / 60,
        )
        with tarfile.open(staged_path) as tar:
            tar.extractall(output_root, filter="data")
    finally:
        staged_path.unlink(missing_ok=True)


def _reject_unsafe_root(output_root: Path) -> None:
    """Refuse a root that must not be wiped: relative, filesystem root, or ``..``."""
    if (
        not output_root.is_absolute()
        or output_root == Path(output_root.anchor)
        or ".." in output_root.parts
    ):
        raise ValueError(f"Refusing unsafe output root: {str(output_root)!r}")
