# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton's QA ``L0_infer`` suite, run against the Dynamo Triton runtime.

``qa/L0_infer/test.sh`` from ``triton-inference-server/server`` drives the whole
suite: with ``SERVER_LAUNCH_MODE=dynamo`` it brings up the Dynamo frontend and
Triton worker for every backend and batching mode, infers against the QA models,
and exits non-zero on the first failure. This module supplies what the script
needs from CI — a checkout of the ``qa`` tree, a ``DATADIR`` of QA model
repositories, and a core-dump sweeper — and reports the run as one pytest case.
qa_utils.py holds the parts that are not specific to L0_infer, including where
artifacts land.

The suite takes hours and holds a GPU for its entire run, so it carries its own
``triton_qa`` marker and is selected on its own rather than with the rest of the
Triton pool. Its only time bound is the CI step's ``gpu_test_timeout_minutes``.

Environment:
    TRITON_SERVER_BRANCH_NAME: server branch or tag holding the QA tree
        (default ``main``), named as the server's own qa/L2_build_presets does.
    TRITON_ARTIFACTORY_USER, TRITON_ARTIFACTORY_TOKEN: credentials for the QA
        model download. Without them the test skips, unless ``DATADIR`` is staged.
    DATADIR: pre-staged QA model repositories, one directory per repository;
        skips the download entirely.
    NVIDIA_TRITON_SERVER_VERSION: Triton release the runtime image ships
        (``"26.07"``). Picks the QA model set and is read by the worker.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest
from qa_models import MODEL_REPOS, download_qa_models, resolve_model_type
from qa_utils import artifact_dir, collect_artifacts, fetch_qa_suite, run_qa_test_sh

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.triton,
    pytest.mark.triton_qa,
    pytest.mark.gpu_1,
    pytest.mark.slow,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
]

_DEFAULT_SERVER_BRANCH = "main"


@pytest.fixture(scope="session")
def triton_version() -> str:
    """Triton release of the runtime image, e.g. ``26.07``."""
    version = os.environ.get("NVIDIA_TRITON_SERVER_VERSION", "").strip()
    if not version:
        pytest.skip(
            "NVIDIA_TRITON_SERVER_VERSION is unset, so the QA model set for this "
            "Triton release cannot be selected"
        )
    return version


@pytest.fixture(scope="session")
def require_gpu() -> None:
    """Fail unless a GPU is visible, before the download and checkout run.

    The Triton runtime image ships without torch, so CI leaves this check to the
    test (see verify_gpu in .github/workflows/shared-test.yml). A staged DATADIR
    gets the models onto a GPU-less runner without ever querying a GPU, and
    test.sh would then fail hundreds of models in.
    """
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        pytest.fail(f"L0_infer needs a GPU, and nvidia-smi did not run: {error}")

    names = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if query.returncode != 0 or not names:
        pytest.fail(
            f"L0_infer needs a GPU, and nvidia-smi found none: {query.stderr.strip()}"
        )

    logger.info("Running on %d GPU(s): %s", len(names), ", ".join(names))


@pytest.fixture(scope="session")
def qa_data_dir(triton_version: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``DATADIR`` holding the QA model repositories ``test.sh`` copies from."""
    staged = os.environ.get("DATADIR", "").strip()
    if staged:
        # An explicit DATADIR is operator intent, so a wrong one is reported
        # here instead of failing models deep inside test.sh.
        staged_dir = Path(staged)
        missing = [repo for repo in MODEL_REPOS if not (staged_dir / repo).is_dir()]
        if missing:
            pytest.fail(f"DATADIR {staged} is missing model repositories: {missing}")
        logger.info("Using staged QA model repositories under %s", staged)
        return staged_dir

    user = os.environ.get("TRITON_ARTIFACTORY_USER", "")
    token = os.environ.get("TRITON_ARTIFACTORY_TOKEN", "")
    if not user or not token:
        pytest.skip(
            "QA models require TRITON_ARTIFACTORY_USER and "
            "TRITON_ARTIFACTORY_TOKEN, or a DATADIR pointing at a staged download"
        )

    model_type = resolve_model_type()
    output_root = (
        tmp_path_factory.mktemp("qa-models") / f"{triton_version}_{model_type}"
    )
    return download_qa_models(
        upstream_version=triton_version,
        output_root=output_root,
        user=user,
        token=token,
        model_type=model_type,
    )


@pytest.fixture(scope="session")
def l0_infer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``qa/L0_infer`` of a sparse server checkout, holding the script to run."""
    branch = (
        os.environ.get("TRITON_SERVER_BRANCH_NAME", "").strip()
        or _DEFAULT_SERVER_BRANCH
    )
    return fetch_qa_suite(tmp_path_factory.mktemp("server-qa"), "L0_infer", branch)


@pytest.mark.usefixtures("require_gpu")
def test_l0_infer(
    request: pytest.FixtureRequest,
    triton_version: str,
    qa_data_dir: Path,
    l0_infer_dir: Path,
) -> None:
    """Run the QA L0_infer suite with the Dynamo Triton runtime as the server."""
    env = os.environ.copy()
    env.pop("TRITON_ARTIFACTORY_USER", None)
    env.pop("TRITON_ARTIFACTORY_TOKEN", None)
    env.update(
        {
            "NVIDIA_TRITON_SERVER_VERSION": triton_version,
            "DATADIR": str(qa_data_dir),
            "SERVER_LAUNCH_MODE": "dynamo",
        }
    )

    artifacts = artifact_dir(request.node.name)
    log_path = artifacts / "test.sh.log.txt"

    returncode = run_qa_test_sh(
        l0_infer_dir, env=env, log_path=log_path, args=[triton_version]
    )
    collect_artifacts(l0_infer_dir, artifacts)

    assert (
        returncode == 0
    ), f"L0_infer exited {returncode}; see {log_path} and artifacts under {artifacts}"
