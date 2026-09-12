# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plumbing shared by the Triton QA suites that run against the Dynamo runtime.

Every ``qa/<suite>/test.sh`` in ``triton-inference-server/server`` is driven the
same way: fetch the ``qa`` tree, run the script, then move its logs, results and
core dumps where CI collects them. What stays with a test module is the
suite-specific part -- which models the run needs, and which environment
``test.sh`` reads.

Environment:
    GITHUB_WORKSPACE: set under GitHub Actions; artifacts go to the
        ``test-results/logs`` tree that .github/actions/pytest-local uploads.
        Off CI they go to a temp directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

SERVER_REPO_URL = "https://github.com/triton-inference-server/server.git"

# test.sh writes all of these into its own directory.
_ARTIFACT_PATTERNS = ("*.log", "core*", "test_results.txt")


def fetch_qa_suite(destination: Path, suite: str, ref: str) -> Path:
    """Sparse-checkout the server repo's ``qa`` tree, returning ``qa/<suite>``.

    The scripts source their siblings under ``qa/common`` and write into their
    own directory, so the whole tree is fetched into a writable location. ``ref``
    is a branch or tag, which must exist on the remote.
    """
    logger.info("Fetching %s at %s into %s", SERVER_REPO_URL, ref, destination)

    # Shallow and blobless, so only the qa tree's blobs are fetched: --sparse
    # clones the root files alone, then sparse-checkout adds qa.
    for command in (
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            ref,
            SERVER_REPO_URL,
            str(destination),
        ],
        ["git", "-C", str(destination), "sparse-checkout", "set", "qa"],
    ):
        subprocess.run(command, check=True)

    suite_dir = destination / "qa" / suite
    if not (suite_dir / "test.sh").is_file():
        raise FileNotFoundError(f"{SERVER_REPO_URL}@{ref} has no qa/{suite}/test.sh")
    return suite_dir


def run_qa_test_sh(
    suite_dir: Path,
    *,
    env: dict[str, str],
    log_path: Path,
    args: Sequence[str] = (),
) -> int:
    """Run a suite's test.sh to completion and return its exit code.

    Waiting in short slices keeps core dumps swept while the suite runs. The
    script gets its own process group, so an interrupt reaches the frontend,
    worker, etcd and NATS it starts in the background; bash gives its background
    jobs SIGINT ignored, which is why they are signalled as a group.
    """
    logger.info(
        "Running %s test.sh in %s (log %s)", suite_dir.name, suite_dir, log_path
    )
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            ["bash", "-ex", "./test.sh", *args],
            cwd=suite_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    return process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    _sweep_core_dumps(suite_dir)
        except BaseException:
            _terminate_group(process)
            raise


def _terminate_group(process: subprocess.Popen) -> None:
    """Stop the suite's process group, escalating to SIGKILL if it survives."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            return
        except OSError as error:
            logger.warning("Could not signal the test.sh group: %s", error)
            return
        try:
            process.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            logger.warning("test.sh group outlived %s", number.name)


def artifact_dir(name: str) -> Path:
    """Create and return a directory for diagnostics that outlive the run.

    Under GitHub Actions this is the ``test-results/logs`` tree that
    .github/actions/pytest-local uploads. Off CI it is a temp directory, which
    keeps output out of the working tree.
    """
    workspace = os.environ.get("GITHUB_WORKSPACE")
    root = (
        Path(workspace) / "test-results" / "logs"
        if workspace
        else Path(tempfile.gettempdir()) / "dynamo_tests"
    )
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def collect_artifacts(suite_dir: Path, destination: Path) -> None:
    """Move test.sh's logs, results, and core dumps where CI uploads them."""
    _sweep_core_dumps(suite_dir)
    for pattern in _ARTIFACT_PATTERNS:
        for path in sorted(suite_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                shutil.move(str(path), str(destination / path.name))
            except OSError:
                logger.warning("Could not collect %s", path, exc_info=True)


def _sweep_core_dumps(directory: Path) -> None:
    """Keep the first core dump as ``core`` and drop the rest.

    A crash loop leaves one ``core.<pid>`` per process, each the size of the
    worker's address space, so unswept they exhaust the workspace disk and only
    the first is worth uploading. The naming cannot be changed from the
    container: ``/proc/sys`` is read-only and ``kernel.core_uses_pid`` is not
    namespaced, so ``sysctl -w kernel.core_uses_pid=0`` exits 0 and changes
    nothing.
    """
    keeper = directory / "core"
    for dump in sorted(directory.glob("core.*")):
        try:
            if keeper.exists():
                dump.unlink()
            else:
                dump.replace(keeper)
        except OSError:
            logger.debug("Could not sweep %s", dump, exc_info=True)
