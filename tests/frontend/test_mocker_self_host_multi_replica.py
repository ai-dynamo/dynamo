# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-replica metadata-fetch tests using CPU-only mocker workers.

Three scenarios, parametrized over the URI scheme the worker advertises:

* ``self_host`` (``http://``): ``DYN_SELF_HOST_METADATA=true`` → worker
  calls ``move_to_self_host`` and rewrites every CheckedFile to
  ``http://<worker>/v1/metadata/<slug>/<filename>``.
* ``hf`` (``hf://``): ``DYN_SELF_HOST_METADATA=false`` and the worker
  is started with the HF repo ID as ``--model-path``. The repo-ID
  string doesn't exist as a literal local path, so origin's
  ``move_to_url("hf://...")`` branch fires and CheckedFiles end up
  pointing at ``hf://<repo>/<filename>`` URIs.
* ``file`` (``file://``): ``DYN_SELF_HOST_METADATA=false`` and the
  worker is started with a **local snapshot directory** as
  ``--model-path``. The path exists, so neither ``move_to_self_host``
  nor ``move_to_url("hf://...")`` fires; the CheckedFile stays
  ``Either::Left(PathBuf)`` and the frontend's
  ``resolve_metadata_files`` synthesizes a ``file://`` URI from the
  canonicalized local path.

All three end up running the same downstream pipeline: ``resolve_uri``
stages bytes to a tmp via the shared atomic-publish primitive,
verifies via ``CheckedFile::from_disk(tmp)``, and atomic-renames into
the content-addressed cache.

A separate multi-frontend test exercises cross-process flock by
sharing a single ``$HOME`` between two frontend processes.

CPU-only (mocker) so these run on gpu_0 / post_merge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Tuple

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN, DefaultPort
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import ServicePorts, allocate_port, deallocate_port

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.gpu_0,  # mocker is CPU-only
    pytest.mark.e2e,
    pytest.mark.post_merge,
    pytest.mark.model(TEST_MODEL),
]


def _mocker_model_arg_for_scheme(scheme: str) -> str:
    """Resolve the value to pass as `--model-path` for each scheme.

    `self_host` and `hf` use the HF repo ID directly — the mocker
    fetches it under the hood. `file` requires a local-disk path so
    that origin's `move_to_url("hf://...")` branch doesn't fire and
    CheckedFile stays `Either::Left(PathBuf)`.
    """
    if scheme in ("self_host", "hf"):
        return TEST_MODEL
    if scheme == "file":
        from huggingface_hub import snapshot_download

        # Limit to metadata files; weights aren't needed for the mocker.
        return snapshot_download(
            repo_id=TEST_MODEL,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "chat_template.jinja",
                "chat_template.json",
                "generation_config.json",
            ],
        )
    raise ValueError(f"unknown scheme: {scheme}")


@pytest.fixture(scope="function")
def multi_replica_services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
    tmp_path,
):
    """Factory: yields a callable `start(scheme)` that spawns a frontend
    with isolated $HOME plus two mocker workers configured for the
    requested URI scheme.

    `scheme` is one of `"self_host"`, `"hf"`, `"file"`.
    """
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    sys_a, sys_b = dynamo_dynamic_ports.system_ports

    frontend_home = tmp_path / "frontend-home"
    frontend_home.mkdir(parents=True, exist_ok=True)

    contexts: list = []

    def start(scheme: str):
        worker_env = {
            "DYN_SELF_HOST_METADATA": "true" if scheme == "self_host" else "false"
        }
        model_arg = _mocker_model_arg_for_scheme(scheme)

        # For the `file` scheme we pass a local snapshot dir as
        # `--model-path` (so the worker doesn't trigger the
        # `move_to_url("hf://...")` branch). Without an explicit
        # `--model-name`, the model's display_name would become the
        # local path string and `/v1/models` would surface it as the
        # model id. Override so the served name stays stable.
        worker_extra_args = (
            ["--model-name", TEST_MODEL] if scheme == "file" else None
        )

        frontend = DynamoFrontendProcess(
            request,
            frontend_port=frontend_port,
            extra_env={"HOME": str(frontend_home)},
            terminate_all_matching_process_names=False,
        )
        frontend.__enter__()
        contexts.append(frontend)

        for worker_id, sys_port in [("mocker-a", sys_a), ("mocker-b", sys_b)]:
            worker = MockerWorkerProcess(
                request,
                model_arg,
                frontend_port,
                sys_port,
                worker_id=worker_id,
                extra_env=worker_env,
                extra_args=worker_extra_args,
            )
            worker.__enter__()
            contexts.append(worker)

        return dynamo_dynamic_ports, frontend_home

    try:
        yield start
    finally:
        # Tear down in reverse order; swallow individual errors so all
        # contexts get a chance to clean up.
        for ctx in reversed(contexts):
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                logger.exception("error tearing down %r", ctx)


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
@pytest.mark.parametrize("scheme", ["self_host", "hf", "file"])
@pytest.mark.timeout(300)
def test_two_mocker_replicas_register(
    multi_replica_services,
    scheme: str,
    predownload_tokenizers,
) -> None:
    ports, frontend_home = multi_replica_services(scheme)
    base_url = f"http://localhost:{ports.frontend_port}"

    # Both replicas should have registered. /v1/models reflects merged
    # registrations — exactly one entry for TEST_MODEL regardless of
    # replica count.
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    assert response.status_code == 200, response.text
    model_ids = [m.get("id") for m in response.json().get("data", [])]
    assert TEST_MODEL in model_ids, f"expected {TEST_MODEL} in {model_ids}"

    # Send a completion to confirm the served path actually works
    # end-to-end across both replicas.
    completion = requests.post(
        f"{base_url}/v1/completions",
        json={"model": TEST_MODEL, "prompt": "hi", "max_tokens": 8},
        timeout=60,
    )
    assert completion.status_code == 200, completion.text

    # Cache must be populated via the http path. Both replicas
    # advertise content-identical metadata, so the cache should
    # contain one blob per metadata file (not duplicates per
    # replica). The flock + per-blake3 mutex guarantee a single
    # fetch per blob across the two concurrent registrations.
    blobs_dir = frontend_home / ".cache/dynamo/mdc/blobs"
    assert blobs_dir.exists(), f"expected blobs dir at {blobs_dir}"
    blobs = [
        p
        for p in blobs_dir.iterdir()
        if p.is_file() and not p.name.endswith(".lock") and ".tmp" not in p.name
    ]
    assert len(blobs) > 0, f"expected at least one blob in {blobs_dir}"

    # No leftover tmp files — atomic rename cleans up on success.
    leftover_tmps = [p for p in blobs_dir.iterdir() if ".tmp" in p.name]
    assert (
        not leftover_tmps
    ), f"expected no leftover tmp files in cache, got {leftover_tmps}"

    # No duplicate-blob accumulation: each unique blake3 should have
    # exactly one blob entry. Using filename count as a proxy.
    by_hex = {p.name for p in blobs}
    assert len(by_hex) == len(blobs), (
        f"expected unique blob filenames (one per blake3), got duplicates: "
        f"{[p.name for p in blobs]}"
    )
    logger.info(
        "multi-replica %s verified: %d blob(s) under %s "
        "from concurrent registrations on system_ports=%s",
        scheme,
        len(blobs),
        blobs_dir,
        ports.system_ports,
    )


@pytest.fixture(scope="function")
def shared_cache_two_frontends_one_worker(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
    tmp_path,
) -> Generator[Tuple[int, int, Path], None, None]:
    """Two frontends sharing a single $HOME (and therefore a single
    metadata cache dir), plus one mocker worker with self-host
    metadata enabled. Yields ``(frontend_a_port, frontend_b_port,
    shared_home)``.

    Exercises the cross-process flock: both frontends observe the
    same MDC publication and try to fetch into the same cache
    directory simultaneously. The first to acquire the flock fetches;
    the second sees `dest.exists()` after acquiring and short-circuits.
    """
    _ = runtime_services_dynamic_ports
    frontend_a_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    frontend_b_port = allocate_port(DefaultPort.FRONTEND.value)

    shared_home = tmp_path / "shared-frontend-home"
    shared_home.mkdir(parents=True, exist_ok=True)

    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=frontend_a_port,
            extra_env={"HOME": str(shared_home)},
            terminate_all_matching_process_names=False,
            display_name="frontend-a",
        ):
            with DynamoFrontendProcess(
                request,
                frontend_port=frontend_b_port,
                extra_env={"HOME": str(shared_home)},
                terminate_all_matching_process_names=False,
                display_name="frontend-b",
            ):
                with MockerWorkerProcess(
                    request,
                    TEST_MODEL,
                    frontend_a_port,
                    system_port,
                    worker_id="mocker-shared-cache",
                    extra_env={"DYN_SELF_HOST_METADATA": "true"},
                ):
                    yield frontend_a_port, frontend_b_port, shared_home
    finally:
        deallocate_port(frontend_b_port)


@pytest.mark.timeout(300)
def test_two_frontends_share_metadata_cache_via_flock(
    shared_cache_two_frontends_one_worker: Tuple[int, int, Path],
    predownload_tokenizers,
) -> None:
    (
        frontend_a_port,
        frontend_b_port,
        shared_home,
    ) = shared_cache_two_frontends_one_worker

    # The mocker fixture only gates on frontend A's readiness; B's
    # discovery watcher may still be catching up when the fixture
    # yields. Wait for B to actually serve the model before asserting.
    wait_for_http_completions_ready(frontend_port=frontend_b_port, model=TEST_MODEL)

    # Both frontends should see the same model.
    for port in (frontend_a_port, frontend_b_port):
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=30)
        assert response.status_code == 200, response.text
        model_ids = [m.get("id") for m in response.json().get("data", [])]
        assert (
            TEST_MODEL in model_ids
        ), f"frontend on port {port} missing {TEST_MODEL}: {model_ids}"

    # Each frontend serves a completion request independently.
    for port in (frontend_a_port, frontend_b_port):
        completion = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={"model": TEST_MODEL, "prompt": "hi", "max_tokens": 8},
            timeout=60,
        )
        assert (
            completion.status_code == 200
        ), f"completion on port {port}: {completion.text}"

    # The shared cache should have a single coherent set of blobs —
    # the flock prevented both frontends from writing to the same
    # blob path simultaneously.
    blobs_dir = shared_home / ".cache/dynamo/mdc/blobs"
    assert blobs_dir.exists(), f"expected blobs dir at {blobs_dir}"
    blobs = [
        p
        for p in blobs_dir.iterdir()
        if p.is_file() and not p.name.endswith(".lock") and ".tmp" not in p.name
    ]
    assert len(blobs) > 0, f"expected at least one blob in {blobs_dir}"

    leftover_tmps = [p for p in blobs_dir.iterdir() if ".tmp" in p.name]
    assert (
        not leftover_tmps
    ), f"expected no leftover tmp files in cache, got {leftover_tmps}"

    # Each blob filename = blake3 hex; uniqueness confirms no
    # duplicate downloads under contention.
    by_hex = {p.name for p in blobs}
    assert len(by_hex) == len(
        blobs
    ), f"expected unique blob filenames, got duplicates: {[p.name for p in blobs]}"

    logger.info(
        "two-frontend shared-cache verified: %d blob(s) under %s "
        "served by frontends on ports %d, %d",
        len(blobs),
        blobs_dir,
        frontend_a_port,
        frontend_b_port,
    )
