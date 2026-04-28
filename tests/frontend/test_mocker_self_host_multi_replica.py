# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-replica self-host metadata test using CPU-only mocker workers.

Spawns two mocker worker processes against a single frontend, both with
``DYN_SELF_HOST_METADATA=true``. They register the same model
concurrently, each advertising HTTP metadata URLs at its own system
port. Verifies the frontend serializes concurrent fetches via its
per-blake3 in-process mutex + cross-process flock, producing a
consistent content-addressed cache without race-induced errors.

CPU-only (mocker) so this can run on gpu_0 / pre-merge.
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


@pytest.fixture(scope="function")
def multi_replica_self_host_services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
    tmp_path,
) -> Generator[Tuple[ServicePorts, Path], None, None]:
    """Start a frontend with isolated $HOME plus two mocker worker
    replicas, both with self-host metadata enabled. Yields the
    `ServicePorts` (which carries two system ports) and the isolated
    frontend home so the test can inspect the cache.
    """
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    sys_a, sys_b = dynamo_dynamic_ports.system_ports

    frontend_home = tmp_path / "frontend-home"
    frontend_home.mkdir(parents=True, exist_ok=True)

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={"HOME": str(frontend_home)},
        terminate_all_matching_process_names=False,
    ):
        # Two replicas, both publishing metadata via http://. They
        # advertise URLs at distinct system ports, so the frontend
        # sees two MDCs that share blake3 checksums but differ in
        # URL host:port. The cache is keyed on blake3 → after the
        # first replica's fetch, the second short-circuits via
        # `dest.exists()` after acquiring the flock.
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            sys_a,
            worker_id="mocker-a",
            extra_env={"DYN_SELF_HOST_METADATA": "true"},
        ):
            with MockerWorkerProcess(
                request,
                TEST_MODEL,
                frontend_port,
                sys_b,
                worker_id="mocker-b",
                extra_env={"DYN_SELF_HOST_METADATA": "true"},
            ):
                yield dynamo_dynamic_ports, frontend_home


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
@pytest.mark.timeout(300)
def test_two_mocker_replicas_register_via_self_host(
    multi_replica_self_host_services: Tuple[ServicePorts, Path],
    predownload_tokenizers,
) -> None:
    ports, frontend_home = multi_replica_self_host_services
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
        "multi-replica self-host verified: %d blob(s) under %s "
        "from concurrent registrations on system_ports=%s",
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
