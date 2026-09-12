# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run shared deployment API checks against four mixed-version component pairs."""

import asyncio
import json
import logging
import os
import tomllib
import uuid
from pathlib import Path

import pytest

from tests.deploy.api_checks import check_deployment_api
from tests.deploy.dgd_utils import ManagedDeployment
from tests.deploy.n2_utils import MODELS, compatibility_spec, version_matrix
from tests.utils.client import wait_for_model_availability
from tests.utils.test_output import resolve_test_output_path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def version_pairs(request):
    frontend = request.config.getoption("--frontend-image")
    worker = request.config.getoption("--image")
    if not frontend or not worker:
        raise pytest.UsageError("N-2 requires --frontend-image and --image")
    if not request.config.getoption("--model-cache-pvc"):
        raise pytest.UsageError("N-2 requires a prepared shared --model-cache-pvc")
    line = os.environ.get("N2_RELEASE_LINE") or ".".join(
        tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"][
            "version"
        ].split(".")[:2]
    )
    releases = json.loads((ROOT / "tests/deploy/n2/releases.json").read_text())
    return version_matrix(releases, line, frontend, worker)


@pytest.fixture
async def mixed_deployment(request, version_pairs, pair_index, scenario, namespace):
    pair = version_pairs[pair_index]
    output = Path(resolve_test_output_path(request.node.name))
    output.mkdir(parents=True, exist_ok=True)
    spec = compatibility_spec(
        ROOT,
        pair,
        scenario,
        "n2-" + uuid.uuid4().hex[:10],
        request.config.getoption("--model-cache-pvc"),
        request.config.getoption("--model-cache-mount"),
    )
    spec.save(str(output / "deployment.yaml"))
    (output / "versions.json").write_text(
        json.dumps(
            {
                "pair": pair.name,
                "frontend": pair.frontend,
                "worker": pair.worker,
                "scenario": scenario,
            },
            indent=2,
        )
    )
    managed = ManagedDeployment(
        str(output),
        spec,
        namespace,
        skip_service_restart=True,
        readiness_timeout=1200,
        fail_fast_startup=True,
    )
    try:
        async with managed as deployment:
            yield deployment, output
    finally:
        if managed.cleanup_errors:
            # Stop after pytest reports this item's original failure and teardown.
            request.session.shouldstop = (
                "N-2 cleanup failed; remaining pairs are not validated"
            )


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.sglang
@pytest.mark.framework_agnostic
@pytest.mark.core
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("scenario", ["embedding", "chat"])
@pytest.mark.parametrize(
    "pair_index",
    range(4),
    ids=["old-frontend-1", "old-worker-1", "old-frontend-2", "old-worker-2"],
)
async def test_n2_compatibility(mixed_deployment, scenario):
    deployment, output = mixed_deployment
    model = MODELS[scenario][0]
    pods = await asyncio.to_thread(deployment.get_pods, ["Frontend"])
    assert len(pods["Frontend"]) == 1
    forward = await asyncio.to_thread(
        deployment.port_forward, pods["Frontend"][0], 8000
    )
    assert forward is not None, "Frontend port forwarding failed"
    base = f"http://127.0.0.1:{forward.local_port}"
    endpoint = "/v1/embeddings" if scenario == "embedding" else "/v1/chat/completions"
    payload = {"model": model, "input": "test"} if scenario == "embedding" else None
    assert await asyncio.to_thread(
        wait_for_model_availability,
        base,
        endpoint,
        model,
        logger,
        payload=payload,
    ), f"Model {model} did not become available"
    await asyncio.to_thread(
        check_deployment_api,
        base,
        model,
        scenario,
        output / "responses",
        endpoint=endpoint,
    )
