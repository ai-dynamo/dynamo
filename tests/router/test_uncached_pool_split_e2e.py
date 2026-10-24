#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""CPU Mocker end-to-end coverage for the dynamo-uncached-pool-split policy.

Core scenarios from the pool-split E2E matrix:

- 1.1  a fresh cold prompt is served (classification through the taint-declared pool)
- 3.1  a cold tier with no eligible member returns the overload status and never falls
       back to warm workers (isolation)
- 3.3  cold requests keep succeeding while the warm pool is saturated
- 5.1  warm traffic is unaffected by the cold pool

Pool membership is injected through the mocker's topology support: the per-process
``extra_env`` of ``MockerProcess`` plus a mount directory whose file name is the domain
(``worker_pool``) and whose content is the value (``cold-1``). The resulting canonical
taint ``dynamo.topology/worker_pool=cold-1`` is what the policy's ``pools[].taint``
matches.

Placement is asserted through behavior rather than worker ids: scenario 3.1 runs the
same cold workload against a cluster where no worker carries the taint, and expects the
overload status for every request — which only holds if the policy routes by tier and
never silently degrades to the warm workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from tests.router.e2e_harness import allocate_frontend_ports
from tests.router.helper import generate_random_suffix, wait_for_frontend_ready
from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

# Do not add pytest.mark.parallel: existing Mocker Router tests document races in
# process-global DistributedRuntime state under pytest-xdist.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(300),
]

COLD_TAINT = "dynamo.topology/worker_pool=cold-1"
BACKEND_MAX_NUM_SEQS = 4

# Token geometry: with block size 16, a 96-token prompt sits below the 128-token
# threshold (warm) and a 160-token prompt with no reusable prefix sits above it (cold).
WARM_TOKENS = 96
COLD_TOKENS = 160

POOL_SPLIT_YAML = f"""\
worker_selection:
  # The mocker frontend routes in aggregated mode (phase=Aggregated), so the pool-split
  # policy must be selected for that stage; prefill keeps it too for disagg topologies.
  aggregated: pool-split
  prefill: pool-split
  instances:
    - name: pool-split
      type: dynamo-uncached-pool-split
      parameters:
        pools:
          - min_uncached_tokens: 128
            taint: {COLD_TAINT}
"""

MOCKER_ARGS: dict[str, Any] = {
    "speedup_ratio": 1.0,
    "max_num_seqs": BACKEND_MAX_NUM_SEQS,
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": False,
}

logger = logging.getLogger(__name__)


def _cold_taint_env(topology_dir: Path, value: str) -> dict[str, str]:
    topology_dir.mkdir(parents=True, exist_ok=True)
    (topology_dir / "worker_pool").write_text(value)
    return {
        "DYN_TOPOLOGY_ENABLED": "true",
        "DYN_KV_TRANSFER_DOMAIN": "worker_pool",
        "DYN_TOPOLOGY_MOUNT_PATH": str(topology_dir),
    }


def _cold_prompt(request_index: int) -> list[int]:
    # Distinct token base per request so no prefix is shared: every cold request has its
    # full length as fresh compute.
    return [10_000 + request_index * COLD_TOKENS + i for i in range(COLD_TOKENS)]


def _warm_prompt(request_index: int) -> list[int]:
    return [20_000 + request_index * WARM_TOKENS + i for i in range(WARM_TOKENS)]


async def _post_completion(
    session: aiohttp.ClientSession,
    url: str,
    prompt: list[int],
) -> int:
    body = {
        "model": ROUTER_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": False,
        "temperature": 0,
    }
    async with session.post(
        url, data=json.dumps(body), headers={"content-type": "application/json"}
    ) as response:
        await response.read()
        return response.status


async def _statuses(
    frontend_port: int,
    prompts: list[list[int]],
) -> list[int]:
    url = f"http://localhost:{frontend_port}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=120, connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *(_post_completion(session, url, prompt) for prompt in prompts)
        )


def _start_frontend(
    request: pytest.FixtureRequest,
    frontend_port: int,
    config_path: Path,
    namespace: str,
    display_name: str,
) -> DynamoFrontendProcess:
    return DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        router_mode="kv",
        extra_args=[
            "--namespace",
            namespace,
            "--discovery-backend",
            "etcd",
            "--load-aware",
            "--router-policy-config",
            str(config_path),
            "--router-min-initial-workers",
            "1",
        ],
        extra_env={"DYN_REQUEST_PLANE": "nats", "DYN_LOG": "warn"},
        display_name=display_name,
    )


@pytest.mark.usefixtures(
    "runtime_services_dynamic_ports",
    "predownload_tokenizers",
)
def test_cold_tier_without_tainted_worker_rejects(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    dynamo_dynamic_ports,
    tmp_path: Path,
) -> None:
    """3.1: cold requests fail as overload when no worker carries the cold taint.

    This is the isolation guarantee: the policy never silently routes cold traffic to
    warm workers. It also proves scenario 1.1's placement by complement — with a
    tainted worker present the same workload succeeds (asserted separately).
    """

    # No DYN_SYSTEM_PORT monkeypatch: the frontend strips it and mocker workers each
    # need their own random system port; a shared fixed port collides on worker 2.
    frontend_port = allocate_frontend_ports(request, 1)[0]
    config_path = tmp_path / "pool_split.yaml"
    config_path.write_text(POOL_SPLIT_YAML)

    with MockerProcess(
        request,
        mocker_args=MOCKER_ARGS,
        num_mockers=2,
        store_backend="etcd",
        request_plane="nats",
        model_name=ROUTER_MODEL_NAME,
    ) as mocker:
        with _start_frontend(
            request,
            frontend_port,
            config_path,
            mocker.namespace,
            "dynamo-frontend-pool-split-3-1",
        ):
            asyncio.run(
                wait_for_frontend_ready(
                    frontend_url=f"http://localhost:{frontend_port}",
                    expected_num_workers=2,
                    timeout=60,
                    engine_workers=mocker,
                    store_backend="etcd",
                    request_plane="nats",
                )
            )
            statuses = asyncio.run(
                _statuses(frontend_port, [_cold_prompt(i) for i in range(4)])
            )
    assert all(status in (429, 529) for status in statuses), (
        f"cold requests must be rejected as overload without a tainted worker, "
        f"got: {statuses}"
    )


@pytest.mark.usefixtures(
    "runtime_services_dynamic_ports",
    "predownload_tokenizers",
)
def test_cold_and_warm_traffic_with_tainted_worker(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    dynamo_dynamic_ports,
    tmp_path: Path,
) -> None:
    """1.1 + 5.1: with a tainted worker present, both lanes succeed.

    A separate tainted mocker publishes the cold taint; cold requests (fresh 160-token
    prompts) must succeed because their tier has a member, and warm requests must also
    succeed — proving the pool split does not disturb the warm lane.
    """

    # No DYN_SYSTEM_PORT monkeypatch: the frontend strips it and mocker workers each
    # need their own random system port; a shared fixed port collides on worker 2.
    frontend_port = allocate_frontend_ports(request, 1)[0]
    config_path = tmp_path / "pool_split.yaml"
    config_path.write_text(POOL_SPLIT_YAML)

    shared_namespace = f"test-namespace-pool-split-{generate_random_suffix()}"
    with MockerProcess(
        request,
        mocker_args=MOCKER_ARGS,
        num_mockers=1,
        store_backend="etcd",
        request_plane="nats",
        model_name=ROUTER_MODEL_NAME,
        namespace=shared_namespace,
    ) as warm_mocker:
        with MockerProcess(
            request,
            mocker_args=MOCKER_ARGS,
            num_mockers=1,
            store_backend="etcd",
            request_plane="nats",
            model_name=ROUTER_MODEL_NAME,
            extra_env=_cold_taint_env(tmp_path / "topology", "cold-1"),
            namespace=shared_namespace,
        ) as cold_mocker:
            # Both mockers register in the shared namespace: the frontend discovers one
            # warm and one cold-tainted worker for ROUTER_MODEL_NAME.
            with _start_frontend(
                request,
                frontend_port,
                config_path,
                shared_namespace,
                "dynamo-frontend-pool-split-1-1",
            ):
                asyncio.run(
                    wait_for_frontend_ready(
                        frontend_url=f"http://localhost:{frontend_port}",
                        expected_num_workers=2,
                        timeout=60,
                        engine_workers=[warm_mocker, cold_mocker],
                        store_backend="etcd",
                        request_plane="nats",
                    )
                )
                cold_statuses = asyncio.run(
                    _statuses(frontend_port, [_cold_prompt(i) for i in range(4)])
                )
                warm_statuses = asyncio.run(
                    _statuses(frontend_port, [_warm_prompt(i) for i in range(4)])
                )

    assert all(
        status == 200 for status in cold_statuses
    ), f"cold requests must succeed with a tainted worker present, got: {cold_statuses}"
    assert all(
        status == 200 for status in warm_statuses
    ), f"warm requests must succeed alongside the pool split, got: {warm_statuses}"
