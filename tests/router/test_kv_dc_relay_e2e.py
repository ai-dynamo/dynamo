# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest

from dynamo.llm import KvDcRelay, KvRouter, KvRouterConfig
from tests.router.common import _create_kv_router_with_timeout
from tests.router.helper import (
    managed_runtime,
    send_request_via_python_kv_router,
    wait_for_workers_ready,
)
from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
]

BLOCK_SIZE = 16


def _member_blocks(stats: dict) -> dict[int, int]:
    return {
        member["worker_id"]: member["blocks"]
        for member in stats["aggregation"]["members"]
    }


async def _wait_for_stats(relay: KvDcRelay, predicate, timeout: float = 20) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await relay.stats()
        if predicate(last):
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"KV DC Relay stats did not converge: {last}")


async def _send(router: KvRouter, worker_id: int, token_ids: list[int]) -> None:
    assert await send_request_via_python_kv_router(
        kv_python_router=router,
        model_name=ROUTER_MODEL_NAME,
        token_ids=token_ids,
        stop_conditions={"ignore_eos": True, "max_tokens": 1},
        worker_id=worker_id,
        dp_rank=0,
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_kv_dc_relay_deduplicates_workers_and_restores_missed_events(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
):
    mocker_args = {
        "speedup_ratio": 100.0,
        "block_size": BLOCK_SIZE,
        "durable_kv_events": False,
    }
    with (
        MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=2,
            request_plane=request_plane,
        ) as mockers,
        managed_runtime(request_plane=request_plane) as runtime,
    ):
        endpoint = runtime.endpoint(
            f"{mockers.namespace}.{mockers.component_name}.generate"
        )
        router = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint,
                block_size=BLOCK_SIZE,
                kv_router_config=KvRouterConfig(),
            ),
            num_workers=2,
            engine_workers=mockers,
        )

        async def run() -> None:
            worker_ids = sorted(
                await wait_for_workers_ready(
                    endpoint,
                    router,
                    expected_num_workers=2,
                    model_name=ROUTER_MODEL_NAME,
                )
            )
            relay = KvDcRelay(endpoint, ROUTER_MODEL_NAME, "test-dc")
            await relay.start()
            try:
                baseline = await _wait_for_stats(
                    relay,
                    lambda stats: (
                        stats["recovery"]["rank_count"] == 2
                        and stats["recovery"]["recovering_rank_count"] == 0
                    ),
                )
                baseline_members = _member_blocks(baseline)
                common_tokens = list(range(100_000, 100_000 + BLOCK_SIZE * 4))

                await _send(router, worker_ids[0], common_tokens)
                first = await _wait_for_stats(
                    relay,
                    lambda stats: (
                        _member_blocks(stats).get(worker_ids[0], 0)
                        > baseline_members.get(worker_ids[0], 0)
                    ),
                )
                first_members = _member_blocks(first)
                first_member_delta = first_members[
                    worker_ids[0]
                ] - baseline_members.get(worker_ids[0], 0)
                first_unique_delta = (
                    first["aggregation"]["unique_block_count"]
                    - baseline["aggregation"]["unique_block_count"]
                )
                assert first_member_delta == first_unique_delta > 0

                await _send(router, worker_ids[1], common_tokens)
                shared = await _wait_for_stats(
                    relay,
                    lambda stats: (
                        _member_blocks(stats).get(worker_ids[1], 0)
                        > baseline_members.get(worker_ids[1], 0)
                    ),
                )
                shared_members = _member_blocks(shared)
                assert (
                    shared_members[worker_ids[1]]
                    - baseline_members.get(worker_ids[1], 0)
                    == first_member_delta
                )
                assert (
                    shared["aggregation"]["unique_block_count"]
                    == first["aggregation"]["unique_block_count"]
                )
            finally:
                await relay.shutdown()

            missed_tokens = list(range(200_000, 200_000 + BLOCK_SIZE * 3))
            await _send(router, worker_ids[0], missed_tokens)

            restored = KvDcRelay(endpoint, ROUTER_MODEL_NAME, "test-dc")
            await restored.start()
            try:
                recovered = await _wait_for_stats(
                    restored,
                    lambda stats: (
                        stats["recovery"]["rebuild_count"] >= 2
                        and stats["recovery"]["recovering_rank_count"] == 0
                        and stats["aggregation"]["contribution_count"]
                        > shared["aggregation"]["contribution_count"]
                    ),
                )
                recovered_members = _member_blocks(recovered)
                assert recovered_members[worker_ids[0]] > shared_members[worker_ids[0]]
                assert recovered_members[worker_ids[1]] == shared_members[worker_ids[1]]
                assert recovered["recovery"]["pending_live_event_count"] == 0
            finally:
                await restored.shutdown()

        asyncio.run(run())
