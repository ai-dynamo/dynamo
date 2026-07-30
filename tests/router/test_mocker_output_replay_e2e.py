# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

from dynamo.llm import KvRouter, KvRouterConfig
from tests.router.common import _create_kv_router_with_timeout
from tests.router.helper import managed_runtime, wait_for_workers_ready
from tests.router.mocker_process import MockerProcess
from tests.utils.constants import ROUTER_MODEL_NAME

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
BLOCK_SIZE = 16
NUM_MOCKERS = 1
SPEEDUP_RATIO = 100.0

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_NAME),
]


def _write_response_replay_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as trace:
        for row in rows:
            trace.write(json.dumps(row))
            trace.write("\n")


def _preprocessed_request(
    *,
    token_ids: list[int],
    max_tokens: int,
    replay_key: str,
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "token_ids": token_ids,
        "stop_conditions": {
            "ignore_eos": True,
            "max_tokens": max_tokens,
        },
        "sampling_options": {},
        "output_options": {},
        "eos_token_ids": [],
        "annotations": [f"output_replay_id:{replay_key}"],
        "routing": {
            "expected_output_tokens": max_tokens,
        },
    }


async def _collect_output_token_ids(
    router: KvRouter,
    request: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    stream = await router.generate_from_request(request)
    output_token_ids: list[int] = []
    terminal: dict[str, Any] | None = None

    async for response in stream:
        if not isinstance(response, dict):
            continue

        token_ids = response.get("token_ids")
        if isinstance(token_ids, list):
            output_token_ids.extend(token_ids)

        if response.get("finish_reason") is not None:
            terminal = response

    assert (
        terminal is not None
    ), "generate_from_request stream did not emit terminal chunk"
    return output_token_ids, terminal


def _terminal_kv_hit_rate(terminal: dict[str, Any]) -> float:
    routing_data = terminal.get("routing_data")
    assert isinstance(routing_data, dict), f"missing routing_data: {terminal}"

    timing = routing_data.get("timing")
    assert isinstance(timing, dict), f"missing routing_data.timing: {terminal}"

    kv_hit_rate = timing.get("kv_hit_rate")
    assert isinstance(kv_hit_rate, (int, float)), f"missing kv_hit_rate: {terminal}"
    return float(kv_hit_rate)


async def _wait_for_overlap_blocks(
    router: KvRouter,
    token_ids: list[int],
    expected_overlap_blocks: int,
) -> None:
    last_overlap: float | None = None
    for _ in range(40):
        _, _, overlap_blocks = await router.best_worker(token_ids)
        last_overlap = float(overlap_blocks)
        if math.isclose(last_overlap, expected_overlap_blocks, abs_tol=1e-9):
            return
        await asyncio.sleep(0.25)

    raise AssertionError(
        f"expected {expected_overlap_blocks} overlap blocks, got {last_overlap}"
    )


async def _run_multi_turn_replay(
    *,
    endpoint,
    router: KvRouter,
    first_output_token_ids: list[int],
    second_output_token_ids: list[int],
) -> None:
    await wait_for_workers_ready(endpoint, router, NUM_MOCKERS, MODEL_NAME)

    first_input_token_ids = list(range(100_001, 100_001 + BLOCK_SIZE))
    first_request = _preprocessed_request(
        token_ids=first_input_token_ids,
        max_tokens=len(first_output_token_ids),
        replay_key="conversation:0",
    )
    returned_first_ids, _ = await _collect_output_token_ids(router, first_request)
    assert returned_first_ids == first_output_token_ids

    second_turn_suffix_token_ids = list(range(400_001, 400_001 + BLOCK_SIZE - 1))
    second_input_token_ids = (
        first_input_token_ids + returned_first_ids + second_turn_suffix_token_ids
    )
    assert len(second_input_token_ids) % BLOCK_SIZE == 0

    expected_overlap_blocks = (
        len(first_input_token_ids) + len(returned_first_ids)
    ) // BLOCK_SIZE
    expected_total_blocks = len(second_input_token_ids) // BLOCK_SIZE
    expected_hit_rate = expected_overlap_blocks / expected_total_blocks

    await _wait_for_overlap_blocks(
        router,
        second_input_token_ids,
        expected_overlap_blocks,
    )

    second_request = _preprocessed_request(
        token_ids=second_input_token_ids,
        max_tokens=len(second_output_token_ids),
        replay_key="conversation:1",
    )
    returned_second_ids, terminal = await _collect_output_token_ids(
        router, second_request
    )
    assert returned_second_ids == second_output_token_ids
    assert math.isclose(
        _terminal_kv_hit_rate(terminal),
        expected_hit_rate,
        abs_tol=1e-9,
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_mocker_output_replay_generate_from_request_multi_turn(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    request_plane,
    tmp_path,
):
    replay_trace_path = tmp_path / "response-replay.jsonl"
    # The final generated token does not have KV yet. Generate one extra token
    # so the first full output block is actually cached for the next turn.
    first_output_token_ids = list(range(200_001, 200_001 + BLOCK_SIZE + 1))
    second_output_token_ids = [300_001, 300_002, 300_003, 300_004]
    _write_response_replay_trace(
        replay_trace_path,
        [
            {
                "session_id": "conversation",
                "output_length": len(first_output_token_ids),
                "output_token_ids": first_output_token_ids,
            },
            {
                "session_id": "conversation",
                "output_length": len(second_output_token_ids),
                "output_token_ids": second_output_token_ids,
            },
        ],
    )

    mocker_args = {
        "speedup_ratio": SPEEDUP_RATIO,
        "block_size": BLOCK_SIZE,
        "response_replay_trace_path": replay_trace_path,
    }

    with (
        MockerProcess(
            request,
            mocker_args=mocker_args,
            num_mockers=NUM_MOCKERS,
            request_plane=request_plane,
        ) as mockers,
        managed_runtime(request_plane=request_plane) as runtime,
    ):
        logger.info("Started mocker replay test endpoint: %s", mockers.endpoint)
        endpoint = runtime.endpoint(
            f"{mockers.namespace}.{mockers.component_name}.generate"
        )
        router_config = KvRouterConfig(router_track_output_blocks=True)
        router = _create_kv_router_with_timeout(
            router_factory=lambda: KvRouter(
                endpoint=endpoint,
                block_size=BLOCK_SIZE,
                kv_router_config=router_config,
            ),
            num_workers=NUM_MOCKERS,
            engine_workers=mockers,
        )

        asyncio.run(
            _run_multi_turn_replay(
                endpoint=endpoint,
                router=router,
                first_output_token_ids=first_output_token_ids,
                second_output_token_ids=second_output_token_ids,
            )
        )
