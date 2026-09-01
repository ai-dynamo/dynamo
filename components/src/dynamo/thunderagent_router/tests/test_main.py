# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for replica attribution on the router response path."""

from __future__ import annotations

import pytest

from dynamo.thunderagent_router.__main__ import ThunderAgentRouterHandler
from dynamo.thunderagent_router.router import PauseDecision

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _handler() -> ThunderAgentRouterHandler:
    handler = object.__new__(ThunderAgentRouterHandler)
    handler._worker_id_extract_warned = False
    return handler


def test_extract_worker_replica_prefers_decode_attribution():
    chunk = {
        "routing_data": {
            "worker_id": {
                "prefill_worker_id": 10,
                "prefill_dp_rank": 1,
                "decode_worker_id": 11,
                "decode_dp_rank": 3,
            }
        }
    }

    assert _handler()._extract_worker_replica(chunk) == (11, 3)


def test_extract_worker_replica_falls_back_to_prefill_attribution():
    chunk = {
        "routing_data": {
            "worker_id": {
                "prefill_worker_id": 10,
                "prefill_dp_rank": 1,
            }
        }
    }

    assert _handler()._extract_worker_replica(chunk) == (10, 1)


def test_extract_worker_id_accepts_legacy_worker_without_rank():
    chunk = {"routing_data": {"worker_id": {"decode_worker_id": 11}}}

    handler = _handler()

    assert handler._extract_worker_replica(chunk) is None
    assert handler._extract_worker_id(chunk) == 11


def test_extract_worker_replica_keeps_decode_preference_when_rank_is_missing():
    chunk = {
        "routing_data": {
            "worker_id": {
                "decode_worker_id": 11,
                "prefill_worker_id": 10,
                "prefill_dp_rank": 1,
            }
        }
    }

    assert _handler()._extract_worker_replica(chunk) is None


class _FakeScheduler:
    def __init__(self, replica: tuple[int, int] | None) -> None:
        self._replica = replica
        self.assigned: list[tuple[str, tuple[int, int]]] = []

    async def before_request(self, program_id: str, **_kwargs) -> PauseDecision:
        return PauseDecision(program_id, assigned_replica_hint=self._replica)

    async def assign_replica(self, program_id: str, replica: tuple[int, int]) -> None:
        self.assigned.append((program_id, replica))

    def record_output_tokens(self, _program_id: str, _tokens: int) -> None:
        pass

    async def after_request(self, *_args) -> None:
        pass


class _FakeKvRouter:
    def __init__(self, response_replica: tuple[int, int]) -> None:
        self._response_replica = response_replica
        self.request = None

    async def generate_from_request(self, request):
        self.request = request
        worker_id, dp_rank = self._response_replica

        async def response_stream():
            yield {
                "token_ids": [42],
                "routing_data": {
                    "worker_id": {
                        "decode_worker_id": worker_id,
                        "decode_dp_rank": dp_rank,
                    }
                },
            }

        return response_stream()


def _initialized_handler(
    scheduler: _FakeScheduler, kv_router: _FakeKvRouter
) -> ThunderAgentRouterHandler:
    handler = _handler()
    handler._scheduler = scheduler  # type: ignore[assignment]
    handler._kv_router = kv_router  # type: ignore[assignment]
    handler._stat_requests_total = 0
    handler._stat_program_requests = 0
    handler._stat_passthrough_requests = 0
    handler._stat_session_final_requests = 0
    return handler


@pytest.mark.asyncio
async def test_program_request_pins_worker_and_data_parallel_rank():
    scheduler = _FakeScheduler((11, 3))
    kv_router = _FakeKvRouter((11, 3))
    handler = _initialized_handler(scheduler, kv_router)

    chunks = [
        chunk
        async for chunk in handler.generate(
            {"token_ids": [1, 2], "agent_context": {"session_id": "p1"}}
        )
    ]

    assert chunks
    assert kv_router.request["routing"] == {
        "backend_instance_id": 11,
        "dp_rank": 3,
    }


@pytest.mark.asyncio
async def test_cold_start_records_selected_worker_and_data_parallel_rank():
    scheduler = _FakeScheduler(None)
    kv_router = _FakeKvRouter((21, 2))
    handler = _initialized_handler(scheduler, kv_router)

    _ = [
        chunk
        async for chunk in handler.generate(
            {"token_ids": [1], "agent_context": {"session_id": "p1"}}
        )
    ]

    assert scheduler.assigned == [("p1", (21, 2))]
