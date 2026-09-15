# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the router entrypoint: replica extraction and the sticky pin.

The three pieces that turn a placement into a pin the KvRouter accepts. Pure dict
functions, no runtime needed.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from dynamo.thunderagent_router.__main__ import ThunderAgentRouterHandler
from dynamo.thunderagent_router.program_state import ReplicaKey
from dynamo.thunderagent_router.router import PauseDecision

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def make_handler() -> ThunderAgentRouterHandler:
    """A handler with only the state the tested methods touch."""
    return ThunderAgentRouterHandler(runtime=None, config=None)  # type: ignore[arg-type]


def chunk_with(worker_info: Any) -> dict[str, Any]:
    return {"routing_data": {"worker_id": worker_info}}


@pytest.mark.parametrize(
    "worker_info, expected",
    [
        (
            {
                "decode_worker_id": 7,
                "decode_dp_rank": 3,
                "prefill_worker_id": 9,
                "prefill_dp_rank": 1,
            },
            (7, 3),
        ),
        ({"prefill_worker_id": 9, "prefill_dp_rank": 1}, (9, 1)),
        ({"decode_worker_id": 7, "decode_dp_rank": 0}, (7, 0)),
    ],
    ids=["prefers_decode", "falls_back_to_prefill", "rank_zero_is_a_rank"],
)
def test_extract_replica_takes_both_halves_from_one_phase(worker_info, expected):
    assert make_handler()._extract_replica(chunk_with(worker_info)) == expected


def test_extract_replica_never_pairs_a_decode_worker_with_a_prefill_rank():
    """Per-field fallback would return ``(7, 1)`` here -- a replica that never served
    anything, because ``decode_dp_rank`` is only written when a rank was supplied."""
    worker_info = {
        "decode_worker_id": 7,
        "prefill_worker_id": 9,
        "prefill_dp_rank": 1,
    }

    assert make_handler()._extract_replica(chunk_with(worker_info)) == (9, 1)


@pytest.mark.parametrize(
    "chunk",
    [
        "not a dict",
        {},
        {"routing_data": "not a dict"},
        {"routing_data": {}},
        {"routing_data": {"worker_id": "not a dict"}},
        {"routing_data": {"worker_id": {"decode_worker_id": 7}}},
        {"routing_data": {"worker_id": {"decode_dp_rank": 3}}},
        {"routing_data": {"worker_id": {"decode_worker_id": "7", "decode_dp_rank": 3}}},
    ],
    ids=[
        "not_a_dict",
        "no_routing_data",
        "routing_data_not_a_dict",
        "no_worker_id",
        "worker_id_not_a_dict",
        "worker_without_rank",
        "rank_without_worker",
        "worker_id_not_an_int",
    ],
)
def test_extract_replica_returns_none_on_an_incomplete_payload(chunk):
    """Half an answer is not usable: the caller must leave the program unplaced."""
    assert make_handler()._extract_replica(chunk) is None


class FakeScheduler:
    """Records what the handler asked for and answers with a fixed decision."""

    def __init__(
        self,
        replica_hint: Optional[ReplicaKey],
        *,
        admission_epoch: int = 1,
    ) -> None:
        self._decision = PauseDecision(
            program_id="p1",
            assigned_replica_hint=replica_hint,
            admission_epoch=admission_epoch,
        )
        self.back_fills: list[tuple[Optional[ReplicaKey], int]] = []

    async def before_request(self, program_id, estimated_prompt_tokens=0):
        return self._decision

    async def assign_worker(self, program_id, replica, *, admission_epoch):
        self.back_fills.append((replica, admission_epoch))
        return replica is not None

    def record_output_tokens(self, program_id, delta_tokens):
        pass

    async def after_request(self, program_id, prompt_tokens, completion_tokens):
        pass


class FakeKvRouter:
    """Captures the outgoing request and replays fixed chunks."""

    def __init__(self, chunks: Optional[list[dict[str, Any]]] = None) -> None:
        self.chunks = chunks if chunks is not None else []
        self.sent: Optional[dict[str, Any]] = None

    async def generate_from_request(self, preprocessed):
        self.sent = preprocessed

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


async def drive(handler: ThunderAgentRouterHandler) -> None:
    request = {
        "token_ids": [1, 2, 3],
        "agent_context": {"session_id": "p1"},
    }
    async for _ in handler.generate(request):
        pass


@pytest.mark.asyncio
async def test_a_placed_program_pins_both_backend_id_and_dp_rank():
    """The actual fix for the reported 500: ``resolve_pinned_worker_rank`` rejects a
    ``backend_instance_id`` with no rank once the worker owns more than one."""
    handler = make_handler()
    handler._scheduler = FakeScheduler(replica_hint=(7, 3))  # type: ignore[assignment]
    handler._kv_router = FakeKvRouter()  # type: ignore[assignment]

    await drive(handler)

    sent = handler._kv_router.sent
    assert sent is not None
    assert sent["routing"] == {
        "backend_instance_id": 7,
        "dp_rank": 3,
    }
    assert handler._stat_unpinned_turns == 0


@pytest.mark.asyncio
async def test_an_unplaced_program_sends_no_pin_and_is_counted():
    """A turn that lost sticky affinity must be visible: without the counter it is
    indistinguishable in the logs from a pinned one."""
    handler = make_handler()
    handler._scheduler = FakeScheduler(replica_hint=None)  # type: ignore[assignment]
    handler._kv_router = FakeKvRouter()  # type: ignore[assignment]

    await drive(handler)

    sent = handler._kv_router.sent
    assert sent is not None
    assert sent["routing"] is None
    assert handler._stat_unpinned_turns == 1


@pytest.mark.asyncio
async def test_an_unplaced_program_back_fills_the_replica_from_the_first_chunk():
    handler = make_handler()
    scheduler = FakeScheduler(replica_hint=None, admission_epoch=4)
    handler._scheduler = scheduler  # type: ignore[assignment]
    handler._kv_router = FakeKvRouter(  # type: ignore[assignment]
        [chunk_with({"decode_worker_id": 7, "decode_dp_rank": 3})]
    )

    await drive(handler)

    assert scheduler.back_fills == [((7, 3), 4)]


@pytest.mark.asyncio
async def test_a_rankless_first_chunk_offers_nothing_to_back_fill():
    """The scheduler is asked to record None rather than a worker without a rank."""
    handler = make_handler()
    scheduler = FakeScheduler(replica_hint=None)
    handler._scheduler = scheduler  # type: ignore[assignment]
    handler._kv_router = FakeKvRouter(  # type: ignore[assignment]
        [chunk_with({"decode_worker_id": 7})]
    )

    await drive(handler)

    assert scheduler.back_fills == [(None, 1)]


@pytest.mark.asyncio
async def test_a_placed_program_does_not_consult_the_first_chunk():
    """Admission already named the replica; the response is not asked to confirm it."""
    handler = make_handler()
    scheduler = FakeScheduler(replica_hint=(7, 3))
    handler._scheduler = scheduler  # type: ignore[assignment]
    handler._kv_router = FakeKvRouter(  # type: ignore[assignment]
        [chunk_with({"decode_worker_id": 9, "decode_dp_rank": 1})]
    )

    await drive(handler)

    assert scheduler.back_fills == []
