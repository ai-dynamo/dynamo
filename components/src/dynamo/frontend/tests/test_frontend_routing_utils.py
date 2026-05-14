#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.utils import FrontendRoundRobinRouter

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _FakeClient:
    def __init__(self, instance_sequences):
        self._instance_sequences = list(instance_sequences)
        self._instance_idx = 0
        self.direct_calls = []

    def instance_ids(self):
        if self._instance_idx < len(self._instance_sequences):
            ids = self._instance_sequences[self._instance_idx]
            self._instance_idx += 1
            return ids
        return self._instance_sequences[-1]

    async def wait_for_instances(self):
        return [9, 10]

    async def direct(self, request, instance_id, annotated=True):
        self.direct_calls.append((request, instance_id, annotated))
        return {
            "instance": str(instance_id),
            "request": request,
            "annotated": annotated,
        }


@pytest.mark.asyncio
async def test_frontend_round_robin_router_balances_sorted_instance_ids():
    client = _FakeClient([[20, 10, 30], [20, 10, 30], [20, 10, 30], [20, 10, 30]])
    router = FrontendRoundRobinRouter(client, "dynamo.backend.generate")

    results = []
    for idx in range(4):
        results.append(await router.generate({"seq": idx}, annotated=False))

    assert [item["instance"] for item in results] == ["10", "20", "30", "10"]
    assert [call[2] for call in client.direct_calls] == [False, False, False, False]


@pytest.mark.asyncio
async def test_frontend_round_robin_router_refreshes_membership_each_request():
    client = _FakeClient([[2, 1], [3, 2, 1], [3, 2, 1]])
    router = FrontendRoundRobinRouter(client, "dynamo.backend.generate")

    first = await router.generate({"seq": 0}, annotated=False)
    second = await router.generate({"seq": 1}, annotated=False)
    third = await router.generate({"seq": 2}, annotated=False)

    assert first["instance"] == "1"
    assert second["instance"] == "2"
    assert third["instance"] == "3"


@pytest.mark.asyncio
async def test_frontend_round_robin_router_waits_for_instances_when_empty():
    client = _FakeClient([[]])
    router = FrontendRoundRobinRouter(client, "dynamo.backend.generate")

    result = await router.generate({"seq": 0}, annotated=False)

    assert result["instance"] == "9"


@pytest.mark.asyncio
async def test_frontend_round_robin_router_raises_when_no_instances_ever_appear():
    client = _FakeClient([[]])
    client.wait_for_instances = _empty_instances
    router = FrontendRoundRobinRouter(client, "dynamo.backend.generate")

    with pytest.raises(RuntimeError, match="No active backend instances available"):
        await router.generate({"seq": 0}, annotated=False)


@pytest.mark.asyncio
async def test_frontend_round_robin_router_rejects_unexpected_kwargs():
    client = _FakeClient([[1]])
    router = FrontendRoundRobinRouter(client, "dynamo.backend.generate")

    with pytest.raises(TypeError, match="Unsupported kwargs"):
        await router.generate({"seq": 0}, annotated=False, foo=1)


async def _empty_instances():
    return []
