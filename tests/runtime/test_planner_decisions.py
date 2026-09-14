# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import uuid

import httpx
import pytest

from dynamo._core import (
    DistributedRuntime,
    VirtualConnectorClient,
    VirtualConnectorCoordinator,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.timeout(60),
]


def decision_tuple(decision):
    return (
        decision.num_prefill_workers,
        decision.num_decode_workers,
        decision.decision_id,
    )


def encoded(value):
    return base64.b64encode(value.encode()).decode()


@pytest.mark.parametrize("discovery_backend", ["etcd"], indirect=True)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize("event_plane", ["zmq"], indirect=True)
def test_planner_publishes_complete_decisions(
    runtime_services_dynamic_ports, dynamo_dynamic_ports, monkeypatch
):
    _, etcd = runtime_services_dynamic_ports
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(dynamo_dynamic_ports.system_ports[0]))
    loop = asyncio.new_event_loop()
    runtime = DistributedRuntime(loop, "etcd", "tcp", event_plane="zmq")

    async def exercise_decisions():
        namespace = f"planner-{uuid.uuid4().hex}"
        prefix = f"v1/{namespace}/planner/"

        def coordinator():
            return VirtualConnectorCoordinator(runtime, namespace, 1, 0, 5)

        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{etcd.port}", timeout=5
        ) as http:

            async def rpc(path, body):
                response = await http.post(f"/v3/{path}", json=body)
                response.raise_for_status()
                return response.json()

            async def snapshot():
                result = await rpc(
                    "kv/range",
                    {"key": encoded(prefix), "range_end": encoded(prefix[:-1] + "0")},
                )
                values = {
                    base64.b64decode(kv["key"]).decode(): base64.b64decode(kv["value"])
                    for kv in result.get("kvs", [])
                }
                return int(result["header"]["revision"]), values

            coord = coordinator()
            await coord.async_init()
            client = VirtualConnectorClient(runtime, namespace)
            assert decision_tuple(await client.get()) == (-1, -1, -1)

            for prefill, decode, expected in [
                (None, 2, (-1, 2, 0)),
                (1, None, (1, 2, 1)),
                (5, 8, (5, 8, 2)),
                (0, None, (0, 8, 3)),
            ]:
                revision, _ = await snapshot()
                await coord.update_scaling_decision(prefill, decode)
                next_revision, values = await snapshot()
                # A single etcd write must publish all three fields, including partial updates.
                assert next_revision == revision + 1
                assert set(values) <= {
                    prefix + "scaling_decision",
                    prefix + "scaled_decision_id",
                }
                record = json.loads(values[prefix + "scaling_decision"])
                assert record == dict(
                    zip(
                        ("num_prefill_workers", "num_decode_workers", "decision_id"),
                        expected,
                    )
                )
                event = await client.get()
                assert decision_tuple(event) == expected
                assert decision_tuple(coord.read_state()) == expected
                await client.complete(event)
                await coord.wait_for_scaling_completion()
                assert await coord.is_scaling_ready()

            restarted = coordinator()
            await restarted.async_init()
            assert decision_tuple(restarted.read_state()) == (0, 8, 3)

            # A real etcd write rejection must preserve both persisted and local state.
            revision, before = await snapshot()
            await rpc("maintenance/alarm", {"action": "ACTIVATE", "alarm": "NOSPACE"})
            with pytest.raises(Exception, match="(?i)(space|quota)"):
                await restarted.update_scaling_decision(7, 9)
            assert decision_tuple(restarted.read_state()) == (0, 8, 3)
            assert await snapshot() == (revision, before)
            await rpc("maintenance/alarm", {"action": "DEACTIVATE", "alarm": "NOSPACE"})
            await restarted.update_scaling_decision(7, 9)
            assert decision_tuple(await client.get()) == (7, 9, 4)

            # A coordinated upgrade must keep the legacy decision ID above old acknowledgements.
            namespace = f"planner-legacy-{uuid.uuid4().hex}"
            prefix = f"v1/{namespace}/planner/"
            for key, value in {
                "num_prefill_workers": "1",
                "num_decode_workers": "2",
                "decision_id": "42",
                "scaled_decision_id": "42",
            }.items():
                await rpc(
                    "kv/put", {"key": encoded(prefix + key), "value": encoded(value)}
                )
            migrated = coordinator()
            await migrated.async_init()
            legacy_client = VirtualConnectorClient(runtime, namespace)
            assert decision_tuple(migrated.read_state()) == (1, 2, 42)
            assert decision_tuple(await legacy_client.get()) == (1, 2, 42)
            await migrated.update_scaling_decision(None, 8)
            assert decision_tuple(await legacy_client.get()) == (1, 8, 43)
            migrated_restart = coordinator()
            await migrated_restart.async_init()
            assert decision_tuple(migrated_restart.read_state()) == (1, 8, 43)

    try:
        loop.run_until_complete(exercise_decisions())
    finally:
        runtime.shutdown()
        loop.close()
