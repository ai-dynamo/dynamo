# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
]


@pytest.mark.timeout(60)
def test_planner_control_uses_the_actual_system_listener():
    # Runtime initialization is process-global; isolate its ephemeral listener
    # from other tests without requiring etcd, NATS, or a model download.
    script = textwrap.dedent(
        """
        import asyncio
        import aiohttp
        from dynamo.runtime import DistributedRuntime
        from dynamo.planner.tests.unit.test_control_api import _planner

        async def main():
            runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp", event_plane="zmq")
            async def legacy_echo(body):
                return body
            runtime.register_engine_route("legacy/echo", legacy_echo)
            planner = _planner("agg", enable_runtime_reconfiguration=True)
            planner.runtime = runtime
            base = runtime.system_status_server_url() + "/engine/planner"
            async with aiohttp.ClientSession() as client:
                payload = {"status": 418, "body": {"value": 7}}
                async with client.post(runtime.system_status_server_url() + "/engine/legacy/echo", json=payload) as response:
                    assert response.status == 200
                    assert await response.json(content_type=None) == payload
                async with client.get(base + "/config") as response:
                    assert response.status == 404
                await planner._async_init()
                async with client.get(base + "/config") as response:
                    assert response.status == 200
                    current = await response.json()
                async with client.patch(base + "/config", data="{", headers={"Content-Type": "application/json"}) as response:
                    assert response.status == 400
                    assert (await response.json())["error"] == "Invalid JSON"
                update = {
                    "instance_id": current["instance_id"],
                    "expected_revision": current["revision"],
                    "request_id": "wire-test",
                    "config": {"max_gpu_budget": 24},
                }
                async with client.patch(base + "/config", json=update) as response:
                    assert response.status == 202, await response.text()
                for _ in range(200):
                    async with client.get(base + "/operations/wire-test") as response:
                        assert response.status == 200
                        operation = await response.json()
                    if operation["state"] == "applied":
                        break
                    assert operation["state"] != "failed", operation
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("update did not activate")
                async with client.get(base + "/config") as response:
                    applied = await response.json()
                assert applied["revision"] == 1
                assert applied["config"]["max_gpu_budget"] == 24
                async with client.patch(base + "/config", json=update) as response:
                    assert response.status == 202
                    assert (await response.json())["state"] == "applied"
                update["request_id"] = "stale"
                async with client.patch(base + "/config", json=update) as response:
                    assert response.status == 409
                update["expected_revision"] = 1
                update["config"] = {"ais_perf_model": None}
                async with client.patch(base + "/config", json=update) as response:
                    assert response.status == 422
                async with client.delete(base + "/config") as response:
                    assert response.status == 405
                await planner._shutdown_runtime()
                async with client.get(base + "/config") as response:
                    assert response.status == 404
            runtime.shutdown()

        asyncio.run(main())
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ | {"DYN_SYSTEM_PORT": "0", "DYN_SYSTEM_HOST": "127.0.0.1"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
