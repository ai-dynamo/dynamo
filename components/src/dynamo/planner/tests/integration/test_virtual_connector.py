# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest

from dynamo._core import (
    DistributedRuntime,
    VirtualConnectorClient,
    VirtualConnectorCoordinator,
)
from dynamo.planner import SubComponentType, TargetReplica, VirtualConnector
from dynamo.planner.monitoring.worker_info import build_worker_info_from_defaults

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.sglang,
    pytest.mark.planner,
]
logger = logging.getLogger(__name__)

NAMESPACE = "test_virtual_connector"
ETCD_STARTUP_TIMEOUT = 5
ETCD_SHUTDOWN_TIMEOUT = 5


@contextmanager
def _isolated_etcd(tmp_path):
    tmp_path.mkdir()
    data_dir = tmp_path / "data"
    log_path = tmp_path / "etcd.log"
    command = [
        "etcd",
        "--logger",
        "zap",
        "--data-dir",
        str(data_dir),
        "--listen-client-urls",
        "http://localhost:0",
        "--advertise-client-urls",
        "http://localhost:0",
        "--listen-peer-urls",
        "http://localhost:0",
        "--initial-advertise-peer-urls",
        "http://localhost:0",
        "--initial-cluster",
        "default=http://localhost:0",
    ]

    with log_path.open("w", encoding="utf-8") as log_writer:
        etcd = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=log_writer,
        )

        try:
            client_port = None
            deadline = time.monotonic() + ETCD_STARTUP_TIMEOUT
            with log_path.open(encoding="utf-8") as log_reader:
                while time.monotonic() < deadline:
                    if etcd.poll() is not None:
                        break

                    line = log_reader.readline()
                    if not line:
                        time.sleep(0.01)
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "serving client" not in entry.get("msg", ""):
                        continue

                    match = re.search(r":(\d+)$", entry.get("address", ""))
                    if match:
                        client_port = int(match.group(1))
                        break

            if client_port is None:
                details = log_path.read_text(encoding="utf-8")
                raise RuntimeError(f"etcd failed to start:\n{details}")

            yield client_port
        finally:
            if etcd.poll() is None:
                etcd.terminate()
                try:
                    etcd.wait(timeout=ETCD_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    etcd.kill()
                    etcd.wait(timeout=ETCD_SHUTDOWN_TIMEOUT)
            shutil.rmtree(tmp_path, ignore_errors=True)


class DefaultWorkerInfoProvider:
    def get_worker_info(self, sub_component_type, backend="vllm"):
        return build_worker_info_from_defaults(backend, sub_component_type)


def get_runtime():
    """Get or create a DistributedRuntime instance.

    This handles the case where a worker is already initialized (common in CI)
    by using the detached() method to reuse the existing runtime.
    """
    try:
        # Try to use existing runtime (common in CI where tests run in same process)
        _runtime_instance = DistributedRuntime.detached()
    except Exception:
        # If no existing runtime, create a new one
        loop = asyncio.get_running_loop()
        _runtime_instance = DistributedRuntime(loop, "etcd", "nats")

    return _runtime_instance


# Fails in CI after 30+ minutes with:
# pyo3_runtime.PanicException: Cannot drop a runtime in a context where blocking is not allowed. This happens when a runtime is dropped from within an asynchronous context.
# Disabling until we have a faster CI to iterate with.
@pytest.mark.skip("See comment in source")
def test_main():
    """
    Connect a VirtualConnector (Dynamo Planner) and a VirtualConnectorClient (customer), and scale.
    """
    asyncio.run(async_internal(get_runtime()))


async def next_scaling_decision(c):
    """Move the second decision in to a separate task so we can `.wait` for it."""
    replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=5),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=8),
    ]
    await c.set_component_replicas(replicas, blocking=False)


async def async_internal(distributed_runtime):
    # This is Dynamo Planner
    c = VirtualConnector(
        distributed_runtime,
        NAMESPACE,
        worker_info_provider=DefaultWorkerInfoProvider(),
        model_name="sglang",
    )
    await c.async_init()
    replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=1),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=2),
    ]
    await c.set_component_replicas(replicas, blocking=False)

    # This is the client
    client = VirtualConnectorClient(distributed_runtime, NAMESPACE)
    event = await client.get()
    # Here the client would do the scaling
    assert event.num_prefill_workers == 1
    assert event.num_decode_workers == 2
    assert event.decision_id == 0
    await client.complete(event)

    await c._wait_for_scaling_completion()

    # Second decision with wait

    task = asyncio.create_task(next_scaling_decision(c))
    await client.wait()
    await task

    event = await client.get()
    assert event.num_prefill_workers == 5
    assert event.num_decode_workers == 8
    assert event.decision_id == 1
    await client.complete(event)

    await c._wait_for_scaling_completion()

    # Now scale to zero
    replicas = [
        TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=0),
        TargetReplica(sub_component_type=SubComponentType.DECODE, desired_replicas=0),
    ]
    await c.set_component_replicas(replicas, blocking=False)
    event = await client.get()
    assert event.num_prefill_workers == 0
    assert event.num_decode_workers == 0
    await client.complete(event)


@pytest.mark.timeout(15)
def test_wait_for_unacknowledged_decision(tmp_path):
    # DistributedRuntime is process-global, so keep it out of the pytest worker.
    with _isolated_etcd(tmp_path / "etcd") as etcd_port:
        subprocess.run(
            [sys.executable, __file__],
            check=True,
            env={**os.environ, "ETCD_ENDPOINTS": f"http://localhost:{etcd_port}"},
            timeout=30,
        )


async def _wait_for_unacknowledged_decision():
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp", event_plane="zmq"
    )
    try:
        coord = VirtualConnectorCoordinator(runtime, NAMESPACE, 1, 30, 5)
        await coord.async_init()
        await coord.update_scaling_decision(1, 2)

        # A late consumer must find the decision even though its watch starts afterward.
        client = VirtualConnectorClient(runtime, NAMESPACE)
        await asyncio.wait_for(client.wait(), timeout=5)
        first = await client.get()
        assert (
            first.num_prefill_workers,
            first.num_decode_workers,
            first.decision_id,
        ) == (1, 2, 0)
        await client.complete(first)
        await coord.wait_for_scaling_completion()

        waiter = asyncio.ensure_future(client.wait())
        try:
            done, _ = await asyncio.wait([waiter], timeout=0.2)
            assert not done, "An acknowledged decision must not wake the next wait"
            await client.complete(first)
            done, _ = await asyncio.wait([waiter], timeout=0.2)
            assert not done, "An acknowledgement update is not a new decision"

            await coord.update_scaling_decision(0, 3)
            await asyncio.wait_for(waiter, timeout=5)
            second = await client.get()
            assert (
                second.num_prefill_workers,
                second.num_decode_workers,
                second.decision_id,
            ) == (0, 3, 1)
            await client.complete(second)
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(_wait_for_unacknowledged_decision())
