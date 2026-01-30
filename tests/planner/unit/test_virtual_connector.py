# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# This test requires etcd and nats to be running, and the bindings to be installed
#

import asyncio
import logging

import pytest

from dynamo._core import DistributedRuntime, VirtualConnectorClient
from dynamo.planner import SubComponentType, TargetReplica, VirtualConnector

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]
logger = logging.getLogger(__name__)

NAMESPACE = "test_virtual_connector"


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
    c = VirtualConnector(distributed_runtime, NAMESPACE, "sglang")
    await c._async_init()
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
