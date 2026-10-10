# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kvbm accepts a dynamo._core.DistributedRuntime as `drt` and keeps it alive.

This is the path DYN_RUNTIME_ENABLED_KVBM=1 takes. Building the leader needs no
GPU or external services: the runtime uses in-memory discovery, and the leader
only binds its two ZMQ sockets.
"""

import asyncio
import sys

import pytest

kvbm = pytest.importorskip("kvbm", reason="kvbm package not installed")
from dynamo._core import DistributedRuntime  # noqa: E402
from tests.utils.port_utils import reserved_ports  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
]


@pytest.fixture
def leader_env(monkeypatch):
    # Same ZMQ port base as the other kvbm tests; the defaults (56001/56002) are fixed.
    with reserved_ports(2, 20001) as (pub_port, ack_port):
        monkeypatch.setenv("DYN_KVBM_LEADER_ZMQ_HOST", "127.0.0.1")
        monkeypatch.setenv("DYN_KVBM_LEADER_ZMQ_PUB_PORT", str(pub_port))
        monkeypatch.setenv("DYN_KVBM_LEADER_ZMQ_ACK_PORT", str(ack_port))
        # The leader refuses to start without a cache tier.
        monkeypatch.setenv("DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS", "1")
        # No worker ever joins, so end the leader's handshake loop soon after the test.
        monkeypatch.setenv("DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS", "1")
        yield


@pytest.mark.timeout(30)
async def test_leader_holds_distributed_runtime(leader_env):
    drt = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    try:
        with pytest.raises(TypeError, match="dynamo._core.DistributedRuntime"):
            kvbm.KvbmLeader(1, drt=object())

        base = sys.getrefcount(drt)
        leader = kvbm.KvbmLeader(1, drt=drt)
        assert sys.getrefcount(drt) == base + 1

        del leader
        assert sys.getrefcount(drt) == base
    finally:
        drt.shutdown()
