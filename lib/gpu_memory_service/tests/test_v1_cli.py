# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import threading
from contextlib import ExitStack

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.core.server.allocations import GMSAllocationManager
from gpu_memory_service.core.server.gms import GMS
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1 import cli

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@pytest.mark.timeout(10)
def test_sidecar_couples_listener_lifecycle_and_recovers_stale_sockets(
    tmp_path,
) -> None:
    paths = [tmp_path / domain for domain in ("weights.sock", "kv-cache.sock")]
    active = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    active.bind(str(paths[0]))
    active.listen()
    try:
        gms = GMS("GPU-0", GMSAllocationManager(FakeVMM(granularity=64), 0))
        with pytest.raises(RuntimeError, match="GMS already running"):
            GMSRPCServer(str(paths[0]), gms)
        assert paths[0].exists()
    finally:
        active.close()

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(paths[1]))
    stale.close()

    with ExitStack() as stack:
        servers = [
            stack.enter_context(
                GMSRPCServer(
                    str(path),
                    GMS(
                        "GPU-0",
                        GMSAllocationManager(FakeVMM(granularity=64), 0),
                    ),
                )
            )
            for path in paths
        ]
        serving = [threading.Event(), threading.Event()]
        stopped = [threading.Event(), threading.Event()]
        for server, serving_event, stopped_event in zip(servers, serving, stopped):
            serve_forever = server.serve_forever

            def observed_serve_forever(
                serve_forever=serve_forever,
                serving_event=serving_event,
                stopped_event=stopped_event,
            ) -> None:
                serving_event.set()
                try:
                    serve_forever(poll_interval=0.01)
                finally:
                    stopped_event.set()

            server.serve_forever = observed_serve_forever

        def stop_weights() -> None:
            for event in serving:
                if not event.wait(5):
                    return
            servers[0].shutdown()

        stopper = threading.Thread(target=stop_weights, daemon=True)
        stopper.start()
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            cli.run_servers(servers)
        stopper.join(timeout=5)
        assert not stopper.is_alive()
        assert all(event.is_set() for event in stopped)

    assert all(not path.exists() for path in paths)
