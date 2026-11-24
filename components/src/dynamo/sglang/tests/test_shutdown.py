# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dynamo.sglang.shutdown import coordinate_non_leader_shutdown

pytestmark = [
    pytest.mark.integration,
    pytest.mark.sglang,
    pytest.mark.framework_agnostic,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.mark.asyncio
async def test_follower_waits_for_scheduler_exit_without_reaping_it():
    # Regression: a follower's empty request count allowed cleanup to kill its
    # schedulers while the leader still needed them to drain admitted work.
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    ) as scheduler:
        shutdown = SimpleNamespace(prefill=True, wait_for_children=None)
        engine = SimpleNamespace(
            server_args=SimpleNamespace(node_rank=1),
            get_all_child_pids=lambda: [scheduler.pid],
        )
        coordinate_non_leader_shutdown(engine, shutdown)
        waiting = asyncio.create_task(shutdown.wait_for_children())
        try:
            await asyncio.sleep(0)
            assert not waiting.done()
            assert scheduler.poll() is None
            scheduler.stdin.close()
            # The engine owns reaping. A zombie must count as exited, otherwise
            # shutdown waits until its deadline despite rank 0 having finished.
            await asyncio.wait_for(waiting, timeout=2)
            assert scheduler.wait(timeout=1) == 0
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            if scheduler.poll() is None:
                scheduler.kill()
                scheduler.wait(timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_pid_list", [False, True])
async def test_follower_without_scheduler_pids_waits_until_cancelled(empty_pid_list):
    # No observable processes is not evidence that rank 0 has drained. The
    # coordinator's drain deadline must remain the fallback, with no join thread.
    shutdown = SimpleNamespace(prefill=True, wait_for_children=None)
    engine = SimpleNamespace(server_args=SimpleNamespace(node_rank=1))
    if empty_pid_list:
        engine.get_all_child_pids = lambda: []
    coordinate_non_leader_shutdown(engine, shutdown)
    waiting = asyncio.create_task(shutdown.wait_for_children())
    try:
        await asyncio.sleep(0)
        assert not waiting.done()
    finally:
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, timeout=1)
