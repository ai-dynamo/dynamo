# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess probes for shutdown; run through test_worker_shutdown_process.py."""

import asyncio
import ctypes
import os
import signal
import sys

from dynamo.common.backend.engine import EngineConfig, RawEngine
from dynamo.common.backend.worker import ShutdownConfig, Worker, WorkerConfig
from dynamo.llm import ModelInput


class ProbeEngine(RawEngine):
    @classmethod
    async def from_args(cls, argv=None):
        raise NotImplementedError

    async def start(self, worker_id):
        # Signal after the Rust listener is installed but before start returns.
        asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.05)
        return EngineConfig(model="shutdown-probe")

    async def generate(self, request, context):
        yield request

    async def cleanup(self):
        print("CLEANUP_STARTED", flush=True)
        if sys.argv[1] == "sdk-wedged":
            ctypes.PyDLL(None).sleep(30)
        print("ENGINE_CLEANED", flush=True)


async def sdk_probe():
    worker = Worker(
        ProbeEngine(),
        WorkerConfig(
            namespace="shutdown-probe",
            model_input=ModelInput.Text,
            endpoint_types="images",
            discovery_backend="mem",
            request_plane="tcp",
            event_plane="zmq",
            shutdown=ShutdownConfig(total_secs=0.5, router_grace_secs=0),
        ),
    )
    await worker.run()
    print("WORKER_RETURNED", flush=True)
    # Stay alive past total + cleanup floor: the host must not be killed after
    # the awaited library call has completed its engine and runtime teardown.
    await asyncio.sleep(6)
    print("HOST_SURVIVED", flush=True)


if __name__ == "__main__":
    asyncio.run(sdk_probe())
