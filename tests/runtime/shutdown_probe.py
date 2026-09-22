# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess probes for shutdown; run through test_worker_shutdown_process.py."""

import asyncio
import ctypes
import importlib
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from types import ModuleType
from unittest.mock import patch

from dynamo._core import DistributedRuntime
from dynamo.common.backend.engine import EngineConfig, RawEngine
from dynamo.common.backend.worker import ShutdownConfig, Worker, WorkerConfig
from dynamo.common.utils.worker_shutdown import WorkerShutdown, serve_endpoint
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


def embedding_processes():
    """Use the production process supervisor without loading GPU engine packages.

    Only unused engine-construction imports are stubbed. Subprocesses, monitoring,
    signal delivery, the shutdown coordinator, and TCP bindings are real.
    """
    imports = {
        "vllm": {},
        "vllm.config": {"VllmConfig": object},
        "vllm.usage.usage_lib": {"UsageContext": object},
        "vllm.v1.engine.async_llm": {"AsyncLLM": object},
        "vllm.v1.engine.utils": {
            "get_engine_zmq_addresses": None,
            "launch_core_engines": None,
        },
        "vllm.v1.executor": {"Executor": object},
    }
    modules = {}
    for name, attributes in imports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        return importlib.import_module("dynamo.vllm.embedding_worker_processes")


async def python_probe(group=None):
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(f"shutdown{uuid.uuid4().hex}.worker.generate")
    shutdown = WorkerShutdown(runtime, [endpoint], asyncio.Event())
    if group is not None:
        shutdown.notify_children = group.begin_shutdown
        shutdown.wait_for_children = group.wait_for_shutdown
    completed = False

    async def pull_handler(request, context=None):
        nonlocal completed
        yield {"seq": 0}
        await asyncio.sleep(0.7)
        completed = True
        yield {"seq": 1}

    async def push_handler(request, context=None, response_sender=None):
        async for item in pull_handler(request, context):
            if response_sender is None:
                yield item
            else:
                response_sender.send(item)
        if response_sender is not None:
            response_sender.close()

    async def worker():
        try:
            await serve_endpoint(
                endpoint,
                push_handler if sys.argv[1] == "python-push" else pull_handler,
                shutdown=shutdown,
            )
        finally:
            assert completed, "engine freed before its admitted request completed"
            if group is not None:
                assert all(child.poll() == 0 for _, child in group.children)
                print("CHILDREN_DRAINED", flush=True)
                group.cleanup()
            # Positive remainder < 0.3s is insufficient, but the 5s floor is.
            await asyncio.sleep(0.5)
            print("ENGINE_CLEANED", flush=True)

    serving = asyncio.create_task(shutdown.run(worker()))
    client = await endpoint.client()
    instances = await asyncio.wait_for(client.wait_for_instances(), 5)
    stream = await client.direct({}, instances[0], annotated=False)
    assert await anext(stream) == {"seq": 0}
    if group is None:
        os.kill(os.getpid(), signal.SIGTERM)
    else:
        assert os.getpgrp() == os.getpid(), "probe must own its process group"
        os.killpg(os.getpgrp(), signal.SIGTERM)
    while shutdown.accepting:
        await asyncio.sleep(0)
    if sys.argv[1] == "python-escalate":
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("second signal did not terminate the worker")
    # This router observes discovery removal immediately. The separate TCP
    # rejection test covers stale routers that still address a closed gate.
    while client.instance_ids():
        await asyncio.sleep(0)
    print("ADMISSION_CLOSED", flush=True)
    assert [item async for item in stream] == [{"seq": 1}]
    await serving
    assert completed
    print("RUNTIME_FINISHED", flush=True)


async def embedding_child():
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    shutdown = WorkerShutdown(runtime, [], asyncio.Event())

    async def worker():
        print("CHILD_READY", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(float(sys.argv[2]))

    # Bound the child even if the outer probe fails before signalling it.
    async with asyncio.timeout(10):
        await shutdown.run(worker())


def embedding_probe():
    processes = embedding_processes()
    delay = "0.1" if sys.argv[1] == "embedding-idle" else "0.9"
    resource = tempfile.TemporaryDirectory()
    with resource:
        with subprocess.Popen(
            [sys.executable, __file__, "embedding-child", delay],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as child:
            try:
                assert child.stdout.readline().strip() == "CHILD_READY"
                group = processes.EmbeddingWorkerProcessGroup(
                    children=[(1, child)],
                    engine_manager=None,
                    rpc_directory=resource,
                    previous_rpc_base_path=None,
                )
                group.start_monitor()
                asyncio.run(python_probe(group))
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode.startswith("sdk-"):
        asyncio.run(sdk_probe())
    elif mode == "embedding-child":
        asyncio.run(embedding_child())
    elif mode.startswith("embedding-"):
        embedding_probe()
    else:
        asyncio.run(python_probe())
