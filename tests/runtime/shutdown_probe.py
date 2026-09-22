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
import time
import uuid
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from dynamo._core import DistributedRuntime
from dynamo.common.backend.engine import EngineConfig, RawEngine
from dynamo.common.backend.worker import ShutdownConfig, Worker, WorkerConfig
from dynamo.common.utils.worker_shutdown import WorkerShutdown, serve_endpoint
from dynamo.llm import ModelInput


def signal_self():
    print(f"SIGTERM_AT={time.monotonic()}", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)


class ProbeEngine(RawEngine):
    @classmethod
    async def from_args(cls, argv=None):
        raise NotImplementedError

    async def start(self, worker_id):
        # Signal after the Rust listener is installed but before start returns.
        asyncio.get_running_loop().call_soon(signal_self)
        await asyncio.sleep(0.05)
        return EngineConfig(model="shutdown-probe")

    async def generate(self, request, context):
        yield request

    async def cleanup(self):
        print("CLEANUP_STARTED", flush=True)
        if sys.argv[1] == "sdk-wedged":
            ctypes.PyDLL(None).sleep(30)
        if sys.argv[1] == "sdk-failed":
            raise RuntimeError("synthetic cleanup failure")
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
            shutdown=ShutdownConfig(
                total_secs=0.5, router_grace_secs=0, cleanup_timeout_secs=0.2
            ),
        ),
    )
    if sys.argv[1] == "sdk-failed":
        try:
            await worker.run()
        except Exception as error:
            assert "synthetic cleanup failure" in str(error), error
            print("CLEANUP_FAILED", flush=True)
            return
        raise AssertionError("worker reported success after cleanup failed")
    await worker.run()
    print("WORKER_RETURNED", flush=True)
    # Stay alive past the total: the host must not be killed after
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
            if sys.argv[1] == "python-wedged":
                ctypes.PyDLL(None).sleep(30)
            assert completed, "engine freed before its admitted request completed"
            if group is not None:
                assert all(child.poll() == 0 for _, child in group.children)
                print("CHILDREN_DRAINED", flush=True)
                group.cleanup()
            # Cleanup must fit the reserved allowance inside the total.
            await asyncio.sleep(0.5)
            print("ENGINE_CLEANED", flush=True)

    serving = asyncio.create_task(shutdown.run(worker()))
    client = await endpoint.client()
    instances = await asyncio.wait_for(client.wait_for_instances(), 5)
    stream = await client.direct({}, instances[0], annotated=False)
    assert await anext(stream) == {"seq": 0}
    if group is None:
        signal_self()
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


async def gateway_probe():
    # Exercise the production supervisor and real process-group delivery;
    # only GPU engine construction and shared-memory setup are unused stubs.
    sgl = ModuleType("sglang")
    sgl.Engine = SimpleNamespace(async_generate=None, _resolve_routed_dp_rank=None)
    mixin = ModuleType("sglang.srt.managers.multi_tokenizer_mixin")
    mixin.write_data_for_multi_tokenizer = None
    with patch.dict(sys.modules, {"sglang": sgl, mixin.__name__: mixin}):
        gateway = importlib.import_module("dynamo.sglang.gateway")
        runtime = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        stop = asyncio.Event()
        shutdown = WorkerShutdown(runtime, [], stop)
        spawned = []
        popen = subprocess.Popen

        def launch(_command, **kwargs):
            child = popen(
                [sys.executable, __file__, "embedding-child", "0.1"],
                stdout=subprocess.PIPE,
                text=True,
                **kwargs,
            )
            spawned.append(child)
            return child

        with patch.object(gateway.subprocess, "Popen", launch):
            serving = asyncio.create_task(
                shutdown.run(
                    gateway.serve_via_gateway_children(
                        SimpleNamespace(_multi_tokenizer_shm=object()),
                        1,
                        stop,
                        shutdown=shutdown,
                    )
                )
            )
            try:
                while not spawned:
                    await asyncio.sleep(0)
                child = spawned[0]
                assert await asyncio.to_thread(child.stdout.readline) == "CHILD_READY\n"
                assert os.getpgrp() == os.getpid()
                os.killpg(os.getpgrp(), signal.SIGTERM)
                await serving
                assert child.returncode == 0, child.returncode
                assert not shutdown.accepting
                print(
                    "ADMISSION_CLOSED\nCHILDREN_DRAINED\nENGINE_CLEANED\nRUNTIME_FINISHED",
                    flush=True,
                )
            finally:
                for child in spawned:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
                    child.stdout.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode.startswith("sdk-"):
        asyncio.run(sdk_probe())
    elif mode == "embedding-child":
        asyncio.run(embedding_child())
    elif mode.startswith("embedding-"):
        embedding_probe()
    elif mode == "gateway-group":
        asyncio.run(gateway_probe())
    else:
        asyncio.run(python_probe())
