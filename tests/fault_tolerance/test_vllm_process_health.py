"""Isolated GPU checks for local process death and discovery withdrawal."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

import psutil
import pytest
import requests

from dynamo.runtime import DistributedRuntime
from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import allocate_ports, deallocate_ports

MODEL = "Qwen/Qwen3-0.6B"
KV_BYTES = 256 * 1024 * 1024

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.vllm,
    pytest.mark.gpu_2,
    pytest.mark.e2e,
    pytest.mark.post_merge,
    pytest.mark.model(MODEL),
    pytest.mark.profiled_vram_gib(8.0),
    pytest.mark.requested_vllm_kv_cache_bytes(KV_BYTES),
    pytest.mark.timeout(420),
    pytest.mark.asyncio,
]


@contextmanager
def owned_process(command, env, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


async def wait_until(predicate, timeout=180):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Condition did not become true"
        await asyncio.sleep(0.05)


async def post(port, path, body):
    response = await asyncio.to_thread(
        requests.post, f"http://127.0.0.1:{port}{path}", json=body, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    assert result.get("status") != "error", result
    return result


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    ports = allocate_ports(5, DynamoPortRange.FRONTEND.value)
    namespace = "health_" + uuid.uuid4().hex
    discovery = tmp_path / "discovery"
    discovery.mkdir()
    for name, value in {
        "DYN_FILE_KV": str(discovery),
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_NAMESPACE": namespace,
        "DYN_SYSTEM_PORT": str(ports[4]),
    }.items():
        monkeypatch.setenv(name, value)
    env = os.environ.copy()
    env.update(
        DYN_ENGINE_HEALTH_CHECK_INTERVAL="0.1",
        DYN_ENGINE_HEALTH_CHECK_TIMEOUT="0.2",
        DYN_ENGINE_HEALTH_SHUTDOWN_TIMEOUT="4",
        DYN_SYSTEM_PORT=str(ports[3]),
    )
    frontend = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(ports[0]),
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--namespace",
        namespace,
    ]
    worker = [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        MODEL,
        "--endpoint",
        f"{namespace}.worker.generate",
        "--enforce-eager",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "2",
        "--kv-cache-memory-bytes",
        os.environ.get("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", str(KV_BYTES)),
        "--gpu-memory-utilization",
        "0.1",
        "--enable-sleep-mode",
        "--enable-rl",
        "--discovery-backend",
        "file",
        "--request-plane",
        "tcp",
        "--worker-extension-cls",
        "tests.fault_tolerance.health_test_worker.HealthTestWorkerExtension",
    ]
    try:
        with owned_process(frontend, env, tmp_path / "frontend.log"):
            yield namespace, ports, env, worker, tmp_path
    finally:
        deallocate_ports(ports)


async def discovery_client(namespace):
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    client = await runtime.endpoint(f"{namespace}.worker.generate").client()
    return runtime, client


async def completion(port):
    return await post(
        port, "/v1/completions", {"model": MODEL, "prompt": "Hello", "max_tokens": 4}
    )


async def wait_for_model(port):
    deadline = time.monotonic() + 30
    while True:
        response = await asyncio.to_thread(
            requests.get, f"http://127.0.0.1:{port}/v1/models", timeout=5
        )
        response.raise_for_status()
        if any(model["id"] == MODEL for model in response.json()["data"]):
            return
        assert time.monotonic() < deadline, "Frontend did not discover the model"
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("victim", ["execution_worker", "engine_core"])
async def test_process_death_withdraws_before_cleanup(cluster, victim):
    namespace, ports, env, command, directory = cluster
    env["DYN_SYSTEM_PORT"] = str(ports[1])
    with owned_process(command, env, directory / "target.log") as target:
        runtime, client = await discovery_client(namespace)
        try:
            await wait_until(lambda: len(client.instance_ids()) == 1)
            target_id = client.instance_ids()[0]
            children = psutil.Process(target.pid).children(recursive=True)
            core = next(
                child
                for child in children
                if child.name().startswith("VLLM::EngineCore")
            )
            execution = next(
                child for child in children if child.name().startswith("VLLM::Worker")
            )
            env["DYN_SYSTEM_PORT"] = str(ports[2])
            with owned_process(command, env, directory / "healthy.log") as healthy:
                await wait_until(lambda: len(client.instance_ids()) == 2)
                await wait_for_model(ports[0])
                await completion(ports[0])
                started = time.monotonic()
                if victim == "execution_worker":
                    core.suspend()
                    execution.kill()
                else:
                    core.kill()
                await wait_until(
                    lambda: target_id not in client.instance_ids(), timeout=3
                )
                withdrawn = time.monotonic()
                if victim == "execution_worker":
                    assert (
                        target.poll() is None
                    ), "Withdrawal must precede blocked cleanup"
                await wait_until(lambda: target.poll() is not None, timeout=8)
                assert target.returncode != 0
                assert healthy.poll() is None
                result = await completion(ports[0])
                assert result["choices"]
                (directory / "result.json").write_text(
                    json.dumps(
                        {
                            "victim": victim,
                            "withdrawal_seconds": withdrawn - started,
                            "exit_seconds": time.monotonic() - started,
                            "exit_code": target.returncode,
                            "remaining_instances": client.instance_ids(),
                            "healthy_worker_served": True,
                        },
                        indent=2,
                    )
                )
        finally:
            runtime.shutdown()


async def test_long_update_pause_sleep_and_worker_death(cluster):
    namespace, ports, env, command, directory = cluster
    env["DYN_SYSTEM_PORT"] = str(ports[1])
    with owned_process(command, env, directory / "lifecycle.log") as worker:
        runtime, client = await discovery_client(namespace)
        try:
            await wait_until(lambda: len(client.instance_ids()) == 1)
            await wait_for_model(ports[0])
            await completion(ports[0])
            await post(ports[1], "/engine/pause_generation", {})
            await asyncio.sleep(1)
            assert worker.poll() is None
            await post(
                ports[1],
                "/engine/update_weights_from_distributed",
                {
                    "engine_rpc": "health_test_block",
                    "delay": 2,
                    "reset_prefix_cache": False,
                },
            )
            assert worker.poll() is None
            await post(ports[1], "/engine/resume_generation", {})
            await completion(ports[0])
            await post(ports[1], "/engine/control/sleep", {"level": 1})
            await wait_until(lambda: not client.instance_ids())
            await asyncio.sleep(1)
            assert worker.poll() is None
            await post(ports[1], "/engine/control/wake_up", {})
            await wait_until(lambda: len(client.instance_ids()) == 1)
            await completion(ports[0])
            await post(ports[1], "/engine/pause_generation", {})
            update = asyncio.create_task(
                post(
                    ports[1],
                    "/engine/update_weights_from_distributed",
                    {
                        "engine_rpc": "health_test_block",
                        "delay": 15,
                        "reset_prefix_cache": False,
                    },
                )
            )
            await asyncio.sleep(0.5)
            children = psutil.Process(worker.pid).children(recursive=True)
            execution = next(
                child for child in children if child.name().startswith("VLLM::Worker")
            )
            execution.kill()
            await wait_until(lambda: not client.instance_ids(), timeout=3)
            await wait_until(lambda: worker.poll() is not None, timeout=8)
            await asyncio.gather(update, return_exceptions=True)
            assert worker.returncode != 0
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "pause_survived": True,
                        "sleep_survived": True,
                        "long_control_rpc_survived": True,
                        "inference_after_wake": True,
                        "death_during_control_rpc_detected": True,
                        "exit_code": worker.returncode,
                    },
                    indent=2,
                )
            )
        finally:
            runtime.shutdown()
