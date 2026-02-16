# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router benchmark tests (production models, benchmark profile) for vLLM.

These tests follow the flow in benchmarks/router/README.md: etcd/NATS, workers,
router, and the same benchmark script (prefix_ratio_benchmark.py) with reduced
parameters. etcd and NATS are started by the runtime_services_dynamic_ports
fixture (no separate CI step).

This module does not import tests.router.common so it can run in environments
where dynamo.llm KvRouter bindings are not available (e.g. minimal CI wheels).

Local run with a smaller model (single GPU, no 120B):
  ROUTER_BENCHMARK_MODEL=facebook/opt-125m ROUTER_BENCHMARK_TP=1 \\
  python -m pytest tests/router/test_router_benchmark_vllm.py -v --timeout=600 \\
  -p no:mypy -o 'addopts=-v --timeout=600'

Requires: nats-server, etcd, and vLLM (run_engines.sh) on PATH.
"""

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest

from tests.utils.constants import DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports


class KVRouterProcess(ManagedProcess):
    """Manages the KV router process using dynamo.frontend (benchmark-only, no KvRouter import)."""

    def __init__(
        self,
        request,
        block_size: int,
        frontend_port: int,
        namespace: str,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        durable_kv_events: bool = False,
        **kwargs,
    ):
        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--kv-cache-block-size",
            str(block_size),
            "--router-mode",
            "kv",
            "--http-port",
            str(frontend_port),
            "--store-kv",
            store_backend,
            "--namespace",
            namespace,
        ]
        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane
        if durable_kv_events:
            env["DYN_DURABLE_KV_EVENTS"] = "true"
        super().__init__(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[frontend_port],
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", self._check_ready)
            ],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            **kwargs,
        )
        self.port = frontend_port

    def _check_ready(self, response):
        return response.status_code == 200


async def wait_for_frontend_ready(
    frontend_url: str, expected_num_workers: int = 2, timeout: int = 120
):
    """Wait for backend worker(s) to be ready via the HTTP frontend (OpenAI API)."""
    models_url = f"{frontend_url}/v1/models"
    chat_url = f"{frontend_url}/v1/chat/completions"
    start_time = asyncio.get_event_loop().time()
    logger_local = logging.getLogger(__name__)
    logger_local.info(
        "Waiting for %s workers on HTTP frontend (timeout=%ss)...",
        expected_num_workers,
        timeout,
    )
    model_name = None
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Timeout waiting for vLLM workers. Waited {elapsed:.1f}s, no workers registered."
            )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        if len(models) > 0:
                            model_name = models[0].get("id")
                            logger_local.info(
                                "Workers registered. Found %s model(s): %s",
                                len(models),
                                [m.get("id") for m in models],
                            )
                            break
        except Exception as e:
            logger_local.debug("Error checking models endpoint: %s", e)
        await asyncio.sleep(1)
    logger_local.info("Waiting for chat completions pipeline...")
    test_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
        "stream": False,
    }
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                "Timeout waiting for chat completions pipeline. Waited {:.1f}s.".format(
                    elapsed
                )
            )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(chat_url, json=test_payload) as response:
                    if response.status == 200:
                        logger_local.info("Chat completions pipeline ready!")
                        return
        except Exception as e:
            logger_local.debug("Error testing chat completions: %s", e)
        await asyncio.sleep(1)

logger = logging.getLogger(__name__)

# Production default; override with ROUTER_BENCHMARK_MODEL (and optionally ROUTER_BENCHMARK_TP) for local/small-model runs.
MODEL_NAME = os.environ.get("ROUTER_BENCHMARK_MODEL", "openai/gpt-oss-120b")
# Tensor parallel size: use env for local small-model runs (e.g. 1), else default 2 for production.
TENSOR_PARALLEL_SIZE = int(os.environ.get("ROUTER_BENCHMARK_TP", "2"))
BLOCK_SIZE = 64
# Default namespace used by dynamo.vllm when DYN_NAMESPACE is not set (see runtime_args.py).
DEFAULT_NAMESPACE = "dynamo"

# Minimal parameters for prefix_ratio_benchmark.py (same script as benchmarks/router/README.md)
BENCHMARK_REQUESTS = 5
BENCHMARK_CONCURRENCY = 2
BENCHMARK_ISL = 256
BENCHMARK_OSL = 32
BENCHMARK_PREFIX_RATIO = 0.5

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.router_benchmark,
    pytest.mark.nightly,
    pytest.mark.gpu_4,
    pytest.mark.vllm,
    pytest.mark.slow,
    pytest.mark.model(MODEL_NAME),
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("store_kv", ["etcd"], indirect=True)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize("durable_kv_events", [True], indirect=True)
def test_gpt_oss_tp2(
    request,
    runtime_services_dynamic_ports,
    request_plane,
    store_kv,
    durable_kv_events,
    tmp_path,
):
    """
    Router benchmark with GPT-OSS-120B, tensor parallelism 2.

    Follows benchmarks/router/README.md: launch 2 vLLM workers (TP=2, 4 GPUs),
    start Dynamo router, run the same prefix_ratio_benchmark.py with reduced
    parameters. etcd and NATS (with JetStream) are started by
    runtime_services_dynamic_ports (no separate step).
    """
    nats_process, etcd_process = runtime_services_dynamic_ports
    assert etcd_process is not None and nats_process is not None

    repo_root = Path(__file__).resolve().parent.parent.parent
    router_bench_dir = repo_root / "benchmarks" / "router"
    run_engines = router_bench_dir / "run_engines.sh"
    prefix_bench = router_bench_dir / "prefix_ratio_benchmark.py"
    if not run_engines.exists():
        pytest.skip(f"run_engines.sh not found at {run_engines}")
    if not prefix_bench.exists():
        pytest.skip(f"prefix_ratio_benchmark.py not found at {prefix_bench}")

    frontend_port = allocate_ports(1, DefaultPort.FRONTEND.value)[0]
    request.addfinalizer(lambda: deallocate_ports([frontend_port]))

    env = os.environ.copy()
    env.setdefault("DYNAMO_HOME", str(repo_root))
    if durable_kv_events:
        env["DYN_DURABLE_KV_EVENTS"] = "true"

    num_workers = 2 if TENSOR_PARALLEL_SIZE >= 2 else 1
    proc = subprocess.Popen(
        [
            "bash",
            str(run_engines),
            "--num-workers",
            str(num_workers),
            "--model-path",
            MODEL_NAME,
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
        ],
        cwd=str(repo_root),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    request.addfinalizer(lambda: _kill_process_group(proc))

    # Allow time for workers to load the model (longer for large models).
    model_load_wait = 600 if "gpt-oss-120b" in MODEL_NAME else 180
    logger.info(f"Waiting up to {model_load_wait}s for workers to load {MODEL_NAME}...")
    time.sleep(min(30, model_load_wait))  # Brief wait before starting router

    with KVRouterProcess(
        request,
        block_size=BLOCK_SIZE,
        frontend_port=frontend_port,
        namespace=DEFAULT_NAMESPACE,
        store_backend=store_kv,
        request_plane=request_plane,
        durable_kv_events=durable_kv_events,
    ):
        logger.info(f"Starting KV router on port {frontend_port}")
        frontend_url = f"http://localhost:{frontend_port}"

        import asyncio

        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=num_workers,
                timeout=model_load_wait - 30,
            )
        )

        # Same benchmark as benchmarks/router/README.md (Step 4), with reduced params.
        benchmark_out = tmp_path / "kv_router"
        benchmark_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(prefix_bench),
            "--url",
            frontend_url,
            "--model",
            MODEL_NAME,
            "--requests",
            str(BENCHMARK_REQUESTS),
            "--concurrency",
            str(BENCHMARK_CONCURRENCY),
            "--isl",
            str(BENCHMARK_ISL),
            "--osl",
            str(BENCHMARK_OSL),
            "--prefix-ratios",
            str(BENCHMARK_PREFIX_RATIO),
            "--output-dir",
            str(benchmark_out),
            "--collect-gpu-sku",
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
        ]
        logger.info(
            "Running prefix_ratio_benchmark.py (same as benchmarks/router/README.md)"
        )
        try:
            subprocess.run(
                cmd,
                cwd=str(router_bench_dir),
                env=env,
                check=True,
                timeout=300,
                capture_output=False,
            )
        except FileNotFoundError as e:
            pytest.skip(
                f"prefix_ratio_benchmark.py deps not available (e.g. aiperf): {e}"
            )
        # Copy benchmark results to test-results/ so CI can store them as artifacts
        results_json = benchmark_out / "results_summary.json"
        if results_json.exists():
            test_results_dir = repo_root / "test-results"
            test_results_dir.mkdir(parents=True, exist_ok=True)
            dest = test_results_dir / "router_benchmark_results.json"
            shutil.copy2(results_json, dest)
            logger.info("Benchmark results copied to %s for CI artifact", dest)
        logger.info("Router benchmark (test_gpt_oss_tp2) completed successfully")


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate process group so run_engines.sh and all worker children exit."""
    if proc.poll() is not None:
        return
    if proc.pid is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError, AttributeError) as e:
        logger.warning(f"Error killing process group: {e}")
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
