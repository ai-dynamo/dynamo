# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router benchmark tests (production models, benchmark profile) for vLLM.

These tests follow the flow in benchmarks/router/README.md: etcd/NATS, workers,
router, and the same benchmark script (prefix_ratio_benchmark.py) with reduced
parameters. etcd and NATS are started by the runtime_services_dynamic_ports
fixture (no separate CI step).
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.router.common import KVRouterProcess, wait_for_frontend_ready
from tests.utils.constants import DefaultPort
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"
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
def test_gpt_oss_tp2(
    request,
    runtime_services_dynamic_ports,
    request_plane,
    tmp_path,
):
    """
    Router benchmark with GPT-OSS-120B, tensor parallelism 2.

    Follows benchmarks/router/README.md: launch 2 vLLM workers (TP=2, 4 GPUs),
    start Dynamo router, run the same prefix_ratio_benchmark.py with reduced
    parameters. etcd and NATS are started by runtime_services_dynamic_ports
    (no separate step).
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

    proc = subprocess.Popen(
        [
            "bash",
            str(run_engines),
            "--num-workers",
            "2",
            "--model-path",
            MODEL_NAME,
            "--tensor-parallel-size",
            "2",
        ],
        cwd=str(repo_root),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    request.addfinalizer(lambda: _kill_process_group(proc))

    # Allow time for workers to load the large model (GPT-OSS-120B).
    model_load_wait = 600
    logger.info(f"Waiting up to {model_load_wait}s for workers to load {MODEL_NAME}...")
    time.sleep(min(120, model_load_wait))  # Minimum wait before starting router

    with KVRouterProcess(
        request,
        block_size=BLOCK_SIZE,
        frontend_port=frontend_port,
        namespace=DEFAULT_NAMESPACE,
        store_backend="etcd",
        request_plane=request_plane,
    ):
        logger.info(f"Starting KV router on port {frontend_port}")
        frontend_url = f"http://localhost:{frontend_port}"

        import asyncio

        asyncio.run(
            wait_for_frontend_ready(
                frontend_url=frontend_url,
                expected_num_workers=2,
                timeout=model_load_wait - 120,
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
