# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Timing notes (measured in an SGLang-enabled container):
# - GPU-1 subset (`-m "gpu_1"`): 92.35s total for 2 tests (+ 1 skipped).
# These tests load a real model and can be slow/flaky when GPU resources are contended,
# so we set explicit pytest timeouts to fail fast on hangs (see per-test markers below).
import logging
import os
import time
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (  # utilities
    _test_router_basic,
    _test_router_decisions,
    _test_router_indexers_sync,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.constants import DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sglang,
    pytest.mark.model(MODEL_NAME),
]
SPEEDUP_RATIO = 10.0
NUM_REQUESTS = 10
PAGE_SIZE = 16  # SGLang uses "page_size" instead of "block_size"


def allocate_frontend_ports(request, count: int) -> list[int]:
    """Allocate random free frontend ports for xdist-safe execution."""
    ports = allocate_ports(count, DefaultPort.FRONTEND.value)
    request.addfinalizer(lambda: deallocate_ports(ports))
    return ports


# Shared test payload for all tests
TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "In a quiet meadow tucked between rolling hills, a plump gray rabbit nibbled on clover beneath the shade of a gnarled oak tree. Its ears twitched at the faint rustle of leaves, but it remained calm, confident in the safety of its burrow just a few hops away. The late afternoon sun warmed its fur, and tiny dust motes danced in the golden light as bees hummed lazily nearby. Though the rabbit lived a simple life, every day was an adventure of scents, shadows, and snacks—an endless search for the tastiest patch of greens and the softest spot to nap.",
        }
    ],
    "stream": True,
    "max_tokens": 10,
}

# Shared SGLang configuration for all tests
# mem_fraction_static limits actual VRAM allocation (required for multi-worker on same GPU)
SGLANG_ARGS: Dict[str, Any] = {
    "page_size": PAGE_SIZE,
    "model": MODEL_NAME,
    "mem_fraction_static": 0.4,  # Limit VRAM allocation per worker (equivalent to vLLM's gpu_memory_utilization)
    "context_length": 1024,  # Limit context length to reduce KV cache size (equivalent to vLLM's max_model_len)
    "disable_cuda_graph": True,  # Disable CUDA graphs for faster startup & lower memory (equivalent to vLLM's enforce_eager)
}


class SGLangProcess:
    """Manages SGLang workers using dynamo.sglang (HTTP API + KV events).

    This is a drop-in replacement for MockerProcess that uses real SGLang workers.
    The key difference: dynamo.sglang automatically handles:
    - HTTP API serving
    - KV cache event publishing (ZMQ → NATS bridge)
    - Integration with dynamo.frontend router
    """

    def __init__(
        self,
        request,
        sglang_args: Optional[Dict[str, Any]] = None,
        num_workers: int = 2,
        single_gpu: bool = False,
        data_parallel_size: Optional[int] = None,
    ):
        """Initialize SGLang workers with dynamo integration.

        Args:
            request: pytest request fixture for log directory
            sglang_args: Configuration dict with keys:
                - page_size: KV cache page size (default: 16)
                - model: Model name/path (default: TinyLlama-1.1B)
                - mem_fraction_static: Fraction of GPU memory to allocate (optional)
                - context_length: Maximum sequence length (optional)
                - disable_cuda_graph: Disable CUDA graphs (default: False)
            num_workers: Number of SGLang worker processes
            single_gpu: If True, all workers share GPU 0
            data_parallel_size: If set, enables data parallelism with this many ranks (num_workers must equal data_parallel_size)
        """
        # Generate unique namespace for isolation
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "backend"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.worker_processes = []

        if sglang_args is None:
            sglang_args = {}

        page_size = sglang_args.get("page_size", PAGE_SIZE)
        model = sglang_args.get("model", MODEL_NAME)
        mem_fraction_static = sglang_args.get("mem_fraction_static")
        context_length = sglang_args.get("context_length")
        disable_cuda_graph = sglang_args.get("disable_cuda_graph", False)

        self.model_name = model

        for worker_idx in range(num_workers):
            # Calculate GPU device for this process
            if single_gpu:
                # Force all processes to GPU 0 (for single-GPU testing)
                gpu_device = "0"
            elif data_parallel_size is not None:
                # Worker sees dp_rank GPUs (each DP rank gets its own GPU)
                worker_start_gpu = worker_idx * data_parallel_size
                gpu_device = ",".join(
                    str(i)
                    for i in range(
                        worker_start_gpu, worker_start_gpu + data_parallel_size
                    )
                )
            else:
                # No DP; worker sees one GPU
                gpu_device = str(worker_idx)

            command = [
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                model,
                "--page-size",
                str(page_size),
            ]

            # Disable CUDA graphs for faster startup & lower memory
            if disable_cuda_graph:
                command.append("--disable-cuda-graph")

            # Limit VRAM allocation (required for multi-worker on same GPU)
            if mem_fraction_static is not None:
                command.extend(["--mem-fraction-static", str(mem_fraction_static)])

            # Add optional context_length if specified
            if context_length is not None:
                command.extend(["--context-length", str(context_length)])

            if data_parallel_size is not None:
                # Add DP configuration
                command.extend(
                    [
                        "--dp-size",
                        str(data_parallel_size),
                    ]
                )

            # Add per-worker KV events config for ZMQ publishing
            # Each worker needs a unique port to avoid conflicts
            kv_events_port = 20080 + worker_idx
            kv_events_config = f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{kv_events_port}"}}'
            command.extend(["--kv-events-config", kv_events_config])

            env = os.environ.copy()  # Copy parent environment
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_device,
                    "DYN_NAMESPACE": self.namespace,
                    "PYTHONHASHSEED": "0",  # for deterministic event id's
                }
            )

            # Create managed process for the worker
            process = ManagedProcess(
                command=command,
                env=env,
                timeout=120,  # Allow time for model loading
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_existing=False,
            )
            self.worker_processes.append(process)
            if data_parallel_size is not None:
                logger.info(
                    f"Created {data_parallel_size} DP ranks per worker on GPU(s) {gpu_device} "
                    f"(mem_frac={mem_fraction_static}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )
            else:
                logger.info(
                    f"Created SGLang worker {worker_idx} on GPU {gpu_device} "
                    f"(mem_frac={mem_fraction_static}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )

    def __enter__(self):
        """Start all SGLang worker processes with sequential initialization.

        Workers are started sequentially with a delay between each to avoid
        resource contention during initialization. This prevents
        shared memory handle allocation failures when multiple workers
        try to initialize simultaneously on the same GPU.
        """
        logger.info(
            f"[SGLangProcess] Starting {len(self.worker_processes)} worker processes sequentially..."
        )

        # Start each process sequentially, waiting for initialization before next
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[SGLangProcess] Starting SGLang worker {i}...")
            try:
                # Manually initialize the process without blocking on health checks
                process._logger = logging.getLogger(process.__class__.__name__)
                process._command_name = process.command[0]
                os.makedirs(process.log_dir, exist_ok=True)
                log_name = f"{process._command_name}.log.txt"
                process._log_path = os.path.join(process.log_dir, log_name)

                if process.data_dir:
                    process._remove_directory(process.data_dir)

                process._terminate_existing()
                logger.info(
                    f"[SGLangProcess] Launching process {i} (pid will be assigned)..."
                )
                process._start_process()  # Start the process but don't wait
                logger.info(
                    f"[SGLangProcess] Worker {i} launched with PID: {process.proc.pid if process.proc else 'unknown'}"
                )
                time.sleep(process.delayed_start)

                # Wait for initialization before starting next worker
                # This prevents shared memory contention
                if i < len(self.worker_processes) - 1:
                    init_delay = 5  # seconds
                    logger.info(
                        f"[SGLangProcess] Waiting {init_delay}s for worker {i} to initialize before starting next worker..."
                    )
                    time.sleep(init_delay)

            except Exception:
                logger.exception(f"[SGLangProcess] Failed to start worker {i}")
                # Clean up on failure
                try:
                    process.__exit__(None, None, None)
                except Exception as cleanup_err:
                    logger.warning(
                        f"[SGLangProcess] Error during cleanup: {cleanup_err}"
                    )
                raise

        logger.info(
            f"[SGLangProcess] All {len(self.worker_processes)} workers launched with sequential initialization."
        )
        logger.info("[SGLangProcess] Waiting for health checks to complete...")

        # Now wait for health checks for all processes
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[SGLangProcess] Checking health for worker {i}...")
            try:
                elapsed = process._check_ports(process.timeout)
                process._check_urls(process.timeout - elapsed)
                process._check_funcs(process.timeout - elapsed)
                logger.info(f"[SGLangProcess] Worker {i} health checks passed")
            except Exception:
                logger.error(f"[SGLangProcess] Worker {i} health check failed")
                # Clean up all processes on failure
                self.__exit__(None, None, None)
                raise

        logger.info(
            "[SGLangProcess] All workers started successfully and passed health checks!"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all SGLang worker processes gracefully."""
        for i, process in enumerate(self.worker_processes):
            logger.info(f"Stopping SGLang worker {i}")
            process.__exit__(exc_type, exc_val, exc_tb)

        # Add delay to ensure full cleanup of NATS/ETCD/ZMQ resources
        # This prevents test isolation issues when running multiple tests
        logger.info("Waiting for SGLang worker resources to fully clean up...")
        time.sleep(2)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.timeout(150)  # ~3x average (~46s/test), rounded up
def test_sglang_kv_router_basic(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    """
    Quick e2e sanity test for KV router with SGLang engine instances.
    """

    # runtime_services starts etcd and nats
    N_SGLANG_WORKERS = 2
    logger.info(f"Starting SGLang KV router test with {N_SGLANG_WORKERS} workers")

    try:
        # Start SGLang workers
        logger.info(f"Starting {N_SGLANG_WORKERS} SGLang workers")
        sglang_workers = SGLangProcess(
            request,
            sglang_args=SGLANG_ARGS,
            num_workers=N_SGLANG_WORKERS,
            single_gpu=True,  # fit workers into one GPU
        )
        logger.info(f"All SGLang workers using namespace: {sglang_workers.namespace}")
        sglang_workers.__enter__()

        # Run basic router test (starts router internally and waits for workers to be ready)
        frontend_port = allocate_frontend_ports(request, 1)[0]
        _test_router_basic(
            engine_workers=sglang_workers,
            block_size=PAGE_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            frontend_timeout=180,  # 3 minutes should be plenty for TinyLlama
            store_backend="etcd",  # Explicit for clarity
        )

    finally:
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.skip(reason="Broken by sglang changes")
# TODO: Re-enable this test once https://github.com/sgl-project/sglang/pull/14934 is merged
def test_router_decisions_sglang_multiple_workers(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    # runtime_services starts etcd and nats
    logger.info("Starting SGLang router prefix reuse test with two workers")
    N_WORKERS = 2

    try:
        # Start 2 worker processes on the same GPU
        logger.info("Starting 2 SGLang worker processes on single GPU (mem_frac=0.4)")
        sglang_workers = SGLangProcess(
            request,
            sglang_args=SGLANG_ARGS,
            num_workers=N_WORKERS,
            single_gpu=True,  # Worker uses GPU 0
        )
        logger.info(f"All SGLang workers using namespace: {sglang_workers.namespace}")

        # Initialize SGLang workers
        sglang_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        namespace = runtime.namespace(sglang_workers.namespace)
        component = namespace.component("backend")
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            sglang_workers, endpoint, MODEL_NAME, request, test_dp_rank=False
        )

    finally:
        # Clean up SGLang workers
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)


@pytest.mark.gpu_2
@pytest.mark.timeout(600)  # 10 min max (multi-GPU + DP startup variance)
def test_router_decisions_sglang_dp(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    """Validate KV cache prefix reuse with SGLang by sending progressive requests with overlapping prefixes.
    Same flow as test_router_decisions_sglang_multiple_workers; force first request to (worker_id, dp_rank=1).
    Dump events from router and verify:
        * All but one (worker_id, dp_rank) should have no events (due to prefix reuse)
        * The (worker_id, dp_rank) with events should have exactly 4 events (one per request)
        * All events should be on the forced (worker_id, dp_rank=1) (verifying forced routing and prefix reuse)
    """
    N_WORKERS = 1
    DP_SIZE = 2

    try:
        logger.info("Starting 2 SGLang DP ranks (dp_size=2) (mem_frac=0.4)")
        sglang_workers = SGLangProcess(
            request,
            sglang_args=SGLANG_ARGS,
            num_workers=N_WORKERS,  # Ignored when data_parallel_size is set
            single_gpu=False,
            data_parallel_size=DP_SIZE,  # Creates DP_SIZE processes (one per rank)
        )
        logger.info(f"All SGLang workers using namespace: {sglang_workers.namespace}")
        sglang_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        # Use the namespace from the SGLang workers
        namespace = runtime.namespace(sglang_workers.namespace)
        component = namespace.component("backend")  # endpoint is backend.generate
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            sglang_workers, endpoint, MODEL_NAME, request, test_dp_rank=True
        )

    finally:
        # Clean up SGLang workers
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.timeout(150)  # ~3x average (~46s/test), rounded up
def test_sglang_indexers_sync(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    """
    Test that two KV routers have synchronized indexer states after processing requests
    with SGLang workers. This test verifies that both routers converge to the same internal state.
    """
    logger.info("Starting SGLang indexers sync test")
    N_SGLANG_WORKERS = 2

    try:
        # Start SGLang workers
        logger.info(f"Starting {N_SGLANG_WORKERS} SGLang workers")
        sglang_workers = SGLangProcess(
            request,
            sglang_args=SGLANG_ARGS,
            num_workers=N_SGLANG_WORKERS,
            single_gpu=True,  # fit workers into one GPU
        )
        logger.info(f"All SGLang workers using namespace: {sglang_workers.namespace}")
        sglang_workers.__enter__()

        # Use the common test implementation (creates its own runtimes for each router)
        # Note: Consumer verification is done inside _test_router_indexers_sync while routers are alive
        _test_router_indexers_sync(
            engine_workers=sglang_workers,
            block_size=PAGE_SIZE,
            model_name=MODEL_NAME,
            num_workers=N_SGLANG_WORKERS,
            store_backend="etcd",
        )

        logger.info("SGLang indexers sync test completed successfully")

    finally:
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)
