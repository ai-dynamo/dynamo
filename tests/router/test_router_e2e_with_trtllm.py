# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Timing notes (measured in a TRT-LLM-enabled container):
# - GPU-1 subset (`-m "gpu_1"`): 136.36s total for 3 tests.
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
TRTLLM_BLOCK_SIZE = 32  # fixed internally to 32

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.trtllm,
    pytest.mark.model(MODEL_NAME),
]
NUM_REQUESTS = 10


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

# Shared TRT-LLM configuration for all tests
# free_gpu_memory_fraction limits actual VRAM allocation (required for multi-worker on same GPU)
TRTLLM_ARGS: Dict[str, Any] = {
    "kv_block_size": TRTLLM_BLOCK_SIZE,
    "model": MODEL_NAME,
    "free_gpu_memory_fraction": 0.4,  # Limit VRAM allocation per worker
    "max_seq_len": 1024,  # Limit context length to reduce KV cache size
}


class TRTLLMProcess:
    """Manages TRT-LLM workers using dynamo.trtllm (HTTP API + KV events).

    This is a drop-in replacement for MockerProcess that uses real TRT-LLM workers.
    The key difference: dynamo.trtllm automatically handles:
    - HTTP API serving
    - KV cache event publishing
    - Integration with dynamo.frontend router
    """

    def __init__(
        self,
        request,
        trtllm_args: Optional[Dict[str, Any]] = None,
        num_workers: int = 2,
        single_gpu: bool = False,
    ):
        """Initialize TRT-LLM workers with dynamo integration.

        Args:
            request: pytest request fixture for log directory
            trtllm_args: Configuration dict with keys:
                - kv_block_size: KV cache block size (default: 32)
                - model: Model name/path (default: TinyLlama-1.1B)
                - free_gpu_memory_fraction: Fraction of GPU memory to allocate (optional)
                - max_seq_len: Maximum sequence length (optional)
            num_workers: Number of TRT-LLM worker processes
            single_gpu: If True, all workers share GPU 0

        Note: TRT-LLM doesn't support data parallelism like vLLM (dp_rank is always 0).
              Tensor parallelism (TP) is supported but creates 1 worker spanning multiple GPUs,
              not multiple routing targets.
        """
        # Generate unique namespace for isolation
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "tensorrt_llm"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.worker_processes = []

        if trtllm_args is None:
            trtllm_args = {}

        model = trtllm_args.get("model", MODEL_NAME)
        free_gpu_memory_fraction = trtllm_args.get("free_gpu_memory_fraction")
        max_seq_len = trtllm_args.get("max_seq_len")

        self.model_name = model

        for worker_idx in range(num_workers):
            # Calculate GPU device for this process
            if single_gpu:
                # Force all processes to GPU 0 (for single-GPU testing)
                gpu_device = "0"
            else:
                # Each worker sees one GPU
                gpu_device = str(worker_idx)

            # Single-node TRT-LLM workers use python3 -m dynamo.trtllm directly
            # (trtllm-llmapi-launch is only needed for multi-node MPI deployments)
            command = [
                "python3",
                "-m",
                "dynamo.trtllm",
                "--model-path",
                model,
                "--kv-block-size",
                str(TRTLLM_BLOCK_SIZE),
                # Enable KV events publishing for router integration
                "--publish-events-and-metrics",
            ]

            # Limit VRAM allocation (required for multi-worker on same GPU)
            if free_gpu_memory_fraction is not None:
                command.extend(
                    ["--free-gpu-memory-fraction", str(free_gpu_memory_fraction)]
                )

            # Add optional max_seq_len if specified
            if max_seq_len is not None:
                command.extend(["--max-seq-len", str(max_seq_len)])

            # Each TRT-LLM worker needs a unique DYN_SYSTEM_PORT to avoid conflicts.
            # See examples/backends/trtllm/launch/disagg_same_gpu.sh for reference.
            system_port = 8081 + worker_idx

            env = os.environ.copy()  # Copy parent environment
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_device,
                    "DYN_NAMESPACE": self.namespace,
                    "PYTHONHASHSEED": "0",  # for deterministic event id's
                    # Set unique system port for each worker to avoid port conflicts
                    "DYN_SYSTEM_PORT": str(system_port),
                }
            )

            # Create managed process for the worker
            process = ManagedProcess(
                command=command,
                env=env,
                timeout=180,  # Allow time for model loading (TRT-LLM may take longer)
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_existing=False,
            )
            self.worker_processes.append(process)
            logger.info(
                f"Created TRT-LLM worker {worker_idx} on GPU {gpu_device} "
                f"(gpu_mem_frac={free_gpu_memory_fraction}, system_port={system_port}) "
                f"with endpoint: {self.endpoint}"
            )

    def __enter__(self):
        """Start all TRT-LLM worker processes with sequential initialization.

        Workers are started sequentially with a delay between each to avoid
        resource contention during initialization. This prevents
        MPI initialization conflicts when multiple workers
        try to initialize simultaneously on the same GPU.
        """
        logger.info(
            f"[TRTLLMProcess] Starting {len(self.worker_processes)} worker processes sequentially..."
        )

        # Start each process sequentially, waiting for initialization before next
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[TRTLLMProcess] Starting TRT-LLM worker {i}...")
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
                    f"[TRTLLMProcess] Launching process {i} (pid will be assigned)..."
                )
                process._start_process()  # Start the process but don't wait
                logger.info(
                    f"[TRTLLMProcess] Worker {i} launched with PID: {process.proc.pid if process.proc else 'unknown'}"
                )
                time.sleep(process.delayed_start)

                # Wait for initialization before starting next worker
                # This prevents MPI initialization conflicts
                if i < len(self.worker_processes) - 1:
                    init_delay = 5  # seconds
                    logger.info(
                        f"[TRTLLMProcess] Waiting {init_delay}s for worker {i} to initialize before starting next worker..."
                    )
                    time.sleep(init_delay)

            except Exception:
                logger.exception(f"[TRTLLMProcess] Failed to start worker {i}")
                # Clean up on failure
                try:
                    process.__exit__(None, None, None)
                except Exception as cleanup_err:
                    logger.warning(
                        f"[TRTLLMProcess] Error during cleanup: {cleanup_err}"
                    )
                raise

        logger.info(
            f"[TRTLLMProcess] All {len(self.worker_processes)} workers launched with sequential initialization."
        )
        logger.info("[TRTLLMProcess] Waiting for health checks to complete...")

        # Now wait for health checks for all processes
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[TRTLLMProcess] Checking health for worker {i}...")
            try:
                elapsed = process._check_ports(process.timeout)
                process._check_urls(process.timeout - elapsed)
                process._check_funcs(process.timeout - elapsed)
                logger.info(f"[TRTLLMProcess] Worker {i} health checks passed")
            except Exception:
                logger.error(f"[TRTLLMProcess] Worker {i} health check failed")
                # Clean up all processes on failure
                self.__exit__(None, None, None)
                raise

        logger.info(
            "[TRTLLMProcess] All workers started successfully and passed health checks!"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all TRT-LLM worker processes gracefully."""
        for i, process in enumerate(self.worker_processes):
            logger.info(f"Stopping TRT-LLM worker {i}")
            process.__exit__(exc_type, exc_val, exc_tb)

        # Add delay to ensure full cleanup of NATS/ETCD/MPI resources
        # This prevents test isolation issues when running multiple tests
        logger.info("Waiting for TRT-LLM worker resources to fully clean up...")
        time.sleep(2)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.timeout(150)  # ~3x average (~45s/test), rounded up
def test_trtllm_kv_router_basic(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    """
    Quick e2e sanity test for KV router with TRT-LLM engine instances.
    """

    # runtime_services starts etcd and nats
    N_TRTLLM_WORKERS = 2
    logger.info(f"Starting TRT-LLM KV router test with {N_TRTLLM_WORKERS} workers")

    try:
        # Start TRT-LLM workers
        logger.info(f"Starting {N_TRTLLM_WORKERS} TRT-LLM workers")
        trtllm_workers = TRTLLMProcess(
            request,
            trtllm_args=TRTLLM_ARGS,
            num_workers=N_TRTLLM_WORKERS,
            single_gpu=True,  # fit workers into one GPU
        )
        logger.info(f"All TRT-LLM workers using namespace: {trtllm_workers.namespace}")
        trtllm_workers.__enter__()

        # Run basic router test (starts router internally and waits for workers to be ready)
        frontend_port = allocate_frontend_ports(request, 1)[0]
        _test_router_basic(
            engine_workers=trtllm_workers,
            block_size=TRTLLM_BLOCK_SIZE,
            request=request,
            frontend_port=frontend_port,
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            frontend_timeout=180,  # 3 minutes should be plenty for TinyLlama
            store_backend="etcd",  # Explicit for clarity
        )

    finally:
        if "trtllm_workers" in locals():
            trtllm_workers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.timeout(150)  # ~3x average (~45s/test), rounded up
def test_router_decisions_trtllm_multiple_workers(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    # runtime_services starts etcd and nats
    logger.info("Starting TRT-LLM router prefix reuse test with two workers")
    N_WORKERS = 2

    try:
        # Start 2 worker processes on the same GPU
        logger.info(
            "Starting 2 TRT-LLM worker processes on single GPU (gpu_mem_frac=0.4)"
        )
        trtllm_workers = TRTLLMProcess(
            request,
            trtllm_args=TRTLLM_ARGS,
            num_workers=N_WORKERS,
            single_gpu=True,  # Worker uses GPU 0
        )
        logger.info(f"All TRT-LLM workers using namespace: {trtllm_workers.namespace}")

        # Initialize TRT-LLM workers
        trtllm_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        namespace = runtime.namespace(trtllm_workers.namespace)
        component = namespace.component("tensorrt_llm")
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            trtllm_workers,
            endpoint,
            MODEL_NAME,
            request,
            test_dp_rank=False,
            block_size=TRTLLM_BLOCK_SIZE,
        )

    finally:
        # Clean up TRT-LLM workers
        if "trtllm_workers" in locals():
            trtllm_workers.__exit__(None, None, None)


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.timeout(150)  # ~3x average (~45s/test), rounded up
def test_trtllm_indexers_sync(
    request, runtime_services_dynamic_ports, predownload_models, set_ucx_tls_no_mm
):
    """
    Test that two KV routers have synchronized indexer states after processing requests
    with TRT-LLM workers. This test verifies that both routers converge to the same internal state.
    """
    logger.info("Starting TRT-LLM indexers sync test")
    N_TRTLLM_WORKERS = 2

    try:
        # Start TRT-LLM workers
        logger.info(f"Starting {N_TRTLLM_WORKERS} TRT-LLM workers")
        trtllm_workers = TRTLLMProcess(
            request,
            trtllm_args=TRTLLM_ARGS,
            num_workers=N_TRTLLM_WORKERS,
            single_gpu=True,  # fit workers into one GPU
        )
        logger.info(f"All TRT-LLM workers using namespace: {trtllm_workers.namespace}")
        trtllm_workers.__enter__()

        # Use the common test implementation (creates its own runtimes for each router)
        # Note: Consumer verification is done inside _test_router_indexers_sync while routers are alive
        _test_router_indexers_sync(
            engine_workers=trtllm_workers,
            block_size=TRTLLM_BLOCK_SIZE,
            model_name=MODEL_NAME,
            num_workers=N_TRTLLM_WORKERS,
            store_backend="etcd",
        )

        logger.info("TRT-LLM indexers sync test completed successfully")

    finally:
        if "trtllm_workers" in locals():
            trtllm_workers.__exit__(None, None, None)
