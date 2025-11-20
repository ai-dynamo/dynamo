# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (  # utilities
    _test_router_basic,
    _test_router_decisions,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPEEDUP_RATIO = 10.0
PORTS = [
    8011,
    8022,
]  # Frontend ports: use PORTS[0] for single router, PORTS for multi-router
NUM_REQUESTS = 10
BLOCK_SIZE = 16

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


class VLLMProcess:
    """Manages vLLM workers using dynamo.vllm (HTTP API + KV events).

    This is a drop-in replacement for MockerProcess that uses real vLLM workers.
    The key difference: dynamo.vllm automatically handles:
    - HTTP API serving
    - KV cache event publishing (ZMQ → NATS bridge)
    - Integration with dynamo.frontend router
    """

    def __init__(
        self,
        request,
        vllm_args: Optional[Dict[str, Any]] = None,
        num_workers: int = 2,
        single_gpu: bool = False,
        data_parallel_size: Optional[int] = None,
    ):
        """Initialize vLLM workers with dynamo integration.

        Args:
            request: pytest request fixture for log directory
            vllm_args: Configuration dict with keys:
                - block_size: KV cache block size (default: 16)
                - model: Model name/path (default: TinyLlama-1.1B)
                - gpu_memory_utilization: GPU memory fraction per worker (default: 0.9)
                - max_model_len: Maximum sequence length (optional)
                - speedup_ratio: IGNORED (vLLM runs at real speed)
            num_workers: Number of vLLM worker processes
            single_gpu: If True, all workers share GPU 0 (requires gpu_memory_utilization < 1.0/num_workers)
            data_parallel_size: If set, enables data parallelism with this many ranks (num_workers must equal data_parallel_size)
        """
        # Generate unique namespace for isolation
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "backend"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.worker_processes = []

        if vllm_args is None:
            vllm_args = {}

        block_size = vllm_args.get("block_size", BLOCK_SIZE)
        model = vllm_args.get("model", MODEL_NAME)
        gpu_memory_utilization = vllm_args.get("gpu_memory_utilization", 0.9)
        max_model_len = vllm_args.get("max_model_len")

        self.model_name = model

        # Create vLLM worker processes
        # Matches test.sh behavior:
        # - When data_parallel_size is set, launch one process per DP rank
        # - Each process gets --data-parallel-rank and --data-parallel-size
        # - Each process runs on its own GPU via CUDA_VISIBLE_DEVICES
        # - --connector nixl enables KV cache transfer between ranks

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
                "dynamo.vllm",
                "--model",
                model,
                "--block-size",
                str(block_size),
                "--enforce-eager",  # Disable CUDA graphs for faster startup
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
            ]

            # Add optional max_model_len if specified
            if max_model_len is not None:
                command.extend(["--max-model-len", str(max_model_len)])

            if data_parallel_size is not None:
                # Add DP configuration for external load balancing
                # See: https://docs.vllm.ai/en/v0.10.0/serving/data_parallel_deployment.html#external-load-balancing
                command.extend(
                    [
                        "--data-parallel-size",
                        str(data_parallel_size),
                        # "--data-parallel-address", "127.0.0.1",  # Required for DP coordination
                        # "--data-parallel-rpc-port", "13345",  # RPC port for DP coordination
                        # "--connector", "nixl",  # Required for KV transfer between DP ranks
                    ]
                )

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
                    f"(gpu_memory_utilization={gpu_memory_utilization}) "
                    f"with endpoint: {self.endpoint}"
                )
            else:
                logger.info(
                    f"Created vLLM worker {worker_idx} on GPU {gpu_device} "
                    f"(gpu_memory_utilization={gpu_memory_utilization}) "
                    f"with endpoint: {self.endpoint}"
                )

    def __enter__(self):
        """Start all vLLM worker processes with sequential initialization.

        Workers are started sequentially with a delay between each to avoid
        NIXL/UCX resource contention during initialization. This prevents
        UCX shared memory handle allocation failures when multiple workers
        try to initialize simultaneously on the same GPU.
        """
        logger.info(
            f"[VLLMProcess] Starting {len(self.worker_processes)} worker processes sequentially..."
        )

        # Start each process sequentially, waiting for NIXL initialization before next
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[VLLMProcess] Starting vLLM worker {i}...")
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
                    f"[VLLMProcess] Launching process {i} (pid will be assigned)..."
                )
                process._start_process()  # Start the process but don't wait
                logger.info(
                    f"[VLLMProcess] Worker {i} launched with PID: {process.proc.pid if process.proc else 'unknown'}"
                )
                time.sleep(process.delayed_start)

                # Wait for NIXL initialization before starting next worker
                # This prevents UCX shared memory contention
                if i < len(self.worker_processes) - 1:
                    nixl_init_delay = 5  # seconds
                    logger.info(
                        f"[VLLMProcess] Waiting {nixl_init_delay}s for worker {i} to initialize NIXL before starting next worker..."
                    )
                    time.sleep(nixl_init_delay)

            except Exception:
                logger.exception(f"[VLLMProcess] Failed to start worker {i}")
                # Clean up on failure
                try:
                    process.__exit__(None, None, None)
                except Exception as cleanup_err:
                    logger.warning(f"[VLLMProcess] Error during cleanup: {cleanup_err}")
                raise

        logger.info(
            f"[VLLMProcess] All {len(self.worker_processes)} workers launched with sequential initialization."
        )
        logger.info("[VLLMProcess] Waiting for health checks to complete...")

        # Now wait for health checks for all processes
        for i, process in enumerate(self.worker_processes):
            logger.info(f"[VLLMProcess] Checking health for worker {i}...")
            try:
                elapsed = process._check_ports(process.timeout)
                process._check_urls(process.timeout - elapsed)
                process._check_funcs(process.timeout - elapsed)
                logger.info(f"[VLLMProcess] Worker {i} health checks passed")
            except Exception:
                logger.error(f"[VLLMProcess] Worker {i} health check failed")
                # Clean up all processes on failure
                self.__exit__(None, None, None)
                raise

        logger.info(
            "[VLLMProcess] All workers started successfully and passed health checks!"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all vLLM worker processes gracefully."""
        for i, process in enumerate(self.worker_processes):
            logger.info(f"Stopping vLLM worker {i}")
            process.__exit__(exc_type, exc_val, exc_tb)

        # Add delay to ensure full cleanup of NATS/ETCD/ZMQ resources
        # This prevents test isolation issues when running multiple tests
        logger.info("Waiting for vLLM worker resources to fully clean up...")
        time.sleep(2)


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.vllm
@pytest.mark.skip(reason="All vLLM tests disabled for now")
@pytest.mark.model(MODEL_NAME)
def test_vllm_kv_router_basic(request, runtime_services, predownload_tokenizers):
    """
    Quick e2e sanity test for KV router with vLLM engine instances.
    """

    # runtime_services starts etcd and nats
    N_VLLM_WORKERS = 2
    logger.info(f"Starting vLLM KV router test with {N_VLLM_WORKERS} workers")

    vllm_args = {
        "block_size": BLOCK_SIZE,
        "model": MODEL_NAME,
        "gpu_memory_utilization": 0.35,
        "max_model_len": 1024,  # Limit context length to reduce KV cache size
    }

    try:
        # Start vLLM workers
        logger.info(f"Starting {N_VLLM_WORKERS} vLLM workers")
        vllm_workers = VLLMProcess(
            request,
            vllm_args=vllm_args,
            num_workers=N_VLLM_WORKERS,
            single_gpu=True,  # fit workers into one GPU
        )
        logger.info(f"All vLLM workers using namespace: {vllm_workers.namespace}")
        vllm_workers.__enter__()

        # Run basic router test (starts router internally, vLLM workers need frontend readiness check)
        _test_router_basic(
            engine_workers=vllm_workers,
            block_size=BLOCK_SIZE,
            request=request,
            frontend_port=PORTS[0],
            test_payload=TEST_PAYLOAD,
            num_requests=NUM_REQUESTS,
            wait_for_frontend=True,  # vLLM workers need time to load models
            frontend_timeout=180,  # 3 minutes should be plenty for TinyLlama
            store_backend="etcd",  # Explicit for clarity
        )

    finally:
        if "vllm_workers" in locals():
            vllm_workers.__exit__(None, None, None)


@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.gpu_1
@pytest.mark.skip(reason="All vLLM tests disabled for now")
@pytest.mark.model(MODEL_NAME)
def test_router_decisions_vllm_multiple_workers(
    request, runtime_services, predownload_tokenizers
):
    # runtime_services starts etcd and nats
    logger.info("Starting vLLM router prefix reuse test with two workers")

    # Create vLLM args - one worker with dp_size=2, sharing GPU 0
    vllm_args = {
        "block_size": BLOCK_SIZE,
        "model": MODEL_NAME,
        "gpu_memory_utilization": 0.35,
        "max_model_len": 1024,  # Limit context length to reduce KV cache size
    }
    N_WORKERS = 2

    try:
        # Start 2 worker processes (dp_rank 0 and dp_rank 1) on the same GPU
        logger.info(
            "Starting 2 vLLM worker processes with dp_size=2 on single GPU (gpu_memory_utilization=0.35, max_model_len=1024)"
        )
        vllm_workers = VLLMProcess(
            request,
            vllm_args=vllm_args,
            num_workers=N_WORKERS,  # One worker process with dp_size=2
            single_gpu=True,  # Worker uses GPU 0
        )
        logger.info(f"All vLLM workers using namespace: {vllm_workers.namespace}")

        # Initialize vLLM workers
        vllm_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        namespace = runtime.namespace(vllm_workers.namespace)
        component = namespace.component("backend")
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            vllm_workers, endpoint, MODEL_NAME, request, test_dp_rank=False
        )

    finally:
        # Clean up vLLM workers
        if "vllm_workers" in locals():
            vllm_workers.__exit__(None, None, None)


@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.gpu_2
@pytest.mark.skip(reason="All vLLM tests disabled for now")
@pytest.mark.model(MODEL_NAME)
def test_router_decisions_vllm_dp(request, runtime_services, predownload_tokenizers):
    """Validate KV cache prefix reuse with vLLM by sending progressive requests with overlapping prefixes.
    Same flow as test_router_decisions_vllm_multiple_workers; force first request to (worker_id, dp_rank=1).
    Dump events from router and verify:
        * All but one (worker_id, dp_rank) should have no events (due to prefix reuse)
        * The (worker_id, dp_rank) with events should have exactly 4 events (one per request)
        * All events should be on the forced (worker_id, dp_rank=1) (verifying forced routing and prefix reuse)
    """
    # Create vLLM args - one worker with dp_size=2, sharing GPU 0
    vllm_args = {
        "block_size": BLOCK_SIZE,
        "model": MODEL_NAME,
        "gpu_memory_utilization": 0.35,
        "max_model_len": 1024,  # Limit context length to reduce KV cache size
    }
    N_WORKERS = 1
    DP_SIZE = 2

    try:
        logger.info(
            "Starting 2 vLLM DP ranks (dp_size=2) on single GPU (gpu_memory_utilization=0.35, max_model_len=1024)"
        )
        vllm_workers = VLLMProcess(
            request,
            vllm_args=vllm_args,
            num_workers=N_WORKERS,  # Ignored when data_parallel_size is set
            single_gpu=False,
            data_parallel_size=DP_SIZE,  # Creates DP_SIZE processes (one per rank)
        )
        logger.info(f"All vLLM workers using namespace: {vllm_workers.namespace}")
        vllm_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime()
        # Use the namespace from the vLLM workers
        namespace = runtime.namespace(vllm_workers.namespace)
        component = namespace.component("backend")  # endpoint is backend.generate
        endpoint = component.endpoint("generate")

        _test_router_decisions(
            vllm_workers, endpoint, MODEL_NAME, request, test_dp_rank=True
        )

    finally:
        # Clean up vLLM workers
        if "vllm_workers" in locals():
            vllm_workers.__exit__(None, None, None)
