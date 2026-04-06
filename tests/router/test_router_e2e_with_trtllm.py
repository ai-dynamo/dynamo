# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Timing notes (measured in a TRT-LLM-enabled container):
# - GPU-1 subset (`-m "gpu_1"`): 136.36s total for 3 tests.
# These tests load a real model and can be slow/flaky when GPU resources are contended,
# so we set explicit pytest timeouts to fail fast on hangs (see per-test markers below).
import logging
import os
from typing import Any, Dict, Optional

import pytest

from tests.router.e2e_harness import (
    ManagedEngineProcessMixin,
    run_basic_router_test,
    run_disagg_router_decisions_test,
    run_indexers_sync_test,
    run_router_decisions_test,
)
from tests.router.helper import generate_random_suffix
from tests.utils.constants import DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRTLLM_BLOCK_SIZE = 32  # fixed internally to 32

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.trtllm,
    pytest.mark.model(MODEL_NAME),
]

# Shared TRT-LLM configuration for all tests
# free_gpu_memory_fraction limits actual VRAM allocation (required for multi-worker on same GPU)
TRTLLM_ARGS: Dict[str, Any] = {
    "kv_block_size": TRTLLM_BLOCK_SIZE,
    "model": MODEL_NAME,
    "free_gpu_memory_fraction": 0.4,  # Limit VRAM allocation per worker
    "max_seq_len": 1024,  # Limit context length to reduce KV cache size
}


class TRTLLMProcess(ManagedEngineProcessMixin):
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
        request_plane: str = "tcp",
        store_backend: str = "etcd",
        durable_kv_events: bool = False,
        namespace: Optional[str] = None,
        gpu_start_index: int = 0,
        disaggregation_mode: Optional[str] = None,
    ):
        """Initialize TRT-LLM workers with dynamo integration.

        Args:
            request: pytest request fixture for log directory
            trtllm_args: Configuration dict with keys:
                - kv_block_size: KV cache block size (default: 32)
                - model: Model name/path (default: TinyLlama-1.1B)
                - free_gpu_memory_fraction: Fraction of GPU memory to allocate (optional)
                - max_seq_len: Maximum sequence length (optional)
                - tensor_parallel_size: Number of GPUs for tensor parallelism (optional).
                  When attention DP is enabled, this sets the world size, which then is the attention_dp_size.
                - enable_attention_dp: If True, enable TRT-LLM attention data parallelism.
                  When enabled, attention_dp_size equals tensor_parallel_size, creating
                  multiple routing targets within a single TRT-LLM worker process.
            num_workers: Number of TRT-LLM worker processes
            single_gpu: If True, all workers share GPU 0
            request_plane: Request plane to use ("nats", "tcp", or "http"). Defaults to "tcp".
            store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
            durable_kv_events: If True, use JetStream for durable KV events. Defaults to False (NATS Core mode).

        Note: TRT-LLM supports two forms of parallelism for routing:
              1. Multiple workers (num_workers > 1): Each worker is a separate routing target
              2. Attention DP (enable_attention_dp=True in trtllm_args): Single worker with
                 multiple internal attention DP ranks, each being a separate routing target
        """
        # Generate unique namespace for isolation
        namespace_suffix = generate_random_suffix()
        self.namespace = namespace or f"test-namespace-{namespace_suffix}"
        self.component_name = (
            "prefill" if disaggregation_mode == "prefill" else "tensorrt_llm"
        )
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.worker_processes = []
        self.store_backend = store_backend

        # Dynamically allocate unique system ports (one per worker) to avoid
        # conflicts when tests run in parallel via pytest-xdist.
        self._system_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        request.addfinalizer(lambda: deallocate_ports(self._system_ports))

        if trtllm_args is None:
            trtllm_args = {}

        model = trtllm_args.get("model", MODEL_NAME)
        free_gpu_memory_fraction = trtllm_args.get("free_gpu_memory_fraction")
        max_seq_len = trtllm_args.get("max_seq_len")
        enable_attention_dp = trtllm_args.get("enable_attention_dp", False)
        tensor_parallel_size = trtllm_args.get("tensor_parallel_size")

        self.model_name = model

        for worker_idx in range(num_workers):
            # Calculate GPU device for this process
            if single_gpu:
                # Force all processes to GPU 0 (for single-GPU testing)
                gpu_device = str(gpu_start_index)
            elif enable_attention_dp and tensor_parallel_size:
                # For attention DP, TRT-LLM spawns tensor_parallel_size internal MPI workers.
                # So one process = two attention DP ranks = visibility in to both GPUs.
                gpu_device = ",".join(
                    str(gpu_start_index + i) for i in range(tensor_parallel_size)
                )
            else:
                # Each worker sees one GPU
                gpu_device = str(gpu_start_index + worker_idx)

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

            if disaggregation_mode is not None:
                command.extend(["--disaggregation-mode", disaggregation_mode])

            # Limit VRAM allocation (required for multi-worker on same GPU)
            if free_gpu_memory_fraction is not None:
                command.extend(
                    ["--free-gpu-memory-fraction", str(free_gpu_memory_fraction)]
                )

            # Add optional max_seq_len if specified
            if max_seq_len is not None:
                command.extend(["--max-seq-len", str(max_seq_len)])

            # Use --durable-kv-events to enable JetStream mode (local indexer disabled)
            if durable_kv_events:
                command.append("--durable-kv-events")

            # Set tensor parallel size if specified (needed for attention DP)
            if tensor_parallel_size is not None:
                command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

            # Enable attention data parallelism if requested
            if enable_attention_dp:
                command.append("--enable-attention-dp")

            # Each TRT-LLM worker needs a unique DYN_SYSTEM_PORT to avoid conflicts.
            # Ports are dynamically allocated for xdist-safe parallel execution.
            system_port = self._system_ports[worker_idx]

            env = os.environ.copy()  # Copy parent environment
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_device,
                "DYN_NAMESPACE": self.namespace,
                "DYN_REQUEST_PLANE": request_plane,
                "PYTHONHASHSEED": "0",  # for deterministic event id's
                "DYN_SYSTEM_PORT": str(system_port),
            }

            # Add DYN_FILE_KV if using file storage backend
            if self.store_backend == "file" and "DYN_FILE_KV" in os.environ:
                env_vars["DYN_FILE_KV"] = os.environ["DYN_FILE_KV"]

            env.update(env_vars)

            # Create managed process for the worker
            process = ManagedProcess(
                command=command,
                env=env,
                timeout=180,  # Allow time for model loading (TRT-LLM may take longer)
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_all_matching_process_names=False,
            )
            self.worker_processes.append(process)
            logger.info(
                f"Created TRT-LLM worker {worker_idx} on GPU {gpu_device} "
                f"(gpu_mem_frac={free_gpu_memory_fraction}, system_port={system_port}) "
                f"with endpoint: {self.endpoint}"
            )

    process_name = "TRT-LLM worker"
    cleanup_name = "TRT-LLM worker resources"


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(300)
def test_trtllm_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=TRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=TRTLLM_BLOCK_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(600)  # 10 min max (multi-GPU + DP startup variance)
def test_router_decisions_trtllm_attention_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    """Validate KV cache prefix reuse with TRTLLM by sending progressive requests with overlapping prefixes.
    Same flow as test_router_decisions_trtllm_multiple_workers; force first request to (worker_id, dp_rank=1).
    Dump events from router and verify:
        * All but one (worker_id, dp_rank) should have no events (due to prefix reuse)
        * The (worker_id, dp_rank) with events should have exactly 4 events (one per request)
        * All events should be on the forced (worker_id, dp_rank=1) (verifying forced routing and prefix reuse)
    """
    run_router_decisions_test(
        engine_process_cls=TRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args={
            **TRTLLM_ARGS,
            "enable_attention_dp": True,
            "tensor_parallel_size": 2,
        },
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=TRTLLM_BLOCK_SIZE,
        component_name="tensorrt_llm",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
    )


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(150)  # ~3x average (~45s/test), rounded up
def test_router_decisions_trtllm_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_router_decisions_test(
        engine_process_cls=TRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=TRTLLM_BLOCK_SIZE,
        component_name="tensorrt_llm",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
@pytest.mark.timeout(600)
def test_router_decisions_trtllm_disagg(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_disagg_router_decisions_test(
        engine_process_cls=TRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=TRTLLM_BLOCK_SIZE,
        num_prefill_workers=2,
        num_decode_workers=1,
        prefill_process_kwargs={
            "single_gpu": True,
            "gpu_start_index": 0,
            "disaggregation_mode": "prefill",
        },
        decode_process_kwargs={
            "single_gpu": True,
            "gpu_start_index": 1,
            "disaggregation_mode": "decode",
        },
    )


@pytest.mark.gpu_1
@pytest.mark.nightly
@pytest.mark.timeout(150)  # ~3x average (~45s/test), rounded up
@pytest.mark.parametrize(
    "store_backend,durable_kv_events,request_plane",
    [
        ("etcd", False, "tcp"),
    ],
    ids=["nats_core"],
    indirect=["durable_kv_events", "request_plane"],
)
def test_trtllm_indexers_sync(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    file_storage_backend,
    set_ucx_tls_no_mm,
    store_backend,
    durable_kv_events,
    request_plane,
):
    run_indexers_sync_test(
        engine_process_cls=TRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        request=request,
        runtime_services_dynamic_ports=runtime_services_dynamic_ports,
        store_backend=store_backend,
        durable_kv_events=durable_kv_events,
        request_plane=request_plane,
        block_size=TRTLLM_BLOCK_SIZE,
        model_name=MODEL_NAME,
        num_workers=2,
    )
