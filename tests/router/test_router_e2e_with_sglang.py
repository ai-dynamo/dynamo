# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Timing notes (measured in an SGLang-enabled container):
# - GPU-1 subset (`-m "gpu_1"`): 92.35s total for 2 tests (+ 1 skipped).
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

MODEL_NAME = "silence09/DeepSeek-R1-Small-2layers"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.sglang,
    pytest.mark.model(MODEL_NAME),
]
PAGE_SIZE = 16  # SGLang uses "page_size" instead of "block_size"

# Shared SGLang configuration for all tests
# mem_fraction_static limits actual VRAM allocation (required for multi-worker on same GPU)
SGLANG_ARGS: Dict[str, Any] = {
    "page_size": PAGE_SIZE,
    "model": MODEL_NAME,
    "mem_fraction_static": 0.4,  # Limit VRAM allocation per worker (equivalent to vLLM's gpu_memory_utilization)
    "context_length": 1024,  # Limit context length to reduce KV cache size (equivalent to vLLM's max_model_len)
    "disable_cuda_graph": True,  # Disable CUDA graphs for faster startup & lower memory (equivalent to vLLM's enforce_eager)
}


class SGLangProcess(ManagedEngineProcessMixin):
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
        request_plane: str = "tcp",
        store_backend: str = "etcd",
        durable_kv_events: bool = False,
        namespace: Optional[str] = None,
        gpu_start_index: int = 0,
        disaggregation_mode: Optional[str] = None,
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
            request_plane: Request plane to use ("nats", "tcp", or "http"). Defaults to "tcp".
            store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
            durable_kv_events: If True, use JetStream for durable KV events. Defaults to False (NATS Core mode).
        """
        # Generate unique namespace for isolation
        namespace_suffix = generate_random_suffix()
        self.namespace = namespace or f"test-namespace-{namespace_suffix}"
        self.component_name = (
            "prefill" if disaggregation_mode == "prefill" else "backend"
        )
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.data_parallel_size = data_parallel_size
        self.worker_processes = []
        self.store_backend = store_backend

        # Dynamically allocate unique system and KV event ports (one per worker)
        # to avoid conflicts in parallel test runs.
        self._system_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._kv_event_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        request.addfinalizer(
            lambda: deallocate_ports(self._system_ports + self._kv_event_ports)
        )

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
                gpu_device = str(gpu_start_index)
            elif data_parallel_size is not None:
                # Worker sees dp_rank GPUs (each DP rank gets its own GPU)
                worker_start_gpu = gpu_start_index + worker_idx * data_parallel_size
                gpu_device = ",".join(
                    str(i)
                    for i in range(
                        worker_start_gpu, worker_start_gpu + data_parallel_size
                    )
                )
            else:
                # No DP; worker sees one GPU
                gpu_device = str(gpu_start_index + worker_idx)

            command = [
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                model,
                "--page-size",
                str(page_size),
            ]

            # Disable CUDA graphs for faster startup & lower memory.
            # sglang 0.5.10+ has piecewise CUDA graphs (separate flag) that
            # consume ~7 GB during capture — must also be disabled for
            # multi-worker same-GPU tests to avoid OOM.
            if disable_cuda_graph:
                command.append("--disable-cuda-graph")
                command.append("--disable-piecewise-cuda-graph")

            # Limit VRAM allocation (required for multi-worker on same GPU)
            if mem_fraction_static is not None:
                command.extend(["--mem-fraction-static", str(mem_fraction_static)])

            # Add optional context_length if specified
            if context_length is not None:
                command.extend(["--context-length", str(context_length)])

            if disaggregation_mode is not None:
                command.extend(["--disaggregation-mode", disaggregation_mode])
                command.extend(["--disaggregation-transfer-backend", "nixl"])

            if data_parallel_size is not None:
                # Add DP configuration
                command.extend(
                    [
                        "--dp-size",
                        str(data_parallel_size),
                        "--tp-size",
                        str(data_parallel_size),
                        "--enable-dp-attention",
                    ]
                )

            # Add per-worker KV events config for ZMQ publishing
            # Ports are dynamically allocated for xdist-safe parallel execution.
            kv_events_port = self._kv_event_ports[worker_idx]
            kv_events_config = f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{kv_events_port}"}}'
            command.extend(["--kv-events-config", kv_events_config])

            # Use --durable-kv-events to enable JetStream mode (local indexer disabled)
            if durable_kv_events:
                command.append("--durable-kv-events")

            # Each SGLang worker needs a unique DYN_SYSTEM_PORT to avoid conflicts.
            # Ports are dynamically allocated for xdist-safe parallel execution.
            system_port = self._system_ports[worker_idx]

            env = os.environ.copy()  # Copy parent environment
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_device,
                "DYN_NAMESPACE": self.namespace,
                "DYN_REQUEST_PLANE": request_plane,
                "DYN_SYSTEM_PORT": str(system_port),
                "PYTHONHASHSEED": "0",  # for deterministic event id's
            }

            # Add DYN_FILE_KV if using file storage backend
            if self.store_backend == "file" and "DYN_FILE_KV" in os.environ:
                env_vars["DYN_FILE_KV"] = os.environ["DYN_FILE_KV"]

            env.update(env_vars)

            # Create managed process for the worker
            process = ManagedProcess(
                command=command,
                env=env,
                timeout=120,  # Allow time for model loading
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_all_matching_process_names=False,
            )
            self.worker_processes.append(process)
            if data_parallel_size is not None:
                logger.info(
                    f"Created {data_parallel_size} DP ranks per worker on GPU(s) {gpu_device} "
                    f"(mem_frac={mem_fraction_static}, system_port={system_port}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )
            else:
                logger.info(
                    f"Created SGLang worker {worker_idx} on GPU {gpu_device} "
                    f"(mem_frac={mem_fraction_static}, system_port={system_port}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )

    process_name = "SGLang worker"
    cleanup_name = "SGLang worker resources"


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(150)  # ~3x average (~46s/test), rounded up
def test_sglang_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=PAGE_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.skip(
    reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2784"
)  # Worker #2 dies during startup; KvRouter blocks forever waiting for min_initial_workers=2.
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_router_decisions_sglang_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_router_decisions_test(
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=PAGE_SIZE,
        component_name="backend",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.gpu_2
@pytest.mark.pre_merge
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(600)  # 10 min max (multi-GPU + DP startup variance)
@pytest.mark.skip(
    reason="DYN-2265"
)  # Currently fails probably due to SGLang startup issues when multiple workers on same GPU; re-enable when fixed
def test_router_decisions_sglang_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    """Validate KV cache prefix reuse with SGLang by sending progressive requests with overlapping prefixes.
    Same flow as test_router_decisions_sglang_multiple_workers; force first request to (worker_id, dp_rank=1).
    Dump events from router and verify:
        * All but one (worker_id, dp_rank) should have no events (due to prefix reuse)
        * The (worker_id, dp_rank) with events should have exactly 4 events (one per request)
        * All events should be on the forced (worker_id, dp_rank=1) (verifying forced routing and prefix reuse)
    """
    run_router_decisions_test(
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=PAGE_SIZE,
        component_name="backend",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
        extra_process_kwargs={"data_parallel_size": 2},
    )


@pytest.mark.skip(reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2603")
@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
@pytest.mark.timeout(600)
def test_router_decisions_sglang_disagg(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_disagg_router_decisions_test(
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=PAGE_SIZE,
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


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.parametrize(
    "store_backend,durable_kv_events,request_plane",
    [
        ("etcd", False, "tcp"),
    ],
    ids=["nats_core"],
    indirect=["durable_kv_events", "request_plane"],
)
@pytest.mark.timeout(150)  # ~3x average (~46s/test), rounded up
def test_sglang_indexers_sync(
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
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        runtime_services_dynamic_ports=runtime_services_dynamic_ports,
        store_backend=store_backend,
        durable_kv_events=durable_kv_events,
        request_plane=request_plane,
        block_size=PAGE_SIZE,
        model_name=MODEL_NAME,
        num_workers=2,
    )
