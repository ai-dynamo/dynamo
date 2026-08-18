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
from tests.utils.constants import DynamoPortRange
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import (
    allocate_contiguous_ports,
    allocate_port,
    allocate_ports,
    deallocate_ports,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "silence09/DeepSeek-R1-Small-2layers"

pytestmark = [
    pytest.mark.router,
    pytest.mark.sglang,
]
# SGLang uses "page_size" instead of "block_size". 64 (not 16) because the
# TRT-LLM MLA attention backend only supports page sizes 32/64, and
# dynamo.sglang coerces page_size up to 64 on GPUs where that backend is
# selected (overrides._mla_backend_page_constraints). A 16-token page here
# then desyncs the router radix (16) from the worker KV events (64) and every
# device_blocks assertion sees 0. 64 is honored on every backend, so the
# router and workers always agree.
PAGE_SIZE = 64

# Shared SGLang configuration for all tests.
# Memory is budgeted with the token-cap form (--max-total-tokens +
# --mem-fraction-static 0.9, see tests/README.md "SGLang KV tokens").
SGLANG_ARGS: Dict[str, Any] = {
    "page_size": PAGE_SIZE,
    "model": MODEL_NAME,
    "max_total_tokens": 2048,  # matches the requested_sglang_kv_tokens(2048) markers
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
                - max_total_tokens: Max KV cache tokens; takes precedence over
                  mem_fraction_static and is emitted with --mem-fraction-static 0.9
                  (see tests/README.md "SGLang KV tokens")
                - context_length: Maximum sequence length (optional)
                - disable_cuda_graph: Disable CUDA graphs (default: False)
            num_workers: Number of SGLang worker processes
            single_gpu: If True, all workers share GPU 0
            data_parallel_size: If set, enables this many data-parallel ranks per worker process.
            request_plane: Request plane to use ("nats", "tcp"). Defaults to "tcp".
            store_backend: Storage backend to use ("etcd" or "file"). Defaults to "etcd".
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
        inherited_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_devices = (
            [device.strip() for device in inherited_visible_devices.split(",")]
            if inherited_visible_devices
            else None
        )

        def resolve_gpu_devices(start: int, count: int) -> str:
            logical_indices = range(start, start + count)
            if visible_devices is None:
                return ",".join(str(index) for index in logical_indices)
            if start + count > len(visible_devices):
                raise ValueError(
                    f"SGLang requested {count} GPU(s) starting at logical GPU "
                    f"{start}, but CUDA_VISIBLE_DEVICES exposes only "
                    f"{inherited_visible_devices}"
                )
            return ",".join(visible_devices[index] for index in logical_indices)

        # Dynamically allocate unique system and KV event ports to avoid
        # conflicts in parallel test runs.
        self._system_ports = allocate_ports(num_workers, DynamoPortRange.ROUTER.value)
        kv_event_rank_span = data_parallel_size or 1
        self._kv_event_ports = allocate_contiguous_ports(
            num_workers, kv_event_rank_span, DynamoPortRange.ROUTER.value
        )
        # Forward-pass metrics: SGLang publishes FPM over a per-worker ipc://
        # socket (path derived from the worker's connection_id), so unlike vLLM
        # it never binds this port -- the env var only flips the feature on. One
        # shared value across workers is therefore sufficient (no collision).
        self._fpm_port = allocate_port(DynamoPortRange.FPM.value)
        request.addfinalizer(
            lambda: deallocate_ports(
                self._system_ports + self._kv_event_ports + [self._fpm_port]
            )
        )

        if sglang_args is None:
            sglang_args = {}

        page_size = sglang_args.get("page_size", PAGE_SIZE)
        model = sglang_args.get("model", MODEL_NAME)
        mem_fraction_static = sglang_args.get("mem_fraction_static")
        max_total_tokens = sglang_args.get("max_total_tokens")
        context_length = sglang_args.get("context_length")
        disable_cuda_graph = sglang_args.get("disable_cuda_graph", False)
        # Resolved memory budget, for startup logs (mirrors the command flags).
        mem_budget = (
            f"max_total_tokens={max_total_tokens}, mem_frac=0.9"
            if max_total_tokens is not None
            else f"mem_frac={mem_fraction_static}"
        )

        self.model_name = model

        for worker_idx in range(num_workers):
            # Calculate GPU device for this process
            if single_gpu:
                # Force all processes to GPU 0 (for single-GPU testing)
                gpu_device = resolve_gpu_devices(gpu_start_index, 1)
            elif data_parallel_size is not None:
                # Worker sees dp_rank GPUs (each DP rank gets its own GPU)
                worker_start_gpu = gpu_start_index + worker_idx * data_parallel_size
                gpu_device = resolve_gpu_devices(worker_start_gpu, data_parallel_size)
            else:
                # No DP; worker sees one GPU
                gpu_device = resolve_gpu_devices(gpu_start_index + worker_idx, 1)

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

            # Limit VRAM allocation (required for multi-worker on same GPU).
            # Prefer the token-cap form; see the SGLANG_ARGS comment.
            if max_total_tokens is not None:
                command.extend(
                    [
                        "--max-total-tokens",
                        str(max_total_tokens),
                        "--mem-fraction-static",
                        "0.9",
                    ]
                )
            elif mem_fraction_static is not None:
                command.extend(["--mem-fraction-static", str(mem_fraction_static)])

            # Add optional context_length if specified
            if context_length is not None:
                command.extend(["--context-length", str(context_length)])

            command.extend(build_gpu_mem_args("build_sglang_gpu_mem_args"))

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

            # Add per-worker KV events config for ZMQ publishing. SGLang DP
            # ranks publish at base_port + dp_rank, so DP tests must reserve a
            # contiguous port block and pass the block's base port here.
            kv_events_port = self._kv_event_ports[worker_idx * kv_event_rank_span]
            kv_events_config = f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{kv_events_port}"}}'
            command.extend(["--kv-events-config", kv_events_config])

            # Each SGLang worker needs a unique DYN_SYSTEM_PORT to avoid conflicts.
            # Ports are dynamically allocated for xdist-safe parallel execution.
            system_port = self._system_ports[worker_idx]

            env = os.environ.copy()  # Copy parent environment
            env_vars = {
                "CUDA_VISIBLE_DEVICES": gpu_device,
                "DYN_NAMESPACE": self.namespace,
                "DYN_REQUEST_PLANE": request_plane,
                "DYN_SYSTEM_PORT": str(system_port),
                "DYN_FORWARDPASS_METRIC_PORT": str(self._fpm_port),
                "PYTHONHASHSEED": "0",  # for deterministic event id's
            }
            if data_parallel_size is not None:
                # FlashInfer's fused DeepSeek top-k kernel does not support
                # A10G (SM86). Exercise the equivalent Triton/JIT path until
                # SGLang's capability-aware fallback reaches our container.
                env_vars["SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"] = "1"

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
                    f"({mem_budget}, system_port={system_port}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )
            else:
                logger.info(
                    f"Created SGLang worker {worker_idx} on GPU {gpu_device} "
                    f"({mem_budget}, system_port={system_port}, kv_port={kv_events_port}) "
                    f"with endpoint: {self.endpoint}"
                )

    process_name = "SGLang worker"
    cleanup_name = "SGLang worker resources"


@pytest.mark.e2e
@pytest.mark.model(MODEL_NAME)
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(12.0)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(270)  # 3x ~89s (sglang gpu_1 log)
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


@pytest.mark.e2e
@pytest.mark.model(MODEL_NAME)
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(12.0)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.timeout(300)
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


@pytest.mark.e2e
@pytest.mark.model(MODEL_NAME)
@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.profiled_vram_gib(3.7)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(600)  # 10 min max (multi-GPU + DP startup variance)
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
@pytest.mark.e2e
@pytest.mark.model(MODEL_NAME)
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


# DYN-2784: fixture setup hangs instead of failing. Worker #2 dies during
# SGLangProcess launch and KvRouter then blocks forever on
# min_initial_workers=2; the timeout below cannot fire because the signal is
# swallowed at a C-level syscall.
#
# Skipped outright, because the containment this previously relied on does not
# hold. The note here used to read "passes reliably in pre_merge/post_merge, so
# scope the skip to nightly" -- measured over five days that is 82% of nightly
# runs but also 63% of post_merge, and the runs that do hang burn a whole job:
# observed hangs of 3659s, 4235s and 6077s, each holding a 12 GiB reservation
# with three more workers queued behind it, because the VRAM-aware parallel
# orchestrator has no hard-kill for a child that outlives its declared timeout.
# That came to roughly 220 runner-hours in five days.
#
# skip_in_nightly is kept but is not what makes this safe: it only drops the
# test from the nightly marker filter, and the hang is not nightly-specific.
#
# Re-enable once the launch race is fixed, or once the orchestrator can kill a
# hung child -- either one alone turns this from a lost job into a normal
# failure. The scenario itself is worth keeping; it is the hang that is not
# affordable.
@pytest.mark.skip(
    reason="hangs to the job step cap instead of failing; see the note above"
)
@pytest.mark.e2e
@pytest.mark.model(MODEL_NAME)
@pytest.mark.skip_in_nightly
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.profiled_vram_gib(12.0)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize("event_plane", ["nats"], indirect=True)
@pytest.mark.timeout(320)  # 3x ~106s (sglang gpu_1 log)
def test_sglang_indexers_sync(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
    event_plane,
):
    run_indexers_sync_test(
        engine_process_cls=SGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        runtime_services_dynamic_ports=runtime_services_dynamic_ports,
        store_backend="etcd",
        request_plane=request_plane,
        event_plane=event_plane,
        block_size=PAGE_SIZE,
        model_name=MODEL_NAME,
        num_workers=2,
    )
