# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

# Timing notes (measured locally):
# - GPU-1 subset (`-m "gpu_1 and not gpu_2"`): 130.43s total for 3 tests.
# These tests load a real model and can be slow/flaky when GPU resources are contended,
# so we set explicit pytest timeouts to fail fast on hangs (see per-test markers below).
import json
import logging
import os
from typing import Any, Dict, Optional

import aiohttp
import pytest

from tests.router.e2e_harness import (
    ManagedEngineProcessMixin,
    run_basic_router_test,
    run_disagg_router_decisions_test,
    run_indexers_sync_test,
    run_router_decisions_test,
)
from tests.router.helper import (
    generate_random_suffix,
    get_kv_indexer_command,
    wait_for_indexer_workers_active,
)
from tests.utils.constants import DefaultPort
from tests.utils.device import (
    build_nixl_kv_transfer_config_json,
    detect_target_device,
    get_default_vllm_block_size,
    get_device_visibility_env_var,
    get_gpu_memory_utilization,
)
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.vllm,
    pytest.mark.model(MODEL_NAME),
]
SPEEDUP_RATIO = 10.0
BLOCK_SIZE = get_default_vllm_block_size()  # 64 on XPU (fmha requirement), 16 on CUDA
_GPU_MEM_UTIL = get_gpu_memory_utilization(num_workers=2, single_gpu=True)
_MAX_MODEL_LEN = 768 if detect_target_device() == "xpu" else 1024

# Shared vLLM configuration for all tests
# gpu_memory_utilization limits actual VRAM allocation (required for multi-worker on same GPU)
VLLM_ARGS: Dict[str, Any] = {
    "block_size": BLOCK_SIZE,
    "model": MODEL_NAME,
    "gpu_memory_utilization": _GPU_MEM_UTIL,
    "max_model_len": _MAX_MODEL_LEN,  # Limit context length to reduce KV cache size
    "enforce_eager": True,  # Disable CUDA graphs for faster startup & lower memory
}

VLLM_ARGS_NO_BLOCK_SIZE: Dict[str, Any] = {
    "model": MODEL_NAME,
    "gpu_memory_utilization": _GPU_MEM_UTIL,
    "max_model_len": _MAX_MODEL_LEN,  # Limit context length to reduce KV cache size
    "enforce_eager": True,  # Disable CUDA graphs for faster startup & lower memory
}


class VLLMProcess(ManagedEngineProcessMixin):
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
        request_plane: str = "tcp",
        store_backend: str = "etcd",
        durable_kv_events: bool = False,
        namespace: Optional[str] = None,
        gpu_start_index: int = 0,
        disaggregation_mode: Optional[str] = None,
        standalone_indexer: bool = False,
        zmq_replay: bool = False,
    ):
        """Initialize vLLM workers with dynamo integration.

        Args:
            request: pytest request fixture for log directory
            vllm_args: Configuration dict with keys:
                - model: Model name/path (default: TinyLlama-1.1B)
                - gpu_memory_utilization: Fraction of GPU memory to allocate (optional)
                - num_gpu_blocks_override: Cap on number of KV cache blocks (optional)
                - max_model_len: Maximum sequence length (optional)
                - enforce_eager: Disable CUDA graphs (default: False)
            num_workers: Number of vLLM worker processes
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
        self.worker_id_to_zmq_ports: dict[int, dict[int, str]] = {}
        self._worker_id_to_replay_ports: dict[int, dict[int, str]] = {}
        self.store_backend = store_backend
        self._request = request
        self._request_plane = request_plane
        self._standalone_indexer = standalone_indexer
        self._zmq_replay = zmq_replay
        self._standalone_indexer_port: Optional[int] = None
        self._standalone_indexer_b_port: Optional[int] = None
        self._indexer_process: Optional[ManagedProcess] = None
        self._indexer_b_process: Optional[ManagedProcess] = None

        # Dynamically allocate unique system, KV event, and NIXL side-channel
        # ports (one of each per worker) to avoid conflicts in parallel test runs.
        self._system_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._kv_event_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._nixl_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._replay_ports = (
            allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
            if standalone_indexer and zmq_replay
            else []
        )
        self._indexer_ports = (
            allocate_ports(2, DefaultPort.SYSTEM1.value) if standalone_indexer else []
        )
        if standalone_indexer:
            self._standalone_indexer_port = self._indexer_ports[0]
            self._standalone_indexer_b_port = self._indexer_ports[1]
        request.addfinalizer(
            lambda: deallocate_ports(
                self._system_ports
                + self._kv_event_ports
                + self._nixl_ports
                + self._replay_ports
                + self._indexer_ports
            )
        )

        if vllm_args is None:
            vllm_args = {}

        model = vllm_args.get("model", MODEL_NAME)
        gpu_memory_utilization = vllm_args.get("gpu_memory_utilization")
        num_gpu_blocks_override = vllm_args.get("num_gpu_blocks_override")
        max_model_len = vllm_args.get("max_model_len")
        enforce_eager = vllm_args.get("enforce_eager", False)

        self.model_name = model
        self.block_size = vllm_args.get("block_size", BLOCK_SIZE)

        # Create vLLM worker processes
        # Matches test.sh behavior:
        # - When data_parallel_size is set, launch one process per DP rank
        # - Each process gets --data-parallel-rank and --data-parallel-size
        # - Each process runs on its assigned device visibility env var
        # - --kv-transfer-config enables KV cache transfer between ranks

        for worker_idx in range(num_workers):
            visibility_env_var = get_device_visibility_env_var()
            inherited_visibility = os.environ.get(visibility_env_var)

            # Calculate GPU device for this process
            if single_gpu:
                # On XPU, prefer externally pinned affinity when provided by CI/runtime,
                # but do not override an explicit non-default gpu_start_index.
                if visibility_env_var == "ZE_AFFINITY_MASK" and inherited_visibility:
                    gpu_device = inherited_visibility
                else:
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

            command = ["python3", "-m", "dynamo.vllm", "--model", model]

            if "block_size" in vllm_args:
                command.extend(["--block-size", str(vllm_args["block_size"])])

            if disaggregation_mode is not None:
                command.extend(["--disaggregation-mode", disaggregation_mode])
                command.extend(
                    [
                        "--kv-transfer-config",
                        build_nixl_kv_transfer_config_json(),
                    ]
                )

            # Disable CUDA graphs for faster startup & lower memory
            if enforce_eager:
                command.append("--enforce-eager")

            # Limit VRAM allocation (required for multi-worker on same GPU)
            if gpu_memory_utilization is not None:
                command.extend(
                    ["--gpu-memory-utilization", str(gpu_memory_utilization)]
                )

            # Add optional max_model_len if specified
            if max_model_len is not None:
                command.extend(["--max-model-len", str(max_model_len)])

            # Cap block count for predictable KV cache behavior
            if num_gpu_blocks_override is not None:
                command.extend(
                    ["--num-gpu-blocks-override", str(num_gpu_blocks_override)]
                )

            if data_parallel_size is not None:
                # Add DP configuration for external load balancing
                # See: https://docs.vllm.ai/en/v0.10.0/serving/data_parallel_deployment.html#external-load-balancing
                command.extend(
                    [
                        "--data-parallel-size",
                        str(data_parallel_size),
                        # "--data-parallel-address", "127.0.0.1",  # Required for DP coordination
                        # "--data-parallel-rpc-port", "13345",  # RPC port for DP coordination
                        # "--kv-transfer-config", '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',  # Required for KV transfer between DP ranks
                    ]
                )

            # Use --durable-kv-events to enable JetStream mode (local indexer disabled)
            if durable_kv_events:
                command.append("--durable-kv-events")

            # Ports are dynamically allocated for xdist-safe parallel execution.
            system_port = self._system_ports[worker_idx]
            kv_event_port = self._kv_event_ports[worker_idx]
            nixl_port = self._nixl_ports[worker_idx]
            replay_port = (
                self._replay_ports[worker_idx]
                if worker_idx < len(self._replay_ports)
                else None
            )

            # Pass KV events config explicitly via CLI
            kv_events_cfg: Dict[str, Any] = {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_event_port}",
                "enable_kv_cache_events": True,
            }
            if replay_port is not None:
                kv_events_cfg["replay_endpoint"] = f"tcp://*:{replay_port}"
            command.extend(["--kv-events-config", json.dumps(kv_events_cfg)])

            env = os.environ.copy()  # Copy parent environment
            env_vars = {
                visibility_env_var: gpu_device,
                "DYN_NAMESPACE": self.namespace,
                "DYN_REQUEST_PLANE": request_plane,
                "DYN_SYSTEM_PORT": str(system_port),
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
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
                    f"(gpu_mem={gpu_memory_utilization}, system_port={system_port}) "
                    f"with endpoint: {self.endpoint}"
                )
            else:
                logger.info(
                    f"Created vLLM worker {worker_idx} on GPU {gpu_device} "
                    f"(gpu_mem={gpu_memory_utilization}, system_port={system_port}) "
                    f"with endpoint: {self.endpoint}"
                )

    @property
    def standalone_indexer_url(self) -> Optional[str]:
        if self._standalone_indexer_port is not None:
            return f"http://localhost:{self._standalone_indexer_port}"
        return None

    @property
    def standalone_indexer_b_url(self) -> Optional[str]:
        if self._standalone_indexer_b_port is not None:
            return f"http://localhost:{self._standalone_indexer_b_port}"
        return None

    def __enter__(self):
        if not self._standalone_indexer:
            return super().__enter__()

        indexer_cmd = [
            *get_kv_indexer_command(),
            "--block-size",
            str(self.block_size),
            "--port",
            str(self._standalone_indexer_port),
        ]
        self._indexer_process = ManagedProcess(
            command=indexer_cmd,
            timeout=120,
            display_output=True,
            health_check_ports=[self._standalone_indexer_port],
            health_check_urls=[],
            log_dir=self._request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-kv-indexer",
        )
        logger.info(
            "Starting standalone indexer on port %s", self._standalone_indexer_port
        )
        self._indexer_process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._standalone_indexer:
            for process in self.worker_processes:
                process.__exit__(exc_type, exc_val, exc_tb)
            if self._indexer_b_process is not None:
                self._indexer_b_process.__exit__(exc_type, exc_val, exc_tb)
                self._indexer_b_process = None
            if self._indexer_process is not None:
                self._indexer_process.__exit__(exc_type, exc_val, exc_tb)
                self._indexer_process = None
            return

        super().__exit__(exc_type, exc_val, exc_tb)

    async def launch_workers_with_indexer(self, endpoint):
        if not self._standalone_indexer:
            raise RuntimeError(
                "launch_workers_with_indexer requires standalone_indexer=True"
            )

        client = await endpoint.client()
        known_ids: set[int] = set()
        register_url = f"{self.standalone_indexer_url}/register"

        async with aiohttp.ClientSession() as session:
            for worker_idx, process in enumerate(self.worker_processes):
                process.__enter__()

                new_worker_id = None
                for _ in range(120):
                    ids = set(client.instance_ids())
                    new = ids - known_ids
                    if new:
                        new_worker_id = new.pop()
                        known_ids.add(new_worker_id)
                        break
                    await asyncio.sleep(0.5)

                if new_worker_id is None:
                    raise RuntimeError(
                        f"Timed out waiting for vLLM worker {worker_idx} to register "
                        f"(known_ids={known_ids})"
                    )

                zmq_endpoint = f"tcp://127.0.0.1:{self._kv_event_ports[worker_idx]}"
                replay_endpoint = (
                    f"tcp://127.0.0.1:{self._replay_ports[worker_idx]}"
                    if worker_idx < len(self._replay_ports)
                    else None
                )

                payload = {
                    "instance_id": new_worker_id,
                    "endpoint": zmq_endpoint,
                    "dp_rank": 0,
                    "model_name": self.model_name,
                    "block_size": self.block_size,
                }
                if replay_endpoint is not None:
                    payload["replay_endpoint"] = replay_endpoint

                async with session.post(register_url, json=payload) as resp:
                    if resp.status != 201:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Failed to register vLLM instance {new_worker_id}: "
                            f"{resp.status} {body}"
                        )

                self.worker_id_to_zmq_ports[new_worker_id] = {0: zmq_endpoint}
                if replay_endpoint is not None:
                    self._worker_id_to_replay_ports[new_worker_id] = {
                        0: replay_endpoint
                    }

                logger.info(
                    "vLLM worker %s: worker_id=%s, zmq_endpoint=%s, replay_endpoint=%s",
                    worker_idx,
                    new_worker_id,
                    zmq_endpoint,
                    replay_endpoint,
                )

        await wait_for_indexer_workers_active(
            self.standalone_indexer_url, self.worker_id_to_zmq_ports
        )
        logger.info(
            "All %s vLLM workers launched and registered with indexer",
            self.num_workers,
        )

    def launch_indexer(self):
        if not self._standalone_indexer or self._standalone_indexer_b_port is None:
            raise RuntimeError("launch_indexer requires standalone_indexer=True")
        if not self.worker_id_to_zmq_ports:
            raise RuntimeError("launch_indexer requires workers to be registered first")

        worker_entries = []
        for worker_id, zmq_addresses in self.worker_id_to_zmq_ports.items():
            for dp_rank, zmq_endpoint in zmq_addresses.items():
                worker_entries.append(f"{worker_id}:{dp_rank}={zmq_endpoint}")
        workers_arg = ",".join(worker_entries)

        indexer_b_cmd = [
            *get_kv_indexer_command(),
            "--block-size",
            str(self.block_size),
            "--port",
            str(self._standalone_indexer_b_port),
            "--peers",
            f"http://localhost:{self._standalone_indexer_port}",
            "--workers",
            workers_arg,
            "--model-name",
            self.model_name,
        ]
        self._indexer_b_process = ManagedProcess(
            command=indexer_b_cmd,
            timeout=120,
            display_output=True,
            health_check_ports=[self._standalone_indexer_b_port],
            health_check_urls=[],
            log_dir=self._request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-kv-indexer-b",
        )
        logger.info(
            "Starting standalone indexer B on port %s with peer http://localhost:%s",
            self._standalone_indexer_b_port,
            self._standalone_indexer_port,
        )
        self._indexer_b_process.__enter__()

    process_name = "vLLM worker"
    cleanup_name = "vLLM worker resources"
    init_delay_reason = "initialize NIXL before starting next worker"


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.timeout(
    300
)  # XPU CI observed >160s; raise timeout to reduce false failures
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_vllm_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.timeout(150)  # ~3x average (~43s/test), rounded up
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_vllm_kv_router_without_block_size_specified_in_vllm_args(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS_NO_BLOCK_SIZE,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.xpu_1
@pytest.mark.timeout(150)  # ~3x average (~43s/test), rounded up
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_router_decisions_vllm_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_router_decisions_test(
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
        component_name="backend",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.gpu_2
@pytest.mark.xpu_2
@pytest.mark.nightly
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.timeout(600)  # 10 min max (multi-GPU + DP startup variance)
def test_router_decisions_vllm_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    """Validate KV cache prefix reuse with vLLM by sending progressive requests with overlapping prefixes.
    Same flow as test_router_decisions_vllm_multiple_workers; force first request to (worker_id, dp_rank=1).
    Dump events from router and verify:
        * All but one (worker_id, dp_rank) should have no events (due to prefix reuse)
        * The (worker_id, dp_rank) with events should have exactly 4 events (one per request)
        * All events should be on the forced (worker_id, dp_rank=1) (verifying forced routing and prefix reuse)
    """
    run_router_decisions_test(
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
        component_name="backend",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
        extra_process_kwargs={"data_parallel_size": 2},
    )


@pytest.mark.gpu_2
@pytest.mark.xpu_2
@pytest.mark.nightly
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
def test_router_decisions_vllm_disagg(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_disagg_router_decisions_test(
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
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
@pytest.mark.xpu_1
@pytest.mark.timeout(150)  # ~3x average (~43s/test), rounded up
@pytest.mark.parametrize(
    "store_backend,durable_kv_events,request_plane",
    [
        ("etcd", False, "tcp"),
    ],
    ids=["nats_core"],
    indirect=["durable_kv_events", "request_plane"],
)
def test_vllm_indexers_sync(
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
        engine_process_cls=VLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        runtime_services_dynamic_ports=runtime_services_dynamic_ports,
        store_backend=store_backend,
        durable_kv_events=durable_kv_events,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
        num_workers=2,
        extra_process_kwargs={"standalone_indexer": True, "zmq_replay": True},
    )
