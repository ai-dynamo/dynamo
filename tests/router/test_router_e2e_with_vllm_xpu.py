# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU-specific router e2e tests (multi-card, one device per worker).

This module mirrors the gpu_1 tests from test_router_e2e_with_vllm.py
with the following XPU adaptations:

  * Block size 64 (Intel fmha requirement) instead of 16.
  * Device visibility via ZE_AFFINITY_MASK instead of CUDA_VISIBLE_DEVICES.
  * ``health_check_ports=[kv_event_port]`` on each worker so the health
    check waits until the ZMQ PUB socket is actually listening.  This
    avoids the TOCTOU race where the allocated port is grabbed by
    another process between allocation and bind.
  * Longer timeouts (600s) to accommodate XPU model-loading times.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import pytest

from tests.router.e2e_harness import (
    ManagedEngineProcessMixin,
    run_basic_router_test,
    run_router_decisions_test,
)
from tests.router.helper import generate_random_suffix
from tests.utils.constants import DefaultPort
from tests.utils.device import (
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
    pytest.mark.xpu_2,
    pytest.mark.model(MODEL_NAME),
]

BLOCK_SIZE = get_default_vllm_block_size()
_GPU_MEM_UTIL = get_gpu_memory_utilization()
_MAX_MODEL_LEN = 768

VLLM_ARGS: Dict[str, Any] = {
    "block_size": BLOCK_SIZE,
    "model": MODEL_NAME,
    "gpu_memory_utilization": _GPU_MEM_UTIL,
    "max_model_len": _MAX_MODEL_LEN,
    "enforce_eager": True,
}

VLLM_ARGS_NO_BLOCK_SIZE: Dict[str, Any] = {
    "model": MODEL_NAME,
    "gpu_memory_utilization": _GPU_MEM_UTIL,
    "max_model_len": _MAX_MODEL_LEN,
    "enforce_eager": True,
}


class XPUVLLMProcess(ManagedEngineProcessMixin):
    """Manages vLLM workers on XPU with TOCTOU-safe port handling.

    Key differences from the CUDA VLLMProcess:
      * Uses ``ZE_AFFINITY_MASK`` for device visibility.
      * Adds ``health_check_ports=[kv_event_port]`` so the harness waits
        until the ZMQ PUB socket is bound, catching port-conflict crashes
        immediately rather than hanging.
    """

    def __init__(
        self,
        request,
        vllm_args: Optional[Dict[str, Any]] = None,
        num_workers: int = 2,
        request_plane: str = "tcp",
        store_backend: str = "etcd",
        namespace: Optional[str] = None,
        gpu_start_index: int = 0,
        **kwargs,
    ):
        namespace_suffix = generate_random_suffix()
        self.namespace = namespace or f"test-namespace-{namespace_suffix}"
        self.component_name = "backend"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_workers
        self.worker_processes = []
        self.worker_id_to_zmq_ports: dict[int, dict[int, str]] = {}
        self.store_backend = store_backend
        self._request = request
        self._request_plane = request_plane

        self._system_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._kv_event_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        self._nixl_ports = allocate_ports(num_workers, DefaultPort.SYSTEM1.value)
        request.addfinalizer(
            lambda: deallocate_ports(
                self._system_ports + self._kv_event_ports + self._nixl_ports
            )
        )

        if vllm_args is None:
            vllm_args = {}

        model = vllm_args.get("model", MODEL_NAME)
        gpu_memory_utilization = vllm_args.get("gpu_memory_utilization", _GPU_MEM_UTIL)
        max_model_len = vllm_args.get("max_model_len", _MAX_MODEL_LEN)
        enforce_eager = vllm_args.get("enforce_eager", True)

        self.model_name = model
        self.block_size = vllm_args.get("block_size", BLOCK_SIZE)

        visibility_env_var = get_device_visibility_env_var()
        inherited_visibility = os.environ.get(visibility_env_var)

        for worker_idx in range(num_workers):
            # Map worker index to physical device using inherited mask.
            # ZE_AFFINITY_MASK uses physical indices (not remapped like
            # CUDA_VISIBLE_DEVICES), so we must pick from the inherited
            # list to stay on the cards assigned to this runner.
            device_idx = gpu_start_index + worker_idx
            if visibility_env_var == "ZE_AFFINITY_MASK" and inherited_visibility:
                devices = [d.strip() for d in inherited_visibility.split(",")]
                gpu_device = (
                    devices[device_idx]
                    if device_idx < len(devices)
                    else str(device_idx)
                )
            else:
                gpu_device = str(device_idx)

            command = ["python3", "-m", "dynamo.vllm", "--model", model]

            if "block_size" in vllm_args:
                command.extend(["--block-size", str(vllm_args["block_size"])])

            if enforce_eager:
                command.append("--enforce-eager")

            command.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

            if max_model_len is not None:
                command.extend(["--max-model-len", str(max_model_len)])

            system_port = self._system_ports[worker_idx]
            kv_event_port = self._kv_event_ports[worker_idx]
            nixl_port = self._nixl_ports[worker_idx]

            kv_events_cfg = {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_event_port}",
                "enable_kv_cache_events": True,
            }
            command.extend(["--kv-events-config", json.dumps(kv_events_cfg)])

            env = os.environ.copy()
            env.update(
                {
                    visibility_env_var: gpu_device,
                    "DYN_NAMESPACE": self.namespace,
                    "DYN_REQUEST_PLANE": request_plane,
                    "DYN_SYSTEM_PORT": str(system_port),
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
                    "PYTHONHASHSEED": "0",
                }
            )

            if self.store_backend == "file" and "DYN_FILE_KV" in os.environ:
                env["DYN_FILE_KV"] = os.environ["DYN_FILE_KV"]

            # health_check_ports=[kv_event_port]: the harness will poll
            # until the ZMQ PUB socket is listening, catching EADDRINUSE
            # crashes immediately rather than blocking forever.
            process = ManagedProcess(
                command=command,
                env=env,
                timeout=180,
                display_output=True,
                health_check_ports=[kv_event_port],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_all_matching_process_names=False,
            )
            self.worker_processes.append(process)
            logger.info(
                "Created XPU vLLM worker %d on device %s "
                "(gpu_mem=%s, system_port=%d, kv_event_port=%d)",
                worker_idx,
                gpu_device,
                gpu_memory_utilization,
                system_port,
                kv_event_port,
            )

    process_name = "vLLM worker (XPU)"
    cleanup_name = "vLLM worker resources"
    init_delay_reason = "finish post-init registration before starting next worker"
    init_delay_seconds = 8


# ---------------------------------------------------------------------------
# Multi-card tests (one XPU device per worker)
# ---------------------------------------------------------------------------


@pytest.mark.pre_merge
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_vllm_kv_router_basic_xpu(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=XPUVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        num_workers=2,
        single_gpu=False,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_vllm_kv_router_without_block_size_xpu(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_basic_router_test(
        engine_process_cls=XPUVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS_NO_BLOCK_SIZE,
        num_workers=2,
        single_gpu=False,
        request=request,
        request_plane=request_plane,
        block_size=BLOCK_SIZE,
        model_name=MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_router_decisions_vllm_multiple_workers_xpu(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    run_router_decisions_test(
        engine_process_cls=XPUVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=MODEL_NAME,
        block_size=BLOCK_SIZE,
        component_name="backend",
        num_workers=2,
        single_gpu=False,
        test_dp_rank=False,
    )
