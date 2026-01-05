# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
E2E test for DP attention KV events.

Tests that when running SGLang with DP attention mode (--enable-dp-attention):
1. All DP ranks publish KV events via ZMQ
2. Dynamo receives events from all DP ranks
3. Routing decisions work correctly with dp_rank metadata

With --tp 4 --dp-size 4 --enable-dp-attention:
- attn_tp_size = tp_size // dp_size = 4 // 4 = 1
- All 4 schedulers have attn_tp_rank = 0 (KV events enabled for all)
- Each scheduler publishes to port: base_port + attn_dp_rank
- Ports: 5557, 5558, 5559, 5560 (all unique)

The test verifies:
- KV events are received with dp_rank metadata
- Events come from all DP ranks (not just rank 0)
- Router correctly routes based on prefix cache and dp_rank
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import pytest

from tests.router.common import (
    _test_router_decisions,
    generate_random_suffix,
    get_runtime,
)
from tests.utils.constants import DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)

# Model that supports DP attention with MLA
MODEL_NAME = "silence09/DeepSeek-R1-Small-2layers"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.sglang,
    pytest.mark.model(MODEL_NAME),
    pytest.mark.gpu_4,  # Requires 4 GPUs for tp=4, dp=4
]

PAGE_SIZE = 16


def allocate_frontend_ports(request, count: int) -> list[int]:
    """Allocate random free frontend ports for xdist-safe execution."""
    ports = allocate_ports(count, DefaultPort.FRONTEND.value)
    request.addfinalizer(lambda: deallocate_ports(ports))
    return ports


# Shared test payload
TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you today?",
        }
    ],
    "stream": True,
    "max_tokens": 10,
}

# SGLang configuration for DP attention
# tp=4, dp=4 with enable-dp-attention
SGLANG_DP_ATTENTION_ARGS: Dict[str, Any] = {
    "page_size": PAGE_SIZE,
    "model": MODEL_NAME,
    "context_length": 1024,
    "disable_cuda_graph": True,
    # DP attention specific
    "tp_size": 4,
    "dp_size": 4,
    "enable_dp_attention": True,
}


class SGLangDPAttentionProcess:
    """Manages SGLang with DP attention mode for testing KV events from all ranks.

    Unlike standard DP mode where each process is a separate DP rank,
    DP attention mode runs a single process with all DP ranks inside.
    Each internal scheduler publishes KV events to a unique ZMQ port.
    """

    def __init__(
        self,
        request,
        sglang_args: Optional[Dict[str, Any]] = None,
        base_kv_port: int = 5557,
        request_plane: str = "nats",
        store_backend: str = "etcd",
    ):
        """Initialize SGLang with DP attention mode.

        Args:
            request: pytest request fixture for log directory
            sglang_args: Configuration dict with:
                - page_size: KV cache page size
                - model: Model name/path
                - tp_size: Tensor parallel size
                - dp_size: Data parallel size
                - enable_dp_attention: Enable attention parallelism
            base_kv_port: Base port for ZMQ KV events (each dp_rank adds offset)
            request_plane: Request plane to use ("nats" or "tcp")
            store_backend: Storage backend ("etcd" or "file")
        """
        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "backend"
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.store_backend = store_backend

        if sglang_args is None:
            sglang_args = SGLANG_DP_ATTENTION_ARGS.copy()

        self.page_size = sglang_args.get("page_size", PAGE_SIZE)
        self.model = sglang_args.get("model", MODEL_NAME)
        self.tp_size = sglang_args.get("tp_size", 4)
        self.dp_size = sglang_args.get("dp_size", 4)
        self.enable_dp_attention = sglang_args.get("enable_dp_attention", True)
        self.context_length = sglang_args.get("context_length", 1024)
        self.disable_cuda_graph = sglang_args.get("disable_cuda_graph", True)

        self.base_kv_port = base_kv_port
        self.model_name = self.model

        # With DP attention, number of workers = 1 (single process with all ranks)
        # But instance_ids will return dp_size separate instances
        self.num_workers = 1
        self.data_parallel_size = self.dp_size

        # Build command for DP attention mode
        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            self.model,
            "--page-size",
            str(self.page_size),
            "--tp",
            str(self.tp_size),
            "--dp-size",
            str(self.dp_size),
            "--enable-dp-attention",
            "--trust-remote-code",
        ]

        if self.disable_cuda_graph:
            command.append("--disable-cuda-graph")

        if self.context_length:
            command.extend(["--context-length", str(self.context_length)])

        # KV events config - base port, each scheduler will offset by attn_dp_rank
        kv_events_config = f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{base_kv_port}"}}'
        command.extend(["--kv-events-config", kv_events_config])

        env = os.environ.copy()
        env_vars = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(self.tp_size)),
            "DYN_NAMESPACE": self.namespace,
            "DYN_REQUEST_PLANE": request_plane,
            "PYTHONHASHSEED": "0",
        }

        if self.store_backend == "file" and "DYN_FILE_KV" in os.environ:
            env_vars["DYN_FILE_KV"] = os.environ["DYN_FILE_KV"]

        env.update(env_vars)

        self._process = ManagedProcess(
            command=command,
            env=env,
            timeout=180,  # DP attention needs more time to initialize
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_existing=False,
        )

        logger.info(
            f"Created SGLang DP attention process: tp={self.tp_size}, dp={self.dp_size}, "
            f"kv_base_port={base_kv_port}, endpoint={self.endpoint}"
        )

    def __enter__(self):
        logger.info(f"Starting SGLang DP attention process...")
        self._process.__enter__()

        # Wait additional time for all schedulers to initialize ZMQ publishers
        logger.info("Waiting for all ZMQ publishers to start...")
        time.sleep(10)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping SGLang DP attention process")
        self._process.__exit__(exc_type, exc_val, exc_tb)
        time.sleep(2)


@pytest.mark.gpu_4
@pytest.mark.timeout(300)  # 5 minutes for DP attention initialization
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
def test_dp_attention_kv_events_all_ranks(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    """
    Test that KV events are received from all DP attention ranks.

    Configuration: --tp 4 --dp-size 4 --enable-dp-attention

    This test verifies:
    1. All 4 schedulers start ZMQ publishers on unique ports
    2. KV events include dp_rank metadata
    3. Router can route requests based on prefix cache per dp_rank
    """
    logger.info("Starting DP attention KV events test")

    try:
        # Start SGLang with DP attention
        sglang_workers = SGLangDPAttentionProcess(
            request,
            sglang_args=SGLANG_DP_ATTENTION_ARGS,
            base_kv_port=5557,
            request_plane=request_plane,
        )
        logger.info(f"Using namespace: {sglang_workers.namespace}")
        sglang_workers.__enter__()

        # Get runtime and create endpoint
        runtime = get_runtime(request_plane=request_plane)
        namespace = runtime.namespace(sglang_workers.namespace)
        component = namespace.component("backend")
        endpoint = component.endpoint("generate")

        # Run router decisions test with DP rank verification
        _test_router_decisions(
            sglang_workers,
            endpoint,
            MODEL_NAME,
            request,
            test_dp_rank=True,  # Enable DP rank testing
            block_size=PAGE_SIZE,
        )

    finally:
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)


@pytest.mark.gpu_4
@pytest.mark.timeout(300)
@pytest.mark.parametrize("request_plane", ["nats"], indirect=True)
def test_dp_attention_zmq_ports_bound(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
):
    """
    Verify that all DP attention ranks bind to unique ZMQ ports.

    With tp=4, dp=4, enable_dp_attention:
    - Scheduler 0 (attn_dp_rank=0) -> port 5557
    - Scheduler 1 (attn_dp_rank=1) -> port 5558
    - Scheduler 2 (attn_dp_rank=2) -> port 5559
    - Scheduler 3 (attn_dp_rank=3) -> port 5560
    """
    import socket

    logger.info("Starting DP attention ZMQ ports test")

    try:
        sglang_workers = SGLangDPAttentionProcess(
            request,
            sglang_args=SGLANG_DP_ATTENTION_ARGS,
            base_kv_port=5557,
            request_plane=request_plane,
        )
        sglang_workers.__enter__()

        # Check that all 4 ZMQ ports are bound
        bound_ports = []
        for i in range(4):
            port = 5557 + i
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                bound_ports.append(port)
                logger.info(f"Port {port} is bound (attn_dp_rank={i})")
            else:
                logger.warning(f"Port {port} is NOT bound (attn_dp_rank={i})")

        # Verify all 4 ports are bound
        assert len(bound_ports) == 4, (
            f"Expected all 4 ZMQ ports (5557-5560) to be bound, "
            f"but only found: {bound_ports}"
        )

        logger.info(f"SUCCESS: All 4 ZMQ ports bound: {bound_ports}")

    finally:
        if "sglang_workers" in locals():
            sglang_workers.__exit__(None, None, None)
