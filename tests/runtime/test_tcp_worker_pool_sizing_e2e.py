# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TCP worker-pool startup sizing with a real mocker worker.

Each parameter starts a fresh process so the process-global shared TCP server is
initialized exactly once. The automatic case covers the complete path from
mocker capacity normalization through model attachment recording the hint and
explicit endpoint startup consuming it. Serving before attachment completes
would initialize the pool at the 10,000 fallback instead of the expected 1.5x
engine capacity.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import ServicePorts
from tests.utils.prometheus import sum_metric_samples

POOL_CAPACITY_METRIC = "dynamo_work_handler_pool_capacity"
ENGINE_MAX_NUM_SEQS = 6

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.model(QWEN),
]


@pytest.mark.timeout(120)  # Covers the 60s startup timeout plus teardown margin.
@pytest.mark.parametrize(
    "engine_request_limit, tcp_worker_pool_size, expected_pool_size",
    [
        pytest.param(None, None, 9, id="automatic-1.5x"),
        pytest.param(None, 13, 13, id="tcp-pool-override"),
        pytest.param(17, None, 17, id="engine-limit-override"),
        pytest.param(17, 13, 17, id="engine-limit-precedes-tcp-pool"),
    ],
)
def test_tcp_worker_pool_startup_sizing(
    request,
    file_storage_backend,
    predownload_tokenizers,
    dynamo_dynamic_ports: ServicePorts,
    engine_request_limit: int | None,
    tcp_worker_pool_size: int | None,
    expected_pool_size: int,
):
    system_port = dynamo_dynamic_ports.system_ports[0]
    namespace = f"test-worker-pool-{uuid.uuid4().hex}"
    endpoint = f"dyn://{namespace}.mocker.generate"
    command = [
        sys.executable,
        "-m",
        "dynamo.mocker",
        "--model-path",
        QWEN,
        "--endpoint",
        endpoint,
        "--discovery-backend",
        "file",
        "--num-workers",
        "1",
        "--max-num-seqs",
        str(ENGINE_MAX_NUM_SEQS),
        "--data-parallel-size",
        "1",
    ]

    env = os.environ.copy()
    for name in (
        "DYN_ENGINE_REQUEST_LIMIT",
        "DYN_TCP_WORKER_POOL_SIZE",
        "DYN_TCP_WORK_QUEUE_SIZE",
        "DYN_DYNAMO_REQUEST_QUEUE_LIMIT",
        "DYN_TCP_RPC_PORT",
        "NATS_SERVER",
    ):
        env.pop(name, None)
    env.update(
        {
            "DYN_FILE_KV": str(file_storage_backend),
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_SYSTEM_PORT": str(system_port),
        }
    )
    if engine_request_limit is not None:
        env["DYN_ENGINE_REQUEST_LIMIT"] = str(engine_request_limit)
    if tcp_worker_pool_size is not None:
        env["DYN_TCP_WORKER_POOL_SIZE"] = str(tcp_worker_pool_size)

    metrics_url = f"http://localhost:{system_port}/metrics"

    def reports_expected_pool_capacity(response: requests.Response) -> bool:
        return (
            sum_metric_samples(response.text, POOL_CAPACITY_METRIC)
            == expected_pool_size
        )

    process = ManagedProcess(
        command=command,
        env=env,
        health_check_urls=[(metrics_url, reports_expected_pool_capacity)],
        timeout=60,
        display_output=True,
        terminate_all_matching_process_names=False,
        log_dir=f"{request.node.name}_worker_pool",
        display_name="dynamo-mocker-worker-pool-sizing",
    )
    with process:
        response = requests.get(metrics_url, timeout=5)
        response.raise_for_status()
        assert (
            sum_metric_samples(response.text, POOL_CAPACITY_METRIC)
            == expected_pool_size
        )
