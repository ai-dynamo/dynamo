# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LL frontend + vLLM mocker disagg via disaggregated_params (no bootstrap) — strand flow check.

Verifies, end-to-end, that with the frontend in Least Loaded routing mode, vLLM
mocker prefill/decode workers launched WITHOUT --bootstrap-ports route prefill→decode
over the disaggregated_params (NIXL-style) path and strand/release KV. Correctness is
confirmed via the request pipeline working; the pin lifecycle is observable in the
worker logs (prefill_kv_pin_start / decode_kv_wait_start / prefill_kv_pin_end) when run
with -s and DYN_LOG including mocker::kv_abort=info.
"""

import asyncio
import logging
import time
from typing import Any, Dict

import pytest

from tests.router.helper import (
    generate_random_suffix,
    send_request_with_retry,
    wait_for_frontend_ready,
)
from tests.router.mocker_process import _launch_disagg_workers
from tests.router.router_process import FrontendRouterProcess
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.port_utils import allocate_ports

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
BLOCK_SIZE = 16
SPEEDUP_RATIO = 10.0
BASE_PORT = 8700

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_NAME),
]

TEST_PAYLOAD: Dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "In a quiet meadow tucked between rolling hills, a plump gray "
            "rabbit nibbled on clover beneath the shade of a gnarled oak tree. Its ears "
            "twitched at the faint rustle of leaves, but it remained calm, confident in "
            "the safety of its burrow just a few hops away.",
        }
    ],
    "stream": True,
    "max_tokens": 10,
}


@pytest.mark.timeout(240)
def test_ll_disagg_params_strand_flow(
    request,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
):
    _ = (runtime_services_dynamic_ports, predownload_tokenizers)
    discovery_backend = "etcd"
    request_plane = "tcp"
    shared_namespace = f"test-namespace-{generate_random_suffix()}"
    mocker_args = {"speedup_ratio": SPEEDUP_RATIO, "block_size": BLOCK_SIZE}

    frontend_port = allocate_ports(1, BASE_PORT)[0]
    base_url = f"http://localhost:{frontend_port}"
    chat_url = f"{base_url}/v1/chat/completions"

    with FrontendRouterProcess(
        request,
        BLOCK_SIZE,
        frontend_port,
        shared_namespace,
        discovery_backend,
        enforce_disagg=True,
        request_plane=request_plane,
        event_plane="nats",
        router_mode="least-loaded",
    ):
        time.sleep(1.0)
        with _launch_disagg_workers(
            request,
            shared_namespace,
            "prefill_first",
            prefill_mocker_args=mocker_args,
            decode_mocker_args=mocker_args,
            num_prefill_mockers=2,
            num_decode_mockers=2,
            enable_disagg_bootstrap=False,
            store_backend=discovery_backend,
            request_plane=request_plane,
            event_plane="nats",
        ):
            asyncio.run(
                wait_for_frontend_ready(
                    base_url, test_payload=TEST_PAYLOAD, timeout=120
                )
            )
            # Drive several requests; under LL each prefill is routed by least-loaded,
            # emits disaggregated_params, the decode connects, and the pin releases.
            # Readiness already warmed the pipeline, so requests succeed first-try;
            # the bounded retries (max_retries=3) are CI-noise insurance, kept small
            # so a genuinely broken path fails fast rather than backing off into the
            # test timeout.
            for i in range(8):
                ok = asyncio.run(
                    send_request_with_retry(chat_url, TEST_PAYLOAD, max_retries=3)
                )
                assert ok, f"request {i} failed under LL disagg_params routing"
