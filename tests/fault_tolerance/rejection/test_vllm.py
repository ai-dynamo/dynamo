# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for vLLM worker-side queue-depth rejection."""

import concurrent.futures
import logging
import os
import shutil
import time
from typing import Any

import pytest
import requests

from tests.fault_tolerance.cancellation.utils import (
    CancellableRequest,
    DynamoFrontendProcess,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

VLLM_MAX_NUM_SEQS = 4
QUEUE_REJECT_THRESHOLD = 4

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.vllm,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoVllmQueueRejectionWorkerProcess(ManagedProcess):
    """Process manager for a vLLM worker with queue-depth rejection enabled."""

    def __init__(self, request, frontend_port: int):
        self.system_port = allocate_port(9100)
        self.fpm_port = allocate_port(20380)
        self.frontend_port = frontend_port

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--enforce-eager",
            "--use-vllm-tokenizer",
            "--gpu-memory-utilization",
            "0.45",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            str(VLLM_MAX_NUM_SEQS),
        ]

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")
        env["DYN_LOG"] = "debug"
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)
        env["DYN_FORWARDPASS_METRIC_PORT"] = str(self.fpm_port)
        env["DYN_VLLM_REJECT_QUEUE_THRESHOLD"] = str(QUEUE_REJECT_THRESHOLD)

        log_dir = f"{request.node.name}_vllm_queue_rejection_worker"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{self.system_port}/health", self.is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
            display_name="vllm_queue_rejection_worker",
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            deallocate_port(self.system_port)
            deallocate_port(self.fpm_port)
        except Exception as e:
            logger.warning("Failed to release vLLM queue rejection worker ports: %s", e)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def is_ready(self, response) -> bool:
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info("vLLM queue rejection worker is ready")
                return True
            logger.warning("Worker status is not ready: %s", data.get("status"))
        except ValueError:
            logger.warning("Worker health response is not valid JSON")
        return False


def _chat_payload(prompt: str, max_tokens: int, stream: bool = False) -> dict[str, Any]:
    return {
        "model": FAULT_TOLERANCE_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
    }


def _post_chat(frontend_port: int, prompt: str, max_tokens: int) -> requests.Response:
    return requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=_chat_payload(prompt, max_tokens=max_tokens),
        timeout=180,
    )


def _start_streaming_chat(
    frontend_port: int, prompt: str, max_tokens: int
) -> CancellableRequest:
    request = CancellableRequest()
    request.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=_chat_payload(prompt, max_tokens=max_tokens, stream=True),
        stream=True,
        timeout=180,
    )
    return request


def _wait_for_cancellable_response(
    request: CancellableRequest, timeout: float = 30.0
) -> requests.Response:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if request.exception:
            raise request.exception
        if request.response is not None:
            return request.response
        time.sleep(0.05)
    pytest.fail("Timed out waiting for streaming request response headers")


def _response_contains_load_shed(response: requests.Response) -> bool:
    text = response.text
    return "load_shed" in text or "vllm_queued" in text


def _assert_successful_response(response: requests.Response) -> None:
    assert response.status_code == 200, response.text
    assert not _response_contains_load_shed(response), response.text
    body = response.json()
    assert body["choices"], body


def _wait_for_vllm_queued_depth(
    fpm_port: int,
    target_depth: int,
    timeout: float = 90.0,
) -> int:
    """Wait until InstrumentedScheduler reports target vLLM waiting depth."""
    import zmq

    from dynamo.common.forward_pass_metrics import decode as decode_forward_pass_metrics

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(f"tcp://127.0.0.1:{fpm_port}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    deadline = time.monotonic() + timeout
    last_depth = 0
    try:
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            events = dict(poller.poll(timeout=min(500, remaining_ms)))
            if sock not in events:
                continue

            parts = sock.recv_multipart()
            metrics = decode_forward_pass_metrics(parts[-1])
            if metrics is None:
                continue

            queued = metrics.queued_requests
            last_depth = queued.num_prefill_requests + queued.num_decode_requests
            logger.info(
                "Observed vLLM queued depth=%d (prefill=%d decode=%d)",
                last_depth,
                queued.num_prefill_requests,
                queued.num_decode_requests,
            )
            if last_depth >= target_depth:
                return last_depth
    finally:
        try:
            poller.unregister(sock)
        except Exception:
            pass
        sock.close(linger=0)

    pytest.fail(
        f"Timed out waiting for vLLM queued depth >= {target_depth}; "
        f"last_depth={last_depth}"
    )


def _start_streaming_holders(
    frontend_port: int,
    count: int,
    prompt_prefix: str,
    max_tokens: int = 512,
) -> list[CancellableRequest]:
    return [
        _start_streaming_chat(
            frontend_port,
            f"{prompt_prefix} {idx}: count upward until stopped.",
            max_tokens,
        )
        for idx in range(count)
    ]


def _wait_for_streaming_responses(
    holders: list[CancellableRequest],
    count: int,
) -> None:
    for holder in holders[:count]:
        response = _wait_for_cancellable_response(holder)
        assert response.status_code == 200, response.text


def _post_overflow_requests(
    frontend_port: int,
    count: int = 2,
) -> list[requests.Response]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(
                _post_chat,
                frontend_port,
                f"Overflow request {idx}",
                8,
            )
            for idx in range(count)
        ]
        return [future.result(timeout=60) for future in futures]


@pytest.mark.timeout(240)
@pytest.mark.post_merge
@pytest.mark.gpu_1
def test_vllm_worker_does_not_reject_when_running_capacity_is_full(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    """Do not reject when vLLM only has running work, not queued work.

    vLLM is configured to run four requests concurrently and reject only when
    the scheduler waiting queue reaches four requests. Four streaming requests
    fill running capacity but do not make queued depth reach the threshold, so
    two additional requests should be accepted by the worker instead of being
    load-shed before engine submit.
    """
    with DynamoFrontendProcess(request) as frontend:
        with DynamoVllmQueueRejectionWorkerProcess(
            request, frontend.frontend_port
        ) as worker:
            logger.info("Worker PID: %s", worker.get_pid())

            holders: list[CancellableRequest] = []
            try:
                holders = _start_streaming_holders(
                    frontend.frontend_port,
                    VLLM_MAX_NUM_SEQS,
                    "Running capacity fill request",
                )
                _wait_for_streaming_responses(holders, VLLM_MAX_NUM_SEQS)

                overflow_responses = _post_overflow_requests(frontend.frontend_port)

                assert len(overflow_responses) == 2
                for response in overflow_responses:
                    _assert_successful_response(response)
            finally:
                for holder in holders:
                    holder.cancel()


@pytest.mark.timeout(240)
@pytest.mark.post_merge
@pytest.mark.gpu_1
def test_vllm_worker_rejects_when_scheduler_queue_depth_exceeds_threshold(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    """Reject overflow requests when vLLM waiting queue reaches threshold.

    vLLM is configured to run four requests concurrently. With the rejection
    threshold set to four queued requests, the test holds eight streaming
    requests open, waits until vLLM reports four requests in its scheduler
    waiting queue (four running + four waiting), then sends two overflow
    requests that should be rejected by the worker before engine submit.
    """
    with DynamoFrontendProcess(request) as frontend:
        with DynamoVllmQueueRejectionWorkerProcess(
            request, frontend.frontend_port
        ) as worker:
            logger.info("Worker PID: %s", worker.get_pid())

            holders: list[CancellableRequest] = []
            try:
                holders = _start_streaming_holders(
                    frontend.frontend_port,
                    VLLM_MAX_NUM_SEQS + QUEUE_REJECT_THRESHOLD,
                    "Queue fill request",
                )
                _wait_for_streaming_responses(holders, VLLM_MAX_NUM_SEQS)

                observed_depth = _wait_for_vllm_queued_depth(
                    worker.fpm_port,
                    QUEUE_REJECT_THRESHOLD,
                )
                assert observed_depth >= QUEUE_REJECT_THRESHOLD

                # Give the worker-local subscriber a moment to consume the same
                # scheduler snapshot seen by the test subscriber.
                time.sleep(0.25)

                overflow_responses = _post_overflow_requests(frontend.frontend_port)
                rejected = [
                    response
                    for response in overflow_responses
                    if _response_contains_load_shed(response)
                ]
                assert len(rejected) == 2, [
                    (response.status_code, response.text[:500])
                    for response in overflow_responses
                ]
            finally:
                for holder in holders:
                    holder.cancel()
