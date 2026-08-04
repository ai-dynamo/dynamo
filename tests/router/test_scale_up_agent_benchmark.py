# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the standalone affinity lifecycle workload."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from benchmarks.router.scale_up_agent_benchmark import (
    AgentWorkload,
    CaseConfig,
    CasePorts,
    read_jsonl,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.router,
    pytest.mark.timeout(15),
]


class FakeChatServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeChatHandler)
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.failed_once = False


class FakeChatHandler(BaseHTTPRequestHandler):
    server: FakeChatServer

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        session_id = self.headers.get("x-dynamo-session-id")
        with self.server.lock:
            self.server.requests.append(
                {"session_id": session_id, "body": body, "failed": False}
            )
            should_fail = session_id == "scale-agent-0" and not self.server.failed_once
            if should_fail:
                self.server.failed_once = True
                self.server.requests[-1]["failed"] = True

        if should_fail:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"worker disappeared")
            return

        payload = json.dumps(
            {
                "choices": [{"message": {"content": f"reply for {session_id}"}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_standalone_workload_sends_affinity_sessions_and_retries(tmp_path) -> None:
    server = FakeChatServer()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    frontend_port = server.server_address[1]
    ports = CasePorts(
        nats=0,
        frontend=frontend_port,
        engine_a=0,
        engine_b=0,
        dist_init_a=0,
        dist_init_b=0,
        system_a=0,
        system_b=0,
        kv_events_a=0,
        kv_events_b=0,
        forward_pass_a=0,
        forward_pass_b=0,
    )
    case = CaseConfig(
        routing_path="text",
        scenario="scale-down",
        output_dir=tmp_path,
        namespace="standalone-workload-test",
        ports=ports,
    )
    args = SimpleNamespace(
        agents=3,
        turns=2,
        scale_after_turn=1,
        seed=7,
        initial_words_median=8,
        initial_words_sigma=0,
        initial_words_max=8,
        output_tokens_median=4,
        output_tokens_sigma=0,
        output_tokens_max=4,
        temperature=0,
        followup_words_median=3,
        followup_words_sigma=0,
        followup_words_max=3,
        inter_turn_delay_median_ms=1,
        inter_turn_delay_sigma=0,
        inter_turn_delay_max_ms=1,
        request_timeout=2,
        scale_down_max_retries=1,
        scale_down_retry_delay_ms=1,
        served_model_name="test-model",
    )

    try:
        workload = AgentWorkload(args, case)
        workload.start()
        assert workload.trigger_reached.wait(timeout=5)
        deadline = time.monotonic() + 5
        while workload.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert workload.poll() == 0, workload.error
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    events = read_jsonl(tmp_path / "agent-events.jsonl")
    completions = [event for event in events if event["event"] == "request_complete"]
    retries = [event for event in events if event["event"] == "request_retry"]
    assert len(completions) == 6
    assert all(event["status"] == "success" for event in completions)
    assert len(retries) == 1
    assert retries[0]["session_id"] == "scale-agent-0"

    for agent_id in range(3):
        session_id = f"scale-agent-{agent_id}"
        successful = [
            request
            for request in server.requests
            if request["session_id"] == session_id and not request["failed"]
        ]
        assert [len(request["body"]["messages"]) for request in successful] == [1, 3]
        assert all(request["body"]["model"] == "test-model" for request in successful)
