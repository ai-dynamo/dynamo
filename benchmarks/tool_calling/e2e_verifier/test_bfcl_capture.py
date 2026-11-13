# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from .bfcl_capture import FAILURES_FILE, TRACE_FILE, capture_http

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_capture_keeps_concurrent_case_ids_exact_bodies_and_error_responses(tmp_path):
    body = json.dumps({"messages": [{"role": "user", "content": "x" * 20000}]})
    response = SimpleNamespace(
        status_code=400,
        text='{"error":{"message":"specific rejected parameter"}}',
        content=b'{"error":{"message":"specific rejected parameter"}}',
        headers={"Content-Type": "application/json", "X-Request-ID": "server-request"},
    )
    calls = []

    class Session:
        def send(self, request, **kwargs):
            calls.append((request, kwargs))
            return response

    request = SimpleNamespace(
        method="POST",
        url="http://unused/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
    )

    class Handler:
        def inference(self, test_entry, include_input_log, exclude_state_log):
            return Session().send(request, timeout=123)

    original_inference, original_send = Handler.inference, Session.send
    with capture_http(Handler, Session, tmp_path):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda n: Handler().inference({"id": f"case_{n}"}, False, False),
                    range(8),
                )
            )
    assert all(result is response for result in results)
    assert Handler.inference is original_inference
    assert Session.send is original_send
    assert all(sent is request and kwargs == {"timeout": 123} for sent, kwargs in calls)
    failures = [
        json.loads(line) for line in (tmp_path / FAILURES_FILE).read_text().splitlines()
    ]
    assert {r["case_id"] for r in failures} == {f"case_{n}" for n in range(8)}
    assert all(r["request_sequence"] == 1 for r in failures)
    assert all(r["request"]["body"] == body for r in failures)
    assert all(r["response"]["body"] == response.text for r in failures)
    assert all(r["response"]["status"] == 400 for r in failures)
    assert "secret" not in (tmp_path / TRACE_FILE).read_text()


def test_capture_preserves_transport_exception_and_request_evidence(tmp_path):
    failure = ConnectionError("connection lost")

    class Session:
        def send(self, request, **kwargs):
            raise failure

    class Handler:
        def inference(self, test_entry, *args):
            request = SimpleNamespace(
                method="POST", url="http://unused", body=b"{}", headers={}
            )
            return Session().send(request)

    with capture_http(Handler, Session, tmp_path):
        with pytest.raises(ConnectionError) as raised:
            Handler().inference({"id": "failed_case"})
    assert raised.value is failure
    rows = [
        json.loads(line) for line in (tmp_path / TRACE_FILE).read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == ["request", "transport_error"]
    assert rows[-1]["case_id"] == "failed_case"
    assert rows[-1]["request"]["body"] == "{}"
