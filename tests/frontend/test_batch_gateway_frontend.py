# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Python frontend and native proxy without inference workers."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
from pytest_httpserver import HTTPServer

from tests.utils.managed_process import DynamoFrontendProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.forked,
]


@pytest.fixture
def httpserver() -> Iterator[HTTPServer]:
    # Stop the server thread before a later test forks the pytest process.
    with HTTPServer(host="127.0.0.1", port=0) as server:
        yield server


@pytest.mark.timeout(60)
@pytest.mark.parametrize("enabled", [False, True])
def test_batch_jobs_through_actual_dynamo_frontend(
    request,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    httpserver: HTTPServer,
    enabled: bool,
) -> None:
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.delenv("no_proxy", raising=False)
    for name in (
        "DYN_BATCH_GATEWAY_URL",
        "DYN_FRONTEND_ROUTE_EXTENSIONS",
        "DYN_INTERACTIVE",
        "DYN_KSERVE_GRPC_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)
    batch_body = {
        "input_file_id": "file-input",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    output = '{"custom_id":"one","response":{"status_code":200}}\n'
    httpserver.expect_request(
        "/v1/files",
        method="POST",
        headers={"Authorization": "Bearer test-token", "X-Tenant-Id": "tenant-a"},
    ).respond_with_json({"id": "file-input", "object": "file"})
    httpserver.expect_request(
        "/v1/batches", method="POST", json=batch_body
    ).respond_with_json({"id": "batch-one", "status": "in_progress"})
    httpserver.expect_request("/v1/batches/batch-one", method="GET").respond_with_json(
        {"id": "batch-one", "status": "completed", "output_file_id": "file-output"}
    )
    httpserver.expect_request(
        "/v1/files/file-output/content", method="GET"
    ).respond_with_data(output, content_type="application/x-ndjson")
    args = [
        "--discovery-backend",
        "mem",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    if enabled:
        # The real CLI value must beat the inherited environment all the way to
        # native proxy construction, not merely within the Python parser.
        monkeypatch.setenv("DYN_BATCH_GATEWAY_URL", "http://wrong-gateway.invalid")
        args.extend(["--batch-gateway-url", httpserver.url_for("/")])
    frontend = DynamoFrontendProcess(
        request,
        frontend_port=0,
        extra_args=args,
        extra_env={
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "DYN_TEST_OUTPUT_PATH": str(tmp_path),
        },
    )
    origin = f"http://127.0.0.1:{frontend.frontend_port}"
    frontend.health_check_urls = [f"{origin}/v1/models"]
    frontend.timeout = 20
    with frontend:
        upload = requests.post(
            f"{origin}/v1/files",
            headers={"Authorization": "Bearer test-token", "X-Tenant-Id": "tenant-a"},
            data={"purpose": "batch"},
            files={
                "file": ("input.jsonl", b'{"custom_id":"one"}\n', "application/jsonl")
            },
            timeout=5,
        )
        if not enabled:
            assert upload.status_code == 404
            assert len(httpserver.log) == 0
            return
        assert upload.status_code == 200
        assert upload.json()["id"] == "file-input"
        created = requests.post(f"{origin}/v1/batches", json=batch_body, timeout=5)
        assert created.status_code == 200
        assert created.json()["id"] == "batch-one"
        polled = requests.get(f"{origin}/v1/batches/batch-one", timeout=5)
        assert polled.status_code == 200
        assert polled.json()["status"] == "completed"
        downloaded = requests.get(f"{origin}/v1/files/file-output/content", timeout=5)
        assert downloaded.status_code == 200
        assert downloaded.text == output
        assert requests.get(f"{origin}/v1/models", timeout=5).status_code == 200
        httpserver.check_assertions()
        assert len(httpserver.log) == 4
        logged_upload = httpserver.log[0][0]
        assert logged_upload.form["purpose"] == "batch"
        assert logged_upload.files["file"].read() == b'{"custom_id":"one"}\n'
