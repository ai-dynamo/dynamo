# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import urllib.error
import urllib.request

import pytest

from dynamo.validation import runner

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit]

REQUEST = {"base_url": "http://127.0.0.1:1/v1"}


def post(port: int, name: str, body: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/checks/{name}",
        data=json.dumps(body).encode(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


@pytest.fixture
def serve():
    servers = []

    def start(checks: dict[str, list[str]], timeout: float = 10) -> int:
        server = runner.make_server(0, checks, timeout)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    yield start
    for server in servers:
        server.shutdown()


def test_status_is_the_verdict_and_stderr_the_reason(serve):
    port = serve(
        {
            "pass": ["sh", "-c", 'test "$OPENAI_BASE_URL" = http://127.0.0.1:1/v1'],
            "fail": ["sh", "-c", "echo no tool call >&2; exit 3"],
        }
    )
    assert post(port, "pass", REQUEST) == (200, "pass")
    assert post(port, "fail", REQUEST) == (500, "exit status 3: no tool call")
    assert post(port, "missing", REQUEST)[0] == 404


def test_a_hung_check_fails_at_the_timeout(serve):
    port = serve({"hang": ["sleep", "30"]}, timeout=0.5)
    assert post(port, "hang", REQUEST) == (500, "no verdict within 0.5s")


def test_the_base_url_comes_only_from_the_request(serve, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://somewhere-else/v1")
    monkeypatch.setenv("DYN_VALIDATION_MODEL", "m")
    port = serve(
        {
            "unset": [
                "sh",
                "-c",
                'test -z "${OPENAI_BASE_URL+x}" && test "$DYN_VALIDATION_MODEL" = m',
            ]
        }
    )
    assert post(port, "unset", {"base_url": None})[0] == 200
