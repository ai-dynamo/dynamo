# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dynamo.common.validation import runner

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


class FakeWorker:
    """A worker's validation endpoint: ``/v1/models`` with an instance id
    header, standing in for one worker process, then a restarted one."""

    def __init__(self) -> None:
        self.instance = "a1"
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header(runner.INSTANCE_ID_HEADER, worker.instance)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def worker():
    fake = FakeWorker()
    yield fake
    fake.server.shutdown()


def test_ready_only_for_the_instance_that_passed(worker, tmp_path):
    marker = tmp_path / "runs"
    check = [
        "sh",
        "-c",
        f'test "$OPENAI_BASE_URL" = {worker.base_url} && echo >> {marker}',
    ]
    readiness = runner.Readiness(worker.base_url, {"smoke": check}, ["smoke"], 10)
    assert readiness.ready()[0] is False
    readiness.poll()
    assert readiness.ready() == (True, "pass")
    readiness.poll()
    assert marker.read_text().count("\n") == 1, "a passed instance was checked again"

    # A restarted worker has a new id: not ready until it is checked again.
    worker.instance = "b2"
    ok, why = readiness.ready()
    assert not ok and "b2" in why
    readiness.poll()
    assert readiness.ready() == (True, "pass")
    assert marker.read_text().count("\n") == 2


def test_a_failed_check_holds_that_instance(worker):
    fail = ["sh", "-c", "echo bad template >&2; exit 1"]
    readiness = runner.Readiness(
        worker.base_url, {"ok": ["true"], "tools": fail}, ["ok", "tools"], 10
    )
    readiness.poll()
    ok, why = readiness.ready()
    assert not ok and "tools: exit status 1: bad template" in why
    readiness.poll()
    assert readiness.ready()[0] is False


def test_not_ready_while_the_worker_is_not_answering():
    readiness = runner.Readiness("http://127.0.0.1:1/v1", {"ok": ["true"]}, ["ok"], 10)
    readiness.poll()
    assert readiness.ready() == (
        False,
        "the worker's validation endpoint is not answering",
    )


def test_get_ready_reports_the_verdict(worker, serve):
    readiness = runner.Readiness(worker.base_url, {"ok": ["true"]}, ["ok"], 10)
    server = runner.make_server(0, {"ok": ["true"]}, 10, readiness)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/ready"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url, timeout=10)
        assert error.value.code == 503
        readiness.poll()
        with urllib.request.urlopen(url, timeout=10) as response:
            assert response.status == 200
    finally:
        server.shutdown()
