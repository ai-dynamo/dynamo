# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import urllib.error
import urllib.request

import pytest

from dynamo.validation import runner

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit]


def post(port: int, name: str) -> tuple[int, str]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/validate/{name}",
        data=b'{"DYN_VALIDATION_GATE": "g"}',
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_status_is_the_verdict_and_stderr_the_reason():
    server = runner.make_server(
        0,
        {
            "pass": ["sh", "-c", 'test "$DYN_VALIDATION_GATE" = g'],
            "fail": ["sh", "-c", "echo no tool call >&2; exit 3"],
        },
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert post(port, "pass") == (200, "pass")
        assert post(port, "fail") == (500, "exit status 3: no tool call")
        assert post(port, "missing")[0] == 404
    finally:
        server.shutdown()
