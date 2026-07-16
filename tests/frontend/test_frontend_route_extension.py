# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the frontend route extension mechanism.

Launches ``python -m dynamo.frontend`` with ``--frontend-route-extension``
pointing at a custom route provider, then calls the custom route and validates
the response. No worker or model is needed — extension routes are served by the
frontend independently of the inference path.

Both selector forms are exercised:

* a **registered entry-point name** (via a throwaway ``.dist-info`` on the
  subprocess ``PYTHONPATH``, so the test installs nothing and leaves the shipped
  ``ai-dynamo`` metadata untouched), and
* a direct **``module:function``** path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.frontend.route_extension_provider import HELLO_BODY, HELLO_PATH
from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.parallel,
    # Frontend-only: the custom route is served without a worker or GPU.
    pytest.mark.gpu_0,
]

ENTRY_POINT_NAME = "test-route-extension"
PROVIDER_TARGET = "tests.frontend.route_extension_provider:routes"
# tests/frontend/test_frontend_route_extension.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _pythonpath(*parts: str) -> str:
    entries = [p for p in parts if p]
    if os.environ.get("PYTHONPATH"):
        entries.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(entries)


def _register_entry_point(
    tmp_path: Path, name: str = ENTRY_POINT_NAME, target: str = PROVIDER_TARGET
) -> None:
    """Write a throwaway ``.dist-info`` registering ``name -> target`` under the
    ``dynamo.frontend.routes`` group, without installing a package."""
    dist_info = tmp_path / "route_ext_test-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: route-ext-test\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[dynamo.frontend.routes]\n{name} = {target}\n"
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("selector", ["entry_point", "module_path"])
def test_frontend_route_extension_serves_custom_route(request, tmp_path, selector):
    if selector == "entry_point":
        _register_entry_point(tmp_path)
        extension = ENTRY_POINT_NAME
        pythonpath = _pythonpath(str(tmp_path), str(REPO_ROOT))
    else:
        # module:function path — no packaging, just an importable module.
        extension = PROVIDER_TARGET
        pythonpath = _pythonpath(str(REPO_ROOT))

    with DynamoFrontendProcess(
        request,
        frontend_port=0,
        extra_args=["--frontend-route-extension", extension],
        extra_env={
            "PYTHONPATH": pythonpath,
            # In-process discovery: the frontend boots standalone (no etcd/NATS/
            # worker) since the custom route does not depend on inference.
            "DYN_DISCOVERY_BACKEND": "mem",
        },
    ) as frontend:
        url = f"http://localhost:{frontend.frontend_port}{HELLO_PATH}"

        response = None
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

        assert (
            response is not None and response.status_code == 200
        ), f"custom route {url} never returned 200"
        assert response.json() == HELLO_BODY


@pytest.mark.timeout(120)
def test_frontend_route_extension_rejects_duplicate_routes(tmp_path):
    """Two routes with the same method+path must fail with a clean error, not a
    panic. All Python providers fold into one axum Router, and Router::route
    panics on an overlapping method+path; the loader must reject the collision
    up front instead."""
    _register_entry_point(
        tmp_path,
        name="dup",
        target="tests.frontend.route_extension_provider:duplicate_routes",
    )
    port = allocate_port(DynamoPortRange.FRONTEND.value)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath(str(tmp_path), str(REPO_ROOT))
        env["DYN_DISCOVERY_BACKEND"] = "mem"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(port),
                "--router-mode",
                "round-robin",
                "--frontend-route-extension",
                "dup",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    finally:
        deallocate_port(port)

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"frontend should have exited non-zero:\n{output}"
    assert "duplicate frontend route registered" in output, output
    # The clean error must replace the raw axum overlap panic.
    assert "Overlapping method route" not in output, output
