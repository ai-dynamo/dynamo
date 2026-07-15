# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the frontend route extension mechanism.

Launches ``python -m dynamo.frontend`` with ``--frontend-route-extension``
pointing at a custom route provider, then calls the custom route and validates
the response. No worker or model is needed — extension routes are served by the
frontend independently of the inference path.

The provider is registered by writing a throwaway ``.dist-info`` onto the
subprocess ``PYTHONPATH`` (rather than installing a package), so the test is
self-contained and leaves the shipped ``ai-dynamo`` metadata untouched.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest
import requests

from tests.frontend.route_extension_provider import HELLO_BODY, HELLO_PATH
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.parallel,
    # Frontend-only: the custom route is served without a worker or GPU.
    pytest.mark.gpu_0,
]

EXTENSION_NAME = "test-route-extension"
PROVIDER_TARGET = "tests.frontend.route_extension_provider:routes"
# tests/frontend/test_frontend_route_extension.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _register_provider_entry_point(tmp_path: Path) -> str:
    """Write a throwaway ``.dist-info`` that registers the provider under the
    ``dynamo.frontend_routes`` group, and return a ``PYTHONPATH`` that exposes
    both the entry point and the provider module to the frontend subprocess."""
    dist_info = tmp_path / "route_ext_test-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: route-ext-test\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[dynamo.frontend_routes]\n{EXTENSION_NAME} = {PROVIDER_TARGET}\n"
    )
    parts = [str(tmp_path), str(REPO_ROOT)]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


@pytest.mark.timeout(120)
def test_frontend_route_extension_serves_custom_route(request, tmp_path):
    pythonpath = _register_provider_entry_point(tmp_path)

    with DynamoFrontendProcess(
        request,
        frontend_port=0,
        extra_args=["--frontend-route-extension", EXTENSION_NAME],
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
