# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the model-aware health option through the Python frontend and bindings."""

import time

import pytest
import requests

from tests.frontend.test_frontend_route_extension import STANDALONE_ARGS
from tests.utils.managed_process import DynamoFrontendProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
]


def wait_for_live(frontend):
    url = f"http://localhost:{frontend.frontend_port}/live"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if not frontend.is_running():
            pytest.fail(f"frontend exited during startup:\n{frontend.read_logs()}")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    pytest.fail(f"frontend never became live:\n{frontend.read_logs()}")


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    ("flag", "expected_status"),
    [("--health-check-requires-model", 503), ("--no-health-check-requires-model", 200)],
)
def test_frontend_health_without_workers(request, flag, expected_status):
    with DynamoFrontendProcess(
        request,
        frontend_port=0,
        extra_args=[*STANDALONE_ARGS, flag],
        # CLI must override an enabled environment default.
        extra_env={"DYN_HEALTH_CHECK_REQUIRES_MODEL": "true"},
    ) as frontend:
        wait_for_live(frontend)
        url = f"http://localhost:{frontend.frontend_port}"
        health = requests.get(f"{url}/health", timeout=5)
        assert health.status_code == expected_status
        assert health.json()["status"] == (
            "not_ready" if expected_status == 503 else "healthy"
        )
        models = requests.get(f"{url}/v1/models", timeout=5)
        assert models.status_code == 200
        assert models.json()["data"] == []
        assert requests.get(f"{url}/live", timeout=5).status_code == 200
