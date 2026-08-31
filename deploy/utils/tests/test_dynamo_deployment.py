# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the fail-fast behavior added for #9213.

When a candidate deployment's worker pods enter ``CrashLoopBackOff``,
``DynamoDeploymentClient.wait_for_deployment_ready`` must raise
``DeploymentFailedError`` immediately rather than waiting out the full
``timeout`` — otherwise the thorough-mode profiler burns up to 30 min
of wall-clock per failing candidate.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip the whole module if the deploy.utils runtime deps aren't available
# in this environment. The test doesn't actually exercise these — it just
# needs the import of `dynamo_deployment` to succeed.
pytest.importorskip("aiofiles")
pytest.importorskip("kubernetes_asyncio")
pytest.importorskip("httpx")

from deploy.utils.dynamo_deployment import (  # noqa: E402
    DeploymentFailedError,
    DynamoDeploymentClient,
)

pytestmark = pytest.mark.pre_merge


async def test_wait_for_deployment_ready_raises_deployment_failed_on_crashloop(
    monkeypatch,
):
    client = DynamoDeploymentClient(namespace="ns", deployment_name="dgd-test")
    client.deployment_name = "dgd-test"
    client._original_components = ["PrefillWorker"]
    client.components = ["prefillworker"]

    # DGD CR exists but isn't Ready yet.
    client.custom_api = MagicMock()
    client.custom_api.get_namespaced_custom_object = AsyncMock(
        return_value={"status": {"state": "deploying", "conditions": []}}
    )
    # Simulate a crash on the very first poll.
    client._detect_terminal_pod_failure = AsyncMock(  # type: ignore[method-assign]
        return_value="pod p0 container worker in CrashLoopBackOff"
    )

    # Avoid sleeping in the test.
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("deploy.utils.dynamo_deployment.asyncio.sleep", _no_sleep)

    with pytest.raises(DeploymentFailedError) as excinfo:
        # Pass a generous timeout so a regression (timeout instead of
        # raise) would be obvious.
        await client.wait_for_deployment_ready(timeout=600)

    assert "CrashLoopBackOff" in str(excinfo.value)
    # Confirm we didn't run out the timeout — there should have been at
    # most one DGD status check before the raise.
    assert client.custom_api.get_namespaced_custom_object.await_count == 1
