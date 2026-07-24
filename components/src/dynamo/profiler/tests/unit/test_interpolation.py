# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from deploy.utils.dynamo_deployment import DeploymentFailedError, DynamoDeploymentClient
from dynamo.profiler.interpolation import _wait_for_required_interpolation_deployment

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        DeploymentFailedError("worker entered CrashLoopBackOff"),
    ],
    ids=["timeout", "terminal-failure"],
)
def test_required_interpolation_failure_is_fatal(failure):
    client = AsyncMock(spec=DynamoDeploymentClient)
    client.wait_for_deployment_ready.side_effect = failure
    deployment_clients = [client]

    with pytest.raises(
        RuntimeError,
        match=(
            "Thorough mode requires real-GPU interpolation data; "
            "use pre_deployment_sweeping_mode='rapid'"
        ),
    ) as exc_info:
        asyncio.run(
            _wait_for_required_interpolation_deployment(
                client,
                deployment_clients,
                timeout=20,
                phase="prefill",
            )
        )

    assert exc_info.value.__cause__ is failure
    client.delete_deployment.assert_awaited_once_with()
    assert deployment_clients == []


def test_failed_cleanup_leaves_deployment_for_final_cleanup():
    failure = DeploymentFailedError("worker entered CrashLoopBackOff")
    client = AsyncMock(spec=DynamoDeploymentClient)
    client.wait_for_deployment_ready.side_effect = failure
    client.delete_deployment.side_effect = RuntimeError("API unavailable")
    deployment_clients = [client]

    with pytest.raises(RuntimeError, match="Thorough mode requires real-GPU"):
        asyncio.run(
            _wait_for_required_interpolation_deployment(
                client,
                deployment_clients,
                timeout=20,
                phase="decode",
            )
        )

    client.delete_deployment.assert_awaited_once_with()
    assert deployment_clients == [client]
