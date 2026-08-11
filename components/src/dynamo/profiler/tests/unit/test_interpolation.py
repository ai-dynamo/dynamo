# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deploy.utils.dynamo_deployment import DeploymentFailedError, DynamoDeploymentClient
from dynamo.profiler.interpolation import (
    _wait_for_required_interpolation_deployment,
    run_interpolation,
)
from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    FeaturesSpec,
    MockerSpec,
)
from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _mocker_only_dgdr(search_strategy: str) -> DynamoGraphDeploymentRequestSpec:
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-32B",
        features=FeaturesSpec(mocker=MockerSpec(enabled=True)),
        searchStrategy=search_strategy,
    )


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


def test_run_interpolation_mocker_only_rapid_is_skipped(tmp_path):
    dgdr = _mocker_only_dgdr("rapid")
    ops = ProfilerOperationalConfig(output_dir=str(tmp_path))
    deployment_clients: list[DynamoDeploymentClient] = []
    pick = PickedParallelConfig(tp=1)

    asyncio.run(
        run_interpolation(
            dgdr,
            ops,
            {},
            pick,
            pick,
            "trtllm",
            4096,
            deployment_clients,
        )
    )

    assert deployment_clients == []


def test_run_interpolation_mocker_only_thorough_is_not_skipped(tmp_path):
    def raise_past_skip_gate(*_args, **_kwargs):
        raise RuntimeError("proceeded past skip gate")

    dgdr = _mocker_only_dgdr("thorough")
    ops = ProfilerOperationalConfig(output_dir=str(tmp_path))
    pick = PickedParallelConfig(tp=1)
    fake_config_modifier = type(
        "FakeConfigModifier",
        (),
        {"get_model_name": staticmethod(raise_past_skip_gate)},
    )()

    with patch(
        "dynamo.profiler.interpolation.CONFIG_MODIFIERS",
        {"trtllm": fake_config_modifier},
    ):
        with pytest.raises(RuntimeError, match="proceeded past skip gate"):
            asyncio.run(
                run_interpolation(
                    dgdr,
                    ops,
                    {},
                    pick,
                    pick,
                    "trtllm",
                    4096,
                    [],
                )
            )
