# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mocker-backed smoke test for the event-based FT harness. Uses the
# mocker backend (bundled in every dynamo runtime image) so it runs
# without a GPU; aiperf drives 10 requests, the deployment must serve
# them all with zero errors. Useful as a baseline next to fault tests.

import pytest

from tests.fault_tolerance.deploy.checks import LoadCompleted, MinRequests, ZeroErrors
from tests.fault_tolerance.deploy.events import StartLoad, WaitForLoadCompletion
from tests.fault_tolerance.deploy.scenario import run_scenario
from tests.utils.managed_deployment import DeploymentSpec
from tests.utils.managed_load import LoadConfig


@pytest.mark.k8s
@pytest.mark.e2e
@pytest.mark.weekly
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_sanity_mocker(namespace, image, skip_service_restart, storage_class):
    spec = DeploymentSpec.from_backend("mocker", "agg")
    # The mocker/agg.yaml hardcodes a stale planner-profile path; override
    # to the location the runtime images actually ship with.
    spec["decode"].set_arg(
        "--planner-profile-data",
        "/workspace/tests/planner/profiling_results/H200_TP1P_TP1D",
    )
    # Pull model from the worker's launch args so aiperf and the worker stay
    # in sync no matter what the YAML default is.
    served_model = spec["decode"].model
    await run_scenario(
        deployment_spec=spec,
        events=[
            StartLoad(
                load_config=LoadConfig(
                    model_name=served_model,
                    tokenizer=served_model,
                    request_count=10,
                    concurrency=2,
                    input_tokens_mean=128,
                    output_tokens_mean=32,
                )
            ),
            WaitForLoadCompletion(),
        ],
        checks=[
            LoadCompleted(),
            MinRequests(min_count=10),
            ZeroErrors(),
        ],
        namespace=namespace,
        image=image,
        test_name="test_sanity_mocker",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
