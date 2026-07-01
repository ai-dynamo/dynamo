# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.planner_factory import (
    construct_connector,
    construct_environment,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(**overrides) -> PlannerConfig:
    values = {
        "namespace": "base-ns",
        "backend": "vllm",
        "mode": "disagg",
        "environment": "kubernetes",
    }
    values.update(overrides)
    return PlannerConfig.model_construct(**values)


def test_construct_environment_shares_runtime_namespace_binding():
    connector = MagicMock()
    runtime = MagicMock()
    with (
        patch(
            "dynamo.planner.core.planner_factory.construct_connector",
            return_value=connector,
        ),
        patch(
            "dynamo.planner.core.planner_factory.RuntimeFpmProvider"
        ) as fpm_provider_class,
        patch(
            "dynamo.planner.core.planner_factory.PrometheusTrafficProvider"
        ) as traffic_provider_class,
    ):
        environment = construct_environment(
            config=_config(),
            runtime=runtime,
            require_prefill=True,
            require_decode=True,
        )

    namespace_source = traffic_provider_class.call_args.kwargs["namespace_source"]
    assert namespace_source is environment.runtime_namespace_source
    assert namespace_source is fpm_provider_class.call_args.kwargs["namespace_source"]
    assert namespace_source.runtime_namespace() == "base-ns"


@pytest.mark.parametrize("global_planner_namespace", [None, ""])
def test_global_planner_namespace_uses_explicit_runtime_validation(
    global_planner_namespace,
):
    config = _config(
        environment="global-planner",
        global_planner_namespace=global_planner_namespace,
    )

    with pytest.raises(ValueError, match="global_planner_namespace is required"):
        construct_connector(config, runtime=MagicMock())
