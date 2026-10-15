# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep direct Planner examples independent of profiling ConfigMaps."""

import json
from pathlib import Path

import pytest
import yaml

from dynamo.planner.config.planner_config import (
    PlannerConfig,
    PlannerPreDeploymentSweepMode,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_disagg_planner_example_uses_inline_config_and_runtime_observations(backend):
    root = Path(__file__).resolve().parents[6]
    source = (
        root / "examples" / "backends" / backend / "deploy" / "disagg_planner.yaml"
    ).read_text()
    manifests = list(yaml.safe_load_all(source))
    assert all(manifest["kind"] != "ConfigMap" for manifest in manifests)
    deployments = [
        manifest
        for manifest in manifests
        if manifest["kind"] == "DynamoGraphDeployment"
    ]
    assert len(deployments) == 1
    components = {
        component["name"]: component
        for component in deployments[0]["spec"]["components"]
    }
    planner = components["Planner"]["podTemplate"]["spec"]["containers"][0]
    config_arg = planner["args"][planner["args"].index("--config") + 1]
    config_data = json.loads(config_arg)
    config = PlannerConfig.model_validate(config_data)
    assert config.backend == backend
    assert config.mode == "disagg"
    assert config.pre_deployment_sweeping_mode == PlannerPreDeploymentSweepMode.None_
    assert "profile_results_dir" not in config_data

    for component in components.values():
        pod_spec = component["podTemplate"]["spec"]
        assert all("configMap" not in volume for volume in pod_spec.get("volumes", []))
        for container in pod_spec["containers"]:
            assert "--planner-profile-data" not in container.get("args", [])
