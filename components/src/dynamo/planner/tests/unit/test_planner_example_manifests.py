# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep direct Planner examples independent of profiling ConfigMaps."""

import json
from pathlib import Path
from string import Template

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
    images = {
        "FRONTEND_IMAGE": "registry.example/dynamo-frontend:release",
        "PLANNER_IMAGE": "registry.example/dynamo-planner:release",
        f"{backend.upper()}_IMAGE": f"registry.example/{backend}-runtime:release",
    }
    manifests = list(yaml.safe_load_all(Template(source).substitute(images)))
    assert len(manifests) == 1
    deployment = manifests[0]
    assert deployment["kind"] == "DynamoGraphDeployment"
    assert deployment["apiVersion"] == "nvidia.com/v1beta1"
    components = {
        component["name"]: component for component in deployment["spec"]["components"]
    }

    planner = components["Planner"]["podTemplate"]["spec"]["containers"][0]
    assert (planner["command"] + planner["args"])[:3] == [
        "python3",
        "-m",
        "dynamo.planner",
    ]
    frontend = components["Frontend"]["podTemplate"]["spec"]["containers"][0]
    assert frontend["image"] == images["FRONTEND_IMAGE"]
    assert planner["image"] == images["PLANNER_IMAGE"]
    config_arg = planner["args"][planner["args"].index("--config") + 1]
    config = PlannerConfig.from_config_arg(config_arg)
    assert config.backend == backend
    assert config.mode == "disagg"
    assert config.pre_deployment_sweeping_mode == PlannerPreDeploymentSweepMode.None_
    assert "profile_results_dir" not in json.loads(config_arg)

    for role in ("prefill", "decode"):
        component = components[role]
        assert component["scalingAdapter"]["enabled"] is True
        container = component["podTemplate"]["spec"]["containers"][0]
        assert container["image"] == images[f"{backend.upper()}_IMAGE"]
        assert "--planner-profile-data" not in container["args"]
    for component in components.values():
        assert all(
            "configMap" not in volume
            for volume in component["podTemplate"]["spec"].get("volumes", [])
        )
