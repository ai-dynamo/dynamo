# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/generate_kustomize_openapi.py"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_kustomize_openapi", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_schema_includes_all_operator_crd_versions():
    generator = load_generator_module()
    schema = json.loads(generator.generated_schema())

    expected_definitions = set()
    for crd_path in generator.crd_paths():
        crd = yaml.safe_load(crd_path.read_text(encoding="utf-8"))
        if crd.get("kind") != "CustomResourceDefinition":
            continue
        group = crd["spec"]["group"]
        kind = crd["spec"]["names"]["kind"]
        expected_definitions.update(
            f"{group}.{version['name']}.{kind}" for version in crd["spec"]["versions"]
        )

    assert expected_definitions <= schema["definitions"].keys()


def test_generated_dgd_schema_merges_main_container_env_by_name():
    generator = load_generator_module()
    schema = json.loads(generator.generated_schema())
    dgd = schema["definitions"]["nvidia.com.v1alpha1.DynamoGraphDeployment"]
    env = dgd["properties"]["spec"]["properties"]["services"]["additionalProperties"][
        "properties"
    ]["extraPodSpec"]["properties"]["mainContainer"]["properties"]["env"]

    assert env["x-kubernetes-patch-strategy"] == "merge"
    assert env["x-kubernetes-patch-merge-key"] == "name"
