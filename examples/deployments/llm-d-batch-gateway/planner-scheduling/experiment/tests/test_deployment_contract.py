# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

PLANNER_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = PLANNER_ROOT.parent
SENTINEL_NAMESPACE = "planner-contract-test"
SENTINEL_IMAGE = "registry.example/dynamo:immutable-test-build"
SENTINEL_TENANT = "planner-contract-tenant"
SENTINEL_WARM_REPLICAS = "1"
CONTROL_KEY = "llm-d-async:drain-limit:dynamo-batch"


def _render(path: Path, replacements: dict[str, str]) -> str:
    rendered = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        rendered = rendered.replace(f"${{{name}}}", value)
    assert "${" not in rendered
    return rendered


def test_async_values_preserve_namespace_readiness_and_leased_pool_gates() -> None:
    rendered = _render(
        EXAMPLE_ROOT / "llm-d-async-planner-values.yaml",
        {"NAMESPACE": SENTINEL_NAMESPACE},
    )
    values = yaml.safe_load(rendered)

    queue = values["ap"]["redis"]["queuesConfig"][0]
    assert queue["gate_type"] == "prometheus-query"
    assert queue["gate_params"]["query"] == (
        "min(dynamo_frontend_model_ready{"
        'model="Qwen/Qwen3-0.6B",'
        f'namespace="{SENTINEL_NAMESPACE}"'
        "})"
    )
    pool = values["ap"]["workerPools"][0]
    assert pool["gate_type"] == "wait-on-refuse"
    assert pool["gate_params"]["gate"]["gate_type"] == "redis-leased-rate"
    assert pool["gate_params"]["gate"]["gate_params"]["control_key"] == CONTROL_KEY
    assert values["ap"]["imagePullPolicy"] == "IfNotPresent"


def test_planner_manifest_has_recreate_fence_and_scoped_rbac() -> None:
    rendered = _render(
        PLANNER_ROOT / "planner-poc.yaml",
        {
            "NAMESPACE": SENTINEL_NAMESPACE,
            "PLANNER_IMAGE": SENTINEL_IMAGE,
            "BATCH_TENANT": SENTINEL_TENANT,
            "BATCH_WARM_REPLICAS": SENTINEL_WARM_REPLICAS,
        },
    )
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    assert all(
        document["metadata"].get("namespace") == SENTINEL_NAMESPACE
        for document in documents
    )
    role = next(document for document in documents if document["kind"] == "Role")
    base_rule = next(
        rule
        for rule in role["rules"]
        if rule["resources"] == ["dynamographdeploymentscalingadapters"]
    )
    assert base_rule["resourceNames"] == ["qwen3-0-6b-batch-worker"]
    assert set(base_rule["verbs"]) == {"get", "patch"}
    scale_rule = next(
        rule
        for rule in role["rules"]
        if rule["resources"] == ["dynamographdeploymentscalingadapters/scale"]
    )
    assert scale_rule["resourceNames"] == ["qwen3-0-6b-batch-worker"]
    assert scale_rule["verbs"] == ["patch"]

    role_binding = next(
        document for document in documents if document["kind"] == "RoleBinding"
    )
    assert role_binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "planner-serviceaccount",
            "namespace": SENTINEL_NAMESPACE,
        }
    ]
    assert role_binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": "qwen3-0-6b-batch-planner-scaling-adapter",
    }

    configmap = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap"
        and document["metadata"]["name"] == "qwen3-0-6b-batch-planner-config"
    )
    planner_config = yaml.safe_load(configmap["data"]["planner.yaml"])
    assert planner_config["namespace"] == (f"{SENTINEL_NAMESPACE}-qwen3-0-6b-batch")
    assert planner_config["batch_scheduling"]["gateway"]["tenant"] == (SENTINEL_TENANT)
    assert planner_config["batch_scheduling"]["redis"]["control_key"] == CONTROL_KEY
    assert planner_config["min_endpoint"] == 0
    assert planner_config["batch_scheduling"]["pool"]["min_replicas"] == 1


@pytest.mark.parametrize("warm_replicas", [0, 1])
def test_planner_manifest_renders_configurable_batch_warm_capacity(
    warm_replicas: int,
) -> None:
    rendered = _render(
        PLANNER_ROOT / "planner-poc.yaml",
        {
            "NAMESPACE": SENTINEL_NAMESPACE,
            "PLANNER_IMAGE": SENTINEL_IMAGE,
            "BATCH_TENANT": SENTINEL_TENANT,
            "BATCH_WARM_REPLICAS": str(warm_replicas),
        },
    )
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    configmap = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap"
        and document["metadata"]["name"] == "qwen3-0-6b-batch-planner-config"
    )
    planner_config = yaml.safe_load(configmap["data"]["planner.yaml"])

    assert planner_config["min_endpoint"] == 0
    assert planner_config["batch_scheduling"]["pool"]["min_replicas"] == warm_replicas

    dgd = next(
        document
        for document in documents
        if document["kind"] == "DynamoGraphDeployment"
    )
    planner = dgd["spec"]["components"][0]
    assert planner["replicas"] == 1
    assert planner["runtimeVersionOverride"] == "1.5.0"
    assert planner["podTemplate"]["metadata"]["annotations"] == {
        "nvidia.com/deployment-strategy": "Recreate"
    }
    container = planner["podTemplate"]["spec"]["containers"][0]
    assert container["image"] == SENTINEL_IMAGE
    assert container["imagePullPolicy"] == "IfNotPresent"
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    assert env["DYN_NAMESPACE"] == (f"{SENTINEL_NAMESPACE}-qwen3-0-6b-batch")
    assert env["DYN_PARENT_DGD_K8S_NAMESPACE"] == SENTINEL_NAMESPACE
    assert env["DYN_PARENT_DGD_K8S_NAME"] == "qwen3-0-6b-batch"
    assert env["DYN_PLANNER_BATCH_REDIS_URL"] == ("redis://batch-gateway-valkey:6379/0")


def test_frontend_template_uses_branch_image_and_model_credentials() -> None:
    rendered = _render(
        EXAMPLE_ROOT / "dynamo.yaml",
        {"FRONTEND_IMAGE": SENTINEL_IMAGE},
    )
    dgd = yaml.safe_load(rendered)
    assert dgd["metadata"]["name"] == "qwen3-0-6b-batch"
    frontend = next(
        component
        for component in dgd["spec"]["components"]
        if component["name"] == "Frontend"
    )
    assert frontend["runtimeVersionOverride"] == "1.5.0"
    container = frontend["podTemplate"]["spec"]["containers"][0]
    assert container["image"] == SENTINEL_IMAGE
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert {entry["name"] for entry in container["env"]} == {"HF_HOME"}
    assert container["envFrom"] == [{"secretRef": {"name": "hf-token-secret"}}]
    assert container["volumeMounts"] == [
        {"name": "model-cache", "mountPath": "/opt/models"}
    ]
    worker = next(
        component
        for component in dgd["spec"]["components"]
        if component["name"] == "worker"
    )
    assert worker["scalingAdapter"] == {}
    assert f"{dgd['metadata']['name']}-{worker['name']}" == ("qwen3-0-6b-batch-worker")
