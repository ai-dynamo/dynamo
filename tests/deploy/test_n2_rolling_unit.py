# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock

import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.deploy.test_n2_rolling_upgrade import manifest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
]


def deployment(tmp_path):
    value = manifest(
        "test",
        {"frontend": "fe:1.4.2", "worker": "worker:1.4.2"},
        "embedding",
        {"id": "model", "revision": "fixed"},
        "worker:1.5.0",
    )
    path = tmp_path / "dgd.json"
    path.write_text(json.dumps(value))
    result = ManagedDeployment(str(tmp_path), DeploymentSpec(str(path)), "test")
    result._custom_api = AsyncMock()
    result._custom_api.get_namespaced_custom_object.return_value = value
    return result, value


def test_manifest_enforces_fixed_capacity_without_changing_discovery(tmp_path):
    _, value = deployment(tmp_path)
    components = value["spec"]["components"]
    assert [c["replicas"] for c in components] == [2, 2]
    for component in components:
        annotations = component["podTemplate"]["metadata"]["annotations"]
        assert annotations["nvidia.com/deployment-rolling-update-max-surge"] == "0"
        assert (
            annotations["nvidia.com/deployment-rolling-update-max-unavailable"] == "1"
        )
        env = component["podTemplate"]["spec"]["containers"][0]["env"]
        assert not any(e["name"] == "DYN_NAMESPACE" for e in env)
        assert any(
            e == {"name": "DYN_HEALTH_CHECK_ENABLED", "value": "false"} for e in env
        )
    worker = components[1]["podTemplate"]["spec"]["containers"][0]
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "--embedding-worker" in worker["command"]


async def test_image_update_preserves_unrelated_fields_and_tests_identity(tmp_path):
    managed, live = deployment(tmp_path)
    live["spec"]["components"][1]["podTemplate"]["spec"]["containers"].insert(
        0, {"name": "sidecar", "image": "sidecar:old"}
    )
    await managed.update_component_images({"decode": "worker@sha256:new"}, "1.5.0")
    kwargs = managed._custom_api.patch_namespaced_custom_object.call_args.kwargs
    patch = kwargs["body"]
    assert kwargs["_content_type"] == "application/json-patch+json"
    writes = [p for p in patch if p["op"] != "test"]
    assert writes == [
        {
            "op": "add",
            "path": "/spec/components/1/runtimeVersionOverride",
            "value": "1.5.0",
        },
        {
            "op": "replace",
            "path": "/spec/components/1/podTemplate/spec/containers/1/image",
            "value": "worker@sha256:new",
        },
    ]
    assert any(p["op"] == "test" and p["value"] == "worker:1.4.2" for p in patch)
    assert managed.deployment_spec["decode"].image == "worker@sha256:new"
    assert managed.deployment_spec["Frontend"].image == "fe:1.4.2"


async def test_rejected_patch_does_not_update_local_spec(tmp_path):
    managed, _ = deployment(tmp_path)
    managed._custom_api.patch_namespaced_custom_object.side_effect = ApiException(
        status=409
    )
    with pytest.raises(ApiException):
        await managed.update_component_images({"decode": "new"}, "1.5.0")
    assert managed.deployment_spec["decode"].image == "worker:1.4.2"
