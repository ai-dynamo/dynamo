# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from kubernetes_asyncio.client.exceptions import ApiException

from tests.deploy import test_n2_rolling_upgrade as suite
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.deploy.n2.rolling_contract import Cleanup, require_executed_cases
from tests.deploy.n2.rolling_http import ContractError
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
    frontend_spec = components[0]["podTemplate"]["spec"]
    frontend = frontend_spec["containers"][0]
    assert "initContainers" not in frontend_spec
    assert "volumes" not in frontend_spec
    assert "volumeMounts" not in frontend
    assert all(
        e["name"] not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DYN_SYSTEM_PORT")
        for e in frontend["env"]
    )
    worker_spec = components[1]["podTemplate"]["spec"]
    assert worker_spec["initContainers"][0]["name"] == "model"
    assert worker_spec["volumes"] == [{"name": "model", "emptyDir": {}}]
    assert all(c["runtimeVersionOverride"] == "1.4.2" for c in components)
    worker = worker_spec["containers"][0]
    assert not any(e["name"] == "DYN_SYSTEM_PORT" for e in worker["env"])
    assert worker["volumeMounts"] == [{"name": "model", "mountPath": "/model"}]
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


@pytest.mark.parametrize("timeout", [None, 330])
async def test_kubectl_honors_default_and_explicit_timeout(timeout):
    with patch.object(suite, "command", return_value="") as command:
        await suite.kubectl(
            "ns", "get", "pods", **({} if timeout is None else {"timeout": timeout})
        )
    assert command.call_args.kwargs["timeout"] == (120 if timeout is None else timeout)


@pytest.mark.parametrize("primary_failure", [False, True])
async def test_cleanup_preserves_primary_error_and_runs_remaining_operations(
    primary_failure,
):
    cleanup = Cleanup()
    completed = []

    async def failed_delete():
        raise subprocess.TimeoutExpired("kubectl delete", 330)

    async def next_delete():
        completed.append("remaining resources")

    with pytest.raises(ContractError if primary_failure else ExceptionGroup):
        try:
            if primary_failure:
                raise ContractError("request failed")
        finally:
            active_failure = sys.exc_info()[0] is not None
            await cleanup.attempt("delete deployment", failed_delete)
            await cleanup.attempt("delete quota", next_delete)
            cleanup.finish(active_failure)
    assert completed == ["remaining resources"]
    assert len(cleanup.errors) == 1
    assert "delete deployment" in str(cleanup.errors[0])
    assert isinstance(cleanup.errors[0].__cause__, subprocess.TimeoutExpired)


@pytest.mark.parametrize(
    "count,skipped", [(0, False), (7, False), (8, True), (8, False)]
)
def test_execution_guard_rejects_missing_incomplete_or_skipped_cases(
    tmp_path, count, skipped
):
    root = ET.Element("testsuites")
    for index in range(count):
        case = ET.SubElement(root, "testcase", name=f"test_n2_rolling_upgrade[{index}]")
        if skipped:
            ET.SubElement(case, "skipped")
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path)
    if count == 8 and not skipped:
        require_executed_cases([path])
    else:
        with pytest.raises(ContractError):
            require_executed_cases([path])


@pytest.mark.parametrize("change", ["sglang", "core", "frontend"])
def test_rolling_pr_lane_builds_all_required_candidate_images(change):
    workflow = yaml.safe_load((suite.ROOT / ".github/workflows/pr.yaml").read_text())
    outcomes = {}
    for name in (
        "operator",
        "frontend-build",
        "sglang-build",
        "frontend-copy-to-acr",
        "sglang-copy-to-acr",
        "rolling-upgrade-compatibility",
    ):
        condition = workflow["jobs"][name]["if"]
        condition = re.sub(
            r"needs\.changed-files\.outputs\.(\w+) == 'true'",
            lambda match: str(match[1] in (change, "run_deploy_tests")),
            condition,
        )
        condition = re.sub(
            r"needs\.([\w-]+)\.result == 'success'",
            lambda match: str(outcomes.get(match[1], False)),
            condition,
        )
        condition = (
            condition.replace("always()", "True")
            .replace("!cancelled()", "True")
            .replace("&&", " and ")
            .replace("||", " or ")
        )
        outcomes[name] = eval(" ".join(condition.split()), {"__builtins__": {}})
    assert all(outcomes.values()), outcomes


async def test_watch_resumes_from_last_resource_version_and_fails_on_expired_history(
    tmp_path,
):
    versions = []

    async def events(*args, **kwargs):
        versions.append(kwargs["resource_version"])
        if len(versions) == 2:
            raise ApiException(status=410, reason="expired history")
        yield {
            "type": "BOOKMARK",
            "raw_object": {"metadata": {"resourceVersion": "102"}},
        }

    with patch("kubernetes_asyncio.watch.Watch") as watch:
        watcher = watch.return_value.__aenter__.return_value
        watcher.stream = MagicMock(side_effect=events)
        with pytest.raises(ApiException) as error:
            await suite.watch_pods(AsyncMock(), "ns", "dgd", "100", None, tmp_path)
    assert error.value.status == 410
    assert versions == ["100", "102"]
