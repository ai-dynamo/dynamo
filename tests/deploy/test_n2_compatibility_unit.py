# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from kubernetes_asyncio.client import exceptions

from tests.deploy import test_n2_compatibility as suite
from tests.deploy.dgd_utils import (
    DeploymentSpec,
    DeploymentStartupError,
    ManagedDeployment,
    PodStatusDetail,
)
from tests.deploy.n2_utils import (
    MODELS,
    compatibility_spec,
    runtime_version,
    version_matrix,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.framework_agnostic,
]
ROOT = Path(__file__).resolve().parents[2]


def test_matrix_uses_both_age_directions_without_candidate_controls():
    releases = json.loads((ROOT / "tests/deploy/n2/releases.json").read_text())
    pairs = version_matrix(releases, "1.5", "candidate-fe", "candidate-wk")
    assert [(p.frontend, p.worker) for p in pairs] == [
        (releases["1.4"]["frontend"], "candidate-wk"),
        ("candidate-fe", releases["1.4"]["worker"]),
        (releases["1.3"]["frontend"], "candidate-wk"),
        ("candidate-fe", releases["1.3"]["worker"]),
    ]
    with pytest.raises(KeyError):
        version_matrix(releases, "1.6", "fe", "wk")


@pytest.mark.parametrize("scenario", ["chat", "embedding"])
def test_example_patching_preserves_command_and_sets_component_versions(scenario):
    releases = json.loads((ROOT / "tests/deploy/n2/releases.json").read_text())
    pair = version_matrix(
        releases,
        "1.5",
        "registry:5000/fe:1.5.0-ci-abc",
        "registry:5000/wk:1.5.0-ci-abc",
    )[0]
    spec = compatibility_spec(ROOT, pair, scenario, "test", "shared", "/models")
    components = spec.spec()["spec"]["components"]
    assert components[0]["runtimeVersionOverride"] == "1.4.2"
    assert components[1]["runtimeVersionOverride"] == "1.5.0"
    worker = components[1]["podTemplate"]["spec"]["containers"][0]
    assert worker["command"] == ["python3", "-m", "dynamo.sglang"]
    if scenario == "embedding":
        assert "--embedding-worker" in worker["args"]
        assert "--use-sglang-tokenizer" in worker["args"]
    assert MODELS[scenario][1] in spec["decode"].model
    assert "pvcs" not in spec.spec()["spec"]
    for component in components:
        pod = component["podTemplate"]["spec"]
        assert {
            "name": "shared",
            "persistentVolumeClaim": {"claimName": "shared"},
        } in pod["volumes"]
        main = next(c for c in pod["containers"] if c["name"] == "main")
        assert {"name": "shared", "mountPath": "/models"} in main["volumeMounts"]
        assert {"name": "HF_HOME", "value": "/models"} in main["env"]
    assert runtime_version("registry:5000/fe:1.5.0.dev20260911-ci") == "1.5.0"
    with pytest.raises(ValueError):
        compatibility_spec(ROOT, pair, scenario, "test", "", "/models")


@pytest.mark.parametrize("schema", ["v1alpha1", "v1beta1"])
def test_model_cache_mount_preserves_schema_and_is_idempotent(tmp_path, schema):
    container = {"name": "main", "env": [{"name": "KEEP", "value": "yes"}]}
    if schema == "v1beta1":
        body = {
            "components": [
                {"name": "decode", "podTemplate": {"spec": {"containers": [container]}}}
            ]
        }
    else:
        body = {"services": {"decode": {"envs": [{"name": "KEEP", "value": "yes"}]}}}
    path = tmp_path / "dgd.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": f"nvidia.com/{schema}",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": "test"},
                "spec": body,
            }
        )
    )
    spec = DeploymentSpec(str(path))
    spec.mount_model_cache_pvc("cache", "/cache")
    first = json.dumps(spec.spec(), sort_keys=True)
    spec.mount_model_cache_pvc("cache", "/cache")
    assert json.dumps(spec.spec(), sort_keys=True) == first
    result = spec.spec()["spec"]
    if schema == "v1beta1":
        assert "pvcs" not in result and "envs" not in result
        component = result["components"][0]
        assert "volumeMounts" not in component
        pod = component["podTemplate"]["spec"]
        assert pod["volumes"] == [
            {"name": "cache", "persistentVolumeClaim": {"claimName": "cache"}}
        ]
        assert pod["containers"][0]["volumeMounts"] == [
            {"name": "cache", "mountPath": "/cache"}
        ]
        assert pod["containers"][0]["env"] == [
            {"name": "KEEP", "value": "yes"},
            {"name": "HF_HOME", "value": "/cache"},
        ]
    else:
        assert result["pvcs"] == [{"name": "cache", "create": False}]
        assert result["envs"] == [{"name": "HF_HOME", "value": "/cache"}]
        assert result["services"]["decode"]["volumeMounts"] == [
            {"name": "cache", "mountPoint": "/cache"}
        ]
        assert result["services"]["decode"]["envs"] == [
            {"name": "KEEP", "value": "yes"}
        ]


def test_snapshot_preflight_reads_files_and_rejects_missing_blob(tmp_path):
    pair = version_matrix(
        {
            "1.4": {"frontend": "fe:1.4.2", "worker": "wk:1.4.2"},
            "1.3": {"frontend": "fe:1.3.2", "worker": "wk:1.3.2"},
        },
        "1.5",
        "fe:1.5.0",
        "wk:1.5.0",
    )[0]
    spec = compatibility_spec(ROOT, pair, "chat", "test", "cache", "/models")
    init = spec.spec()["spec"]["components"][1]["podTemplate"]["spec"][
        "initContainers"
    ][0]
    assert init["volumeMounts"] == [
        {"name": "cache", "mountPath": "/models", "readOnly": True}
    ]
    for name in ["config.json", "tokenizer.json", "model.safetensors"]:
        (tmp_path / name).write_bytes(b"test")
    command = [sys.executable, "-c", init["args"][0], str(tmp_path)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model.safetensors").symlink_to(tmp_path / "missing-blob")
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_preparation_job_downloads_the_exact_snapshots_without_a_gpu():
    job = yaml.safe_load((ROOT / "tests/deploy/n2/model-download.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    tree = ast.parse(container["args"][0])
    models = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "models"
    )
    assert models == dict(MODELS.values())
    assert "resources" not in container


async def test_delete_waits_for_cr_and_owned_pods(monkeypatch, tmp_path):
    deployment = ManagedDeployment(
        str(tmp_path),
        DeploymentSpec(str(ROOT / "examples/backends/sglang/deploy/agg.yaml")),
        "test",
    )
    deployment._deployment_name = "test"
    not_found = exceptions.ApiException(status=404)
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(),
        get_namespaced_custom_object=AsyncMock(side_effect=[{}, not_found, not_found]),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(
            side_effect=[
                SimpleNamespace(items=[1]),
                SimpleNamespace(items=[1]),
                SimpleNamespace(items=[]),
            ]
        )
    )
    monkeypatch.setattr("tests.deploy.dgd_utils.asyncio.sleep", AsyncMock())
    await deployment._delete_deployment()
    assert deployment._core_api.list_namespaced_pod.await_count == 3
    assert (
        deployment._custom_api.delete_namespaced_custom_object.call_args.kwargs[
            "body"
        ].propagation_policy
        == "Foreground"
    )


async def test_cleanup_keeps_request_failure_and_records_cleanup_failure(tmp_path):
    deployment = ManagedDeployment(str(tmp_path), SimpleNamespace(name="test"), "test")
    deployment._logger = logging.getLogger(__name__)
    failure = RuntimeError("delete failed")
    deployment._cleanup = AsyncMock(side_effect=failure)
    primary = ValueError("bad response")
    await deployment.__aexit__(ValueError, primary, None)
    assert deployment.cleanup_errors == [failure]
    with pytest.raises(RuntimeError, match="delete failed"):
        await deployment.__aexit__(None, None, None)


@pytest.mark.parametrize("exit_code,fatal", [(0, False), (1, True)])
async def test_restarted_init_container_only_fails_for_nonzero_exit(
    monkeypatch, tmp_path, exit_code, fatal
):
    deployment = ManagedDeployment(
        str(tmp_path),
        SimpleNamespace(name="test", api_version="v1beta1"),
        "test",
        fail_fast_startup=True,
    )
    deployment._custom_api = SimpleNamespace(
        get_namespaced_custom_object=AsyncMock(
            return_value={
                "status": {
                    "state": "successful",
                    "conditions": [{"type": "Ready", "status": "True"}],
                }
            }
        )
    )
    monkeypatch.setattr(
        deployment,
        "_get_pod_status_details",
        AsyncMock(
            return_value=[
                PodStatusDetail(
                    "pod",
                    "init",
                    "Terminated",
                    "Completed" if exit_code == 0 else "Error",
                    exit_code=exit_code,
                    restart_count=2,
                )
            ]
        ),
    )
    if fatal:
        with pytest.raises(DeploymentStartupError):
            await deployment._wait_for_ready(timeout=1)
    else:
        assert await deployment._wait_for_ready(timeout=1)


async def test_delete_timeout_is_reported(monkeypatch, tmp_path):
    deployment = ManagedDeployment(
        str(tmp_path), SimpleNamespace(name="test", api_version="v1beta1"), "test"
    )
    deployment._deployment_name = "test"
    deployment._custom_api = SimpleNamespace(
        delete_namespaced_custom_object=AsyncMock(),
        get_namespaced_custom_object=AsyncMock(return_value={}),
    )
    deployment._core_api = SimpleNamespace(
        list_namespaced_pod=AsyncMock(return_value=SimpleNamespace(items=[1]))
    )
    times = iter([0, 121])
    monkeypatch.setattr(
        "tests.deploy.dgd_utils.time", SimpleNamespace(monotonic=lambda: next(times))
    )
    with pytest.raises(TimeoutError, match="not deleted"):
        await deployment._delete_deployment()


async def test_cleanup_failure_stops_session_without_replacing_body_error(
    monkeypatch, tmp_path
):
    failure = RuntimeError("cleanup failed")

    class Deployment:
        cleanup_errors = [failure]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    options = {"--model-cache-pvc": "shared", "--model-cache-mount": "/models"}
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=options.__getitem__),
        node=SimpleNamespace(name="test"),
        session=SimpleNamespace(shouldstop=False),
    )
    pair = version_matrix(
        {
            "1.4": {"frontend": "fe:1.4.2", "worker": "wk:1.4.2"},
            "1.3": {"frontend": "fe:1.3.1", "worker": "wk:1.3.1"},
        },
        "1.5",
        "fe:1.5.0",
        "wk:1.5.0",
    )
    monkeypatch.setattr(
        suite, "ManagedDeployment", lambda *args, **kwargs: Deployment()
    )
    monkeypatch.setattr(suite, "resolve_test_output_path", lambda _: str(tmp_path))
    fixture = suite.mixed_deployment.__wrapped__(request, pair, 0, "chat", "test")
    await anext(fixture)
    primary = ValueError("bad response")
    with pytest.raises(ValueError, match="bad response"):
        await fixture.athrow(primary)
    assert (
        request.session.shouldstop
        == "N-2 cleanup failed; remaining pairs are not validated"
    )
