# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from tests.deploy.n2_utils import DEFAULT_RELEASE_LINE, MODELS
from tests.frontend import test_strict_schema_upgrade as suite

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.sglang,
    pytest.mark.framework_agnostic,
]
OLD = "registry/sglang:1.5.0"
NEW = "registry/sglang:1.7.0-fixed@sha256:" + "b" * 64


def getoption(options):
    def read_option(name, default=None):
        return options.get(name, default)

    return read_option


@pytest.mark.parametrize(
    "additional_file",
    ["", "tests/frontend/test_strict_schema_upgrade.py"],
    ids=["single-file", "upgrade-matrix"],
)
def test_deploy_action_places_test_files_before_options(tmp_path, additional_file):
    action = yaml.safe_load(
        (suite.ROOT / ".github/actions/dynamo-deploy-test/action.yml").read_text()
    )
    step = next(s for s in action["runs"]["steps"] if s.get("id") == "deploy")
    script = re.sub(r"\$\{\{.*?\}\}", "ci-placeholder", step["run"])
    binary = tmp_path / "bin" / "python"
    binary.parent.mkdir()
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    capture = tmp_path / "args.json"
    env = os.environ.copy()
    env.update(
        PATH=str(binary.parent) + os.pathsep + env["PATH"],
        CAPTURE_FILE=str(capture),
        TEST_FILE="tests/deploy/test_n2_compatibility.py",
        ADDITIONAL_TEST_FILE=additional_file,
        NAMESPACE="default",
        IMAGE="registry/sglang:1.6.0-ci",
        FRONTEND_IMAGE="registry/frontend:1.6.0-ci",
        MODEL_CACHE_PVC="cache",
        EXTRA_PYTEST_ARGS="-m k8s -n 0",
        FRAMEWORK="",
        PROFILE="",
        TEST_NAME="n2-compatibility",
        DYNAMO_TEST_PLATFORM="amd64",
        DD_API_KEY="",
    )
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = json.loads(capture.read_text())
    files = [env["TEST_FILE"]] + ([additional_file] if additional_file else [])
    assert args[: 2 + len(files)] == ["-m", "pytest", *files]
    assert args[2 + len(files)] == "--namespace=default"
    assert "" not in args


@pytest.mark.parametrize(
    "enabled,baseline,target",
    [(True, OLD, NEW), (False, None, None)],
    ids=["explicit-pair", "disabled"],
)
def test_upgrade_matrix_is_bypassed_for_explicit_pair_or_disabled_run(
    enabled, baseline, target
):
    options = {
        "--run-strict-schema-upgrade": enabled,
        "--schema-upgrade-baseline-image": baseline,
        "--schema-upgrade-target-image": target,
    }
    metafunc = SimpleNamespace(
        fixturenames=("upgrade_images",),
        config=SimpleNamespace(getoption=getoption(options)),
        parametrize=MagicMock(),
    )

    suite.pytest_generate_tests(metafunc)

    metafunc.parametrize.assert_not_called()


def test_upgrade_matrix_uses_catalog_workers_and_candidate_image(monkeypatch):
    monkeypatch.delenv("N2_RELEASE_LINE", raising=False)
    candidate = "registry/sglang:1.7.0-fixed"
    options = {
        "--run-strict-schema-upgrade": True,
        "--schema-upgrade-baseline-image": None,
        "--schema-upgrade-target-image": None,
        "--image": candidate,
    }
    metafunc = SimpleNamespace(
        fixturenames=("upgrade_images",),
        config=SimpleNamespace(getoption=getoption(options)),
        parametrize=MagicMock(),
    )

    suite.pytest_generate_tests(metafunc)

    releases = json.loads((suite.ROOT / "tests/deploy/n2/releases.json").read_text())
    major, minor = map(int, DEFAULT_RELEASE_LINE.split("."))
    baseline_releases = [f"{major}.{minor - age}" for age in (1, 2)]
    expected_images = [
        (releases[line]["worker"], candidate) for line in baseline_releases
    ]
    metafunc.parametrize.assert_called_once_with(
        "upgrade_images",
        expected_images,
        indirect=True,
        ids=["old-worker-1", "old-worker-2"],
    )


@pytest.mark.parametrize(
    "baseline,target",
    [
        (None, NEW),
        (OLD, "registry/sglang:latest"),
        (OLD, "registry/sglang:1.7.0@sha256:invalid"),
        (NEW, OLD),
        (OLD, "registry/sglang:1.5.0@sha256:" + "b" * 64),
        (
            OLD + "@sha256:" + "a" * 64,
            "registry/sglang:1.7.0@sha256:" + "a" * 64,
        ),
    ],
)
def test_invalid_upgrade_images_fail_setup(baseline, target):
    with pytest.raises(pytest.UsageError):
        suite._validate_images(baseline, target)


@pytest.mark.parametrize(
    "baseline,target,digest",
    [("1.6.9", "1.6.10", ""), ("1.9.0", "1.10.0", "@sha256:")],
)
def test_other_release_pairs_use_numeric_version_order(baseline, target, digest):
    suite._validate_images(
        f"registry/sglang:{baseline}" + (digest + "a" * 64 if digest else ""),
        f"registry/sglang:{target}" + (digest + "b" * 64 if digest else ""),
    )


@pytest.mark.parametrize(
    "enabled,processor",
    [(False, "sglang"), (True, "sglang"), (False, "dynamo")],
    ids=["sglang-default", "sglang-compat", "rust"],
)
def test_upgrade_manifest_keeps_versions_and_processor_explicit(enabled, processor):
    suite._validate_images(OLD, NEW)
    spec = suite._upgrade_spec(NEW, enabled, "test", "cache", "/models", processor)
    components = {c["name"]: c for c in spec.spec()["spec"]["components"]}
    for component in components.values():
        assert component["runtimeVersionOverride"] == "1.7.0"
        assert component["podTemplate"]["spec"]["containers"][0]["image"] == NEW
    frontend = components["Frontend"]["podTemplate"]["spec"]["containers"][0]
    assert frontend["command"] == ["python3", "-m", "dynamo.frontend"]
    assert frontend["args"][:2] == ["--dyn-chat-processor", processor]
    if processor == "sglang":
        assert frontend["args"][2:] == [
            "--tool-call-parser",
            "qwen25",
            "--reasoning-parser",
            "qwen3",
        ]
    else:
        assert frontend["args"] == ["--dyn-chat-processor", "dynamo"]
    assert next(e["value"] for e in frontend["env"] if e["name"] == suite.FLAG) == str(
        int(enabled)
    )
    worker = components["decode"]["podTemplate"]["spec"]["containers"][0]
    worker_args = worker["args"]
    assert MODELS["chat"][1] in " ".join(worker_args)
    if processor == "sglang":
        assert "--dyn-tool-call-parser" not in worker_args
        assert "--dyn-reasoning-parser" not in worker_args
    else:
        assert worker_args[worker_args.index("--dyn-tool-call-parser") + 1] == "qwen25"
        assert worker_args[worker_args.index("--dyn-reasoning-parser") + 1] == "qwen3"


def completion():
    return {
        "status": 200,
        "body": json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": MODELS["chat"][0],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ),
    }


@pytest.mark.parametrize("default_status", [200, 400])
async def test_default_regression_is_not_hidden_by_successful_migration(
    monkeypatch, tmp_path, default_status
):
    options = {
        "--namespace": "test",
        "--schema-upgrade-model-cache-pvc": "cache",
        "--schema-upgrade-model-cache-mount": "/models",
    }
    request = SimpleNamespace(
        node=SimpleNamespace(name="upgrade"),
        config=SimpleNamespace(getoption=getoption(options)),
    )
    default = (
        completion()
        if default_status == 200
        else {"status": 400, "body": "strict schema rejected"}
    )
    probe = AsyncMock(side_effect=[completion(), default, completion()])
    monkeypatch.setattr(suite, "_probe", probe)
    monkeypatch.setattr(suite, "resolve_test_output_path", lambda _: str(tmp_path))
    if default_status == 400:
        with pytest.raises(AssertionError, match="default: HTTP 400"):
            await suite.test_strict_schema_upgrade(request, (OLD, NEW), "dynamo")
    else:
        await suite.test_strict_schema_upgrade(request, (OLD, NEW), "dynamo")
    assert probe.await_count == 3
    calls = probe.await_args_list
    assert [call.args[:2] for call in calls] == [
        (OLD, False),
        (NEW, False),
        (NEW, True),
    ]
    assert all(call.args[2] is calls[0].args[2] for call in calls)
    assert all(call.args[-1] == "dynamo" for call in calls)
    report = json.loads(next(tmp_path.glob("*/comparison.json")).read_text())
    assert report["processor"] == "dynamo"
    assert report["stages"]["default"]["status"] == default_status
    assert report["stages"]["compat"]["status"] == 200


@pytest.mark.parametrize(
    "probe_error", [False, True], ids=["http-400", "deployment-error"]
)
async def test_invalid_baseline_does_not_claim_an_upgrade_regression(
    monkeypatch, tmp_path, probe_error
):
    options = {
        "--namespace": "test",
        "--model-cache-pvc": "cache",
        "--model-cache-mount": "/shared-models",
    }
    request = SimpleNamespace(
        node=SimpleNamespace(name="upgrade"),
        config=SimpleNamespace(getoption=getoption(options)),
        session=SimpleNamespace(shouldstop=None),
    )
    if probe_error:
        probe = AsyncMock(side_effect=RuntimeError("deployment failed"))
        expected_error = RuntimeError
        match = "deployment failed"
    else:
        probe = AsyncMock(return_value={"status": 400, "body": "baseline rejected"})
        expected_error = AssertionError
        match = "baseline: HTTP 400"
    monkeypatch.setattr(suite, "_probe", probe)
    monkeypatch.setattr(suite, "resolve_test_output_path", lambda _: str(tmp_path))

    with pytest.raises(expected_error, match=match):
        await suite.test_strict_schema_upgrade(request, (OLD, NEW), "sglang")

    assert probe.await_count == 1
    assert probe.await_args.args[5:7] == ("cache", "/shared-models")
    if probe_error:
        assert request.session.shouldstop == (
            "Schema upgrade deployment failed; remaining pairs are not validated"
        )
    else:
        assert request.session.shouldstop is None


@pytest.mark.parametrize("leftovers", [False, True])
async def test_probe_preserves_400_and_stops_on_leftover_pods(
    monkeypatch, tmp_path, leftovers
):
    managed = MagicMock()
    managed.__aenter__ = AsyncMock(return_value=managed)
    managed.__aexit__ = AsyncMock(return_value=False)
    pod = MagicMock()
    managed.get_pods.side_effect = [
        {"Frontend": [pod]},
        {"Frontend": [pod] if leftovers else [], "decode": []},
    ]
    forward = SimpleNamespace(local_port=12345)
    managed.port_forward.return_value = forward
    response = MagicMock(status_code=400, text="strict schema rejected")
    managed.send_request_with_port_forward_retry.return_value = (response, forward)
    monkeypatch.setattr(suite, "ManagedDeployment", lambda *a, **kw: managed)
    monkeypatch.setattr(suite, "wait_for_model_availability", lambda *a: True)
    output = tmp_path / "probe"
    args = (
        NEW,
        False,
        {"model": MODELS["chat"][0]},
        "test",
        "test",
        "cache",
        "/models",
        output,
        "dynamo",
    )
    if leftovers:
        with pytest.raises(AssertionError, match="remaining Pods"):
            await suite._probe(*args)
    else:
        result = await suite._probe(*args)
        assert result == {"status": 400, "body": "strict schema rejected"}
    assert json.loads((output / "response.json").read_text())["status"] == 400
    manifest = yaml.safe_load((output / "deployment.yaml").read_text())
    components = {
        component["name"]: component for component in manifest["spec"]["components"]
    }
    frontend_args = components["Frontend"]["podTemplate"]["spec"]["containers"][0][
        "args"
    ]
    assert frontend_args == ["--dyn-chat-processor", "dynamo"]
    response.close.assert_called_once()
    managed.__aexit__.assert_awaited_once()
