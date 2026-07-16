# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for the Python-to-Go DGD override protocol."""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dynamo.profiler.utils.dgd_override import apply_dgd_overrides

_RUN_INTEGRATION_ENV = "DYNAMO_RUN_DGD_OVERRIDE_GO_INTEGRATION"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.timeout(360),
    pytest.mark.skipif(
        os.environ.get(_RUN_INTEGRATION_ENV) != "1",
        reason=f"set {_RUN_INTEGRATION_ENV}=1 to build and test the Go helper",
    ),
]


@pytest.fixture(scope="session")
def go_override_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.fail("Go is required when the DGD override integration test is enabled")

    repository_root = Path(__file__).resolve().parents[7]
    operator_dir = repository_root / "deploy" / "operator"
    binary = tmp_path_factory.mktemp("dgd-override-go") / "dgd-apply-overrides"
    environment = os.environ.copy()
    environment.update({"CGO_ENABLED": "0", "GOWORK": "off"})

    try:
        result = subprocess.run(
            [
                go,
                "build",
                "-mod=readonly",
                "-trimpath",
                "-ldflags=-s -w",
                "-o",
                str(binary),
                "./cmd/dgd-apply-overrides",
            ],
            cwd=operator_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("building dgd-apply-overrides timed out after 300 seconds")

    if result.returncode != 0:
        pytest.fail(
            "building dgd-apply-overrides failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not os.access(binary, os.X_OK):
        pytest.fail(f"Go build did not produce an executable at {binary}")
    return binary


def _named(items: list[dict], name: str) -> dict:
    matches = [item for item in items if item.get("name") == name]
    assert len(matches) == 1, f"expected one item named {name!r}, got {matches!r}"
    return matches[0]


def test_python_adapter_invokes_real_go_override_engine(
    go_override_binary: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    blueprint = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": "generated",
            "namespace": "default",
            "labels": {"base": "true"},
        },
        "spec": {
            "backendFramework": "vllm",
            "components": [
                {"name": "Frontend", "type": "frontend"},
                {
                    "name": "Worker",
                    "type": "worker",
                    "frontendSidecar": "sidecar",
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "image": "old-image",
                                    "args": ["--base"],
                                    "env": [
                                        {"name": "KEEP", "value": "keep"},
                                        {"name": "CHANGE", "value": "old"},
                                    ],
                                },
                                {"name": "sidecar", "image": "sidecar-image"},
                            ]
                        }
                    },
                },
            ],
        },
    }
    legacy_override = {
        "spec": {
            "services": {
                "Worker": {
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "new-image",
                            "args": ["--override"],
                            "env": [
                                {"name": "CHANGE", "value": "new"},
                                {"name": "ADDED", "value": "added"},
                            ],
                        }
                    }
                }
            }
        }
    }
    original_blueprint = copy.deepcopy(blueprint)
    original_override = copy.deepcopy(legacy_override)

    with caplog.at_level(logging.WARNING):
        effective = apply_dgd_overrides(
            blueprint,
            legacy_override,
            binary_path=str(go_override_binary),
        )

    assert effective["apiVersion"] == "nvidia.com/v1beta1"
    assert effective["kind"] == "DynamoGraphDeployment"
    assert effective["metadata"]["name"] == "generated"
    assert effective["metadata"]["namespace"] == "default"
    assert effective["metadata"]["labels"] == {"base": "true"}

    components = effective["spec"]["components"]
    assert _named(components, "Frontend")["type"] == "frontend"
    worker = _named(components, "Worker")
    assert worker["frontendSidecar"] == "sidecar"
    containers = worker["podTemplate"]["spec"]["containers"]
    assert _named(containers, "sidecar")["image"] == "sidecar-image"
    main = _named(containers, "main")
    assert main["image"] == "new-image"
    assert main["args"] == ["--base", "--override"]
    assert {entry["name"]: entry["value"] for entry in main["env"]} == {
        "KEEP": "keep",
        "CHANGE": "new",
        "ADDED": "added",
    }

    assert blueprint == original_blueprint
    assert legacy_override == original_override
    assert "treating it as nvidia.com/v1alpha1" in caplog.text
