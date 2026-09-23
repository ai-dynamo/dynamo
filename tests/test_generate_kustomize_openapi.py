# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
    rendered = generator.generated_schema()
    schema = json.loads(rendered)

    assert rendered.splitlines()[1] == (
        '  "x-generated-warning": "Generated file. Do not edit this checked-in copy.",'
    )
    assert rendered.splitlines()[2] == (
        '  "x-regenerate-command": "python3 scripts/generate_kustomize_openapi.py",'
    )
    assert (
        schema["x-regenerate-command"]
        == "python3 scripts/generate_kustomize_openapi.py"
    )

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


def test_generated_beta_dgd_schema_merges_shared_env_by_name():
    generator = load_generator_module()
    schema = json.loads(generator.generated_schema())
    dgd = schema["definitions"]["nvidia.com.v1beta1.DynamoGraphDeployment"]
    env = dgd["properties"]["spec"]["properties"]["env"]

    assert env["x-kubernetes-patch-strategy"] == "merge"
    assert env["x-kubernetes-patch-merge-key"] == "name"


def test_missing_beta_shared_env_schema_reports_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    generator = load_generator_module()
    crd = yaml.safe_load(
        (generator.CRD_DIR / "nvidia.com_dynamographdeployments.yaml").read_text(
            encoding="utf-8"
        )
    )
    beta = next(
        version for version in crd["spec"]["versions"] if version["name"] == "v1beta1"
    )
    del beta["schema"]["openAPIV3Schema"]["properties"]["spec"]["properties"]["env"]
    crd_path = tmp_path / "dgd.yaml"
    crd_path.write_text(yaml.safe_dump(crd), encoding="utf-8")
    monkeypatch.setattr(generator, "CRD_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--check"])

    assert generator.main() == 1
    error = capsys.readouterr().err
    assert str(crd_path) in error
    assert "nvidia.com/v1beta1 DynamoGraphDeployment" in error
    assert "spec.env" in error
    assert "Traceback" not in error


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "base_env,patch_env,expected_env",
    [
        pytest.param(
            {"name": "HF_HOME", "value": "/model-cache"},
            {"name": "UCX_TLS", "value": "rc_x"},
            [
                {"name": "MODEL_NAME", "value": "test-model"},
                {"name": "HF_HOME", "value": "/model-cache"},
                {"name": "UCX_TLS", "value": "rc_x"},
            ],
            id="add-variable",
        ),
        pytest.param(
            {"name": "HF_HOME", "value": "/model-cache"},
            {"name": "HF_HOME", "value": "/cluster-cache"},
            [
                {"name": "MODEL_NAME", "value": "test-model"},
                {"name": "HF_HOME", "value": "/cluster-cache"},
            ],
            id="update-variable",
        ),
        pytest.param(
            {"name": "HF_HOME", "value": "/model-cache"},
            {
                "name": "HF_HOME",
                "value": None,
                "valueFrom": {
                    "configMapKeyRef": {"name": "cache-config", "key": "path"}
                },
            },
            [
                {"name": "MODEL_NAME", "value": "test-model"},
                {
                    "name": "HF_HOME",
                    "valueFrom": {
                        "configMapKeyRef": {"name": "cache-config", "key": "path"}
                    },
                },
            ],
            id="literal-to-value-from",
        ),
        pytest.param(
            {
                "name": "HF_HOME",
                "valueFrom": {
                    "configMapKeyRef": {"name": "cache-config", "key": "path"}
                },
            },
            {"name": "HF_HOME", "valueFrom": None, "value": "/cluster-cache"},
            [
                {"name": "MODEL_NAME", "value": "test-model"},
                {"name": "HF_HOME", "value": "/cluster-cache"},
            ],
            id="value-from-to-literal",
        ),
    ],
)
def test_generated_beta_dgd_schema_preserves_shared_env(
    tmp_path: Path, base_env: dict, patch_env: dict, expected_env: list[dict]
):
    """Preserve unrelated variables while merging entries and clearing old sources."""
    executable = os.environ.get("KUSTOMIZE_BIN") or shutil.which("kustomize")
    if executable is None:
        pytest.skip("kustomize is required for the shared environment merge test")

    generator = load_generator_module()
    (tmp_path / "schema.json").write_text(
        generator.generated_schema(), encoding="utf-8"
    )
    base = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "env-merge"},
        "spec": {
            "env": [
                {"name": "MODEL_NAME", "value": "test-model"},
                base_env,
            ]
        },
    }
    patch = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "env-merge"},
        "spec": {"env": [patch_env]},
    }
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": ["base.yaml"],
        "openapi": {"path": "schema.json"},
        "patches": [{"path": "patch.yaml"}],
    }
    for name, document in (
        ("base.yaml", base),
        ("patch.yaml", patch),
        ("kustomization.yaml", kustomization),
    ):
        (tmp_path / name).write_text(yaml.safe_dump(document), encoding="utf-8")

    result = subprocess.run(
        [executable, "build", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    env = yaml.safe_load(result.stdout)["spec"]["env"]
    assert {entry["name"]: entry for entry in env} == {
        entry["name"]: entry for entry in expected_env
    }
    assert len(env) == len(expected_env), "shared environment contains duplicate names"
