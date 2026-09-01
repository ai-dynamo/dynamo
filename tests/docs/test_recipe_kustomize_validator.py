# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD = REPO_ROOT / "recipes" / "templates" / "kustomize"
VALIDATOR = REPO_ROOT / "scripts" / "validate-recipe-kustomization.py"
AGG_BASE = (
    REPO_ROOT
    / "recipes"
    / "templates"
    / "vllm"
    / "agg"
    / "deploy-v1beta1.template.yaml"
)
DISAGG_CASES = (
    (
        "vllm/disagg/deploy-v1beta1.template.yaml",
        "vllm-kv-transfer-config.yaml",
    ),
    (
        "vllm/disagg/deploy-v1beta1-compute-domain.template.yaml",
        "vllm-compute-domain-kv-transfer-config.yaml",
    ),
    (
        "sglang/disagg/deploy-v1beta1.template.yaml",
        "sglang-nixl-backend.yaml",
    ),
    ("trtllm/disagg/deploy-v1beta1.template.yaml", None),
)
AGG_CASES = (
    "vllm/agg/deploy-v1beta1.template.yaml",
    "sglang/agg/deploy-v1beta1.template.yaml",
    "sglang/agg/deploy-v1beta1-compute-domain.template.yaml",
    "trtllm/agg/deploy-v1beta1.template.yaml",
)

PLACEHOLDERS = {
    "your-base-recipe.yaml": "base.yaml",
    "your-model-cache-claim": "cluster-model-cache",
    "your-pull-secret": "cluster-registry-secret",
    "your-node-pool-label": "example.com/node-pool",
    "your-frontend-node-pool": "frontend",
    "your-worker-node-pool": "worker",
    "your-prefill-node-pool": "prefill",
    "your-decode-node-pool": "decode",
    "your-frontend-taint-key": "example.com/frontend",
    "your-frontend-taint-value": "true",
    "your-worker-taint-key": "example.com/worker",
    "your-worker-taint-value": "true",
    "your-prefill-taint-key": "example.com/prefill",
    "your-prefill-taint-value": "true",
    "your-decode-taint-key": "example.com/decode",
    "your-decode-taint-value": "true",
    "your-scheduler-name": "default-scheduler",
    "your-runtime-class-name": "nvidia",
    "your-priority-class-name": "high-priority",
    "your-worker-socket-interface": "eth0",
    "your-prefill-socket-interface": "eth0",
    "your-decode-socket-interface": "eth0",
    "your-worker-rdma-resource-count": "1",
    "your-prefill-rdma-resource-count": "1",
    "your-decode-rdma-resource-count": "1",
    "your-worker-startup-failure-threshold": "720",
    "your-prefill-startup-failure-threshold": "720",
    "your-decode-startup-failure-threshold": "720",
    "your-domain.example/rdma": "example.com/rdma",
    "your-domain.example~1rdma": "example.com~1rdma",
    "your-rdma.example.com/device": "example.com/rdma",
    "your-rdma.example.com~1device": "example.com~1rdma",
    "your-topology-key": "topology.kubernetes.io/zone",
}


def _documents(path: Path) -> list[dict[str, Any]]:
    return [
        document
        for document in yaml.safe_load_all(path.read_text())
        if isinstance(document, dict)
    ]


def _write_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump_all(documents, sort_keys=False))


def _dgd(path: Path) -> dict[str, Any]:
    return next(
        document
        for document in _documents(path)
        if document.get("kind") == "DynamoGraphDeployment"
    )


def _kustomize_bin() -> str:
    executable = os.environ.get("KUSTOMIZE_BIN") or shutil.which("kustomize")
    if executable is None:
        pytest.skip("Kustomize v5.8.1 is required for scaffold integration tests")
    return executable


def _filled_case(
    tmp_path: Path,
    *,
    with_case_override: bool = True,
    base: Path = AGG_BASE,
) -> Path:
    case = tmp_path / "cluster-kustomization"
    shutil.copytree(SCAFFOLD, case)
    shutil.copy2(base, case / "base.yaml")

    for path in case.rglob("*.yaml"):
        text = path.read_text()
        for placeholder, value in PLACEHOLDERS.items():
            text = text.replace(placeholder, value)
        path.write_text(text)

    if with_case_override:
        worker_env = _dgd(case / "base.yaml")["spec"]["components"][1]["podTemplate"][
            "spec"
        ]["containers"][0]["env"]
        env_index = len(worker_env)
        patch = [
            {"op": "test", "path": "/spec/components/1/name", "value": "Worker"},
            {"op": "test", "path": "/spec/components/1/type", "value": "worker"},
            {
                "op": "test",
                "path": "/spec/components/1/podTemplate/spec/containers/0/name",
                "value": "main",
            },
            {
                "op": "test",
                "path": (
                    "/spec/components/1/podTemplate/spec/containers/0/"
                    f"env/{env_index}/name"
                ),
                "value": "NCCL_SOCKET_IFNAME",
            },
            {
                "op": "test",
                "path": (
                    "/spec/components/1/podTemplate/spec/containers/0/"
                    f"env/{env_index}/value"
                ),
                "value": "eth0",
            },
            {
                "op": "replace",
                "path": (
                    "/spec/components/1/podTemplate/spec/containers/0/"
                    f"env/{env_index}/value"
                ),
                "value": "ens1f0",
            },
        ]
        patch_path = case / "patches" / "case-network-override.yaml"
        patch_path.write_text(yaml.safe_dump(patch, sort_keys=False))
        kustomization_path = case / "kustomization.yaml"
        kustomization = yaml.safe_load(kustomization_path.read_text())
        kustomization["patches"] = [
            {
                "target": {
                    "group": "nvidia.com",
                    "version": "v1beta1",
                    "kind": "DynamoGraphDeployment",
                },
                "path": "patches/case-network-override.yaml",
            }
        ]
        kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))
    return case


def _validate(
    case: Path, *, kustomize_bin: Optional[Path] = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(case / "base.yaml"),
            str(case / "kustomization.yaml"),
            "--kustomize-bin",
            str(kustomize_bin or _kustomize_bin()),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _filled_disagg_case(
    tmp_path: Path,
    relative_base: str,
    hook_patch: Optional[str],
) -> Path:
    case = tmp_path / "cluster-kustomization"
    shutil.copytree(SCAFFOLD, case)
    source = REPO_ROOT / "recipes" / "templates" / relative_base
    shutil.copy2(source, case / "base.yaml")

    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["resources"] = ["base.yaml"]
    kustomization["components"] = [
        reference.replace("/agg", "/disagg")
        for reference in kustomization["components"]
    ]
    if hook_patch is not None:
        kustomization["patches"] = [
            {
                "target": {
                    "group": "nvidia.com",
                    "version": "v1beta1",
                    "kind": "DynamoGraphDeployment",
                },
                "path": f"patches/{hook_patch}",
            }
        ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    for path in case.rglob("*.yaml"):
        text = path.read_text()
        for placeholder, value in PLACEHOLDERS.items():
            text = text.replace(placeholder, value)
        text = text.replace(
            "your-kv-transfer-config",
            "cluster-kv-transfer-config",
        ).replace("your-sglang-nixl-backend", "GDS")
        path.write_text(text)
    return case


def _component_guards(
    index: int,
    name: str,
    component_type: str,
) -> list[dict[str, Any]]:
    return [
        {"op": "test", "path": f"/spec/components/{index}/name", "value": name},
        {
            "op": "test",
            "path": f"/spec/components/{index}/type",
            "value": component_type,
        },
    ]


def _identity_guards(
    index: int,
    name: str,
    component_type: str,
) -> list[dict[str, Any]]:
    return _component_guards(index, name, component_type) + [
        {
            "op": "test",
            "path": (f"/spec/components/{index}/podTemplate/spec/containers/0/name"),
            "value": "main",
        },
    ]


def _write_component(
    component: Path,
    operations: list[dict[str, Any]],
) -> None:
    component.mkdir(parents=True, exist_ok=True)
    kustomization: dict[str, Any] = {
        "apiVersion": "kustomize.config.k8s.io/v1alpha1",
        "kind": "Component",
    }
    if operations:
        kustomization["patches"] = [
            {
                "target": {
                    "group": "nvidia.com",
                    "version": "v1beta1",
                    "kind": "DynamoGraphDeployment",
                },
                "path": "patch-dgd.yaml",
            }
        ]
        (component / "patch-dgd.yaml").write_text(
            yaml.safe_dump(operations, sort_keys=False)
        )
    (component / "kustomization.yaml").write_text(
        yaml.safe_dump(kustomization, sort_keys=False)
    )


def _neutral_component_case(
    tmp_path: Path,
    operations: list[dict[str, Any]],
) -> tuple[Path, Path]:
    case = _filled_case(tmp_path, with_case_override=False)
    reference = "components/network-interface/agg"
    component = case / reference
    _write_component(component, operations)
    return case, component


def _seed_worker_container_collection(
    case: Path,
    collection: str,
    containers: list[dict[str, Any]],
) -> None:
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    worker_spec = dgd["spec"]["components"][1]["podTemplate"]["spec"]
    worker_spec[collection] = containers
    _write_documents(base_path, documents)


def _assert_error(result: subprocess.CompletedProcess[str], code: str) -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"ERROR [{code}]" in result.stderr, result.stdout + result.stderr


def test_scaffold_inventory_is_complete() -> None:
    expected = {
        "README.md",
        "kustomization.yaml",
        "patches/vllm-kv-transfer-config.yaml",
        "patches/vllm-compute-domain-kv-transfer-config.yaml",
        "patches/sglang-nixl-backend.yaml",
    }
    for concern in (
        "cache-binding",
        "registry-credentials",
        "probes",
        "scheduling",
        "network-interface",
        "placement",
    ):
        for topology in ("agg", "disagg"):
            expected.add(f"components/{concern}/{topology}/kustomization.yaml")
            expected.add(f"components/{concern}/{topology}/patch-dgd.yaml")
    actual = {
        path.relative_to(SCAFFOLD).as_posix()
        for path in SCAFFOLD.rglob("*")
        if path.is_file()
    }
    assert expected == actual


def test_validator_accepts_ordered_components_and_case_override(tmp_path: Path) -> None:
    result = _validate(_filled_case(tmp_path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "validation passed" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("relative_base", AGG_CASES)
def test_validator_accepts_aggregate_templates(
    tmp_path: Path,
    relative_base: str,
) -> None:
    case = _filled_case(
        tmp_path,
        with_case_override=False,
        base=REPO_ROOT / "recipes" / "templates" / relative_base,
    )
    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""


def test_cache_binding_is_independent_of_volume_and_mount_names(
    tmp_path: Path,
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    for component in dgd["spec"]["components"][1:]:
        pod_spec = component["podTemplate"]["spec"]
        main = pod_spec["containers"][0]
        main["volumeMounts"][0]["name"] = "runtime-cache"
        pod_spec["volumes"][0]["name"] = "runtime-cache"
        pod_spec["volumes"][0]["persistentVolumeClaim"][
            "claimName"
        ] = "shared-model-cache"
    _write_documents(base_path, documents)

    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_accepts_optional_cluster_startup_probe(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    scheduling_index = kustomization["components"].index("components/scheduling/agg")
    kustomization["components"].insert(
        scheduling_index,
        "components/probes/agg",
    )
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))
    probe_operations = yaml.safe_load(
        (case / "components" / "probes" / "agg" / "patch-dgd.yaml").read_text()
    )
    startup_probe = next(
        operation["value"]
        for operation in probe_operations
        if operation["op"] == "add" and operation["path"].endswith("/startupProbe")
    )
    assert startup_probe["failureThreshold"] == 720

    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_dual_startup_probe_ownership(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"][1]["podTemplate"]["spec"]["containers"][0][
        "startupProbe"
    ] = {
        "httpGet": {"path": "/live", "port": "system"},
        "periodSeconds": 10,
        "timeoutSeconds": 5,
        "failureThreshold": 600,
    }
    _write_documents(base_path, documents)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    scheduling_index = kustomization["components"].index("components/scheduling/agg")
    kustomization["components"].insert(
        scheduling_index,
        "components/probes/agg",
    )
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "base-field-ownership")


@pytest.mark.parametrize(("relative_base", "hook_patch"), DISAGG_CASES)
def test_validator_accepts_disaggregated_templates_and_hook_patches(
    tmp_path: Path,
    relative_base: str,
    hook_patch: Optional[str],
) -> None:
    case = _filled_disagg_case(
        tmp_path,
        relative_base,
        hook_patch,
    )

    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_missing_sort_options(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    del kustomization["sortOptions"]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    result = _validate(case)

    _assert_error(result, "component-order")
    assert "expected={'order': 'fifo'}" in result.stderr


@pytest.mark.parametrize(
    "sort_options",
    (
        pytest.param({"order": "legacy"}, id="wrong-order"),
        pytest.param("fifo", id="non-mapping"),
        pytest.param({"order": "fifo", "extra": True}, id="extra-key"),
    ),
)
def test_validator_rejects_non_fifo_sort_options(
    tmp_path: Path,
    sort_options: object,
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["sortOptions"] = sort_options
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "component-order")


def test_validator_rejects_placement_without_scheduling(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"].remove("components/scheduling/agg")
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    result = _validate(case)

    _assert_error(result, "component-dependency")
    assert "scheduling" in result.stderr
    assert "placement" in result.stderr


def test_validator_rejects_placement_before_scheduling(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    selected = kustomization["components"]
    scheduling = selected.index("components/scheduling/agg")
    placement = selected.index("components/placement/agg")
    selected[scheduling], selected[placement] = (
        selected[placement],
        selected[scheduling],
    )
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    result = _validate(case)

    _assert_error(result, "component-order")
    assert "scheduling" in result.stderr
    assert "placement" in result.stderr


def test_validator_rejects_placement_when_scheduling_affinity_parent_missing(
    tmp_path: Path,
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    patch_path = case / "components" / "scheduling" / "agg" / "patch-dgd.yaml"
    operations = yaml.safe_load(patch_path.read_text())
    affinity_path = "/spec/components/1/podTemplate/spec/affinity"
    affinity_add = next(
        operation
        for operation in operations
        if operation["op"] == "add" and operation["path"] == affinity_path
    )
    operations.remove(affinity_add)
    patch_path.write_text(yaml.safe_dump(operations, sort_keys=False))

    result = _validate(case)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "ERROR [component-dependency]" in result.stderr
    assert "scheduling" in result.stderr
    assert "placement" in result.stderr
    assert "ERROR [replay-path]" not in result.stderr


def test_validator_rejects_unguarded_init_container_mutation(
    tmp_path: Path,
) -> None:
    operations = _component_guards(1, "Worker", "worker") + [
        {
            "op": "add",
            "path": ("/spec/components/1/podTemplate/spec/initContainers/0/command"),
            "value": ["sh", "-c", "prepare"],
        }
    ]
    case, _ = _neutral_component_case(tmp_path, operations)
    _seed_worker_container_collection(
        case,
        "initContainers",
        [{"name": "model-prepare", "image": "example.invalid/prepare:latest"}],
    )

    _assert_error(_validate(case), "patch-guard")


def test_validator_rejects_unguarded_ephemeral_container_mutation(
    tmp_path: Path,
) -> None:
    operations = _component_guards(1, "Worker", "worker") + [
        {
            "op": "add",
            "path": (
                "/spec/components/1/podTemplate/spec/" "ephemeralContainers/0/command"
            ),
            "value": ["sh"],
        }
    ]
    case, _ = _neutral_component_case(tmp_path, operations)
    _seed_worker_container_collection(
        case,
        "ephemeralContainers",
        [{"name": "debugger", "image": "example.invalid/debugger:latest"}],
    )

    _assert_error(_validate(case), "patch-guard")


def test_validator_accepts_guarded_init_container_mutation(tmp_path: Path) -> None:
    init_name_path = "/spec/components/1/podTemplate/spec/initContainers/0/name"
    operations = _component_guards(1, "Worker", "worker") + [
        {"op": "test", "path": init_name_path, "value": "model-prepare"},
        {
            "op": "add",
            "path": ("/spec/components/1/podTemplate/spec/initContainers/0/command"),
            "value": ["sh", "-c", "prepare"],
        },
    ]
    case, _ = _neutral_component_case(tmp_path, operations)
    _seed_worker_container_collection(
        case,
        "initContainers",
        [{"name": "model-prepare", "image": "example.invalid/prepare:latest"}],
    )

    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""


def test_validator_accepts_whole_ephemeral_container_append(
    tmp_path: Path,
) -> None:
    operations = _component_guards(1, "Worker", "worker") + [
        {
            "op": "add",
            "path": ("/spec/components/1/podTemplate/spec/ephemeralContainers/-"),
            "value": {
                "name": "debugger",
                "image": "example.invalid/debugger:latest",
                "command": ["sh"],
            },
        }
    ]
    case, _ = _neutral_component_case(tmp_path, operations)
    _seed_worker_container_collection(case, "ephemeralContainers", [])

    result = _validate(case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("rename", "canonical-components"),
        ("reorder", "canonical-components"),
        ("wrong-type", "canonical-components"),
        ("wrong-api", "beta-dgd-count"),
        ("zero-dgd", "beta-dgd-count"),
        ("two-dgds", "beta-dgd-count"),
        ("main-renamed", "kustomize-build"),
        ("main-moved", "kustomize-build"),
        ("duplicate-env", "base-env-ownership"),
        ("base-image-pull-secrets", "base-field-ownership"),
    ),
)
def test_validator_rejects_contract_breaks(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    components = dgd["spec"]["components"]

    if mutation == "rename":
        components[1]["name"] = "Engine"
        _write_documents(base_path, documents)
    elif mutation == "reorder":
        components[0], components[1] = components[1], components[0]
        _write_documents(base_path, documents)
    elif mutation == "wrong-type":
        components[1]["type"] = "decode"
        _write_documents(base_path, documents)
    elif mutation == "wrong-api":
        dgd["apiVersion"] = "nvidia.com/v1alpha1"
        _write_documents(base_path, documents)
    elif mutation == "zero-dgd":
        dgd["kind"] = "ConfigMap"
        _write_documents(base_path, documents)
    elif mutation == "two-dgds":
        duplicate = copy.deepcopy(dgd)
        duplicate["metadata"]["name"] = "second-dgd"
        documents.append(duplicate)
        _write_documents(base_path, documents)
    elif mutation == "main-renamed":
        components[1]["podTemplate"]["spec"]["containers"][0]["name"] = "engine"
        _write_documents(base_path, documents)
    elif mutation == "main-moved":
        components[1]["podTemplate"]["spec"]["containers"].insert(
            0,
            {"name": "sidecar", "image": "example.invalid/sidecar"},
        )
        _write_documents(base_path, documents)
    elif mutation == "duplicate-env":
        components[1]["podTemplate"]["spec"]["containers"][0]["env"].append(
            {"name": "NCCL_SOCKET_IFNAME", "value": "ens2f0"}
        )
        _write_documents(base_path, documents)
    elif mutation == "base-image-pull-secrets":
        components[0]["podTemplate"]["spec"]["imagePullSecrets"] = [
            {"name": "already-owned"}
        ]
        _write_documents(base_path, documents)
    else:
        raise AssertionError(mutation)

    _assert_error(_validate(case), expected_code)


def test_validator_rejects_alpha_patch_target(tmp_path: Path) -> None:
    operations = _identity_guards(1, "Worker", "worker") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "CASE_LOCAL_FLAG", "value": "enabled"},
        }
    ]
    case, component = _neutral_component_case(tmp_path, operations)
    kustomization_path = component / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"][0]["target"]["version"] = "v1alpha1"
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "unsupported-manifest")


def test_validator_rejects_unguarded_container_mutation(
    tmp_path: Path,
) -> None:
    case, _ = _neutral_component_case(
        tmp_path,
        _component_guards(1, "Worker", "worker")
        + [
            {
                "op": "add",
                "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
                "value": {"name": "CASE_LOCAL_FLAG", "value": "enabled"},
            }
        ],
    )

    _assert_error(_validate(case), "patch-guard")


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("rename", "canonical-components"),
        ("reorder", "canonical-components"),
        ("wrong-type", "canonical-components"),
        ("main-renamed", "kustomize-build"),
        ("main-moved", "kustomize-build"),
    ),
)
def test_validator_rejects_disaggregated_identity_breaks(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    case = _filled_disagg_case(tmp_path, *DISAGG_CASES[3])
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    components = dgd["spec"]["components"]
    if mutation == "rename":
        components[1]["name"] = "TrtllmPrefillWorker"
    elif mutation == "reorder":
        components[1], components[2] = components[2], components[1]
    elif mutation == "wrong-type":
        components[1]["type"] = "decode"
    elif mutation == "main-renamed":
        components[1]["podTemplate"]["spec"]["containers"][0]["name"] = "engine"
    elif mutation == "main-moved":
        components[1]["podTemplate"]["spec"]["containers"].insert(
            0,
            {"name": "sidecar", "image": "example.invalid/sidecar"},
        )
    else:
        raise AssertionError(mutation)
    _write_documents(base_path, documents)

    _assert_error(_validate(case), expected_code)


def test_validator_rejects_wrong_framework_hook_old_value(tmp_path: Path) -> None:
    case = _filled_disagg_case(tmp_path, *DISAGG_CASES[0])
    patch_path = case / "patches" / "vllm-kv-transfer-config.yaml"
    operations = yaml.safe_load(patch_path.read_text())
    hook_test = next(
        operation
        for operation in operations
        if operation["op"] == "test" and operation["path"].endswith("/env/0/value")
    )
    hook_test["value"] = "wrong-old-hook-value"
    patch_path.write_text(yaml.safe_dump(operations, sort_keys=False))

    _assert_error(_validate(case), "kustomize-build")


def test_validator_rejects_duplicate_env_added_by_later_layer(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    duplicate_patch = [
        {"op": "test", "path": "/spec/components/1/name", "value": "Worker"},
        {"op": "test", "path": "/spec/components/1/type", "value": "worker"},
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/name",
            "value": "main",
        },
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_SOCKET_IFNAME", "value": "ens2f0"},
        },
    ]
    patch_path = case / "patches" / "duplicate-env.yaml"
    patch_path.write_text(yaml.safe_dump(duplicate_patch, sort_keys=False))
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/duplicate-env.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "replay-duplicate-env")


@pytest.mark.parametrize(
    "mutation_path",
    (
        "/spec",
        "/spec/components",
        "/spec/components/1",
        "/spec/components/1/podTemplate",
        "/spec/components/1/podTemplate/spec",
    ),
)
def test_validator_rejects_mutations_above_guarded_component_structure(
    tmp_path: Path, mutation_path: str
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    dgd = _dgd(case / "base.yaml")
    tokens = mutation_path.lstrip("/").split("/")
    old_value: Any = dgd
    for token in tokens:
        old_value = (
            old_value[int(token)] if isinstance(old_value, list) else old_value[token]
        )
    new_value = yaml.safe_load(yaml.safe_dump(old_value))
    if mutation_path == "/spec":
        new_value["components"][1]["replicas"] = 2
    elif mutation_path == "/spec/components":
        new_value[1]["replicas"] = 2
    elif mutation_path == "/spec/components/1":
        new_value["replicas"] = 2
    elif mutation_path == "/spec/components/1/podTemplate":
        new_value["spec"]["containers"][0]["image"] = "example.invalid/replacement"
    else:
        new_value["containers"][0]["image"] = "example.invalid/replacement"

    patch = [
        {"op": "test", "path": "/spec/components/1/name", "value": "Worker"},
        {"op": "test", "path": "/spec/components/1/type", "value": "worker"},
        {"op": "test", "path": mutation_path, "value": old_value},
        {"op": "replace", "path": mutation_path, "value": new_value},
    ]
    patch_path = case / "patches" / "structural-ancestor.yaml"
    patch_path.write_text(yaml.safe_dump(patch, sort_keys=False))
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"] = []
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/structural-ancestor.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "patch-guard")


def test_validator_rejects_guard_invalidated_by_descendant_mutation(
    tmp_path: Path,
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    worker_env = _dgd(case / "base.yaml")["spec"]["components"][1]["podTemplate"][
        "spec"
    ]["containers"][0]["env"]
    old_entry = worker_env[0]
    old_value = old_entry["value"]
    patch = [
        {"op": "test", "path": "/spec/components/1/name", "value": "Worker"},
        {"op": "test", "path": "/spec/components/1/type", "value": "worker"},
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/name",
            "value": "main",
        },
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/0",
            "value": old_entry,
        },
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/0/value",
            "value": old_value,
        },
        {
            "op": "replace",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/0/value",
            "value": "intermediate-value",
        },
        {
            "op": "replace",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/0",
            "value": {"name": old_entry["name"], "value": "final-value"},
        },
    ]
    patch_path = case / "patches" / "stale-guard.yaml"
    patch_path.write_text(yaml.safe_dump(patch, sort_keys=False))
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/stale-guard.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "patch-guard")


def test_validator_rejects_guard_invalidated_by_list_insertion(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    worker_env = _dgd(case / "base.yaml")["spec"]["components"][1]["podTemplate"][
        "spec"
    ]["containers"][0]["env"]
    old_value = worker_env[1]["value"]
    patch = [
        {"op": "test", "path": "/spec/components/1/name", "value": "Worker"},
        {"op": "test", "path": "/spec/components/1/type", "value": "worker"},
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/name",
            "value": "main",
        },
        {
            "op": "test",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/1/value",
            "value": old_value,
        },
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/0",
            "value": {"name": "INSERTED", "value": "true"},
        },
        {
            "op": "replace",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/1/value",
            "value": "stale-guard-replacement",
        },
    ]
    patch_path = case / "patches" / "shifted-list-guard.yaml"
    patch_path.write_text(yaml.safe_dump(patch, sort_keys=False))
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/shifted-list-guard.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(_validate(case), "patch-guard")


@pytest.mark.parametrize(
    ("selector", "value"),
    (
        ("name", "dgd"),
        ("name", ".*"),
        ("namespace", "default"),
        ("labelSelector", "app.kubernetes.io/name=dgd"),
        ("annotationSelector", "example.com/network=rdma"),
    ),
)
def test_validator_rejects_target_selector(
    tmp_path: Path,
    selector: str,
    value: str,
) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    component_path = (
        case / "components" / "cache-binding" / "agg" / "kustomization.yaml"
    )
    component = yaml.safe_load(component_path.read_text())
    component["patches"][0]["target"][selector] = value
    component_path.write_text(yaml.safe_dump(component, sort_keys=False))

    _assert_error(_validate(case), "unsupported-manifest")


def test_validator_preserves_failed_guard_expectation(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"][1]["podTemplate"]["spec"]["volumes"][0][
        "persistentVolumeClaim"
    ]["claimName"] = "unexpected-cache"
    _write_documents(base_path, documents)

    result = _validate(case)

    _assert_error(result, "kustomize-build")
    assert "expected='shared-model-cache'" in result.stderr
    assert "actual='unexpected-cache'" in result.stderr


def test_validator_rejects_base_owned_device_selector(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = _documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"][1]["podTemplate"]["spec"]["containers"][0]["env"].append(
        {"name": "UCX_NET_DEVICES", "value": "mlx5_0:1"}
    )
    _write_documents(base_path, documents)

    _assert_error(_validate(case), "base-env-ownership")


def test_validator_rejects_unhashable_operation_name(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    component_path = case / "components" / "cache-binding" / "agg" / "patch-dgd.yaml"
    operations = yaml.safe_load(component_path.read_text())
    operations[0]["op"] = {"not": "a string"}
    component_path.write_text(yaml.safe_dump(operations, sort_keys=False))

    result = _validate(case)

    _assert_error(result, "unsupported-manifest")
    assert "Traceback" not in result.stderr


def test_validator_rejects_prerelease_kustomize_version(tmp_path: Path) -> None:
    case = _filled_case(tmp_path, with_case_override=False)
    executable = tmp_path / "kustomize-prerelease"
    executable.write_text("#!/bin/sh\necho v5.8.1-rc.1\n")
    executable.chmod(0o755)

    _assert_error(
        _validate(case, kustomize_bin=executable),
        "kustomize-version",
    )


def test_readme_covers_binding_and_safety_contract() -> None:
    readme = (SCAFFOLD / "README.md").read_text()

    for required in (
        "Kustomize v5.8.1",
        "cache-binding",
        "registry-credentials",
        "probes",
        "scheduling",
        "network-interface",
        "placement",
        "your-worker-startup-failure-threshold",
        "your-storage-class-name",
        "never apply",
        "component-order",
        "component-dependency",
        "Kubernetes 1.33+",
        "MatchLabelKeysInPodAffinity",
        "leader/main worker container",
        "multinode follower probes",
        "UCX_NET_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "networking allowlist",
        "documented design decision",
        "KV_TRANSFER_CONFIG",
        "SGLANG_DISAGGREGATION_NIXL_BACKEND",
        "LoadRestrictionsNone",
        "~1",
        "scripts/validate-recipe-kustomization.py",
        "--kustomize-bin /explicit/path/to/kustomize",
        "deployment operation",
        "not an automatic substitute",
    ):
        assert required in readme

    placeholder_preflight = """if grep -R -n -E --include='*.yaml' \\
  'your-[[:alnum:].~/_-]+' kustomization.yaml components patches
then
  echo 'ERROR: unresolved recipe placeholders remain' >&2
  exit 1
else
  placeholder_scan_status=$?
  if [ "$placeholder_scan_status" -ne 1 ]; then
    exit "$placeholder_scan_status"
  fi
fi"""
    assert placeholder_preflight in readme

    ordered_components = (
        "1. `cache-binding`",
        "2. `registry-credentials`",
        "3. `probes`, when required",
        "4. `scheduling`",
        "5. `network-interface`",
        "6. `placement`, when required",
    )
    component_positions = [readme.index(item) for item in ordered_components]
    assert component_positions == sorted(component_positions)

    assert "rg -n --glob '*.yaml'" not in readme
    assert '--kustomize-bin "$(command -v kustomize)"' not in readme
