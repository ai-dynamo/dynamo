# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytest
import test_recipe_kustomize_validator as core
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(30),
]

SCAFFOLD = core.SCAFFOLD
TEMPLATE_ROOT = core.REPO_ROOT / "recipes" / "templates"
PROVIDER_BASES = (
    core.DISAGG_CASES[0],
    core.DISAGG_CASES[2],
    core.DISAGG_CASES[3],
)
PROVIDER_VALUES = {
    "gke-roce": {
        **{f"your-{role}-default-interface": "eth0" for role in ("prefill", "decode")},
        **{
            f"your-{role}-rdma-interface-{slot}": f"{role}-rdma{slot}"
            for role in ("prefill", "decode")
            for slot in range(4)
        },
        **{
            f"your-{role}-network-{slot}": f"{role}-rdma-{slot}"
            for role in ("prefill", "decode")
            for slot in range(4)
        },
        "your-prefill-nccl-socket-interface": "eth0",
        "your-prefill-gloo-socket-interface": "eth0",
        "your-prefill-cross-nic-value": "0",
        "your-prefill-network-resource-quantity": "2",
        "your-decode-nccl-socket-interface": "eth0",
        "your-decode-gloo-socket-interface": "eth0",
        "your-decode-cross-nic-value": "1",
        "your-decode-network-resource-quantity": "3",
    },
    "ib": {
        "your-prefill-rdma.example.com/resource": "example.com/prefill-rdma",
        "your-prefill-rdma-resource-quantity": "2",
        "your-prefill-nccl-socket-interface": "eth0",
        "your-prefill-gloo-socket-interface": "eth0",
        "your-prefill-ucx-device-list": "mlx5_0:1,mlx5_1:1",
        "your-decode-rdma.example.com/resource": "example.com/decode-rdma",
        "your-decode-rdma-resource-quantity": "3",
        "your-decode-nccl-socket-interface": "ens1f0",
        "your-decode-gloo-socket-interface": "ens1f0",
        "your-decode-ucx-device-list": "mlx5_2:1,mlx5_3:1",
    },
}
CANONICAL_BETA_DISAGG_CASES = (
    "vllm/disagg/deploy-v1beta1.template.yaml",
    "vllm/disagg/deploy-v1beta1-compute-domain.template.yaml",
    "sglang/disagg/deploy-v1beta1.template.yaml",
    "trtllm/disagg/deploy-v1beta1.template.yaml",
)


def _replace_tokens(root: Path, replacements: dict[str, str]) -> None:
    for path in root.rglob("*.yaml"):
        text = path.read_text()
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        path.write_text(text)


def _select_networking_component(
    case: Path,
    reference: str,
    *,
    keep_generic: bool = False,
) -> None:
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    generic = "components/network-interface/disagg"
    generic_index = kustomization["components"].index(generic)
    if keep_generic:
        kustomization["components"].insert(generic_index + 1, reference)
    else:
        kustomization["components"][generic_index] = reference
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))


def _remove_networking_component(case: Path) -> None:
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"] = [
        reference
        for reference in kustomization["components"]
        if reference != "components/network-interface/disagg"
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))


def _filled_provider_case(
    tmp_path: Path,
    provider: str,
    relative_base: str,
    hook_patch: Optional[str],
) -> Path:
    case = core._filled_disagg_case(tmp_path, relative_base, hook_patch)
    reference = f"components/provider-networking/{provider}/disagg"
    destination = case / reference
    assert destination.is_dir(), f"provider source was not copied: {destination}"
    _replace_tokens(destination, PROVIDER_VALUES[provider])
    unresolved = {
        path.relative_to(case).as_posix()
        for path in destination.rglob("*.yaml")
        if "your-" in path.read_text()
    }
    assert unresolved == set()
    _select_networking_component(case, reference)
    return case


def _remove_optional_provider_blocks(case: Path, provider: str) -> None:
    patch_path = (
        case
        / "components"
        / "provider-networking"
        / provider
        / "disagg"
        / "patch-dgd.yaml"
    )
    document = yaml.safe_load(patch_path.read_text())
    for component in document["spec"]["components"]:
        pod_template = component.get("podTemplate")
        if pod_template is None:
            continue
        pod_spec = pod_template["spec"]
        main = pod_spec["containers"][0]
        if provider == "gke-roce":
            main["env"] = [
                entry for entry in main["env"] if entry["name"] != "NCCL_CROSS_NIC"
            ]
            for scope in ("requests", "limits"):
                main["resources"][scope] = {
                    key: value
                    for key, value in main["resources"][scope].items()
                    if not key.endswith(".IP")
                }
        else:
            main.pop("env", None)
            main.pop("volumeMounts", None)
            pod_spec.pop("volumes", None)
    patch_path.write_text(yaml.safe_dump(document, sort_keys=False))


def _rendered_dgd(case: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            core._kustomize_bin(),
            "build",
            str(case),
            "--load-restrictor",
            "LoadRestrictionsNone",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return next(
        document
        for document in yaml.safe_load_all(completed.stdout)
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    )


def _components(dgd: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {component["name"]: component for component in dgd["spec"]["components"]}


def _added_entries(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the keyed list entries that a merge patch added, in render order."""

    names = {entry.get("name") for entry in before}
    added = [entry for entry in after if entry.get("name") not in names]
    assert [entry for entry in after if entry.get("name") in names] == before
    return added


def _added_mapping(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    assert all(after.get(key) == value for key, value in before.items())
    return {key: value for key, value in after.items() if key not in before}


def _network_projection(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_pod = before["podTemplate"]
    after_pod = after["podTemplate"]
    before_spec = before_pod["spec"]
    after_spec = after_pod["spec"]
    before_main = before_spec["containers"][0]
    after_main = after_spec["containers"][0]
    before_resources = before_main["resources"]
    after_resources = after_main["resources"]
    before_mounts = before_main.get("volumeMounts", [])
    before_volumes = before_spec.get("volumes", [])
    return {
        "annotations": after_pod.get("metadata", {}).get("annotations", {}),
        "environment": _added_entries(before_main["env"], after_main["env"]),
        "requests": _added_mapping(
            before_resources["requests"], after_resources["requests"]
        ),
        "limits": _added_mapping(before_resources["limits"], after_resources["limits"]),
        "volumeMounts": _added_entries(
            before_mounts, after_main.get("volumeMounts", [])
        ),
        "volumes": _added_entries(before_volumes, after_spec.get("volumes", [])),
    }


def _expected_gke_annotations(role: str) -> dict[str, str]:
    interfaces = [
        {"interfaceName": "eth0", "network": "default"},
        *[
            {
                "interfaceName": f"{role}-rdma{slot}",
                "network": f"{role}-rdma-{slot}",
            }
            for slot in range(4)
        ],
    ]
    return {
        "networking.gke.io/default-interface": "eth0",
        "networking.gke.io/interfaces": json.dumps(interfaces, separators=(",", ":")),
    }


def _assert_provider_worker_delta(
    provider: str,
    role: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    projection = _network_projection(before, after)
    requests = projection["requests"]
    assert requests == projection["limits"]
    assert all("/" in key and "~1" not in key for key in requests)

    if provider == "gke-roce":
        expected_annotations = _expected_gke_annotations(role)
        assert (
            projection["annotations"]["networking.gke.io/default-interface"] == "eth0"
        )
        assert json.loads(
            projection["annotations"]["networking.gke.io/interfaces"]
        ) == json.loads(expected_annotations["networking.gke.io/interfaces"])
        assert set(projection["annotations"]) == set(expected_annotations)
        assert projection["environment"] == [
            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
            {"name": "GLOO_SOCKET_IFNAME", "value": "eth0"},
            {
                "name": "NCCL_CROSS_NIC",
                "value": "0" if role == "prefill" else "1",
            },
        ]
        quantity = "2" if role == "prefill" else "3"
        assert requests == {
            key: quantity
            for slot in range(4)
            for key in (
                f"networking.gke.io.networks/{role}-rdma-{slot}",
                f"networking.gke.io.networks/{role}-rdma-{slot}.IP",
            )
        }
        assert projection["volumeMounts"] == []
        assert projection["volumes"] == []
    else:
        socket_interface = "eth0" if role == "prefill" else "ens1f0"
        devices = "mlx5_0:1,mlx5_1:1" if role == "prefill" else "mlx5_2:1,mlx5_3:1"
        assert projection["annotations"] == {}
        assert projection["environment"] == [
            {"name": "NCCL_SOCKET_IFNAME", "value": socket_interface},
            {"name": "GLOO_SOCKET_IFNAME", "value": socket_interface},
            {"name": "UCX_NET_DEVICES", "value": devices},
        ]
        quantity = "2" if role == "prefill" else "3"
        assert requests == {f"example.com/{role}-rdma": quantity}
        assert projection["volumeMounts"] == [
            {"name": "ib", "mountPath": "/dev/infiniband"}
        ]
        assert projection["volumes"] == [
            {"name": "ib", "hostPath": {"path": "/dev/infiniband"}}
        ]

    scrubbed = copy.deepcopy(after)
    before_pod = before["podTemplate"]
    scrubbed_pod = scrubbed["podTemplate"]
    scrubbed_pod["metadata"] = copy.deepcopy(before_pod.get("metadata", {}))
    before_spec = before_pod["spec"]
    scrubbed_spec = scrubbed_pod["spec"]
    before_main = before_spec["containers"][0]
    scrubbed_main = scrubbed_spec["containers"][0]
    scrubbed_main["env"] = copy.deepcopy(before_main["env"])
    scrubbed_main["resources"] = copy.deepcopy(before_main["resources"])
    scrubbed_main["volumeMounts"] = copy.deepcopy(before_main.get("volumeMounts", []))
    scrubbed_spec["volumes"] = copy.deepcopy(before_spec.get("volumes", []))
    assert scrubbed == before


def _identity_guards(
    index: int, name: str, component_type: str
) -> list[dict[str, Any]]:
    return [
        {"op": "test", "path": f"/spec/components/{index}/name", "value": name},
        {
            "op": "test",
            "path": f"/spec/components/{index}/type",
            "value": component_type,
        },
        {
            "op": "test",
            "path": f"/spec/components/{index}/podTemplate/spec/containers/0/name",
            "value": "main",
        },
    ]


def _write_component(
    component: Path,
    operations: list[dict[str, Any]],
    *,
    nested: Optional[list[str]] = None,
) -> None:
    component.mkdir(parents=True, exist_ok=True)
    kustomization: dict[str, Any] = {
        "apiVersion": "kustomize.config.k8s.io/v1alpha1",
        "kind": "Component",
    }
    if nested:
        kustomization["components"] = nested
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


def _custom_networking_case(
    tmp_path: Path,
    operations: list[dict[str, Any]],
    *,
    keep_generic: bool = False,
) -> tuple[Path, Path]:
    case = core._filled_disagg_case(
        tmp_path,
        "trtllm/disagg/deploy-v1beta1.template.yaml",
        None,
    )
    reference = "components/provider-networking/test/disagg"
    component = case / reference
    _write_component(component, operations)
    _select_networking_component(case, reference, keep_generic=keep_generic)
    return case, component


def _assert_error(result: subprocess.CompletedProcess[str], code: str) -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"ERROR [{code}]" in result.stderr, result.stdout + result.stderr


def test_provider_source_inventory_is_complete() -> None:
    expected = {
        f"components/provider-networking/{provider}/disagg/{filename}"
        for provider in ("gke-roce", "ib")
        for filename in ("kustomization.yaml", "patch-dgd.yaml")
    }
    actual = {
        path.relative_to(SCAFFOLD).as_posix()
        for path in (SCAFFOLD / "components" / "provider-networking").rglob("*")
        if path.is_file()
    }
    assert actual == expected


@pytest.mark.parametrize("relative_path", CANONICAL_BETA_DISAGG_CASES)
def test_canonical_disaggregated_workers_have_annotation_anchors(
    relative_path: str,
) -> None:
    documents = tuple(yaml.safe_load_all((TEMPLATE_ROOT / relative_path).read_text()))
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    for component in dgd["spec"]["components"]:
        annotations = component["podTemplate"].get("metadata", {}).get("annotations")
        if component["name"] in {"PrefillWorker", "DecodeWorker"}:
            assert annotations == {}
        else:
            assert annotations is None


@pytest.mark.parametrize("provider", tuple(PROVIDER_VALUES))
@pytest.mark.parametrize(("relative_base", "hook_patch"), PROVIDER_BASES)
def test_provider_networking_is_backend_neutral_and_worker_only(
    tmp_path: Path,
    provider: str,
    relative_base: str,
    hook_patch: Optional[str],
) -> None:
    baseline = core._filled_disagg_case(
        tmp_path / "baseline", relative_base, hook_patch
    )
    _remove_networking_component(baseline)
    before = _rendered_dgd(baseline)
    case = _filled_provider_case(
        tmp_path / "provider", provider, relative_base, hook_patch
    )

    result = core._validate(case)
    after = _rendered_dgd(case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    before_components = _components(before)
    after_components = _components(after)
    assert after_components["Frontend"] == before_components["Frontend"]
    for role, name in (("prefill", "PrefillWorker"), ("decode", "DecodeWorker")):
        _assert_provider_worker_delta(
            provider, role, before_components[name], after_components[name]
        )


@pytest.mark.parametrize("provider", tuple(PROVIDER_VALUES))
def test_provider_optional_blocks_can_be_removed_as_complete_units(
    tmp_path: Path,
    provider: str,
) -> None:
    case = _filled_provider_case(tmp_path, provider, *PROVIDER_BASES[2])
    _remove_optional_provider_blocks(case, provider)

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    for role, name in (("prefill", "PrefillWorker"), ("decode", "DecodeWorker")):
        worker = rendered[name]
        main = worker["podTemplate"]["spec"]["containers"][0]
        if provider == "gke-roce":
            added = {
                key
                for key in main["resources"]["requests"]
                if key.startswith("networking.gke.io.networks/")
            }
            assert added == {
                f"networking.gke.io.networks/{role}-rdma-{slot}" for slot in range(4)
            }
            assert all(not key.endswith(".IP") for key in added)
            assert all(entry["name"] != "NCCL_CROSS_NIC" for entry in main["env"])
        else:
            cluster_env = {
                "NCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "UCX_NET_DEVICES",
            }
            assert all(entry["name"] not in cluster_env for entry in main["env"])
            assert all(
                mount.get("mountPath") != "/dev/infiniband"
                for mount in main.get("volumeMounts", [])
            )
            assert all(
                volume.get("hostPath", {}).get("path") != "/dev/infiniband"
                for volume in worker["podTemplate"]["spec"].get("volumes", [])
            )


@pytest.mark.parametrize("provider", tuple(PROVIDER_VALUES))
def test_provider_network_delta_is_replica_count_independent(
    tmp_path: Path,
    provider: str,
) -> None:
    projections: list[dict[str, dict[str, Any]]] = []
    observed_replicas: list[int] = []
    for label, decode_replicas in (("256k", 2), ("1m", 1)):
        baseline = core._filled_disagg_case(
            tmp_path / f"{label}-baseline", *PROVIDER_BASES[0]
        )
        _remove_networking_component(baseline)
        provider_case = _filled_provider_case(
            tmp_path / f"{label}-provider", provider, *PROVIDER_BASES[0]
        )
        for case in (baseline, provider_case):
            base_path = case / "base.yaml"
            documents = core._documents(base_path)
            dgd = next(
                document
                for document in documents
                if document.get("kind") == "DynamoGraphDeployment"
            )
            dgd["spec"]["components"][2]["replicas"] = decode_replicas
            core._write_documents(base_path, documents)
        before = _components(_rendered_dgd(baseline))
        after = _components(_rendered_dgd(provider_case))
        observed_replicas.append(after["DecodeWorker"]["replicas"])
        projections.append(
            {
                role: _network_projection(before[name], after[name])
                for role, name in (
                    ("prefill", "PrefillWorker"),
                    ("decode", "DecodeWorker"),
                )
            }
        )

    assert observed_replicas == [2, 1]
    assert projections[0] == projections[1]


def test_provider_source_placeholder_vocabulary_is_exact() -> None:
    for provider, expected_values in PROVIDER_VALUES.items():
        source = SCAFFOLD / "components" / "provider-networking" / provider / "disagg"
        patch_path = source / "patch-dgd.yaml"
        tokens = {
            token.removesuffix(".IP")
            for token in re.findall(r"your-[A-Za-z0-9_.~/-]+", patch_path.read_text())
        }
        assert tokens == set(expected_values)
        assert "your-" not in (source / "kustomization.yaml").read_text()


def test_validator_rejects_unguarded_networking_container_mutation(
    tmp_path: Path,
) -> None:
    case, _ = _custom_networking_case(
        tmp_path,
        [
            {
                "op": "add",
                "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
                "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
            }
        ],
    )
    _assert_error(core._validate(case), "patch-guard")


@pytest.mark.parametrize("position", ("before-scheduling", "after-placement"))
def test_validator_rejects_networking_outside_the_concern_order(
    tmp_path: Path,
    position: str,
) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    selected = kustomization["components"]
    provider = "components/provider-networking/test/disagg"
    selected.remove(provider)
    if position == "before-scheduling":
        selected.insert(selected.index("components/scheduling/disagg"), provider)
    else:
        selected.insert(selected.index("components/placement/disagg") + 1, provider)
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(core._validate(case), "networking-slot")


def test_validator_rejects_generic_and_provider_networking_slots(
    tmp_path: Path,
) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_SOCKET_IFNAME", "value": "ens1f0"},
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations, keep_generic=True)
    _assert_error(core._validate(case), "networking-slot")


def test_validator_accepts_case_without_networking_component(tmp_path: Path) -> None:
    case = core._filled_disagg_case(
        tmp_path,
        "trtllm/disagg/deploy-v1beta1.template.yaml",
        None,
    )
    _remove_networking_component(case)

    result = core._validate(case)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_network_delta_hidden_under_scheduling_root(
    tmp_path: Path,
) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    provider = "components/provider-networking/test/disagg"
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"].remove(provider)
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))
    scheduling_path = (
        case / "components" / "scheduling" / "disagg" / "kustomization.yaml"
    )
    scheduling = yaml.safe_load(scheduling_path.read_text())
    scheduling["components"] = ["../../provider-networking/test/disagg"]
    scheduling_path.write_text(yaml.safe_dump(scheduling, sort_keys=False))

    _assert_error(core._validate(case), "networking-delta")


@pytest.mark.parametrize("root_kind", ("provider", "private"))
def test_validator_preserves_networking_root_identity_through_nested_components(
    tmp_path: Path,
    root_kind: str,
) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
        }
    ]
    case, root = _custom_networking_case(tmp_path, operations)
    child = root / "mechanism"
    _write_component(child, operations)
    _write_component(root, [], nested=["mechanism"])
    if root_kind == "private":
        private_root = case / "components" / "networking" / "disagg"
        private_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(private_root))
        kustomization_path = case / "kustomization.yaml"
        kustomization = yaml.safe_load(kustomization_path.read_text())
        provider = "components/provider-networking/test/disagg"
        kustomization["components"][
            kustomization["components"].index(provider)
        ] = "components/networking/disagg"
        kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    result = core._validate(case)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_root_patch_as_networking_slot(tmp_path: Path) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/env/-",
            "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    provider = "components/provider-networking/test/disagg"
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"].remove(provider)
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": f"{provider}/patch-dgd.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(core._validate(case), "networking-delta")


def _forbidden_network_delta(name: str) -> list[dict[str, Any]]:
    guards = _identity_guards(1, "PrefillWorker", "prefill")
    if name == "frontend":
        return _identity_guards(0, "Frontend", "frontend") + [
            {
                "op": "add",
                "path": "/spec/components/0/podTemplate/spec/containers/0/env/-",
                "value": {"name": "NCCL_CROSS_NIC", "value": "0"},
            }
        ]
    deltas = {
        "scheduling": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/nodeSelector",
            "value": {"example.com/pool": "gpu"},
        },
        "cache": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/volumeMounts/-",
            "value": {"name": "other-cache", "mountPath": "/cache"},
        },
        "registry": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/imagePullSecrets",
            "value": [{"name": "registry-secret"}],
        },
        "probes": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/startupProbe",
            "value": {"exec": {"command": ["true"]}},
        },
        "namespace": {"op": "add", "path": "/metadata/namespace", "value": "default"},
        "shared-memory": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/volumes/-",
            "value": {
                "name": "dshm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
            },
        },
        "security": {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/containers/0/securityContext/privileged",
            "value": True,
        },
    }
    if name == "serving-command":
        path = "/spec/components/1/podTemplate/spec/containers/0/command"
        return guards + [
            {"op": "test", "path": path, "value": ["python3"]},
            {"op": "replace", "path": path, "value": ["sleep"]},
        ]
    return guards + [deltas[name]]


@pytest.mark.parametrize(
    "concern",
    (
        "frontend",
        "scheduling",
        "cache",
        "registry",
        "probes",
        "namespace",
        "shared-memory",
        "serving-command",
        "security",
    ),
)
def test_validator_rejects_forbidden_provider_concern_delta(
    tmp_path: Path,
    concern: str,
) -> None:
    case, _ = _custom_networking_case(tmp_path, _forbidden_network_delta(concern))
    _assert_error(core._validate(case), "networking-delta")


def test_validator_accepts_provider_defined_annotation_key(tmp_path: Path) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": (
                "/spec/components/1/podTemplate/metadata/annotations/"
                "k8s.v1.cni.cncf.io~1networks"
            ),
            "value": "rdma-net-0,rdma-net-1",
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations)

    result = core._validate(case)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_accepts_provider_defined_environment_and_devices(
    tmp_path: Path,
) -> None:
    main = "/spec/components/1/podTemplate/spec/containers/0"
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": f"{main}/env/-",
            "value": {"name": "FI_PROVIDER", "value": "efa"},
        },
        {
            "op": "add",
            "path": f"{main}/env/-",
            "value": {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "1"},
        },
        {
            "op": "add",
            "path": f"{main}/env/-",
            "value": {"name": "NCCL_IB_HCA", "value": "mlx5_0"},
        },
        {
            "op": "add",
            "path": f"{main}/resources/requests/vpc.amazonaws.com~1efa",
            "value": "1",
        },
        {
            "op": "add",
            "path": f"{main}/resources/limits/vpc.amazonaws.com~1efa",
            "value": "1",
        },
        {
            "op": "add",
            "path": f"{main}/volumeMounts/-",
            "value": {"name": "efa", "mountPath": "/dev/infiniband"},
        },
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/volumes/-",
            "value": {"name": "efa", "hostPath": {"path": "/dev/infiniband"}},
        },
    ]
    case, _ = _custom_networking_case(tmp_path, operations)

    result = core._validate(case)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_accepts_provider_defined_host_path_mount(tmp_path: Path) -> None:
    # GKE RoCE recipes expose the gIB NCCL plugin from a non-/dev host path.
    main = "/spec/components/1/podTemplate/spec/containers/0"
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": f"{main}/volumeMounts/-",
            "value": {"name": "gib", "mountPath": "/usr/local/gib"},
        },
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/volumes/-",
            "value": {"name": "gib", "hostPath": {"path": "/home/kubernetes/bin/gib"}},
        },
    ]
    case, _ = _custom_networking_case(tmp_path, operations)

    result = core._validate(case)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_unpaired_networking_host_volume(tmp_path: Path) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": "/spec/components/1/podTemplate/spec/volumes/-",
            "value": {"name": "ib", "hostPath": {"path": "/dev/infiniband"}},
        },
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    _assert_error(core._validate(case), "networking-delta")


def test_validator_rejects_whole_worker_annotation_map_replacement(
    tmp_path: Path,
) -> None:
    annotations_path = "/spec/components/1/podTemplate/metadata/annotations"
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {"op": "test", "path": annotations_path, "value": {}},
        {
            "op": "replace",
            "path": annotations_path,
            "value": {"networking.gke.io/default-interface": "eth0"},
        },
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    base_path = case / "base.yaml"
    documents = core._documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"][1]["podTemplate"]["metadata"] = {"annotations": {}}
    core._write_documents(base_path, documents)

    _assert_error(core._validate(case), "networking-delta")


@pytest.mark.parametrize("mismatch", ("key", "quantity"))
def test_validator_rejects_networking_resource_pair_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    request_key = "example.com~1prefill-rdma"
    limit_key = "example.com~1other-rdma" if mismatch == "key" else request_key
    limit_quantity = "2" if mismatch == "quantity" else "1"
    base_path = "/spec/components/1/podTemplate/spec/containers/0/resources"
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": f"{base_path}/requests/{request_key}",
            "value": "1",
        },
        {
            "op": "add",
            "path": f"{base_path}/limits/{limit_key}",
            "value": limit_quantity,
        },
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    _assert_error(core._validate(case), "networking-resource-pair")


def test_validator_rejects_unescaped_networking_resource_pointer(
    tmp_path: Path,
) -> None:
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {
            "op": "add",
            "path": (
                "/spec/components/1/podTemplate/spec/containers/0/resources/"
                "requests/example.com/rdma"
            ),
            "value": "1",
        }
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    _assert_error(core._validate(case), "networking-resource-pair")


def test_validator_rejects_base_owned_provider_annotation(tmp_path: Path) -> None:
    annotation_path = (
        "/spec/components/1/podTemplate/metadata/annotations/"
        "networking.gke.io~1default-interface"
    )
    operations = _identity_guards(1, "PrefillWorker", "prefill") + [
        {"op": "add", "path": annotation_path, "value": "eth0"}
    ]
    case, _ = _custom_networking_case(tmp_path, operations)
    base_path = case / "base.yaml"
    documents = core._documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"][1]["podTemplate"]["metadata"] = {
        "annotations": {"networking.gke.io/default-interface": "eth0"}
    }
    core._write_documents(base_path, documents)

    _assert_error(core._validate(case), "base-field-ownership")


def _merge_patch(components: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "canonical-recipe"},
        "spec": {"components": components},
    }


def _worker_env_patch(name: str, env: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "podTemplate": {"spec": {"containers": [{"name": "main", "env": env}]}},
    }


def _custom_merge_networking_case(
    tmp_path: Path, document: dict[str, Any]
) -> tuple[Path, Path]:
    case = core._filled_disagg_case(
        tmp_path,
        "trtllm/disagg/deploy-v1beta1.template.yaml",
        None,
    )
    reference = "components/provider-networking/test/disagg"
    component = case / reference
    component.mkdir(parents=True, exist_ok=True)
    (component / "kustomization.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "kustomize.config.k8s.io/v1alpha1",
                "kind": "Component",
                "patches": [
                    {
                        "target": {
                            "group": "nvidia.com",
                            "version": "v1beta1",
                            "kind": "DynamoGraphDeployment",
                        },
                        "path": "patch-dgd.yaml",
                    }
                ],
            },
            sort_keys=False,
        )
    )
    (component / "patch-dgd.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    _select_networking_component(case, reference)
    return case, component


_MERGE_NETWORK_ENV = [{"name": "FI_PROVIDER", "value": "efa"}]


def test_validator_accepts_merge_patch_networking_component(tmp_path: Path) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("PrefillWorker", _MERGE_NETWORK_ENV),
                _worker_env_patch("DecodeWorker", _MERGE_NETWORK_ENV),
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("PrefillWorker", "DecodeWorker"):
        env = rendered[name]["podTemplate"]["spec"]["containers"][0]["env"]
        assert env[0] == {"name": "FI_PROVIDER", "value": "efa"}
    rendered_components = _rendered_dgd(case)["spec"]["components"]
    assert [component["name"] for component in rendered_components] == [
        "Frontend",
        "PrefillWorker",
        "DecodeWorker",
    ]


def test_validator_accepts_merge_annotations_without_base_anchor(
    tmp_path: Path,
) -> None:
    annotations = {"k8s.v1.cni.cncf.io/networks": "rdma-net-0"}
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                {
                    "name": "PrefillWorker",
                    "podTemplate": {"metadata": {"annotations": annotations}},
                },
                {"name": "DecodeWorker"},
            ]
        ),
    )
    base_path = case / "base.yaml"
    documents = core._documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    for component in dgd["spec"]["components"]:
        component["podTemplate"].pop("metadata", None)
    core._write_documents(base_path, documents)

    result = core._validate(case)

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = _components(_rendered_dgd(case))
    assert rendered["PrefillWorker"]["podTemplate"]["metadata"] == {
        "annotations": annotations
    }
    assert "metadata" not in rendered["DecodeWorker"]["podTemplate"]


def test_validator_rejects_merge_patch_omitting_a_canonical_component(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch([_worker_env_patch("PrefillWorker", _MERGE_NETWORK_ENV)]),
    )
    _assert_error(core._validate(case), "merge-patch")


def test_validator_rejects_merge_patch_naming_unknown_component(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                {"name": "PrefillWorker"},
                {"name": "DecodeWorker"},
                _worker_env_patch("BogusWorker", _MERGE_NETWORK_ENV),
            ]
        ),
    )
    _assert_error(core._validate(case), "merge-patch")


def test_validator_rejects_merge_patch_directives(tmp_path: Path) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch(
                    "PrefillWorker", [{"name": "HF_HUB_OFFLINE", "$patch": "delete"}]
                ),
                {"name": "DecodeWorker"},
            ]
        ),
    )
    _assert_error(core._validate(case), "merge-patch")


def test_validator_rejects_merge_networking_override_of_base_env(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch(
                    "PrefillWorker", [{"name": "HF_HUB_OFFLINE", "value": "0"}]
                ),
                {"name": "DecodeWorker"},
            ]
        ),
    )
    # The base owns HF_HUB_OFFLINE, so ownership is reported as the root cause
    # before the networking contract sees the lowered override.
    _assert_error(core._validate(case), "base-env-ownership")


def test_validator_rejects_merge_networking_frontend_mutation(tmp_path: Path) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                _worker_env_patch("Frontend", _MERGE_NETWORK_ENV),
                {"name": "PrefillWorker"},
                {"name": "DecodeWorker"},
            ]
        ),
    )
    _assert_error(core._validate(case), "networking-delta")


def test_validator_rejects_merge_patch_without_schema_component(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("PrefillWorker", _MERGE_NETWORK_ENV),
                {"name": "DecodeWorker"},
            ]
        ),
    )
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["components"].remove("components/dynamo-openapi")
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))

    _assert_error(core._validate(case), "merge-patch")


def _append_optional_component(case: Path) -> None:
    base_path = case / "base.yaml"
    documents = core._documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"].append(
        {
            "name": "Planner",
            "type": "planner",
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {"name": "main", "image": "example.invalid/planner:latest"}
                    ]
                }
            },
        }
    )
    core._write_documents(base_path, documents)


def test_validator_accepts_merge_prefix_before_optional_component(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("PrefillWorker", _MERGE_NETWORK_ENV),
                _worker_env_patch("DecodeWorker", _MERGE_NETWORK_ENV),
            ]
        ),
    )
    _append_optional_component(case)

    result = core._validate(case)
    rendered = _rendered_dgd(case)

    assert result.returncode == 0, result.stdout + result.stderr
    assert [component["name"] for component in rendered["spec"]["components"]] == [
        "Frontend",
        "PrefillWorker",
        "DecodeWorker",
        "Planner",
    ]
    planner = _components(rendered)["Planner"]
    assert "env" not in planner["podTemplate"]["spec"]["containers"][0]


def test_validator_rejects_merge_patch_skipping_a_canonical_component(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("DecodeWorker", _MERGE_NETWORK_ENV),
            ]
        ),
    )
    _assert_error(core._validate(case), "merge-patch")


def test_validator_accepts_merge_env_value_from_source(tmp_path: Path) -> None:
    sourced = [
        {
            "name": "FI_PROVIDER",
            "valueFrom": {
                "configMapKeyRef": {"name": "efa-settings", "key": "provider"}
            },
        }
    ]
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("PrefillWorker", sourced),
                _worker_env_patch("DecodeWorker", sourced),
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("PrefillWorker", "DecodeWorker"):
        env = rendered[name]["podTemplate"]["spec"]["containers"][0]["env"]
        assert env[0] == sourced[0]


def _read_only_mount_component(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "volumeMounts": [
                            {
                                "name": "ib",
                                "mountPath": "/dev/infiniband",
                                "readOnly": True,
                            }
                        ],
                    }
                ],
                "volumes": [{"name": "ib", "hostPath": {"path": "/dev/infiniband"}}],
            }
        },
    }


def test_validator_accepts_merge_read_only_host_mount(tmp_path: Path) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _read_only_mount_component("PrefillWorker"),
                _read_only_mount_component("DecodeWorker"),
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("PrefillWorker", "DecodeWorker"):
        main = rendered[name]["podTemplate"]["spec"]["containers"][0]
        assert main["volumeMounts"][0] == {
            "name": "ib",
            "mountPath": "/dev/infiniband",
            "readOnly": True,
        }


def test_validator_rejects_merge_host_mount_with_unknown_field(
    tmp_path: Path,
) -> None:
    component = _read_only_mount_component("PrefillWorker")
    mount = component["podTemplate"]["spec"]["containers"][0]["volumeMounts"][0]
    mount["bogusField"] = "x"
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch([{"name": "Frontend"}, component, {"name": "DecodeWorker"}]),
    )
    _assert_error(core._validate(case), "networking-delta")


def _root_merge_patch_case(
    tmp_path: Path,
    document: dict[str, Any],
    *,
    keep_networking: bool,
    optional_component: bool = False,
) -> Path:
    case = core._filled_disagg_case(
        tmp_path,
        "trtllm/disagg/deploy-v1beta1.template.yaml",
        None,
    )
    if optional_component:
        _append_optional_component(case)
    if not keep_networking:
        _remove_networking_component(case)
    (case / "patches" / "case-local.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False)
    )
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/case-local.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))
    return case


@pytest.mark.parametrize("keep_networking", (False, True))
def test_validator_rejects_root_patch_adding_provider_networking_env(
    tmp_path: Path,
    keep_networking: bool,
) -> None:
    case = _root_merge_patch_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_env_patch("PrefillWorker", _MERGE_NETWORK_ENV),
                {"name": "DecodeWorker"},
            ]
        ),
        keep_networking=keep_networking,
    )
    _assert_error(core._validate(case), "networking-delta")


def test_validator_rejects_root_patch_adding_provider_annotation(
    tmp_path: Path,
) -> None:
    case = _root_merge_patch_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                {
                    "name": "PrefillWorker",
                    "podTemplate": {
                        "metadata": {
                            "annotations": {"k8s.v1.cni.cncf.io/networks": "rdma-net-0"}
                        }
                    },
                },
                {"name": "DecodeWorker"},
            ]
        ),
        keep_networking=False,
    )
    _assert_error(core._validate(case), "networking-delta")


def test_validator_accepts_root_patch_networking_for_optional_component(
    tmp_path: Path,
) -> None:
    case = _root_merge_patch_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                {"name": "PrefillWorker"},
                {"name": "DecodeWorker"},
                _worker_env_patch("Planner", _MERGE_NETWORK_ENV),
            ]
        ),
        keep_networking=False,
        optional_component=True,
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    planner_main = rendered["Planner"]["podTemplate"]["spec"]["containers"][0]
    assert planner_main["env"] == _MERGE_NETWORK_ENV
    for name in ("PrefillWorker", "DecodeWorker"):
        env = rendered[name]["podTemplate"]["spec"]["containers"][0]["env"]
        assert all(entry["name"] != "FI_PROVIDER" for entry in env)


def test_validator_accepts_root_hook_move_after_networking_prepend(
    tmp_path: Path,
) -> None:
    case = core._filled_disagg_case(tmp_path, *core.DISAGG_CASES[0])

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("PrefillWorker", "DecodeWorker"):
        env = rendered[name]["podTemplate"]["spec"]["containers"][0]["env"]
        assert env[0] == {
            "name": "KV_TRANSFER_CONFIG",
            "value": "cluster-kv-transfer-config",
        }
        assert env[1]["name"] == "NCCL_SOCKET_IFNAME"


def _attach_root_merge_patch(case: Path, document: dict[str, Any]) -> None:
    (case / "patches" / "case-local.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False)
    )
    kustomization_path = case / "kustomization.yaml"
    kustomization = yaml.safe_load(kustomization_path.read_text())
    kustomization["patches"] = [
        {
            "target": {
                "group": "nvidia.com",
                "version": "v1beta1",
                "kind": "DynamoGraphDeployment",
            },
            "path": "patches/case-local.yaml",
        }
    ]
    kustomization_path.write_text(yaml.safe_dump(kustomization, sort_keys=False))


def _worker_resource_patch(name: str, key: str, quantity: Any) -> dict[str, Any]:
    return {
        "name": name,
        "podTemplate": {
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "resources": {
                            "requests": {key: quantity},
                            "limits": {key: quantity},
                        },
                    }
                ]
            }
        },
    }


def _template_gpu_quantity() -> Any:
    template = TEMPLATE_ROOT / "trtllm" / "disagg" / "deploy-v1beta1.template.yaml"
    dgd = next(
        document
        for document in yaml.safe_load_all(template.read_text())
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    )
    main = dgd["spec"]["components"][1]["podTemplate"]["spec"]["containers"][0]
    return main["resources"]["requests"]["nvidia.com/gpu"]


@pytest.mark.parametrize("keep_networking", (False, True))
def test_validator_rejects_root_patch_replacing_portable_gpu_resources(
    tmp_path: Path,
    keep_networking: bool,
) -> None:
    current = _template_gpu_quantity()
    doubled = str(int(current) * 2) if isinstance(current, str) else int(current) * 2
    case = _root_merge_patch_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_resource_patch("PrefillWorker", "nvidia.com/gpu", doubled),
                {"name": "DecodeWorker"},
            ]
        ),
        keep_networking=keep_networking,
    )
    _assert_error(core._validate(case), "networking-delta")


def test_validator_accepts_root_patch_replacing_component_added_resource(
    tmp_path: Path,
) -> None:
    key = "example.com/prefill-rdma"
    case = _filled_provider_case(tmp_path, "ib", *PROVIDER_BASES[2])
    _attach_root_merge_patch(
        case,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_resource_patch("PrefillWorker", key, "4"),
                {"name": "DecodeWorker"},
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    main = rendered["PrefillWorker"]["podTemplate"]["spec"]["containers"][0]
    assert main["resources"]["requests"][key] == "4"
    assert main["resources"]["limits"][key] == "4"


def _agg_case_with_optional_component(tmp_path: Path) -> Path:
    """Aggregate vLLM case (Frontend, Worker) plus an optional Planner at index 2."""

    case = core._filled_case(tmp_path, with_case_override=False)
    base_path = case / "base.yaml"
    documents = core._documents(base_path)
    dgd = next(
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    )
    dgd["spec"]["components"].append(
        {
            "name": "Planner",
            "type": "planner",
            "replicas": 1,
            "podTemplate": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "image": "example.invalid/planner:latest",
                            "env": [{"name": "PLANNER_MODE", "value": "auto"}],
                        }
                    ]
                }
            },
        }
    )
    core._write_documents(base_path, documents)
    return case


def test_validator_rejects_agg_networking_component_touching_optional_component(
    tmp_path: Path,
) -> None:
    case = _agg_case_with_optional_component(tmp_path)
    patch_path = case / "components" / "network-interface" / "agg" / "patch-dgd.yaml"
    document = yaml.safe_load(patch_path.read_text())
    document["spec"]["components"].append(
        _worker_env_patch("Planner", [{"name": "NCCL_SOCKET_IFNAME", "value": "eth0"}])
    )
    patch_path.write_text(yaml.safe_dump(document, sort_keys=False))

    _assert_error(core._validate(case), "networking-delta")


def test_validator_accepts_agg_root_patch_on_optional_component(
    tmp_path: Path,
) -> None:
    case = _agg_case_with_optional_component(tmp_path)
    added = [{"name": "LOG_LEVEL", "value": "debug"}]
    _attach_root_merge_patch(
        case,
        _merge_patch(
            [
                {"name": "Frontend"},
                {"name": "Worker"},
                _worker_env_patch("Planner", added),
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    planner_env = rendered["Planner"]["podTemplate"]["spec"]["containers"][0]["env"]
    assert planner_env[0] == added[0]
    worker_env = rendered["Worker"]["podTemplate"]["spec"]["containers"][0]["env"]
    assert all(entry["name"] != "LOG_LEVEL" for entry in worker_env)


def test_validator_accepts_root_patch_replacing_core_resource(tmp_path: Path) -> None:
    case = core._filled_case(tmp_path, with_case_override=False)
    _attach_root_merge_patch(
        case,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_resource_patch("Worker", "memory", "123Gi"),
            ]
        ),
    )

    result = core._validate(case)
    rendered = _components(_rendered_dgd(case))

    assert result.returncode == 0, result.stdout + result.stderr
    resources = rendered["Worker"]["podTemplate"]["spec"]["containers"][0]["resources"]
    assert resources["requests"]["memory"] == "123Gi"
    assert resources["limits"]["memory"] == "123Gi"


def test_validator_rejects_networking_component_adding_core_resource(
    tmp_path: Path,
) -> None:
    case, _ = _custom_merge_networking_case(
        tmp_path,
        _merge_patch(
            [
                {"name": "Frontend"},
                _worker_resource_patch("PrefillWorker", "memory", "123Gi"),
                {"name": "DecodeWorker"},
            ]
        ),
    )
    _assert_error(core._validate(case), "networking-delta")
