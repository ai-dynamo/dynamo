# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = REPO_ROOT / "recipes" / "templates"

TEMPLATE_CASES = (
    "sglang/agg/deploy-v1alpha1.template.yaml",
    "sglang/agg/deploy-v1beta1.template.yaml",
    "sglang/agg/deploy-v1beta1-compute-domain.template.yaml",
    "sglang/disagg/deploy-v1alpha1.template.yaml",
    "sglang/disagg/deploy-v1beta1.template.yaml",
    "trtllm/agg/deploy-v1alpha1.template.yaml",
    "trtllm/agg/deploy-v1beta1.template.yaml",
    "trtllm/disagg/deploy-v1alpha1.template.yaml",
    "trtllm/disagg/deploy-v1beta1.template.yaml",
    "vllm/agg/deploy-v1alpha1.template.yaml",
    "vllm/agg/deploy-v1beta1.template.yaml",
    "vllm/disagg/deploy-v1alpha1.template.yaml",
    "vllm/disagg/deploy-v1beta1.template.yaml",
    "vllm/disagg/deploy-v1beta1-compute-domain.template.yaml",
)

VLLM_COMPUTE_DOMAIN_TEMPLATE = "vllm/disagg/deploy-v1beta1-compute-domain.template.yaml"


def _load(relative_path: str) -> tuple[str, dict[str, Any]]:
    text = (TEMPLATE_ROOT / relative_path).read_text()
    documents = tuple(yaml.safe_load_all(text))
    dgds = tuple(
        document
        for document in documents
        if isinstance(document, dict)
        and document.get("kind") == "DynamoGraphDeployment"
    )
    assert len(dgds) == 1, relative_path
    return text, dgds[0]


def _load_documents(relative_path: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        document
        for document in yaml.safe_load_all((TEMPLATE_ROOT / relative_path).read_text())
        if isinstance(document, dict)
    )


def _roles(dgd: dict[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    if dgd["apiVersion"] == "nvidia.com/v1alpha1":
        return tuple(dgd["spec"]["services"].items())
    return tuple(
        (component["name"], component) for component in dgd["spec"]["components"]
    )


def _main_container(dgd: dict[str, Any], role: dict[str, Any]) -> dict[str, Any]:
    if dgd["apiVersion"] == "nvidia.com/v1alpha1":
        return role["extraPodSpec"]["mainContainer"]
    return next(
        container
        for container in role["podTemplate"]["spec"]["containers"]
        if container["name"] == "main"
    )


def _environment(
    dgd: dict[str, Any], role: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    main = _main_container(dgd, role)
    return tuple(role.get("envs", ())) + tuple(main.get("env", ()))


def _has_credential_secret(dgd: dict[str, Any], role: dict[str, Any]) -> bool:
    if dgd["apiVersion"] == "nvidia.com/v1alpha1":
        return "envFromSecret" in role
    main = _main_container(dgd, role)
    return any("secretRef" in source for source in main.get("envFrom", ()))


def _mappings(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def test_template_catalog_has_expected_inventory() -> None:
    actual = tuple(
        sorted(
            path.relative_to(TEMPLATE_ROOT).as_posix()
            for path in TEMPLATE_ROOT.rglob("*.template.yaml")
        )
    )
    expected = tuple(sorted(TEMPLATE_CASES))

    assert actual == expected


def test_all_recipe_yaml_documents_parse() -> None:
    yaml_paths = tuple(sorted(TEMPLATE_ROOT.rglob("*.yaml")))

    assert yaml_paths, "recipe YAML inventory is empty"
    for yaml_path in yaml_paths:
        relative_path = yaml_path.relative_to(REPO_ROOT)
        try:
            tuple(yaml.safe_load_all(yaml_path.read_text()))
        except yaml.YAMLError as error:
            pytest.fail(f"{relative_path}: {error}")


def test_check_yaml_excludes_renderer_owned_templates_but_not_recipe_yaml() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    check_yaml_hook = next(
        hook
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["id"] == "check-yaml"
    )
    exclusion = re.compile(check_yaml_hook["exclude"])
    recipe_paths = tuple(
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(TEMPLATE_ROOT.rglob("*.yaml"))
    )
    interpolation_template_paths = (
        "benchmarks/frontend/dgd/templates/mocker.yaml",
        "benchmarks/frontend/dgd/templates/vllm-gpt-oss-20b.yaml",
        "benchmarks/frontend/dgd/templates/vllm.yaml",
        "examples/deployments/EKS/templates/eksctl.yaml",
    )

    assert len(recipe_paths) == 42
    assert not any(exclusion.match(path) for path in recipe_paths)
    assert all(exclusion.match(path) for path in interpolation_template_paths)


@pytest.mark.parametrize("relative_path", TEMPLATE_CASES)
def test_templates_use_canonical_component_order(relative_path: str) -> None:
    _, dgd = _load(relative_path)
    expected = (
        ("Frontend", "Worker")
        if "/agg/" in relative_path
        else ("Frontend", "PrefillWorker", "DecodeWorker")
    )

    assert tuple(name for name, _ in _roles(dgd)) == expected


@pytest.mark.parametrize("relative_path", TEMPLATE_CASES)
def test_templates_omit_probe_structs(relative_path: str) -> None:
    _, dgd = _load(relative_path)
    for _, role in _roles(dgd):
        main = _main_container(dgd, role)
        assert "livenessProbe" not in main
        assert "readinessProbe" not in main
        assert "startupProbe" not in main


@pytest.mark.parametrize("relative_path", TEMPLATE_CASES)
def test_templates_omit_credentials_and_beta_templates_are_offline(
    relative_path: str,
) -> None:
    _, dgd = _load(relative_path)
    for _, role in _roles(dgd):
        assert not _has_credential_secret(dgd, role)
        main = _main_container(dgd, role)
        assert all("secretRef" not in source for source in main.get("envFrom", ()))
        if dgd["apiVersion"] == "nvidia.com/v1beta1":
            env = {
                entry["name"]: entry.get("value") for entry in _environment(dgd, role)
            }
            assert env["HF_HUB_OFFLINE"] == "1"
            assert env["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("relative_path", TEMPLATE_CASES)
def test_template_manifest_conventions(relative_path: str) -> None:
    text, dgd = _load(relative_path)

    assert all(
        "readOnly" not in mapping["persistentVolumeClaim"]
        for mapping in _mappings(dgd)
        if "persistentVolumeClaim" in mapping
    )
    if dgd["spec"]["backendFramework"] != "sglang":
        assert "--model-path" not in text
    assert not re.search(r"--kv-transfer-config(?:=|\s+)['\"]\{", text)

    if dgd["apiVersion"] == "nvidia.com/v1alpha1":
        assert dgd["spec"]["pvcs"][0]["name"] == "shared-model-cache"
        for _, role in _roles(dgd):
            for mount in role.get("volumeMounts", ()):
                if "cache" in mount["name"]:
                    assert mount == {
                        "name": "shared-model-cache",
                        "mountPoint": "/shared-model-cache",
                    }
    else:
        for _, role in _roles(dgd):
            main = _main_container(dgd, role)
            mounts = main.get("volumeMounts", ())
            volumes = role["podTemplate"]["spec"].get("volumes", ())
            if any(mount["name"] == "shared-model-cache" for mount in mounts):
                assert mounts[0] == {
                    "name": "shared-model-cache",
                    "mountPath": "/shared-model-cache",
                    "readOnly": True,
                }
            if any(volume["name"] == "shared-model-cache" for volume in volumes):
                assert volumes[0]["name"] == "shared-model-cache"
                assert volumes[0]["persistentVolumeClaim"]["claimName"] == (
                    "shared-model-cache"
                )
            assert all(volume["name"] != "dshm" for volume in volumes)
            assert all(mount.get("mountPath") != "/dev/shm" for mount in mounts)

    if "DYN_FORWARDPASS_METRIC_PORT" in text:
        assert re.search(r'value: ""\s+#.*empty.*disable', text, re.IGNORECASE)


@pytest.mark.parametrize("relative_path", TEMPLATE_CASES)
def test_templates_use_exec_form_and_explicit_pull_policy(relative_path: str) -> None:
    text, dgd = _load(relative_path)

    for _, role in _roles(dgd):
        main = _main_container(dgd, role)
        assert main["imagePullPolicy"] == "IfNotPresent"
        assert main["command"] == ["python3"]
        assert isinstance(main["args"], list)
        assert all(isinstance(argument, str) for argument in main["args"])
        assert main["args"][0] == "-m"
        assert main["args"][1].startswith("dynamo.")
    assert "/bin/bash" not in text
    assert "${" not in text


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    (
        ("vllm/agg/deploy-v1alpha1.template.yaml", (("Worker", "20Gi"),)),
        (
            "vllm/disagg/deploy-v1alpha1.template.yaml",
            (("PrefillWorker", "80Gi"), ("DecodeWorker", "80Gi")),
        ),
        ("sglang/agg/deploy-v1alpha1.template.yaml", (("Worker", "16Gi"),)),
        (
            "sglang/disagg/deploy-v1alpha1.template.yaml",
            (("PrefillWorker", "16Gi"), ("DecodeWorker", "16Gi")),
        ),
        ("trtllm/agg/deploy-v1alpha1.template.yaml", (("Worker", "80Gi"),)),
        (
            "trtllm/disagg/deploy-v1alpha1.template.yaml",
            (("PrefillWorker", "16Gi"), ("DecodeWorker", "16Gi")),
        ),
    ),
)
def test_alpha_worker_shared_memory_is_unchanged(
    relative_path: str, expected: tuple[tuple[str, str], ...]
) -> None:
    _, dgd = _load(relative_path)
    roles = dict(_roles(dgd))

    actual = tuple(
        (name, roles[name].get("sharedMemory", {}).get("size")) for name, _ in expected
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    (
        ("vllm/agg/deploy-v1beta1.template.yaml", (("Worker", "64Gi"),)),
        (
            "vllm/disagg/deploy-v1beta1.template.yaml",
            (("PrefillWorker", "64Gi"), ("DecodeWorker", "64Gi")),
        ),
        (
            VLLM_COMPUTE_DOMAIN_TEMPLATE,
            (("PrefillWorker", "200Gi"), ("DecodeWorker", "200Gi")),
        ),
        ("sglang/agg/deploy-v1beta1.template.yaml", (("Worker", "512Gi"),)),
        (
            "sglang/agg/deploy-v1beta1-compute-domain.template.yaml",
            (("Worker", "200Gi"),),
        ),
        (
            "sglang/disagg/deploy-v1beta1.template.yaml",
            (("PrefillWorker", "64Gi"), ("DecodeWorker", "64Gi")),
        ),
        ("trtllm/agg/deploy-v1beta1.template.yaml", (("Worker", "40Gi"),)),
        (
            "trtllm/disagg/deploy-v1beta1.template.yaml",
            (("PrefillWorker", "16Gi"), ("DecodeWorker", "16Gi")),
        ),
    ),
)
def test_beta_worker_shared_memory_is_operator_owned(
    relative_path: str, expected: tuple[tuple[str, str], ...]
) -> None:
    _, dgd = _load(relative_path)
    roles = dict(_roles(dgd))

    actual = tuple((name, roles[name].get("sharedMemorySize")) for name, _ in expected)
    assert actual == expected


def test_sglang_aggregate_scratch_volumes_are_preserved() -> None:
    _, dgd = _load("sglang/agg/deploy-v1beta1.template.yaml")
    worker = dict(_roles(dgd))["Worker"]
    main = _main_container(dgd, worker)
    mounts = {mount["name"]: mount["mountPath"] for mount in main["volumeMounts"]}
    volumes = {
        volume["name"]: volume for volume in worker["podTemplate"]["spec"]["volumes"]
    }

    assert mounts["tmp"] == "/tmp"
    assert mounts["flashinfer-cache"] == "/root/.cache/flashinfer"
    assert volumes["tmp"]["emptyDir"]["sizeLimit"] == "200Gi"
    assert volumes["flashinfer-cache"]["emptyDir"]["sizeLimit"] == "32Gi"


@pytest.mark.parametrize(
    "relative_path",
    tuple(path for path in TEMPLATE_CASES if "v1beta1" in path),
)
def test_beta_workers_use_the_security_default(relative_path: str) -> None:
    _, dgd = _load(relative_path)

    for name, role in _roles(dgd):
        if name == "Frontend":
            assert "securityContext" not in _main_container(dgd, role)
            continue
        assert _main_container(dgd, role)["securityContext"] == {
            "runAsUser": 0,
            "runAsGroup": 0,
            "capabilities": {
                "add": ["IPC_LOCK", "SYS_PTRACE", "SYS_RESOURCE"],
            },
        }


@pytest.mark.parametrize(
    "relative_path",
    tuple(path for path in TEMPLATE_CASES if "v1beta1" in path),
)
def test_beta_frontends_do_not_mount_the_model_cache(relative_path: str) -> None:
    _, dgd = _load(relative_path)
    frontend = dict(_roles(dgd))["Frontend"]
    main = _main_container(dgd, frontend)
    mounts = main.get("volumeMounts", ())
    volumes = frontend["podTemplate"]["spec"].get("volumes", ())
    env = _environment(dgd, frontend)

    assert all(mount.get("name") != "shared-model-cache" for mount in mounts)
    assert all(volume.get("name") != "shared-model-cache" for volume in volumes)
    assert all(entry.get("name") != "HF_HOME" for entry in env)
    assert all("/shared-model-cache" not in value for value in _strings(main))


def test_beta_workers_with_read_only_hf_home_use_writable_modules_cache() -> None:
    for relative_path in (path for path in TEMPLATE_CASES if "v1beta1" in path):
        _, dgd = _load(relative_path)
        for name, role in _roles(dgd):
            if name == "Frontend":
                continue
            environment = _environment(dgd, role)
            if not any(
                entry.get("name") == "HF_HOME"
                and entry.get("value") == "/shared-model-cache"
                for entry in environment
            ):
                continue
            modules_cache_entries = tuple(
                entry
                for entry in environment
                if entry.get("name") == "HF_MODULES_CACHE"
            )
            assert modules_cache_entries == (
                {"name": "HF_MODULES_CACHE", "value": "/tmp/hf_modules"},
            ), f"{relative_path}: {name}"


@pytest.mark.parametrize(
    ("relative_path", "default"),
    (
        # Each template keeps the transfer settings of the recipe it was derived
        # from, so the hook defaults are not identical across the two API shapes.
        (
            "vllm/disagg/deploy-v1alpha1.template.yaml",
            '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        ),
        (
            "vllm/disagg/deploy-v1beta1.template.yaml",
            '{"kv_connector":"NixlConnector","kv_role":"kv_both",'
            '"kv_buffer_device":"cuda"}',
        ),
        (
            VLLM_COMPUTE_DOMAIN_TEMPLATE,
            '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        ),
    ),
)
def test_vllm_disaggregated_transfer_uses_the_env_hook(
    relative_path: str, default: str
) -> None:
    _, dgd = _load(relative_path)

    for name, worker in _roles(dgd):
        if name not in {"PrefillWorker", "DecodeWorker"}:
            continue
        main = _main_container(dgd, worker)
        assert main["env"][0] == {"name": "KV_TRANSFER_CONFIG", "value": default}
        transfer_index = main["args"].index("--kv-transfer-config")
        assert main["args"][transfer_index + 1] == "$(KV_TRANSFER_CONFIG)"


def test_vllm_compute_domain_variant_preserves_the_dra_and_runtime_contract() -> None:
    documents = _load_documents(VLLM_COMPUTE_DOMAIN_TEMPLATE)
    assert tuple(document["kind"] for document in documents) == (
        "ComputeDomain",
        "DynamoGraphDeployment",
    )

    compute_domain, dgd = documents
    channel_template = compute_domain["spec"]["channel"]["resourceClaimTemplate"][
        "name"
    ]
    assert compute_domain["apiVersion"] == "resource.nvidia.com/v1beta1"
    assert compute_domain["spec"]["numNodes"] == 0
    assert channel_template == "dsv4-pro-compute-domain-channel"
    assert dgd["apiVersion"] == "nvidia.com/v1beta1"
    assert dgd["spec"]["backendFramework"] == "vllm"

    roles = dict(_roles(dgd))
    assert tuple(roles) == ("Frontend", "PrefillWorker", "DecodeWorker")
    assert roles["Frontend"]["sharedMemorySize"] == "40Gi"

    expected_common_env = (
        "KV_TRANSFER_CONFIG",
        "MODEL_NAME",
        "SERVED_MODEL_NAME",
        "HF_HOME",
        "HF_MODULES_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLLM_ENGINE_READY_TIMEOUT_S",
        "TILELANG_CLEANUP_TEMP_FILES",
        "VLLM_SERVER_DEV_MODE",
        "VLLM_USE_NCCL_SYMM_MEM",
        "NCCL_CUMEM_ENABLE",
        "NCCL_MNNVL_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_STORE_TIMEOUT",
        "NVIDIA_GDRCOPY",
        "UCX_MEMTYPE_CACHE",
        "UCX_TLS",
        "UCX_CUDA_IPC_ENABLE_MNNVL",
    )
    prefill_only_env = (
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
        "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE",
    )
    expected_env_values = {
        "KV_TRANSFER_CONFIG": ('{"kv_connector":"NixlConnector","kv_role":"kv_both"}'),
        "MODEL_NAME": "deepseek-ai/DeepSeek-V4-Pro",
        "SERVED_MODEL_NAME": "deepseek-ai/DeepSeek-V4-Pro",
        "HF_HOME": "/shared-model-cache",
        "HF_MODULES_CACHE": "/tmp/hf_modules",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "5400",
        "TILELANG_CLEANUP_TEMP_FILES": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_NCCL_SYMM_MEM": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_MNNVL_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_STORE_TIMEOUT": "7200",
        "NVIDIA_GDRCOPY": "1",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_TLS": "cuda_copy,cuda_ipc,tcp",
        "UCX_CUDA_IPC_ENABLE_MNNVL": "y",
    }

    for name, mode in (("PrefillWorker", "prefill"), ("DecodeWorker", "decode")):
        worker = roles[name]
        pod_spec = worker["podTemplate"]["spec"]
        main = _main_container(dgd, worker)

        assert worker["replicas"] == 1
        assert worker["multinode"]["nodeCount"] == 2
        assert worker["sharedMemorySize"] == "200Gi"
        assert pod_spec["resourceClaims"] == [
            {
                "name": "compute-domain-channel",
                "resourceClaimTemplateName": channel_template,
            }
        ]
        assert main["resources"] == {
            "requests": {"nvidia.com/gpu": "4"},
            "limits": {"nvidia.com/gpu": "4"},
            "claims": [{"name": "compute-domain-channel"}],
        }
        assert main["securityContext"] == {
            "runAsUser": 0,
            "runAsGroup": 0,
            "capabilities": {
                "add": ["IPC_LOCK", "SYS_PTRACE", "SYS_RESOURCE"],
            },
        }

        env_names = tuple(entry["name"] for entry in main["env"])
        expected_env = list(expected_common_env)
        if name == "PrefillWorker":
            expected_env[9:9] = prefill_only_env
        assert env_names == tuple(expected_env)
        expected_values = dict(expected_env_values)
        if name == "PrefillWorker":
            expected_values.update(
                {
                    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "1024",
                    "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE": "2048",
                }
            )
        assert {entry["name"]: entry["value"] for entry in main["env"]} == (
            expected_values
        )
        assert "NCCL_SOCKET_IFNAME" not in env_names
        assert "GLOO_SOCKET_IFNAME" not in env_names

        expected_pairs = (
            ("--tensor-parallel-size", "1"),
            ("--pipeline-parallel-size", "1"),
            ("--data-parallel-size", "8"),
            ("--disaggregation-mode", mode),
            ("--kv-transfer-config", "$(KV_TRANSFER_CONFIG)"),
        )
        for flag, value in expected_pairs:
            flag_index = main["args"].index(flag)
            assert main["args"][flag_index + 1] == value


def test_vllm_beta_kv_routing_matches_worker_event_publication() -> None:
    for relative_path in (
        "vllm/agg/deploy-v1beta1.template.yaml",
        "vllm/disagg/deploy-v1beta1.template.yaml",
    ):
        _, dgd = _load(relative_path)
        frontend = dict(_roles(dgd))["Frontend"]
        args = _main_container(dgd, frontend)["args"]
        assert "--no-router-kv-events" in args
        assert "--router-kv-events" not in args


@pytest.mark.parametrize(
    "relative_path",
    (
        "sglang/agg/deploy-v1alpha1.template.yaml",
        "sglang/disagg/deploy-v1alpha1.template.yaml",
    ),
)
def test_sglang_alpha_frontends_use_router_kv_event_flag(
    relative_path: str,
) -> None:
    _, dgd = _load(relative_path)
    frontend = dict(_roles(dgd))["Frontend"]
    args = _main_container(dgd, frontend)["args"]

    assert "--no-kv-events" not in args
    assert args.count("--no-router-kv-events") == 1


@pytest.mark.parametrize(
    ("relative_path", "historical_url"),
    (
        (
            "vllm/agg/deploy-v1alpha1.template.yaml",
            "https://github.com/ai-dynamo/dynamo/blob/"
            "67203f32d2508c96c9387d263e0f02f4f3830f3f/"
            "recipes/llama-3-70b/vllm/agg/deploy.yaml",
        ),
        (
            "vllm/disagg/deploy-v1alpha1.template.yaml",
            "https://github.com/ai-dynamo/dynamo/blob/"
            "67203f32d2508c96c9387d263e0f02f4f3830f3f/"
            "recipes/llama-3-70b/vllm/disagg-multi-node/deploy.yaml",
        ),
    ),
)
def test_vllm_alpha_provenance_uses_pinned_historical_url(
    relative_path: str, historical_url: str
) -> None:
    text, _ = _load(relative_path)
    header = text.split("apiVersion:", 1)[0]
    readme = (TEMPLATE_ROOT / "README.md").read_text()
    example_name = (
        "vLLM aggregate alpha"
        if "/agg/" in relative_path
        else "vLLM disaggregated alpha"
    )
    readme_row = next(
        line
        for line in readme.splitlines()
        if line.startswith(f"| [{example_name}]({relative_path})")
    )

    assert f"# Derived from: {historical_url}." in header
    assert f"]({historical_url})" in readme_row


def test_inkling_hostipc_removal_is_documented_as_cluster_policy() -> None:
    relative_path = "sglang/agg/deploy-v1beta1.template.yaml"
    text, _ = _load(relative_path)
    header = text.split("apiVersion:", 1)[0]
    readme = (TEMPLATE_ROOT / "README.md").read_text()
    readme_row = next(
        line
        for line in readme.splitlines()
        if line.startswith(f"| [SGLang aggregate beta]({relative_path})")
    )

    for provenance in (header, readme_row):
        assert "hostIPC: true" in provenance
        assert "intentionally removed" in provenance
        assert "host IPC is cluster/host policy" in provenance


def test_aggregate_placement_selector_targets_only_workers() -> None:
    patch_path = TEMPLATE_ROOT / "kustomize/components/placement/agg/patch-dgd.yaml"
    patch = yaml.safe_load(patch_path.read_text())
    affinity_operation = next(
        operation
        for operation in patch
        if operation.get("path")
        == "/spec/components/1/podTemplate/spec/affinity/podAffinity"
    )
    required_affinity = affinity_operation["value"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    values = required_affinity[0]["labelSelector"]["matchExpressions"][0]["values"]

    assert values == ["worker"]


def test_disaggregate_placement_selector_targets_prefill_and_decode_workers() -> None:
    patch_path = TEMPLATE_ROOT / "kustomize/components/placement/disagg/patch-dgd.yaml"
    patch = yaml.safe_load(patch_path.read_text())
    affinity_operations = [
        operation
        for operation in patch
        if operation.get("path", "").endswith("/affinity/podAffinity")
    ]

    assert [operation["path"] for operation in affinity_operations] == [
        "/spec/components/1/podTemplate/spec/affinity/podAffinity",
        "/spec/components/2/podTemplate/spec/affinity/podAffinity",
    ]
    for operation in affinity_operations:
        required_affinity = operation["value"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        values = required_affinity[0]["labelSelector"]["matchExpressions"][0]["values"]
        assert values == ["prefill", "decode"]


def test_framework_transfer_override_boundaries_are_documented() -> None:
    sglang_text, sglang_dgd = _load("sglang/disagg/deploy-v1beta1.template.yaml")
    sglang_header = sglang_text.split("apiVersion:", 1)[0]
    assert "SGLANG_DISAGGREGATION_NIXL_BACKEND" in sglang_header
    assert "To change the transfer backend" in sglang_header
    for name, worker in _roles(sglang_dgd):
        if name not in {"PrefillWorker", "DecodeWorker"}:
            continue
        main = _main_container(sglang_dgd, worker)
        assert main["env"][0]["name"] == "SGLANG_DISAGGREGATION_NIXL_BACKEND"

    for relative_path in (
        "trtllm/disagg/deploy-v1alpha1.template.yaml",
        "trtllm/disagg/deploy-v1beta1.template.yaml",
    ):
        text, _ = _load(relative_path)
        header = text.split("apiVersion:", 1)[0]
        assert "no environment-variable override" in header
        assert "cache_transceiver_config" in header
        assert "in the engine ConfigMaps" in header


def test_readme_documents_operator_shared_memory_semantics() -> None:
    readme = (TEMPLATE_ROOT / "README.md").read_text()

    assert "When `sharedMemorySize` is omitted" in readme
    assert "operator injects an 8Gi `/dev/shm` volume" in readme
    assert "A positive `sharedMemorySize`" in readme
    assert "drop any manual mount at `/dev/shm`" in readme
    assert '`sharedMemorySize: "0"` disables operator injection' in readme
    assert "only mode in which a manual `/dev/shm` volume applies" in readme
    assert "catalog does not use it" in readme
