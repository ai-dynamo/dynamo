# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    from dynamo.profiler.utils.dgd_generation import (
        apply_unauthenticated_mode,
        assemble_final_config,
    )
    from dynamo.profiler.utils.dgd_template import load_dgd_template
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
    )
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

_BACKENDS = ["vllm", "sglang", "trtllm"]


def _dgdr(unauthenticated: bool | None) -> DynamoGraphDeploymentRequestSpec:
    kwargs = {}
    if unauthenticated is not None:
        kwargs["unauthenticated"] = unauthenticated
    return DynamoGraphDeploymentRequestSpec(**kwargs)


def _secret_refs(config: dict, component_name: str, container_name: str) -> list[str]:
    component = next(
        component
        for component in config["spec"]["components"]
        if component["name"] == component_name
    )
    container = next(
        container
        for container in component["podTemplate"]["spec"]["containers"]
        if container["name"] == container_name
    )
    return [
        entry["secretRef"]["name"]
        for entry in container.get("envFrom", [])
        if "secretRef" in entry
    ]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("unauthenticated", [None, False])
def test_authenticated_requests_keep_required_secret_reference(
    backend: str, mode: str, unauthenticated: bool | None
) -> None:
    config = load_dgd_template(backend, mode)
    dgdr = _dgdr(unauthenticated)

    apply_unauthenticated_mode(dgdr, config)

    frontend = next(
        component
        for component in config["spec"]["components"]
        if component["type"] == "frontend"
    )
    for container in frontend["podTemplate"]["spec"]["containers"]:
        for entry in container.get("envFrom", []):
            if entry.get("secretRef", {}).get("name") == "hf-token-secret":
                assert "optional" not in entry["secretRef"]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_unauthenticated_removes_secret_reference_from_all_containers(
    backend: str, mode: str
) -> None:
    config = load_dgd_template(backend, mode)
    dgdr = _dgdr(unauthenticated=True)

    apply_unauthenticated_mode(dgdr, config)

    for component in config["spec"]["components"]:
        containers = (
            component.get("podTemplate", {}).get("spec", {}).get("containers", [])
        )
        for container in containers:
            secret_refs = [
                entry["secretRef"]["name"]
                for entry in container.get("envFrom", [])
                if "secretRef" in entry
            ]
            assert "hf-token-secret" not in secret_refs, (
                f"{component['name']}/{container['name']} still references "
                "hf-token-secret"
            )


def test_unauthenticated_preserves_other_env_from_entries() -> None:
    config = load_dgd_template("vllm", "agg")
    frontend_container = next(
        component
        for component in config["spec"]["components"]
        if component["type"] == "frontend"
    )["podTemplate"]["spec"]["containers"][0]
    frontend_container["envFrom"].append(
        {"configMapRef": {"name": "extra-env", "optional": True}}
    )

    apply_unauthenticated_mode(_dgdr(unauthenticated=True), config)

    env_from = frontend_container["envFrom"]
    assert {"configMapRef": {"name": "extra-env", "optional": True}} in env_from
    assert all(
        entry.get("secretRef", {}).get("name") != "hf-token-secret"
        for entry in env_from
    )


def test_unauthenticated_strips_mocker_template() -> None:
    config = load_dgd_template("mocker", "disagg")

    apply_unauthenticated_mode(_dgdr(unauthenticated=True), config)

    for component in config["spec"]["components"]:
        containers = (
            component.get("podTemplate", {}).get("spec", {}).get("containers", [])
        )
        for container in containers:
            assert "hf-token-secret" not in _secret_refs(
                config, component["name"], container["name"]
            )


class _Ops:
    output_dir = "/tmp"


def test_assemble_final_config_applies_unauthenticated_mode() -> None:
    config = load_dgd_template("vllm", "agg")
    dgdr = _dgdr(unauthenticated=True)

    assembled = assemble_final_config(
        dgdr,
        _Ops(),
        config,
        resolved_backend="vllm",
    )

    frontend = next(
        component
        for component in assembled["spec"]["components"]
        if component["type"] == "frontend"
    )
    for container in frontend["podTemplate"]["spec"]["containers"]:
        assert "hf-token-secret" not in _secret_refs(
            assembled, frontend["name"], container["name"]
        )
