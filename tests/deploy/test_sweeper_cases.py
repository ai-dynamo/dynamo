# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live validation for comparison DGDs selected directly from a suite.

The deployment preparation and cluster-inventory flow builds on Ashna Mehrotra's
work in https://github.com/ai-dynamo/dynamo/pull/14031. This version deliberately
does not use a custom report file as its discovery or result protocol.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from dynamo.profiler.tests.sweeper.run_cases import (
    SuiteException,
    default_suite_output_root,
    load_case,
    load_hardware,
    load_recipe,
    load_suite,
    missing_case_inputs,
    record_cluster_result,
    recipe_gpu_count,
    write_discovered_recipe_requirement,
)
from tests.deploy.conftest import DeploymentTarget, _deploy_test_call_report_key
from tests.deploy.dgd_utils import DeploymentSpec
from tests.deploy.dgdr_utils import kubectl
from tests.deploy.test_dgd import test_deployment as run_dgd_deployment_test

pytestmark = [
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.e2e,
    pytest.mark.integration,
    pytest.mark.nightly,
]

_VARIANT_FILES = {
    "profiler-v1beta1": "dgd-profiler-v1beta1.yaml",
    "sweeper-aic": "dgd-sweeper-aic.yaml",
    "sweeper-direct": "dgd-sweeper-direct.yaml",
}
_VARIANTS = {*_VARIANT_FILES, "recipe"}


@dataclass(frozen=True)
class SweeperDeploymentTarget:
    """One generated or recipe DGD selected for live validation."""

    case_name: str
    hardware_name: str
    variant: str
    backend: str
    deployment_mode: str
    output_dir: Path
    yaml_path: Path
    gpu_count: int
    gpu_sku: str
    expected_failure: SuiteException | None = None

    @property
    def test_id(self) -> str:
        return f"{self.hardware_name}-{self.case_name}-{self.variant}"


def _selected_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - _VARIANTS)
    if unknown:
        raise pytest.UsageError(
            f"unknown --sweeper-variants value(s): {', '.join(unknown)}"
        )
    if not variants:
        raise pytest.UsageError("--sweeper-variants must select at least one variant")
    return variants


def _discover_targets(
    suite_path: Path,
    output_root: Path,
    variants: list[str],
    discover_recipe_hardware: bool,
) -> list[SweeperDeploymentTarget]:
    targets = []
    hardware_cache = {}
    for entry in load_suite(suite_path):
        if entry.status is not None and entry.status.status == "skipped":
            continue
        case = None
        if not missing_case_inputs(entry.case):
            if entry.hardware not in hardware_cache:
                hardware_cache[entry.hardware] = load_hardware(entry.hardware)
            case = load_case(
                entry.case,
                hardware_cache[entry.hardware],
                output_root=output_root,
            )
        recipe = load_recipe(entry.case)
        for variant in variants:
            deploy_exception = entry.exception_for("deploy", variant)
            if deploy_exception is not None and deploy_exception.status == "skipped":
                continue
            if variant == "recipe":
                if recipe is None:
                    continue
                if not recipe.path.is_file():
                    if deploy_exception is not None:
                        continue
                    raise pytest.UsageError(
                        f"{entry.case}: recipe source is missing: {recipe.path}"
                    )
                requirement = recipe.requirements.get(entry.hardware)
                if requirement is None and not discover_recipe_hardware:
                    continue
                yaml_path = recipe.path
                recipe_gpus = (
                    requirement.get("gpus") if isinstance(requirement, dict) else None
                )
                gpu_count = (
                    recipe_gpus
                    if isinstance(recipe_gpus, int)
                    else recipe_gpu_count(recipe)
                )
                backend = _recipe_backend(recipe.source)
                deployment_mode = _recipe_deployment_mode(recipe.source)
            else:
                if case is None:
                    continue
                yaml_path = case.generated_dir / _VARIANT_FILES[variant]
                gpu_count = case.dgdr_input["hardware"]["totalGpus"]
                backend = case.dgdr_input["backend"]
                deployment_modes = case.sweeper_input["search_space"].get(
                    "deployment_mode", ["agg"]
                )
                deployment_mode = (
                    deployment_modes[0]
                    if isinstance(deployment_modes, list) and deployment_modes
                    else "agg"
                )
            if not yaml_path.is_file():
                render_exception = entry.exception_for("render", variant)
                if render_exception is None and variant.startswith("sweeper-"):
                    render_exception = entry.exception_for("render", "sweeper")
                if render_exception is not None:
                    continue
                raise pytest.UsageError(
                    f"{entry.hardware}/{entry.case}: requested {variant} artifact is missing: "
                    f"{yaml_path}"
                )
            targets.append(
                SweeperDeploymentTarget(
                    case_name=entry.case,
                    hardware_name=entry.hardware,
                    variant=variant,
                    backend=backend,
                    deployment_mode=deployment_mode,
                    output_dir=output_root / entry.hardware / entry.case,
                    yaml_path=yaml_path,
                    gpu_count=gpu_count,
                    gpu_sku=entry.hardware,
                    expected_failure=deploy_exception,
                )
            )
    if not targets:
        raise pytest.UsageError(f"{suite_path}: no eligible deployment targets")
    return targets


def _recipe_backend(source: str) -> str:
    matches = [
        name for name in ("vllm", "sglang", "trtllm") if name in Path(source).parts
    ]
    if len(matches) != 1:
        raise pytest.UsageError(f"cannot determine recipe backend from {source!r}")
    return matches[0]


def _recipe_deployment_mode(source: str) -> str:
    return (
        "disagg"
        if any(part.startswith("disagg") for part in Path(source).parts)
        else "agg"
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "sweeper_deployment_target" not in metafunc.fixturenames:
        return
    suite_path = metafunc.config.getoption("--sweeper-suite", default=None)
    if suite_path is None:
        metafunc.parametrize(
            "sweeper_deployment_target",
            [
                pytest.param(
                    None, marks=pytest.mark.skip(reason="--sweeper-suite is required")
                )
            ],
        )
        return
    output_root = metafunc.config.getoption(
        "--sweeper-output-dir", default=None
    ) or default_suite_output_root(suite_path)
    targets = _discover_targets(
        suite_path,
        output_root,
        _selected_variants(
            metafunc.config.getoption(
                "--sweeper-variants",
                default="profiler-v1beta1,sweeper-aic,recipe",
            )
        ),
        metafunc.config.getoption("--sweeper-discover-recipe-hardware", default=False),
    )
    parameters = []
    for target in targets:
        marks = []
        if (
            target.expected_failure is not None
            and target.expected_failure.status == "broken"
        ):
            marks.append(
                pytest.mark.xfail(
                    reason=target.expected_failure.describe(), strict=False
                )
            )
        parameters.append(pytest.param(target, marks=marks, id=target.test_id))
    metafunc.parametrize("sweeper_deployment_target", parameters)


def _gpu_family(value: str) -> str:
    normalized = value.lower().replace("-", "").replace("_", "").replace(" ", "")
    for family in ("gb300", "gb200", "b300", "b200", "h200", "h100", "a100"):
        if family in normalized:
            return family
    return normalized


def _cluster_inventory() -> dict[str, Any]:
    result = kubectl("get", "nodes", "-o", "json")
    if result.returncode != 0:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    payload = json.loads(result.stdout)
    products: dict[str, dict[str, int]] = {}
    for node in payload.get("items", []):
        labels = node.get("metadata", {}).get("labels", {})
        capacity = node.get("status", {}).get("capacity", {})
        product = labels.get("nvidia.com/gpu.product") or labels.get(
            "node.kubernetes.io/instance-type"
        )
        gpu_count = capacity.get("nvidia.com/gpu")
        if not product or gpu_count is None:
            continue
        entry = products.setdefault(product, {"nodes": 0, "gpus": 0})
        entry["nodes"] += 1
        entry["gpus"] += int(gpu_count)
    return {"gpuProducts": products}


def _validate_cluster_hardware(
    target: SweeperDeploymentTarget, inventory: dict[str, Any]
) -> None:
    products = inventory.get("gpuProducts")
    if not isinstance(products, dict) or not products:
        pytest.fail(f"could not detect cluster GPU inventory: {inventory}")
    family = _gpu_family(target.gpu_sku)
    compatible_gpus = sum(
        details["gpus"]
        for product, details in products.items()
        if _gpu_family(product) == family
    )
    if compatible_gpus < target.gpu_count:
        pytest.fail(
            f"{target.test_id} requires {target.gpu_count} {target.gpu_sku} GPUs; "
            f"detected {inventory}"
        )


def _replace_resource_names(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_resource_names(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_resource_names(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _write_recipe_discovery(target: SweeperDeploymentTarget) -> None:
    recipe = load_recipe(target.case_name)
    if recipe is None:
        raise pytest.UsageError(
            f"{target.case_name}: recipe.yaml disappeared during test"
        )
    write_discovered_recipe_requirement(
        recipe,
        target.hardware_name,
        target.gpu_count,
    )


@pytest.fixture
def prepared_deployment_path(
    sweeper_deployment_target: SweeperDeploymentTarget | None,
    namespace: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    """Extract one DGD and apply uniquely named supporting resources."""
    if sweeper_deployment_target is None:
        yield None
        return

    documents = [
        document
        for document in yaml.safe_load_all(
            sweeper_deployment_target.yaml_path.read_text()
        )
        if document
    ]
    dgds = [
        document
        for document in documents
        if document.get("kind") == "DynamoGraphDeployment"
    ]
    if len(dgds) != 1:
        pytest.fail(
            f"{sweeper_deployment_target.yaml_path} must contain exactly one "
            "DynamoGraphDeployment"
        )
    resources = [
        document
        for document in documents
        if document.get("kind") != "DynamoGraphDeployment"
    ]
    digest = hashlib.sha1(
        sweeper_deployment_target.test_id.encode(), usedforsecurity=False
    ).hexdigest()[:8]
    replacements = {}
    for resource in resources:
        metadata = resource.setdefault("metadata", {})
        original_name = metadata["name"]
        replacements[original_name] = f"{original_name[:54]}-{digest}"
        metadata["name"] = replacements[original_name]
        metadata["namespace"] = namespace
    model_cache_pvc = request.config.getoption("--model-cache-pvc")
    if model_cache_pvc:
        replacements["model-cache"] = model_cache_pvc
    resources = _replace_resource_names(resources, replacements)
    dgd = _replace_resource_names(dgds[0], replacements)
    prepared_path = tmp_path / sweeper_deployment_target.yaml_path.name
    prepared_path.write_text(yaml.safe_dump(dgd, sort_keys=False))
    if not resources or not namespace:
        yield prepared_path
        return
    resource_yaml = yaml.safe_dump_all(resources, sort_keys=False)
    applied = kubectl("apply", "-n", namespace, "-f", "-", input_=resource_yaml)
    if applied.returncode != 0:
        pytest.fail(
            "failed to apply recipe resources: "
            f"{applied.stderr.strip() or applied.stdout.strip()}"
        )
    try:
        yield prepared_path
    finally:
        deleted = kubectl(
            "delete",
            "-n",
            namespace,
            "-f",
            "-",
            "--ignore-not-found",
            input_=resource_yaml,
        )
        if deleted.returncode != 0:
            pytest.fail(
                "failed to delete recipe resources: "
                f"{deleted.stderr.strip() or deleted.stdout.strip()}"
            )


@pytest.fixture
def cluster_result_recorder(
    request: pytest.FixtureRequest,
    sweeper_deployment_target: SweeperDeploymentTarget | None,
):
    """Append the pytest outcome to an existing optional JSON report."""
    started_at = time.monotonic()
    yield
    if sweeper_deployment_target is None:
        return
    report = request.node.stash.get(_deploy_test_call_report_key, None)
    if report is None:
        return
    if report.skipped:
        status = (
            "expected-failure"
            if getattr(report, "wasxfail", None) is not None
            else None
        )
        if status is None:
            return
    else:
        status = "failed" if report.failed else "passed"
    record_cluster_result(
        sweeper_deployment_target.output_dir,
        variant=sweeper_deployment_target.variant,
        status=status,
        duration_seconds=time.monotonic() - started_at,
        inventory=_cluster_inventory(),
    )


async def test_sweeper_generated_dgd(
    sweeper_deployment_target: SweeperDeploymentTarget | None,
    image: str | None,
    namespace: str,
    skip_service_restart: bool,
    request: pytest.FixtureRequest,
    cluster_result_recorder,
    prepared_deployment_path: Path | None,
) -> None:
    """Deploy one selected DGD and validate readiness and inference."""
    assert sweeper_deployment_target is not None
    if not namespace:
        pytest.skip("--namespace is required for live-cluster validation")
    assert prepared_deployment_path is not None
    _validate_cluster_hardware(sweeper_deployment_target, _cluster_inventory())
    deployment_spec = DeploymentSpec(str(prepared_deployment_path))
    deployment_spec.namespace = namespace
    digest = hashlib.sha1(
        sweeper_deployment_target.test_id.encode(), usedforsecurity=False
    ).hexdigest()[:8]
    deployment_spec.name = f"sweeper-{digest}"
    if image:
        deployment_spec.set_image(image)
    model_cache_pvc = request.config.getoption("--model-cache-pvc")
    if model_cache_pvc:
        deployment_spec.mount_model_cache_pvc(
            model_cache_pvc, request.config.getoption("--model-cache-mount")
        )
    target = DeploymentTarget(
        yaml_path=sweeper_deployment_target.yaml_path,
        framework=sweeper_deployment_target.backend,
        profile=sweeper_deployment_target.deployment_mode,
        source=f"sweeper:{sweeper_deployment_target.case_name}",
    )
    await run_dgd_deployment_test(
        target, deployment_spec, namespace, skip_service_restart, request
    )
    if sweeper_deployment_target.variant == "recipe":
        _write_recipe_discovery(sweeper_deployment_target)
