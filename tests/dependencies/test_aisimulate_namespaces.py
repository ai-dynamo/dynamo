# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Dynamo consumes the canonical AISimulate wheel and namespaces."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from importlib import metadata, resources
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.aiconfigurator,
]

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DISTRIBUTIONS = {"aiconfigurator", "aiconfigurator-core"}
CARGO_LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


@pytest.mark.timeout(30)
def test_operator_schemas_do_not_load_aisimulate_runtime() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import runpy
        import sys

        class WithoutAIS(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'aisimulate', 'aisimulate_core'}:
                    raise ModuleNotFoundError('AIS runtime is unavailable', name=fullname)

        sys.meta_path.insert(0, WithoutAIS())
        namespace = runpy.run_path(sys.argv[1], run_name='__main__')
        try:
            namespace['PlannerConfig'](
                mode='decode',
                ais_perf_model={'roles': {'decode': {
                    'model': 'model', 'system': 'system', 'backend': 'vllm',
                }}},
            )
        except ModuleNotFoundError as error:
            assert error.name == 'aisimulate_core'
        else:
            raise AssertionError('Explicit AIS configuration requires its runtime')
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "deploy/operator/api/scripts/validate_pydantic_models.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _requirement_names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(item).name) for item in requirements}


def _requirements_file_names(path: Path) -> set[str]:
    requirements = [
        line.split("#", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    return _requirement_names(requirements)


def test_no_manifest_installs_retired_aic_distributions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        root_project = tomllib.load(handle)["project"]
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmark_project = tomllib.load(handle)["project"]
    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings_cargo = tomllib.load(handle)
    requirement_sets = [
        (
            "pyproject.toml project.dependencies",
            _requirement_names(root_project["dependencies"]),
        ),
        *(
            (
                f"pyproject.toml project.optional-dependencies.{name}",
                _requirement_names(requirements),
            )
            for name, requirements in root_project["optional-dependencies"].items()
        ),
        (
            "benchmarks/pyproject.toml project.dependencies",
            _requirement_names(benchmark_project["dependencies"]),
        ),
        (
            "container/deps/requirements.frontend.txt",
            _requirements_file_names(ROOT / "container/deps/requirements.frontend.txt"),
        ),
        (
            "container/deps/requirements.planner.txt",
            _requirements_file_names(ROOT / "container/deps/requirements.planner.txt"),
        ),
    ]
    for label, names in requirement_sets:
        retired = names & LEGACY_DISTRIBUTIONS
        assert not retired, f"{label} installs retired distributions: {sorted(retired)}"

    features = bindings_cargo["features"]
    dependencies = bindings_cargo["dependencies"]
    assert "aiconfigurator-core" not in dependencies
    for lockfile in CARGO_LOCKFILES:
        with lockfile.open("rb") as handle:
            packages = tomllib.load(handle)["package"]
        assert all(package["name"] != "aiconfigurator-core" for package in packages)
    assert features["ais-forward-pass"] == ["dep:aisimulate-core"]
    assert "aic-forward-pass" not in features
    with (ROOT / "Cargo.toml").open("rb") as handle:
        canonical_core = tomllib.load(handle)["workspace"]["dependencies"][
            "aisimulate-core"
        ]
    assert dependencies["aisimulate-core"] == {
        **canonical_core,
        "optional": True,
        "features": ["python"],
    }
    requirement = next(
        Requirement(item)
        for item in root_project["dependencies"]
        if canonicalize_name(Requirement(item).name) == "aisimulate"
    )
    if "git" in canonical_core:
        assert requirement.url == (
            f"git+{canonical_core['git']}@{canonical_core['rev']}"
            "#subdirectory=python/aisimulate"
        )
    else:
        assert (
            str(requirement.specifier).replace("==", "=").replace(".dev", "-dev.")
            == canonical_core["version"]
        )


def test_aisimulate_wheel_uses_canonical_import_namespaces() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")

    release = metadata.distribution("aisimulate")
    release_requirements = _requirement_names(release.requires or [])
    release_files = {str(path) for path in release.files or []}

    assert not (release_requirements & LEGACY_DISTRIBUTIONS)
    assert "aisimulate/__init__.py" in release_files
    assert "aisimulate_core/__init__.py" in release_files
    assert not any(
        path.split("/", 1)[0] in {"aiconfigurator", "aiconfigurator_core"}
        for path in release_files
    )

    from aisimulate.sdk.task_v2 import Task
    from aisimulate_core.sdk import RustForwardPassPerfModel

    assert Task is not None
    assert callable(RustForwardPassPerfModel.best_available)
    core_root = resources.files("aisimulate_core")
    assert core_root.joinpath("model_configs/Qwen--Qwen3-32B_config.json").is_file()
    assert core_root.joinpath("systems/h200_sxm.yaml").is_file()
    legacy_cli = next(
        entry for entry in release.entry_points if entry.name == "aiconfigurator"
    )
    assert legacy_cli.value == "aisimulate.legacy_cli.entrypoint:main"


def test_pinned_aisimulate_materializes_shared_host_offload() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")

    # Adapters consume the public launch contract; layout identity remains
    # owned by AISimulate rather than a private helper imported by Dynamo.
    from aisimulate.runner import materialize_engine_launch_config

    config = materialize_engine_launch_config(
        "vllm",
        "",
        {},
        {
            "aic_model_path": "resolved-model",
            "num_gpu_blocks": 4,
            "tensor_parallel_size": 2,
            "kv_cache_bytes_per_token": 1024,
            "kv_bytes_per_token": 4096,
            "native_host_offload": {
                "scope": "cluster_shared",
                "num_host_blocks": 8,
                "h2d_bandwidth_gbps": 7.0,
            },
            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        },
        "aggregated",
    )
    engine = config["engine"]
    host = engine["native_host_offload"]
    assert config["tensor_parallel_size"] == 2
    assert engine["kv_cache_bytes_per_token"] == 1024
    assert engine["kv_transfer_bytes_per_token"] == 4096
    assert host["scope"] == "cluster_shared"
    assert host["num_host_blocks"] == 8
    assert host["h2d_bandwidth_gbps"] == 7.0
    assert json.loads(host["kv_layout_id"]) == {
        "model": "resolved-model",
        "backend": "vllm",
        "tp": 2,
        "block_size": engine["block_size"],
        "bytes_per_token": 1024,
    }
