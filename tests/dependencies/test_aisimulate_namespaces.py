# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Dynamo consumes the canonical AISim wheel and namespaces."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from importlib import metadata, resources
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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
                if fullname.split('.')[0] in {'aisim', 'aisim_core'}:
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
            assert error.name == 'aisim_core'
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
    assert features["ais-forward-pass"] == ["dep:aisim-core"]
    assert "aic-forward-pass" not in features
    with (ROOT / "Cargo.toml").open("rb") as handle:
        canonical_core = tomllib.load(handle)["workspace"]["dependencies"]["aisim-core"]
    assert dependencies["aisim-core"] == {
        **canonical_core,
        "optional": True,
        "features": ["python"],
    }
    requirement = next(
        Requirement(item)
        for item in root_project["dependencies"]
        if canonicalize_name(Requirement(item).name) == "aisim"
    )
    if "git" in canonical_core:
        assert requirement.url == (
            f"git+{canonical_core['git']}.git@{canonical_core['rev']}"
            "#subdirectory=python/aisim"
        )
    else:
        assert (
            str(requirement.specifier).replace("==", "=").replace(".dev", "-dev.")
            == canonical_core["version"]
        )


def test_aisimulate_wheel_uses_canonical_import_namespaces() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISim supports Python 3.11 through 3.13")

    release = metadata.distribution("aisim")
    release_requirements = _requirement_names(release.requires or [])
    release_files = {str(path) for path in release.files or []}

    assert not (release_requirements & LEGACY_DISTRIBUTIONS)
    assert "aisim/__init__.py" in release_files
    assert "aisim_core/__init__.py" in release_files
    assert not any(
        path.split("/", 1)[0] in {"aiconfigurator", "aiconfigurator_core"}
        for path in release_files
    )

    from aisim.sdk.task_v2 import Task
    from aisim_core.sdk import RustForwardPassPerfModel

    assert Task is not None
    assert callable(RustForwardPassPerfModel.best_available)
    core_root = resources.files("aisim_core")
    assert core_root.joinpath("model_configs/Qwen--Qwen3-32B_config.json").is_file()
    assert core_root.joinpath("systems/h200_sxm.yaml").is_file()
    legacy_cli = next(
        entry for entry in release.entry_points if entry.name == "aiconfigurator"
    )
    assert legacy_cli.value == "aisim.legacy_cli.entrypoint:main"


def test_pinned_aisimulate_materializes_shared_host_offload() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISim supports Python 3.11 through 3.13")

    # Adapters consume the public launch contract; layout identity remains
    # owned by AISim rather than a private helper imported by Dynamo.
    from aisim.runner import materialize_engine_launch_config

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


@pytest.mark.parametrize("failure", [None, "source", "payload", "version"])
def test_staged_wheel_provenance_matches_reviewed_git_payload(
    tmp_path, monkeypatch, failure
):
    helper_path = ROOT / "container/aisim_source_provenance.py"
    helper_spec = importlib.util.spec_from_file_location(
        "aisim_source_provenance", helper_path
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    requirement_path = ROOT / "container/deps/requirements.aisim.txt"
    pin = Requirement(
        next(
            line
            for line in requirement_path.read_text().splitlines()
            if line.startswith("aisim @")
        )
    )
    commit = pin.url.split("@", 1)[1].split("#", 1)[0]
    origin = {
        "url": "https://github.com/ai-dynamo/aisimulate.git",
        "vcs_info": {"vcs": "git", "commit_id": commit},
        "subdirectory": "python/aisim",
    }
    if failure == "source":
        origin["vcs_info"]["commit_id"] = "0" * 40
    installed = tmp_path / "installed/aisim/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"canonical source")
    distribution = SimpleNamespace(
        version="0.14.0",
        read_text=lambda _: json.dumps(origin),
        locate_file=lambda name: tmp_path / "installed" / name,
    )
    monkeypatch.setattr(
        helper.importlib.metadata, "distribution", lambda _: distribution
    )
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "aisim-0.14.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "aisim/__init__.py",
            b"tampered" if failure == "payload" else installed.read_bytes(),
        )
        archive.writestr(
            "aisim-0.14.0.dist-info/METADATA",
            "Name: aisim\nVersion: "
            + ("0.13.0" if failure == "version" else "0.14.0")
            + "\n",
        )
    if failure is not None:
        with pytest.raises(
            ValueError,
            match={
                "source": "source differs",
                "payload": "payload differs",
                "version": "metadata differs",
            }[failure],
        ):
            helper.write_provenance(requirement_path, wheelhouse)
        assert not (wheelhouse / "aisim-source.json").exists()
        assert not (wheelhouse / "aisim-override.txt").exists()
    else:
        helper.write_provenance(requirement_path, wheelhouse)
        proof = json.loads((wheelhouse / "aisim-source.json").read_text())
        assert (wheelhouse / "aisim-override.txt").read_text() == (
            f"aisim @ file:///opt/dynamo/wheelhouse/{wheel.name}\n"
        )
        assert proof["source"] == origin
        assert proof["wheel"] == wheel.name
        assert proof["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
