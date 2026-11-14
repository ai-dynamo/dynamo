# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Dynamo on one published AISimulate release."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import textwrap
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from tests.wheels.smoke_install import AISIMULATE_FIND_LINKS

pytestmark = [
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

ROOT = Path(__file__).resolve().parents[2]
AISIMULATE_REQUIREMENTS = ROOT / "container/deps/requirements.aisimulate.txt"
LOCKFILES = (
    ROOT / "Cargo.lock",
    ROOT / "lib/bindings/python/Cargo.lock",
    ROOT / "lib/bindings/kvbm/Cargo.lock",
)


def _root_configs() -> tuple[dict, dict]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    with (ROOT / "Cargo.toml").open("rb") as handle:
        cargo = tomllib.load(handle)
    return pyproject, cargo


def _python_requirement(pyproject: dict, *, extra: str | None = None) -> Requirement:
    requirements = (
        pyproject["project"]["dependencies"]
        if extra is None
        else pyproject["project"]["optional-dependencies"][extra]
    )
    matches = [
        Requirement(requirement)
        for requirement in requirements
        if canonicalize_name(Requirement(requirement).name) == "aisimulate"
    ]
    assert len(matches) == 1, "ai-dynamo must declare one AISimulate dependency"
    return matches[0]


def _requirements_file_aisimulate_requirement(path: Path) -> Requirement:
    matches: list[Requirement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith("--"):
            continue
        parsed = Requirement(requirement)
        if canonicalize_name(parsed.name) == "aisimulate":
            matches.append(parsed)
    assert len(matches) == 1, f"{path} must declare one AISimulate dependency"
    return matches[0]


def _exact_version(requirement: Requirement) -> Version:
    assert requirement.url is None, "AISimulate must resolve from PyPI"
    specifiers = list(requirement.specifier)
    assert (
        len(specifiers) == 1 and specifiers[0].operator == "=="
    ), "AISimulate must use one exact PyPI version"
    return Version(specifiers[0].version)


def _locked_cargo_version(path: Path) -> Version:
    with path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    matches = [package for package in packages if package["name"] == "aisimulate-core"]
    assert len(matches) == 1, f"expected one aisimulate-core package in {path}"

    package = matches[0]
    assert package.get("source") == (
        "registry+https://github.com/rust-lang/crates.io-index"
    ), f"aisimulate-core must resolve from crates.io in {path}"
    assert re.fullmatch(
        r"[0-9a-f]{64}", str(package.get("checksum", ""))
    ), f"aisimulate-core must have a registry checksum in {path}"
    return Version(str(package["version"]))


def test_dynamo_pins_matching_published_aisimulate_releases() -> None:
    pyproject, cargo = _root_configs()
    python_requirement = _python_requirement(pyproject, extra="ais")
    python_version = _exact_version(python_requirement)
    container_requirement = _requirements_file_aisimulate_requirement(
        AISIMULATE_REQUIREMENTS
    )

    assert python_requirement.marker is not None
    environment = default_environment()
    environment["python_version"] = "3.10"
    assert not python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.11"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.12"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.13"
    assert python_requirement.marker.evaluate(environment)
    environment["python_version"] = "3.14"
    assert not python_requirement.marker.evaluate(environment)
    assert container_requirement.marker is None
    assert _exact_version(container_requirement) == python_version
    with (ROOT / "benchmarks/pyproject.toml").open("rb") as handle:
        benchmarks = tomllib.load(handle)
    assert _exact_version(_python_requirement(benchmarks)) == python_version
    assert "aisimulate" not in pyproject.get("tool", {}).get("uv", {}).get(
        "sources", {}
    )

    cargo_dependency = cargo["workspace"]["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & cargo_dependency.keys()
    cargo_requirement = str(cargo_dependency["version"])
    assert cargo_requirement.startswith(
        "="
    ), "aisimulate-core must use one exact crates.io version"
    cargo_version = Version(cargo_requirement.removeprefix("="))

    with (ROOT / "lib/bindings/python/Cargo.toml").open("rb") as handle:
        bindings = tomllib.load(handle)
    bindings_dependency = bindings["dependencies"]["aisimulate-core"]
    assert not {"path", "git", "rev", "branch", "tag"} & bindings_dependency.keys()
    assert bindings_dependency["version"] == cargo_requirement

    assert cargo_version == python_version
    assert all(_locked_cargo_version(path) == cargo_version for path in LOCKFILES)


def test_container_stages_the_published_aisimulate_wheel() -> None:
    pyproject, _ = _root_configs()
    python_version = _exact_version(_python_requirement(pyproject, extra="ais"))
    container_version = _exact_version(
        _requirements_file_aisimulate_requirement(AISIMULATE_REQUIREMENTS)
    )
    wheel_builder = (ROOT / "container/templates/wheel_builder.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert container_version == python_version
    assert "requirements.aisimulate.txt" in wheel_builder
    assert (
        "--requirement /opt/dynamo/container/deps/requirements.aisimulate.txt"
        in wheel_builder
    )
    assert "--only-binary=:all:" in wheel_builder
    assert "--no-deps" in wheel_builder
    assert "--no-index" in wheel_builder
    assert AISIMULATE_FIND_LINKS == "https://pypi.nvidia.com/aisimulate/"
    assert f"--find-links {AISIMULATE_FIND_LINKS}" in wheel_builder
    assert "COPY aisimulate" not in wheel_builder
    assert "/opt/dynamo/aisimulate" not in wheel_builder
    assert not (ROOT / "aisimulate").exists()
    planner_template = (ROOT / "container/templates/planner.Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "/opt/dynamo/wheelhouse/aisimulate*.whl" in planner_template


def test_planner_ci_image_collects_unified_cli_e2e_tests() -> None:
    planner_dockerfile = ROOT / "container/templates/planner.Dockerfile"
    if not planner_dockerfile.is_file():
        pytest.skip("planner Dockerfile is not staged in this component image")
    planner_template = planner_dockerfile.read_text(encoding="utf-8")

    assert "components/src/dynamo/replay/tests/e2e" in planner_template
    assert "components/src/dynamo/replay/tests/test_main.py" not in planner_template


@pytest.mark.parametrize(
    "required,wheel_present,success",
    [("0", True, True), ("1", True, True), ("true", True, False), ("1", False, True)],
)
def test_ci_ais_install_arguments_are_explicit(
    tmp_path: Path, required: str, wheel_present: bool, success: bool
) -> None:
    dockerfile = (ROOT / "container/Dockerfile.test").read_text(encoding="utf-8")
    start = dockerfile.index('    case "$DYNAMO_TEST_REQUIRE_AIS_RUNTIME"')
    end = dockerfile.index("    check_python=python3", start)
    # Exercise the actual Docker RUN argument construction without pip, a
    # package import, or a network request. Invalid modes fail; the final
    # Planner image has no wheelhouse but must still require its exact AIS pin.
    script = dockerfile[start:end].replace("\\\n", "")
    script = script.replace("/opt/dynamo/wheelhouse", shlex.quote(str(tmp_path)))
    script += "printf '%s\\n' \"$@\""
    wheel = tmp_path / "ai_dynamo-1.7.0-py3-none-any.whl"
    if wheel_present:
        wheel.write_bytes(b"argument-construction fixture; never installed")
    result = subprocess.run(
        ["bash", "-eu", "-c", script],
        env={**os.environ, "DYNAMO_TEST_REQUIRE_AIS_RUNTIME": required},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is success, result.stderr
    if success:
        arguments = result.stdout.splitlines()
        assert arguments[:2] == ["--requirement", "/tmp/requirements.test.txt"]
        if required == "1":
            assert "/tmp/requirements.aisimulate.txt" in arguments
            assert (str(wheel) + "[ais]" in arguments) is wheel_present
        else:
            assert not any(
                "aisimulate" in value or "[ais]" in value for value in arguments
            )


def test_ci_requires_ais_in_its_owning_lanes() -> None:
    dockerfile = (ROOT / "container/Dockerfile.test").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/shared-build-image.yml").read_text(
        encoding="utf-8"
    )
    assert "ARG DYNAMO_TEST_REQUIRE_AIS_RUNTIME=0" in dockerfile
    planner_stage = dockerfile.split("FROM test_distroless AS test_planner\n", 1)[1]
    assert 'RUN test "$DYNAMO_TEST_REQUIRE_AIS_RUNTIME" = 1' in planner_stage
    assert (
        "TEST_REQUIRE_AIS_RUNTIME: ${{ inputs.framework == 'dynamo' && "
        "(inputs.target == 'runtime' || inputs.target == 'planner') && '1' || '0' }}"
    ) in workflow
    assert (
        '--build-arg DYNAMO_TEST_REQUIRE_AIS_RUNTIME="${TEST_REQUIRE_AIS_RUNTIME}"'
        in workflow
    )


@pytest.mark.parametrize(
    "relative", ["container", "container/deps", "container/templates"]
)
def test_ci_metadata_directories_are_readable_and_traversable(relative: str) -> None:
    directory = ROOT / relative
    assert directory.is_dir()
    # The mode check also catches the regression when a privileged test process
    # could bypass it. os.access verifies the actual CI runner's access too.
    assert directory.stat().st_mode & 0o555 == 0o555, directory
    assert os.access(directory, os.R_OK | os.X_OK), directory
    for name in directory.iterdir():
        if name.is_file():
            assert os.access(name, os.R_OK), name


@pytest.fixture
def installed_aisimulate() -> metadata.Distribution:
    """Set DYNAMO_TEST_REQUIRE_AIS_RUNTIME=1 for a required opt-in installation."""
    try:
        return metadata.distribution("aisimulate")
    except metadata.PackageNotFoundError:
        if os.environ.get("DYNAMO_TEST_REQUIRE_AIS_RUNTIME") == "1":
            pytest.fail(
                "The requested ai-dynamo[ais] test environment lacks AISimulate"
            )
        pytest.skip("AIS installation check requires the ai-dynamo[ais] environment")


def test_installed_aisimulate_matches_the_declared_release(
    installed_aisimulate: metadata.Distribution,
) -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        pytest.skip("AISimulate supports Python 3.11 through 3.13")
    pyproject, _ = _root_configs()
    expected = _exact_version(_python_requirement(pyproject, extra="ais"))

    assert Version(installed_aisimulate.version) == expected


def test_ai_dynamo_registers_only_its_aisimulate_providers() -> None:
    pyproject, _ = _root_configs()
    project = pyproject["project"]

    extras = set(project.get("optional-dependencies", {}))
    assert {"sweeper", "simulate", "simulation"}.isdisjoint(extras)
    assert project["entry-points"]["aisimulate.sweep_config_providers"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }
    assert project["entry-points"]["aisimulate.runner_factories"] == {
        "dynamo": "dynamo.replay.simulation:DynamoReplayRunnerFactory"
    }
    assert project["entry-points"]["aisimulate.config_adapters"] == {
        "dynamo.planner": "dynamo.planner.simulation:create_provider",
        "dynamo.router": "dynamo.router.simulation:create_provider",
    }


def test_base_ai_dynamo_metadata_does_not_activate_offline_ais() -> None:
    """Check the installed wheel metadata, not only the source declaration."""
    requirements = [
        Requirement(raw)
        for raw in metadata.requires("ai-dynamo") or []
        if canonicalize_name(Requirement(raw).name) == "aisimulate"
    ]
    assert len(requirements) == 1, "one optional AIS requirement must remain published"
    requirement = requirements[0]
    assert requirement.marker is not None
    pyproject, _ = _root_configs()
    assert not any(
        canonicalize_name(Requirement(raw).name) == "aisimulate"
        for raw in pyproject["project"]["dependencies"]
    )
    assert _exact_version(requirement) == _exact_version(
        _python_requirement(pyproject, extra="ais")
    )
    environment = default_environment()
    for python_version in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        environment.update(python_version=python_version, extra="")
        assert not requirement.marker.evaluate(environment)
        environment["extra"] = "ais"
        assert requirement.marker.evaluate(environment) is (
            python_version in {"3.11", "3.12", "3.13"}
        )


@pytest.mark.parametrize(
    ("action", "expected_output"),
    [
        (
            "sys.argv = ['dynamo.frontend', '--help']\n"
            "runpy.run_module('dynamo.frontend', run_name='__main__')\n",
            "--http-port",
        ),
        (
            "from dynamo.frontend.frontend_args import FrontendConfig, build_frontend_parser\n"
            "parser, _ = build_frontend_parser({})\n"
            "config = FrontendConfig.from_cli_args(parser.parse_args([]))\n"
            "config.validate()\n"
            "assert config.router_prefill_load_model != 'ais'\n"
            "print('Base frontend configuration validates without AIS')\n",
            "Base frontend configuration validates without AIS",
        ),
        (
            "import argparse\n"
            "from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig\n"
            "parser = argparse.ArgumentParser()\n"
            "FrontendArgGroup().add_arguments(parser)\n"
            "args = parser.parse_args([\n"
            "    '--router-mode', 'kv', '--router-prefill-load-model', 'ais',\n"
            "    '--dyn-chat-processor', 'dynamo', '--router-track-prefill-tokens',\n"
            "    '--ais-model-path', 'fixture-model', '--ais-system', 'fixture-system',\n"
            "    '--ais-backend', 'trtllm',\n"
            "])\n"
            "config = FrontendConfig.from_cli_args(args)\n"
            "try:\n"
            "    config.validate()\n"
            "except ModuleNotFoundError as error:\n"
            "    assert error.name == 'aisimulate_core'\n"
            "    print('Explicit AIS mode requires the AIS runtime')\n"
            "else:\n"
            "    raise AssertionError('Explicit AIS mode silently lost its runtime')\n",
            "Explicit AIS mode requires the AIS runtime",
        ),
    ],
    ids=["frontend-help", "frontend-config", "requested-ais"],
)
@pytest.mark.timeout(30)
def test_optional_ais_boundaries(action: str, expected_output: str) -> None:
    script = (
        textwrap.dedent(
            """
        import importlib.abc
        import runpy
        import sys

        blocked = {'aisimulate', 'aisimulate_core'}
        assert not any(name.split('.')[0] in blocked for name in sys.modules)

        class WithoutAIS(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError('Optional AIS runtime is unavailable', name=fullname)

        sys.meta_path.insert(0, WithoutAIS())
        """
        )
        + action
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_output in result.stdout
