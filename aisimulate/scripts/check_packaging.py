#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate source-level packaging contracts for experimental Spica."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
AISIMULATE_ROOT = REPO_ROOT / "aisimulate"


def _project(path: Path) -> dict:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def _check() -> list[str]:
    errors: list[str] = []
    dynamo = _project(REPO_ROOT / "pyproject.toml")
    aisimulate = _project(AISIMULATE_ROOT / "pyproject.toml")

    if "spica" in dynamo.get("optional-dependencies", {}):
        errors.append("ai-dynamo must not expose a Spica optional extra")

    if aisimulate.get("name") != "aisimulate":
        errors.append("the standalone distribution must be named aisimulate")

    with (AISIMULATE_ROOT / "pyproject.toml").open("rb") as project_file:
        aisimulate_config = tomllib.load(project_file)
    wheel_packages = (
        aisimulate_config.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages")
    )
    if wheel_packages != ["spica"]:
        errors.append("the aisimulate wheel must publish canonical package spica")

    dependencies = set(aisimulate.get("dependencies", []))
    required = {
        "ai-dynamo>=1.3.0,<2.0.0",
        "aiconfigurator>=0.9.0,<0.10.0",
        "chex==0.1.87",
        "filterpy==1.4.5",
        "flax==0.10.0",
        "google-vizier[jax]==0.1.21",
        "jax==0.4.38",
        "jaxlib==0.4.38",
        "optax==0.2.4",
        "pmdarima==2.1.1",
        "prometheus-api-client==0.6.0",
        "prophet==1.2.1",
        "scikit-learn==1.7.2",
        "scipy>=1.14.0,<2.0",
        "tfp-nightly[jax]==0.26.0.dev20260626",
    }
    missing = sorted(required - dependencies)
    if missing:
        errors.append(f"aisimulate is missing runtime dependencies: {missing}")
    if any(item.startswith("tensorflow-probability") for item in dependencies):
        errors.append(
            "Spica must use one tensorflow_probability provider; "
            "remove tensorflow-probability"
        )

    if aisimulate.get("scripts"):
        errors.append("aisimulate must not expose console scripts")
    if not (AISIMULATE_ROOT / "spica/__main__.py").is_file():
        errors.append("canonical `python -m spica` entry point is missing")
    if (REPO_ROOT / "components/src/dynamo/profiler/spica").exists():
        errors.append("Spica source must not remain under dynamo.profiler")

    profiler_init = (
        REPO_ROOT / "components/src/dynamo/profiler/__init__.py"
    ).read_text()
    if "spica" in profiler_init:
        errors.append("dynamo.profiler must not re-export Spica")

    wheel_template = (
        REPO_ROOT / "container/templates/wheel_builder.Dockerfile"
    ).read_text()
    planner_feature = '{% if target == "planner" %},mocker-kvbm-offload{% endif %}'
    if wheel_template.count(planner_feature) != 2:
        errors.append(
            "mocker-kvbm-offload must be enabled only for both planner wheel build paths"
        )
    if (
        "uv build --wheel --out-dir /opt/dynamo/dist /opt/dynamo/aisimulate"
        not in wheel_template
    ):
        errors.append(
            "the planner wheel builder must build the aisimulate distribution"
        )

    planner_template = (
        REPO_ROOT / "container/templates/planner.Dockerfile"
    ).read_text()
    if "/opt/dynamo/wheelhouse/aisimulate*.whl" not in planner_template:
        errors.append("the planner image must install the aisimulate wheel")
    if "aisimulate /workspace/aisimulate" not in planner_template:
        errors.append("the planner image must copy aisimulate tests and examples")

    return errors


def main() -> int:
    errors = _check()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Spica packaging contracts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
