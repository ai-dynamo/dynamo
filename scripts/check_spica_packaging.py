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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _planner_requirements() -> set[str]:
    requirements = REPO_ROOT / "container/deps/requirements.planner.txt"
    return {
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _check() -> list[str]:
    errors: list[str] = []
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)["project"]

    spica = set(project["optional-dependencies"]["spica"])
    planner = _planner_requirements()
    missing = sorted(spica - planner)
    if missing:
        errors.append(f"Spica extra missing from planner requirements: {missing}")

    predictor_dependencies = {
        "filterpy==1.4.5",
        "pmdarima==2.1.1",
        "prophet==1.2.1",
        "scikit-learn==1.7.2",
        "scipy>=1.14.0,<2.0",
    }
    missing_predictors = sorted(predictor_dependencies - spica)
    if missing_predictors:
        errors.append(
            f"Spica extra missing predictor dependencies: {missing_predictors}"
        )

    combined = spica | planner
    if any(item.startswith("tensorflow-probability") for item in combined):
        errors.append(
            "Spica must use one tensorflow_probability provider; remove tensorflow-probability"
        )
    nightly = "tfp-nightly[jax]==0.26.0.dev20260626; python_version < '3.13'"
    if nightly not in spica or nightly not in planner:
        errors.append(
            "Spica extra and planner requirements must pin the same tfp-nightly[jax]"
        )

    scripts = project.get("scripts", {})
    if "spica" in scripts:
        errors.append("Spica must not expose a console script")
    if (REPO_ROOT / "components/src/spica").exists():
        errors.append("Spica must not expose a top-level compatibility package")

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
