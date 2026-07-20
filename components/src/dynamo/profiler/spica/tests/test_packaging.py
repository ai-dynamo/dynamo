# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for the experimental Spica feature."""

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]


def test_spica_extra_matches_planner_image_requirements():
    project_text = (_REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r"(?ms)^spica = \[\n(.*?)^\]", project_text)
    assert match is not None
    spica_dependencies = set(ast.literal_eval(f"[{match.group(1)}]"))

    planner_dependencies = {
        line.strip()
        for line in (_REPO_ROOT / "container" / "deps" / "requirements.planner.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert spica_dependencies <= planner_dependencies


def test_spica_has_no_console_script():
    project_text = (_REPO_ROOT / "pyproject.toml").read_text()
    scripts = re.search(r"(?ms)^\[project\.scripts\]\s*(.*?)(?=^\[|\Z)", project_text)

    assert scripts is None or re.search(r"(?m)^spica\s*=", scripts.group(1)) is None


def test_spica_has_no_compatibility_alias_or_profiler_reexport():
    assert not (_REPO_ROOT / "components" / "src" / "spica").exists()

    profiler_init = (
        _REPO_ROOT / "components" / "src" / "dynamo" / "profiler" / "__init__.py"
    ).read_text()
    assert "spica" not in profiler_init


def test_runtime_wheel_enables_spica_replay_features():
    dockerfile = (
        _REPO_ROOT / "container" / "templates" / "wheel_builder.Dockerfile"
    ).read_text()

    required_features = "aic-forward-pass,mocker-kvbm-offload"
    assert dockerfile.count(required_features) == 2
