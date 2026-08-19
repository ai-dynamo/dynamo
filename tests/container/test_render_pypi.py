# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_RENDER_PATH = _REPO_ROOT / "container" / "render.py"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _load_render_module():
    spec = importlib.util.spec_from_file_location(
        "dynamo_container_render", _RENDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_index_environment_follows_interleaved_mount_comments():
    dockerfile = (
        "RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \\\n"
        "    # Keep the package cache shared across builds.\n"
        "    --mount=type=cache,target=/root/.cache/uv \\\n"
        "    uv pip install --requirement /tmp/requirements.txt\n"
    )

    rendered = _load_render_module()._inject_python_index_mounts(dockerfile)

    assert rendered.index("id=pypi-netrc") < rendered.index("type=bind")
    assert rendered.index("type=bind") < rendered.index("type=cache")
    assert rendered.index("type=cache") < rendered.index("export NETRC=")
    assert rendered.index("export NETRC=") < rendered.index("uv pip install")


def test_python_index_mounts_ignore_full_line_comments():
    dockerfile = (
        "RUN echo no-package-download \\\n"
        "    # uv pip install --requirement /tmp/requirements.txt\n"
        "    && true\n"
    )

    rendered = _load_render_module()._inject_python_index_mounts(dockerfile)

    assert rendered == dockerfile


def test_vllm_omni_installer_gets_python_index_mounts():
    template = (
        _REPO_ROOT / "container" / "templates" / "vllm_runtime.Dockerfile"
    ).read_text()
    instruction = next(
        item
        for item in re.split(r"(?=^[A-Z]+\b)", template, flags=re.MULTILINE)
        if "bash /tmp/install_vllm_omni.sh" in item
    )

    rendered = _load_render_module()._inject_python_index_mounts(instruction)

    assert "id=pip-index-url,env=PIP_INDEX_URL" in rendered
    assert "id=uv-default-index,env=UV_DEFAULT_INDEX" in rendered
    assert "id=pypi-netrc,target=/run/secrets/pypi-netrc" in rendered


def test_compliance_extract_forwards_python_index_secrets():
    action = (
        _REPO_ROOT / ".github" / "actions" / "compliance-extract" / "action.yml"
    ).read_text()

    assert action.count("--secret id=pip-index-url,env=PIP_INDEX_URL") == 2
    assert action.count("--secret id=uv-default-index,env=UV_DEFAULT_INDEX") == 2
    assert action.count("--secret id=pypi-netrc,src=${NETRC}") == 2
