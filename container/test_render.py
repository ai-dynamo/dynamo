# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_RENDER_PATH = Path(__file__).with_name("render.py")
_RENDER_SPEC = importlib.util.spec_from_file_location(
    "dynamo_container_render", _RENDER_PATH
)
assert _RENDER_SPEC is not None and _RENDER_SPEC.loader is not None
_RENDER_MODULE = importlib.util.module_from_spec(_RENDER_SPEC)
_RENDER_SPEC.loader.exec_module(_RENDER_MODULE)
_inject_python_index_secrets = _RENDER_MODULE._inject_python_index_secrets

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "command",
    [
        "uv pip install maturin[patchelf]",
        "uv build --wheel",
        'git clone "https://github.com/ai-dynamo/nixl.git" && ninja',
    ],
)
def test_python_package_command_gets_optional_index_secrets(command):
    dockerfile = f"RUN {command}\n"

    rendered = _inject_python_index_secrets(dockerfile)

    assert "id=pip-index-url,env=PIP_INDEX_URL" in rendered
    assert "id=uv-default-index,env=UV_DEFAULT_INDEX" in rendered
    assert "id=pypi-netrc,target=/run/secrets/pypi-netrc" in rendered
    assert rendered.index("id=pypi-netrc") < rendered.index("export NETRC=")
    assert rendered.index("export NETRC=") < rendered.index(command)


def test_existing_mounts_stay_before_the_shell_command():
    dockerfile = (
        "RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \\\n"
        "    export UV_CACHE_DIR=/root/.cache/uv && \\\n"
        "    uv pip install aiofiles\n"
    )

    rendered = _inject_python_index_secrets(dockerfile)

    assert rendered.index("id=pypi-netrc") < rendered.index("type=cache")
    assert rendered.index("type=cache") < rendered.index("export NETRC=")
    assert rendered.index("export NETRC=") < rendered.index("export UV_CACHE_DIR")


def test_runner_retry_policy_overrides_template_default():
    dockerfile = "RUN export UV_HTTP_RETRIES=5 && \\\n" "    uv pip install aiofiles\n"

    rendered = _inject_python_index_secrets(dockerfile)

    assert "UV_HTTP_RETRIES=5" not in rendered
    assert rendered.count("UV_HTTP_RETRIES=10") == 2


def test_non_python_run_is_unchanged():
    dockerfile = "RUN cargo build --release\n"

    assert _inject_python_index_secrets(dockerfile) == dockerfile
