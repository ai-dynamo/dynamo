# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from render import _inject_python_index_secrets

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_python_install_gets_optional_index_secrets():
    dockerfile = "RUN uv pip install maturin[patchelf]\n"

    rendered = _inject_python_index_secrets(dockerfile)

    assert "id=pip-index-url,env=PIP_INDEX_URL" in rendered
    assert "id=uv-default-index,env=UV_DEFAULT_INDEX" in rendered
    assert "id=pypi-netrc,target=/run/secrets/pypi-netrc" in rendered
    assert rendered.index("id=pypi-netrc") < rendered.index("export NETRC=")
    assert rendered.index("export NETRC=") < rendered.index("uv pip install")


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


def test_non_python_run_is_unchanged():
    dockerfile = "RUN cargo build --release\n"

    assert _inject_python_index_secrets(dockerfile) == dockerfile
