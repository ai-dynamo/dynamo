# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render gates for the release/1.3 SGLang EFA image composition."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
    pytest.mark.sglang,
    pytest.mark.core,
]

_REPO = Path(__file__).resolve().parents[2]
_CONTAINER = _REPO / "container"


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "dynamo_container_render", _CONTAINER / "render.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(
    framework: str,
    *,
    cuda_version: str = "13.0",
    make_efa: bool,
    platform: str = "multi",
) -> str:
    renderer = _load_renderer()
    context = yaml.safe_load((_CONTAINER / "context.yaml").read_text())
    args = SimpleNamespace(
        framework=framework,
        device="cuda",
        target="runtime",
        platform=platform,
        cuda_version=cuda_version,
        make_efa=make_efa,
    )
    template = renderer._make_jinja_env(_CONTAINER).get_template("Dockerfile.template")
    rendered = template.render(
        context=context,
        **renderer._render_context(args, context),
    )
    return re.sub(r"\n{3,}", "\n\n", rendered)


@pytest.mark.parametrize("platform", ["amd64", "arm64", "multi"])
def test_release13_sglang_efa_stack_is_pinned(platform: str):
    rendered = _render("sglang", make_efa=True, platform=platform)

    for required in (
        "ARG EFA_VERSION=1.49.0",
        "cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309",
        "./efa_installer.sh -y --build-ngc",
        "--skip-mpi --skip-plugin --no-verify",
        'libfabric1-aws)" = "2.4.0amzn5.0"',
        "5d5f48961cf9b30def1c8cdb0961b0683f0102ca9898641ef90f2d205379a5fd",
        "24291af4829bcdddc2d47d7eb35366c19246170091202b93814fda9b67f32248",
        "0f3c0aec918502aaad4343b99f1cc94cea1e85e0",
        "d1e60ec95140ff018a1db6057e3f51dfa4c0f562",
        "659b5b21f4d0d7ad0d79c54a8bd47be642ca21a2",
        "d605856c2409b5a2204a780ca4e1738445a0496b68a52d83f7f5350403687524",
        "validate_pr1966_semantics.py",
        ".nv_fatbin",
        "FROM aws_base AS sglang_nixl_efa_builder",
        "FROM aws_base AS aws",
        "ENV LD_PRELOAD=/opt/amazon/efa/lib/libfabric.so.1",
    ):
        assert required in rendered

    assert "libfabric overlay:" not in rendered
    assert "libfabric.so.1.31.1 =>" not in rendered


def test_non_efa_sglang_render_has_no_release_overlay():
    rendered = _render("sglang", make_efa=False, platform="amd64")

    assert "aws-efa-installer" not in rendered
    assert "nixl-pr1966" not in rendered
    assert "sglang_nixl_efa_builder" not in rendered


@pytest.mark.parametrize(
    ("framework", "cuda_version"),
    [("vllm", "13.0"), ("trtllm", "13.1")],
)
def test_other_efa_frameworks_keep_the_generic_stack(framework: str, cuda_version: str):
    rendered = _render(framework, cuda_version=cuda_version, make_efa=True)

    assert "ARG EFA_VERSION=1.47.0" in rendered
    assert "libfabric overlay:" in rendered
    assert "FROM ${EFA_BASE_IMAGE} AS aws" in rendered
    assert "nixl-pr1966" not in rendered
    assert "sglang_nixl_efa_builder" not in rendered
