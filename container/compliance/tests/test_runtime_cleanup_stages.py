# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compliance corrections must describe the final image, never a cleaned scanner copy."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "compliance_test_renderer", ROOT / "container/render.py"
)
render = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render)
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]


def rendered(framework, target="runtime", efa=False):
    context = yaml.safe_load((ROOT / "container/context.yaml").read_text())
    args = SimpleNamespace(
        framework=framework,
        target=target,
        device="cuda",
        cuda_version="13.1" if framework == "trtllm" else "13.0",
        platform="linux/amd64,linux/arm64",
        make_efa=efa,
    )
    return (
        render._make_jinja_env(ROOT / "container")
        .get_template("Dockerfile.template")
        .render(context=context, **render._render_context(args, context))
    )


@pytest.mark.parametrize(
    "framework,target,efa",
    [
        ("vllm", "runtime", False),
        ("sglang", "runtime", False),
        ("trtllm", "runtime", False),
        ("dynamo", "frontend", False),
        ("vllm", "runtime", True),
        ("sglang", "runtime", True),
        ("trtllm", "runtime", True),
    ],
)
def test_packaging_cleanup_changes_a_runtime_ancestor(framework, target, efa):
    text = rendered(framework, target, efa)
    stage = None
    cleanup_stages = []
    for line in text.splitlines():
        if line.startswith("FROM "):
            stage = line.rsplit(" AS ", 1)[-1].strip()
        if "python3 -m compliance.trim_packaging_docs" in line:
            cleanup_stages.append(stage)
    expected = "pre_frontend" if target == "frontend" else "pre_runtime"
    assert expected in cleanup_stages
    assert set(cleanup_stages) <= {expected, "aws"}
    context = yaml.safe_load((ROOT / "container/context.yaml").read_text())
    stem = (
        context["dynamo"]["frontend_baseline_sbom"]
        if target == "frontend"
        else context[framework]["cuda13.1" if framework == "trtllm" else "cuda13.0"][
            "baseline_sbom"
        ]
    )
    argument = f"--baseline-sbom /tmp/compliance/compliance/base_sboms/{stem}-${{TARGETARCH}}.cdx.json"
    assert text.count(argument) == len(cleanup_stages)
    if efa:
        assert "aws" in cleanup_stages
    assert f"FROM {expected} AS " in text


def test_graphics_cleanup_is_after_headless_rebuild_and_runtime_only():
    text = rendered("vllm")
    assert text.index("-DWITH_OPENGL=OFF") < text.index("--require-headless-opencv")
    assert "--package libglvnd0" not in text
    assert "--require-headless-opencv" not in rendered("vllm", "dev")
