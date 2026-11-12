# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EFA package installation must precede the inventory copied into its image."""

import re
from argparse import Namespace
from pathlib import Path

import pytest
import render as container_render
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _render(framework, target="runtime", make_efa=True, platform="multi"):
    directory = Path(__file__).resolve().parents[2]
    context = yaml.safe_load((directory / "context.yaml").read_text())
    args = Namespace(
        framework=framework,
        target=target,
        make_efa=make_efa,
        device="cuda",
        cuda_version="13.1" if framework == "trtllm" else "13.0",
        platform=platform,
    )
    return (
        container_render._make_jinja_env(directory)
        .get_template("Dockerfile.template")
        .render(context=context, **container_render._render_context(args, context))
    )


def _stage_graph(rendered):
    """Read FROM/COPY/bind-mount edges from the actual rendered Dockerfile."""
    stages, parents, defaults = {}, {}, {}
    current = None
    for line in rendered.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        arg = re.match(r"ARG ([A-Z_]+)=(.*)", line)
        if arg and current is None:
            defaults[arg[1]] = arg[2].strip('"')
        stage = re.match(r"FROM (?:--platform=\S+ )?(\S+) AS (\S+)", line)
        if stage:
            base, current = stage.groups()
            assert current not in stages, f"Duplicate stage: {current}"
            stages[current] = []
            parents[current] = {
                re.sub(r"\$\{([^}]+)\}", lambda m: defaults.get(m[1], m[0]), base)
            }
        elif current:
            stages[current].append(line)
            parents[current].update(re.findall(r"--from=([^\s]+)", line))
            parents[current].update(re.findall(r",from=([^,\s]+)", line))
    edges = {
        name: dependencies & stages.keys() for name, dependencies in parents.items()
    }

    def visit(name, ancestors):
        assert name not in ancestors, f"Stage cycle: {' -> '.join([*ancestors, name])}"
        for dependency in edges[name]:
            visit(dependency, [*ancestors, name])

    for name in stages:
        visit(name, [])
    return {name: "\n".join(body) for name, body in stages.items()}, parents


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("platform", ["amd64", "arm64", "multi"])
def test_efa_inventory_reads_installed_packages_before_final_legal_copy(
    framework, platform
):
    rendered = _render(framework, platform=platform)
    stages, parents = _stage_graph(rendered)
    assert "pre_efa" in stages
    assert parents["pre_efa"] >= {"pre_runtime"}
    assert "./efa_installer.sh" in stages["pre_efa"]
    assert parents["licenses"] >= {"pre_efa"}
    assert "python3 -m compliance.generators" in stages["licenses"]
    assert parents["runtime"] >= {"pre_efa", "licenses"}
    assert parents["aws"] == {"runtime"}
    assert parents["compliance_artifact"] >= {"licenses"}
    assert parents["sources_collect"] >= {"pre_efa"}
    assert rendered.count("./efa_installer.sh") == 1
    assert len(re.findall(r"^COPY --from=licenses /legal /legal$", rendered, re.M)) == 1


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
def test_non_efa_runtime_keeps_its_existing_inventory_parent(framework):
    stages, parents = _stage_graph(_render(framework, make_efa=False))
    assert "pre_efa" not in stages
    assert "aws" not in stages
    assert parents["licenses"] >= {"pre_runtime"}
    assert parents["runtime"] >= {"pre_runtime", "licenses"}


@pytest.mark.parametrize("framework", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("target", ["dev", "local-dev"])
def test_efa_development_targets_keep_post_dev_installation(framework, target):
    rendered = _render(framework, target=target)
    stages, parents = _stage_graph(rendered)
    assert "pre_efa" not in stages
    assert "licenses" not in stages
    assert parents["aws"] >= {"dev"}
    assert "./efa_installer.sh" in stages["aws"]
    assert rendered.count("./efa_installer.sh") == 1


@pytest.mark.parametrize(
    "target,parent", [("frontend", "pre_frontend"), ("planner", "planner_builder")]
)
def test_planner_and_frontend_inventory_parents_are_unchanged(target, parent):
    stages, parents = _stage_graph(_render("dynamo", target=target, make_efa=False))
    assert "pre_efa" not in stages
    assert parents["licenses"] >= {parent}
