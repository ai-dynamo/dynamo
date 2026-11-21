# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest lint for SGLang elastic-EP deployments (NVBug 6733937 / DYN-4325).

SGLang's ``server_args.py::_handle_elastic_ep`` exits at startup with
"Elastic EP runtime scale-up requires decode and prefill CUDA graphs to be
disabled." ``--moe-a2a-backend nixl`` disables only the prefill graph, so a
manifest that relies on it still crash-loops. Every elastic-EP launch must
disable both graphs explicitly, or pass the blanket ``--disable-cuda-graph``.

The whole template directory is walked, not one file, so a new elastic-EP
manifest that omits the flags also fails here.
"""

from pathlib import Path

import pytest

from tests.utils.manifest_lint import (
    container_argv,
    has_flag,
    iter_containers,
    iter_yaml_documents,
    split_shell,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.parallel,
]

SGLANG_TEMPLATES = Path(__file__).parent / "deploy" / "templates" / "sglang"

ELASTIC_EP_FLAG = "--elastic-ep-backend"
REQUIRED_FLAGS = (
    ("--cuda-graph-backend-decode", "disabled"),
    ("--cuda-graph-backend-prefill", "disabled"),
)
BLANKET_FLAG = "--disable-cuda-graph"


def missing_cuda_graph_flags(argv: list[str]) -> list[str]:
    """Return the CUDA-graph flags an elastic-EP launch is missing."""
    if has_flag(argv, BLANKET_FLAG):
        return []
    return [
        f"{flag} {value}"
        for flag, value in REQUIRED_FLAGS
        if not has_flag(argv, flag, value)
    ]


def manifest_violations(root: Path) -> tuple[int, list[str]]:
    """Return how many elastic-EP containers were checked and which fail."""
    checked = 0
    violations = []
    for path, doc in iter_yaml_documents(root):
        for location, container in iter_containers(doc):
            argv = container_argv(container)
            if not has_flag(argv, ELASTIC_EP_FLAG):
                continue
            checked += 1
            missing = missing_cuda_graph_flags(argv)
            if missing:
                violations.append(f"{path.name}: {location} is missing {missing}")
    return checked, violations


def shell_commands(path: Path) -> list[str]:
    """Return the logical commands of a shell script, joining ``\\`` continuations
    and skipping comment lines."""
    commands = []
    current = ""
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        commands.append(current + stripped)
        current = ""
    if current:
        commands.append(current)
    return commands


def test_elastic_ep_manifests_disable_both_cuda_graphs():
    checked, violations = manifest_violations(SGLANG_TEMPLATES)
    # Guard against a moved directory or a schema change making this vacuous.
    assert (
        checked
    ), f"no container with {ELASTIC_EP_FLAG} found under {SGLANG_TEMPLATES}"
    assert not violations, "\n".join(violations)


def test_elastic_ep_launcher_disables_both_cuda_graphs():
    checked = 0
    violations = []
    for path in sorted(SGLANG_TEMPLATES.rglob("*.sh")):
        for command in shell_commands(path):
            argv = split_shell(command)
            if not has_flag(argv, ELASTIC_EP_FLAG):
                continue
            checked += 1
            missing = missing_cuda_graph_flags(argv)
            if missing:
                violations.append(
                    f"{path.name}: elastic-EP command is missing {missing}"
                )
    assert (
        checked
    ), f"no shell command with {ELASTIC_EP_FLAG} found under {SGLANG_TEMPLATES}"
    assert not violations, "\n".join(violations)


def test_manifest_lint_reports_only_noncompliant_containers(tmp_path):
    # v1beta1 multi-document manifest: argv-list, --flag=value and bash -lc forms
    # pass; the last worker omits the decode flag.
    (tmp_path / "multi.yaml").write_text(
        """
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: good
    podTemplate:
      spec:
        containers:
        - name: main
          args: [--elastic-ep-backend, mooncake,
                 --cuda-graph-backend-decode, disabled,
                 --cuda-graph-backend-prefill=disabled]
  - name: blanket
    podTemplate:
      spec:
        containers:
        - name: main
          command: [/bin/bash, -lc]
          args: ["python3 -m dynamo.sglang --elastic-ep-backend mooncake --disable-cuda-graph"]
---
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: bad
    podTemplate:
      spec:
        containers:
        - name: main
          args: [--elastic-ep-backend, mooncake,
                 --cuda-graph-backend-prefill, disabled]
"""
    )
    checked, violations = manifest_violations(tmp_path)
    assert checked == 3
    assert violations == [
        "multi.yaml: spec.components[bad].podTemplate.spec.containers[main] "
        "is missing ['--cuda-graph-backend-decode disabled']"
    ]
