# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest lint for SGLang elastic-EP deployments (NVBug 6733937, fixed in #14568).

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
    container_commands,
    has_flag,
    iter_containers,
    iter_yaml_documents,
    shell_commands,
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


def command_violations(commands: list[list[str]], where: str) -> tuple[int, list[str]]:
    """Return how many elastic-EP commands were checked and which fail."""
    checked = 0
    violations = []
    for argv in commands:
        if not has_flag(argv, ELASTIC_EP_FLAG):
            continue
        checked += 1
        missing = missing_cuda_graph_flags(argv)
        if missing:
            violations.append(f"{where} is missing {missing}")
    return checked, violations


def manifest_violations(root: Path) -> tuple[int, list[str]]:
    """Check every container under ``root``'s YAML documents."""
    checked = 0
    violations = []
    for path, doc in iter_yaml_documents(root):
        for location, container in iter_containers(doc):
            count, found = command_violations(
                container_commands(container), f"{path.name}: {location}"
            )
            checked += count
            violations.extend(found)
    return checked, violations


def launcher_violations(root: Path) -> tuple[int, list[str]]:
    """Check every command in ``root``'s shell scripts."""
    checked = 0
    violations = []
    for path in sorted(root.rglob("*.sh")):
        count, found = command_violations(
            shell_commands(path.read_text()), f"{path.name}: elastic-EP command"
        )
        checked += count
        violations.extend(found)
    return checked, violations


def test_elastic_ep_manifests_disable_both_cuda_graphs():
    checked, violations = manifest_violations(SGLANG_TEMPLATES)
    # Guard against a moved directory or a schema change making this vacuous.
    assert (
        checked
    ), f"no container with {ELASTIC_EP_FLAG} found under {SGLANG_TEMPLATES}"
    assert not violations, "\n".join(violations)


def test_elastic_ep_launcher_disables_both_cuda_graphs():
    checked, violations = launcher_violations(SGLANG_TEMPLATES)
    assert (
        checked
    ), f"no shell command with {ELASTIC_EP_FLAG} found under {SGLANG_TEMPLATES}"
    assert not violations, "\n".join(violations)


def test_manifest_lint_reports_only_noncompliant_containers(tmp_path):
    # v1beta1 multi-document manifest: argv-list, --flag=value and bash -lc forms
    # pass; "bad" omits the decode flag, and in "split" the blanket flag belongs
    # to a different command than the worker.
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
  - name: split
    podTemplate:
      spec:
        containers:
        - name: main
          command: [/bin/sh, -ec]
          args: ["sglang serve --elastic-ep-backend mooncake; echo --disable-cuda-graph"]
"""
    )
    checked, violations = manifest_violations(tmp_path)
    assert checked == 4
    assert violations == [
        "multi.yaml: spec.components[bad].podTemplate.spec.containers[main] "
        "is missing ['--cuda-graph-backend-decode disabled']",
        "multi.yaml: spec.components[split].podTemplate.spec.containers[main] "
        "is missing ['--cuda-graph-backend-decode disabled', "
        "'--cuda-graph-backend-prefill disabled']",
    ]


GOOD_LAUNCH = """\
kubectl exec "$pod" -- bash -lc "
  nohup sglang serve --model-path '#model' \\
    --elastic-ep-backend mooncake --port $((30000 + offset)) \\
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \\
    > /tmp/joiner.log 2>&1 &
" 2>&1
"""


def test_launcher_lint_accepts_compliant_launch(tmp_path):
    (tmp_path / "launch.sh").write_text(GOOD_LAUNCH)
    assert launcher_violations(tmp_path) == (1, [])


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            "sglang serve --elastic-ep-backend mooncake # --disable-cuda-graph",
            id="flag-in-comment",
        ),
        pytest.param(
            "sglang serve --elastic-ep-backend mooncake; echo --disable-cuda-graph",
            id="flag-in-other-command",
        ),
        pytest.param(
            GOOD_LAUNCH + "bash -lc 'sglang serve --elastic-ep-backend mooncake'",
            id="quoted-script",
        ),
    ],
)
def test_launcher_lint_reports_noncompliant_launch(tmp_path, script):
    (tmp_path / "launch.sh").write_text(script)
    checked, violations = launcher_violations(tmp_path)
    assert checked >= 1
    assert violations == [
        "launch.sh: elastic-EP command is missing "
        "['--cuda-graph-backend-decode disabled', "
        "'--cuda-graph-backend-prefill disabled']"
    ]
