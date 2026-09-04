# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay :class:`ArgV` over every shipped deployment manifest.

Unit tests prove the constructs this module *intends* to handle. This one proves
the corpus contains nothing else. It is the test that would have caught the
defect it was written for: ``-lc`` containers were 100 of the 184 shell-invoked
containers in ``recipes/`` and ``examples/``, and every helper that scanned them
returned nothing while reporting the flag as absent.

Skipped when the manifests are not present, so the harness stays installable and
testable on its own.
"""

import pathlib
import subprocess

import pytest
from dynamo_test.argv import ArgForm, ArgV

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOTS = [REPO / "recipes", REPO / "examples"]

pytestmark = pytest.mark.skipif(
    not all(r.is_dir() for r in ROOTS),
    reason="deployment manifests are not present next to the harness",
)


def _containers(path):
    """Every container in every DGD document, across both schema versions."""
    try:
        documents = list(yaml.safe_load_all(path.read_text()))
    except Exception:
        return
    for doc in documents:
        if not isinstance(doc, dict) or doc.get("kind") != "DynamoGraphDeployment":
            continue
        spec = doc.get("spec") or {}
        # v1alpha1
        for name, service in (spec.get("services") or {}).items():
            main = ((service or {}).get("extraPodSpec") or {}).get("mainContainer")
            if main:
                yield name, main
        # v1beta1
        for component in spec.get("components") or []:
            name = (component or {}).get("name", "<unnamed>")
            if component.get("container"):
                yield name, component["container"]
            pod_spec = (component.get("podTemplate") or {}).get("spec") or {}
            for container in pod_spec.get("containers") or []:
                yield name, container
            main = (component.get("extraPodSpec") or {}).get("mainContainer")
            if main:
                yield name, main


def _shell_containers():
    for root in ROOTS:
        for path in sorted(root.rglob("*.yaml")):
            for name, container in _containers(path):
                if not isinstance(container, dict):
                    continue
                argv = ArgV.from_container(
                    container, source=f"{path.relative_to(REPO)}[{name}]"
                )
                if argv.form is ArgForm.SHELL:
                    yield argv


@pytest.fixture(scope="module")
def shell_commands():
    found = list(_shell_containers())
    assert found, "expected the corpus to contain shell-invoked containers"
    return found


def test_every_shipped_shell_command_is_parseable(shell_commands):
    """``shlex`` failed on four of these; a parse failure must not be silent."""
    unparseable = [
        (a.source, a.parse_error) for a in shell_commands if not a.is_parseable
    ]
    assert unparseable == []


def test_the_login_shell_majority_is_detected(shell_commands):
    """``-lc`` outnumbers ``-c``; a predicate that matches only ``-c`` sees half."""
    login = [a for a in shell_commands if a.command[-1] != "-c"]
    assert len(login) > len(shell_commands) / 3, (
        f"only {len(login)} of {len(shell_commands)} use a non-bare-c shell flag; "
        "if the corpus really changed this much, re-measure before relaxing this"
    )


def test_editing_any_shipped_command_preserves_everything_else(shell_commands):
    """Two edits per manifest; every untouched line must be byte-identical.

    This is the property a token round-trip cannot hold: it flattens
    continuations, drops comments, and re-quotes operators into arguments.
    """
    damage = []
    for argv in shell_commands:
        before = argv.as_shell_string()

        edited = argv
        for flag in ("--model-path", "--model", "--served-model-name"):
            if edited.get(flag).is_known:
                edited = edited.set(flag, "sentinel/model")
                break
        edited = edited.set("--dyn-corpus-probe", "42")
        after = edited.as_shell_string()

        if len(after.splitlines()) != len(before.splitlines()):
            damage.append(f"{argv.source}: line count changed")
            continue
        for old, new in zip(before.splitlines(), after.splitlines()):
            if old == new:
                continue
            if "sentinel/model" not in new and "--dyn-corpus-probe" not in new:
                damage.append(f"{argv.source}: unrelated line changed: {old.strip()!r}")

        if len(edited.as_container_args()) != 1:
            damage.append(f"{argv.source}: shell args must stay a single string")

    assert damage == []


def test_comments_survive_an_edit(shell_commands):
    """The comments explain why each flag is set; losing them loses the reason."""
    damage = []
    for argv in shell_commands:
        before = argv.as_shell_string()
        comments = [ln for ln in before.splitlines() if ln.lstrip().startswith("#")]
        if not comments:
            continue
        after = argv.set("--dyn-corpus-probe", "42").as_shell_string()
        if [ln for ln in after.splitlines() if ln.lstrip().startswith("#")] != comments:
            damage.append(argv.source)
    assert damage == []


@pytest.mark.skipif(
    subprocess.run(["which", "bash"], capture_output=True).returncode != 0,
    reason="bash is required to syntax-check the rewritten commands",
)
def test_every_edited_command_still_parses_under_bash(shell_commands):
    """The independent check: bash's own parser, not ours."""
    rejected = []
    for argv in shell_commands:
        after = argv.set("--dyn-corpus-probe", "42").as_shell_string()
        result = subprocess.run(
            ["bash", "-n", "-c", after], capture_output=True, text=True
        )
        if result.returncode != 0:
            rejected.append(f"{argv.source}: {result.stderr.strip()[:120]}")
    assert rejected == []
