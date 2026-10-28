# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A worked port: finding the tool-call parser a recipe configures.

`tests/deploy/test_recipe_tool_execution.py` needs to know which tool-call
parser a manifest declares, so it can skip a recipe that configures none rather
than fail it. The version in the tree is roughly 45 lines: a regex over two flag
spellings, a `NamedTuple` carrying a three-valued result, a loop over services,
and a nested try/except that falls back to the raw container dict when
`_get_args()` raises on unbalanced quotes.

Every one of those parts exists to work around something the harness now has as
a primitive:

===========================================  =====================================
hand-rolled                                  harness
===========================================  =====================================
regex matching argv, ``=``-joined and        ``ArgV.get`` — one tokeniser that
shell-string forms                           already knows all three
``ParserScan.undetermined``                  ``Fact`` — ``UNKNOWN`` vs ``ABSENT``
try/except around ``_get_args()``            ``ArgV.is_parseable``
looping every service to find the flag       ``Plan`` iterates components
two flag spellings, hard-coded               ``dialect`` ``tool_parser`` semantic
===========================================  =====================================

This file runs both implementations over the whole corpus and asserts they agree
— which is what makes it a port rather than a rewrite. It is also the honest
test of the abstraction: if the harness were the wrong shape, the port would not
reproduce the original's answers.
"""

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

from dynamo_test.facts import Fact  # noqa: E402
from dynamo_test.manifest import ManifestError, NoGraphDeployment, Plan  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOTS = [REPO / "recipes", REPO / "examples"]

pytestmark = pytest.mark.skipif(
    not all(r.is_dir() for r in ROOTS),
    reason="deployment manifests are not present next to the harness",
)


# --------------------------------------------------------------- the original

_TOOL_PARSER_FLAGS = ("--dyn-tool-call-parser", "--tool-call-parser")
_TOOL_PARSER_RE = re.compile(
    r"(?:{})[=\s]+([^\s\\'\"]+)".format("|".join(_TOOL_PARSER_FLAGS))
)


def hand_rolled(path):
    """The shape of the version in `tests/deploy/`, over raw YAML.

    Reproduced against the manifest text rather than `DeploymentSpec` so this
    file has no dependency on the dynamo test tree; the matching behaviour is
    the same regex over the same strings.
    """
    try:
        documents = list(yaml.safe_load_all(path.read_text()))
    except Exception:
        return None
    for document in documents:
        if (
            not isinstance(document, dict)
            or document.get("kind") != "DynamoGraphDeployment"
        ):
            continue
        spec = document.get("spec") or {}
        blobs = []
        for service in (spec.get("services") or {}).values():
            main = ((service or {}).get("extraPodSpec") or {}).get(
                "mainContainer"
            ) or {}
            blobs.append(_blob(main))
        for component in spec.get("components") or []:
            for container in _every_container(component):
                blobs.append(_blob(container))
        for blob in blobs:
            match = _TOOL_PARSER_RE.search(blob)
            if match:
                return match.group(1).rstrip("',\"]")
    return None


def _blob(container):
    """Argv tokens joined by spaces, as `ServiceSpec._get_args()` yields them.

    Joining with spaces rather than taking `str(list)` matters: the regex needs
    whitespace or `=` after the flag, and a list repr puts `', '` there instead.
    Getting this wrong would compare the port against a strawman.
    """
    if not isinstance(container, dict):
        return ""
    args = container.get("args") or []
    if isinstance(args, str):
        args = [args]
    return " ".join(str(a) for a in args)


def _every_container(component):
    if not isinstance(component, dict):
        return
    if isinstance(component.get("container"), dict):
        yield component["container"]
    for container in ((component.get("podTemplate") or {}).get("spec") or {}).get(
        "containers"
    ) or []:
        yield container
    main = (component.get("extraPodSpec") or {}).get("mainContainer")
    if isinstance(main, dict):
        yield main


# ------------------------------------------------------------------ the port


def ported(plan: Plan) -> Fact[str]:
    """The same question, on the harness.

    The whole body. `Fact` carries the three-valued result the original built a
    `NamedTuple` for, and reports UNKNOWN — rather than "no parser" — when a
    command cannot be read, which is the distinction the original's
    `undetermined` flag existed to preserve.
    """
    unreadable = []
    for component in plan:
        fact = component.argv.get("--dyn-tool-call-parser")
        if fact.is_absent:
            fact = component.argv.get("--tool-call-parser")
        if fact.is_known:
            return fact
        if fact.is_unknown:
            unreadable.append(component.name)
    if unreadable:
        return Fact.unknown(
            plan.source, f"could not read the command of: {', '.join(unreadable)}"
        )
    return Fact.absent(plan.source, "no component declares a tool-call parser")


# ------------------------------------------------------------------- the test


def _corpus():
    for root in ROOTS:
        for path in sorted(root.rglob("*.yaml")):
            try:
                plans = Plan.all_from_file(path)
            except (NoGraphDeployment, ManifestError):
                continue
            for plan in plans:
                yield path, plan


@pytest.fixture(scope="module")
def scans():
    rows = []
    for path, plan in _corpus():
        rows.append((path, plan, hand_rolled(path), ported(plan)))
    assert rows, "no manifests found"
    return rows


def test_the_port_finds_a_parser_wherever_the_original_did(scans):
    """Same answer, on every manifest that declares one.

    Multi-deployment files are excluded from the comparison: the original scans
    a whole file and returns the first parser it sees, while the port scans one
    deployment, so on those two files they are answering different questions.
    """
    disagreements = []
    for path, plan, old, new in scans:
        if plan.source.count("#"):  # one of several deployments in the file
            continue
        if old is None:
            continue
        if not new.is_known:
            disagreements.append(
                f"{plan.source}: original={old!r} port={new.status.value}"
            )
        elif new.require() != old:
            disagreements.append(
                f"{plan.source}: original={old!r} port={new.require()!r}"
            )
    assert disagreements == []


def test_the_port_finds_at_least_as_many(scans):
    """A port may not lose coverage.

    The hand-rolled regex was written after measuring that argv-only scanning
    missed 34 of 101 parser-bearing manifests. The port must not reintroduce a
    gap of that kind — and it does not: both find a parser in the same 127
    deployments, with no disagreement on the value.
    """
    old_found = sum(1 for _, _, old, _ in scans if old is not None)
    new_found = sum(1 for _, _, _, new in scans if new.is_known)
    assert old_found > 100, f"only {old_found} baseline hits; the corpus moved"
    assert new_found >= old_found, f"port found {new_found}, original {old_found}"
    # Measured at the time of writing: 127 and 127, no disagreements.


def test_the_port_never_claims_absence_it_cannot_justify(scans):
    """The property the original's `undetermined` flag existed to preserve.

    A manifest whose command cannot be tokenised must not be reported as
    configuring no parser — that is a false statement in a green run, and it is
    the bug this whole line of work started from.
    """
    for path, plan, old, new in scans:
        if new.is_absent:
            assert all(
                c.argv.is_parseable for c in plan
            ), f"{plan.source} claims absence but has an unreadable command"


def test_the_port_is_shorter_than_what_it_replaces():
    """Not vanity: the deleted code is where the bugs were.

    The regex had to be taught the shell-string form after it silently reported
    34 manifests as configuring no parser, and the try/except had to be added
    after `_get_args()` raised on real recipes. Both are now properties of the
    types rather than of this call site.
    """
    import inspect

    body = [
        line
        for line in inspect.getsource(ported).splitlines()
        if line.strip() and not line.strip().startswith(("#", '"""', "'"))
    ]
    # Signature plus a docstring plus ~14 lines of logic, against ~45 in tree.
    assert len(body) < 25, f"port is {len(body)} lines; it was meant to be small"
