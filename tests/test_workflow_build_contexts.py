# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two operator build paths must pass the same named build contexts.

`COPY --from=<name>` resolves `<name>` as an *image* when no matching
`--build-context <name>=...` is supplied, so omitting one does not fail at the COPY.
It fails while resolving an image nobody publishes:

    failed to resolve source metadata for docker.io/library/compliance:latest: ...
    403 Forbidden

That is far enough from the missing flag that `build-on-demand.yml` shipped without it
and failed every operator build until someone tried to use it.

The requirement is taken from `.github/actions/build-deploy-component` -- the path CI
exercises on every PR, and therefore the one known to work -- rather than derived from
`COPY --from=` lines in the Dockerfile. Those over-state it: the `tester` stage needs an
`operator-chart` context, but no push build reaches that stage, so nothing supplies it
and nothing should.
"""

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
REFERENCE_ACTION = (
    REPO_ROOT / ".github" / "actions" / "build-deploy-component" / "action.yml"
)
OPERATOR_DIR = "deploy/operator"

_BUILD_CONTEXT = re.compile(r"--build-context\s+([A-Za-z0-9_.-]+)=")
# A `run:` block's buildx invocation, up to the next step or blank line.
_BUILDX_INVOCATION = re.compile(r"docker buildx build(?:.|\n)*?(?=\n\n|\n\s{6}-|\Z)")


def reference_build_contexts() -> set[str]:
    """Named contexts the known-good operator build path supplies."""
    return set(_BUILD_CONTEXT.findall(REFERENCE_ACTION.read_text(encoding="utf-8")))


def test_reference_action_supplies_at_least_one_build_context() -> None:
    """Guards the premise below rather than trusting it.

    If the reference action stops passing any context this test would otherwise pass
    vacuously, hiding a regression instead of reporting one.
    """
    assert reference_build_contexts(), (
        f"{REFERENCE_ACTION.relative_to(REPO_ROOT)} passes no --build-context; "
        "either the operator Dockerfile no longer needs one, or this test has gone stale"
    )


def test_inline_operator_builds_match_the_reference_path() -> None:
    """A raw `docker buildx build` of the operator must pass the same contexts.

    Only inline `run:` builds are checked. Steps delegating to build-deploy-component
    inherit the flags from the action and are correct by construction.
    """
    required = reference_build_contexts()
    missing: list[str] = []

    for workflow in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        if f"working-directory: ./{OPERATOR_DIR}" not in text:
            continue
        for invocation in _BUILDX_INVOCATION.findall(text):
            if "-f Dockerfile" not in invocation:
                continue
            supplied = set(_BUILD_CONTEXT.findall(invocation))
            for context in sorted(required - supplied):
                missing.append(f"{workflow.name}: missing --build-context {context}=")

    assert not missing, (
        "an inline operator build omits a named build context that "
        f"{REFERENCE_ACTION.relative_to(REPO_ROOT)} supplies, so the name resolves as an "
        "image and the build fails far from the cause:\n  " + "\n  ".join(missing)
    )
