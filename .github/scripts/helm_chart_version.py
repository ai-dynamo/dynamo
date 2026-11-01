# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm Chart.yaml version rewrite shared by release and nightly packaging.

Imported as a sibling module by apply_dev_version.py (release branch cut) and
prepare_nightly_helm.py (nightly chart packaging). cut-release-branch.yml runs
apply_dev_version.py from RUNNER_TEMP, so it copies this file there from the
same commit.
"""
from __future__ import annotations

import re
from pathlib import Path

# Top-level version/appVersion, line-start anchored (platform has no appVersion).
_TOP_LEVEL = re.compile(
    r'^(?P<pre>(?:appVersion|version)\s*:\s*)(?P<q>"?)[^"\n]*(?P=q)(?P<post>\s*)$',
    re.MULTILINE,
)

# dynamo-operator (file:// subchart) exact pin; the hop is bounded to the entry
# so it can't reach nats/etcd/....
_OPERATOR_PIN = re.compile(
    r"(?m)^(\s*-\s+name:\s*dynamo-operator\s*\n"
    r"(?:(?!\s*-\s)[^\n]*\n)*?"
    r'\s*version\s*:\s*)("?)[^"\n]*\2(\s*)$'
)


def set_chart_versions(path: Path, new: str, expect_operator_pin: bool = False) -> None:
    """Set top-level version/appVersion and the dynamo-operator pin to `new`.

    Top-level fields are set unconditionally: a rewrite keyed on the old version
    leaves stale values when a reused branch widens the helm subset. With
    `expect_operator_pin`, raise before writing unless exactly one operator pin
    matched.
    """
    text = path.read_text(encoding="utf-8")

    text, n_top = _TOP_LEVEL.subn(
        lambda m: f"{m.group('pre')}{m.group('q')}{new}{m.group('q')}{m.group('post')}",
        text,
    )
    if n_top == 0:
        raise RuntimeError(f"no top-level version/appVersion in {path}")

    text, n_dep = _OPERATOR_PIN.subn(
        lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(2)}{m.group(3)}", text
    )
    if expect_operator_pin and n_dep != 1:
        raise RuntimeError(
            f"expected exactly one dynamo-operator dependency pin in {path}, found {n_dep}"
        )

    # No generic indented-version rewrite: it would clobber a third-party pin
    # equal to the workspace version (nats is pinned 1.3.2).
    path.write_text(text, encoding="utf-8")
