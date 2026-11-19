# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
from difflib import unified_diff
from pathlib import Path

import pytest
import yaml


def repository_root() -> Path:
    if workspace := os.environ.get("GITHUB_WORKSPACE"):
        return Path(workspace).resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "components/src/dynamo/aisimulate").is_dir():
        return cwd

    return Path(__file__).resolve().parents[6]


def assert_yaml_golden(golden: Path, actual: str) -> None:
    expected = golden.read_text() if golden.exists() else None
    if expected is not None and list(yaml.safe_load_all(expected)) == list(
        yaml.safe_load_all(actual)
    ):
        return

    proposal = golden.with_name(f"{golden.stem}.new{golden.suffix}")
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(actual)
    root = repository_root()
    golden_arg, proposal_arg = (
        shlex.quote(str(path.relative_to(root) if path.is_relative_to(root) else path))
        for path in (golden, proposal)
    )
    review_source = golden_arg if expected is not None else "/dev/null"
    diff = "\n".join(
        unified_diff(
            (expected or "").splitlines(),
            actual.splitlines(),
            fromfile=review_source,
            tofile=proposal_arg,
            lineterm="",
        )
    )
    pytest.fail(
        f"DGD golden is missing or differs: {golden}\n"
        f"Actual output was written to {proposal}; the golden was not changed.\n"
        f"Review this proposal: diff -u {review_source} {proposal_arg}\n"
        f"Accept only after reviewing this file: mv {proposal_arg} {golden_arg}\n"
        f"Then rerun this case.\n\n{diff}",
        pytrace=False,
    )
