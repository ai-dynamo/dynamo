# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterable
from difflib import unified_diff
from pathlib import Path
from typing import Any

import pytest
import yaml


def repository_root() -> Path:
    candidates = [Path.cwd()]
    if workspace := os.environ.get("GITHUB_WORKSPACE"):
        candidates.append(Path(workspace))

    for candidate in candidates:
        root = candidate.resolve()
        if (root / "components/src/dynamo/aisimulate").is_dir():
            return root

    return Path(__file__).resolve().parents[6]


def _load_structured(path: Path, content: str) -> Any:
    if path.suffix == ".json":
        return json.loads(content)
    if path.suffix in {".yaml", ".yml"}:
        return list(yaml.safe_load_all(content))
    raise ValueError(f"unsupported golden format: {path}")


def compare_structured_golden(golden: Path, actual: Path) -> str | None:
    actual_content = actual.read_text()
    expected_content = golden.read_text() if golden.exists() else None
    if expected_content is not None and _load_structured(
        golden, expected_content
    ) == _load_structured(actual, actual_content):
        return None

    proposal = golden.with_name(f"{golden.stem}.new{golden.suffix}")
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(actual_content)
    root = repository_root()
    golden_arg, proposal_arg = (
        shlex.quote(str(path.relative_to(root) if path.is_relative_to(root) else path))
        for path in (golden, proposal)
    )
    review_source = golden_arg if expected_content is not None else "/dev/null"
    diff = "\n".join(
        unified_diff(
            (expected_content or "").splitlines(),
            actual_content.splitlines(),
            fromfile=review_source,
            tofile=proposal_arg,
            lineterm="",
        )
    )
    return (
        f"Golden is missing or differs: {golden}\n"
        f"Actual output was written to {proposal}; the golden was not changed.\n"
        f"Review this proposal: diff -u {review_source} {proposal_arg}\n"
        f"Accept only after reviewing this file: mv {proposal_arg} {golden_arg}\n"
        f"Then rerun this case.\n\n{diff}"
    )


def assert_structured_goldens(pairs: Iterable[tuple[Path, Path]]) -> None:
    mismatches = [
        mismatch
        for golden, actual in pairs
        if (mismatch := compare_structured_golden(golden, actual)) is not None
    ]
    if mismatches:
        pytest.fail("\n\n".join(mismatches), pytrace=False)
