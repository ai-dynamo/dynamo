# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read immutable Git source without checking out or executing it."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout


def commit(repo: Path, revision: str) -> str:
    # Require immutable input, and avoid Git option/revision-expression injection.
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit SHA")
    resolved = git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}").strip()
    if resolved != revision:
        raise ValueError("revision must identify a commit, not an annotated tag object")
    return resolved
