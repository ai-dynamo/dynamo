# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether stale generated output is this branch's fault or main's.

The API references are generated from source that lives outside docs/. A global
freshness check therefore fails on any commit that touches a documented symbol,
including commits this branch never made: four commits landing on main across
24 Python source files is enough to turn README.mdx, frontend.mdx and
common.mdx stale on a branch that touched none of them.

That is not a defect the branch can fix. Regenerating buys hours until the next
unrelated merge, so the gate reduces to "whoever is unlucky at merge time
regenerates", which is noise rather than signal, and noise on a required check
trains people to re-run it rather than read it.

``attributable`` narrows the gate to what a branch controls. Staleness blocks
only when the branch itself touched something feeding the generator: a source
root, a generated page, or the generator scripts. Otherwise the drift is
reported and the check passes, because main moved and the next regeneration on
main resolves it.

This deliberately does not diff the base tree's generated output. Doing so
would need a second checkout and a second griffe load per generator, and it
answers a question the changed-file set already answers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


def changed_paths(since: str, repo_root: Path) -> frozenset[str] | None:
    """Repo-relative paths this branch changed since ``since``.

    Returns None when the range cannot be resolved -- a shallow clone without
    the base commit, a detached build, a ref that does not exist. Callers must
    treat None as "cannot attribute" and fall back to the strict gate, so a
    broken range can never turn the check into a silent pass.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", f"{since}...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return frozenset(line for line in proc.stdout.splitlines() if line)


def _attributable(
    changed: frozenset[str] | None,
    *,
    sources: tuple[str, ...],
    output_paths: frozenset[str],
    script_dir: str,
) -> bool:
    """True when this branch could have caused the drift, or we cannot tell.

    ``sources`` entries are matched as either an exact file or a directory
    prefix, because the generators differ: the Python one reads two source
    trees, while the Rust one reads Cargo.toml and releases.data.ts by name.

    Fails closed on None: an unresolvable range keeps the strict behaviour.
    """
    if changed is None:
        return True
    prefixes = (script_dir, *sources)
    for path in changed:
        if path in output_paths or path in sources:
            return True
        if any(path.startswith(f"{prefix}/") for prefix in prefixes):
            return True
    return False


def blames_branch(
    since: str,
    *,
    repo_root: Path,
    sources: tuple[str, ...],
    outputs: Iterable[Path],
    script_dir: Path,
) -> bool:
    """Whether drift is this branch's to fix, given what it changed since ``since``.

    The one entry point the generators call. Each differs only in which
    sources it reads; resolving the changed set and normalising output and
    script paths to repo-relative POSIX is identical for all three, so it
    lives here rather than three times at the call sites.
    """
    return _attributable(
        changed_paths(since, repo_root),
        sources=sources,
        output_paths=frozenset(
            path.resolve().relative_to(repo_root).as_posix() for path in outputs
        ),
        script_dir=script_dir.resolve().relative_to(repo_root).as_posix(),
    )
