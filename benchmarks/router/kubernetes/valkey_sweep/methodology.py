# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the local driver and rendered Kubernetes stack to one clean commit."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any

from .artifacts import canonical_digest
from .cluster import MANIFEST_FILES, SWEEP_DIR, render_stack


REPOSITORY = SWEEP_DIR.parents[3]
METHOD_FILES = (
    SWEEP_DIR / "Dockerfile",
    SWEEP_DIR / "__init__.py",
    SWEEP_DIR / "aiperf.in",
    SWEEP_DIR / "aiperf_runner.py",
    SWEEP_DIR / "artifacts.py",
    SWEEP_DIR / "build.in",
    SWEEP_DIR / "cluster.py",
    SWEEP_DIR / "crick-build.in",
    SWEEP_DIR / "ha.py",
    SWEEP_DIR / "methodology.py",
    SWEEP_DIR / "model.py",
    SWEEP_DIR / "network.py",
    SWEEP_DIR / "pylock.aiperf.toml",
    SWEEP_DIR / "pylock.build.toml",
    SWEEP_DIR / "pylock.crick-build.toml",
    SWEEP_DIR / "start_valkey.py",
    SWEEP_DIR / "sweep.py",
    *MANIFEST_FILES,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(REPOSITORY), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(REPOSITORY), *arguments),
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def methodology_binding(image: str) -> dict[str, Any]:
    relative = [path.relative_to(REPOSITORY).as_posix() for path in METHOD_FILES]
    revision = _git("rev-parse", "HEAD")
    if len(revision) != 40:
        raise RuntimeError(f"invalid methodology Git revision: {revision}")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *relative)
    if status:
        raise RuntimeError(f"benchmark methodology is not a clean checkout:\n{status}")
    contents = {
        path.relative_to(REPOSITORY).as_posix(): path.read_bytes()
        for path in METHOD_FILES
    }
    drift = [
        name
        for name, content in contents.items()
        if content != _git_bytes("show", f"{revision}:{name}")
    ]
    final_status = _git(
        "status", "--porcelain", "--untracked-files=all", "--", *relative
    )
    if drift or final_status or _git("rev-parse", "HEAD") != revision:
        raise RuntimeError(
            f"benchmark methodology changed while it was being bound: "
            f"blob_drift={drift}, status={final_status!r}"
        )
    files = {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()
    }
    manifest_texts = tuple(
        contents[path.relative_to(REPOSITORY).as_posix()].decode()
        for path in MANIFEST_FILES
    )
    rendered = render_stack(image, manifest_texts).encode()
    unsigned = {
        "git_revision": revision,
        "git_dirty": False,
        "files": files,
        "rendered_stack_sha256": hashlib.sha256(rendered).hexdigest(),
    }
    return {**unsigned, "methodology_digest": canonical_digest(unsigned)}


def verify_methodology(binding: dict[str, Any], core_revision: str) -> None:
    digest = binding.get("methodology_digest")
    unsigned = {
        key: value for key, value in binding.items() if key != "methodology_digest"
    }
    if digest != canonical_digest(unsigned):
        raise RuntimeError("methodology binding fingerprint is invalid")
    if (
        binding.get("git_dirty") is not False
        or binding.get("git_revision") != core_revision
    ):
        raise RuntimeError(
            f"methodology revision does not match image core: "
            f"methodology={binding.get('git_revision')}, core={core_revision}"
        )
