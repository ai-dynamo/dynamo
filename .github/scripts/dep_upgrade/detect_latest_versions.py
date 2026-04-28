#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect latest upstream version for a Dynamo framework dependency.

Stdout: one-line JSON: {"framework", "current", "latest", "upgrade_needed"}.
Exit 0 on success regardless of upgrade_needed; non-zero only on real errors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

GH_API = "https://api.github.com"

FRAMEWORK_SOURCES: dict[str, dict[str, object]] = {
    "trtllm": {
        "github_repo": "NVIDIA/TensorRT-LLM",
        "current_regex": re.compile(r"^\s*pip_wheel:\s*tensorrt-llm==(\S+)\s*$", re.M),
    },
}


def gh_releases(repo: str) -> list[dict]:
    req = urllib.request.Request(f"{GH_API}/repos/{repo}/releases?per_page=30")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def parse_version(v: str) -> tuple:
    """Parse '1.3.0rc12' / '1.2.1' / '1.3.0' into a sortable tuple.

    Stable releases sort after pre-releases of the same X.Y.Z.
    """
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:(rc|post|dev)\.?(\d+))?$", v.lstrip("v"))
    if not m:
        return (0, 0, 0, ("", v))
    major, minor, patch, suf, sufnum = m.groups()
    suf_key = suf or "z"
    return (int(major), int(minor), int(patch), (suf_key, int(sufnum or 0)))


def latest_release(repo: str, include_prereleases: bool) -> str:
    candidates: list[str] = []
    for rel in gh_releases(repo):
        if rel.get("draft"):
            continue
        if rel.get("prerelease") and not include_prereleases:
            continue
        candidates.append(rel["tag_name"].lstrip("v"))
    if not candidates:
        raise SystemExit(f"no releases found for {repo}")
    return max(candidates, key=parse_version)


def current_pin(framework: str, repo_root: Path) -> str:
    src = FRAMEWORK_SOURCES[framework]
    text = (repo_root / "container" / "context.yaml").read_text()
    m = src["current_regex"].search(text)  # type: ignore[union-attr]
    if not m:
        raise SystemExit(f"no current pin found for {framework}")
    return m.group(1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", required=True, choices=sorted(FRAMEWORK_SOURCES))
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument(
        "--no-prereleases",
        action="store_true",
        help="exclude rc/post/dev tags",
    )
    args = p.parse_args()

    src = FRAMEWORK_SOURCES[args.framework]
    cur = current_pin(args.framework, args.repo_root)
    latest = latest_release(src["github_repo"], not args.no_prereleases)  # type: ignore[arg-type]
    print(
        json.dumps(
            {
                "framework": args.framework,
                "current": cur,
                "latest": latest,
                "upgrade_needed": parse_version(latest) > parse_version(cur),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
