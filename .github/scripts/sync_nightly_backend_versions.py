#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sync nightly backend versions from container/context.yaml into the docs.

Regenerates the marked blocks in docs/reference/nightly-release-info.md
(current backend versions + backend version history) and refreshes the
``main (ToT)`` row in docs/reference/support-matrix.md -- all derived from
container/context.yaml, the source of truth for delivered image versions.

Current versions are read from container/context.yaml at HEAD; the history
table is reconstructed from the git log of that file. Output is deterministic,
so re-running with no source change rewrites nothing (idempotent).

Usage:
    sync_nightly_backend_versions.py            # rewrite in place (default)
    sync_nightly_backend_versions.py --check    # exit 1 if any file is stale
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

CONTEXT = "container/context.yaml"
NIGHTLY_PAGE = "docs/reference/nightly-release-info.md"
SUPPORT_MATRIX = "docs/reference/support-matrix.md"

# First nightly built with the current *-runtime image layout / .devYYYYMMDD
# wheel scheme. History before this predates pullable nightlies, so floor here.
FLOOR = "2026-04-24"


@dataclass(frozen=True)
class Framework:
    label: str  # display name
    key: str  # top-level context.yaml key
    device: str  # sub-key holding runtime_image_tag
    version_re: re.Pattern[str]  # accepts only real backend tags


# Order matters: this is the column order used in the generated tables. Keep the
# device keys in sync with the CUDA targets nightly actually builds
# (see .github/workflows/nightly-ci.yml). The version_re drops pre-layout
# base-image tags (e.g. vLLM 13.0.2, TRT-LLM 26.02) that the history walk would
# otherwise pick up from old commits, keeping only genuine backend versions.
FRAMEWORKS = [
    Framework("vLLM", "vllm", "cuda13.0", re.compile(r"^v\d+\.\d+\.\d+")),
    Framework("SGLang", "sglang", "cuda13.0", re.compile(r"^v\d+\.\d+")),
    Framework("TensorRT-LLM", "trtllm", "cuda13.1", re.compile(r"^\d+\.\d+\.\d+rc\d+")),
]


def parse_version(tag: str) -> str:
    """The framework version embedded in a runtime image tag.

    e.g. ``v0.24.0-ubuntu2404`` -> ``v0.24.0``, ``1.3.0rc20`` -> ``1.3.0rc20``.
    """
    return tag.split("-")[0]


def tag_from_context(doc: dict, fw: Framework) -> str | None:
    node = doc.get(fw.key)
    if not isinstance(node, dict):
        return None
    dev = node.get(fw.device)
    if not isinstance(dev, dict):
        return None
    return dev.get("runtime_image_tag")


def git(args: list[str], repo_root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True)


def read_current(repo_root: Path) -> dict[str, tuple[str, str]]:
    """{label: (version, raw_runtime_image_tag)} from context.yaml at HEAD."""
    doc = yaml.safe_load((repo_root / CONTEXT).read_text())
    out: dict[str, tuple[str, str]] = {}
    for fw in FRAMEWORKS:
        tag = tag_from_context(doc, fw)
        if tag is None:
            raise SystemExit(
                f"{CONTEXT}: no runtime_image_tag for {fw.key}.{fw.device}"
            )
        out[fw.label] = (parse_version(tag), tag)
    return out


def read_history(repo_root: Path) -> dict[str, list[tuple[str, str]]]:
    """Reconstruct per-framework version change-points from git history.

    Returns {label: [(version, start_date), ...]} oldest-first.
    """
    lines = (
        git(["log", "--reverse", "--format=%H|%cs", "--", CONTEXT], repo_root)
        .strip()
        .splitlines()
    )
    changes: dict[str, list[tuple[str, str]]] = {fw.label: [] for fw in FRAMEWORKS}
    for line in lines:
        sha, date = line.split("|", 1)
        try:
            blob = git(["show", f"{sha}:{CONTEXT}"], repo_root)
            doc = yaml.safe_load(blob)
        except (subprocess.CalledProcessError, yaml.YAMLError):
            continue  # file absent or unparsable at this commit
        if not isinstance(doc, dict):
            continue
        for fw in FRAMEWORKS:
            tag = tag_from_context(doc, fw)
            if not tag:
                continue
            v = parse_version(tag)
            if not fw.version_re.match(v):
                continue  # base-image tag from before the runtime layout existed
            pts = changes[fw.label]
            if not pts or pts[-1][0] != v:
                pts.append((v, date))
    return changes


def render_current(current: dict[str, tuple[str, str]]) -> str:
    rows = [
        "| Backend | Version | `context.yaml` runtime image tag |",
        "|---------|---------|----------------------------------|",
    ]
    for fw in FRAMEWORKS:
        version, tag = current[fw.label]
        rows.append(f"| {fw.label} | `{version}` | `{tag}` |")
    return "\n".join(rows)


def render_history(changes: dict[str, list[tuple[str, str]]]) -> str:
    sections: list[str] = []
    for fw in FRAMEWORKS:
        pts = changes[fw.label]
        rows: list[str] = []
        for i in range(len(pts) - 1, -1, -1):
            version, start = pts[i]
            end = "present" if i == len(pts) - 1 else pts[i + 1][1]
            if end != "present" and end < FLOOR:
                continue  # entirely before the first pullable nightly
            disp_start = start if start >= FLOOR else f"≤{FLOOR}"
            rows.append(f"| `{version}` | {disp_start} → {end} |")
        if not rows:
            continue
        sections.append(
            "\n".join(
                [
                    f"### {fw.label}",
                    "",
                    "| Version | In nightlies |",
                    "|---------|--------------|",
                    *rows,
                ]
            )
        )
    return "\n\n".join(sections)


def replace_block(text: str, name: str, inner: str) -> str:
    # Markers use the MDX comment form {/* ... */} -- the form Fern renders
    # invisibly on frontmatter pages ( <!-- --> is not MDX-safe here ). The
    # BEGIN/END markers are preserved; everything between them is regenerated,
    # with one blank line on each side so the table parses as its own block.
    pat = re.compile(
        r"(?s)(\{/\* BEGIN:" + re.escape(name) + r"\b[^\n]*?\*/\})"
        r".*?"
        r"(\{/\* END:" + re.escape(name) + r" \*/\})"
    )
    new, n = pat.subn(lambda m: f"{m.group(1)}\n\n{inner}\n\n{m.group(2)}", text)
    if n != 1:
        raise SystemExit(
            f"{NIGHTLY_PAGE}: expected exactly 1 '{name}' marker block, found {n}"
        )
    return new


# main (ToT) row of the Backend Dependencies table in support-matrix.md.
# Columns: Dynamo | SGLang | TensorRT-LLM | vLLM | NIXL. We rewrite the three
# framework cells and leave the trailing NIXL cell untouched.
TOT_RE = re.compile(r"(?m)^\| \*\*main \(ToT\)\*\* \| `[^`]+` \| `[^`]+` \| `[^`]+` \|")


def update_tot(text: str, current: dict[str, tuple[str, str]]) -> str:
    def bare(version: str) -> str:
        # support-matrix cells omit the leading `v` (e.g. `0.24.0`).
        return version[1:] if version.startswith("v") else version

    sglang = bare(current["SGLang"][0])
    trtllm = bare(current["TensorRT-LLM"][0])
    vllm = bare(current["vLLM"][0])
    repl = f"| **main (ToT)** | `{sglang}` | `{trtllm}` | `{vllm}` |"
    new, n = TOT_RE.subn(repl, text)
    if n != 1:
        raise SystemExit(
            f"{SUPPORT_MATRIX}: expected exactly 1 main (ToT) row, found {n}"
        )
    return new


def build(repo_root: Path) -> dict[Path, str]:
    """Compute the desired content of every file this script owns."""
    current = read_current(repo_root)
    changes = read_history(repo_root)

    page_path = repo_root / NIGHTLY_PAGE
    page = page_path.read_text()
    page = replace_block(page, "backend-versions", render_current(current))
    page = replace_block(page, "backend-history", render_history(changes))

    sm_path = repo_root / SUPPORT_MATRIX
    sm = update_tot(sm_path.read_text(), current)

    return {page_path: page, sm_path: sm}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any owned file is out of date (do not write)",
    )
    args = ap.parse_args()

    outputs = build(args.repo_root)
    stale = {p: new for p, new in outputs.items() if p.read_text() != new}

    if args.check:
        for p in stale:
            print(f"out of date: {p.relative_to(args.repo_root)}")
        if stale:
            print("run: python .github/scripts/sync_nightly_backend_versions.py")
            return 1
        print("nightly backend docs are up to date")
        return 0

    for p, new in outputs.items():
        if p in stale:
            p.write_text(new)
            print(f"updated {p.relative_to(args.repo_root)}")
        else:
            print(f"unchanged {p.relative_to(args.repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
