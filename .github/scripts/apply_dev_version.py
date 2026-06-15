#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every Dynamo package version and cross-ref.

Invoked by nightly CI on the runner, before `docker buildx build`. Takes one
argument -- a suffix like '.dev20260423' -- and rewrites, in place:
  - [project].version in every Dynamo pyproject.toml (PEP 440 form)
  - [package].version / [workspace.package].version in every Cargo.toml
    (SemVer form: dash instead of dot before 'dev', so '1.1.0-dev20260423')
  - The `ai-dynamo-runtime==1.1.0` pin in the root pyproject
  - The `version = "1.1.0"` pins on dynamo-*/kvbm-* path deps in root Cargo.toml

Empty suffix is a no-op, so safe to run unconditionally in every workflow.

With `--set-version X.Y.Z[.postN]` it instead SETS an absolute release version:
it replaces the current workspace version M wherever it appears in those same
files, plus the Helm Chart.yaml version/appVersion/dependency sites. Python
keeps PEP 440 form ('0.8.1.post1'); Cargo/Helm use SemVer build metadata
('0.8.1+post1'). Sites holding an independent version (not M) are left alone.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT_TARGETS = [
    "pyproject.toml",
    "lib/bindings/python/pyproject.toml",
    "lib/bindings/kvbm/pyproject.toml",
    "lib/gpu_memory_service/pyproject.toml",
]

# Sub-crate Cargo files with an EXPLICIT [package].version (not workspace-inherited).
# kvbm-config uses `version.workspace = true`, so it's intentionally omitted.
# lib/runtime/examples/Cargo.toml is also omitted: it's a nested workspace (own
# [workspace.package]) used only for local example binaries, not shipped in any
# wheel, and nothing outside that workspace pins its version.
# Root Cargo.toml is handled separately by rewrite_root_cargo.
SUBCRATE_CARGO_TARGETS = [
    "lib/bindings/python/Cargo.toml",
    "lib/bindings/python/codegen/Cargo.toml",
    "lib/bindings/kvbm/Cargo.toml",
    "lib/kvbm-common/Cargo.toml",
    "lib/kvbm-engine/Cargo.toml",
    "lib/kvbm-kernels/Cargo.toml",
    "lib/kvbm-logical/Cargo.toml",
    "lib/kvbm-physical/Cargo.toml",
]

# Helm charts carry the unified version in version / appVersion / dependency
# version. Each entry is (helm_subset_token, Chart.yaml path); a chart is bumped
# only when its token is in the --helm subset. operator is a subchart of platform,
# so it rides the "platform" token. Only touched in --set-version (release) mode;
# nightly never bumps charts.
HELM_CHART_TARGETS = [
    ("platform", "deploy/helm/charts/platform/Chart.yaml"),
    ("platform", "deploy/helm/charts/platform/components/operator/Chart.yaml"),
    ("snapshot", "deploy/helm/charts/snapshot/Chart.yaml"),
]

# First-party image `tag:` sites in values.yaml. Each entry is
# (container_token, helm_token, values.yaml path, image repository). The tag is set
# to the release version only if the chart is published (helm_token in --helm) AND
# its image is published (container_token in --containers). If the chart is
# published but the image is excluded, the tag is PINNED to the last-published value
# so the chart never references a missing image; if the chart is not published the
# site is left untouched. The operator tag is written explicitly here, decoupling it
# from its `tag: "" -> .Chart.AppVersion` inheritance. 3rd-party tags (etcd/nats) are
# never matched (different repositories).
HELM_IMAGE_TAG_SITES = [
    ("operator", "platform", "deploy/helm/charts/platform/values.yaml",
     "nvcr.io/nvidia/ai-dynamo/kubernetes-operator"),
    ("operator", "platform", "deploy/helm/charts/platform/components/operator/values.yaml",
     "nvcr.io/nvidia/ai-dynamo/kubernetes-operator"),
    ("snapshot", "snapshot", "deploy/helm/charts/snapshot/values.yaml",
     "nvcr.io/nvidia/ai-dynamo/snapshot-agent"),
]

# Normalized subset universes for --containers / --helm token validation.
CONTAINER_TOKENS = {
    "vllm-runtime", "vllm-efa", "sglang-runtime", "sglang-efa",
    "trtllm-runtime", "trtllm-efa", "frontend", "operator", "planner", "snapshot",
}
HELM_TOKENS = {"platform", "snapshot"}

# X.Y.Z.postN -> X.Y.Z+postN so SemVer ecosystems (Cargo, Helm) stay valid.
SET_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\.(post\d+))?$")

# Line-anchored: matches `version = "X.Y.Z"` lines. Skips `version.workspace = true`
# (no quotes) and `version = { ... }` (no string). Safe for sub-crate Cargo.tomls
# whose only `version = "..."` line is the [package] one; external-crate deps use
# the `name = { version = "..." }` inline-table form which this regex skips.
VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*")([^"]+)(")\s*$', re.MULTILINE)

# Root pyproject cross-ref to the runtime wheel.
PY_RUNTIME_PIN_RE = re.compile(r'("ai-dynamo-runtime==)([^"]+)(")')


def pep440(suffix: str, base: str) -> str:
    # suffix already starts with '.' (dev release) or '+' (local-only).
    return base + suffix


def semver(suffix: str, base: str) -> str:
    # Convert a PEP 440-style '.devN' into SemVer '-devN'.
    if suffix.startswith("."):
        return base + "-" + suffix[1:]
    return base + suffix


def _pep440_tail(suffix: str) -> str:
    # The trailing text that pep440() appends; used to detect "already stamped".
    return suffix


def _semver_tail(suffix: str) -> str:
    # The trailing text that semver() appends; used to detect "already stamped".
    return "-" + suffix[1:] if suffix.startswith(".") else suffix


def rewrite_pyproject(path: Path, suffix: str, is_root: bool) -> None:
    text = path.read_text()

    current = VERSION_LINE_RE.search(text)
    if current is None:
        raise RuntimeError(f"no [project].version in {path}")
    if current.group(2).endswith(_pep440_tail(suffix)):
        return  # already stamped -- idempotent no-op

    def _bump(m: re.Match) -> str:
        return f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}"

    text, n = VERSION_LINE_RE.subn(_bump, text, count=1)
    assert n == 1  # guaranteed by the search above

    if is_root:
        text = PY_RUNTIME_PIN_RE.sub(
            lambda m: f"{m.group(1)}{pep440(suffix, m.group(2))}{m.group(3)}",
            text,
        )
    path.write_text(text)


def rewrite_subcrate_cargo(path: Path, suffix: str) -> None:
    text = path.read_text()
    tail = _semver_tail(suffix)

    def _bump(m: re.Match) -> str:
        base = m.group(2)
        if base.endswith(tail):
            return m.group(0)  # already stamped
        return f"{m.group(1)}{semver(suffix, base)}{m.group(3)}"

    text = VERSION_LINE_RE.sub(_bump, text)
    path.write_text(text)


def rewrite_root_cargo(root: Path, suffix: str) -> None:
    """Root Cargo.toml has three kinds of `version = "..."` sites:
      1. [workspace.package].version                          -- bump
      2. Internal path-dep pins in [workspace.dependencies],  -- bump (must match (1))
         e.g. `dynamo-runtime = { path = "lib/runtime", version = "1.1.0" }`
      3. External-crate deps, e.g. `anyhow = { version = "1" }` -- leave alone

    (1) and (2) always use the SAME literal string. Anchor on it, then rewrite
    only `version = "<that exact string>"` occurrences. This bumps (1) and (2)
    in one pass while leaving (3) untouched (they hold other values like "1",
    "0.45.0", "=0.19.3", etc.). An explicit "already stamped" guard makes this
    idempotent -- re-running with the same suffix is a no-op.
    """
    path = root / "Cargo.toml"
    text = path.read_text()

    m = re.search(
        r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"',
        text,
    )
    if not m:
        raise RuntimeError("no [workspace.package].version in root Cargo.toml")
    base = m.group(1)
    if base.endswith(_semver_tail(suffix)):
        return  # already stamped -- idempotent no-op
    new = semver(suffix, base)

    text = re.sub(
        rf'(\bversion\s*=\s*"){re.escape(base)}(")',
        lambda mm: f"{mm.group(1)}{new}{mm.group(2)}",
        text,
    )
    path.write_text(text)


def _workspace_version(root: Path) -> str:
    text = (root / "Cargo.toml").read_text()
    m = re.search(r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("no [workspace.package].version in root Cargo.toml")
    return m.group(1)


def _semver_form(new: str) -> str:
    m = SET_RE.match(new)
    if not m:
        raise RuntimeError(f"--set-version must be X.Y.Z or X.Y.Z.postN (got '{new}')")
    base = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return f"{base}+{m.group(4)}" if m.group(4) else base


def set_pyproject(path: Path, old: str, new: str, is_root: bool) -> None:
    text = VERSION_LINE_RE.sub(
        lambda m: f"{m.group(1)}{new}{m.group(3)}" if m.group(2) == old else m.group(0),
        path.read_text(),
        count=1,
    )
    if is_root:
        text = PY_RUNTIME_PIN_RE.sub(
            lambda m: f"{m.group(1)}{new}{m.group(3)}" if m.group(2) == old else m.group(0),
            text,
        )
    path.write_text(text)


def set_cargo(path: Path, old: str, new: str) -> None:
    text = re.sub(
        rf'(\bversion\s*=\s*"){re.escape(old)}(")',
        lambda m: f"{m.group(1)}{new}{m.group(2)}",
        path.read_text(),
    )
    path.write_text(text)


def set_helm(path: Path, old: str, new: str) -> None:
    pat = re.compile(
        r'^(?P<pre>\s*(?:appVersion|version)\s*:\s*)(?P<q>"?)'
        + re.escape(old)
        + r'(?P=q)(?P<post>\s*)$',
        re.MULTILINE,
    )
    text = pat.sub(
        lambda m: f"{m.group('pre')}{m.group('q')}{new}{m.group('q')}{m.group('post')}",
        path.read_text(),
    )
    path.write_text(text)


def set_helm_values_tag(path: Path, repo: str, new: str) -> None:
    # Set the `tag:` that follows the image `repository: <repo>` line to `new`,
    # regardless of its current value (the published image tag is the release tag).
    pat = re.compile(
        r'(repository:\s*"?' + re.escape(repo) + r'"?\s*\n(?:[^\n]*\n)*?\s*tag:\s*)"?[^"\n]*"?',
        re.MULTILINE,
    )
    text, n = pat.subn(lambda m: f"{m.group(1)}{new}", path.read_text(), count=1)
    if n != 1:
        raise RuntimeError(f"could not find image tag for {repo} in {path}")
    path.write_text(text)


def _current_image_tag(path: Path, repo: str, fallback: str) -> str:
    # The tag currently set for `repo` in values.yaml. An empty tag ('' inherits the
    # chart appVersion) resolves to the pre-bump workspace version `fallback`.
    m = re.search(
        r'repository:\s*"?' + re.escape(repo) + r'"?\s*\n(?:[^\n]*\n)*?\s*tag:\s*"?([^"\n]*)"?',
        path.read_text(),
    )
    cur = m.group(1).strip() if m else ""
    return cur or fallback


def _parse_subset(spec: str, universe: set[str]) -> set[str]:
    spec = (spec or "all").strip()
    if spec == "all":
        return set(universe)
    if spec in ("", "none"):
        return set()
    sel = {t.strip() for t in spec.split(",") if t.strip()}
    unknown = sel - universe
    if unknown:
        raise RuntimeError(f"unknown subset token(s) {sorted(unknown)}; valid: {sorted(universe)}")
    return sel


def set_release_version(root: Path, new_version: str, containers: set[str], helm: set[str]) -> None:
    old = _workspace_version(root)
    semver = _semver_form(new_version)
    # Package identity -- always bumped (the version wheels/crates/images carry).
    for rel in PYPROJECT_TARGETS:
        set_pyproject(root / rel, old, new_version, is_root=(rel == "pyproject.toml"))
    set_cargo(root / "Cargo.toml", old, semver)
    for rel in SUBCRATE_CARGO_TARGETS:
        set_cargo(root / rel, old, semver)
    # Chart identity -- only for charts in the --helm subset.
    for token, rel in HELM_CHART_TARGETS:
        if token in helm:
            set_helm(root / rel, old, semver)
    # First-party image reference tags: published image -> new version (NGC tag
    # form == new_version, not SemVer); chart published but image excluded -> pin to
    # last-published; chart not published -> untouched.
    for ctoken, htoken, rel, repo in HELM_IMAGE_TAG_SITES:
        if htoken not in helm:
            continue
        path = root / rel
        tag = new_version if ctoken in containers else _current_image_tag(path, repo, old)
        set_helm_values_tag(path, repo, tag)
    print(f"set_release_version: {old} -> py={new_version} semver={semver} "
          f"containers={sorted(containers)} helm={sorted(helm)}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suffix", nargs="?", default="", help="e.g. .dev20260423 (empty = no-op)")
    ap.add_argument("root", nargs="?", default=".", help="repo root")
    ap.add_argument("--set-version", dest="set_version", default="",
                    help="set an absolute release version X.Y.Z[.postN] instead of appending a suffix")
    ap.add_argument("--containers", default="all",
                    help="normalized container subset (all|none|csv) gating image-tag bumps")
    ap.add_argument("--helm", default="all",
                    help="helm chart subset (all|none|csv of platform,snapshot) gating chart bumps")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    if args.set_version:
        containers = _parse_subset(args.containers, CONTAINER_TOKENS)
        helm = _parse_subset(args.helm, HELM_TOKENS)
        set_release_version(root, args.set_version, containers, helm)
        return 0

    if not args.suffix:
        print("apply_dev_version: empty suffix, no-op", file=sys.stderr)
        return 0

    for rel in PYPROJECT_TARGETS:
        rewrite_pyproject(root / rel, args.suffix, is_root=(rel == "pyproject.toml"))
    rewrite_root_cargo(root, args.suffix)
    for rel in SUBCRATE_CARGO_TARGETS:
        rewrite_subcrate_cargo(root / rel, args.suffix)

    print(f"apply_dev_version: stamped suffix '{args.suffix}'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
