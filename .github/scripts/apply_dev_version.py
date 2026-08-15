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

With `--set-version X.Y.Z[.devN|.postN]` it instead SETS an absolute release
version: it replaces the current workspace version M wherever it appears in those
same files, plus the Helm Chart.yaml version/appVersion/dependency sites. Python
keeps PEP 440 form ('0.8.1.post1', '0.8.1.dev3'); for Cargo/Helm a .devN becomes a
SemVer pre-release ('0.8.1-dev3', sorts before 0.8.1) and a .postN becomes SemVer
build metadata ('0.8.1+post1'). Sites holding an independent version (not M) are
left alone.
"""
from __future__ import annotations

import argparse
import re
import subprocess
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

# Member manifests that pin the workspace version inside a dependency
# inline table, e.g. backend-common's
# `dynamo-llm = { path = "../llm", version = "1.4.0", default-features = false }`
# (a direct path dep because cargo cannot express `workspace = true` +
# `default-features = false`). VERSION_LINE_RE is line-anchored and
# intentionally skips inline tables, so these files get the same exact-string
# rewrite as the root path-dep pins (the pinned value always equals
# [workspace.package].version). Covered by BOTH modes: the dev-suffix stamp
# (rewrite_root_cargo) and --set-version (set_release_version).
WORKSPACE_PIN_CARGO_TARGETS = [
    "lib/backend-common/Cargo.toml",
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

# Container token -> the NGC repo release.yml actually publishes at :<version>.
# Used by --image-refs to rewrite the `my-registry`/`my-tag` placeholders in docs,
# examples and deploy manifests ONLY for images this release publishes, so the tree
# never advertises a tag that will not exist.
# The `-efa` tokens are deliberately ABSENT: they publish <repo>:<version>-efa, so an
# EFA-only selection must leave the plain <repo>:<version> references alone.
# Images with no token (fastvideo-runtime, epp-image, nixlbench, tensorrt-llm, dynamo)
# are never published by release.yml, so their placeholders stay placeholders.
IMAGE_REF_TOKENS = {
    "vllm-runtime": "vllm-runtime",
    "sglang-runtime": "sglang-runtime",
    "trtllm-runtime": "tensorrtllm-runtime",
    "frontend": "dynamo-frontend",
    "operator": "kubernetes-operator",
    "planner": "dynamo-planner",
    "snapshot": "snapshot-agent",
}
GA_REGISTRY = "nvcr.io/nvidia/ai-dynamo"
PLACEHOLDER_REGISTRY = "my-registry"

# .devN is a PRE-release (sorts before X.Y.Z) -> SemVer '-devN'; .postN is a
# post-release -> SemVer build metadata '+postN'. Both keep PEP 440 form for Python.
SET_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\.(dev\d+|post\d+))?$")

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

    pin_re = re.compile(rf'(\bversion\s*=\s*"){re.escape(base)}(")')
    text = pin_re.sub(lambda mm: f"{mm.group(1)}{new}{mm.group(2)}", text)
    path.write_text(text)

    # Bump the same literal pin where it lives in member manifests (inline
    # dep tables that VERSION_LINE_RE deliberately skips). The early
    # "already stamped" return above keeps this idempotent.
    for rel in WORKSPACE_PIN_CARGO_TARGETS:
        p = root / rel
        if not p.exists():
            continue
        t = p.read_text()
        t2 = pin_re.sub(lambda mm: f"{mm.group(1)}{new}{mm.group(2)}", t)
        if t2 != t:
            p.write_text(t2)


def _workspace_version(root: Path) -> str:
    text = (root / "Cargo.toml").read_text()
    m = re.search(r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("no [workspace.package].version in root Cargo.toml")
    return m.group(1)


def _semver_form(new: str) -> str:
    m = SET_RE.match(new)
    if not m:
        raise RuntimeError(f"--set-version must be X.Y.Z, X.Y.Z.devN, or X.Y.Z.postN (got '{new}')")
    base = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    suffix = m.group(4)
    if not suffix:
        return base
    # dev -> pre-release '-devN' (sorts before base); post -> build metadata '+postN'.
    return f"{base}-{suffix}" if suffix.startswith("dev") else f"{base}+{suffix}"


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
    text = path.read_text()

    # Top-level version/appVersion set unconditionally: a rewrite keyed on `old`
    # leaves stale values when a reused branch widens the helm subset.
    top = re.compile(
        r'^(?P<pre>(?:appVersion|version)\s*:\s*)(?P<q>"?)[^"\n]*(?P=q)(?P<post>\s*)$',
        re.MULTILINE,
    )
    text, n_top = top.subn(
        lambda m: f"{m.group('pre')}{m.group('q')}{new}{m.group('q')}{m.group('post')}", text
    )
    if n_top == 0:
        raise RuntimeError(f"no top-level version/appVersion in {path}")

    # dynamo-operator (file:// subchart) pin always rides the workspace version;
    # the hop is bounded to the entry so it can't reach nats/etcd/....
    text = re.sub(
        r'(?m)^(\s*-\s+name:\s*dynamo-operator\s*\n'
        r'(?:(?!\s*-\s)[^\n]*\n)*?'
        r'\s*version\s*:\s*)("?)[^"\n]*\2(\s*)$',
        lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(2)}{m.group(3)}",
        text,
    )

    # Deliberately no generic indented-version rewrite: dynamo-operator is the
    # only first-party dep pin, and a keyed catch-all would clobber a foreign
    # pin that equals the workspace version (nats is pinned 1.3.2).
    path.write_text(text)


# Bounds the repository->tag hop at the next `repository:` line, so a block
# with no tag fails loudly instead of rewriting another image's tag.
_TAG_HOP = r'(?:(?![^\n]*repository:)[^\n]*\n)*?'


def set_helm_values_tag(path: Path, repo: str, new: str) -> None:
    # Set the `tag:` that follows the image `repository: <repo>` line to `new`,
    # regardless of its current value (the published image tag is the release tag).
    pat = re.compile(
        r'(repository:\s*"?' + re.escape(repo) + r'"?\s*\n' + _TAG_HOP + r'\s*tag:\s*)"?[^"\n]*"?',
        re.MULTILINE,
    )
    text, n = pat.subn(lambda m: f"{m.group(1)}{new}", path.read_text(), count=1)
    if n != 1:
        raise RuntimeError(f"could not find image tag for {repo} in {path}")
    path.write_text(text)


def _current_image_tag(path: Path, repo: str) -> str:
    # The tag currently set for `repo` in values.yaml ('' if unset/missing —
    # an empty tag inherits the chart appVersion at deploy time, so there is no
    # recorded last-published tag to pin to).
    m = re.search(
        r'repository:\s*"?' + re.escape(repo) + r'"?\s*\n' + _TAG_HOP + r'\s*tag:\s*"?([^"\n]*)"?',
        path.read_text(),
    )
    return m.group(1).strip() if m else ""


def _tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    files = []
    for rel in out.split("\0"):
        if not rel or rel.startswith(".github/"):
            # .github holds the release tooling itself — rewriting it would corrupt
            # the very placeholder patterns this step relies on.
            continue
        p = root / rel
        if p.is_file() and not p.is_symlink():
            files.append(p)
    return files


def rewrite_image_refs(root: Path, new_version: str, containers: set[str],
                       old_version: str) -> tuple[int, int]:
    """Point first-party image references at the GA registry + release version —
    but ONLY for images this release actually publishes.

    Rewrites, per selected image:
        my-registry/<img>:my-tag        -> <GA>/<img>:<new>
        <GA>/<img>:my-tag               -> <GA>/<img>:<new>   (tag-only placeholder)
        <GA>/<img>:<old>                -> <GA>/<img>:<new>   (re-cut at a new version)

    Anything not in the selection keeps its placeholder, so a container-only release
    can never ship a doc telling users to pull an image that was never built.
    Returns (files_changed, refs_rewritten)."""
    images = sorted({IMAGE_REF_TOKENS[t] for t in containers if t in IMAGE_REF_TOKENS})
    if not images:
        print("rewrite_image_refs: no publishable image selected; placeholders left intact",
              file=sys.stderr)
        return (0, 0)

    # old_version comes from Cargo.toml, i.e. the SemVer form (1.3.0+post1,
    # 1.3.1-dev0), but image tags carry the PEP 440 form (1.3.0.post1, 1.3.1.dev0).
    # Match BOTH or a re-cut off a .devN/.postN branch silently leaves every image
    # reference pinned to the previous release.
    old_literals = ["my-tag"]
    if old_version and old_version != new_version:
        old_pep = re.sub(r"[-+](dev|post)", r".\1", old_version)
        for t in dict.fromkeys((old_version, old_pep)):
            if t and t != new_version:
                old_literals.append(t)
    tag_alt = "|".join(re.escape(t) for t in old_literals)
    reg_alt = f"(?:{re.escape(PLACEHOLDER_REGISTRY)}|{re.escape(GA_REGISTRY)})"
    # The tag must END here: `(?![\w.-])` refuses to match a PREFIX of a longer tag.
    # Without it `:1.3.0` would also match inside `:1.3.0-nemotron`, `:1.2.0-efa`,
    # `:1.3.0-cuda13` and `:1.4.0.dev1`, rewriting them to a bare version and
    # silently destroying the variant suffix.
    pats = [
        (img, re.compile(rf"{reg_alt}/{re.escape(img)}:(?:{tag_alt})(?![\w.-])"), f"{GA_REGISTRY}/{img}:{new_version}")
        for img in images
    ]
    # Untagged prose references (`my-registry/vllm-runtime` with no `:tag`) still
    # move to the GA registry — but only for selected images, so an unpublished
    # image is never given a real registry path. Runs after the tagged patterns,
    # which have already consumed the `<reg>/<img>:<tag>` forms.
    pats += [
        (img, re.compile(rf"{re.escape(PLACEHOLDER_REGISTRY)}/{re.escape(img)}(?![\w.:-])"), f"{GA_REGISTRY}/{img}")
        for img in images
    ]

    files_changed = refs = 0
    for path in _tracked_files(root):
        try:
            text = original = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable — nothing to substitute
        # Fast path — must test every literal the regex can match, including the
        # PEP 440 spelling of the old version and the untagged placeholder registry.
        if not any(t in text for t in old_literals) and PLACEHOLDER_REGISTRY not in text:
            continue
        for _img, pat, repl in pats:
            text, n = pat.subn(repl, text)
            refs += n
        if text != original:
            path.write_text(text)
            files_changed += 1

    print(f"rewrite_image_refs: {refs} reference(s) in {files_changed} file(s) -> "
          f"{GA_REGISTRY}/<image>:{new_version} for {images}", file=sys.stderr)

    # Advisory only: a re-cut of a branch that previously shipped a wider selection
    # legitimately still carries those older refs, so warn rather than fail.
    unselected = sorted(set(IMAGE_REF_TOKENS.values()) - set(images))
    stale = []
    for path in _tracked_files(root):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for img in unselected:
            if f"{GA_REGISTRY}/{img}:{new_version}" in text:
                stale.append(f"{path.relative_to(root)} -> {img}")
    if stale:
        print(f"::warning::{len(stale)} reference(s) point at unselected image(s) at "
              f"{new_version}: {stale[:8]}{' …' if len(stale) > 8 else ''}", file=sys.stderr)
    return (files_changed, refs)


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


def set_release_version(root: Path, new_version: str, containers: set[str], helm: set[str],
                        image_refs: bool = False) -> None:
    old = _workspace_version(root)
    semver = _semver_form(new_version)

    def _exists(rel: str) -> bool:
        # source_ref may predate a target (older release branches / main SHAs);
        # a path that doesn't exist there has nothing to stamp.
        if (root / rel).exists():
            return True
        print(f"set_release_version: skip {rel} (absent at this source ref)", file=sys.stderr)
        return False

    # Package identity -- ALWAYS bumped, regardless of the wheels/crates selection:
    # the containers embed wheels built from this tree, so a container-only release
    # still needs the workspace/pyproject versions stamped or the shipped image would
    # carry the previous version. (wheels/crates are intentionally not passed in.)
    for rel in PYPROJECT_TARGETS:
        if rel == "pyproject.toml" or _exists(rel):
            set_pyproject(root / rel, old, new_version, is_root=(rel == "pyproject.toml"))
    set_cargo(root / "Cargo.toml", old, semver)
    for rel in SUBCRATE_CARGO_TARGETS:
        if _exists(rel):
            set_cargo(root / rel, old, semver)
    # Workspace-version pins in member manifests (inline dep tables, e.g.
    # backend-common's dynamo-llm pin) carry the same literal as the root
    # pins; set_cargo's exact-string replace stamps them identically.
    for rel in WORKSPACE_PIN_CARGO_TARGETS:
        if _exists(rel):
            set_cargo(root / rel, old, semver)
    # Chart identity -- only for charts in the --helm subset.
    for token, rel in HELM_CHART_TARGETS:
        if token in helm and _exists(rel):
            set_helm(root / rel, old, semver)
    # First-party image tags: published image -> new_version (NGC tag form, not
    # SemVer); published chart with excluded image -> pin to the recorded tag; no
    # recorded tag (''/'my-tag') -> fail at cut time, the chart could never resolve.
    for ctoken, htoken, rel, repo in HELM_IMAGE_TAG_SITES:
        if htoken not in helm or not _exists(rel):
            continue
        path = root / rel
        if ctoken in containers:
            tag = new_version
        else:
            tag = _current_image_tag(path, repo)
            if tag in ("", "my-tag"):
                raise RuntimeError(
                    f"chart '{htoken}' is selected but its image '{ctoken}' is excluded and "
                    f"{rel} records no previously published tag for {repo}; either add "
                    f"'{ctoken}' to the container selection or drop '{htoken}' from the helm selection")
        set_helm_values_tag(path, repo, tag)
    print(f"set_release_version: {old} -> py={new_version} semver={semver} "
          f"containers={sorted(containers)} helm={sorted(helm)}", file=sys.stderr)
    # Docs / examples / deploy manifests: selection-gated so the release branch never
    # advertises an image tag this release does not publish.
    if image_refs:
        rewrite_image_refs(root, new_version, containers, old)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suffix", nargs="?", default="", help="e.g. .dev20260423 (empty = no-op)")
    ap.add_argument("root", nargs="?", default=".", help="repo root")
    ap.add_argument("--set-version", dest="set_version", default="",
                    help="set an absolute release version X.Y.Z[.devN|.postN] instead of appending a suffix")
    ap.add_argument("--containers", default="all",
                    help="normalized container subset (all|none|csv) gating image-tag bumps")
    ap.add_argument("--helm", default="all",
                    help="helm chart subset (all|none|csv of platform,snapshot) gating chart bumps")
    ap.add_argument("--image-refs", action="store_true",
                    help="also point the my-registry/my-tag placeholders in docs, examples and "
                         "deploy manifests at the GA registry + release version — only for images "
                         "in --containers; unselected images keep their placeholder")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    if args.set_version:
        containers = _parse_subset(args.containers, CONTAINER_TOKENS)
        helm = _parse_subset(args.helm, HELM_TOKENS)
        set_release_version(root, args.set_version, containers, helm, image_refs=args.image_refs)
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
