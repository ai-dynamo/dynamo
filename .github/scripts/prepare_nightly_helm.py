#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Point the Helm charts at this nightly run's staged images, in place.

Run by release.yml's nightly helm step right before `helm dep build` +
`helm package`. Nightly has no release branch to record a bump on, so the
rewrite is ephemeral: checkout at the source SHA, mutate, package, discard.

Per chart token: Chart.yaml versions -- including the platform->dynamo-operator
exact dependency pin (must move in lockstep) and the operator subchart's
appVersion (`helm package --app-version` stamps only the top-level chart; the
subchart's feeds the image-tag default and `--operator-version`) -- the
top-level chart name (-> dedicated -nightly chart names), plus values.yaml
image sites renamed to the *-nightly NGC repos at this run's dated tag.
Third-party references are never touched; power-agent is excluded.

Every rewrite requires exactly one match and raises otherwise, so a chart
restructure breaks this step loudly instead of shipping stale references.

The default DGDR profiler image (put into DGDR spec.image when unset) is
derived as dynamo-planner:<appVersion>, which cannot exist for pre-release
appVersions -- so the platform rewrite pins the dgdrDefaultImage value to the
dated nightly planner image. rc charts retain the stale derived default.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# token -> Chart.yaml files; the operator subchart rides the platform token.
# Snapshot is an external OCI dependency of the platform chart
# (github.com/ai-dynamo/snapshot), pinned in
# deploy/helm/charts/platform/Chart.yaml, not rewritten here.
CHART_TARGETS: dict[str, list[str]] = {
    "platform": [
        "deploy/helm/charts/platform/Chart.yaml",
        "deploy/helm/charts/platform/components/operator/Chart.yaml",
    ],
}

# token -> (top-level Chart.yaml, nightly chart name). Nightly charts publish
# under dedicated -nightly names, mirroring the container repo convention; the
# operator subchart keeps its name (values keys reference it).
NIGHTLY_CHART_NAMES: dict[str, tuple[str, str]] = {
    "platform": ("deploy/helm/charts/platform/Chart.yaml", "dynamo-platform-nightly"),
}

# token -> (values.yaml path, repository in the file, nightly repository).
IMAGE_SITES: dict[str, list[tuple[str, str, str]]] = {
    "platform": [
        (
            "deploy/helm/charts/platform/values.yaml",
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator",
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator-nightly",
        ),
        (
            "deploy/helm/charts/platform/components/operator/values.yaml",
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator",
            "nvcr.io/nvidia/ai-dynamo/kubernetes-operator-nightly",
        ),
    ],
}

# platform only: pin the default DGDR profiler image (DGDR spec.image when
# unset) to the dated nightly planner — the derived default
# (dynamo-planner:<appVersion>) cannot exist for pre-release chart versions.
DGDR_DEFAULT_IMAGE_SITE = (
    "deploy/helm/charts/platform/components/operator/values.yaml",
    "nvcr.io/nvidia/ai-dynamo/dynamo-planner-nightly",
)

# Bounds the repository->tag hop at the next `repository:` line, so an image
# block with no tag fails loudly instead of rewriting another image's tag.
_TAG_HOP = r"(?:(?![^\n]*repository:)[^\n]*\n)*?"


def set_chart_versions(path: Path, new: str, expect_operator_pin: bool) -> None:
    text = path.read_text()

    # Top-level version/appVersion, line-start anchored (platform has no appVersion).
    top = re.compile(
        r'^(?P<pre>(?:appVersion|version)\s*:\s*)(?P<q>"?)[^"\n]*(?P=q)(?P<post>\s*)$',
        re.MULTILINE,
    )
    text, n_top = top.subn(
        lambda m: f"{m.group('pre')}{m.group('q')}{new}{m.group('q')}{m.group('post')}",
        text,
    )
    if n_top == 0:
        raise RuntimeError(f"no top-level version/appVersion in {path}")

    # dynamo-operator exact pin; hop bounded so third-party pins stay untouched.
    dep = re.compile(
        r"(?m)^(\s*-\s+name:\s*dynamo-operator\s*\n"
        r"(?:(?!\s*-\s)[^\n]*\n)*?"
        r'\s*version\s*:\s*)("?)[^"\n]*\2(\s*)$'
    )
    text, n_dep = dep.subn(
        lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(2)}{m.group(3)}", text
    )
    if expect_operator_pin and n_dep != 1:
        raise RuntimeError(
            f"expected exactly one dynamo-operator dependency pin in {path}, found {n_dep}"
        )

    path.write_text(text)
    print(f"  {path}: version -> {new}")


def set_dgdr_default_image(path: Path, image: str) -> None:
    pat = re.compile(r'^(\s*dgdrDefaultImage:\s*)"?[^"\n]*"?(\s*)$', re.MULTILINE)
    text, n = pat.subn(lambda m: f'{m.group(1)}"{image}"{m.group(2)}', path.read_text())
    if n != 1:
        raise RuntimeError(
            f"expected exactly one dgdrDefaultImage in {path}, found {n}"
        )
    path.write_text(text)
    print(f"  {path}: dgdrDefaultImage -> {image}")


def set_chart_name(path: Path, new: str) -> None:
    # Line-start anchored: never matches indented `- name:` entries
    # (dependencies, maintainers).
    pat = re.compile(r'^(name\s*:\s*)("?)[^"\n]*\2(\s*)$', re.MULTILINE)
    text, n = pat.subn(
        lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(2)}{m.group(3)}",
        path.read_text(),
    )
    if n != 1:
        raise RuntimeError(f"expected exactly one top-level name in {path}, found {n}")
    path.write_text(text)
    print(f"  {path}: name -> {new}")


def set_image_site(path: Path, current_repo: str, nightly_repo: str, tag: str) -> None:
    # One bounded match renames the repo and pins its tag together; the quote
    # backref `\2` also stops prefix matches (e.g. repo + "-nightly").
    pat = re.compile(
        r"(repository:\s*)(\"?)"
        + re.escape(current_repo)
        + r"\2(\s*\n"
        + _TAG_HOP
        + r'\s*tag:\s*)"?[^"\n]*"?',
        re.MULTILINE,
    )
    text, n = pat.subn(
        lambda m: f'{m.group(1)}{m.group(2)}{nightly_repo}{m.group(2)}{m.group(3)}"{tag}"',
        path.read_text(),
    )
    if n != 1:
        raise RuntimeError(
            f"expected exactly one image site for {current_repo} in {path}, found {n}"
        )
    path.write_text(text)
    print(f"  {path}: {current_repo} -> {nightly_repo}:{tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chart-version",
        required=True,
        help="Nightly chart version, X.Y.Z-dev.YYYYMMDD.gSHA7 (prepare-release helm_chart_version).",
    )
    parser.add_argument(
        "--image-tag",
        required=True,
        help="Dated image tag on the -nightly NGC repos, YYYYMMDD-sha7 (prepare-release ngc_version_tag).",
    )
    parser.add_argument(
        "--charts",
        required=True,
        help="Comma-separated chart tokens to rewrite (subset of: platform).",
    )
    parser.add_argument("--root", default=".", help="Repository root (default: cwd).")
    args = parser.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+-dev\.\d{8}\.g[0-9a-f]{7}", args.chart_version):
        parser.error(
            f"--chart-version must be X.Y.Z-dev.YYYYMMDD.gSHA7 (got '{args.chart_version}')"
        )
    if not re.fullmatch(r"\d{8}-[0-9a-f]{7}", args.image_tag):
        parser.error(f"--image-tag must be YYYYMMDD-sha7 (got '{args.image_tag}')")

    tokens = [t for t in args.charts.split(",") if t]
    unknown = sorted(set(tokens) - set(CHART_TARGETS))
    if not tokens or unknown:
        parser.error(
            f"--charts must be a non-empty subset of {sorted(CHART_TARGETS)} (got '{args.charts}')"
        )

    root = Path(args.root)
    for token in tokens:
        print(f"{token}:")
        for rel in CHART_TARGETS[token]:
            set_chart_versions(
                root / rel,
                args.chart_version,
                expect_operator_pin=rel.endswith("platform/Chart.yaml"),
            )
        chart_path, nightly_name = NIGHTLY_CHART_NAMES[token]
        set_chart_name(root / chart_path, nightly_name)
        for rel, current_repo, nightly_repo in IMAGE_SITES[token]:
            set_image_site(root / rel, current_repo, nightly_repo, args.image_tag)
        if token == "platform":
            rel, planner_repo = DGDR_DEFAULT_IMAGE_SITE
            set_dgdr_default_image(root / rel, f"{planner_repo}:{args.image_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
