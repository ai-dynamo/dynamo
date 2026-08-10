#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Point the Helm charts at this nightly run's staged images, in place.

Invoked by release.yml's nightly helm step in the runner workspace, right
before `helm dep build` + `helm package`. rc/GA releases record chart versions
and image tags on their release branch; nightly has no release branch, so the
rewrite is ephemeral: checkout at the source SHA, mutate, package, discard.

Per selected chart token this rewrites:
  platform:
    - deploy/helm/charts/platform/Chart.yaml: top-level version, plus the
      dynamo-operator file:// dependency pin -- the pin is exact, so it must
      move in lockstep with the subchart version or `helm dep build` fails
      its constraint check.
    - deploy/helm/charts/platform/components/operator/Chart.yaml: version AND
      appVersion. `helm package --app-version` stamps only the top-level
      chart; the subchart's appVersion feeds the operator image-tag default
      (`tag: ""` falls through to .Chart.AppVersion) and the
      `--operator-version` arg (semver-validated at operator startup), so it
      must be rewritten in source before `helm dep build` packs the subchart.
    - values.yaml (platform and subchart copies): the operator image
      repository is renamed to its -nightly NGC repo and the tag pinned to
      this run's dated tag -- exactly the image the copy step staged.
  snapshot:
    - deploy/helm/charts/snapshot/Chart.yaml: version and appVersion.
    - values.yaml: snapshot-agent repository -> snapshot-agent-nightly, tag
      pinned (that tag is a hard literal with no appVersion fallback).

Third-party references (nats, etcd, busybox, kai-scheduler, grove) are never
touched. power-agent is excluded: no release path publishes its image.

Every rewrite requires exactly one match and raises otherwise, so a chart
restructure breaks this step loudly instead of publishing a chart that points
at a stale or missing image.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (helm token) -> Chart.yaml files carrying the chart version. The operator
# subchart rides the "platform" token, mirroring the release-branch bump
# tooling's HELM_CHART_TARGETS.
CHART_TARGETS: dict[str, list[str]] = {
    "platform": [
        "deploy/helm/charts/platform/Chart.yaml",
        "deploy/helm/charts/platform/components/operator/Chart.yaml",
    ],
    "snapshot": [
        "deploy/helm/charts/snapshot/Chart.yaml",
    ],
}

# (helm token) -> first-party image sites in values.yaml:
# (path, repository as written in the file, nightly repository to write).
# The subchart values are rewritten too so the defaults are correct when the
# operator chart is consumed directly.
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
    "snapshot": [
        (
            "deploy/helm/charts/snapshot/values.yaml",
            "nvcr.io/nvidia/ai-dynamo/snapshot-agent",
            "nvcr.io/nvidia/ai-dynamo/snapshot-agent-nightly",
        ),
    ],
}

# Bounds the repository->tag hop at the next `repository:` line, so an image
# block with no tag fails loudly instead of rewriting another image's tag.
_TAG_HOP = r"(?:(?![^\n]*repository:)[^\n]*\n)*?"


def set_chart_versions(path: Path, new: str, expect_operator_pin: bool) -> None:
    text = path.read_text()

    # Top-level version/appVersion (line-start anchored, so dependency and
    # kubeVersion lines never match). The platform chart has no appVersion
    # line -- `helm package --app-version` injects it at package time.
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

    # dynamo-operator (file:// subchart) exact pin. Hop bounded to the entry so
    # it can't reach the nats/etcd/kai/grove pins, which stay untouched.
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


def set_image_site(path: Path, current_repo: str, nightly_repo: str, tag: str) -> None:
    # Rename the repository and pin the tag in one bounded match, so they can
    # never come from different image blocks. `\2` (the closing quote backref)
    # right after the repo also stops the pattern matching a repo that merely
    # starts with `current_repo`.
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
        help="Comma-separated chart tokens to rewrite (subset of: platform,snapshot).",
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
        for rel, current_repo, nightly_repo in IMAGE_SITES[token]:
            set_image_site(root / rel, current_repo, nightly_repo, args.image_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
