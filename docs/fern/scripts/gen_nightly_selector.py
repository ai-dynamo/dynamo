#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the nightly dimension of the install selectors.

Emits ``docs/fern/components/nightly-selector-data.generated.ts``: for each
backend, the ``NIGHTLY_VERSIONS_BACK`` most recent backend versions a nightly
shipped, each paired with the newest nightly that shipped it. The module is
gitignored and rebuilt by the docs workflow on every publish, so the site never
serves a pin that a human forgot to refresh.

Data sources (all authoritative and anonymous):
  * which nightlies exist, and the commit each was built from -- the dated
    ``YYYYMMDD-<sha>`` tags on the public ``*-runtime-nightly`` NGC repos. The
    tag names its own commit, so no build-time inference is needed.
  * backend versions per nightly -- ``container/context.yaml`` at that commit.
  * wheel version -- ``pyproject.toml`` at that commit, confirmed against the
    pypi.nvidia.com ``ai-dynamo`` index. A night whose wheel was skipped or
    garbage-collected keeps its container command and drops its wheel command,
    so a dead install line is never emitted.

Stable and source-build entries are NOT generated here; they stay in
``components/releases.data.ts``, which remains the source of truth for released
artifacts.

Needs full history: run ``git fetch --unshallow`` first on a shallow clone, or
the commit lookups silently lose old nightlies.

Usage:
    gen_nightly_selector.py            # write the TS module
    gen_nightly_selector.py --stdout   # print it instead
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs/fern/components/nightly-selector-data.generated.ts"
CONTEXT = "container/context.yaml"
PYPI = "https://pypi.nvidia.com"
NGC_NAMESPACE = "nvidia/ai-dynamo"

# How many distinct backend versions each selector row offers.
NIGHTLY_VERSIONS_BACK = 3
# Dated tags to walk back through when hunting for those versions.
MAX_TAGS = 120

TIMEOUT = 30


@dataclass(frozen=True)
class Framework:
    backend: str  # selector id, matches InstallBackend
    image: str  # NGC image stem
    key: str  # top-level container/context.yaml key
    device: str  # sub-key holding runtime_image_tag
    version_re: "re.Pattern[str]"  # rejects pre-layout base-image tags


FRAMEWORKS = [
    Framework("sglang", "sglang", "sglang", "cuda13.0", re.compile(r"^v?\d+\.\d+")),
    Framework(
        "trtllm",
        "tensorrtllm",
        "trtllm",
        "cuda13.1",
        re.compile(r"^\d+\.\d+\.\d+(rc\d+)?"),
    ),
    Framework("vllm", "vllm", "vllm", "cuda13.0", re.compile(r"^v?\d+\.\d+\.\d+")),
]

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def blob_at(sha: str, path: str) -> str | None:
    try:
        return git(["show", f"{sha}:{path}"])
    except subprocess.CalledProcessError:
        return None


def context_at(sha: str) -> dict | None:
    blob = blob_at(sha, CONTEXT)
    if blob is None:
        return None
    try:
        doc = yaml.safe_load(blob)
    except yaml.YAMLError:
        return None
    return doc if isinstance(doc, dict) else None


def backend_version(doc: dict, fw: Framework) -> str | None:
    """``v0.27.1-ubuntu2404`` -> ``0.27.1``; rejects base-image tags."""
    node = doc.get(fw.key)
    if not isinstance(node, dict):
        return None
    device = node.get(fw.device)
    if not isinstance(device, dict):
        return None
    tag = device.get("runtime_image_tag")
    if not isinstance(tag, str) or not fw.version_re.match(tag):
        return None
    version = re.sub(r"^v", "", tag)
    return re.sub(r"-ubuntu\d+$|-cu\d+-runtime$", "", version)


def base_version_at(sha: str) -> str | None:
    blob = blob_at(sha, "pyproject.toml")
    if blob is None:
        return None
    m = re.search(r'(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"', blob)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# NGC (public images, anonymous pull token)
# --------------------------------------------------------------------------- #
def ngc_tag_list(repo: str) -> list[str]:
    token_url = f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull"
    token = json.load(urllib.request.urlopen(token_url, timeout=TIMEOUT))["token"]
    url = f"https://nvcr.io/v2/{repo}/tags/list?n=1000"
    tags: list[str] = []
    while url:
        resp = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
            timeout=TIMEOUT,
        )
        tags += json.load(resp).get("tags") or []
        # Docker Registry v2 paginates via `Link: <...>; rel="next"`.
        m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
        url = f"https://nvcr.io{m.group(1)}" if m else None
    return tags


def dated_tags(image: str) -> list[tuple[str, str]]:
    """``[(yyyymmdd, short_sha), ...]`` newest first, for a ``-nightly`` repo."""
    repo = f"{NGC_NAMESPACE}/{image}-runtime-nightly"
    try:
        tags = ngc_tag_list(repo)
    except Exception as exc:  # noqa: BLE001 - any network failure degrades the same way
        warn(f"NGC tag list for {repo} failed: {exc}")
        return []
    found = {}
    for tag in tags:
        m = re.fullmatch(r"(\d{8})-([0-9a-f]{7,40})", tag)
        if m:
            found[m.group(1)] = m.group(2)
    return sorted(found.items(), reverse=True)[:MAX_TAGS]


# --------------------------------------------------------------------------- #
# PyPI
# --------------------------------------------------------------------------- #
def published_wheels() -> set[str] | None:
    """Published ``ai-dynamo`` dev versions; ``None`` when the index is unreachable."""
    try:
        html = (
            urllib.request.urlopen(f"{PYPI}/ai-dynamo/", timeout=TIMEOUT)
            .read()
            .decode()
        )
    except Exception as exc:  # noqa: BLE001
        warn(f"pypi.nvidia.com index fetch failed: {exc}; wheel commands omitted")
        return None
    return set(re.findall(r"ai_dynamo-(\d+\.\d+\.\d+\.dev\d{8})", html))


def wheel_for(yyyymmdd: str, sha: str, published: set[str] | None) -> str | None:
    base = base_version_at(sha)
    if not base:
        return None
    version = f"{base}.dev{yyyymmdd}"
    if published is None or version not in published:
        return None
    return version


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def pretty_date(yyyymmdd: str) -> str:
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:]))
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def build() -> list[dict]:
    published = published_wheels()
    rows: list[dict] = []

    for fw in FRAMEWORKS:
        tags = dated_tags(fw.image)
        if not tags:
            warn(f"{fw.backend}: no dated nightly tags found; skipping")
            continue

        # Group nightly tags by the backend version each shipped, newest first,
        # so a version's whole run is available when picking a representative.
        runs: dict[str, list[tuple[str, str]]] = {}
        order: list[str] = []
        unresolved = 0
        for yyyymmdd, sha in tags:
            doc = context_at(sha)
            if doc is None:
                # Commit missing (shallow clone, or the branch was GC'd). Guessing
                # what it carried is how stale pins get invented.
                unresolved += 1
                continue
            version = backend_version(doc, fw)
            if not version:
                continue
            if version not in runs:
                if len(order) >= NIGHTLY_VERSIONS_BACK:
                    break
                runs[version] = []
                order.append(version)
            runs[version].append((yyyymmdd, sha))

        if unresolved:
            warn(
                f"{fw.backend}: {unresolved} nightly tag(s) name a commit this clone "
                "does not have; unshallow the checkout"
            )
        if len(order) < NIGHTLY_VERSIONS_BACK:
            warn(
                f"{fw.backend}: only {len(order)} distinct backend version(s) across "
                f"{len(tags)} nightly tags"
            )

        for index, version in enumerate(order):
            nights = runs[version]
            # Prefer the newest night that also published a wheel, so the container
            # and wheel commands describe the same build. Fall back to the newest
            # night, which still has an immutable container tag.
            chosen = None
            for yyyymmdd, sha in nights:
                wheel = wheel_for(yyyymmdd, sha, published)
                if wheel:
                    chosen = (yyyymmdd, sha, wheel)
                    break
            if chosen is None:
                chosen = (nights[0][0], nights[0][1], None)
            yyyymmdd, sha, wheel = chosen
            rows.append(
                {
                    "backend": fw.backend,
                    "backendVersion": version,
                    "dynamo": wheel,
                    "date": pretty_date(yyyymmdd),
                    "tag": f"{yyyymmdd}-{sha}",
                    "latest": index == 0,
                }
            )

    return rows


def as_ts(rows: list[dict]) -> str:
    def ts(value) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        return json.dumps(value)

    lines = [
        "/*",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        " * SPDX-License-Identifier: Apache-2.0",
        " *",
        " * GENERATED FILE - DO NOT EDIT.",
        " * Written by docs/fern/scripts/gen_nightly_selector.py and gitignored;",
        " * the Fern docs workflow rebuilds it on every publish.",
        " */",
        "",
        "export interface NightlyBackendBuild {",
        '  backend: "sglang" | "trtllm" | "vllm";',
        "  backendVersion: string;",
        "  /** Newest nightly wheel that shipped this backend version, null when unpublished. */",
        "  dynamo: string | null;",
        "  date: string;",
        "  /** Immutable NGC nightly tag, YYYYMMDD-<short sha>. */",
        "  tag: string;",
        "  /** Tip of main: the rolling *-runtime-nightly:latest tag points here. */",
        "  latest?: boolean;",
        "}",
        "",
        "export const NIGHTLY_BACKEND_BUILDS: NightlyBackendBuild[] = [",
    ]
    for row in rows:
        fields = ", ".join(
            f"{key}: {ts(value)}"
            for key, value in row.items()
            if not (key == "latest" and not value)
        )
        lines.append(f"  {{ {fields} }},")
    lines += ["];", "", "export default NIGHTLY_BACKEND_BUILDS;", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="print instead of write")
    args = parser.parse_args()

    rows = build()
    module = as_ts(rows)

    if args.stdout:
        print(module)
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(module)
    print(f"{OUT.relative_to(REPO_ROOT)}: wrote {len(rows)} nightly build(s)")
    if not rows:
        warn("no nightly data resolved; the selector will hide its nightly channel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
