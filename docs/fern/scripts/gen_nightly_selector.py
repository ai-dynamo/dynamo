#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the nightly dimension of the install selectors.

Emits ``docs/fern/components/nightly-selector-data.generated.ts``: for each
backend, the ``NIGHTLY_VERSIONS_BACK`` most recent backend versions a nightly
shipped, each paired with the newest nightly that shipped it, plus the recent
nightly-wheel ledger used by Release Artifacts. The module is gitignored and
rebuilt by the docs workflow on every publish, so the site never serves a pin
that a human forgot to refresh.

Data sources (all authoritative and anonymous):
  * which nightlies exist, and the commit each was built from -- the dated
    ``YYYYMMDD-<sha>`` tags on the public ``*-runtime-nightly`` NGC repos. The
    tag names its own commit, so no build-time inference is needed.
  * backend versions per nightly -- ``container/context.yaml`` at that commit.
  * wheel version -- ``pyproject.toml`` at that commit, confirmed against the
    pypi.nvidia.com ``ai-dynamo`` index. A night whose wheel was skipped or
    garbage-collected keeps its container command and drops its wheel command,
    so a dead install line is never emitted.
  * the Release Artifacts ledger -- the same wheel indexes, and nothing else. A
    row is one nightly whose required wheels all published, so the ledger names
    the newest installable nightly. The selectors' ``latest`` rows take their
    wheel from that same set, so the ledger is never older than they are. It can
    be newer: a ``latest`` row falls back to the wheel of its own container-tag
    night when a backend pin moved since that night.

Stable and source-build entries are NOT generated here; they stay in
``components/releases.data.ts``, which remains the source of truth for released
artifacts.

Needs full history: run ``git fetch --unshallow`` first on a shallow clone, or
the commit lookups silently lose old nightlies.

Usage:
    gen_nightly_selector.py            # write the TS module
    gen_nightly_selector.py --stdout   # print it instead
    gen_nightly_selector.py --offline  # write an empty module, no network
    gen_nightly_selector.py --out PATH # write somewhere other than the source tree
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "docs/fern/components/nightly-selector-data.generated.ts"
CONTEXT = "container/context.yaml"
PYPROJECT = "pyproject.toml"
# Files that pin a backend version: context.yaml for the container, pyproject
# for the wheel extras. A commit touching either can move a row's claim.
PIN_FILES = (CONTEXT, PYPROJECT)
PYPI = "https://pypi.nvidia.com"
NGC_NAMESPACE = "nvidia/ai-dynamo"

# How many distinct backend versions each selector row offers.
NIGHTLY_VERSIONS_BACK = 3
# How many dated nightly wheels the Release Artifacts ledger shows.
NIGHTLY_LEDGER_BUILDS = 3
# Dated tags to walk back through when hunting for those versions.
MAX_TAGS = 120
# Packages a ledger row must have published before the row is written. These are
# the wheels its install commands resolve: ai-dynamo pins ai-dynamo-runtime to
# the same version, so a night missing either one is not installable and must not
# be advertised.
NIGHTLY_LEDGER_REQUIRED_PACKAGES = ["ai-dynamo", "ai-dynamo-runtime"]
# Packages the ledger advertises when that night published them. Not required:
# kvbm is deprecated with removal targeted for v1.6.0, and a package that stops
# publishing must drop out of the Packages column rather than freeze the ledger.
NIGHTLY_LEDGER_OPTIONAL_PACKAGES = ["kvbm"]
# Every package the ledger can advertise, the required ones first.
NIGHTLY_PACKAGES = [
    *NIGHTLY_LEDGER_REQUIRED_PACKAGES,
    *NIGHTLY_LEDGER_OPTIONAL_PACKAGES,
]

TIMEOUT = 30
# Endpoint-unreachable failures degrade gracefully (skip the backend); anything
# else -- a malformed or unexpected response -- must fail the generation run.
TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError)


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
    blob = blob_at(sha, PYPROJECT)
    if blob is None:
        return None
    m = re.search(r'(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"', blob)
    return m.group(1) if m else None


def pins_moved_since(sha: str) -> bool:
    """Did any backend pin change between ``sha`` and ``HEAD``? Unknown counts as yes."""
    try:
        return bool(git(["log", "--oneline", f"{sha}..HEAD", "--", *PIN_FILES]).strip())
    except subprocess.CalledProcessError:
        return True


# --------------------------------------------------------------------------- #
# NGC (public images, anonymous pull token)
# --------------------------------------------------------------------------- #
def ngc_tag_list(repo: str) -> list[str]:
    token_url = f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull"
    with urllib.request.urlopen(token_url, timeout=TIMEOUT) as resp:
        token = json.load(resp)["token"]
    url = f"https://nvcr.io/v2/{repo}/tags/list?n=1000"
    tags: list[str] = []
    while url:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
            timeout=TIMEOUT,
        ) as resp:
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
    except TRANSPORT_ERRORS as exc:
        warn(f"NGC tag list for {repo} failed: {exc}")
        return []
    except Exception as exc:
        warn(f"NGC tag list for {repo} returned an unexpected response: {exc}")
        raise
    found = {}
    for tag in tags:
        m = re.fullmatch(r"(\d{8})-([0-9a-f]{7,40})", tag)
        if m:
            found[m.group(1)] = m.group(2)
    return sorted(found.items(), reverse=True)[:MAX_TAGS]


# --------------------------------------------------------------------------- #
# PyPI
# --------------------------------------------------------------------------- #
def published_wheels(package: str) -> set[str] | None:
    """Published dev versions for ``package``; ``None`` when its index is unreachable."""
    try:
        with urllib.request.urlopen(f"{PYPI}/{package}/", timeout=TIMEOUT) as resp:
            html = resp.read().decode()
    except TRANSPORT_ERRORS as exc:
        warn(f"pypi.nvidia.com index fetch for {package} failed: {exc}")
        return None
    except Exception as exc:
        warn(
            f"pypi.nvidia.com index for {package} returned an unexpected response: {exc}"
        )
        raise
    normalized = package.replace("-", "_")
    return set(
        re.findall(rf"{re.escape(normalized)}-(\d+\.\d+\.\d+\.dev\d{{8}})", html)
    )


def published_nightly_packages() -> dict[str, set[str]] | None:
    """Published dev versions per ledger package; ``None`` when a required one fails.

    A required package missing from the index would leave every ledger row
    unverifiable, so that fails the run. An optional package drops out of the
    Packages column instead: kvbm's removal is targeted for v1.6.0, and its index
    going away must not stop every docs publish.
    """
    published: dict[str, set[str]] = {}
    for package in NIGHTLY_LEDGER_REQUIRED_PACKAGES:
        versions = published_wheels(package)
        if versions is None:
            return None
        published[package] = versions
    for package in NIGHTLY_LEDGER_OPTIONAL_PACKAGES:
        published[package] = published_wheels(package) or set()
    return published


def wheel_for(yyyymmdd: str, sha: str, published: set[str]) -> str | None:
    base = base_version_at(sha)
    if not base:
        return None
    version = f"{base}.dev{yyyymmdd}"
    return version if version in published else None


def run_wheel(nights: list[tuple[str, str]], published: set[str]) -> str | None:
    """Newest wheel published by a night that shipped this backend version."""
    for yyyymmdd, sha in nights:
        wheel = wheel_for(yyyymmdd, sha, published)
        if wheel:
            return wheel
    return None


def wheel_date(version: str) -> str:
    return version.partition(".dev")[2]


def version_key(version: str) -> tuple[str, tuple[int, int, int]]:
    """Order dev versions by night first, then by base version."""
    base, _, yyyymmdd = version.partition(".dev")
    major, minor, patch = (int(part) for part in base.split("."))
    return (yyyymmdd, (major, minor, patch))


def newest_published(published: set[str]) -> str | None:
    """The newest installable dev wheel, independent of any NGC tag."""
    if not published:
        return None
    return max(published, key=version_key)


def ledger_version_published(version: str, published: dict[str, set[str]]) -> bool:
    """Whether every required wheel for ``version`` is on the index.

    A ledger row and a wheel command make the same claim, so this is also the
    installability rule: ``ai-dynamo`` pins ``ai-dynamo-runtime`` to the exact
    version, and a night missing either wheel offers nothing to install.
    """
    return all(
        version in published[package] for package in NIGHTLY_LEDGER_REQUIRED_PACKAGES
    )


def installable_wheels(published: dict[str, set[str]]) -> set[str]:
    """Versions that pass ``ledger_version_published``, i.e. that pip can resolve.

    Both the selector rows and the ledger rows resolve their wheel commands from
    this one set, which is what keeps their newest nightly the same version.
    ``ai-dynamo`` is the package the selector pins, and a version it never
    published is not installable whatever its companions did.
    """
    return {
        version
        for version in published["ai-dynamo"]
        if ledger_version_published(version, published)
    }


def ledger_packages_published(
    version: str, published: dict[str, set[str]]
) -> list[str]:
    """The advertised packages that published ``version``, required ones first."""
    return [package for package in NIGHTLY_PACKAGES if version in published[package]]


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def pretty_date(yyyymmdd: str) -> str:
    d = date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:]))
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


@dataclass(frozen=True)
class NightlyBackendBuild:
    """One selector row. Field names are the generated TS property names."""

    backend: str
    backendVersion: str
    dynamo: str | None
    date: str
    tag: str
    latest: bool


@dataclass(frozen=True)
class NightlyBuild:
    """One Release Artifacts ledger row: a nightly whose wheels all published."""

    version: str
    date: str
    packages: list[str]


def build_ledger(
    wheels: set[str], published: dict[str, set[str]]
) -> list[NightlyBuild]:
    """The ``NIGHTLY_LEDGER_BUILDS`` newest nights of the installable wheel train.

    ``wheels`` is ``installable_wheels()``: the same set the selectors' ``latest``
    rows pick their wheel from, so the ledger cannot name an older nightly than
    they do. ``published`` is here for the Packages column, which names the
    packages that actually published that night.
    """
    ledger: list[NightlyBuild] = []
    for version in sorted(wheels, key=version_key, reverse=True):
        ledger.append(
            NightlyBuild(
                version=version,
                date=pretty_date(wheel_date(version)),
                packages=ledger_packages_published(version, published),
            )
        )
        if len(ledger) == NIGHTLY_LEDGER_BUILDS:
            break
    return ledger


def build(published: set[str]) -> list[NightlyBackendBuild]:
    rows: list[NightlyBackendBuild] = []

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
            if index == 0:
                # Latest row: the container side points at the rolling
                # *-runtime-nightly:latest tag, so the wheel may come from a later
                # night -- the two jobs do not always land together. Only while it
                # provably carries this row's backend version, though: no earlier
                # than the container commit, and no pin moved since. Otherwise use
                # this version's own run, whose commits were read.
                yyyymmdd, sha = nights[0]
                wheel = newest_published(published)
                if not wheel or wheel_date(wheel) < yyyymmdd or pins_moved_since(sha):
                    wheel = run_wheel(nights, published)
            else:
                # Pinned rows describe one immutable build, so the wheel has to
                # come from the exact commit the container tag names. Fall back to
                # the newest night in the group, which still has an immutable
                # container tag, when none of its nights published a wheel.
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
                NightlyBackendBuild(
                    backend=fw.backend,
                    backendVersion=version,
                    dynamo=wheel,
                    date=pretty_date(yyyymmdd),
                    tag=f"{yyyymmdd}-{sha}",
                    latest=index == 0,
                )
            )

    # The ledger is not assembled here. It describes the wheel train, not one
    # backend selector row, and every runtime repository carrying a dated tag
    # that night says nothing about which wheels exist. build_ledger() reads the
    # same installable wheel set the rows above take their commands from.
    return rows


def as_ts(rows: list[NightlyBackendBuild], ledger: list[NightlyBuild]) -> str:
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
        "export interface NightlyBuild {",
        "  version: string;",
        "  date: string;",
        "  packages: string[];",
        "  note?: string;",
        "}",
        "",
        "export const NIGHTLY_BACKEND_BUILDS: NightlyBackendBuild[] = [",
    ]
    for row in rows:
        fields = ", ".join(
            f"{key}: {ts(value)}"
            for key, value in asdict(row).items()
            if not (key == "latest" and not value)
        )
        lines.append(f"  {{ {fields} }},")
    lines += ["];", "", "export const NIGHTLY_BUILDS: NightlyBuild[] = ["]
    for build in ledger:
        fields = ", ".join(
            f"{key}: {ts(value)}" for key, value in asdict(build).items()
        )
        lines.append(f"  {{ {fields} }},")
    lines += ["];", "", "export default NIGHTLY_BACKEND_BUILDS;", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="print instead of write")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the network and write an empty module",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="destination module (default: the source tree copy)",
    )
    args = parser.parse_args(argv)

    rows: list[NightlyBackendBuild] = []
    ledger: list[NightlyBuild] = []
    if args.offline:
        # Local previews and the composition replay only need the module to
        # exist so the component import resolves; the selector and the ledger
        # render their unavailable state from an empty list.
        warn("offline: writing an empty module, nightly rows omitted")
    else:
        # These guards protect the same invariant: the publish job syncs whatever
        # this writes, so a module that resolved nothing would replace the live
        # selector's rows with nothing. Fail and leave the published copy alone.
        published = published_nightly_packages()
        if published is None:
            print(
                "error: a required pypi.nvidia.com package index is unreachable; "
                "refusing to write a module with incomplete package claims",
                file=sys.stderr,
            )
            return 1
        # One installable wheel set feeds both views, so the ledger cannot name
        # an older nightly than the selector rows do. NGC tags are not consulted:
        # a backend's tag list going missing must not cost the ledger a row.
        wheels = installable_wheels(published)
        rows = build(wheels)
        if not rows:
            print(
                "error: no nightly data resolved; refusing to write an empty module",
                file=sys.stderr,
            )
            return 1
        ledger = build_ledger(wheels, published)
        if len(ledger) < NIGHTLY_LEDGER_BUILDS:
            print(
                f"error: fewer than {NIGHTLY_LEDGER_BUILDS} complete nightly ledger "
                "rows resolved; refusing to replace the published ledger",
                file=sys.stderr,
            )
            return 1

    module = as_ts(rows, ledger)
    if args.stdout:
        print(module)
        return 0

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(module)
    print(f"{out}: wrote {len(rows)} nightly build(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
