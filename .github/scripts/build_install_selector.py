#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the data for the "Get Dynamo" install-command selector.

Produces ``docs/data/install-selector.json`` (+ a TS module the Fern component
imports): for each backend framework, the stable and nightly Dynamo builds that
ship each backend version, with the complete, copy-pastable install command per
(channel, version, install form). With ``--refresh-support-matrix`` it also
rewrites the ``main (ToT)`` row of the Support Matrix from ``container/context.yaml``.

Data sources (all authoritative, no credentials required):
  * stable release -> backend version -- the "Backend Dependencies" table in
    ``docs/reference/support-matrix.md``.
  * backend version -> nightly window -- git history of ``container/context.yaml``.
  * per-window latest PUBLISHED nightly -- the live NGC tag list (``nvcr.io``
    anonymous pull token) gives the exact ``YYYYMMDD-<sha>`` tag; the wheel version
    derives from ``pyproject.toml`` at that ``<sha>`` and is confirmed against the
    ``pypi.nvidia.com`` index. GC'd or skipped nights never appear, so a dead
    command is never emitted.

Scope: the ``STABLE_RELEASES_BACK`` most recent stable releases and nightly builds
pinned within ``NIGHTLY_DAYS_BACK`` days of the latest published nightly.

Usage:
    build_install_selector.py                        # write JSON + TS module
    build_install_selector.py --refresh-support-matrix   # also refresh the ToT row
    build_install_selector.py --stdout               # print JSON to stdout
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/install-selector.json"
TS_OUT = "fern/components/install-selector-data.ts"
CONTEXT = "container/context.yaml"
SUPPORT_MATRIX = "docs/reference/support-matrix.md"
REGISTRY = "nvcr.io/nvidia/ai-dynamo"
PYPI = "https://pypi.nvidia.com"

STABLE_RELEASES_BACK = 3  # show only the N most recent stable releases
NIGHTLY_DAYS_BACK = 30  # show nightly builds pinned within N days of the latest

# label -> (image stem, wheel extra, has PyPI wheel)
META = {
    "vLLM": ("vllm", "vllm", True),
    "SGLang": ("sglang", "sglang", True),
    "TensorRT-LLM": ("tensorrtllm", "trtllm", False),
}


# --------------------------------------------------------------------------- #
# container/context.yaml — current + historical backend versions (from git)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Framework:
    label: str  # display name
    key: str  # top-level context.yaml key
    device: str  # sub-key holding runtime_image_tag
    version_re: "re.Pattern[str]"  # accepts only real backend tags


# Device keys track the CUDA targets nightly builds. version_re drops pre-layout
# base-image tags (e.g. vLLM 13.0.2) so the history walk keeps only real versions.
FRAMEWORKS = [
    Framework("vLLM", "vllm", "cuda13.0", re.compile(r"^v\d+\.\d+\.\d+")),
    Framework("SGLang", "sglang", "cuda13.0", re.compile(r"^v\d+\.\d+")),
    Framework(
        "TensorRT-LLM", "trtllm", "cuda13.1", re.compile(r"^\d+\.\d+\.\d+(rc\d+)?")
    ),
]


def parse_version(tag: str) -> str:
    """``v0.24.0-ubuntu2404`` -> ``v0.24.0``; ``1.3.0rc20`` -> ``1.3.0rc20``."""
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
        version = parse_version(tag)
        if not fw.version_re.match(version):
            raise SystemExit(
                f"{CONTEXT}: {fw.key}.{fw.device} tag {tag!r} is not a recognized backend version"
            )
        out[fw.label] = (version, tag)
    return out


def read_history(repo_root: Path) -> dict[str, list[tuple[str, str]]]:
    """{label: [(version, start_date), ...]} oldest-first, from git history."""
    lines = (
        git(["log", "--reverse", "--format=%H|%cs", "--", CONTEXT], repo_root)
        .strip()
        .splitlines()
    )
    changes: dict[str, list[tuple[str, str]]] = {fw.label: [] for fw in FRAMEWORKS}
    for line in lines:
        sha, date = line.split("|", 1)
        try:
            doc = yaml.safe_load(git(["show", f"{sha}:{CONTEXT}"], repo_root))
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
                continue
            pts = changes[fw.label]
            if not pts or pts[-1][0] != v:
                pts.append((v, date))
    return changes


# --------------------------------------------------------------------------- #
# Support Matrix — main (ToT) row (Dynamo | SGLang | TensorRT-LLM | vLLM | NIXL);
# we rewrite the three framework cells and leave the trailing NIXL cell alone.
# --------------------------------------------------------------------------- #
TOT_RE = re.compile(r"(?m)^\| \*\*main \(ToT\)\*\* \| `[^`]+` \| `[^`]+` \| `[^`]+` \|")


def update_tot(text: str, current: dict[str, tuple[str, str]]) -> str:
    def bare(version: str) -> str:
        return version[1:] if version.startswith("v") else version

    repl = (
        f"| **main (ToT)** | `{bare(current['SGLang'][0])}` "
        f"| `{bare(current['TensorRT-LLM'][0])}` | `{bare(current['vLLM'][0])}` |"
    )
    new, n = TOT_RE.subn(repl, text)
    if n != 1:
        raise SystemExit(
            f"{SUPPORT_MATRIX}: expected exactly 1 main (ToT) row, found {n}"
        )
    return new


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def norm(label: str, version: str) -> str:
    if label in ("vLLM", "SGLang") and not version.startswith("v"):
        return "v" + version
    return version


def compact(iso_date: str) -> str:
    return iso_date.replace("-", "")


def iso(compact_date: str) -> str:
    return f"{compact_date[:4]}-{compact_date[4:6]}-{compact_date[6:]}"


# --------------------------------------------------------------------------- #
# NGC (anonymous, public images)
# --------------------------------------------------------------------------- #
def _ngc_token(repo: str) -> str:
    url = f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull"
    return json.load(urllib.request.urlopen(url, timeout=30))["token"]


def ngc_tag_list(repo: str) -> list[str]:
    token = _ngc_token(repo)
    url = f"https://nvcr.io/v2/{repo}/tags/list?n=1000"
    tags: list[str] = []
    while url:
        resp = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
            timeout=30,
        )
        tags += json.load(resp).get("tags") or []
        # Docker Registry v2 paginates via a `Link: <...>; rel="next"` header.
        m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
        url = f"https://nvcr.io{m.group(1)}" if m else None
    return tags


def ngc_dated_tags(image: str) -> dict[int, tuple[str, str]]:
    """{yyyymmdd:int -> (compact_date, short_sha)} for a public ``-nightly`` image."""
    out: dict[int, tuple[str, str]] = {}
    for t in ngc_tag_list(f"nvidia/ai-dynamo/{image}-runtime-nightly"):
        m = re.fullmatch(r"(\d{8})-([0-9a-f]{7,40})", t)
        if m:
            out[int(m.group(1))] = (m.group(1), m.group(2))
    return out


def ngc_release_tags(image: str) -> set[str]:
    """Plain ``X.Y.Z`` release tags published for a public ``-runtime`` image."""
    return {
        t
        for t in ngc_tag_list(f"nvidia/ai-dynamo/{image}-runtime")
        if re.fullmatch(r"\d+\.\d+\.\d+", t)
    }


# --------------------------------------------------------------------------- #
# stable map (support-matrix.md), limited to the newest releases
# --------------------------------------------------------------------------- #
_ROW = re.compile(
    r"^\|\s*\*\*(v\d+\.\d+\.\d+)\*\*\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|"
)
# _ROW hard-codes the column order; fail loudly if the table header ever drifts.
_HEADER_RE = re.compile(
    r"^\|\s*\*\*Dynamo\*\*\s*\|\s*\*\*SGLang\*\*\s*\|\s*\*\*TensorRT-LLM\*\*\s*\|\s*\*\*vLLM\*\*\s*\|"
)


def stable_by_framework(live: dict[str, set[str] | None]) -> dict[str, list[dict]]:
    text = (REPO_ROOT / SUPPORT_MATRIX).read_text()
    if not any(_HEADER_RE.match(line.strip()) for line in text.splitlines()):
        raise SystemExit(
            f"{SUPPORT_MATRIX}: Backend Dependencies header changed; _ROW column order is stale"
        )
    releases: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        release, sglang, trtllm, vllm = m.groups()
        releases[release] = {"SGLang": sglang, "TensorRT-LLM": trtllm, "vLLM": vllm}

    def relkey(r: str) -> tuple[int, ...]:
        return tuple(int(x) for x in r.lstrip("v").split("."))

    ordered = sorted(releases, key=relkey, reverse=True)[:STABLE_RELEASES_BACK]
    out: dict[str, list[dict]] = {lbl: [] for lbl in META}
    for release in ordered:
        relnum = release.lstrip("v")
        for label in META:
            img = META[label][0]
            live_tags = live.get(img)
            if live_tags is not None and relnum not in live_tags:
                continue
            bver = norm(label, releases[release][label])
            out[label].append({"backend_version": bver, "release": relnum})
    return out


# --------------------------------------------------------------------------- #
# nightly windows + wheel version (git)
# --------------------------------------------------------------------------- #
def windows(label: str, changes: dict[str, list[tuple[str, str]]]):
    pts = changes[label]
    return [
        (v, start, pts[i + 1][1] if i + 1 < len(pts) else None)
        for i, (v, start) in enumerate(pts)
    ]


def latest_in_window(
    dated: dict[int, tuple[str, str]], start_iso: str, end_iso: str | None
):
    s = int(compact(start_iso))
    e = int(compact(end_iso)) if end_iso else None
    hits = [d for d in dated if d >= s and (e is None or d < e)]
    return dated[max(hits)] if hits else None


def base_version_at(sha: str) -> str | None:
    try:
        blob = git(["show", f"{sha}:pyproject.toml"], REPO_ROOT)
    except Exception:
        return None
    m = re.search(r'(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"', blob)
    return m.group(1) if m else None


def pypi_dev_versions() -> set[str] | None:
    """Published ``ai-dynamo`` nightly dev versions on the NVIDIA index.

    ``None`` means the index couldn't be reached — callers then keep the derived
    pin rather than dropping every wheel command.
    """
    try:
        html = urllib.request.urlopen(f"{PYPI}/ai-dynamo/", timeout=30).read().decode()
    except Exception as exc:
        print(f"warning: pypi.nvidia.com index fetch failed: {exc}", file=sys.stderr)
        return None
    return set(re.findall(r"ai_dynamo-(\d+\.\d+\.\d+\.dev\d{8})", html))


def wheel_version_for(date: str, sha: str, published: set[str] | None) -> str | None:
    """``<pyproject version at sha>.dev<date>`` if that wheel is actually published."""
    base = base_version_at(sha)
    if not base:
        return None
    version = f"{base}.dev{date}"
    if published is not None and version not in published:
        return None  # container shipped but the wheel didn't — don't emit a 404
    return version


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #
def run(image_ref: str) -> str:
    return f"docker run --gpus all --network host --rm -it {image_ref}"


def stable_commands(label: str, release: str) -> dict[str, str]:
    img, extra, wheel = META[label]
    cmds = {"container": run(f"{REGISTRY}/{img}-runtime:{release}")}
    if wheel:
        flag = "--prerelease=allow " if label == "SGLang" else ""
        cmds["wheel"] = f'uv pip install {flag}"ai-dynamo[{extra}]=={release}"'
    return cmds


def nightly_latest_commands(
    label: str, wheel_version: str | None = None
) -> dict[str, str]:
    img, extra, wheel = META[label]
    cmds = {"container": run(f"{REGISTRY}/{img}-runtime-nightly:latest")}
    # Pin the wheel: --extra-index-url pools with public PyPI, so an unpinned --pre
    # install can resolve a stable release. Omit it when we can't pin (rather than
    # emit an unpinned command that may point at a stable version).
    if wheel and wheel_version:
        cmds[
            "wheel"
        ] = f'uv pip install --pre --extra-index-url {PYPI}/ "ai-dynamo[{extra}]=={wheel_version}"'
    return cmds


def nightly_pinned_commands(
    label: str, date: str, sha: str, wheel_version: str | None
) -> dict[str, str]:
    img, extra, wheel = META[label]
    cmds = {"container": run(f"{REGISTRY}/{img}-runtime-nightly:{date}-{sha}")}
    if wheel and wheel_version:
        cmds[
            "wheel"
        ] = f'uv pip install --pre --extra-index-url {PYPI}/ "ai-dynamo[{extra}]=={wheel_version}"'
    return cmds


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build() -> dict:
    changes = read_history(REPO_ROOT)

    dated_by_img: dict[str, dict] = {}
    reltags_by_img: dict[str, set[str] | None] = {}
    for label in META:
        img = META[label][0]
        try:
            dated_by_img[img] = ngc_dated_tags(img)
        except Exception as exc:  # keep build resilient; current build still works
            print(f"warning: NGC nightly tags failed for {img}: {exc}", file=sys.stderr)
            dated_by_img[img] = {}
        try:
            reltags_by_img[img] = ngc_release_tags(img)
        except Exception as exc:
            # None means "couldn't verify" — keep all releases rather than dropping
            # every stable option (which would blank the default view).
            print(f"warning: NGC release tags failed for {img}: {exc}", file=sys.stderr)
            reltags_by_img[img] = None

    published = pypi_dev_versions()
    stable = stable_by_framework(reltags_by_img)
    data: dict[str, dict] = {}

    for fw in FRAMEWORKS:
        label = fw.label
        img, extra, has_wheel = META[label]
        entry: dict = {
            "label": label,
            "image": img,
            "extra": extra,
            "wheel": has_wheel,
            "stable": [],
            "nightly": [],
        }

        for s in stable[label]:
            entry["stable"].append(
                {
                    "backend_version": s["backend_version"],
                    "dynamo": s["release"],
                    "commands": stable_commands(label, s["release"]),
                }
            )

        dated = dated_by_img[img]
        cutoff = None
        if dated:
            newest = max(dated)
            cutoff = int(
                (
                    datetime.strptime(str(newest), "%Y%m%d")
                    - timedelta(days=NIGHTLY_DAYS_BACK)
                ).strftime("%Y%m%d")
            )

        wins = windows(label, changes)
        for idx in range(len(wins) - 1, -1, -1):  # newest-first
            version, start, end = wins[idx]
            if end is None:  # current build
                hit = latest_in_window(dated, start, None)
                wheel_version = wheel_version_for(*hit, published) if hit else None
                entry["nightly"].append(
                    {
                        "backend_version": version,
                        "latest": True,
                        "commands": nightly_latest_commands(label, wheel_version),
                    }
                )
                continue
            hit = latest_in_window(dated, start, end)
            if not hit:
                continue
            date, sha = hit
            if cutoff is not None and int(date) < cutoff:
                continue
            wheel_version = wheel_version_for(date, sha, published)
            entry["nightly"].append(
                {
                    "backend_version": version,
                    "window": [start, end],
                    "pin_date": iso(date),
                    "note": f"In nightlies {start} → {end}; pinned to {iso(date)} (latest in range).",
                    "commands": nightly_pinned_commands(
                        label, date, sha, wheel_version
                    ),
                }
            )

        data[img] = entry

    return {
        "note": "Generated by .github/scripts/build_install_selector.py — do not edit.",
        "frameworks": data,
    }


def as_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def as_ts_module(data: dict) -> str:
    """A TypeScript module the Fern component imports (avoids JSON-loader config)."""
    header = (
        "// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// Generated by .github/scripts/build_install_selector.py — do not edit.\n\n"
    )
    body = json.dumps(data["frameworks"], indent=2, ensure_ascii=False)
    return (
        f"{header}export const INSTALL_DATA = {body};\n\nexport default INSTALL_DATA;\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stdout", action="store_true", help="print JSON instead of writing"
    )
    ap.add_argument(
        "--refresh-support-matrix",
        action="store_true",
        help="also rewrite the main (ToT) row in support-matrix.md",
    )
    ap.add_argument("--out", default=OUT, help="JSON data path")
    ap.add_argument(
        "--ts-out", default=TS_OUT, help="TypeScript module path for the component"
    )
    args = ap.parse_args()

    data = build()
    json_text = as_json(data)

    if args.stdout:
        sys.stdout.write(json_text)
        return 0

    for rel, text in ((args.out, json_text), (args.ts_out, as_ts_module(data))):
        target = REPO_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"wrote {rel}")

    if args.refresh_support_matrix:
        sm = REPO_ROOT / SUPPORT_MATRIX
        sm.write_text(update_tot(sm.read_text(), read_current(REPO_ROOT)))
        print(f"refreshed {SUPPORT_MATRIX} main (ToT) row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
