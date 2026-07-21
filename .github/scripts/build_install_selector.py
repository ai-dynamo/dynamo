#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the data file for the "Get Dynamo" install-command selector.

Produces ``docs/data/install-selector.json``: for each backend framework, the
stable and nightly Dynamo builds that ship each backend version, with the
complete, copy-pastable install command per (channel, version, install form).

Data sources (all authoritative, no credentials required):
  * stable release -> backend version -- the "Backend Dependencies" table in
    ``docs/reference/support-matrix.md``.
  * backend version -> nightly window -- git history of ``container/context.yaml``
    (reused from ``sync_nightly_backend_versions``).
  * per-window latest PUBLISHED nightly -- the live NGC tag list
    (``nvcr.io`` anonymous pull token) gives the exact ``YYYYMMDD-<sha>`` tag; the
    wheel version derives from ``pyproject.toml`` at that ``<sha>``. GC'd or
    skipped nights simply never appear, so a dead command is never emitted.

Scope: the ``STABLE_RELEASES_BACK`` most recent stable releases and nightly builds
pinned within ``NIGHTLY_DAYS_BACK`` days of the latest published nightly.

Usage:
    build_install_selector.py                 # write docs/data/install-selector.json
    build_install_selector.py --stdout        # print JSON to stdout
    build_install_selector.py --check         # exit 1 if the on-disk file is stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sync_nightly_backend_versions as sync  # noqa: E402  (reuse history helpers)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/install-selector.json"
TS_OUT = "fern/components/install-selector-data.ts"
SUPPORT_MATRIX = "docs/reference/support-matrix.md"
REGISTRY = "nvcr.io/nvidia/ai-dynamo"
PYPI = "https://pypi.nvidia.com"

STABLE_RELEASES_BACK = 3  # show only the N most recent stable releases
NIGHTLY_DAYS_BACK = 30    # show nightly builds pinned within N days of the latest

# label (as used by sync.FRAMEWORKS) -> (image stem, wheel extra, has PyPI wheel)
META = {
    "vLLM": ("vllm", "vllm", True),
    "SGLang": ("sglang", "sglang", True),
    "TensorRT-LLM": ("tensorrtllm", "trtllm", False),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def norm(label: str, version: str) -> str:
    """Display form: vLLM/SGLang carry a leading ``v``; TensorRT-LLM does not."""
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
    """All tags for a public nvcr.io repo (anonymous pull token)."""
    req = urllib.request.Request(
        f"https://nvcr.io/v2/{repo}/tags/list",
        headers={"Authorization": f"Bearer {_ngc_token(repo)}"},
    )
    return json.load(urllib.request.urlopen(req, timeout=30)).get("tags") or []


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
        t for t in ngc_tag_list(f"nvidia/ai-dynamo/{image}-runtime")
        if re.fullmatch(r"\d+\.\d+\.\d+", t)
    }


# --------------------------------------------------------------------------- #
# stable map (support-matrix.md), limited to the newest releases
# --------------------------------------------------------------------------- #
_ROW = re.compile(
    r"^\|\s*\*\*(v\d+\.\d+\.\d+)\*\*\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|"
)


def stable_by_framework(live: dict[str, set[str]]) -> dict[str, list[dict]]:
    """{label: [{backend_version, releases:[newest..]}]} for the newest releases.

    Limited to the ``STABLE_RELEASES_BACK`` newest releases, and only releases whose
    ``{image}-runtime:{release}`` tag is actually published.
    """
    text = (REPO_ROOT / SUPPORT_MATRIX).read_text()
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
            if relnum not in live.get(img, set()):
                continue  # no published image for this framework at this release
            bver = norm(label, releases[release][label])
            existing = next((e for e in out[label] if e["backend_version"] == bver), None)
            if existing:
                existing["releases"].append(relnum)
            else:
                out[label].append({"backend_version": bver, "releases": [relnum]})
    return out


# --------------------------------------------------------------------------- #
# nightly windows (git) + wheel version (git)
# --------------------------------------------------------------------------- #
def windows(label: str, changes: dict[str, list[tuple[str, str]]]):
    """[(version, start_iso, end_iso|None)] oldest-first; last is the current build."""
    pts = changes[label]
    return [
        (v, start, pts[i + 1][1] if i + 1 < len(pts) else None)
        for i, (v, start) in enumerate(pts)
    ]


def latest_in_window(dated: dict[int, tuple[str, str]], start_iso: str, end_iso: str | None):
    s = int(compact(start_iso))
    e = int(compact(end_iso)) if end_iso else None
    hits = [d for d in dated if d >= s and (e is None or d < e)]
    return dated[max(hits)] if hits else None


def base_version_at(sha: str) -> str | None:
    try:
        blob = sync.git(["show", f"{sha}:pyproject.toml"], REPO_ROOT)
    except Exception:
        return None
    m = re.search(r'(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"', blob)
    return m.group(1) if m else None


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


def nightly_latest_commands(label: str) -> dict[str, str]:
    img, extra, wheel = META[label]
    cmds = {"container": run(f"{REGISTRY}/{img}-runtime-nightly:latest")}
    if wheel:
        cmds["wheel"] = f'uv pip install --pre --extra-index-url {PYPI}/ "ai-dynamo[{extra}]"'
    return cmds


def nightly_pinned_commands(label: str, date: str, sha: str, wheel_version: str | None) -> dict[str, str]:
    img, extra, wheel = META[label]
    cmds = {"container": run(f"{REGISTRY}/{img}-runtime-nightly:{date}-{sha}")}
    if wheel and wheel_version:
        cmds["wheel"] = (
            f'uv pip install --pre --extra-index-url {PYPI}/ "ai-dynamo[{extra}]=={wheel_version}"'
        )
    return cmds


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build() -> dict:
    changes = sync.read_history(REPO_ROOT)

    dated_by_img: dict[str, dict] = {}
    reltags_by_img: dict[str, set] = {}
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
            print(f"warning: NGC release tags failed for {img}: {exc}", file=sys.stderr)
            reltags_by_img[img] = set()

    stable = stable_by_framework(reltags_by_img)
    data: dict[str, dict] = {}

    for fw in sync.FRAMEWORKS:
        label = fw.label
        img, extra, has_wheel = META[label]
        entry: dict = {"label": label, "image": img, "extra": extra, "wheel": has_wheel,
                       "stable": [], "nightly": []}

        for s in stable[label]:
            entry["stable"].append({
                "backend_version": s["backend_version"],
                "dynamo": s["releases"][0],
                "also": s["releases"][1:],
                "commands": stable_commands(label, s["releases"][0]),
            })

        dated = dated_by_img[img]
        cutoff = None
        if dated:
            newest = max(dated)
            cutoff = int(
                (datetime.strptime(str(newest), "%Y%m%d") - timedelta(days=NIGHTLY_DAYS_BACK)).strftime("%Y%m%d")
            )

        wins = windows(label, changes)
        for idx in range(len(wins) - 1, -1, -1):  # newest-first
            version, start, end = wins[idx]
            if end is None:  # current build
                entry["nightly"].append({
                    "backend_version": version, "latest": True,
                    "commands": nightly_latest_commands(label),
                })
                continue
            hit = latest_in_window(dated, start, end)
            if not hit:
                continue  # no published (or GC'd) nightly in range -> omit
            date, sha = hit
            if cutoff is not None and int(date) < cutoff:
                continue  # older than the nightly window we surface
            base = base_version_at(sha)
            wheel_version = f"{base}.dev{date}" if base else None
            entry["nightly"].append({
                "backend_version": version, "window": [start, end], "pin_date": iso(date),
                "note": f"In nightlies {start} → {end}; pinned to {iso(date)} (latest in range).",
                "commands": nightly_pinned_commands(label, date, sha, wheel_version),
            })

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
    return f"{header}export const INSTALL_DATA = {body};\n\nexport default INSTALL_DATA;\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="print JSON instead of writing")
    ap.add_argument("--check", action="store_true", help="exit 1 if the on-disk JSON is stale")
    ap.add_argument("--out", default=OUT, help="JSON data path")
    ap.add_argument("--ts-out", default=TS_OUT, help="TypeScript module path for the component")
    args = ap.parse_args()

    data = build()
    json_text = as_json(data)

    if args.check:
        target = REPO_ROOT / args.out
        current = target.read_text() if target.exists() else ""
        if current != json_text:
            print(f"{args.out} is stale — run build_install_selector.py", file=sys.stderr)
            return 1
        return 0
    if args.stdout:
        sys.stdout.write(json_text)
        return 0

    for rel, text in ((args.out, json_text), (args.ts_out, as_ts_module(data))):
        target = REPO_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
