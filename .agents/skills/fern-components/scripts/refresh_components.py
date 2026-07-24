#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh helper for the fern-components skill.

Tracks the Fern component-library source pages in `fern-api/docs` and detects
when they drift from what this skill documents. Internet is required only when
running this script -- normal use of the skill is fully offline.

Source of truth: https://github.com/fern-api/docs (branch `main`)
Component pages: fern/products/docs/pages/component-library/**/*.mdx (+ a few related pages)

The manifest (`manifest.json`, next to this script's parent dir) records the git
blob SHA of every tracked page as of the last sync. `--check` compares the live
repo tree against that manifest and reports added / removed / changed pages, so a
refresh only has to re-read what actually moved.

Typical workflow:
    python3 refresh_components.py --check                 # what changed upstream?
    python3 refresh_components.py --fetch --out /tmp/fern # download changed pages
    # ...Claude updates references/components-reference.md + SKILL.md from those...
    python3 refresh_components.py --sync                  # record new SHAs as current

Fetching prefers the `gh` CLI (authenticated, higher rate limits); it falls back
to the public GitHub API + raw.githubusercontent.com over urllib when `gh` is
absent. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import date, timezone, datetime
from pathlib import Path

REPO = "fern-api/docs"
BRANCH = "main"
COMPONENT_DIR = "fern/products/docs/pages/component-library"
# Pages the reference is built from that live OUTSIDE the component-library dir.
EXTRA_PAGES = [
    "fern/products/docs/pages/customization/custom-react-components.mdx",
    "fern/products/docs/pages/navigation/tabs.mdx",
]

SKILL_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = SKILL_ROOT / "manifest.json"


# --------------------------------------------------------------------------- #
# Fetching (gh preferred, urllib fallback)
# --------------------------------------------------------------------------- #
def _have_gh() -> bool:
    return shutil.which("gh") is not None


def _gh_json(endpoint: str):
    out = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "fern-components-skill"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_tree() -> dict[str, str]:
    """Return {path: blob_sha} for every tracked page in the live repo."""
    endpoint = f"repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    try:
        data = _gh_json(endpoint) if _have_gh() else _http_json(f"https://api.github.com/{endpoint}")
    except Exception as exc:  # noqa: BLE001 -- surface a clean message, any failure mode
        raise SystemExit(f"error: could not fetch repo tree ({exc}).\n"
                         f"Check your internet connection, or authenticate `gh auth login`.")
    if data.get("truncated"):
        raise SystemExit("error: GitHub returned a truncated tree; cannot reliably diff.")

    tracked: dict[str, str] = {}
    for entry in data.get("tree", []):
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        in_component_dir = (
            path.startswith(COMPONENT_DIR)
            and path.endswith(".mdx")
            and not path.startswith("fern/translations")
        )
        if in_component_dir or path in EXTRA_PAGES:
            tracked[path] = entry["sha"]
    return dict(sorted(tracked.items()))


def fetch_content(path: str) -> str:
    """Return the decoded text of a single page from the live repo."""
    if _have_gh():
        b64 = subprocess.run(
            ["gh", "api", f"repos/{REPO}/contents/{path}?ref={BRANCH}", "--jq", ".content"],
            capture_output=True, text=True, check=True,
        ).stdout
        return base64.b64decode(b64).decode("utf-8", "replace")
    raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    req = urllib.request.Request(raw, headers={"User-Agent": "fern-components-skill"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def write_manifest(pages: dict[str, str]) -> None:
    manifest = {
        "source": {"repo": REPO, "branch": BRANCH, "component_dir": COMPONENT_DIR,
                   "extra_pages": EXTRA_PAGES},
        "fetched_at": date.today().isoformat(),
        "page_count": len(pages),
        "pages": pages,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def diff(old: dict[str, str], new: dict[str, str]):
    old_p, new_p = set(old), set(new)
    added = sorted(new_p - old_p)
    removed = sorted(old_p - new_p)
    changed = sorted(p for p in (old_p & new_p) if old[p] != new[p])
    return added, removed, changed


def short(path: str) -> str:
    """Trim the long repo prefix for readable output."""
    return path.replace(COMPONENT_DIR + "/", "").replace("fern/products/docs/pages/", "")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check() -> int:
    manifest = load_manifest()
    if not manifest:
        print("No manifest yet. Run `--sync` to record the current upstream state.")
        return 2
    old = manifest.get("pages", {})
    new = fetch_tree()
    added, removed, changed = diff(old, new)
    if not (added or removed or changed):
        print(f"In sync with {REPO}@{BRANCH} "
              f"(last synced {manifest.get('fetched_at', '?')}, {len(old)} pages).")
        return 0
    print(f"DRIFT vs {REPO}@{BRANCH} (last synced {manifest.get('fetched_at', '?')}):\n")
    for label, items in (("new", added), ("removed", removed), ("changed", changed)):
        for p in items:
            print(f"  {label:8} {short(p)}")
    print("\nRun `--fetch` to download the changed pages, update the reference, then `--sync`.")
    return 2


def cmd_fetch(out_dir: Path, fetch_all: bool) -> int:
    manifest = load_manifest()
    old = manifest.get("pages", {})
    new = fetch_tree()
    added, _removed, changed = diff(old, new)
    targets = sorted(new) if (fetch_all or not manifest) else sorted(set(added) | set(changed))
    if not targets:
        print("Nothing to fetch: already in sync. (Use `--all` to force a full download.)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        dest = out_dir / short(path).replace("/", "__")
        dest.write_text(fetch_content(path))
        print(f"  wrote {dest}")
    print(f"\n{len(targets)} page(s) written to {out_dir}.")
    print("Update references/components-reference.md (and the SKILL.md table) from these, "
          "then run `--sync` to record the new SHAs.")
    return 0


def cmd_sync() -> int:
    new = fetch_tree()
    prior = load_manifest().get("pages", {})
    added, removed, changed = diff(prior, new)
    write_manifest(new)
    if prior:
        print(f"Manifest synced: {len(new)} pages "
              f"(+{len(added)} / -{len(removed)} / ~{len(changed)} since last sync).")
    else:
        print(f"Manifest initialized: {len(new)} pages recorded from {REPO}@{BRANCH}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh/track the Fern component-library source pages.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="Report drift vs the manifest (default). Exit 2 if drift, 0 if in sync.")
    g.add_argument("--fetch", action="store_true",
                   help="Download changed pages (or --all) to --out for re-summarizing.")
    g.add_argument("--sync", action="store_true",
                   help="Record the current upstream SHAs as the manifest (init, or after updating).")
    ap.add_argument("--all", action="store_true", help="With --fetch, download every tracked page.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/fern-components-refresh"),
                    help="Output dir for --fetch (default: /tmp/fern-components-refresh).")
    args = ap.parse_args()

    if args.fetch:
        return cmd_fetch(args.out, args.all)
    if args.sync:
        return cmd_sync()
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
