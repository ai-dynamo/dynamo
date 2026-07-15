#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-time sync for Dynamo Enhancement Proposals (DEPs).

Fetches DEP bodies from `ai-dynamo/enhancements` and emits generated MDX pages
under `docs/proposals/_generated/`. The generated dir is gitignored — the
`ai-dynamo/dynamo` repo NEVER contains a copy of a DEP body. The Fern docs
build (both the local `fern check` step and the `fern generate --docs` step)
runs this script first so the nav entries in `docs/index.yml` that point at
`_generated/*.mdx` resolve.

Manifest: `fern/scripts/deps.json` — list of DEPs to sync. Each entry names
its GitHub source (owner/repo/ref/path), the output slug, and the DEP number
plus PR/tracking-issue overrides that feed the metadata card + comment mirror
on the rendered page.

Transforms applied to each source markdown (see PROPOSAL COPY IN THE BRANCH
`docs/proposals/0000-nova.mdx` for the reference hand-copy the pipeline
reproduces):

  1. Drop the source's H1 title (Fern renders the front-matter `title` as H1).
  2. Parse the bold-key metadata block (`**Key**: Value`) into fields; feed
     them into `<DepMetadata>`. `[TBD]` / `N/A` placeholders are dropped.
  3. Demote every ATX heading one level (H1 -> H2, ..., H6 stays H6). Skips
     hashes inside fenced code blocks so code samples are unaffected.
  4. Escape stray `<digit` occurrences in prose (e.g. `<5us` -> `&lt;5us`)
     that MDX otherwise parses as an unclosed JSX tag. Fenced code blocks
     are left alone.
  5. Wrap the transformed body with imports + `<DepMetadata>` + a
     `<PrInlineComments>` mount whose owner/repo/pr/path point at the
     enhancements repo (the runtime in `fern/js/dep-pr-comments.js` reads
     those props at load time).

Usage:
  python3 fern/scripts/sync_deps.py                 # sync all manifest entries
  python3 fern/scripts/sync_deps.py --root ./       # explicit repo root
  python3 fern/scripts/sync_deps.py --dry-run       # print what would change
  python3 fern/scripts/sync_deps.py --only nova     # sync one entry by slug
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

# Bold-key metadata line: `**Key**: Value`. The enhancements repo template
# (NNNN-complete-template.md) uses exactly this shape for every field.
_META_LINE = re.compile(r"^\*\*(?P<key>[^*]+?)\*\*\s*:\s*(?P<value>.*)$")

# Placeholder values that mean "not yet filled in" — drop the field entirely so
# the DepMetadata card doesn't render "Sponsor: [TBD]" et al.
_PLACEHOLDER_RE = re.compile(r"^\s*(?:\[?tbd\]?|n/?a|none|todo|—|-)\s*$", re.IGNORECASE)


@dataclass
class ParsedDep:
    """Structured view of one DEP markdown source."""

    title: str
    fields: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_dep_source(text: str) -> ParsedDep:
    """Parse a DEP markdown source into title + bold-key fields + body.

    The enhancements repo uses no YAML front-matter. Its convention is:

        # <Title>

        **Status**: Draft

        **Authors**: [...]

        ... more **Key**: Value ...

        # Summary

        ...body...

    We split at the first "content line" after the metadata block: an ATX
    heading (`# `, `## `, ...) or an ordinary paragraph. Blank lines and
    additional `**Key**: Value` lines stay inside the metadata block.
    """
    lines = text.splitlines()

    # 1. Title = first ATX H1. Non-H1 headings before it (unlikely) are body.
    title = ""
    body_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            body_start = idx + 1
            break

    # 2. Walk metadata lines until the first non-metadata content line.
    fields: dict[str, str] = {}
    cursor = body_start
    n = len(lines)
    while cursor < n:
        line = lines[cursor]
        stripped = line.strip()
        if stripped == "":
            cursor += 1
            continue

        match = _META_LINE.match(stripped)
        if match:
            fields[match.group("key").strip()] = match.group("value").strip()
            cursor += 1
            continue

        # First non-blank, non-metadata line — body starts here.
        break

    body = "\n".join(lines[cursor:])
    # Preserve source's trailing newline behavior for stable diffs.
    if text.endswith("\n") and not body.endswith("\n"):
        body += "\n"
    return ParsedDep(title=title, fields=fields, body=body)


def useful_fields(fields: dict[str, str]) -> dict[str, str]:
    """Strip fields whose value is `[TBD]` / `N/A` / empty."""
    return {
        k: v for k, v in fields.items() if v.strip() and not _PLACEHOLDER_RE.match(v)
    }


# --------------------------------------------------------------------------- #
# Transforms                                                                  #
# --------------------------------------------------------------------------- #

# Fenced-code delimiter: three or more backticks (or tildes), optional info.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>```+|~~~+)")

# ATX heading: 1-6 leading '#', at least one required trailing space+text.
# We deliberately require the space to skip `#anchor` / `#not-a-heading` uses.
_ATX_RE = re.compile(r"^(?P<hashes>#{1,6})(?P<rest>\s+.*)$")


def _iter_lines_with_fence(text: str):
    """Yield (line, in_fence) for each line, tracking fenced-code state."""
    in_fence = False
    fence_marker: str | None = None
    for line in text.splitlines(keepends=False):
        m = _FENCE_RE.match(line)
        if m and (not in_fence or line.strip().startswith(fence_marker or "")):
            if not in_fence:
                in_fence = True
                fence_marker = m.group("fence")[:3]  # ``` or ~~~
                yield line, False  # the opening fence itself is NOT "in" a fence
                continue
            # Closing fence of the same kind ends the block.
            if fence_marker and line.strip().startswith(fence_marker):
                yield line, True  # still "in fence" for the closing marker
                in_fence = False
                fence_marker = None
                continue
        yield line, in_fence


def demote_headings(text: str) -> str:
    """Demote every ATX heading one level, skipping fenced-code content.

    H1 -> H2, H2 -> H3, ..., H5 -> H6. H6 stays H6 (can't demote to H7).
    Lines that only *look* like a heading (`#no-space-here`) are ignored so
    Markdown/HTML anchors survive untouched.
    """
    out_lines: list[str] = []
    for line, in_fence in _iter_lines_with_fence(text):
        if in_fence:
            out_lines.append(line)
            continue
        m = _ATX_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        hashes = m.group("hashes")
        rest = m.group("rest")
        if len(hashes) < 6:
            hashes = hashes + "#"
        out_lines.append(hashes + rest)
    result = "\n".join(out_lines)
    if text.endswith("\n"):
        result += "\n"
    return result


# `<` immediately followed by a digit — matches `<5us`, `<10ms`, ...
# NOT matches `<Component ...>` (letter after `<`) or `< 5us` (space) which
# MDX parses fine.
_STRAY_LT_RE = re.compile(r"<(?=\d)")


def escape_stray_lt(text: str) -> str:
    """Escape `<digit` occurrences outside fenced code blocks."""
    out_lines: list[str] = []
    for line, in_fence in _iter_lines_with_fence(text):
        if in_fence:
            out_lines.append(line)
        else:
            out_lines.append(_STRAY_LT_RE.sub("&lt;", line))
    result = "\n".join(out_lines)
    if text.endswith("\n"):
        result += "\n"
    return result


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

# Ordered mapping from source bold-keys to DepMetadata component props. Order
# controls emission order on the rendered card.
_FIELD_MAP: list[tuple[str, str]] = [
    ("Category", "category"),
    ("Owning SIG", "owningSig"),
    ("Participating SIGs", "participatingSigs"),
    ("Authors", "authors"),
    ("Sponsor", "sponsor"),
    ("Required Reviewers", "requiredReviewers"),
    ("Review Date", "reviewDate"),
    ("Replaces", "replaces"),
    ("Replaced By", "replacedBy"),
]


def _yaml_string(value: str) -> str:
    """Double-quote-escape a string for YAML front-matter."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _jsx_string(value: str) -> str:
    """Quote a string for a JSX attribute (double-quoted attribute)."""
    return '"' + value.replace("\\", "\\\\").replace('"', "&quot;") + '"'


def render_mdx(*, entry: dict[str, Any], parsed: ParsedDep) -> str:
    """Assemble the full generated MDX for one DEP entry.

    Layout:
      1. YAML front-matter (SPDX + title + subtitle + pr + tracking-issue)
      2. "GENERATED FILE" HTML comment linking back to source
      3. `import { DepMetadata } ...` + `import { PrInlineComments } ...`
      4. `<DepMetadata ... />` populated from source metadata + manifest
      5. Transformed body (headings demoted, stray `<` escaped, source
         metadata block already stripped)
      6. `<PrInlineComments ... />` mount pointing at the source repo/PR
    """
    fields = useful_fields(parsed.fields)
    source = entry.get("source", {}) or {}
    owner = source.get("owner", "ai-dynamo")
    repo = source.get("repo", "enhancements")
    ref = source.get("ref", "main")
    src_path = source.get("path", "")
    pr = entry.get("pr")
    tracking_issue_url = entry.get("tracking_issue_url") or ""
    tracking_issue_number = entry.get("tracking_issue")
    subtitle = entry.get("subtitle") or ""
    slug = entry["output"]
    dep = entry.get("dep") or ""
    source_url = (
        f"https://github.com/{owner}/{repo}/blob/{ref}/{src_path}"
        if src_path
        else f"https://github.com/{owner}/{repo}"
    )
    # Body preparation: demote headings, escape stray <digit.
    body = demote_headings(parsed.body)
    body = escape_stray_lt(body)
    body = body.rstrip() + "\n"

    # ---------------- front-matter ---------------- #
    fm_lines: list[str] = [
        "---",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# GENERATED FILE — DO NOT EDIT.",
        f"# Synced by fern/scripts/sync_deps.py from {source_url}",
        f"title: {_yaml_string(parsed.title or dep or slug)}",
    ]
    if subtitle:
        fm_lines.append(f"subtitle: {_yaml_string(subtitle)}")
    if pr is not None:
        fm_lines.append(f"pr: {int(pr)}")
    if tracking_issue_number is not None:
        fm_lines.append(f"tracking-issue: {int(tracking_issue_number)}")
    else:
        fm_lines.append("tracking-issue:")
    fm_lines.append("---")

    # ---------------- generated-marker comment ---------------- #
    marker = textwrap.dedent(
        f"""
        {{/*
          GENERATED FILE — DO NOT EDIT.

          This page is regenerated on every Fern docs build by
          `fern/scripts/sync_deps.py`, which pulls the DEP body from the
          public `ai-dynamo/enhancements` repository. The `ai-dynamo/dynamo`
          repo intentionally does NOT commit the DEP body; edit the source
          instead:

              {source_url}

          Manifest entry: fern/scripts/deps.json
          Slug:           {slug}
        */}}
        """
    ).strip()

    # ---------------- imports ---------------- #
    imports = (
        'import { DepMetadata } from "@/components/DepMetadata";\n'
        'import { PrInlineComments } from "@/components/PrInlineComments";'
    )

    # ---------------- DepMetadata ---------------- #
    dep_meta_attrs: list[str] = []
    if dep:
        dep_meta_attrs.append(f"  dep={_jsx_string(str(dep))}")
    status = fields.get("Status", "Draft")
    dep_meta_attrs.append(f"  status={_jsx_string(status)}")
    for src_key, prop in _FIELD_MAP:
        if src_key in fields:
            dep_meta_attrs.append(f"  {prop}={_jsx_string(fields[src_key])}")
    dep_meta_attrs.append(f"  owner={_jsx_string(owner)}")
    dep_meta_attrs.append(f"  repo={_jsx_string(repo)}")
    if pr is not None:
        dep_meta_attrs.append(f"  pr={{{int(pr)}}}")
    if tracking_issue_url:
        dep_meta_attrs.append(f"  trackingIssue={_jsx_string(tracking_issue_url)}")
    dep_meta = "<DepMetadata\n" + "\n".join(dep_meta_attrs) + "\n/>"

    # ---------------- PrInlineComments ---------------- #
    pr_attrs: list[str] = []
    if pr is not None:
        pr_attrs.append(f"  pr={{{int(pr)}}}")
    pr_attrs.append(f"  owner={_jsx_string(owner)}")
    pr_attrs.append(f"  repo={_jsx_string(repo)}")
    if src_path:
        pr_attrs.append(f"  path={_jsx_string(src_path)}")
    if tracking_issue_number is not None:
        pr_attrs.append(f"  issue={{{int(tracking_issue_number)}}}")
    pr_mount = "<PrInlineComments\n" + "\n".join(pr_attrs) + "\n/>"

    parts = [
        "\n".join(fm_lines),
        "",
        marker,
        "",
        imports,
        "",
        dep_meta,
        "",
        body.rstrip(),
        "",
        "{/* Read-only mirror of the source PR's comments (fetched from",
        f"    {owner}/{repo}#{pr if pr is not None else 'TBD'} at page load).",
        "    Runtime: fern/js/dep-pr-comments.js reads the data-* attributes.",
        "*/}",
        "",
        pr_mount,
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Fetch + manifest driver                                                     #
# --------------------------------------------------------------------------- #


def _raw_url(owner: str, repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def fetch_source(entry: dict[str, Any], *, timeout: int = 30) -> str:
    """Fetch the raw markdown for one DEP entry from raw.githubusercontent.com.

    Uses `urllib` (stdlib) so this script has no third-party deps. If
    `GITHUB_TOKEN` is set in the environment, we still don't send it — the
    raw endpoint is public and rate-limited generously per-IP.
    """
    source = entry["source"]
    url = _raw_url(source["owner"], source["repo"], source["ref"], source["path"])
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-dynamo-fern-sync-deps/1.0",
            "Accept": "text/plain, */*;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"[sync-deps] HTTP {exc.code} fetching {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"[sync-deps] URL error fetching {url}: {exc.reason}") from exc
    return data


def load_manifest(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "fern" / "scripts" / "deps.json"
    if not manifest_path.exists():
        raise SystemExit(f"[sync-deps] manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("deps", [])
    if not isinstance(entries, list):
        raise SystemExit(f"[sync-deps] manifest {manifest_path}: 'deps' must be a list")
    for entry in entries:
        for required in ("output", "source"):
            if required not in entry:
                raise SystemExit(
                    f"[sync-deps] manifest entry missing '{required}': {entry}"
                )
        source = entry["source"]
        for required in ("owner", "repo", "ref", "path"):
            if required not in source:
                raise SystemExit(
                    f"[sync-deps] manifest entry.source missing '{required}': {entry}"
                )
    return entries


def sync_entry(
    entry: dict[str, Any],
    *,
    root: Path,
    dry_run: bool = False,
) -> Path:
    text = fetch_source(entry)
    parsed = parse_dep_source(text)
    mdx = render_mdx(entry=entry, parsed=parsed)
    out_dir = root / "docs" / "proposals" / "_generated"
    out_path = out_dir / f"{entry['output']}.mdx"
    if dry_run:
        print(f"[sync-deps] would write {out_path} ({len(mdx)} bytes)")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mdx, encoding="utf-8")
    print(f"[sync-deps] wrote {out_path} ({len(mdx)} bytes)")
    return out_path


def _resolve_root(cli_root: str | None) -> Path:
    """Locate the repo root by walking up from the script until we find fern/."""
    if cli_root:
        return Path(cli_root).resolve()
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "fern" / "docs.yml").exists():
            return parent
    raise SystemExit("[sync-deps] could not locate repo root (no fern/docs.yml)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync DEP bodies from ai-dynamo/enhancements into "
        "docs/proposals/_generated/ (see fern/scripts/deps.json).",
    )
    parser.add_argument(
        "--root",
        help="Repo root (defaults to auto-detect from this script's location).",
    )
    parser.add_argument(
        "--only",
        help="Sync only manifest entries whose 'output' contains this substring.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written but do not write files.",
    )
    args = parser.parse_args(argv)

    root = _resolve_root(args.root)
    entries = load_manifest(root)
    if args.only:
        entries = [e for e in entries if args.only in e["output"]]
        if not entries:
            print(f"[sync-deps] --only={args.only!r} matched no manifest entries")
            return 1

    for entry in entries:
        sync_entry(entry, root=root, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
