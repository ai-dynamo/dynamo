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

# Manifest `output` slug: becomes a filename via out_dir / f"{output}.mdx".
# Restrict to a strict subset so a malicious/malformed entry cannot escape
# docs/proposals/_generated/ via `..`, `/`, or leading `.` (dotfiles).
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

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
    """Double-quote-escape a string for YAML front-matter.

    YAML double-quoted strings accept C-style escapes; embedded literal
    control characters break the front-matter block. We escape the ones
    that actually appear in DEP metadata: backslash, double-quote, CR,
    LF, and tab.
    """
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        + '"'
    )


def _jsx_string(value: str) -> str:
    """Quote a string for a JSX attribute (double-quoted attribute).

    JSX attributes are HTML-attribute-shaped. Replace control chars with
    a single space so a stray newline in a manifest field doesn't split
    the tag; escape backslash and double-quote conservatively.
    """
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', "&quot;")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        + '"'
    )


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
    marker_body = f"""
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
    marker = textwrap.dedent(marker_body).strip()

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
# Sidebar status map                                                          #
# --------------------------------------------------------------------------- #
#
# The Proposals sidebar renders a right-aligned lifecycle status pill on each
# DEP link. The pill is populated at runtime by fern/js/dep-status-pills.js
# from a `slug → status` map on window.__DEP_STATUS. That map is materialised
# at Fern-build time by the helpers below and written to
# fern/js/dep-status-data.js. Two sources feed it:
#
#   1. Synced DEPs      — the parsed `**Status**:` field of the source
#                         markdown, keyed by manifest `output` slug (same
#                         slug used in docs/index.yml).
#   2. Hand-authored    — the `status="..."` prop on <DepMetadata /> in
#      DEPs (proposals/*.mdx) hand-authored proposals under docs/proposals/,
#                         keyed by filename stem (matches docs/index.yml).
#
# When both sources supply the same slug the synced entry wins (upstream is
# the authoritative source of truth for synced DEPs).

# Regex targeting a `status="..."` (or `status='...'`) attribute on the
# <DepMetadata /> component. The component is often written across multiple
# lines, so we allow arbitrary whitespace + other props between `<DepMetadata`
# and the `status` attribute. The opening quote is captured and the same quote
# must close it, so both single- and double-quoted attribute values are
# extracted. Anchored to the `<DepMetadata` opening tag so a `status=` on a
# different component cannot be misattributed to the DEP.
_DEP_META_STATUS_RE = re.compile(
    r"<DepMetadata\b[^>]*?\bstatus\s*=\s*(?P<quote>[\"'])(?P<status>.*?)(?P=quote)",
    re.DOTALL,
)


def _extract_status_from_mdx(text: str) -> str | None:
    """Extract the `status` prop from a `<DepMetadata />` component.

    Returns the string value of the first matching `status=".."` on a
    `<DepMetadata` tag, or None when no such prop is present (e.g. README /
    non-DEP pages, or a DEP-shaped page that predates the metadata card).
    """
    match = _DEP_META_STATUS_RE.search(text)
    if not match:
        return None
    return match.group("status")


# Hand-authored DEPs live directly under docs/proposals/ as *.mdx files with
# a `<DepMetadata />` card. Meta pages (Overview + Template) are hard-excluded
# by filename because they either lack a card (README.mdx) or are a template
# whose status ("Draft") is not a real DEP state.
_META_PAGE_STEMS = {"README", "TEMPLATE"}


def discover_hand_authored_statuses(root: Path) -> dict[str, str]:
    """Walk docs/proposals/ for hand-authored DEPs and return {slug: status}.

    Slug = filename stem (the same convention docs/index.yml uses for
    `slug:` entries). Files under docs/proposals/_generated/ are the
    synced DEPs — they are handled by the manifest path and MUST NOT
    appear here (or a stale committed generated file would override the
    fresh sync).
    """
    proposals = root / "docs" / "proposals"
    if not proposals.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(proposals.glob("*.mdx")):
        if path.stem in _META_PAGE_STEMS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        status = _extract_status_from_mdx(text)
        if status:
            out[path.stem] = status
    return out


def entry_status(parsed: ParsedDep) -> str:
    """Return the sidebar status for a synced DEP.

    Uses the parsed `**Status**:` field of the upstream markdown when
    present (filtered through useful_fields so a `[TBD]` / `N/A`
    placeholder is treated as missing). Defaults to "Draft" — the DEP-0001
    default lifecycle state.
    """
    fields = useful_fields(parsed.fields)
    return fields.get("Status", "Draft")


def merge_status_maps(
    hand_authored: dict[str, str], synced: dict[str, str]
) -> dict[str, str]:
    """Combine hand-authored + synced status maps; synced entries win on
    conflict (upstream is authoritative for synced DEPs)."""
    merged: dict[str, str] = {}
    merged.update(hand_authored)
    merged.update(synced)
    return merged


def render_status_data_js(status_map: dict[str, str]) -> str:
    """Render the JS data file that fern/js/dep-status-pills.js consumes.

    Emits `window.__DEP_STATUS = { "<slug>": "<Status>", ... };` with keys
    sorted alphabetically for stable git diffs. Runs no template engine —
    plain JSON dump wrapped in a single `window.__DEP_STATUS = <obj>;`
    assignment. An empty map is rendered as `window.__DEP_STATUS = {};`
    (still valid — the runtime silently no-ops when the map is empty).
    """
    header = (
        "/*\n"
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION "
        "& AFFILIATES. All rights reserved.\n"
        " * SPDX-License-Identifier: Apache-2.0\n"
        " *\n"
        " * GENERATED FILE — DO NOT EDIT.\n"
        " *\n"
        " * Regenerated on every Fern docs build by "
        "fern/scripts/sync_deps.py.\n"
        " * Consumed by fern/js/dep-status-pills.js to render the "
        "right-aligned\n"
        " * lifecycle status pill on each DEP link in the Proposals "
        "sidebar.\n"
        " *\n"
        " * Sources:\n"
        " *   - synced DEPs        parsed `**Status**:` from the source "
        "markdown\n"
        " *                        listed in fern/scripts/deps.json.\n"
        ' *   - hand-authored DEPs `status="..."` on <DepMetadata /> in\n'
        " *                        docs/proposals/*.mdx (excluding "
        "README + TEMPLATE).\n"
        " */\n"
    )
    if not status_map:
        return header + "window.__DEP_STATUS = {};\n"
    # `json.dumps` gives us the correct escapes for embedded quotes /
    # backslashes / control chars. `sort_keys=True` guarantees stable
    # diffs; `indent=2` keeps the file readable.
    body = json.dumps(status_map, sort_keys=True, indent=2, ensure_ascii=False)
    return header + f"window.__DEP_STATUS = {body};\n"


def write_status_data_js(root: Path, status_map: dict[str, str]) -> Path:
    """Write the rendered status data JS to fern/js/dep-status-data.js.

    Creates parent directories as needed. Returns the output path.
    """
    out_path = root / "fern" / "js" / "dep-status-data.js"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_status_data_js(status_map), encoding="utf-8")
    return out_path


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
        output = entry["output"]
        if not isinstance(output, str) or not _SLUG_RE.match(output):
            raise SystemExit(
                f"[sync-deps] manifest entry 'output' must match "
                f"[A-Za-z0-9][A-Za-z0-9_-]* (got {output!r}); output "
                f"becomes docs/proposals/_generated/<output>.mdx and must "
                f"not escape that dir."
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
) -> tuple[Path, str]:
    """Sync one DEP entry, returning (output path, resolved status).

    The status is fed back to `main()` so it can build the sidebar status
    map (fern/js/dep-status-data.js) without re-parsing generated MDX.
    """
    text = fetch_source(entry)
    parsed = parse_dep_source(text)
    status = entry_status(parsed)
    mdx = render_mdx(entry=entry, parsed=parsed)
    out_dir = root / "docs" / "proposals" / "_generated"
    out_path = out_dir / f"{entry['output']}.mdx"
    if dry_run:
        print(f"[sync-deps] would write {out_path} ({len(mdx)} bytes)")
        return out_path, status
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mdx, encoding="utf-8")
    print(f"[sync-deps] wrote {out_path} ({len(mdx)} bytes)")
    return out_path, status


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

    synced_statuses: dict[str, str] = {}
    for entry in entries:
        _, status = sync_entry(entry, root=root, dry_run=args.dry_run)
        synced_statuses[entry["output"]] = status

    # Build the sidebar status map. Hand-authored statuses are picked up from
    # the tree regardless of whether --only filtered the manifest.
    hand_authored = discover_hand_authored_statuses(root)
    status_map = merge_status_maps(hand_authored, synced_statuses)

    # Only a FULL run (no --only, no --dry-run) may write dep-status-data.js.
    # A --only run synced just a subset, so its status_map omits the unsynced
    # DEPs and would TRUNCATE the committed snapshot. A --dry-run must not touch
    # the tree at all. In both cases, print the map a full run WOULD write and
    # skip the write, so a partial or dry run never clobbers the committed
    # fern/js/dep-status-data.js. (That file is committed, so `fern check` after
    # a partial/dry run still resolves the docs.yml `js:` reference.)
    if args.dry_run or args.only:
        reason = "--dry-run" if args.dry_run else "--only (partial sync)"
        print(
            f"[sync-deps] {reason}: NOT writing fern/js/dep-status-data.js — a "
            f"partial or dry run must not clobber the committed snapshot. The "
            f"status map a full run would write "
            f"({len(status_map)} DEPs: {len(hand_authored)} hand-authored + "
            f"{len(synced_statuses)} synced):"
        )
        print(render_status_data_js(status_map))
        return 0

    data_path = write_status_data_js(root, status_map)
    print(
        f"[sync-deps] wrote {data_path} "
        f"({len(status_map)} DEPs: "
        f"{len(hand_authored)} hand-authored + "
        f"{len(synced_statuses)} synced)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
