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

"""Tests for fern/scripts/sync_deps.py transforms.

Runs without pytest via `python3 test_sync_deps.py`; exit code 0 on pass.
Each `test_*` function is a red/green anchor for one transform. Fixtures are
inline so this file stays hermetic — no live GitHub calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import sync_deps  # noqa: E402

NOVA_LIKE = """# Nova: Active Messaging as a Foundational Network Primitive

**Status**: Draft

**Authors**: [@ryanolson](https://github.com/ryanolson)

**Category**: Architecture

**Replaces**: N/A

**Replaced By**: N/A

**Sponsor**: [TBD]

**Required Reviewers**: [@grahamking](https://github.com/grahamking), [@biswapanda](https://github.com/biswapanda)

**Review Date**: [TBD]

**Pull Request**: [TBD]

**Implementation PR / Tracking Issue**: [TBD]

# Summary

Nova provides a transport-agnostic active messaging layer.

# Motivation

## Problem Statement

Sub-10us latency means <5us is a design target.

```rust
struct Envelope {
    header: Vec<u8>,
    payload: Vec<u8>,
}
```

### Deep Section

More text with <3ms budget.
"""


class ParseMetadataTests(unittest.TestCase):
    def test_extracts_title_and_bold_key_fields(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        self.assertEqual(
            parsed.title,
            "Nova: Active Messaging as a Foundational Network Primitive",
        )
        self.assertEqual(parsed.fields["Status"], "Draft")
        self.assertEqual(
            parsed.fields["Authors"],
            "[@ryanolson](https://github.com/ryanolson)",
        )
        self.assertEqual(parsed.fields["Category"], "Architecture")
        self.assertEqual(
            parsed.fields["Required Reviewers"],
            "[@grahamking](https://github.com/grahamking), [@biswapanda](https://github.com/biswapanda)",
        )

    def test_body_starts_after_metadata_block(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        # First line of the body must be the first non-metadata heading from
        # the source (H1 "# Summary" — demotion happens later).
        first_line = parsed.body.lstrip().splitlines()[0]
        self.assertEqual(first_line, "# Summary")

    def test_tbd_and_na_values_are_dropped(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        useful = sync_deps.useful_fields(parsed.fields)
        self.assertNotIn("Replaces", useful)  # N/A
        self.assertNotIn("Replaced By", useful)  # N/A
        self.assertNotIn("Sponsor", useful)  # [TBD]
        self.assertNotIn("Review Date", useful)  # [TBD]
        self.assertNotIn("Pull Request", useful)  # [TBD]
        self.assertNotIn("Implementation PR / Tracking Issue", useful)  # [TBD]
        self.assertIn("Status", useful)
        self.assertIn("Authors", useful)


class DemoteHeadingsTests(unittest.TestCase):
    def test_h1_becomes_h2(self):
        out = sync_deps.demote_headings("# Summary\n\ntext")
        self.assertEqual(out, "## Summary\n\ntext")

    def test_h2_becomes_h3(self):
        out = sync_deps.demote_headings("## Motivation\n\ntext")
        self.assertEqual(out, "### Motivation\n\ntext")

    def test_h6_stays_h6(self):
        # Max heading depth is H6; demoting H6 would produce invalid H7.
        out = sync_deps.demote_headings("###### Deep\n\n")
        self.assertEqual(out, "###### Deep\n\n")

    def test_ignores_headings_inside_fenced_code(self):
        src = "```md\n# Not A Heading\n```\n"
        self.assertEqual(sync_deps.demote_headings(src), src)

    def test_ignores_hash_that_is_not_atx_heading(self):
        # A "#" without a following space is a comment / anchor / etc, not
        # an ATX heading — leave it alone.
        src = "#no-space-not-heading\n\ntext"
        self.assertEqual(sync_deps.demote_headings(src), src)


class EscapeStrayLtTests(unittest.TestCase):
    def test_escapes_lt_digit_outside_code(self):
        src = "Latency <5us and <10us matter."
        out = sync_deps.escape_stray_lt(src)
        self.assertEqual(out, "Latency &lt;5us and &lt;10us matter.")

    def test_leaves_lt_digit_inside_fenced_code(self):
        # Inline `<5us` inside a fenced block must stay untouched or the
        # rendered code sample no longer matches the source.
        src = "```\nassert x <5;\n```\n"
        self.assertEqual(sync_deps.escape_stray_lt(src), src)

    def test_leaves_jsx_like_tags(self):
        src = "See <Component name='x' /> and <PrInlineComments />."
        self.assertEqual(sync_deps.escape_stray_lt(src), src)

    def test_leaves_vec_generic_inside_code(self):
        src = "```rust\nlet v: Vec<u8> = Vec::new();\n```\n"
        self.assertEqual(sync_deps.escape_stray_lt(src), src)


class RenderMdxTests(unittest.TestCase):
    def _entry(self, **overrides):
        base = {
            "output": "dep-nova-synced",
            "dep": "Nova",
            "source": {
                "owner": "ai-dynamo",
                "repo": "enhancements",
                "ref": "ryan/nova",
                "path": "deps/0000-nova.md",
            },
            "pr": 61,
            "tracking_issue_url": None,
        }
        base.update(overrides)
        return base

    def test_renders_frontmatter_and_wrappers(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        mdx = sync_deps.render_mdx(entry=self._entry(), parsed=parsed)

        # YAML front-matter block at the top with title and pr.
        self.assertTrue(mdx.startswith("---\n"))
        self.assertIn(
            'title: "Nova: Active Messaging as a Foundational Network Primitive"',
            mdx,
        )
        self.assertIn("pr: 61", mdx)

        # SPDX headers are inside the front-matter so Fern doesn't render
        # them as page content.
        self.assertIn("SPDX-License-Identifier: Apache-2.0", mdx)

        # Component imports are present.
        self.assertIn('import { DepMetadata } from "@/components/DepMetadata";', mdx)
        self.assertIn(
            'import { PrInlineComments } from "@/components/PrInlineComments";',
            mdx,
        )

        # DepMetadata is emitted with the DEP number and status pill.
        self.assertIn('dep="Nova"', mdx)
        self.assertIn('status="Draft"', mdx)
        self.assertIn('category="Architecture"', mdx)
        self.assertIn('authors="[@ryanolson](https://github.com/ryanolson)"', mdx)
        self.assertIn('owner="ai-dynamo"', mdx)
        self.assertIn('repo="enhancements"', mdx)
        self.assertIn("pr={61}", mdx)

        # PrInlineComments mirror mounts against the enhancements repo,
        # with the file path filter derived from source.path.
        self.assertIn("<PrInlineComments", mdx)
        self.assertIn('path="deps/0000-nova.md"', mdx)

    def test_body_headings_are_demoted(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        mdx = sync_deps.render_mdx(entry=self._entry(), parsed=parsed)

        # Source has "# Summary" — after demote this should be "## Summary"
        # and NOT collide with the front-matter title (H1 comes from Fern).
        self.assertIn("\n## Summary\n", mdx)
        self.assertIn("\n## Motivation\n", mdx)
        self.assertIn("\n### Problem Statement\n", mdx)
        self.assertIn("\n#### Deep Section\n", mdx)
        # The literal source H1 line must NOT appear (dropped by parser).
        self.assertNotIn(
            "# Nova: Active Messaging as a Foundational Network Primitive",
            mdx.replace(
                'title: "Nova: Active Messaging as a Foundational Network Primitive"',
                "",
            ),
        )

    def test_stray_lt_is_escaped_in_prose(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        mdx = sync_deps.render_mdx(entry=self._entry(), parsed=parsed)
        # Prose <5us / <3ms must be escaped so MDX doesn't parse them as JSX.
        self.assertIn("&lt;5us", mdx)
        self.assertIn("&lt;3ms", mdx)
        # Inside the rust fence, "Vec<u8>" must survive intact.
        self.assertIn("Vec<u8>", mdx)

    def test_generated_marker_comment_is_present(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        mdx = sync_deps.render_mdx(entry=self._entry(), parsed=parsed)
        # A visible-in-source marker so reviewers understand the file is
        # not hand-edited.
        self.assertIn("GENERATED FILE — DO NOT EDIT", mdx)
        self.assertIn("fern/scripts/sync_deps.py", mdx)
        self.assertIn(
            "https://github.com/ai-dynamo/enhancements/blob/ryan/nova/deps/0000-nova.md",
            mdx,
        )


class SlugValidationTests(unittest.TestCase):
    """Manifest 'output' becomes a filename via out_dir / f"{output}.mdx".
    An unsafe value can escape docs/proposals/_generated/. load_manifest
    must reject anything that isn't a bare slug.
    """

    def _manifest(self, output_value):
        return {
            "deps": [
                {
                    "output": output_value,
                    "source": {
                        "owner": "ai-dynamo",
                        "repo": "enhancements",
                        "ref": "main",
                        "path": "deps/x.md",
                    },
                }
            ]
        }

    def _load(self, output_value, tmp_path: Path) -> None:
        manifest = tmp_path / "fern" / "scripts" / "deps.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            __import__("json").dumps(self._manifest(output_value)),
            encoding="utf-8",
        )
        sync_deps.load_manifest(tmp_path)

    def test_accepts_plain_slug(self):
        with tempfile.TemporaryDirectory() as td:
            self._load("dep-nova-synced", Path(td))  # must not raise

    def test_accepts_alphanumeric_and_underscore(self):
        with tempfile.TemporaryDirectory() as td:
            self._load("dep_v1_2_synced", Path(td))  # must not raise

    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                self._load("../../etc/passwd", Path(td))
            self.assertIn("output", str(ctx.exception).lower())

    def test_rejects_slash_in_slug(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("nova/other", Path(td))

    def test_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("/tmp/pwn", Path(td))

    def test_rejects_empty_string(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("", Path(td))

    def test_rejects_null_byte(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("nova\x00.mdx", Path(td))

    def test_rejects_dot_prefix(self):
        # Prevent .hidden / . / .. slugs

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load(".secret", Path(td))


class YamlStringEscapeTests(unittest.TestCase):
    """_yaml_string must produce a valid YAML double-quoted string. Newlines
    and carriage returns MUST be escaped so a multi-line title/subtitle does
    not break the front-matter block.
    """

    def test_escapes_newline(self):
        s = sync_deps._yaml_string("line1\nline2")
        # No literal newline should remain inside the quoted string
        self.assertNotIn("\n", s.strip('"'))
        # \n escape sequence should be present
        self.assertIn("\\n", s)

    def test_escapes_carriage_return(self):
        s = sync_deps._yaml_string("crlf\r\nline")
        self.assertNotIn("\r", s.strip('"'))
        self.assertIn("\\r", s)

    def test_escapes_backslash_and_quote_still(self):
        s = sync_deps._yaml_string('back\\slash and "quote"')
        # Regressions on existing escape behavior
        self.assertIn("\\\\", s)
        self.assertIn('\\"', s)


class ExtractStatusFromMdxTests(unittest.TestCase):
    """Sidebar status pill comes from the SAME truth as the on-page card.

    For hand-authored DEPs (docs/proposals/*.mdx) that means the
    `status="..."` prop on the `<DepMetadata ... />` component. The
    extractor must survive: multi-line JSX, other props before/after
    `status`, single-quoted attribute values, and DEPs that do not have
    a DepMetadata card (rare — Overview / Template — those aren't DEPs).
    """

    def test_extracts_status_from_multiline_component(self):
        text = '<DepMetadata\n  dep="0001"\n  status="Draft"\n  category="Process"\n/>'
        self.assertEqual(sync_deps._extract_status_from_mdx(text), "Draft")

    def test_extracts_status_from_inline_component(self):
        text = '<DepMetadata dep="0000" status="Under Review" />'
        self.assertEqual(sync_deps._extract_status_from_mdx(text), "Under Review")

    def test_extracts_single_quoted_status(self):
        # Single-quoted JSX attribute values must be extracted too, so the
        # docstring's single-quote claim holds. The opening quote is captured
        # and the same quote closes it.
        text = "<DepMetadata dep='0000' status='Accepted' />"
        self.assertEqual(sync_deps._extract_status_from_mdx(text), "Accepted")

    def test_returns_none_when_no_dep_metadata(self):
        text = "# Just a regular markdown file with no DepMetadata."
        self.assertIsNone(sync_deps._extract_status_from_mdx(text))

    def test_returns_none_when_dep_metadata_lacks_status(self):
        # DepMetadata without a status prop — no meaningful sidebar label.
        text = '<DepMetadata dep="0000" category="Process" />'
        self.assertIsNone(sync_deps._extract_status_from_mdx(text))

    def test_ignores_status_prop_on_other_components(self):
        # A `status=` prop on some other component must not be
        # misattributed to a DEP.
        text = '<SomethingElse status="ShouldNotMatch" />'
        self.assertIsNone(sync_deps._extract_status_from_mdx(text))


class DiscoverHandAuthoredStatusesTests(unittest.TestCase):
    """docs/proposals/*.mdx is scanned for hand-authored DEPs. README + TEMPLATE
    are meta pages (not DEPs) and MUST be excluded. Files without a
    `<DepMetadata status="..." />` prop are skipped. Slug = filename stem
    (matches the slug convention used in docs/index.yml)."""

    def _write(self, root: Path, rel: str, contents: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def test_discovers_hand_authored_deps_excludes_readme_and_template(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "docs/proposals/0000-example-dep.mdx",
                '<DepMetadata dep="0000" status="Draft" />',
            )
            self._write(
                root,
                "docs/proposals/0001-dep-process.mdx",
                '<DepMetadata dep="0001" status="Under Review" />',
            )
            # Meta pages: no DEP status in the sidebar.
            self._write(
                root,
                "docs/proposals/README.mdx",
                "# Overview",
            )
            self._write(
                root,
                "docs/proposals/TEMPLATE.mdx",
                '<DepMetadata dep="TEMPLATE" status="Draft" />',
            )
            got = sync_deps.discover_hand_authored_statuses(root)
        self.assertEqual(
            got,
            {"0000-example-dep": "Draft", "0001-dep-process": "Under Review"},
        )

    def test_skips_generated_dir(self):
        # docs/proposals/_generated/*.mdx are synced files — they are
        # covered by the manifest path, NOT by the hand-authored scan.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "docs/proposals/_generated/dep-nova-synced.mdx",
                '<DepMetadata dep="Nova" status="Draft" />',
            )
            got = sync_deps.discover_hand_authored_statuses(root)
        self.assertEqual(got, {})

    def test_skips_files_without_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "docs/proposals/only-prose.mdx",
                "# Not a DEP\n\nNo DepMetadata here.",
            )
            got = sync_deps.discover_hand_authored_statuses(root)
        self.assertEqual(got, {})


class EntryStatusTests(unittest.TestCase):
    """Synced DEP status comes from the parsed `**Status**:` field of the
    upstream markdown; missing / empty falls to 'Draft'."""

    def test_returns_parsed_status(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        self.assertEqual(sync_deps.entry_status(parsed), "Draft")

    def test_defaults_to_draft_when_missing(self):
        parsed = sync_deps.ParsedDep(title="X", fields={}, body="")
        self.assertEqual(sync_deps.entry_status(parsed), "Draft")

    def test_defaults_to_draft_when_placeholder(self):
        # `[TBD]` / `N/A` are filtered by useful_fields — treat as unset.
        parsed = sync_deps.ParsedDep(title="X", fields={"Status": "[TBD]"}, body="")
        self.assertEqual(sync_deps.entry_status(parsed), "Draft")


class RenderStatusDataJsTests(unittest.TestCase):
    """The status data file is a plain JS assignment to `window.__DEP_STATUS`.
    Keys are DEP slugs (matching docs/index.yml), values are lifecycle labels
    that fern/js/dep-status-pills.js maps to a colour variant via
    `variant()` (verbatim copy of DepMetadata.statusVariant())."""

    def test_emits_valid_js_with_spdx_header(self):
        js = sync_deps.render_status_data_js(
            {"0000-example-dep": "Draft", "dep-nova-synced": "Under Review"}
        )
        self.assertIn("SPDX-License-Identifier: Apache-2.0", js)
        self.assertIn("GENERATED FILE", js)
        self.assertIn("window.__DEP_STATUS", js)
        self.assertIn('"0000-example-dep": "Draft"', js)
        self.assertIn('"dep-nova-synced": "Under Review"', js)

    def test_keys_are_sorted_for_stable_diffs(self):
        # Deterministic output: even if we accept dict insertion order in
        # Python 3.7+, we want git diffs to reflect real changes, not
        # scanner traversal order.
        js = sync_deps.render_status_data_js(
            {"zzz-late": "Draft", "aaa-early": "Draft"}
        )
        pos_early = js.index('"aaa-early"')
        pos_late = js.index('"zzz-late"')
        self.assertLess(pos_early, pos_late)

    def test_empty_map_emits_empty_object_literal(self):
        # If sync fails or no DEPs exist we still emit a valid file so
        # docs.yml's js: reference resolves. The runtime silently no-ops
        # on an empty map.
        js = sync_deps.render_status_data_js({})
        self.assertIn("window.__DEP_STATUS = {};", js)

    def test_escapes_status_double_quotes(self):
        # Defense-in-depth: status labels shouldn't contain a bare " but
        # if a manifest ever carries one, the emitted JS must still parse.
        js = sync_deps.render_status_data_js({"x": 'A"B'})
        # Must NOT leave a bare " in the JS string literal (would break
        # syntax); must have a backslash escape sequence.
        self.assertIn('"A\\"B"', js)


class BuildStatusMapTests(unittest.TestCase):
    """Combine synced-entry statuses with hand-authored statuses. Synced
    entries win when both sources supply the same slug (upstream is the
    authoritative source for synced DEPs)."""

    def test_merges_synced_and_hand_authored(self):
        synced = {"dep-nova-synced": "Draft"}
        hand = {"0000-example-dep": "Draft", "0001-dep-process": "Draft"}
        merged = sync_deps.merge_status_maps(hand, synced)
        self.assertEqual(
            merged,
            {
                "0000-example-dep": "Draft",
                "0001-dep-process": "Draft",
                "dep-nova-synced": "Draft",
            },
        )

    def test_synced_wins_on_conflict(self):
        # Should never happen (slug space is disjoint by convention) but
        # if a manifest slug collides with a hand-authored basename, the
        # synced (upstream) source of truth wins.
        synced = {"nova": "Accepted"}
        hand = {"nova": "Draft"}
        merged = sync_deps.merge_status_maps(hand, synced)
        self.assertEqual(merged, {"nova": "Accepted"})


class WriteStatusDataJsTests(unittest.TestCase):
    """End-to-end: writing the JS data file to fern/js/dep-status-data.js."""

    def test_writes_file_under_fern_js(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "fern" / "js").mkdir(parents=True)
            status_map = {"0000-example-dep": "Draft"}
            out = sync_deps.write_status_data_js(root, status_map)
        # Path check
        self.assertEqual(out.name, "dep-status-data.js")
        self.assertEqual(out.parent.name, "js")
        self.assertEqual(out.parent.parent.name, "fern")

    def test_content_matches_render(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "fern" / "js").mkdir(parents=True)
            status_map = {"a": "Draft", "b": "Under Review"}
            out = sync_deps.write_status_data_js(root, status_map)
            got = out.read_text(encoding="utf-8")
        self.assertEqual(got, sync_deps.render_status_data_js(status_map))


class MainWriteGateTests(unittest.TestCase):
    """main() may write fern/js/dep-status-data.js ONLY on a full run.

    A --dry-run must not touch the tree, and a --only (partial) run must not
    clobber the committed snapshot with a truncated status map. In both cases
    main() prints the map a full run would write and skips the write.
    """

    def _setup_root(self, td: str) -> Path:
        root = Path(td)
        # Minimal repo skeleton main() needs: fern/docs.yml (the root marker),
        # the manifest, and the dirs the writers touch.
        (root / "fern").mkdir()
        (root / "fern" / "docs.yml").write_text("x\n", encoding="utf-8")
        (root / "fern" / "js").mkdir()
        (root / "fern" / "scripts").mkdir()
        (root / "docs" / "proposals").mkdir(parents=True)
        manifest = {
            "deps": [
                {
                    "output": "dep-nova-synced",
                    "dep": "Nova",
                    "source": {
                        "owner": "ai-dynamo",
                        "repo": "enhancements",
                        "ref": "main",
                        "path": "deps/0000-nova.md",
                    },
                },
                {
                    "output": "dep-other-synced",
                    "dep": "Other",
                    "source": {
                        "owner": "ai-dynamo",
                        "repo": "enhancements",
                        "ref": "main",
                        "path": "deps/0001-other.md",
                    },
                },
            ]
        }
        (root / "fern" / "scripts" / "deps.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root

    def test_full_run_writes_status_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            data = root / "fern" / "js" / "dep-status-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root)])
            self.assertEqual(rc, 0)
            self.assertTrue(data.exists())
            self.assertIn("window.__DEP_STATUS", data.read_text(encoding="utf-8"))

    def test_dry_run_does_not_write_status_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            data = root / "fern" / "js" / "dep-status-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root), "--dry-run"])
            # File must not exist — a dry run must not touch the tree.
            self.assertEqual(rc, 0)
            self.assertFalse(data.exists())

    def test_only_does_not_write_status_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            data = root / "fern" / "js" / "dep-status-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root), "--only", "nova"])
            # Partial sync must not clobber the committed snapshot.
            self.assertEqual(rc, 0)
            self.assertFalse(data.exists())


class ParseAuthorsTests(unittest.TestCase):
    """Authors field -> [{label, handle}]. The canonical shape is a
    comma-separated list of `[@handle](https://github.com/handle)` markdown
    links; a bare `@handle` also resolves a login; plain text yields a label
    with handle=None; a non-GitHub href contributes a label but no handle."""

    def test_parses_single_github_link(self):
        got = sync_deps.parse_authors("[@ryanolson](https://github.com/ryanolson)")
        self.assertEqual(got, [{"label": "@ryanolson", "handle": "ryanolson"}])

    def test_parses_multiple_authors(self):
        got = sync_deps.parse_authors(
            "[@grahamking](https://github.com/grahamking), "
            "[@biswapanda](https://github.com/biswapanda)"
        )
        self.assertEqual(
            got,
            [
                {"label": "@grahamking", "handle": "grahamking"},
                {"label": "@biswapanda", "handle": "biswapanda"},
            ],
        )

    def test_bare_handle_resolves_login(self):
        got = sync_deps.parse_authors("@dagil-nvidia")
        self.assertEqual(got, [{"label": "@dagil-nvidia", "handle": "dagil-nvidia"}])

    def test_plain_text_has_no_handle(self):
        got = sync_deps.parse_authors("Dynamo Docs Infrastructure (example)")
        self.assertEqual(
            got,
            [{"label": "Dynamo Docs Infrastructure (example)", "handle": None}],
        )

    def test_non_github_href_keeps_label_drops_handle(self):
        got = sync_deps.parse_authors("[Jane Doe](https://example.com/jane)")
        self.assertEqual(got, [{"label": "Jane Doe", "handle": None}])

    def test_empty_value_is_empty_list(self):
        self.assertEqual(sync_deps.parse_authors(""), [])
        self.assertEqual(sync_deps.parse_authors(None), [])


class ExtractDepMetaPropTests(unittest.TestCase):
    """Generic <DepMetadata> prop extractor: any prop, single/double quotes,
    multi-line tags, and never misattributed to a different component."""

    MULTILINE = (
        '<DepMetadata\n  dep="0003"\n  status="Under Review"\n'
        '  owningSig="SIG-Router"\n  category="Architecture"\n/>'
    )

    def test_extracts_owning_sig(self):
        self.assertEqual(
            sync_deps._extract_dep_meta_prop(self.MULTILINE, "owningSig"),
            "SIG-Router",
        )

    def test_extracts_category(self):
        self.assertEqual(
            sync_deps._extract_dep_meta_prop(self.MULTILINE, "category"),
            "Architecture",
        )

    def test_extracts_single_quoted_prop(self):
        text = "<DepMetadata dep='0000' category='Process' />"
        self.assertEqual(sync_deps._extract_dep_meta_prop(text, "category"), "Process")

    def test_returns_none_when_prop_absent(self):
        self.assertIsNone(sync_deps._extract_dep_meta_prop(self.MULTILINE, "sponsor"))

    def test_not_misattributed_to_other_component(self):
        text = '<SomethingElse owningSig="Nope" />'
        self.assertIsNone(sync_deps._extract_dep_meta_prop(text, "owningSig"))


class ExtractFrontmatterTitleTests(unittest.TestCase):
    def test_extracts_quoted_title(self):
        text = '---\ntitle: "DEP-0001: The DEP Process"\npr: 1\n---\nbody\n'
        self.assertEqual(
            sync_deps._extract_frontmatter_title(text), "DEP-0001: The DEP Process"
        )

    def test_extracts_unquoted_title(self):
        text = "---\ntitle: Plain Title\n---\nbody\n"
        self.assertEqual(sync_deps._extract_frontmatter_title(text), "Plain Title")

    def test_returns_none_without_title(self):
        text = "---\npr: 1\n---\nbody\n"
        self.assertIsNone(sync_deps._extract_frontmatter_title(text))


class DiscoverHandAuthoredDepsTests(unittest.TestCase):
    """docs/proposals/*.mdx -> full index records. Same exclusions as the
    status discovery (README + TEMPLATE dropped, _generated skipped, non-DEP
    files skipped), but each record carries the card fields the grid needs."""

    def _write(self, root: Path, rel: str, contents: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def test_builds_records_with_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "docs/proposals/0003-router.mdx",
                '---\ntitle: "DEP-0003: Router v2"\n---\n'
                '<DepMetadata\n  dep="0003"\n  status="Under Review"\n'
                '  owningSig="SIG-Router"\n  category="Architecture"\n'
                '  authors="[@tanmayv25](https://github.com/tanmayv25)"\n/>',
            )
            self._write(root, "docs/proposals/README.mdx", "# Overview")
            self._write(
                root,
                "docs/proposals/TEMPLATE.mdx",
                '<DepMetadata dep="NNNN" status="Draft" />',
            )
            got = sync_deps.discover_hand_authored_deps(root)
        self.assertEqual(len(got), 1)
        rec = got[0]
        self.assertEqual(rec["slug"], "0003-router")
        self.assertEqual(rec["dep"], "0003")
        self.assertEqual(rec["title"], "DEP-0003: Router v2")
        self.assertEqual(rec["status"], "Under Review")
        self.assertEqual(rec["sig"], "SIG-Router")
        self.assertEqual(rec["category"], "Architecture")
        self.assertEqual(rec["submitter"], "tanmayv25")

    def test_skips_generated_and_non_dep(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "docs/proposals/_generated/dep-nova-synced.mdx",
                '<DepMetadata dep="Nova" status="Draft" />',
            )
            self._write(root, "docs/proposals/only-prose.mdx", "# Not a DEP")
            got = sync_deps.discover_hand_authored_deps(root)
        self.assertEqual(got, [])


class SyncedIndexRecordTests(unittest.TestCase):
    def test_builds_from_entry_and_parsed(self):
        parsed = sync_deps.parse_dep_source(NOVA_LIKE)
        entry = {"output": "dep-nova-synced", "dep": "Nova"}
        rec = sync_deps.synced_index_record(entry, parsed)
        self.assertEqual(rec["slug"], "dep-nova-synced")
        self.assertEqual(rec["dep"], "Nova")
        self.assertEqual(rec["status"], "Draft")
        self.assertEqual(rec["category"], "Architecture")
        self.assertEqual(rec["submitter"], "ryanolson")
        # N/A owning SIG absent -> field omitted.
        self.assertNotIn("sig", rec)


class BuildIndexRecordsTests(unittest.TestCase):
    def test_merges_sorted_by_slug_synced_wins(self):
        hand = [
            {"slug": "zzz", "dep": "9", "title": "Z", "status": "Draft"},
            {"slug": "aaa", "dep": "1", "title": "A", "status": "Draft"},
        ]
        synced = [
            {"slug": "aaa", "dep": "1", "title": "A-upstream", "status": "Accepted"}
        ]
        got = sync_deps.build_index_records(hand, synced)
        self.assertEqual([r["slug"] for r in got], ["aaa", "zzz"])
        # synced wins on the shared slug
        self.assertEqual(got[0]["status"], "Accepted")
        self.assertEqual(got[0]["title"], "A-upstream")


class RenderIndexDataJsTests(unittest.TestCase):
    def test_emits_valid_js_with_header(self):
        js = sync_deps.render_index_data_js(
            [{"slug": "0003-router", "dep": "0003", "title": "R", "status": "Draft"}]
        )
        self.assertIn("SPDX-License-Identifier: Apache-2.0", js)
        self.assertIn("GENERATED FILE", js)
        self.assertIn("window.__DEP_INDEX", js)
        self.assertIn('"slug": "0003-router"', js)

    def test_empty_list_emits_empty_array(self):
        self.assertIn("window.__DEP_INDEX = [];", sync_deps.render_index_data_js([]))


class WriteIndexDataJsTests(unittest.TestCase):
    def test_writes_file_under_fern_js(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "fern" / "js").mkdir(parents=True)
            out = sync_deps.write_index_data_js(
                root, [{"slug": "a", "dep": "1", "title": "A", "status": "Draft"}]
            )
        self.assertEqual(out.name, "dep-index-data.js")
        self.assertEqual(out.parent.name, "js")
        self.assertEqual(out.parent.parent.name, "fern")


class MainIndexWriteGateTests(unittest.TestCase):
    """main() writes fern/js/dep-index-data.js on a full run only; a --dry-run
    or --only run must not touch it (parallels MainWriteGateTests)."""

    def _setup_root(self, td: str) -> Path:
        root = Path(td)
        (root / "fern").mkdir()
        (root / "fern" / "docs.yml").write_text("x\n", encoding="utf-8")
        (root / "fern" / "js").mkdir()
        (root / "fern" / "scripts").mkdir()
        (root / "docs" / "proposals").mkdir(parents=True)
        manifest = {
            "deps": [
                {
                    "output": "dep-nova-synced",
                    "dep": "Nova",
                    "source": {
                        "owner": "ai-dynamo",
                        "repo": "enhancements",
                        "ref": "main",
                        "path": "deps/0000-nova.md",
                    },
                }
            ]
        }
        (root / "fern" / "scripts" / "deps.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root

    def test_full_run_writes_index_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            index = root / "fern" / "js" / "dep-index-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root)])
            self.assertEqual(rc, 0)
            self.assertTrue(index.exists())
            text = index.read_text(encoding="utf-8")
            self.assertIn("window.__DEP_INDEX", text)
            self.assertIn('"dep-nova-synced"', text)

    def test_dry_run_does_not_write_index_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            index = root / "fern" / "js" / "dep-index-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root), "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse(index.exists())

    def test_only_does_not_write_index_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._setup_root(td)
            index = root / "fern" / "js" / "dep-index-data.js"
            with mock.patch.object(sync_deps, "fetch_source", return_value=NOVA_LIKE):
                rc = sync_deps.main(["--root", str(root), "--only", "nova"])
            self.assertEqual(rc, 0)
            self.assertFalse(index.exists())


if __name__ == "__main__":
    unittest.main()
