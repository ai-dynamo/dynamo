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

import sys
import unittest
from pathlib import Path

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
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            self._load("dep-nova-synced", Path(td))  # must not raise

    def test_accepts_alphanumeric_and_underscore(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            self._load("dep_v1_2_synced", Path(td))  # must not raise

    def test_rejects_parent_traversal(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                self._load("../../etc/passwd", Path(td))
            self.assertIn("output", str(ctx.exception).lower())

    def test_rejects_slash_in_slug(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("nova/other", Path(td))

    def test_rejects_absolute_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("/tmp/pwn", Path(td))

    def test_rejects_empty_string(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("", Path(td))

    def test_rejects_null_byte(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                self._load("nova\x00.mdx", Path(td))

    def test_rejects_dot_prefix(self):
        # Prevent .hidden / . / .. slugs
        import tempfile

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


if __name__ == "__main__":
    unittest.main()
