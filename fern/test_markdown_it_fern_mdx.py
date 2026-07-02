# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from markdown_it import MarkdownIt

from fern.ast_roundtrip_spike import round_trip
from fern.markdown_it_fern_mdx import fern_mdx_plugin


class FernMdxPluginTest(unittest.TestCase):
    def test_component_children_are_parsed_as_markdown(self) -> None:
        source = """<Tabs>
  <Tab
    title="A > B"
    predicate={value => value}
  >
    [link](../../target.txt)

    ```markdown
    [example](../../target.txt)
    ```
  </Tab>
</Tabs>
"""
        tokens = MarkdownIt("commonmark").use(fern_mdx_plugin).parse(source)

        self.assertEqual(
            [token.type for token in tokens],
            [
                "fern_mdx_container_open",
                "fern_mdx_container_open",
                "paragraph_open",
                "inline",
                "paragraph_close",
                "fence",
                "fern_mdx_container_close",
                "fern_mdx_container_close",
            ],
        )
        self.assertIn("link_open", [token.type for token in tokens[3].children or []])
        self.assertNotIn(
            "link_open", [token.type for token in tokens[5].children or []]
        )

    def test_mdformat_round_trip_preserves_mdx_structure(self) -> None:
        source = """<Tabs>
  <Tab title="A">
    [link](../../target.txt)

    ```markdown
    [example](../../target.txt)
    ```
  </Tab>
</Tabs>

- {/* an inline list comment */}
"""

        self.assertEqual(round_trip(source), source)

    def test_inline_mdx_comment_is_not_parsed_as_markdown(self) -> None:
        source = "before {/* [example](../../target.txt) */} after"
        tokens = MarkdownIt("commonmark").use(fern_mdx_plugin).parse(source)

        inline_types = [token.type for token in tokens[1].children or []]
        self.assertIn("fern_mdx_raw_inline", inline_types)
        self.assertNotIn("link_open", inline_types)


if __name__ == "__main__":
    unittest.main()
