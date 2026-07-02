# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fern.link_rewrite_spike import (
    build_parser,
    confirm_candidates,
    process_file,
    scan_candidates,
)


class LinkRewriteParserOracleTest(unittest.TestCase):
    def test_parser_confirms_mdx_links_and_rejects_examples(self) -> None:
        source = """<Info>
  [component](../../target.txt)
</Info>

<Tabs>
  <Tab
    title="A > B"
    predicate={value => value}
  >
    [tab content](../../target.txt)

    ```markdown
    [fenced example](../../target.txt)
    ```
  </Tab>
</Tabs>

[reference link][target]

{/* [comment example](../../target.txt) */}
[target]: <../../target.txt>

`[inline example](../../target.txt)`
"""
        candidates = scan_candidates(source)

        confirm_candidates(build_parser(), source, candidates)

        confirmed = [candidate for candidate in candidates if candidate.parser_kinds]
        rejected = [candidate for candidate in candidates if not candidate.parser_kinds]
        self.assertEqual(len(confirmed), 3)
        self.assertEqual(len(rejected), 3)
        self.assertEqual(
            [sorted(candidate.parser_kinds) for candidate in confirmed],
            [["link"], ["link"], ["definition", "link"]],
        )

    def test_unresolved_target_skips_other_replacements_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory).resolve()
            docs_root = repo_root / "docs"
            source_path = docs_root / "guide" / "page.md"
            output_path = repo_root / "output" / "guide" / "page.md"
            source_path.parent.mkdir(parents=True)
            output_path.parent.mkdir(parents=True)
            (repo_root / "target.txt").write_text("fixture\n", encoding="utf-8")
            source = "[existing](../../target.txt)\n[missing](../../missing.txt)\n"
            source_path.write_text(source, encoding="utf-8")
            output_path.write_text(source, encoding="utf-8")

            result = process_file(
                build_parser(),
                source_path,
                output_path,
                docs_root,
                repo_root,
                "ai-dynamo/dynamo",
                "main",
            )

            self.assertEqual(result["replacements"], [])
            self.assertEqual(len(result["skippedReplacements"]), 1)
            self.assertEqual(len(result["unresolved"]), 1)
            self.assertEqual(output_path.read_text(encoding="utf-8"), source)


if __name__ == "__main__":
    unittest.main()
