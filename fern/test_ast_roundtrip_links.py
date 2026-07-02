# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fern.ast_roundtrip_spike import round_trip
from fern.markdown_it_relative_links import LinkRewriteConfig, LinkRewriteResult


class AstRoundTripLinksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo_root = Path(self.temporary_directory.name).resolve()
        self.docs_root = self.repo_root / "docs"
        self.source_path = self.docs_root / "guide" / "page.mdx"
        self.source_path.parent.mkdir(parents=True)
        (self.docs_root / "inside.md").write_text("inside\n", encoding="utf-8")
        (self.repo_root / "target.txt").write_text("target\n", encoding="utf-8")
        (self.repo_root / "image.png").write_bytes(b"image")
        (self.repo_root / "assets").mkdir()

    def render(self, source: str) -> tuple[str, LinkRewriteResult]:
        result = LinkRewriteResult()
        rendered = round_trip(
            source,
            LinkRewriteConfig(
                source_path=self.source_path,
                docs_root=self.docs_root,
                repo_root=self.repo_root,
                repository="ai-dynamo/dynamo",
                ref="v1.2.3",
                result=result,
            ),
        )
        return rendered, result

    def test_rewrites_ast_links_images_directories_and_references(self) -> None:
        source = """<Info>
  [file](../../target.txt?raw=1#L1)
  ![image](../../image.png#preview)
  [directory](../../assets/)
  [inside](../inside.md)
  [reference][target]
  [shared link][shared]
  ![shared image][shared]

  {/* [comment](../../missing.txt) */}

  ```markdown
  [example](../../missing.txt)
  ```
</Info>

[target]: <../../target.txt>
[shared]: ../../image.png
"""

        rendered, result = self.render(source)

        self.assertIn(
            "https://github.com/ai-dynamo/dynamo/blob/v1.2.3/target.txt?raw=1#L1",
            rendered,
        )
        self.assertIn(
            "https://raw.githubusercontent.com/ai-dynamo/dynamo/v1.2.3/image.png#preview",
            rendered,
        )
        self.assertIn(
            "https://github.com/ai-dynamo/dynamo/tree/v1.2.3/assets", rendered
        )
        self.assertIn("[inside](../inside.md)", rendered)
        self.assertIn(
            "[target]: https://github.com/ai-dynamo/dynamo/blob/v1.2.3/target.txt",
            rendered,
        )
        self.assertIn(
            "[shared]: https://raw.githubusercontent.com/ai-dynamo/dynamo/v1.2.3/image.png",
            rendered,
        )
        self.assertIn("[comment](../../missing.txt)", rendered)
        self.assertIn("[example](../../missing.txt)", rendered)
        self.assertEqual(len(result.replacements), 5)
        self.assertEqual(result.skipped_replacements, [])
        self.assertEqual(result.unresolved, [])
        shared = next(
            replacement
            for replacement in result.replacements
            if replacement["destination"] == "../../image.png"
        )
        self.assertEqual(shared["parserKinds"], ["image", "link"])
        self.assertEqual(shared["syntax"], "reference")

    def test_unresolved_target_skips_other_rewrites_in_file(self) -> None:
        source = """[existing](../../target.txt)
[missing](../../missing.txt)
"""

        rendered, result = self.render(source)

        self.assertIn("[existing](../../target.txt)", rendered)
        self.assertIn("[missing](../../missing.txt)", rendered)
        self.assertEqual(result.replacements, [])
        self.assertEqual(len(result.skipped_replacements), 1)
        self.assertEqual(len(result.unresolved), 1)
        self.assertEqual(result.unresolved[0]["reason"], "destination does not exist")


if __name__ == "__main__":
    unittest.main()
