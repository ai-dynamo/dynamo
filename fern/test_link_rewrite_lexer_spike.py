# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fern.link_rewrite_lexer_spike import lex_markdown, process_file


class LinkRewriteLexerTest(unittest.TestCase):
    def test_lexes_links_and_protects_non_markdown_regions(self) -> None:
        source = """---
title: "[metadata](../../target.txt)"
---

<Info note="[attribute](../../target.txt)">
  [component](../../target.txt "title")
</Info>

{/* [MDX comment](../../target.txt) */}
<!-- [HTML comment](../../target.txt) -->

[angle](<../../target(foo).txt>)
[balanced](../../target(foo).txt)
[![badge](../../pixel.png)](../../target.txt)
![reference image][pixel]
[local README][readme]

[pixel]: ../../pixel.png
[readme]: <README.md>

```bash
cat <<EOF
[fenced](../../target.txt)
EOF
```

`[inline code](../../target.txt)`
"""

        result = lex_markdown(source)

        self.assertEqual(
            [
                (candidate.destination, candidate.syntax, sorted(candidate.kinds))
                for candidate in result.candidates
            ],
            [
                ("../../target.txt", "inline", ["link"]),
                ("../../target(foo).txt", "inline", ["link"]),
                ("../../target(foo).txt", "inline", ["link"]),
                ("../../pixel.png", "inline", ["image"]),
                ("../../target.txt", "inline", ["link"]),
                ("../../pixel.png", "reference", ["definition", "image"]),
                ("README.md", "reference", ["definition", "link"]),
            ],
        )
        self.assertEqual(
            result.protected_counts,
            {
                "fenced-code": 1,
                "frontmatter": 1,
                "html-comment": 1,
                "html-or-jsx-tag": 2,
                "inline-code": 1,
                "mdx-comment": 1,
            },
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
