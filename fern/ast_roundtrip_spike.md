<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Markdown AST round-trip spike

This spike evaluates whether a Python Markdown parser and renderer can safely
round-trip the Dynamo documentation before preprocessing links, comments, and
GitHub callouts.

Run it without modifying the source tree:

```bash
uv run fern/ast_roundtrip_spike.py \
  --input docs \
  --output /tmp/dynamo-docs-python-roundtrip \
  --report /tmp/dynamo-docs-python-roundtrip.json \
  --force
```

The prototype uses `mdformat` with its GFM, GFM-alert, and YAML-frontmatter
extensions. `mdformat` is built on `markdown-it-py`, renders Markdown rather
than HTML, validates that its parsed Markdown AST remains equivalent within
its supported grammar, and is idempotent on the current Dynamo corpus. That
validation does not establish MDX equivalence.

An isolated candidate screen produced these results:

| Parser/renderer | Parse failures | Non-idempotent files |
| --- | ---: | ---: |
| `mdformat` / `markdown-it-py` | 0 | 0 |
| Mistune `MarkdownRenderer` | 1 | 7 |
| Marko `MarkdownRenderer` | 0 | 8 |

## Initial result

| Input | Files | Changed | Parse failures | Non-idempotent |
| --- | ---: | ---: | ---: | ---: |
| `.md` | 240 | 210 | 0 | 0 |
| `.mdx` | 24 | 16 | 0 | 0 |

The `.md` result is promising but creates broad formatting churn, especially
in large tables: output grows by 506,175 bytes (20.2%). The `.mdx` result is
not safe enough for a full-tree parse-and-render pipeline: `mdformat` does not
implement the MDX grammar, and it restructures fenced Markdown nested inside
JSX components such as `<Tab>`. For example,
`docs/kubernetes/gateway-api/quickstart.mdx` gains synthetic four-backtick
wrappers and moves content across JSX boundaries.

The generated tree still passes `fern check` with 0 errors and 2 warnings.
That confirms Fern's structural check does not detect this MDX semantic drift
and cannot serve as the round-trip equivalence test by itself.

The next experiment should use `markdown-it-py` for ordinary `.md` documents
and source-preserving destination edits for `.mdx`, unless a mature Python MDX
parser and Markdown renderer can be identified.
