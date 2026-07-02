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

The prototype uses `mdformat` with its GFM, GFM-alert, YAML-frontmatter, and
local Fern MDX extensions. `mdformat` is built on `markdown-it-py`, renders
Markdown rather than HTML, validates that its parsed Markdown AST remains
equivalent within its supported grammar, and is idempotent on the current
Dynamo corpus.

The local [`markdown_it_fern_mdx.py`](markdown_it_fern_mdx.py) module is a real
`markdown-it-py` plugin rather than a pre-parse source transformation. Its
block rules emit paired container tokens for standalone uppercase MDX tags,
recursively parse their children as Markdown, and emit raw tokens for MDX
comments and self-closing tags. It also implements `mdformat`'s
`update_mdit`/`RENDERERS` extension interface, so the parser-oracle and
parse-and-render spikes use the same MDX grammar.

Run the focused plugin fixtures with:

```bash
uv run --no-project --python 3.12 \
  --with mdformat==1.0.0 \
  --with mdformat-frontmatter==2.1.2 \
  --with mdformat-gfm==1.0.0 \
  --with mdformat-gfm-alerts==2.0.0 \
  python -m unittest fern.test_markdown_it_fern_mdx
```

An isolated candidate screen produced these results:

| Parser/renderer | Parse failures | Non-idempotent files |
| --- | ---: | ---: |
| `mdformat` / `markdown-it-py` | 0 | 0 |
| Mistune `MarkdownRenderer` | 1 | 7 |
| Marko `MarkdownRenderer` | 0 | 8 |

## Result without the MDX plugin

| Input | Files | Changed | Parse failures | Non-idempotent |
| --- | ---: | ---: | ---: | ---: |
| `.md` | 240 | 210 | 0 | 0 |
| `.mdx` | 24 | 16 | 0 | 0 |

The `.md` result creates broad formatting churn, especially
in large tables: output grows by 506,175 bytes (20.2%). The `.mdx` result is
not safe enough for a full-tree parse-and-render pipeline: `mdformat` does not
implement the MDX grammar, and it restructures fenced Markdown nested inside
JSX components such as `<Tab>`. For example,
`docs/kubernetes/gateway-api/quickstart.mdx` gains synthetic four-backtick
wrappers and moves content across JSX boundaries.

The generated tree still passes `fern check` with 0 errors and 2 warnings.
That confirms Fern's structural check does not detect this MDX semantic drift
and cannot serve as the round-trip equivalence test by itself.

## Result with the MDX plugin

| Input | Files | Changed | Parse failures | Non-idempotent |
| --- | ---: | ---: | ---: | ---: |
| `.md` | 240 | 208 | 0 | 0 |
| `.mdx` | 24 | 17 | 0 | 0 |

The changed-file count is not a semantic-quality score: the plugin exposes
Markdown inside MDX containers to `mdformat`, so that content is now formatted
instead of being treated as one opaque HTML block. The important improvement
is structural. On `docs/kubernetes/gateway-api/quickstart.mdx`, the unextended
renderer changed 230 lines and moved fenced content across JSX boundaries;
the plugin-enabled renderer adds only three blank lines between sibling
`<Tab>` components. It no longer synthesizes four-backtick wrappers or moves
content outside its component.

This establishes that one parser plugin can remove the duplicated MDX source
masking and make both experiments understand the current Fern component
shape. It does not make full-tree rendering the preferred preprocessor:
ordinary Markdown still sees broad table and whitespace churn, and this is a
targeted Fern component grammar rather than a complete MDX implementation.

The follow-up [source-preserving link rewrite spike](link_rewrite_spike.md)
therefore remains the safer preprocessor shape: it uses the same plugin with
`markdown-it-py` as a syntax oracle, then changes destination spans only in the
original source.
