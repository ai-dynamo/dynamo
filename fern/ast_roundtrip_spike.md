<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Markdown AST round-trip spike

This spike evaluates whether a Python Markdown parser and renderer can serve as
the unified preprocessing pipeline for Dynamo documentation.

Run it without modifying the source tree:

```bash
uv run fern/ast_roundtrip_spike.py \
  --input docs \
  --repo-root . \
  --output /tmp/dynamo-docs-ast-preprocessed \
  --report /tmp/dynamo-docs-ast-preprocessed.json \
  --ref main \
  --strict \
  --force
```

The prototype uses `mdformat` with its GFM, GFM-alert, YAML-frontmatter, and
local Fern extensions. `mdformat` is built on `markdown-it-py`, renders
Markdown rather than HTML, validates that its parsed Markdown AST remains
equivalent within its supported grammar, and is idempotent on the current
Dynamo corpus.

The spike uses two local `markdown-it-py` plugins:

- [`markdown_it_fern_mdx.py`](markdown_it_fern_mdx.py) emits paired container
  tokens for standalone uppercase MDX tags, recursively parses their children
  as Markdown, and emits raw tokens for MDX comments and self-closing tags.
- [`markdown_it_relative_links.py`](markdown_it_relative_links.py) runs after
  inline parsing and rewrites repository-relative link and image tokens before
  `mdformat` renders the tree.

Both modules implement `mdformat`'s parser-extension interface. No pre-parse
source transformation or post-render link scanner is required.

Run the focused plugin fixtures with:

```bash
uv run --no-project --python 3.12 \
  --with mdformat==1.0.0 \
  --with mdformat-frontmatter==2.1.2 \
  --with mdformat-gfm==1.0.0 \
  --with mdformat-gfm-alerts==2.0.0 \
  python -m unittest \
    fern.test_ast_roundtrip_links \
    fern.test_markdown_it_fern_mdx \
    fern.test_link_rewrite_spike
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

The `.md` result creates broad formatting churn, especially in large tables:
output grows by 506,175 bytes (20.2%). The `.mdx` result is not safe enough for
a full-tree parse-and-render pipeline: `mdformat` does not
implement the MDX grammar, and it restructures fenced Markdown nested inside
JSX components such as `<Tab>`. For example,
`docs/kubernetes/gateway-api/quickstart.mdx` gains synthetic four-backtick
wrappers and moves content across JSX boundaries.

The generated tree still passes `fern check` with 0 errors and 2 warnings.
That confirms Fern's structural check does not detect this MDX semantic drift
and cannot serve as the round-trip equivalence test by itself.

## Latest AST transformation result

| Input | Files | Changed | Unchanged | Source bytes | Output bytes | Growth | Parse failures | Non-idempotent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `.md` | 240 | 213 | 27 | 2,505,077 | 3,014,978 | 509,901 (20.4%) | 0 | 0 |
| `.mdx` | 24 | 17 | 7 | 239,433 | 250,481 | 11,048 (4.6%) | 0 | 0 |
| **Total** | **264** | **230** | **34** | **2,744,510** | **3,265,459** | **520,949 (19.0%)** | **0** | **0** |

The changed-file count is not a semantic-quality score: the plugin exposes
Markdown inside MDX containers to `mdformat`, so that content is now formatted
instead of being treated as one opaque HTML block. The important improvement
is structural. On `docs/kubernetes/gateway-api/quickstart.mdx`, the unextended
renderer changed 230 lines and moved fenced content across JSX boundaries;
the plugin-enabled renderer adds only three blank lines between sibling
`<Tab>` components. It no longer synthesizes four-backtick wrappers or moves
content outside its component.

The result still includes broad, intentional Markdown normalization, especially
for tables. The MDX plugin is also a targeted Fern component grammar rather
than a complete MDX implementation. Within those accepted boundaries, the
combined AST pipeline is structurally stable and idempotent.

## Relative link transformation

The relative-link plugin operates on parsed tokens rather than source spans:

1. A core rule runs after inline parsing and collects `link_open` and `image`
   tokens. It also groups reference-style uses by their resolved definition.
2. Links that remain inside `docs/`, fragments, root-relative paths, and
   already-absolute URLs are left unchanged.
3. A relative destination that escapes `docs/` must resolve to an existing
   path inside the repository. Files use a GitHub `/blob/<ref>/` URL,
   directories use `/tree/<ref>/`, and images use
   `raw.githubusercontent.com`. Query strings and fragments are preserved.
4. Reference definitions and all tokens that use them are updated together.
   A reference shared by link and image syntax uses the raw image URL, matching
   the source-preserving implementation.
5. Rewrites are atomic per file. If any eligible target cannot be resolved,
   all otherwise-valid replacements in that file are reported as skipped.
   `--strict` makes unresolved targets fail the command.

Because rendering already consumes the token tree, this approach does not
need source positions, unique sentinel URLs, candidate scanning, or
parser-to-source reconciliation.

The current corpus produced 100 replacements in 25 files, with no skipped or
unresolved targets. The multiset of `(file, original destination, replacement,
parser kind)` records exactly matches the
[source-preserving link rewrite spike](link_rewrite_spike.md). The final tree
is also byte-for-byte identical to running that source rewrite first and then
performing the AST round trip.

## Validation

The final plugin implementation passed these checks:

- The plugin-enabled round trip completed all 264 Markdown and MDX files with
  no parser failures or non-idempotent files.
- The AST link transformation generated the same 100 replacement records as
  the source-preserving implementation, with no skipped or unresolved targets.
- The direct AST output was byte-for-byte identical to the output from a
  source-preserving rewrite followed by the same round-trip renderer.
- All seven focused AST, MDX plugin, and parser-oracle tests passed. Coverage
  includes files, directories, images, query strings, fragments, reference
  definitions, shared image/link references, unresolved-target atomicity,
  nested components, multiline JSX attributes, fenced examples, inline MDX
  comments, and MDX comments used as list-item placeholders.
- Ruff reported no lint or formatting issues in the spike implementation and
  tests.
- Both `fern check` and `fern docs broken-links` passed against a Fern tree
  built from the plugin-enabled round-trip output.
