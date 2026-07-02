<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parser-free Markdown link rewrite spike

This spike implements the source-preserving link rewrite without a Markdown
parser or third-party Python dependencies. It uses a purpose-built lexer to
identify protected source regions and Markdown destination spans.

Run it without modifying the source tree:

```bash
uv run fern/link_rewrite_lexer_spike.py \
  --input docs \
  --repo-root . \
  --output /tmp/dynamo-lexer-link-spike \
  --report /tmp/dynamo-lexer-link-spike.json \
  --ref main \
  --strict \
  --force
```

Run the focused regression fixtures with:

```bash
uv run --no-project --python 3.12 \
  python -m unittest fern.test_link_rewrite_lexer_spike
```

## Lexer strategy

The implementation makes two passes over each source file:

1. Mark YAML frontmatter, MDX comments, HTML comments, raw code elements,
   fenced code blocks, inline code spans, and HTML or JSX tags as protected.
2. Outside those ranges, recognize inline links, images, nested badge images,
   and reference definitions with a small state machine. It handles escapes,
   angle-bracket destinations, balanced parentheses, optional titles, and
   reference-style image versus link usage.

Recognized relative paths are resolved on disk. Targets within `docs/` remain
relative; existing files outside `docs/` become `/blob/<ref>/` URLs,
directories become `/tree/<ref>/` URLs, and images use an exact
`raw.githubusercontent.com` URL. Replacements are applied in reverse source
order and the rewritten file is lexed again to verify every new destination.
If any recognized target in a file is unresolved, that file is left unchanged.

## Corpus comparison

| Measurement | Parser oracle | Parser-free lexer |
| --- | ---: | ---: |
| Markdown and MDX files | 264 | 264 |
| Recognized source destinations | 3,083 | 3,072 |
| Outside-docs rewrites | 100 | 100 |
| Changed files | 25 | 25 |
| Skipped replacements | 0 | 0 |
| Unresolved targets | 0 | 0 |
| Mapping or lexer errors | 0 | 0 |

The generated trees are byte-for-byte identical. The lexer intentionally omits
11 absolute author links embedded in YAML `subtitle` values; the CommonMark
parser sees those as body Markdown because it does not understand frontmatter.
None of those links are rewrite candidates.

The focused parser-comparison fixture also produces identical output: an
indented Fern-component link, angle-bracket reference definition, and image are
rewritten, while fenced and inline-code examples are not. A larger adversarial
fixture verifies frontmatter, HTML and MDX comments, JSX attributes, a shell
heredoc inside a fence, balanced and angle-bracket destinations, nested badge
images, and image reference definitions. It rewrites the six real destinations
and leaves all seven protected regions unchanged.

The generated corpus passes `fern check` and `fern docs broken-links` after the
normal callout conversion step.

## Tradeoffs

The parser-free result is viable for the current corpus, but the apparent
simplicity does not survive the edge cases. The standalone lexer is about 1,000
lines, compared with about 650 lines for the standalone parser-oracle spike,
because it owns more Markdown and MDX-adjacent syntax.

Its deliberate grammar boundaries are:

- Inline code spans are matched within one source line. Multiline CommonMark
  code spans are not supported.
- Fenced code is protected at any indentation, including inside Fern
  components. Legacy four-space indented code blocks are not recognized; the
  documentation style guide requires fenced code blocks.
- HTML and JSX tag recognition is lexical, not a complete JSX expression
  parser. Explicit `href` and `src` attributes are protected, not rewritten.
- Reference and nested-image handling covers the forms in this corpus but does
  not claim full CommonMark conformance for malformed or deeply nested input.

Those boundaries should become regression fixtures if this implementation is
promoted from a spike; the current tests cover the corpus-relevant and
fail-closed cases described above. The parser-oracle version keeps the same
byte-preserving mutation model while delegating the final “is this Markdown?”
decision to a mature grammar, so it remains the lower-maintenance option
despite its one small Python dependency.
