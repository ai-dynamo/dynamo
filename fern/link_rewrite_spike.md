<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source-preserving Markdown link rewrite spike

This spike tests a hybrid preprocessing strategy: use a Markdown parser to
decide which source fragments are links, but edit only the destination bytes
in the original file. It avoids the formatting churn and MDX corruption found
by the full parse-and-render experiment.

Run it without modifying the source tree:

```bash
uv run fern/link_rewrite_spike.py \
  --input docs \
  --repo-root . \
  --output /tmp/dynamo-source-link-spike \
  --report /tmp/dynamo-source-link-spike.json \
  --ref main \
  --strict \
  --force
```

The output rewrites relative links that resolve outside `docs/`, but remain
inside the repository, to exact GitHub URLs. Files use `/blob/<ref>/`,
directories use `/tree/<ref>/`, and images use an exact
`raw.githubusercontent.com` URL. Query strings and fragments are preserved.

## Source mapping strategy

The implementation does not depend on AST line or column metadata:

1. A scanner locates possible inline-link and reference-definition destination
   spans.
2. Every candidate span is replaced in memory with a unique temporary URL.
3. `markdown-it-py` parses that instrumented document once. Only candidate IDs
   emitted as links, images, or definitions are accepted.
4. A parser-only view blanks MDX comments and standalone Fern component tags,
   and removes structural indentation from their children. This exposes links
   inside components such as `<Info>` and `<Tab>` without changing source.
5. Accepted relative destinations are resolved on disk. The tool refuses to
   rewrite a file when the parser and scanner cannot be reconciled or when a
   parser-confirmed target cannot be resolved.
6. Replacements are applied from the end of the original source toward the
   beginning. The rewritten document is parsed again to verify that every new
   URL remains a real Markdown destination.

Because the parser returns candidate IDs instead of source positions, duplicate
URLs, links on the same line, multiline JSX tags, and shifting replacement
lengths do not make the mapping ambiguous.

## Corpus result

The current `docs/` corpus produced:

| Measurement | Result |
| --- | ---: |
| Markdown and MDX files | 264 |
| Scanner candidates | 3,116 |
| Parser-confirmed destinations | 3,083 |
| Parser-rejected examples | 33 |
| Outside-docs rewrites | 100 |
| Changed files | 25 |
| Unresolved targets | 0 |
| Parser/scanner mapping errors | 0 |

The 33 rejected candidates are all example syntax in fenced code, inline code,
regular-expression text, or an MDX comment. A focused fixture also verified an
indented link inside a Fern component, an angle-bracket reference definition,
an image, a fenced example, and inline code. The first three were rewritten;
both code examples were left unchanged.

The generated docs changed 99 source lines across 25 `.md` files. Every diff
hunk changes only link destinations; no prose, tables, fences, frontmatter,
HTML, or JSX was rendered or reformatted. The generated Fern tree passes both
`fern check` and `fern docs broken-links`.

## Remaining boundaries

- This handles Markdown links, images, and reference definitions. Explicit
  HTML or JSX `href` and `src` attributes need a separate syntax-aware pass.
- The parser-only MDX view supports the standalone Fern component structure in
  the current corpus; it is not a complete MDX parser. Scanner candidates that
  are not parser-confirmed remain visible in the JSON report.
- The spike copies a tree for inspection. Integrating it into publishing should
  combine this pass, callout conversion, GitHub `main`-to-tag pinning, and the
  comment transformation behind one preprocessing entry point.
