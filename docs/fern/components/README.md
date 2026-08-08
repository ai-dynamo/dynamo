<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fern Custom Components

Components registered through `experimental.mdx-components: ./components` in
[`docs.yml`](../docs.yml). This file holds their page-usage examples.

## Why The Examples Live Here And Not In The Components

Fern's bundler scans every component source for import specifiers and treats
anything that is neither relative nor in its allowlist (`react`, `react-dom`,
`@mdx-js/react`, `next`) as a third-party dependency, then shells out to
`npx rolldown` to bundle it — a registry download on every docs build.

The scan is a regular expression over raw text. It does not skip comments. A
usage example written in a docblock is therefore indistinguishable from a real
dependency, and three of them were enough to make docs previews fail
intermittently with `rolldown exited with code 127` and
`ERR_MODULE_NOT_FOUND`.

The scan only looks at `.js`, `.jsx`, `.ts` and `.tsx`. Markdown is invisible to
it, so examples here can use ordinary syntax and stay copy-pasteable.
`../scripts/check_component_imports.py` enforces that no component source
carries a quoted non-relative specifier, in code or in a comment.

## Page Usage

Import the component, then place it in the page body. Ambient use without an
import renders `Unsupported JSX tag`. The `@/` prefix resolves to the `fern/`
root and is rewritten to a relative path at publish time.

### RecipeStyles

Once per recipe/benchmark page, and on the two landing READMEs, immediately
after the frontmatter.

```mdx
import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />
```

### ReferenceStyles

Once per Reference page that uses the Reference components, immediately after
the frontmatter.

```mdx
import { ReferenceStyles } from "@/components/ReferenceStyles";

<ReferenceStyles />
```

### LandingStyles

Once on `welcome.mdx` and `community/README.mdx`, immediately after the
imports.

```mdx
import { LandingStyles } from "@/components/LandingStyles";

<LandingStyles />
```

### BlogStyles

Once on the digest landing page and every article page, immediately after the
imports.

```mdx
import { BlogStyles } from "@/components/BlogStyles";

<BlogStyles />
```

### TerminalDemo

Props are documented in the component's own header.

```mdx
import { TerminalDemo } from "@/components/TerminalDemo";

<TerminalDemo
  src="/dynamo/assets/dynamo-demo.cast"
  startAt={0}
  endAt={18}
  idleTimeLimit={2}
  speed={1.2}
/>
```

## Adding A Component

- No third-party dependencies. There is no `package.json` alongside `fern/`, so
  anything outside Fern's allowlist has to be loaded at runtime instead — see
  `TerminalDemo.tsx`, which pulls asciinema-player from a CDN for this reason.
- Put the usage example in this file, not in the component's docblock.
- CSS bundles that must survive the production theme go in a page-level
  `<style>` block. Use `dangerouslySetInnerHTML` when the CSS contains `>`
  child combinators or `&`, as `RecipeStyles`, `LandingStyles` and `BlogStyles`
  do.
- Keep backticks out of any string inside a CSS template literal, comments
  included. A raw backtick closes the literal and the file stops compiling.
