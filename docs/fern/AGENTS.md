<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent instructions — docs, examples, recipes

When creating or editing files under `docs/`, `examples/`, or `recipes/`, follow the
[documentation style guide](pages/community/contributing/documentation/documentation-style-guide.md).
For **where** a new page belongs in the tab structure, read
[`pages/AGENTS.md`](pages/AGENTS.md) first — placement is decided before anything below matters.

## Authoring non-negotiables

- SPDX header on every file: frontmatter `#` form for Fern docs, `<!-- -->` for plain READMEs,
  full Apache block for code/config; copyright range `2025-2026`.
- Fern docs: `---` frontmatter with SPDX + at least one metadata key (`title`/`subtitle`/
  `sidebar-title`). Fern renders the page H1 from the nav `page:`, so do **not** add a body `# H1`
  (it duplicates the title); start the body at `##`.
- Every new page needs a `- page:` entry in `index.yml`. A page not in the nav is unreachable.
- Admonitions follow the source extension: use Fern callout components (`<Note>`, `<Tip>`,
  `<Info>`, `<Warning>`, `<Error>`) in `.mdx`, and GitHub-style blockquotes (`> [!NOTE]`) in `.md`.
- Links: relative + extension within `docs/`; absolute `github.com/ai-dynamo/dynamo` URLs for
  targets outside `docs/` (no `../` escapes).
- Code fences language-tagged (`bash`, not `sh`); backend casing vLLM / SGLang / TensorRT-LLM.
- No internal/sensitive refs (NVBug/JIRA IDs, internal hosts, secrets, TODO/FIXME) in shipped docs.
- Write for humans: no marketing/bombast, no filler, be concrete.

`scripts/docs_lint.py` enforces the deterministic subset (SPDX, frontmatter, body `# H1`, link
scope, dangling nav paths, internal references) as the `Docs Lint` job on every pull request, and
`fern check` plus `fern docs broken-links` run alongside it. Reproduce all three locally before
pushing (see [Validate](#validate)).

## This directory

`docs/fern/` holds both the content tree and the site configuration. Content lives in `pages/`;
everything else here is machinery:

| Path | Role |
|---|---|
| `pages/` | Every docs page — see [`pages/AGENTS.md`](pages/AGENTS.md) |
| `index.yml` | Navigation: tab map + per-tab layout. The only place a page becomes reachable |
| `docs.yml` | Site config, locales, landing page, versions, and `redirects:` |
| `main.css` | Site styles, including the pure-CSS recipe target-picker vocabulary |
| `components/` | React `.tsx` components used by `.mdx` pages |
| `templates/` | Page skeletons for new backend, component, and feature docs |
| `scripts/` | Build and sync tooling (callout conversion, translation links, snapshot rewrites) |
| `translations/` | Locale mirrors of `pages/` (`zh-CN/pages/<same relative path>`) |
| `assets/` | Images, diagrams, fonts |

Two gates on the machinery:

- Editing `main.css` requires running `python3 docs/fern/scripts/sync_site_css.py` so the footer's
  CSS mirror stays in sync. Pre-commit enforces this.
- The `docs-website` branch is CI-managed and must **never** be edited by hand. All authoring
  happens on `main` or a feature branch based on it.

## Validate

```bash
python3 scripts/docs_lint.py --scan docs              # the `Docs Lint` pull request job
fern check                                            # nav + frontmatter structure
fern docs broken-links                                # link resolution
python3 docs/fern/pages/recipes/_catalog/validate.py  # recipe or benchmark changes only
```

For how the site builds and publishes, see
[Building and Publishing](pages/community/contributing/documentation/building-and-publishing.md).
