---
name: dynamo-docs
description: Adds, updates, moves, or removes content on the Dynamo Fern docs site, local code examples, and translations while preserving the documentation style guide. Use for changes under docs/ or examples/, or when a page needs its frontmatter, headings, links, callouts, terminology, or navigation corrected.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - docs
    - fern
    - style-guide
---

# Dynamo Docs Maintenance

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Use this skill to maintain the Dynamo Fern documentation site and the READMEs under `examples/`. Runnable Kubernetes
deployment examples live in [`examples/deployments/kubernetes`](../../../examples/deployments/kubernetes).

## Read first

- [`docs/fern/AGENTS.md`](../../../docs/fern/AGENTS.md) — required documentation conventions.
- [`docs/fern/pages/community/contributing/documentation/documentation-style-guide.md`](../../../docs/fern/pages/community/contributing/documentation/documentation-style-guide.md) — source of truth for frontmatter, structure, prose, links, and validation.
- [`docs/fern/index.yml`](../../../docs/fern/index.yml) — live navigation; choose the nearest existing page before adding a new one.

## Branch rule

Make source edits on `main` or a feature branch based on `main`. The `docs-website` branch is CI-managed and must not be edited manually.

## Required conventions

- Add SPDX headers to every changed file. Fern pages use the two `#` SPDX lines inside `---` frontmatter; plain READMEs use the HTML-comment form; code and configuration use the full Apache block.
- Fern pages need frontmatter with `title`, `subtitle`, or `sidebar-title`; do not add a body `# H1`, because Fern renders the title from navigation.
- Use relative links with extensions within `docs/`. For code, external repositories, or other targets outside `docs/`, use an absolute GitHub URL. Do not use `../` to escape `docs/`.
- Keep backend spelling exact: vLLM, SGLang, TensorRT-LLM. Use Kubernetes in prose, not `k8s`.
- Use Fern callout components in `.mdx`; use GitHub-style admonitions in `.md`. Tag every code fence with a language.
- Do not ship internal URLs, issue identifiers, secrets, or `TODO`/`FIXME` markers.

## Operations

### Add or update a page

1. Locate the closest existing page in `docs/fern/index.yml`, then add the source file alongside it with a kebab-case name.
2. Start Fern pages with SPDX frontmatter and a short purpose statement. Begin content at `##`.
3. Add or update the matching `page:` entry in `docs/fern/index.yml`. A page absent from navigation is not published.
4. Prefer one page type per page: tutorial, how-to, reference, or explanation. Split unrelated reader goals and link the canonical page.
5. Update internal links and anchors after renames or moves; remove navigation entries and inbound references before deleting a page.

### Link runnable deployment examples

Kubernetes deployment manifests belong under `examples/deployments/kubernetes`. Link to the narrowest matching directory under:

`https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/kubernetes`

The supported topology set includes aggregated and disaggregated vLLM, TensorRT-LLM, and SGLang deployments. It also includes KV-cache offload, multimodal, and multi-node vLLM examples.

### Maintain local code examples

- Keep each example scoped to one workflow and include a README that states prerequisites, commands, expected result, and cleanup when resources persist.
- Link local example READMEs from the nearest relevant documentation page.
- Keep canonical explanatory material in `docs/`; an example README should explain only the code or command in its directory.

### Translation and versioned navigation

- Preserve the source page's information hierarchy when updating `.zh-CN` content; do not translate identifiers, flags, file paths, or URLs.
- Treat versioned navigation as a compatibility surface. Update only the version that contains the changed page, and do not copy current-only pages into historical releases.

## Validation

Run the narrowest useful checks for the changed files:

```bash
git diff --check
```

For a navigation or docs-site change, inspect every edited `page:` path in `docs/fern/index.yml` and run the available Fern validation command. For Markdown and README changes, verify links, anchors, SPDX form, and code-fence language tags. For configuration examples, parse the changed YAML before commit.

## Review checklist

- Frontmatter has SPDX and metadata; no duplicate body H1.
- New, moved, and deleted pages are reflected in the right navigation entry.
- Links resolve, use the correct relative/absolute form, and have descriptive text.
- Terminology, heading case, and admonition syntax are consistent with the style guide.
- A runnable deployment link targets the narrowest matching path under `examples/deployments/kubernetes`.
