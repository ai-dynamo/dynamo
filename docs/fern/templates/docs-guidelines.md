---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Documentation Content Guidelines
subtitle: Choose the right Dynamo documentation surface and authoring guide.
---

Use this page to decide where content belongs and which focused authoring guide to follow. For prose,
frontmatter, links, terminology, and validation requirements that apply to every page, see the
[Documentation Style Guide](../pages/community/contributing/documentation/documentation-style-guide.md).

## Choose an Authoring Guide

Give each page one primary reader goal.

| Content | Reader goal | Typical location | Authoring guide |
|---|---|---|---|
| User-facing guide | Get Dynamo running or complete a task | Kubernetes Guide, CLI Guide, or Use Cases | [Writing User-Facing Guides](user-facing-guides.md) |
| Knowledge base | Understand architecture, design, or implementation | Developer Guide | [Writing Knowledge Base Pages](knowledge-base-pages.md) |
| Reference | Look up an exact field, flag, API, environment variable, or contract | Reference | [Writing Reference Pages](reference-pages.md) |
| Home page | Understand what Dynamo is and choose the next action | Home | [Designing the Home Page](home-page.md) |
| Blog post | Read a dated technical or project article | Blog | [Writing Blog Posts](blog-posts.md) |
| Recipe or feature benchmark | Deploy a validated configuration or review measured evidence | Recipes | [Recipe and Feature Benchmark Authoring](../pages/recipes/_catalog/README.md) |

For custom layouts and styling, see [Site Design and Styling](site-design.md). For translated pages,
see [Maintaining Translations](translations.md).

## Decide Whether to Add a Page

Before creating a page, ask:

- Can the information fit cleanly into an existing page?
- Does it describe a substantial workflow or a distinct body of technical knowledge?
- Is the reader trying to follow instructions, understand a concept, or look up an exact value?
- Will a new page make the topic easier to find, or fragment related information?

Prefer extending an existing page when the new material serves the same reader goal.

## Organize a Topic

Use an overview page when a topic has multiple workflows or optional branches:

```text
Topic
├── Overview
├── Subtopic 1
└── Subtopic 2
```

The overview should introduce the topic, provide a basic path or high-level summary, and link to the
next pages. Deeper pages should cover optional or advanced material that readers can choose after
understanding the overview.

## Place Source Files

Match the source tree to the site structure:

- Use top-level directories that match the tab directories under `docs/fern/pages/`.
- Use nested directories for sidebar sections and topic groups.
- Keep filenames aligned with page titles and URL slugs.
- Do not use `README.md` as a published page name.
- Keep canonical user documentation under `docs/`. A README beside code, an example, or a recipe
  should document that local artifact and link to the canonical documentation when appropriate.
