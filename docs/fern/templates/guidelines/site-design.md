---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Site Design and Styling
subtitle: Add custom Dynamo documentation layouts without breaking the published site.
---

Use the default Fern presentation for standard documentation pages. Add custom layout or styling
only when a landing page, catalog, or specialized reference surface needs behavior that the standard
components do not provide.

## Make Conditional Content Clear

Use tabs when readers follow the same sequence with different backends, deployment modes,
providers, or environments. When a tab contains prose, multiple code blocks, or other substantial
content, give the tab panel a clear visual boundary so readers can tell which content changes with
the selection.

Do not use tabs to hide materially different procedures. Use separate sections or pages when the
sequence itself changes.

## Use the Existing Style Systems

Follow the established page-scoped style component for specialized surfaces:

| Surface | Style component |
|---|---|
| Home and Community | `LandingStyles` |
| Blog | `BlogStyles` |
| Recipes and feature benchmarks | `RecipeStyles` |
| Custom Reference components | `ReferenceStyles` |

Import the component explicitly and render it once near the start of the page. Do not duplicate its
rules in page-local inline styles.

## Choose Between Page-Scoped and Shared CSS

The shared NVIDIA global theme can replace the project stylesheet link during publishing. Custom
landing and catalog styles therefore use page-level style components so they survive the published
build.

Keep genuinely shared site chrome rules in `docs/fern/main.css`. After changing it, run:

```bash
python3 docs/fern/scripts/sync_site_css.py
```

The script mirrors the shared stylesheet into `CustomFooter.tsx`; pre-commit verifies that the copy
is current. Do not put Home, Blog, Recipe, or custom Reference component rules in `main.css` when
their page-scoped style component owns them.

## Verify Published Behavior

Make source changes on `main` or a feature branch based on `main`. The `docs-website` branch is
CI-managed and must not be edited directly.

For layout or styling bugs, verify both the local Fern preview and the generated publication
structure. The shared theme, rewritten paths, and injected components can make production behavior
differ from a basic local render. Use `docs/fern/scripts/simulate_docs_website.sh` when the change
depends on the publication workflow.
