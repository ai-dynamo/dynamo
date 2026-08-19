---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Designing the Home Page
subtitle: Keep the Dynamo landing page focused, distinctive, and easy to navigate.
---

The Home page is a landing surface, not a conventional documentation article. It should explain
what Dynamo is, establish the primary value proposition, and help visitors choose their next action.

For general page placement, see the
[Documentation Content Guidelines](docs-guidelines.md). For custom styling, see
[Site Design and Styling](site-design.md).

## Keep the Landing Page Focused

- Lead with a hero that communicates what Dynamo does.
- Give the hero one primary action instead of several competing calls to action.
- Add supporting sections only when they help a new visitor understand Dynamo or choose a next step.
- Avoid duplicating social and community links that already appear in global notification or
  community surfaces.

## Use the Landing Layout

Use `layout: page` to remove the normal documentation sidebar and table of contents. Do not add a
separate Home sidebar or navigation tree. Hide page actions when they do not make sense for the
landing experience.

The current Home page lives at `docs/fern/pages/home/index.mdx` and is the only entry in the Home
tab.

## Use Purpose-Built Components

Compose the page from focused components rather than recreating a marketing layout in raw Markdown.
Import and render `LandingStyles` once when using the Home or Community landing components.
