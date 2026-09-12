---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Maintaining Translations
subtitle: Keep translated Dynamo pages aligned with their English source pages.
---

Translation sources mirror the English documentation tree. Keep the source path, filename, page
type, and navigation intent aligned so Fern can pair each translated page with its English page.

## Mirror the English Source Tree

Chinese translation sources live under `docs/fern/translations/zh-CN/pages/`. Use the same relative
path, filename, and extension as the English source under `docs/fern/pages/`.

When an English page moves, changes extension, or is removed, update its translation in the same
change when that translation exists. Preserve heading anchors and relative links where possible.

## Edit Source Mirrors Only

On the `docs-website` branch, the publication workflow copies translation sources into
`fern/translations/zh-CN/pages-dev/`. Release automation creates versioned snapshots from the
corresponding release source.

Edit translation sources on `main`; do not hand-edit `pages-dev` or versioned translation snapshots
on `docs-website`.

## Apply the Documentation Standards

Translated pages follow the same frontmatter, SPDX, heading, link, component, and validation rules
as their English source. Keep product names, backend names, commands, flags, and code identifiers in
their required technical form.
