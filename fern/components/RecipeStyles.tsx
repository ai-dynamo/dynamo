/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Recipe & Feature Benchmark component styles.
 *
 * Delivered as a page-level <style> block (NOT via the docs.yml `css:` field)
 * so it survives the shared NVIDIA global theme, which replaces project `css`
 * at publish. Mirrors the prod-proven pattern in NVIDIA-NeMo/DataDesigner
 * (fern/components/BlogCard.tsx), which runs the same `global-theme: nvidia`
 * with no `css:` field and injects product CSS this exact way.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. Use <RecipeStyles /> once at the
 * top of any recipe/benchmark page.
 *
 * PROTOTYPE: currently a single sentinel rule to prove the mechanism end-to-end
 * (a component-emitted <style> renders AND a :has() selector applies). The full
 * `.dynamo-*` stylesheet moves here once the mechanism is confirmed.
 */
const RECIPE_CSS = `
/* SENTINEL: crimson picker title proves this component's CSS is applied.
   The ::after via body:has(...) also proves a :has() selector survives —
   that combinator is the core of the pure-CSS picker. */
.dynamo-target-picker-title { color: rgb(220, 20, 60) !important; }
body:has(.dynamo-target-picker) .dynamo-target-picker-title::after {
  content: " ✓ styled-by-component";
  font-size: 11px;
  color: rgb(220, 20, 60);
}
`;

export function RecipeStyles() {
  return <style dangerouslySetInnerHTML={{ __html: RECIPE_CSS }} />;
}
