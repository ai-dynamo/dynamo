/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Content-column alignment for docs pages that set hide-toc.
 *
 * From 80rem up, Fern shows the table of contents (#fern-toc) in a column to
 * the right of the content. hide-toc removes that column, so the content
 * wrapper grows into its width. The article is width-capped and centered in
 * the wrapper, so it moves right of where pages with a TOC put it, by about
 * 240px at a 1920px viewport, and the column jumps sideways when a reader
 * moves between the page and its neighbors. The rule below keeps the TOC
 * column empty instead. It uses the width variable and breakpoint that Fern
 * gives #fern-toc, so the page lines up with its neighbors at every width.
 *
 * The rule keys on :has(> #fern-toc) rather than a sibling combinator because
 * DOM order differs between builds: the TOC aside comes before the content
 * wrapper, and the hosted build also renders the sidebar aside after it.
 *
 * Delivered as a page-level <style> block (NOT via the docs.yml `css:` field)
 * so it survives the shared NVIDIA global theme, which replaces project `css`
 * in every hosted build, PR previews included. Same pattern as
 * LandingStyles.tsx, BlogStyles.tsx and RecipeStyles.tsx, which handle
 * hide-toc for their own wider layouts. Injected via dangerouslySetInnerHTML
 * because a text child is escaped on render, turning the > child combinators
 * into &gt; and dropping the rule.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. It must be imported (ambient
 * use renders "Unsupported JSX tag"), then placed once, right after the
 * imports. The page-usage example lives in README.md, for the reason recorded
 * there.
 */
const HIDE_TOC_CSS = `
/* Hold the hidden TOC column open so the article lines up with pages that
   have a TOC. */
@media (min-width: 80rem) {
  .fern-main:not(:has(> #fern-toc)) > .fern-layout-content-wrapper {
    margin-right: var(--sticky-aside-width);
  }
}
`;

export function HideTocStyles() {
  return <style dangerouslySetInnerHTML={{ __html: HIDE_TOC_CSS }} />;
}
