/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Giscus comment widget for Dynamo Enhancement Proposal (DEP) pages.
 *
 * Giscus (https://giscus.app) renders a GitHub-Discussions-backed comment
 * thread on the public docs page, so any GitHub user can join the open-ended
 * debate on a DEP without needing write access to the repo. This is the
 * "open-ended debate" layer of the hybrid DEP workflow (line-level review
 * still happens on the draft PR; the tracking issue is the anchor).
 *
 * Why this shape:
 *   - Server component (no "use client"), mirroring RecipeStyles.tsx. Fern
 *     registers components via docs.yml `experimental.mdx-components:
 *     ./components`. Components must be IMPORTED on each page (ambient use
 *     renders "Unsupported JSX tag"):
 *       import { GiscusComments } from "@/components/GiscusComments";
 *     The @/ prefix resolves to the fern/ root and is rewritten at publish.
 *   - The <script> is giscus's official embed, verbatim as JSX. giscus's
 *     client.ts reads its own `document.currentScript.dataset`, so the tag
 *     must render into the page where the widget should appear. This is the
 *     same delivery mechanism Fern already uses for external JS (docs.yml
 *     `js:` field) and for inline markup (RecipeStyles).
 *   - The wrapper CSS is injected as a page-level <style> (not via docs.yml
 *     `css:`) so it survives the shared NVIDIA global theme at publish, per
 *     the constraint documented in RecipeStyles.tsx.
 *
 * CONFIG (repo-wide, populated below):
 *   data-repo-id / data-category-id are the GitHub node IDs for
 *   `ai-dynamo/dynamo` and its "Ideas" Discussions category, resolved from the
 *   GraphQL API (see the constants below). GitHub Discussions is enabled on the
 *   repo. `data-mapping="pathname"` means each DEP page auto-creates/binds its
 *   own Discussion the first time someone comments, so no per-page config is
 *   needed. See docs/proposals/README.md.
 *
 *   REMAINING FOLLOW-UP (org-admin action, cannot be done from this repo): the
 *   giscus GitHub App (https://github.com/apps/giscus) must be installed on
 *   `ai-dynamo/dynamo` with write access to Discussions before the widget can
 *   accept comments. Until then the box renders but posting returns an error.
 *
 * VERIFY IN FIRST PREVIEW: confirm the widget renders inline at the bottom of
 * the page (not hoisted). If a future Fern/React upgrade hoists `async` scripts
 * with `src`, drop the `async` attribute so the tag stays in document order.
 */

// Resolved 2026-07-14 from the GitHub GraphQL API for ai-dynamo/dynamo:
//   repository.id                          -> data-repo-id
//   discussionCategories "Ideas".id        -> data-category-id
// Re-fetch with:
//   gh api graphql -f query='query{repository(owner:"ai-dynamo",name:"dynamo"){id discussionCategories(first:30){nodes{id name}}}}'
const GISCUS_REPO = "ai-dynamo/dynamo";
const GISCUS_REPO_ID = "R_kgDOOCjvsg";
const GISCUS_CATEGORY = "Ideas";
const GISCUS_CATEGORY_ID = "DIC_kwDOOCjvss4Cuvrp";

const GISCUS_CSS = `
.dep-giscus {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--grayscale-a4, #e5e5e5);
}
.dep-giscus-heading {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.dep-giscus-note {
    font-size: 0.9rem;
    color: var(--grayscale-a9, #777);
    margin-bottom: 1rem;
}
`;

export function GiscusComments() {
  return (
    <div className="dep-giscus">
      <style dangerouslySetInnerHTML={{ __html: GISCUS_CSS }} />
      <p className="dep-giscus-heading">Discussion</p>
      <p className="dep-giscus-note">
        Comments are powered by GitHub Discussions via giscus. Sign in with any
        GitHub account to join the debate on this proposal. If the thread does
        not load, use the &quot;Discuss this DEP&quot; link near the top of the
        page.
      </p>
      <script
        src="https://giscus.app/client.js"
        data-repo={GISCUS_REPO}
        data-repo-id={GISCUS_REPO_ID}
        data-category={GISCUS_CATEGORY}
        data-category-id={GISCUS_CATEGORY_ID}
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossOrigin="anonymous"
        async
      />
    </div>
  );
}
