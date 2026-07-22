/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PrInlineComments — SSR mount point for the read-only GitHub comment mirror.
 *
 * The runtime lives in `fern/js/dep-pr-comments.js` (injected by Fern via
 * `docs.yml` `js:`) and, reading this div's data-* attributes, renders three
 * read-only surfaces:
 *   1. inline PR review line comments  -> anchored <mark> highlights + cards,
 *   2. the PR conversation             -> a "Revision discussion" thread,
 *   3. the tracking-issue comments     -> a "Design discussion" thread.
 * Each surface deep-links back to GitHub for replies; the page never posts.
 * See that file for the WHY (two earlier attempts to run the runtime from
 * inside this component failed on the live preview).
 *
 * This component renders ONLY a static <div id="dep-pr-comments" data-*=...>.
 * SSR of a plain div is reliable; the client script finds it, reads its data
 * attributes, and populates it. No React state, no useEffect, no injected
 * <script> — everything the runtime needs is on the mount div's attributes.
 *
 * Registered via docs.yml `experimental.mdx-components: ./components`. Import
 * per page: `import { PrInlineComments } from "@/components/PrInlineComments";`
 */

interface PrInlineCommentsProps {
  /** Pull request number: source of inline review + revision-discussion comments. */
  pr: number;
  /** Tracking-issue number: source of the design-discussion thread. Omit to show the empty "open a tracking issue" state. */
  issue?: number;
  /** Repository owner (default "ai-dynamo"). */
  owner?: string;
  /** Repository name (default "dynamo"). */
  repo?: string;
  /**
   * Only show inline review comments whose GitHub `path` matches this value.
   * Default is empty (no filter): the runtime shows every inline review
   * comment on the PR. Set this on multi-file PRs so a DEP page only mirrors
   * comments on its own markdown file. Must be the repo-relative path GitHub
   * uses (e.g. `docs/proposals/0000-example-dep.mdx`).
   */
  path?: string;
  /** If true, comment cards are expanded on first render. Default collapsed. */
  autoOpen?: boolean;
}

export function PrInlineComments({
  pr,
  issue,
  owner = "ai-dynamo",
  repo = "dynamo",
  path = "",
  autoOpen = false,
}: PrInlineCommentsProps) {
  return (
    <div
      id="dep-pr-comments"
      data-pr={String(pr)}
      data-issue={issue ? String(issue) : ""}
      data-owner={owner}
      data-repo={repo}
      data-path={path}
      data-auto-open={autoOpen ? "true" : "false"}
    />
  );
}
