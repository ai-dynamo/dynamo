/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PrInlineComments — SSR mount point for the read-only PR line-comment overlay.
 *
 * The runtime that fetches the PR's line comments and anchors them into
 * `.fern-prose` lives in `fern/js/dep-pr-comments.js` and is injected by Fern
 * via `docs.yml` `js:`. See that file for the WHY (two earlier attempts to run
 * the runtime from inside this component failed on the live preview).
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
  /** Pull request number whose review line comments are mirrored. */
  pr: number;
  /** Repository owner (default "ai-dynamo"). */
  owner?: string;
  /** Repository name (default "dynamo"). */
  repo?: string;
  /** Only show comments on this file path (default the example DEP). */
  path?: string;
  /** If true, comment cards are expanded on first render. Default collapsed. */
  autoOpen?: boolean;
}

export function PrInlineComments({
  pr,
  owner = "ai-dynamo",
  repo = "dynamo",
  path = "docs/proposals/0000-example-dep.mdx",
  autoOpen = false,
}: PrInlineCommentsProps) {
  return (
    <div
      id="dep-pr-comments"
      data-pr={String(pr)}
      data-owner={owner}
      data-repo={repo}
      data-path={path}
      data-auto-open={autoOpen ? "true" : "false"}
    />
  );
}
