---
name: dep-update
description: Advances a Dynamo Enhancement Proposal through its lifecycle under the current model -- a short follow-up PR in ai-dynamo/enhancements that updates BOTH the status field and the status-banner admonition in sync (Draft -> Under Review -> Accepted/Rejected/Deferred -> Implemented -> Replaced). Approval is by the owning SIG's chairs/tech leads; the rendered metadata pill and sidebar pill update on the next docs build. Use when moving a DEP through review, recording a decision, or marking it implemented or replaced.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - dep
    - enhancement-proposal
    - enhancements
    - sig
    - lifecycle
---

# Skill: Advance a DEP's Lifecycle

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Move a DEP through its lifecycle. Under the current model a DEP is a markdown
file `deps/NNNN-slug.md` in **`ai-dynamo/enhancements`**, and its status is
advanced by a **short follow-up PR** that edits the DEP's `status` field and its
status-banner admonition together. This replaces the deprecated flow that
flipped `dep:draft` / `dep:under-review` / `dep:approved` labels on a GitHub
*issue* in `ai-dynamo/dynamo`.

See `docs/proposals/README.mdx` and `docs/proposals/0001-dep-process.mdx` for
the model, roles, and the status-banner convention. Do not restate the whole
model here.

## When to Use

When requesting a decision on a DEP (moving it to Under Review), recording an
Accepted / Rejected / Deferred decision, marking it Implemented once the work
ships, or marking it Replaced when a later DEP supersedes it.

## Lifecycle

The enum is standardized to this set (the docs render layer implements it); the
DEP process around it is still under review in DEP-0001:

**Draft → Under Review → Accepted / Rejected / Deferred → Implemented → Replaced**

- **Draft** — shape under discussion. The PR may merge early so the DEP is
  discoverable and rendered; merging as Draft does not imply acceptance.
- **Under Review** — the author requests a decision; the owning SIG's approvers
  engage on the PR (line-level) and the tracking issue (design).
- **Accepted / Rejected / Deferred** — maintainers record the decision in a
  short follow-up PR that updates the status.
- **Implemented** — the work items have merged; the DEP names the shipping
  release.
- **Replaced** — a later DEP supersedes this one.

**Approvers** are the owning SIG's chairs or technical leads (the DEP's
`required-reviewers`); they speak for the SIG and decide when it is accepted.
Maintainers record the decision on the SIG's behalf.

## Status ↔ Banner Mapping

Keep the `status` field and the banner admonition in sync — the field carries
the full enum value, the banner carries the reader-facing callout. The docs
pipeline converts the GitHub-style admonition to a native callout
(`fern/convert_callouts.py`).

| Status | Banner admonition | Renders as |
|--------|-------------------|------------|
| Draft | `> [!WARNING]` | Warning |
| Under Review | `> [!IMPORTANT]` | Info |
| Accepted / Implemented | `> [!NOTE]` | Note |
| Rejected / Deferred / Replaced | `> [!CAUTION]` | Error |

## Workflow

### 1. Open the Follow-Up PR

```bash
cd /path/to/enhancements-checkout          # gh repo clone ai-dynamo/enhancements
git checkout main && git pull
git checkout -b dep-<slug>-status-<new-status>
# edit deps/NNNN-slug.md — see the two edits below
git add deps/NNNN-slug.md
git commit -s -m "docs: advance DEP-NNNN to <New Status>"
git push -u origin dep-<slug>-status-<new-status>
gh pr create --repo ai-dynamo/enhancements \
  --title "DEP-NNNN: advance to <New Status>" \
  --body "Records the <New Status> decision. Tracking issue: #<N_ISSUE>."
```

### 2. Make Both Edits in Sync

Update the status field **and** the banner in the same PR. For example, moving
from Draft to Under Review:

```md
**Status**: Under Review
```

```md
> [!IMPORTANT]
> **Status: Under Review.** The author has requested a decision; the owning SIG's approvers are reviewing.
```

When marking **Implemented**, name the shipping release in the DEP body. When
marking **Replaced**, link the superseding DEP (and its number) in the body.

### 3. Merge Under the Owning SIG's Approval

The follow-up PR merges once the owning SIG's approvers (the DEP's
`required-reviewers`) approve. For an Accepted / Rejected / Deferred decision, a
maintainer records it on the SIG's behalf. Keep GitHub the source of truth —
the decision lives in the merged PR, not on the docs page.

### 4. Confirm the Rendered Pills Update

No manual docs edit is needed. On the next Fern build,
`fern/scripts/sync_deps.py` re-reads the DEP's status and regenerates
`fern/js/dep-status-data.js` and `fern/js/dep-index-data.js`, so every surface
updates to match the new status from one source: the on-page `<DepMetadata>`
pill and its **lifecycle stepper**, the right-aligned Proposals-sidebar pill
(`fern/js/dep-status-pills.js`), and the DEP's card in the **registry index**
(`/proposals/registry`). The `/proposals/<slug>` URL is stable across status
changes. Getting a DEP rendered in the first place is the `dep-render` skill.

## Notes

- One status change per follow-up PR keeps the decision auditable.
- The banner and the `status` field must never disagree — a page that reads
  "Draft" in the field but shows a Note banner misleads readers about whether a
  proposal is ratified.
- DEP-0001 is a *draft* proposal, but the lifecycle enum itself is settled — the
  render layer implements exactly this set. Phrase advice as that model and
  point to `docs/proposals/0001-dep-process.mdx` for the full specification.
