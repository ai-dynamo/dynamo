---
name: dep-create
description: Authors a new Dynamo Enhancement Proposal (DEP) the SIG-owned way -- drafts the prose, then opens a tracking issue and a deps/NNNN-slug.md pull request in the ai-dynamo/enhancements repo (KEP-style sections, Draft status), and notes the optional Linear DYN-#### execution rollup. Use when proposing a cross-cutting architecture or process change, or filing a retroactive DEP.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - dep
    - enhancement-proposal
    - enhancements
    - sig
    - github
---

# Skill: Author a New DEP

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Author a new Dynamo Enhancement Proposal (DEP) under the current model:
DEPs are markdown files `deps/NNNN-slug.md` in the public
**`ai-dynamo/enhancements`** repository, owned by a SIG. Each is added by a
**pull request** (the revision, where reviewers leave line-level comments)
anchored by a **tracking issue** (the durable design discussion, open to any
GitHub user). The Dynamo docs site renders the DEP under the Proposals tab and
mirrors its GitHub review read-only.

This replaces the deprecated flow that filed DEPs as GitHub *issues* with
`dep:*` labels inside `ai-dynamo/dynamo`.

Ground everything in the two canonical files (read both before writing; do not
restate the whole model here):

- `docs/proposals/README.mdx` — the model and where each artifact lives.
- `docs/proposals/0001-dep-process.mdx` — the meta-DEP: lifecycle, roles,
  front-matter specification, and the honest caveats on still-contested points.

## When to Use

When the user wants to propose a new architecture, design, or process change,
especially a cross-cutting one that spans the Router, Planner, KVBM, Deploy,
the LLM engine, or OPS. Also when filing a retroactive DEP for work already
merged. If the proposal only touches one code area and does not change a public
API, a communication plane, or a multi-component contract, a DEP may be
overkill — confirm with the user first.

## Workflow

### 1. Draft the Prose

Write the proposal itself with the `write-proposal` approach (grounded,
concise, argument-first), then let `edit-docs` and `de-ai-text` polish it. The
DEP body is a real document, not a form to fill: lead with the recommendation
and the motivation, and answer the strongest objection head-on.

**Ask for source material.** Prompt the user for a Google Doc, Confluence page,
or other NVIDIA-internal document with background, customer context, or
detailed requirements. Read it (gdocs, Confluence MCP, WebFetch) and cite it in
the References section — that internal link is the record for context that
cannot appear in the public DEP.

**Customer name stripping.** Before opening anything, scan the summary,
motivation, proposal, and every other field for specific customer, company, or
partner names and replace them with generic references (e.g. "a customer", "a
cloud partner", "an enterprise user"). DEPs are public — no customer names in
the markdown, the PR, or the tracking issue.

### 2. Determine the Owning SIG / Area

A SIG owns a DEP, not a code repo, and SIGs map onto the existing CODEOWNERS
area taxonomy. Pick the owning area (and thus the owning SIG) from
`.github/codeowners/areas.yaml` in `ai-dynamo/dynamo` — for example `frontend`,
`router`, `kv-memory`, `planner`, `operator`, `backend-vllm`, `backend-trtllm`,
`backend-sglang`, `observability`, `fault-tolerance`, `runtime`, `ops`, `docs`,
`process`. Each area is backed by an `@ai-dynamo/dynamo-<area>-codeowners` team;
use `.github/codeowners/who_owns.py` to confirm who reviews a path. A
cross-cutting DEP names one owning SIG plus optional participating SIGs.

### 3. Open the Tracking Issue

The tracking issue is the DEP's stable anchor and its durable, open-ended
design-discussion thread. Open it in `ai-dynamo/enhancements`:

```bash
gh issue create --repo ai-dynamo/enhancements \
  --title "DEP: <short descriptive title>" \
  --body-file /tmp/dep-tracking-issue.md
# record the issue number -> N_ISSUE
```

### 4. Open the DEP PR

Add `deps/NNNN-slug.md` on a branch and open the PR. `NNNN` is the next free
number in `deps/` (list the directory first and coordinate with the SIG if
unsure); `slug` is a short kebab-case title.

```bash
gh repo clone ai-dynamo/enhancements /tmp/enhancements   # or reuse a checkout
cd /tmp/enhancements
git checkout -b dep-<slug>
# write deps/NNNN-slug.md — see the skeleton below
git add deps/NNNN-slug.md
git commit -s -m "docs: add DEP-NNNN <title> (draft)"
git push -u origin dep-<slug>
gh pr create --repo ai-dynamo/enhancements \
  --title "DEP-NNNN: <title>" \
  --body "Tracking issue: #<N_ISSUE>

<one-paragraph summary>"
```

**DEP file skeleton.** Follow the KEP-style section layout the enhancements
repo uses (its `NNNN-complete-template.md`), which is mirrored by
`docs/proposals/TEMPLATE.mdx`. Carry the DEP-0001 metadata keys — `number`,
`title`, `status`, `category`, `owning-sig`, `participating-sigs` (optional),
`authors`, `sponsor`, `required-reviewers`, `review-date`, `tracking-issue`,
`pr` — and open with a status banner that matches `status`. The enhancements
template writes each field as a `**Key**: Value` metadata block (this is what
`fern/scripts/sync_deps.py` parses into the rendered metadata card); the exact
serialization is settled by DEP-0001 and the enhancements template, so match
whatever the repo's current template uses rather than inventing a shape.

```md
# DEP-NNNN: <Title>

**Status**: Draft

**Category**: Architecture

**Owning SIG**: SIG-<Area>

**Authors**: [@handle](https://github.com/handle)

**Sponsor**: <owning SIG, or a maintainer shepherding on its behalf>

**Required Reviewers**: [@approver-one](https://github.com/approver-one)

**Review Date**: <target date>

**Tracking Issue**: https://github.com/ai-dynamo/enhancements/issues/<N_ISSUE>

**PR**: <this PR URL>

> [!WARNING]
> **Status: Draft.** This proposal is under active discussion and is not an accepted decision.

## Summary

<one paragraph>

## Motivation

<what problem, why now>

## Proposal

<the design; subsections as needed>

## Alternate Solutions

<options considered and rejected — or N/A>
```

New DEPs open as **Draft** with the `> [!WARNING]` banner. The full
status-to-banner mapping and the lifecycle are owned by the `dep-update` skill;
advancing the status is a separate follow-up PR, not part of authoring.

### 5. Note the Linear Execution Rollup (Optional)

For internal execution, a companion Linear issue (`DYN-####`) links
implementation PRs across the `ai-dynamo/*` code repos back to the DEP — the
cross-repo rollup GitHub's own same-repo issue linking cannot provide. This is
internal-only; the public design record stays the DEP doc plus its tracking
issue. Create the Linear issue only with explicit user approval.

### 6. Report Back

Report the **DEP PR** URL, the **tracking issue** URL, the optional
**`DYN-####`**, and — once `fern/scripts/sync_deps.py` picks it up on the next
docs build — the rendered **Proposals page** at
`docs.nvidia.com/dynamo/.../proposals/<slug>`. Getting the DEP rendered on the
docs site is the `dep-render` skill.

## Notes

- The DEP markdown in `ai-dynamo/enhancements` is the source of truth; the
  docs page is a read-only mirror. Every reply happens on GitHub.
- Reviewers leave line-level comments on the PR (the revision) and open-ended
  design discussion on the tracking issue. Both surface on the rendered page.
- DEP-0001 is itself a *draft* proposal. Where a specific (exact lifecycle
  state names, the final front-matter set) is still under review, defer to
  `docs/proposals/0001-dep-process.mdx` rather than hard-asserting it.
- Retroactive DEP: same flow, but open it already reflecting the shipped state
  (reference the merged PRs, and set the status the work is actually in).
