---
name: dep-status
description: Checks Dynamo Enhancement Proposal status and finds DEPs under the current model -- DEPs are markdown files in ai-dynamo/enhancements with a status field plus a tracking issue, surfaced on the docs-site Proposals tab. Lists open DEP PRs and deps/*.md files (parsing their status), lists tracking issues, and points at the Linear DYN-#### execution rollup. Use when asking where a DEP stands or which DEPs exist for an area.
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

# Skill: Check DEP Status / Find DEPs

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Report where a DEP stands and find the DEPs for an area. Under the current
model a DEP is a markdown file `deps/NNNN-slug.md` in **`ai-dynamo/enhancements`**
whose `status` field carries its lifecycle state, anchored by a tracking issue
and rendered on the Dynamo docs site's Proposals tab. This replaces the
deprecated flow that read DEP *issues* by `dep:*` labels in `ai-dynamo/dynamo`.

See `docs/proposals/README.mdx` and `docs/proposals/0001-dep-process.mdx` for
the model — do not restate it here.

## When to Use

When the user wants the status of one or more DEPs, wants to see what is under
review, wants the DEPs related to a component/area, or wants a triage summary.

## Workflow

### 1. List Open DEP PRs

The revision of each in-flight DEP is an open PR against `deps/`:

```bash
gh pr list --repo ai-dynamo/enhancements --state open \
  --json number,title,headRefName,labels,author,updatedAt
```

### 2. List DEP Markdown Files and Parse Status

The `deps/` directory holds every accepted/landed DEP; each file's `status`
field is the source of truth for its lifecycle state:

```bash
# List DEP markdown files
gh api repos/ai-dynamo/enhancements/contents/deps \
  --jq '.[] | select(.name | endswith(".md")) | .name'

# Read one DEP's status (the enhancements template uses a **Status**: block;
# a YAML `status:` front-matter is also matched as a fallback)
gh api repos/ai-dynamo/enhancements/contents/deps/NNNN-slug.md \
  --jq '.content' | base64 -d | grep -iE '^\*\*Status\*\*:|^status:'
```

The lifecycle states are Draft → Under Review → Accepted / Rejected / Deferred
→ Implemented → Replaced (proposed in DEP-0001; treat exact names as the model
the docs render layer currently implements, and defer to
`docs/proposals/0001-dep-process.mdx` where still under review).

### 3. List Tracking Issues

Each DEP has a tracking issue in the same repo hosting its design discussion:

```bash
gh issue list --repo ai-dynamo/enhancements --state all \
  --search 'DEP in:title' \
  --json number,title,state,labels,updatedAt
```

### 4. Note the Linear Execution Rollup

For internal execution status, each DEP may have a companion Linear issue
(`DYN-####`) that links implementation PRs across the `ai-dynamo/*` code repos.
It is internal-only — surface it for NVIDIANs, but the public status is the
DEP's `status` field and its tracking issue.

### 5. Point at the Rendered Page

Every DEP renders at a stable `docs.nvidia.com/dynamo/.../proposals/<slug>` URL
with a status pill on the metadata card and a matching sidebar pill. That page
is the shareable public link for a DEP's current state and its mirrored
discussion.

### 6. Format as a Summary Table

```text
| DEP | Title | Status | Owning SIG / Area | Tracking issue | PR |
|-----|-------|--------|-------------------|----------------|----|
| 0001 | The DEP Process | Draft | SIG-Process / process | ai-dynamo/enhancements#123 | ai-dynamo/enhancements#124 |
```

Pull each cell from the real DEP markdown and its GitHub objects; never
hand-wave a status. If a value is unknown, write `?` and say why.

## Notes

- For a full triage view, combine open DEP PRs (in flight) with the `deps/*.md`
  files (landed) so both in-review and merged DEPs appear.
- Owning SIG / area comes from the DEP's `owning-sig` field; it maps onto the
  CODEOWNERS area taxonomy (`.github/codeowners/areas.yaml`).
- Unauthenticated `gh api` calls share GitHub's per-IP rate limit; authenticate
  `gh` (it uses your token) to avoid 403s on larger sweeps.
