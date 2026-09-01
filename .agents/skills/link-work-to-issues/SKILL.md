---
name: link-work-to-issues
description: Ensures every piece of work in ai-dynamo/dynamo starts from a tracked issue and every pull request references one. Use when starting new work, opening or editing a pull request, or when the "PR Issue Link" check fails and needs remediation.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - linear
    - workflow
---

# Skill: Link Work to Issues

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Every pull request in this repository references the issue it implements, so
work is traceable to a tracked task and release tracking stays live without
manual reconciliation. The `PR Issue Link` CI check enforces this: advisory
today, required from the date named in its failure message.

## When to Use

- Starting any new piece of work in this repository.
- Opening a pull request, or editing one whose `PR Issue Link` check failed.
- Deciding how to reference a DEP or another long-running tracking issue.

## Workflow

### Start From an Issue

Before writing code, find or create the issue for the work and start from it.
The issue carries the context, the discussion history, and the record of why
the change exists; an agent pointed at the issue first inherits all of it.

- **Community contributors**: create or claim a GitHub issue on
  `ai-dynamo/dynamo` (`gh issue create`). Contributions are expected to have
  one.
- **NVIDIA engineers**: start from the Linear issue. Creating the issue in the
  DGH Linear team mirrors it to GitHub automatically, and GitHub issues mirror
  back into DGH, so either side works.
- Bug found mid-task that needs its own fix: create an issue for it rather
  than folding untracked work into the current PR.

### Name the Branch

When a Linear issue exists, include its ID in the branch name
(`user/dyn-1234-short-description`). Linear's GitHub integration links the PR
from the branch name alone, and the CI check accepts it.

### Reference the Issue in the Pull Request

Put at least one of these in the PR title or description:

| Situation | Reference form |
|---|---|
| This PR completes the issue | `Fixes #123` or `Closes DYN-1234` |
| The issue outlives this PR (tracking issue, DEP) | `Part of #123` or `refs DYN-1234` |
| The issue lives in a sibling org repository | `ai-dynamo/<repo>#12` or the full issue URL |

Closing keywords close the issue on merge; non-closing forms link without
closing, which is what a DEP tracking issue or a multi-PR feature issue needs.
The reference must point at a real issue: the check verifies existence and
does not count pull-request numbers.

### Remediate a Failing Check

The `PR Issue Link` check failing means no verifiable reference was found.
Add one using the table above and edit the PR description; editing re-triggers
the check. If no issue exists yet, create one first (see Start From an Issue).
While the check is advisory it does not block merges; its failure message
names the date it becomes required.

## Related

- `dep-create` for proposing architecture changes as DEP issues; reference a
  DEP from implementing PRs with a non-closing form (`Part of #<dep>`).
- The check's implementation lives at
  `.github/workflows/pr-issue-link.yml` and `pr_issue_link.py`.
