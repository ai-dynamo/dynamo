---
name: dynamo-issue-first
description: Ensures every piece of work in ai-dynamo/dynamo starts from a tracked issue and every pull request references one. Use when starting new work, creating or sorting its Linear issue (release, feature project, milestone), opening or editing a pull request, or when the "PR Issue Link" check fails and needs remediation.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - linear
    - workflow
---

# Skill: Issue-First Development

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

## Purpose

Every pull request in this repository references the issue it implements, so
work is traceable to a tracked task and release tracking stays live without
manual reconciliation. The `PR Issue Link` CI check enforces this on non-draft
PRs into `main`: advisory today, required once the release branch named in its
message is cut.

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
- **Release features**: each feature in a release has a tracking ticket in
  Linear with its work issues linked as Related, so that ticket is where to
  look up which feature a work issue belongs to.

### Sort the Issue (NVIDIA Engineers)

When you create or update the Linear issue for your work, set its release and
feature project yourself. You have the context; nobody sorts it after you.

- **Content.** The Linear issue holds what the public PR must not: customer
  names, NVBug ids and repro steps, internal threads, and roadmap context. Keep
  that detail in an issue in your own team or DYN, never in DGH, which mirrors
  to public GitHub. The PR references the issue and describes only the
  technical change.
- **Release.** Look up the Linear "Dynamo" release pipeline every time (Linear
  MCP `list_releases`, archived off) and take the release in "In Development".
  Never hardcode it or derive it from the version on `main`, which lags after a
  branch cut. Two exceptions: a QA bug against the release in QA takes that
  release, and a cherry-pick takes each release it ships in, patch releases
  included (see `dynamo-release-cherry-pick`). Never add a patch release
  otherwise.
- **Project.** Feature projects carry the newest label in the DYNO "Dynamo
  Roadmap" project label group and are named `<window>-AREA-NN Title`. Pick the
  one in the current window whose scope matches the work. If none clearly fits,
  or the window has no projects yet, leave it empty; never guess or create one.
  Add a milestone only when the project has one named exactly like the release.
- **Write.** Use the Linear MCP `save_issue` with `project` and `addReleases`
  (pass `id` for an existing issue), or `issueUpdate` with `addedReleaseIds`.
  Never `setReleases` or a replacing `labels` / `labelIds`, which drop what
  others added. Read the issue back to confirm the project and release stuck.
- The project is the feature: do not mark it with labels or sub-issues, and ask
  before moving an issue out of a project someone else set.

### Name the Branch

When a Linear issue exists, include its ID in the branch name
(`user/dyn-1234-short-description`). Linear's GitHub integration links the PR
from the branch name alone, and the CI check accepts it. Linear treats that
link as closing, so leave the ID out of the branch name for a QA bug (below).

### Reference the Issue in the Pull Request

Put at least one of these in the PR title or description:

| Situation | Reference form |
|---|---|
| This PR completes the issue | `Fixes #123` or `Closes DYN-1234` |
| The issue outlives this PR (multi-PR feature, tracking issue) | `Part of #123` or `refs DYN-1234` |
| The issue is a QA bug (a `qa:*` label) | `refs DYN-1234` or `Part of DYN-1234` in the description only, never a closing keyword, and the key out of the title |
| The work implements a DEP | `Part of #<dep>` **and** a work issue of its own |
| The issue lives in a sibling org repository | `ai-dynamo/<repo>#12` or the full issue URL |

Closing keywords close the issue on merge; non-closing forms link without
closing, which is what a multi-PR feature issue needs. A QA bug must stay open
until QA verifies the fix on a release candidate: closing it on the `main`
merge closes the synced NVBug too early. `dynamo-release-cherry-pick` covers the
release side. The reference must point at a real issue: the check verifies existence and does not count
pull-request numbers.

A DEP does not satisfy the check on its own. It is a proposal umbrella that
stays open across many pull requests, so it records what the work is part of
and not what one change is. Keep the `Part of #<dep>` reference and file a
work issue for the change as well.

### Remediate a Failing Check

The `PR Issue Link` check failing means no verifiable reference was found.
Add one using the table above and edit the PR description; editing re-triggers
the check. If no issue exists yet, create one first (see Start From an Issue).
Until the next minor release branch is cut the check passes with a warning;
its message names the branch that switches it on.

## Related

- `dep-create` for proposing architecture changes as DEP issues. An
  implementing PR carries `Part of #<dep>` alongside its own work issue; the
  DEP reference alone fails the check.
- The check's implementation lives at
  `.github/workflows/pr-issue-link.yml` and `pr_issue_link.py`.
