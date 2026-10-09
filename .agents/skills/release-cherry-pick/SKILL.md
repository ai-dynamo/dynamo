---
name: release-cherry-pick
description: Prepares and follows through Dynamo release-branch cherry-picks after a fix has merged to main, including request and approval gates, signed-off cherry-picks, release PR metadata, and release CI ownership. Use when asked to backport, cherry-pick, or pick to release a Dynamo change on a release/X.Y.Z branch. Do not use for ordinary main-branch fixes.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - git
    - github
    - linear
    - release
---

# Dynamo Release Cherry-Picks

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Backport an approved Dynamo change without prematurely closing its tracking
issue. Cherry-pick requests live in Linear; do not use the retired release
canvas table.

## Establish the Release Request

Identify the target `release/X.Y.Z` branch, the merged main PR and commit, and
the matching Linear issue before changing the release branch.

- The normal path starts with a fix merged to `main`. Use a release-only fix
  only for changes that cannot sensibly land on `main`, such as a release
  version bump or code that `main` has already replaced.
- If the fix is not merged, stop release-branch work and complete the main PR
  first using the `issue-first` skill. A closing reference such as
  `Fixes DYN-123` belongs to the main PR only.
- For an internal fix, use a DYN issue. Set its Releases field to the target
  release and add `cherry-pick:requested`. QA bugs may already have this label
  from NVBugs synchronization.
- A release approver owns the decision. Proceed only after they replace the
  request label with `cherry-pick:approved` and record approval in a comment.
  A release-only fix instead receives `cherry-pick:release-only`.
- A declined request is removed from the release and has the reason recorded
  in a Linear comment. Do not create a release PR for it.

## Prepare the Release Branch

Start from the current target release branch using the local Git strategy
appropriate to the task. Keep the Linear issue ID out of both the release
branch name and the PR title because the Linear integration can otherwise
close the issue before QA verifies it. A suitable branch shape is
`<user>/cherrypick-<short-name>`.

For a normal cherry-pick:

```bash
git cherry-pick -x --signoff <main-merge-commit>
```

Use the exact commit that merged the main PR and preserve the `-x` provenance
and sign-off.

For an approved release-only fix, branch from `release/X.Y.Z`, make the
release-specific change, and create a commit with `--signoff`. There is no main
commit to cherry-pick. The Linear comment must explain why the change cannot
go through `main`.

## Validate and Open the Release PR

Open the PR against the target release branch. Before creating it, verify all
of these invariants:

- The branch name and PR title contain no issue ID.
- The description contains `Refs DYN-123`, never `Fixes`, `Closes`, or another
  closing form for the Linear issue.
- Any ordering or PR dependencies are stated explicitly.
- The cherry-picked commit retains its source SHA annotation and sign-off.

## Own the Result Through the RC

During the QA period, the next-RC cutoff is 3 PM PT each day. The release PR
must be approved and passing release-branch CI before that cutoff. For a late
or urgent fix, ping `@release-support`.

The PR author owns CI failures until the PR merges. If a test fails, check with
the release PiC whether it is flaky; only the release PiC decides whether a
failing PR may merge. The release PiC also merges the PR and replaces
`cherry-pick:approved` with the `cherry-pick:rcN` label for the RC containing
the fix.

## Approval Criteria

Requests are more likely to be approved when they are P0 for the release,
small or mostly documentation, early in QA, or required by a customer. They
are less likely to be approved when they are not P0, have broad or breaking
scope, arrive late in QA after RC3, or lack concrete customer benefit.

Changes with a `feat` PR type require extra approval.

## Track the Request

Use the target release's Linear cherry-pick view to follow every request and
its status. Add the `Assignee is me` filter to narrow the view to your own
requests.
