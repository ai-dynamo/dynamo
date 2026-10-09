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
issue or bypassing release ownership.

## Establish the Release Request

Identify the target `release/X.Y.Z` branch, the merged main PR and commit, and
the matching Linear issue before changing the release branch.

- The normal path starts with a fix merged to `main`. Use a release-only fix
  only for changes that cannot sensibly land on `main`, such as a release
  version bump or code that `main` has already replaced.
- If the fix is not merged, stop release-branch work and complete the main PR
  first. The main branch and main PR may carry the issue ID and a closing
  reference such as `Fixes DYN-123`; those forms belong to the main PR only.
- For NVIDIA work, use the existing DYN issue as the request record. Do not
  create another issue without the user's authorization.
- Set the issue's Releases field to the target release and request the
  `cherry-pick:requested` label. QA bugs may already have a synced request.
- A release approver owns the decision. Proceed only after they replace the
  request label with `cherry-pick:approved` and record approval in a comment.
  A release-only fix instead receives `cherry-pick:release-only`.
- Do not self-approve, infer approval from urgency, or proceed after a declined
  request. Report the recorded reason when a request is declined.

Updating Linear, pushing a branch, opening or editing a PR, and messaging
release support are external actions. Perform only the actions the user has
authorized.

## Prepare the Release Branch

Use a clean worktree based on the current target release branch. Keep the
Linear issue ID out of both the release branch name and the PR title because
the Linear integration can otherwise close the issue before QA verifies it.
A suitable branch shape is `<user>/cherrypick-<short-name>`.

For a normal cherry-pick:

```bash
git fetch origin release/X.Y.Z
git worktree add <path> -b <user>/cherrypick-<short-name> origin/release/X.Y.Z
cd <path>
git cherry-pick -x --signoff <main-merge-commit>
```

Use the exact commit that merged the main PR. Preserve the `-x` provenance and
DCO sign-off. Resolve conflicts from source files; regenerate generated files
instead of hand-editing them. Never skip or weaken tests to make the backport
pass.

For an approved release-only fix, branch from `release/X.Y.Z`, make the
smallest release-specific change, and create a signed-off commit. There is no
main commit to cherry-pick. The Linear comment must explain why the change
cannot go through `main`.

## Validate and Open the Release PR

Run validation proportional to the affected code and the release risk. Follow
the scoped `AGENTS.md` instructions and preserve long-running command output
for diagnosis.

Open the PR against the target release branch. Before creating it, verify all
of these invariants:

- The branch name and PR title contain no issue ID.
- The description contains `Refs DYN-123`, never `Fixes`, `Closes`, or another
  closing form for the Linear issue.
- The description is self-contained, with `Summary` and `Validation`, and
  links the merged main PR or explains the release-only exception.
- Any ordering or PR dependencies are stated explicitly.
- The cherry-picked commit retains its source SHA annotation and sign-off.

Treat the release PR like any other PR: obtain review, diagnose its failures,
and keep release-branch CI green. Use the `pr-monitor` skill when CI needs
investigation.

## Own the Result Through the RC

During the QA period, the next-RC cutoff is 3 PM Pacific each day. The release
PR must be approved and passing release-branch CI before that cutoff. Escalate
late or urgent fixes to release support when the user authorizes the message.

Do not classify a failure as an ignorable flake on your own. Investigate it,
then ask the release PiC; only the release PiC decides whether a failing PR may
merge. The release PiC also merges the PR and replaces
`cherry-pick:approved` with the `cherry-pick:rcN` label for the RC containing
the fix.
