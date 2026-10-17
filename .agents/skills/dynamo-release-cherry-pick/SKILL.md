---
name: dynamo-release-cherry-pick
description: Describes how a fix reaches a Dynamo release branch after code freeze - cherry-picking a merged main fix into release/X.Y.Z, landing a release-only fix, or sponsoring an external contributor's Cherry-Pick Request - with the cherry-pick Linear labels, the Releases field, non-closing issue links, and the daily RC cutoff. Use when backporting a fix to a release branch, opening a PR against release/X.Y.Z, filing or sponsoring a Cherry-Pick Request, or asking how the cherry-pick labels work.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - release
    - cherry-pick
    - linear
    - workflow
---

# Skill: Release Cherry-Pick

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

After code freeze, `release/X.Y.Z` takes changes only through pull requests
tracked on a Linear issue. Every fix lands on `main` first unless it is a
release-only fix. This skill is guidance: do not add labels, edit Linear,
comment, or open pull requests unless the user asks.

## Labels and Fields

| Stage | Label | Set by |
|---|---|---|
| Request | `cherry-pick:requested` | Engineer, or the sponsor for an external request |
| Decision | `cherry-pick:approved` or `cherry-pick:rejected` | Release approver |
| Landed | `cherry-pick:rcN` | Release PiC, at merge |

The release approver (the release TPM) and the release PiC (person in charge
of merges) are named in the branch-cut announcement for each release.

- The stage labels form one exclusive group, so an issue holds one at a time.
- `cherry-pick:release-only` is a separate tag outside the group. The release
  approver adds it beside `cherry-pick:approved` for a release-only change,
  and it stays on through `cherry-pick:rcN`.
- A pick merged while RCn is the current release candidate gets
  `cherry-pick:rc(n+1)`, the build it first ships in.
- Add the target release to the issue's **Releases** field (for example
  `Dynamo v1.6.0`) without removing releases already there, and read the
  issue back to confirm it stuck. Without it the pick does not appear on that release's
  cherry-pick board.
- `cherry-pick:rcN` is single-select, so an issue records one landing RC. An
  issue may carry several release PRs only when they all land in the same RC.
  See Edge Cases for the split rules.
- The issue can live in any Linear team that carries the cherry-pick labels.
  A team without them uses DYN.

## Linking: Never Close the Issue Early

For a QA bug (an issue with a `qa:*` label), both the main PR and the release
PR reference the issue with a non-closing word in the PR body: `Refs DYN-123`
or `Part of DYN-123`. Never `Fixes`, `Closes`, or `Resolves`. A closing link
moves the issue to Done when the main PR merges, and the NVBugs sync then
closes the NVBug before the fix reaches the release and QA verifies it. Linear
also treats an issue key in a branch name or PR title as a closing link, so
keep the key out of both for these PRs.

For an internal issue that is not a QA bug, a closing word on the main PR is
fine. The release PR always uses `Refs`, with the key out of its branch name
and title.

## Before You Start

Check by content that the fix is not already on the release branch. Picks
are often adapted or folded into another PR, so commit ancestry, `git cherry`,
and patch-id can report a present fix as missing. Search for a changed line:

```bash
git fetch origin
git grep -n '<changed line>' origin/release/X.Y.Z -- <path>
```

## Internal Engineers

1. **Merge the fix to `main`** first, linked as described above.
2. **Open the release PR right away.** Do not wait for approval.

   ```bash
   git switch -c <user>/cherry-pick-<desc> origin/release/X.Y.Z
   git cherry-pick -x --signoff <main-merge-sha>   # add -m 1 for a merge commit
   git push -u origin HEAD
   gh pr create --base release/X.Y.Z \
     --title "<type>(<scope>): [cherry-pick X.Y.Z] <desc> (#<main-PR>)" \
     --body "Refs DYN-123"
   ```

   `-x` records the source commit and `--signoff` keeps DCO. On a conflict,
   resolve, `git add`, and `git cherry-pick --continue`. Never force-push to
   `release/*`.
3. **Request.** Once the release PR shows on the Linear issue, add
   `cherry-pick:requested` and set the Releases field.
4. **Get release-branch CI green** before the daily 3:00 PM PT RC cutoff.
5. **Approval and merge.** The release approver approves in Linear. The
   release PiC merges the release PR and sets `cherry-pick:rcN`. QA verifies
   the fix on that RC.

`feat` changes need extra approval. Talk to the TPM before requesting one.

## Release-Only Fixes

A release-only fix never goes to `main`: a release version bump, a fix to
code `main` has already rewritten, or a temporary workaround `main` solves
another way. Branch from `origin/release/X.Y.Z`, commit with `--signoff`, and
open the PR against `release/X.Y.Z` with `Refs <issue key>` and no
`(#<main-PR>)` suffix. Add `cherry-pick:requested`, set the Releases field,
and explain in a Linear comment why the change cannot go to `main`. The
approver adds `cherry-pick:release-only` at approval.

## External Contributors

Contributors outside the `ai-dynamo` org work only on GitHub:

1. Merge the fix to `main` through a normal PR (skip for a release-only fix).
2. File the **Cherry-Pick Request** issue
   (`.github/ISSUE_TEMPLATE/cherry_pick_request.yml`). Title it
   `[Cherry-Pick Request] [release/X.Y.Z]: <desc>` with the real version. The
   template applies `cherry-pick:requested` and asks for the merged main PR,
   target release, type, reason, and dependencies.

The contributor never opens the release PR. The issue mirrors to a DGH
Linear issue automatically; labels sync by name, the Releases field does not.

A **sponsor** owns everything after that. The sponsor is the maintainer who
merged the main PR or, for a release-only fix, the maintainer the release
approver assigns. The sponsor adds the release to the DGH issue's Releases
field, opens the release PR with `Refs DGH-NNN` in its body so Linear attaches
it, and keeps its CI green, following the internal steps above.

## Edge Cases

| Situation | What to do |
|---|---|
| A pick is reverted | `cherry-pick:rcN` no longer holds. Return to `cherry-pick:requested` and open a fresh release PR; the re-land gets a new `rcN`. |
| Several main PRs, one release PR | Attach the shared release PR to every issue and list each main PR in its title or body. |
| Same fix, two release lines, same RC | Add each release to the Releases field and open one release PR per branch, all on one issue. One approval covers them. |
| Same fix, two release lines, different RCs | Use one issue per release line, so each carries its own Releases entry and `rcN`. |
| Two release PRs on one branch, same RC | Attach both to the issue (a pick split in two, or a pick plus its prerequisite). Merge both before setting `rcN`. |
| A second release PR after the first landed | Open a new issue for the follow-up, linked to the first. The landed issue keeps its `rcN`. |

## Related

- `dynamo-issue-first` for linking work to issues on `main`. For QA bugs, the
  non-closing rule above overrides both its `Fixes` form and its convention
  of putting the issue key in the branch name.
