---
name: reviewers
description: Finds active, eligible GitHub approvers for a pull request or local diff using CODEOWNERS, submitted reviews, current team membership, recent review activity, changed-code blame, and component history. Use when the user asks who should review or approve a PR, invokes /reviewers or $reviewers, or wants a reviewer plan directly in Codex. Returns grouped GitHub reviewers and scoped path summaries without using Slack, sending messages, or persisting state.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - github
    - pull-request
    - reviewers
---

# GitHub PR Reviewers

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Build the canonical reviewer plan for an `ai-dynamo/dynamo` pull request and render
it directly in Codex. Own reviewer scope, coverage, ranking, and path summaries here.
Slack-specific workflows must reuse this plan instead of reimplementing selection.

## Input

Accept a GitHub PR URL, a PR number inside the matching clone, or no argument:

1. **Explicit PR:** Use the supplied URL or number.
2. **Current branch:** Run `gh pr view` when no PR was supplied.
3. **Local diff:** If the branch has no PR, compare `HEAD` with the merge base of
   the repository's default branch. Local diffs have no submitted-review coverage.

Require a local clone because CODEOWNERS matching and old-side blame use repository
objects. Report the resolved mode before the reviewer plan.

Default to two candidates per owner group. Use three only when the third candidate
has meaningful evidence. Honor `--per-team N` or `-n N` when supplied.

## Output contract

Write human-readable output in English. Render only uncovered owner groups. Use
GitHub team names and logins, never Slack handles or IDs:

```text
@ai-dynamo/dynamo-ops-codeowners — @nv-tusharma @dillon-cullinan
- `.github/{actions,workflows}/` — Adds DGDR deploy-test CI wiring.
- `.github/filters.yaml` — Routes DGDR changes into deploy CI.
```

For each group:

- Put the exact CODEOWNERS team first, followed by two or three ranked candidates.
- Add one or two scoped path bullets; never exceed three.
- Summarize at a component or parent-directory level. Use brace notation when useful.
- Describe the implementation in 4–8 words, not a concern or desired verification.
- Repeat a candidate under every owner group they can cover; do not label them as
  cross-group.

If all required groups are covered, say no additional reviewer request is needed. If
a group has no eligible active candidate, explain the missing signal instead of
inventing a reviewer.

Do not send messages, resolve Slack identities, emit Slack mention syntax, or persist
selection state.

## Canonical reviewer plan

Keep this structure internally so wrapper skills can consume it:

```json
{
  "repo": "ai-dynamo/dynamo",
  "pr": 11946,
  "url": "https://github.com/ai-dynamo/dynamo/pull/11946",
  "author": "sttts",
  "scope": [
    {"path": ".github/workflows/", "owners": ["@ai-dynamo/dynamo-ops-codeowners"]}
  ],
  "groups": [
    {
      "owner": "@ai-dynamo/dynamo-ops-codeowners",
      "covered": false,
      "reviewers": ["nv-tusharma", "dillon-cullinan"],
      "paths": [
        {
          "path": ".github/{actions,workflows}/",
          "summary": "Adds DGDR deploy-test CI wiring."
        }
      ]
    }
  ]
}
```

`scope` contains every sorted changed `{path, owners}` record, including covered
groups. `groups` contains every required owner group with current coverage. Populate
`reviewers` only for uncovered groups.

## Workflow

### 1. Resolve PR metadata

For a PR, fetch:

```bash
gh pr view <N> --repo <owner/repo> \
  --json number,title,url,state,author,baseRefName,baseRefOid,headRefOid,files,reviews
gh api repos/<owner>/<repo>/pulls/<N>/files --paginate
```

Reject a closed or merged PR unless the user requests historical analysis. Fetch the
base branch when `baseRefOid` is missing locally.

For a local diff, synthesize equivalent metadata from the merge base, current `HEAD`,
`git diff`, and the authenticated GitHub user.

### 2. Resolve required owners

Read `CODEOWNERS` at `baseRefOid`; fall back to the working tree only when the object
cannot be read. Apply GitHub semantics exactly:

- last matching pattern wins;
- leading `/`, trailing `/`, `*`, and `**` retain CODEOWNERS behavior;
- an empty owner list explicitly clears ownership;
- co-owned paths require one group per owner team.

Map every changed path to its final owner set and keep sorted `{path, owners}` records.

### 3. Resolve coverage

Fetch current membership for every required GitHub owner team. Count human submitted
reviews in states `APPROVED`, `CHANGES_REQUESTED`, and `COMMENTED`. Ignore `PENDING`,
`DISMISSED`, bots, and ordinary issue comments.

Treat a group as covered when any counted reviewer is a current member of that exact
team. The same review may cover multiple groups. Keep covered groups in the internal
plan but omit them from direct output.

### 4. Select eligible candidates

For each uncovered group:

1. Start with current members of the exact GitHub owner team.
2. Exclude the PR author, bots, departed members, and everyone who already submitted
   a counted review on this PR.
3. Require at least one `ai-dynamo/dynamo` pull-request review contribution in the
   last 90 days. Prefer GraphQL `pullRequestReviewContributions`; use general activity
   only when contribution data is unavailable.
4. Rank remaining candidates by:
   - old-side blame counts on modified source hunks at `baseRefOid`;
   - recent reviews of PRs touching the same paths or component;
   - eligibility across other still-uncovered owner groups;
   - outstanding review-request load only as a tie-breaker.

Skip pure additions for blame. Downweight generated files and lockfiles unless that
owner group explicitly owns them.

When blame has no useful signal, use recent component review history, then recent
commits under the changed directories. Never select solely because someone belongs
to many teams.

### 5. Build path summaries

Group changed files by meaningful component or parent directory for each owner team.
Prefer the smallest set that explains why the group is required. Generate terse,
implementation-focused summaries from the patch.

### 6. Render

Report the resolved mode, then render uncovered groups using the output contract.
Mention covered groups once in a short trailing note. Do not append a flat mention
list.

## Edge cases

- If no changed path has an owner, report that no CODEOWNERS group matched.
- If a team lookup returns 404 or permission denied, report that group as unresolved.
- If recent review-contribution data is unavailable, state which fallback was used.
