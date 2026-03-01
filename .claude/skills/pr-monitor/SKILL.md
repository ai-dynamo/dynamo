---
name: pr-monitor
description: Check CI status, analyze failures, and explain skips for a Dynamo PR
user-invocable: true
disable-model-invocation: true
---

# PR CI Monitor

Perform a full health check on a Dynamo pull request. Takes a PR number as argument (e.g., `/dynamo:pr-monitor 6554`).

## Step 1: PR Overview

Gather PR metadata and determine what CI should look like for this PR.

```bash
gh pr view $PR_NUMBER --repo ai-dynamo/dynamo --json title,body,author,state,isDraft,additions,deletions,changedFiles,labels,reviewDecision,headRefName,baseRefName

gh pr diff $PR_NUMBER --repo ai-dynamo/dynamo --name-only
```

**Check if full CI should be running.** The dynamo repo has two tiers of CI:

- **Lightweight (`pre-merge.yml`)**: Triggers on all `pull_request` events. Runs pre-commit, copyright checks, DCO, and optionally rust-clippy/rust-tests (if rust filter matches). This always runs.
- **Full pipeline (`pr.yaml`, `container-validation-dynamo.yml`)**: Triggers on `push` to `main` or `pull-request/[0-9]+` branches ONLY. This includes docker builds, GPU tests, deploy tests. It does NOT trigger on regular PR branches.

To get full CI on a PR, a `pull-request/$PR_NUMBER` branch must exist (created by copy-pr-bot after an NVIDIA maintainer approves). Check:

```bash
gh api repos/ai-dynamo/dynamo/branches/pull-request/$PR_NUMBER 2>/dev/null
```

If the branch does not exist, full CI has not been triggered. Common reasons:

1. **Awaiting approval**: An NVIDIA maintainer needs to comment `/ok to test <commit_sha>` on the PR to create the branch and trigger full CI. This applies to both fork PRs and internal PRs from authors not yet in the approval list.
2. **DCO failure**: Commits are not signed (`Signed-off-by` line missing). Check for a DCO bot comment. Fix: author signs their commits (see `DCO.md`).
3. **Draft PR**: Some checks may not run until marked ready for review.

**Verify which workflows actually ran** against the PR's HEAD commit:

```bash
# Get the HEAD SHA
HEAD_SHA=$(gh pr view $PR_NUMBER --repo ai-dynamo/dynamo --json headRefOid --jq '.headRefOid')

# Check which workflow runs exist for this SHA
gh api "repos/ai-dynamo/dynamo/actions/runs?head_sha=$HEAD_SHA" --jq '.workflow_runs[] | {name: .name, status: .status, conclusion: .conclusion}'
```

Compare the workflows that ran against what's expected. If only `Pre Merge`, `Copyright Checks`, `DCO Commenter`, etc. appear but NOT `PR` or `Dynamo Validation`, then full CI was not triggered.

**Determine which CI filters are active.** If full CI ran, read the `changed-files` job output from the actual workflow run — this is the authoritative source for which filters are `true`:

```bash
# Find the PR workflow run ID
PR_RUN_ID=$(gh api "repos/ai-dynamo/dynamo/actions/runs?head_sha=$HEAD_SHA" --jq '.workflow_runs[] | select(.name == "PR") | .id')

# Get the changed-files job output (look for the filter results in the logs)
gh api "repos/ai-dynamo/dynamo/actions/runs/$PR_RUN_ID/jobs" --jq '.jobs[] | select(.name == "changed-files") | {id: .id, status: .status, conclusion: .conclusion}'
```

If full CI hasn't run, fall back to fetching `filters.yaml` and matching manually:

```bash
gh api repos/ai-dynamo/dynamo/contents/.github/filters.yaml --jq '.content' | base64 -d
```

Key rules for manual matching:
- `core` being true triggers ALL framework pipelines (vllm, sglang, trtllm).
- Filters use YAML anchors (e.g., `*ci`) — resolve these when reading.
- Negation patterns like `!**/*.md` in a filter mean markdown-only changes do NOT trigger that filter.

**Jobs that always run in `pr.yaml` regardless of filters:**
- `changed-files`, `deploy-operator`, `backend-status-check`, `clean-k8s-builder`, `cleanup` — these run unconditionally or with `if: always()`.

## Step 2: CI Status Dashboard

Fetch all check runs. Note: `gh pr checks` returns non-zero exit codes when checks are pending or failed — this is normal, not an error.

```bash
gh pr checks $PR_NUMBER --repo ai-dynamo/dynamo
```

If no checks are returned at all, refer back to the Step 1 diagnosis (DCO, approval, draft status).

**Ignore external CI checks** (e.g., GitLab mirror `ci/gitlab/*`). These are NVIDIA-internal pipelines that cannot be inspected from GitHub. Only analyze GitHub Actions checks.

**Distinguish two situations:**
1. **Full CI triggered** (workflow runs include `PR`, `Dynamo Validation`): Analyze all jobs normally.
2. **Only lightweight CI ran** (`Pre Merge` and utility workflows only): Report this clearly. The filter predictions from Step 1 describe what *would* run once full CI triggers, but there's nothing to analyze yet beyond the lightweight checks.

If full CI ran, **identify the critical path** — which checks are most relevant to this PR's changes based on the filter mapping from Step 1.

Produce a concise dashboard grouped by status:
- **Failed** — needs immediate attention
- **Pending/In-progress** — still running (note which are critical path)
- **Passed** — healthy (just count, don't enumerate unless asked)
- **Skipped** — handled in Step 4

## Step 3: Failure Analysis

For each failed GitHub Actions job, drill into the logs to extract root cause.

First, identify the failed jobs within each run:

```bash
gh api repos/ai-dynamo/dynamo/actions/runs/$RUN_ID/jobs --jq '.jobs[] | select(.conclusion == "failure") | {name: .name, id: .id, html_url: .html_url}'
```

Then fetch logs for specific failed jobs. The check URL format from `gh pr checks` is `https://github.com/.../actions/runs/{RUN_ID}/job/{JOB_ID}`. Extract the `RUN_ID` (first number):

```bash
gh run view $RUN_ID --repo ai-dynamo/dynamo --log-failed 2>&1 | tail -200
```

Note: `--log-failed` concatenates all failed job logs, which can be noisy. For multi-failure runs, prefer fetching per-job to isolate root causes.

For each failure, report:
- **Job name** and which workflow it belongs to
- **Root cause** — the actual error (compilation error, test assertion, timeout, infra issue, DCO sign-off failure, etc.)
- **Relevant log excerpt** — the key lines (max 20 lines), not the full dump
- **Suggested fix** — concrete action the PR author can take
- If it looks like an infra flake, include: `gh run rerun $RUN_ID --repo ai-dynamo/dynamo --failed`

If there are no failures, say so and move on.

## Step 4: Skip & Discrepancy Analysis

This step is **exception-based only**. Do NOT enumerate expected skips — that's noise.

**If full CI did not trigger**, skip this step entirely — there are no jobs to analyze. The absence of full CI was already explained in Steps 1-2.

**If full CI ran**, compare the filter results from Step 1 against actual CI results from Step 2. Only report **surprises**:

- A filter should be `true` (files matched its paths) but the corresponding job was skipped or missing
- A gate job (`backend-status-check`, `dynamo-status-check`) was skipped — these use `if: always()` and should always run
- All framework pipelines skipped when `core` files changed (core should trigger all frameworks)

**Things that are NOT surprises** (do not report):
- Jobs skipped because their filter is `false` (e.g., docs jobs skipped when no docs changed)
- Multi-GPU tests skipped on pre-merge (these are gated to post-merge/nightly)
- arm64 copy-to-acr jobs skipped (only triggered on merge to main)
- arm64 GPU tests skipped (GPU tests are amd64-only)
- Downstream jobs skipped because their upstream was legitimately skipped
- External CI (GitLab) status — ignored entirely
- `deploy-operator` running despite `operator=false` — this job always runs in `pr.yaml`

If everything matches expectations, say "No unexpected skips or discrepancies" and move on.

## Step 5: Actionable Summary

Synthesize into a concise report:

**PR Health: [PASSING | FAILING | PENDING | CI NOT TRIGGERED | PARTIAL — lightweight only]**

**If full CI not triggered:**
- "Full CI awaiting approval — an NVIDIA maintainer needs to comment `/ok to test <sha>` to create the `pull-request/$PR_NUMBER` branch."
- "DCO check failed — commits need to be signed. See DCO.md."
- "Draft PR — some checks may not run until marked ready for review."
- Note which lightweight checks passed/failed.

**Blocking issues** — failures that must be fixed before merge:
- One-line root cause + suggested fix for each
- Note if `backend-status-check` or `dynamo-status-check` gate is failing

**Non-blocking issues** (if any):
- Flaky tests, infra timeouts, unexpected skips

**Critical path status** — the checks most relevant to this PR's changes:
- List them with current status (passed/failed/pending/not triggered)
- If pending, suggest re-checking in ~15 minutes

**Next steps** — concrete actions ordered by priority:
- "An NVIDIA maintainer should comment `/ok to test <sha>`" if full CI hasn't triggered
- "Sign your commits with `git commit --amend -s`" for DCO failures
- "Fix X in file Y" for code failures
- Re-run command for infra flakes: `gh run rerun $RUN_ID --repo ai-dynamo/dynamo --failed`
- "No action needed — CI is green" if everything passed

## Step 6: Monitor Pending Checks

If any checks are still pending or in-progress, offer to monitor them.

**List remaining checks:**

```bash
gh pr checks $PR_NUMBER --repo ai-dynamo/dynamo | grep -E 'pending|queued|in_progress'
```

Report:
- How many checks are still pending
- Which ones are on the critical path
- Estimated wait time based on similar completed jobs (e.g., if `vllm-cuda12.9-amd64 / Test` took 20m and `vllm-cuda13.0-amd64 / Test` is still running, estimate ~20m remaining)

If the user wants to wait, poll periodically:

```bash
# Re-check status
gh pr checks $PR_NUMBER --repo ai-dynamo/dynamo | grep -cE 'pass|fail|skipped'  # completed count
gh pr checks $PR_NUMBER --repo ai-dynamo/dynamo | grep -cE 'pending|queued|in_progress'  # remaining count
```

When all checks complete, re-run the summary from Step 5 with final results. Report any checks that changed from pending to failed since the last check.

## Behavior Notes

- **Concurrency cancellation**: If a PR has rapid pushes, earlier runs get cancelled. Note if you see cancelled runs and suggest checking the latest run instead.
- **Large log output**: Always truncate to relevant sections. Never dump more than 50 lines of raw log in the summary.
- **Rate limits**: If `gh` commands fail due to rate limiting, report what you could gather and suggest retrying later.
- **Multiple workflows**: A single push can trigger `pr.yaml`, `pre-merge.yml`, and `container-validation-dynamo.yml`. Check all of them.
- **`pull-request/[0-9]+` branches**: Created by copy-pr-bot after maintainer approval. Required for full CI — applies to both fork and internal PRs.
- **`external-contribution` label**: Fork PRs get this label automatically. Its presence confirms the PR is from an external contributor.
- **External CI (GitLab)**: Ignore `ci/gitlab/*` checks entirely. These are NVIDIA-internal and cannot be diagnosed from GitHub.
