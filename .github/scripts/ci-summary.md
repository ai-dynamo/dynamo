<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR CI summary

`pr-ci-summary.yml` maintains one `github-actions[bot]` comment per PR with
results for **Pre Merge**, **PR**, and **PR-XPU**. It updates when one of those
workflows starts or completes. This is a snapshot, not a continuously updated
view of every check on the PR. Other checks remain visible in GitHub's Checks tab.

The comment includes the exact PR head SHA, latest workflow run and attempt,
job counts (including skipped and cancelled jobs), failed step names, short
failure excerpts, and links to the original runs and jobs. A missing full PR run
is reported as absent, with a reminder that full CI requires approval; missing
XPU CI is shown as optional/not run.
Skipped jobs are never counted as passed tests. Counts describe jobs, not tests.

The reporting shape is inspired by the
[FlashInfer CI comment](https://github.com/flashinfer-ai/flashinfer/pull/4604#issuecomment-5715075212).
This implementation uses GitHub Actions metadata and masked job logs. It does
not classify failures as new, existing, flaky, or infrastructure-related, or
compare different nightly configurations. Excerpts are diagnostic evidence,
not a root-cause determination.

## Collection and publication

1. Resolve the triggering run to an open PR. Full CI uses the exact
   `pull-request/N` branch name; Pre Merge uses GitHub's PR associations, with
   an exact fork-owner/branch lookup when associations are absent. Validate the
   repository and current PR head SHA before accepting the association.
2. Serialize publication by PR number. Every update collects the newest run for
   each supported workflow at the current PR head. GitHub's `filter=latest` jobs
   view includes successful jobs retained across partial reruns.
3. Download at most ten failed job log tails, each capped at 1 MiB and a short
   request timeout. Extract bounded error lines, strip terminal formatting,
   redact common credential forms, and escape untrusted display text. Missing,
   expired, oversized, or unavailable logs still produce job and step links.
4. Recheck the PR head and run snapshots before updating. Use the hidden
   `dynamo-ci-summary` marker and the GitHub Actions bot identity to find the
   comment. Human comments and other bot comments are never update targets.

The privileged reporter checks out only the default-branch commit provided by
`workflow_run`, never PR code. Downloaded logs are parsed only as text. The
reporter needs `contents: read`, `actions: read`, and `pull-requests: write`;
producers need no extra permissions or artifacts. GitHub's token is not sent to
log-storage URLs. Fern preview comments use their own marker so they cannot
replace the CI summary.

## Validation and rollout

Run the dependency-free tests locally:

```sh
node --test .github/scripts/ci-summary*.test.cjs
python3 scripts/check_action_pins.py
```

The workflow also runs these Node tests with read-only permissions on PRs that
change the reporter. `publishSummary({ ..., dryRun: true })` returns the proposed
comment without listing or writing comments; an injected `loadLog(jobId)` allows
read-only replay against saved or live data.

`workflow_run` reporting becomes active only after the workflow and scripts
reach the default branch. Validate a fork PR, a copied full-CI branch, and a
partial rerun after merge. The PR tests cannot prove that a default-branch
reporter has successfully posted a live comment.

To adopt this in another repository, change the monitored workflow names in the
YAML and the file/event mapping in `ci-summary.cjs`, then replace the copied-branch
PR mapping if that repository uses a different approval flow.
