<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runbook — nightly artifact publication failure

Triage guide for the PagerDuty incident raised by
[`notify-pagerduty.yml`](workflows/notify-pagerduty.yml) when the nightly pipeline fails to
publish artifacts.

> This repository is public. Registry hostnames, Artifactory paths, credential owners and
> rotation procedures live in the internal ops-docs runbook — this file covers only what is
> already visible in the workflow definitions. Follow the internal runbook for anything
> requiring credentials.

## What paged you

The nightly (`nightly-ci.yml`, cron `0 8 * * *` UTC = midnight PST) publishes via
`release.yml`. Incidents come from one of two jobs:

| `dedup_key` contains | Meaning |
|---|---|
| `nightly-artifact-publication` | The publish ran and a stage failed. |
| `nightly-not-published` | A build gate failed, so `release` was skipped and **nothing** was published. |

Only unattended cron runs page. RC/GA releases and manual re-runs never do — see the gating
comment in `release.yml`.

## First: read `custom_details`

The incident body carries `failing_stages`, each stage's job result, `ngc_version_tag`,
`wheel_version`, `source_commit`, and a direct `run_url` to the failing Actions run attempt.
Start there — it identifies which system to look at before you open anything.

## Stage-by-stage

### `prepare-release` (severity: error)
Version/tag computation failed, so nothing downstream ran. Almost always a repo-state problem
(unexpected version in the source tree) rather than an infrastructure one. Check the job log;
this rarely needs Ops-Support credentials.

### `NGC publish` (severity: critical)
`release-publish` crane-copies runtime images ECR → NGC. A failure here means the core runtime
images did **not** reach NGC, so the floating `:nightly` tags are stale for consumers.

Usual causes, in order of likelihood:
1. **Expired/rotated NGC publish credentials** — the login step fails. Most common.
2. **Source image missing in ECR** — post-merge CI did not push images for `source_commit`.
   Confirm post-merge CI succeeded for that SHA before blaming the release job.
3. **NGC-side outage or rate limiting.**

Note: only *core runtime* copy failures fail this job. Optional images (operator, planner,
frontend, snapshot-agent, EFA variants) are fail-soft and only increment
`skipped_image_copies`, which is reported for correlation but **never pages on its own**.

### `Artifactory wheels` (severity: critical)
`stage-wheels-artifactory` extracts wheels from the runtime image and uploads them. A failure
means `pip install` of the nightly wheels is broken for consumers.

Usual causes: expired Artifactory token; the upload rejected (non-201); or the wheel-version
verification tripping the UTC-midnight date-drift check — the nightly runs *at* 00:00 PST, so
a run straddling UTC midnight can compute a `wheel_version` date that no longer matches.

### `GitLab release trigger` (severity: error)
Artifacts **were** published successfully, but the downstream GitLab release-automation
pipeline (nSpect/security registration and scans) never started. Consumers are unaffected;
compliance/scan registration for this nightly is missing. Lower urgency than the two above.

A `skipped` result here is *not* a failure — it is the documented `skip_gitlab_pipeline`
emergency bypass and never pages.

### `nightly-not-published`
One of the gating builds (vllm / sglang / trtllm / dynamo-pipeline) failed, so the release job
was skipped by design to avoid publishing a partial release. This is a **development**
regression, not an Ops-Support infrastructure problem — the fix is in the build, and the
failure is already reported in the nightly Slack alert. Hand it to the owning team.

## After you fix it

- Incidents do **not** auto-resolve. Close the incident manually once publication succeeds.
- Re-running the failed jobs in the same Actions run reuses the same `dedup_key`, so it will
  not open a second incident.
- To republish without waiting for the next cron, re-run the `release` job from the nightly
  run, or `workflow_dispatch` `release.yml` with the same `commit_sha`.

## Muting

For a planned NGC or Artifactory maintenance window, set the repository variable
`PAGERDUTY_ENABLED` to `false`. The notifier then logs a notice and sends nothing. **Set it
back to `true` afterwards** — there is no automatic re-enable.

If the `PAGERDUTY_OPS_SUPPORT_ROUTING_KEY` secret is unset or empty, the notifier logs a
warning and exits successfully; paging is disabled but the nightly is unaffected.
