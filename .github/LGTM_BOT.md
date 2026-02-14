<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LGTM Bot & CI Error Classifier

CI automation for merge readiness tracking and failure diagnosis in the
ai-dynamo/dynamo repository.

## Overview

The system has two integrated components:

1. **LGTM Bot** — Tracks merge readiness on every PR (checklist comment,
   `lgtm` label management, CODEOWNERS review requests, stale review nudges).
   On-demand diagnostics via `/diagnose`.

2. **Error Classifier** — LLM-powered CI failure analysis. Runs automatically
   as the final step in CI pipelines and on-demand via `/diagnose`.
   Categorizes failures as **infrastructure errors** (network, runner, flaky
   infra) or **code errors** (build, test, compilation, runtime).

## Slash Commands

| Command | Who Can Run | Description |
|---------|------------|-------------|
| `/diagnose` | Any contributor (not `NONE` association) | Full merge-readiness diagnostic: CI status with LLM-powered failure analysis, reviews, CODEOWNERS, unresolved conversations, conflicts, DCO, PR title |
| `/lgtm-bot diagnose` | (alias) | Backward-compatible alias for `/diagnose` |
| `/ok to test <sha>` | Maintainers | Trigger full CI suite for a specific commit (required for fork PRs and unsigned commits) |

## Workflow Files

| File | Trigger | Purpose |
|------|---------|---------|
| `workflows/lgtm-bot.yml` | `workflow_run`, `pull_request_review`, `pull_request_target` | Main LGTM label management and merge checklist |
| `workflows/lgtm-bot-pr-open.yml` | `pull_request_target` (opened/synchronize/reopened) | PR onboarding: CODEOWNERS review requests, reviewer assignment, visibility comment |
| `workflows/lgtm-bot-diagnose.yml` | `/diagnose` comment | Runs `lgtm_bot_diagnose.py` — full PR diagnostic with error classification |
| `workflows/classify-workflow-errors-reusable.yml` | Called by other workflows | Reusable workflow that runs `classify_workflow_errors.py` |

The reusable error classifier is called as a final job in these pipelines:
- `pre-merge.yml`
- `pr.yaml`
- `ci-test-suite.yml`
- `container-validation-dynamo.yml`
- `copyright-checks.yml`
- `build-frontend-image.yaml`

## Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `lgtm_bot_diagnose.py` | `.github/scripts/` | Diagnose PR merge readiness, optionally invoke error classifier |
| `classify_workflow_errors.py` | `.github/scripts/` | Classify all failed jobs in a workflow run |
| `error_classification/` | repo root | Shared library: LLM client, config, PR comment posting |

## Environment Variables & Secrets

### Required Secrets

| Secret | Used By | Purpose |
|--------|---------|---------|
| `NVIDIA_INFERENCE_API_KEY` | Diagnose, Classify | API key for NVIDIA Inference API (LLM-powered analysis). Without this, the diagnose command still works but skips AI suggestions. |
| `LGTM_BOT_REVIEWERS_YAML` | PR Onboarding | Optional YAML roster for individual reviewer assignment (see Reviewer Assignment section) |

> `GITHUB_TOKEN` is automatically provided by GitHub Actions — no manual
> configuration needed.

### Optional Variables

| Variable (GitHub vars) | Default | Purpose |
|------------------------|---------|---------|
| `NVIDIA_INFERENCE_MODEL` | `aws/anthropic/bedrock-claude-opus-4-6` | Override the LLM model used for error classification |
| `LGTM_BOT_REVIEWERS_PER_TEAM` | `1` | Number of individual reviewers to pick per team/roster key |
| `SLACK_LGTM_ENABLED` | (unset) | Enable Slack notifications for LGTM label changes |

### Internal Environment (set by workflows, not user-configured)

| Variable | Purpose |
|----------|---------|
| `API_FORMAT` | API protocol (`openai` for NVIDIA Inference API) |
| `API_BASE_URL` | API endpoint (`https://inference-api.nvidia.com/v1`) |
| `ENABLE_ERROR_CLASSIFICATION` | Enable classification (`true`/`false`) |
| `ENABLE_PR_COMMENTS` | Post results as PR comments (`true`/`false`) |
| `MAX_PARALLEL_JOBS` | Max parallel jobs to analyze (default `5`) |
| `WORKFLOW_RUN_ID` | Target workflow run ID (set automatically) |
| `OUTPUT_ONLY` | Return markdown to stdout instead of posting comment |

## Reviewer Assignment

Reviewer assignment runs in the `pr-onboard` job when a PR is opened:

1. **Team reviews**: The bot parses [`CODEOWNERS`](../CODEOWNERS), matches
   changed files to team ownership rules (last matching rule wins), and
   requests reviews from the matched teams.

2. **Individual reviewers** (two strategies, both active simultaneously):
   - **Option 1 (team-based)**: Resolves CODEOWNERS team slugs to member
     logins via GitHub API, picks random eligible members (excluding PR
     author). Count controlled by `LGTM_BOT_REVIEWERS_PER_TEAM` var (default 1).
   - **Option 2 (roster)**: Reads YAML roster from `LGTM_BOT_REVIEWERS_YAML`
     secret, matches changed file paths against roster keys.

3. The bot posts a comment listing which teams are required and which files
   triggered each team.

4. The `/diagnose` command shows pending CODEOWNERS reviews in its
   diagnostic output.

### Roster YAML Format

The `LGTM_BOT_REVIEWERS_YAML` secret uses a simple `key: [users]` format:

```yaml
default: [user1, user2]
rust: [user3, user4]
python: [user5]
```

- `default` is always included when the secret is set.
- Other keys match if any changed file path contains the key or ends with
  `.{key}`.

### Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LGTM_BOT_REVIEWERS_PER_TEAM` | `1` | Number of individual reviewers to pick per team/roster key |

## Error Categories

The classifier assigns each failure to one of two categories:

| Category | Examples |
|----------|---------|
| `infrastructure_error` | Network timeouts, runner OOM, Docker pull failures, flaky infrastructure |
| `code_error` | Build failures, test failures, compilation errors, runtime exceptions |

## Permissions

The workflows require these GitHub Actions permissions:

```yaml
permissions:
  pull-requests: write   # Post/update diagnostic comments
  checks: read           # Read CI check results
  statuses: read         # Read commit status checks
  contents: read         # Read CODEOWNERS, repo files
  actions: read          # List workflow runs
```
