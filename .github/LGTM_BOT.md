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

# LGTM Bot

Automated merge readiness tracking and AI-powered CI diagnostics.

## What It Does

| Feature | Description |
|---------|-------------|
| **`lgtm` label** | Applied when ALL gates pass (CI + approved review + no changes requested). Removed on new pushes. |
| **Merge Checklist** | Auto-updating PR comment showing CI, review, and merge status at a glance. |
| **Required Reviewers** | Requests CODEOWNERS team reviews on PR open. Posts a comment for external contributors who cannot see the CODEOWNERS sidebar. |
| **`/diagnose`** | AI-powered CI failure analysis with root cause and fix suggestions. Groups failures into blocking vs non-blocking. Fetches internal GitLab CI logs for external contributors. |

## Slash Commands

| Command | Access | Description |
|---------|--------|-------------|
| `/diagnose` | Any contributor (not first-time) | Analyze CI failures with LLM, grouped by blocking/non-blocking |

## Workflows

| File | Trigger | Purpose |
|------|---------|---------|
| `lgtm-bot.yml` | CI completion, reviews, PR sync | Label + checklist + reviewer request |
| `lgtm-bot-diagnose.yml` | `/diagnose` comment | AI diagnostics (GitHub Actions + GitLab) |

## Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `GITHUB_TOKEN` | Auto-provided | GitHub API access |
| `LLM_API_KEY` | Optional | NVIDIA Inference API key for AI analysis |
| `TOKEN` (GITLAB env) | Optional | GitLab `PRIVATE-TOKEN` for fetching internal CI logs |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `nvidia/llama-3.3-70b-instruct` | LLM model name |
| `LLM_API_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API base URL |
| `LLM_MAX_RPM` | `10` | Rate limit (requests per minute) |

## Architecture

```
.github/workflows/lgtm-bot.yml          → label + checklist + CODEOWNERS
.github/workflows/lgtm-bot-diagnose.yml → /diagnose trigger (env: GITLAB)
.github/scripts/lgtm_bot_diagnose.py    → consolidated diagnostics entry point
error_classification/llm_client.py      → LLM API client
error_classification/config.py          → env config
```

`lgtm_bot_diagnose.py` handles GitHub Actions checks, GitLab CI log fetching,
LLM classification, blocking/non-blocking grouping, and comment posting.
GitHub API via `gh api`; GitLab API via `requests` with `PRIVATE-TOKEN`.
