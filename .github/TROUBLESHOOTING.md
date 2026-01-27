# CI Troubleshooting Guide

This guide covers common CI issues and how to resolve them.

## Table of Contents

- [CI Not Triggered on My PR](#ci-not-triggered-on-my-pr)
  - [Internal NVIDIA PRs: GPG Signing Required](#internal-nvidia-prs-gpg-signing-required)
  - [External Contributions: CI Not Running](#external-contributions-ci-not-running)
- [Required Checks Failing](#required-checks-failing)
  - [DCO Check Failed](#dco-check-failed)
  - [Pre-commit Check Failed](#pre-commit-check-failed)
  - [Copyright Check Failed](#copyright-check-failed)
- [Other Issues](#other-issues)

---

## CI Not Triggered on My PR

If you see checks stuck at "Waiting for status to be reported" or no CI runs at all:

```
⏳ backend-status-check    Expected — Waiting for status to be reported
```

### Understanding Which Checks Run When

| Check | Runs on direct PR? | Requires copy-pr-bot? |
|-------|-------------------|----------------------|
| `pre-commit` | ✅ Yes | No |
| `copyright-checks` | ✅ Yes | No |
| `DCO` | ✅ Yes | No |
| `dynamo-status-check` | ✅ Yes | No |
| `label-pr` | ✅ Yes | No |
| `pr-reminder` (external PRs) | ✅ Yes | No |
| `backend-status-check` | ❌ No | **Yes** |
| `frontend-status-check` | ❌ No | **Yes** |

If only `backend-status-check` or `frontend-status-check` are stuck at "Waiting", this is expected - they require a maintainer to trigger via copy-pr-bot.

### Auto-Labeling

The `label-pr` workflow automatically adds labels based on changed files:
- **Backend labels**: `backend::vllm`, `backend::sglang`, `backend::trtllm`
- **Component labels**: `router`, `frontend`, `planner`
- **Deployment labels**: `deployment::k8s`
- **Other labels**: `build`, `ci`, `documentation`, `multimodal`

External PRs also get the `external-contribution` label automatically.

### Internal NVIDIA PRs: GPG Signing Required

**Symptom**: CI checks show "Waiting for status" and never start for PRs from NVIDIA employees.

**Cause**: Commits are not GPG signed. NVIDIA internal policy requires GPG-signed commits for CI to trigger.

**Solution**: Sign your commits with GPG.


---

## Required Checks Failing

### DCO Check Failed

**Symptom**: DCO check shows ❌ and a bot comments about missing sign-off.

**Cause**: One or more commits are missing the `Signed-off-by` line.

**Solution**: Add sign-off to your commits.



See [DCO.md](../DCO.md) for detailed instructions.

---

### Pre-commit Check Failed

**Symptom**: `pre-commit` check fails with formatting or linting errors.

**Solution**: Run pre-commit locally and push the fixes.

```bash
# Install pre-commit (one time)
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files

# Commit the fixes
git add -A
git commit -s -m "fix: apply pre-commit formatting"
git push
```

**Common Issues**:
- **Black formatting**: Python code style
- **isort**: Import ordering
- **Trailing whitespace**: Extra spaces at end of lines
- **End of file**: Missing newline at end of file
- **YAML/JSON**: Syntax validation

---

### Copyright Check Failed

**Symptom**: `copyright-checks` fails with missing header errors.

**Solution**: Add the required copyright header to new files.

```
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

---

## Other Issues

### PR Title Validation Failed

**Symptom**: `lint-pr-title` check fails.

**Cause**: PR title doesn't follow conventional commits format.

**Solution**: Edit PR title to match format:
```
<type>: <description>

Valid types: feat, fix, docs, test, ci, refactor, perf, chore, revert, style, build
```

**Examples**:
- ✅ `feat: add new router endpoint`
- ✅ `fix: resolve memory leak in worker`
- ❌ `Added new feature` (missing type)
- ❌ `FEAT: something` (type should be lowercase)

---

### Stale PR/Issue Closed

**Symptom**: Your PR or issue was automatically closed.

**Cause**: The `stale_cleaner` workflow closes items after 30 days of inactivity + 5 day warning.

**Solution**:
- **To prevent**: Add the `backlog` label to exempt from stale cleanup
- **To reopen**: Comment on the issue/PR to reopen it

---

### Checks Running on Wrong Files

**Symptom**: Backend builds running when you only changed documentation.

**Cause**: The `filters.yaml` patterns may match your changes unexpectedly.

**Check which filters match**:
1. Review [filters.yaml](./filters.yaml)
2. Verify your changed files against the patterns
3. `has_code_changes` is broad and includes many paths

---


## Related Documentation

- [PR Workflow](./PR_WORKFLOW.md) - Understanding PR checks
- [Nightly Workflow](./NIGHTLY_WORKFLOW.md) - Nightly CI pipeline
- [DCO Guide](../DCO.md) - Developer Certificate of Origin

