# CI Troubleshooting Guide

Quick fixes for common CI issues. For workflow details, see [CI_WORKFLOWS.md](./CI_WORKFLOWS.md).

## Table of Contents

- [CI Not Triggered on My PR](#ci-not-triggered-on-my-pr)
- [Required Checks Failing](#required-checks-failing)
- [Other Issues](#other-issues)

---

## CI Not Triggered on My PR

If you see checks stuck at "Waiting for status to be reported" or no CI runs at all:

```
⏳ backend-status-check    Expected — Waiting for status to be reported
```

**Understanding which checks run when**:

- **4 checks run immediately**: pre-commit, copyright-checks, DCO, dynamo-status-check
- **1 check needs copy-pr-bot**: backend-status-check

If only `backend-status-check` is stuck at "Waiting", this is expected - it requires a maintainer to trigger via copy-pr-bot.

See [CI_WORKFLOWS.md](./CI_WORKFLOWS.md) for full details on all checks.

### Internal NVIDIA PRs: GPG Signing Required

**Symptom**: All CI checks stuck at "Waiting for status" (not just backend-status-check).

**Cause**: NVIDIA internal policy requires GPG-signed commits for CI to trigger.

**Quick fix**:
```bash
# If you don't have GPG set up yet, follow GitHub's GPG guide
# Then sign your existing commits:
git commit --amend -S --no-edit
git push --force-with-lease
```

**Full GPG setup**: See GitHub's documentation on [signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification).


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

Edit your PR title to follow conventional commits format:
```
<type>: <description>

Valid types: feat, fix, docs, test, ci, refactor, perf, chore, revert, style, build

Examples:
✅ feat: add new router endpoint
✅ fix: resolve memory leak in worker
❌ Added new feature (missing type)
```

---

### Stale PR/Issue Closed

PRs/issues are auto-closed after 30 days of inactivity. Add the `backlog` label to prevent this.

---


## Related Documentation

- [CI Workflows](./CI_WORKFLOWS.md) - How PR, post-merge, and nightly CI work
- [DCO Guide](../DCO.md) - Developer Certificate of Origin

