# Dynamo CI/CD

GitHub Actions workflows and configuration for the Dynamo CI/CD system.

## Quick Links

| Document | Description |
|----------|-------------|
| [CI Workflows](./CI_WORKFLOWS.md) | How PR, post-merge, and nightly CI work |
| [Troubleshooting](./TROUBLESHOOTING.md) | Common CI issues and how to fix them |

---

## Workflow Files

### Direct PR Checks (run immediately on PRs)

| Workflow | File | Purpose |
|----------|------|---------|
| Pre-commit | `pre-merge.yml` | Python formatting (black, isort), YAML validation |
| Copyright | `copyright-checks.yml` | SPDX copyright header validation |
| Core Build | `container-validation-dynamo.yml` | Builds `dynamo:latest`, runs Rust + pytest |
| PR Title Lint | `lint-pr-title.yaml` | Validates conventional commit format |
| DCO Comment | `dco_comment.yml` | Posts DCO fix instructions |
| Rust Checks | `pre-merge-rust.yml` | cargo fmt, clippy, tests (on `*.rs` changes) |
| Docs Links | `docs-link-check.yml` | Validates documentation links |
| CodeQL | `codeql.yml` | Security analysis on Python |
| Label PR | `label-pr.yml` | Auto-labels PRs by changed files |
| PR Reminder | `pr_full_ci_reminder.yaml` | Comments on external PRs |

### Backend/Frontend Builds (run on `pull-request/N` branches via copy-pr-bot)

| Workflow | File | Purpose |
|----------|------|---------|
| Backend Builds | `pr.yaml` | Builds vLLM/SGLang/TRT-LLM + tests + deployment tests |
| Frontend Build | `build-frontend-image.yaml` | Builds frontend container (amd64, arm64) |
| GitLab CI | `trigger_ci.yml` | Mirrors to GitLab for internal tests |

### Scheduled / Post-Merge / Release

| Workflow | File | Purpose |
|----------|------|---------|
| Nightly CI | `nightly-ci.yml` | Daily comprehensive tests (12 AM PST) |
| Post-Merge CI | `post-merge-ci.yml` | Full CI after merge to main/release |
| CI Test Suite | `ci-test-suite.yml` | Reusable workflow (nightly + post-merge) |
| Release | `release.yml` | Automated release builds and publishing |
| Docs Publish | `generate-docs.yml` | Publishes docs to S3 |
| Stale Cleaner | `stale_cleaner.yml` | Closes stale issues/PRs |
| Test Report | `test_report.yaml` | Test result summaries |

---

## Custom Actions

| Action | Purpose |
|--------|---------|
| `changed-files` | Detects which components changed |
| `docker-build` | Builds containers (multi-arch, multi-CUDA) |
| `docker-login` | Authenticates with ECR, NGC, ACR |
| `docker-tag-push` | Tags and pushes images |
| `pytest` | Runs pytest in containers with GPU detection |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `copy-pr-bot.yaml` | Copy PR bot configuration |
| `dco.yml` | Developer Certificate of Origin settings |
| `filters.yaml` | Path patterns for conditional workflow execution |
| `labeler.yml` | Auto-labeling rules for PRs based on changed files |
| `pull_request_template.md` | PR description template |
| `release.yml` | Auto-generated release notes categories |

---

## Self-Hosted Runners

| Runner Label | Arch | GPU | Purpose |
|--------------|------|-----|---------|
| `prod-builder-amd-v1` | amd64 | No | CPU-only builds |
| `prod-builder-amd-gpu-v1` | amd64 | Yes | GPU builds and tests |
| `prod-builder-arm-v1` | arm64 | No | ARM64 builds |
| `prod-default-v1` | amd64 | Yes | Kubernetes deployment tests |

---

## Workflow Modes

Dynamo has two CI architectures optimized for different purposes:

### PR Mode (`pr.yaml`)
- **Goal**: Fast feedback on PRs
- **Strategy**: Conditional - only builds changed frameworks
- **Runs on**: `pull-request/N` branches (via copy-pr-bot)

### Full Test Suite (`ci-test-suite.yml`)
- **Goal**: Comprehensive validation
- **Strategy**: Always builds all frameworks
- **Runs via**: `nightly-ci.yml` (daily) and `post-merge-ci.yml` (after merge)

---

## Further Reading

- [Test Documentation](../tests/README.md) - pytest markers and test configuration
- [DCO Guide](../DCO.md) - Developer Certificate of Origin
