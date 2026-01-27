# Dynamo CI/CD

GitHub Actions workflows and configuration for the Dynamo CI/CD system.

## Quick Links

| Document | Description |
|----------|-------------|
| [PR Workflow](./PR_WORKFLOW.md) | PR CI flow diagrams and required checks |
| [Nightly Workflow](./NIGHTLY_WORKFLOW.md) | Nightly build and test pipeline diagrams |
| [Troubleshooting](./TROUBLESHOOTING.md) | Common CI issues and how to fix them |

---

## Required Checks

| Check | Workflow | Trigger | Description |
|-------|----------|---------|-------------|
| `pre-commit` | `pre-merge.yml` | Direct | Code formatting and linting |
| `copyright-checks` | `copyright-checks.yml` | Direct | Copyright header validation |
| `DCO` | DCO App | Direct | Developer Certificate of Origin signature |
| `dynamo-status-check` | `container-validation-dynamo.yml` | Direct | Core Dynamo container build and tests |
| `backend-status-check` | `pr.yaml` | copy-pr-bot | Backend builds (vLLM, SGLang, TRT-LLM) and deployment tests |

> **Note**: Checks marked "copy-pr-bot" require a maintainer to trigger CI. See [Troubleshooting](./TROUBLESHOOTING.md).

---

## Workflows

### Direct PR Checks (run immediately)

| Workflow | File | Purpose |
|----------|------|---------|
| Pre-commit | `pre-merge.yml` | Python formatting (black, isort), YAML validation, whitespace fixes |
| Copyright | `copyright-checks.yml` | Validates SPDX copyright headers on source files |
| Core Build | `container-validation-dynamo.yml` | Builds `dynamo:latest`, runs Rust checks and pytest |
| PR Title Lint | `lint-pr-title.yaml` | Validates conventional commit format, adds labels |
| DCO Comment | `dco_comment.yml` | Posts fix instructions when DCO check fails |
| Rust Checks | `pre-merge-rust.yml` | cargo fmt, clippy, tests (only on `*.rs` changes) |
| Docs Links | `docs-link-check.yml` | Validates internal/external documentation links |
| CodeQL | `codeql.yml` | Security analysis on Python code |
| Label PR | `label-pr.yml` | Automatically labels PRs based on changed files |
| PR Reminder | `pr_full_ci_reminder.yaml` | Comments on external PRs about CI process |

### Via copy-pr-bot (run on `pull-request/N` branches)

| Workflow | File | Purpose |
|----------|------|---------|
| Backend Builds & Tests | `pr.yaml` | Builds vLLM (CUDA 12.9/13), SGLang (CUDA 12.9/13), TRT-LLM (CUDA 13); runs framework tests and deployment tests |
| Frontend Build | `build-frontend-image.yaml` | Builds frontend container for both amd64 and arm64 architectures |
| GitLab CI | `trigger_ci.yml` | Mirrors to GitLab, triggers internal test infrastructure |

### Scheduled / Other

| Workflow | File | Purpose |
|----------|------|---------|
| Nightly CI | `nightly-ci.yml` | Daily builds and comprehensive tests (12:00 AM PST) |
| Post-Merge CI | `post-merge-ci.yml` | Runs full CI suite after merge to main/release branches |
| CI Test Suite | `ci-test-suite.yml` | Reusable workflow for nightly and post-merge pipelines |
| Docs Publish | `generate-docs.yml` | Builds and publishes documentation to S3 |
| Stale Cleaner | `stale_cleaner.yml` | Closes stale issues/PRs after 30 days |
| Test Report | `test_report.yaml` | Generates test result summaries |

---

## Path Filters

Workflows use the custom `changed-files` action to determine which jobs should run:

| Filter | Paths | Used By |
|--------|-------|---------|
| `core` | `components/**`, `lib/**`, `tests/**`, `container/**`, `*.py`, `*.rs` | `pr.yaml` (gates all backend jobs) |
| `vllm` | `container/Dockerfile.vllm`, `components/src/dynamo/vllm/**`, `container/deps/requirements.vllm.txt`, `examples/backends/vllm/**` | `pr.yaml`, `trigger_ci.yml` |
| `sglang` | `container/Dockerfile.sglang`, `components/src/dynamo/sglang/**`, `examples/backends/sglang/**` | `pr.yaml`, `trigger_ci.yml` |
| `trtllm` | `container/Dockerfile.trtllm`, `components/src/dynamo/trtllm/**`, `container/deps/trtllm/**`, `examples/backends/trtllm/**` | `pr.yaml`, `trigger_ci.yml` |
| `frontend` | `components/src/dynamo/frontend/**`, `lib/llm/src/**` | `build-frontend-image.yaml` |
| `operator` | `deploy/operator/**`, `deploy/helm/**` | `pr.yaml` |
| `deploy` | `examples/backends/**/deploy/**` | `pr.yaml` (deployment tests) |

---

## Custom Actions

| Action | Purpose |
|--------|---------|
| `changed-files` | Detects which components changed to determine which jobs to run |
| `docker-build` | Builds Dynamo containers (multi-arch, multi-framework, multi-CUDA) |
| `docker-login` | Authenticates with ECR, NGC, ACR registries |
| `docker-tag-push` | Tags and pushes images to registries |
| `pytest` | Runs pytest in containers with GPU detection |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `filters.yaml` | Path patterns for conditional workflow execution |
| `labeler.yml` | Auto-labeling rules for PRs based on changed files |
| `release.yml` | Auto-generated release notes categories |
| `dco.yml` | Developer Certificate of Origin settings |
| `copy-pr-bot.yaml` | Copy PR bot configuration |
| `pull_request_template.md` | PR description template |

---

## Self-Hosted Runners

CI runs on production self-hosted runners with GPU support:

| Runner Label | Architecture | GPU | Purpose |
|--------------|--------------|-----|---------|
| `prod-builder-amd-v1` | amd64 | No | CPU-only builds (arm64, frontend) |
| `prod-builder-amd-gpu-v1` | amd64 | Yes | GPU builds and tests (vLLM, SGLang, TRT-LLM) |
| `prod-builder-arm-v1` | arm64 | No | ARM64 builds (cross-platform support) |
| `prod-default-v1` | amd64 | Yes | Kubernetes deployment tests |

---

## Post-Merge CI

After PRs merge to `main` or `release/*`:

- **Full CI suite** runs via `post-merge-ci.yml` (uses `ci-test-suite.yml`)
- **All backend builds** for all CUDA versions (12.9, 13.0)
- **All framework tests** (unit, integration, e2e)
- **Docs link check** runs in full mode (external links)
- **GitLab CI** triggers additional internal tests
- **Rust checks** always run (not just on `*.rs` changes)
- **Slack notifications** sent to ops team

---

## GitLab CI

The `trigger_ci.yml` workflow mirrors to GitLab and triggers internal pipelines. GitLab CI is **not a required check** - it provides supplementary validation on NVIDIA infrastructure.

---

## Further Reading

- [Test Documentation](../tests/README.md) - pytest markers and test configuration
- [DCO Guide](../DCO.md) - Developer Certificate of Origin
