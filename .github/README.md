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
| `Build and Test - dynamo` | `container-validation-dynamo.yml` | Direct | Core Dynamo container build and tests |
| `backend-status-check` | `container-validation-backends.yml` | copy-pr-bot | Backend builds (vLLM, SGLang, TRT-LLM) |

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

### Via copy-pr-bot (run on `pull-request/N` branches)

| Workflow | File | Purpose |
|----------|------|---------|
| Backend Builds | `container-validation-backends.yml` | Builds vLLM, SGLang, TRT-LLM containers; runs framework tests |
| GitLab CI | `trigger_ci.yml` | Mirrors to GitLab, triggers internal test infrastructure |

### Scheduled / Other

| Workflow | File | Purpose |
|----------|------|---------|
| Nightly CI | `nightly-ci.yml` | Daily builds and comprehensive tests (12:00 AM PST) |
| Docs Publish | `generate-docs.yml` | Builds and publishes documentation to S3 |
| Stale Cleaner | `stale_cleaner.yml` | Closes stale issues/PRs after 30 days |
| Test Report | `test_report.yaml` | Generates test result summaries |

---

## Path Filters (`filters.yaml`)

| Filter | Paths | Used By |
|--------|-------|---------|
| `has_code_changes` | `components/**`, `lib/**`, `tests/**`, `container/**`, `*.py`, `*.rs` | Backend builds |
| `vllm` | `Dockerfile.vllm`, `components/dynamo/vllm/**` | vLLM-specific jobs |
| `sglang` | `Dockerfile.sglang`, `components/dynamo/sglang/**` | SGLang-specific jobs |
| `trtllm` | `Dockerfile.trtllm`, `components/dynamo/trtllm/**` | TRT-LLM-specific jobs |
| `docs` | `docs/**`, `**/*.md`, `**/*.rst` | Docs link check |

---

## Custom Actions

| Action | Purpose |
|--------|---------|
| `docker-build` | Builds Dynamo containers (multi-arch, multi-framework) |
| `docker-login` | Authenticates with ECR, NGC, ACR registries |
| `docker-tag-push` | Tags and pushes images to registries |
| `pytest` | Runs pytest in containers with GPU detection |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `filters.yaml` | Path patterns for conditional workflow execution |
| `release.yml` | Auto-generated release notes categories |
| `dco.yml` | Developer Certificate of Origin settings |
| `copy-pr-bot.yaml` | Copy PR bot configuration |
| `pull_request_template.md` | PR description template |

---

## Post-Merge CI

After PRs merge to `main` or `release/*`:

- **All backend builds** run (not just changed frameworks)
- **Docs link check** runs in full mode (external links)
- **GitLab CI** triggers additional internal tests
- **Rust checks** always run (not just on `*.rs` changes)

---

## GitLab CI

The `trigger_ci.yml` workflow mirrors to GitLab and triggers internal pipelines. GitLab CI is **not a required check** - it provides supplementary validation on NVIDIA infrastructure.

---

## Further Reading

- [Test Documentation](../tests/README.md) - pytest markers and test configuration
- [DCO Guide](../DCO.md) - Developer Certificate of Origin
