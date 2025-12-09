# Dynamo CI/CD

This directory contains GitHub Actions workflows, custom actions, and configuration files for the Dynamo CI/CD system.

## Quick Links

| Document | Description |
|----------|-------------|
| [PR Workflow](./PR_WORKFLOW.md) | How CI runs on pull requests, required checks, and post-merge testing |
| [Nightly Workflow](./NIGHTLY_WORKFLOW.md) | Scheduled nightly builds, tests, and multi-arch container publishing |
| [Troubleshooting](./TROUBLESHOOTING.md) | Common CI issues and how to fix them |

---

## Directory Structure

```
.github/
├── README.md                    # This file - entry point for CI documentation
├── PR_WORKFLOW.md               # Pull request CI documentation
├── NIGHTLY_WORKFLOW.md          # Nightly CI documentation
├── TROUBLESHOOTING.md           # Common issues and solutions
│
├── workflows/                   # GitHub Actions workflow definitions
│   ├── pre-merge.yml            # Pre-commit hooks (formatting, linting)
│   ├── pre-merge-rust.yml       # Rust-specific checks (cargo fmt, clippy, tests)
│   ├── container-validation-dynamo.yml    # Core Dynamo build and tests
│   ├── container-validation-backends.yml  # Backend builds (vLLM, SGLang, TRT-LLM)
│   ├── copyright-checks.yml     # Copyright header validation
│   ├── lint-pr-title.yaml       # Conventional commit PR title validation
│   ├── dco_comment.yml          # DCO failure helper comments
│   ├── docs-link-check.yml      # Documentation link validation
│   ├── codeql.yml               # Security analysis
│   ├── nightly-ci.yml           # Scheduled nightly builds and tests
│   ├── generate-docs.yml        # Documentation generation and publishing
│   ├── trigger_ci.yml           # GitLab CI mirror (internal testing)
│   ├── test_report.yaml         # Test result summary generation
│   ├── pr_full_ci_reminder.yaml # External contributor reminder
│   └── stale_cleaner.yml        # Stale issue/PR cleanup
│
├── actions/                     # Reusable composite actions
│   ├── docker-build/            # Build Dynamo container images
│   ├── docker-login/            # Login to container registries (ECR, NGC, ACR)
│   ├── docker-tag-push/         # Tag and push images to registries
│   └── pytest/                  # Run pytest in containers
│
├── scripts/                     # Helper scripts for workflows
│   └── parse_buildkit_output.py # Parse Docker build metrics
│
├── ISSUE_TEMPLATE/              # GitHub issue templates
│   ├── bug_report.yml           # Bug report template
│   ├── feature_request.yml      # Feature request template
│   └── config.yml               # Issue template configuration
│
├── filters.yaml                 # Path filters for conditional workflow execution
├── release.yml                  # Release notes configuration
├── dco.yml                      # DCO (Developer Certificate of Origin) config
├── copy-pr-bot.yaml             # Copy PR bot configuration
└── pull_request_template.md     # PR description template
```

---

## Key Configuration Files

### `filters.yaml`

Defines path patterns used to determine which workflows should run based on changed files:

| Filter | Description | Triggers |
|--------|-------------|----------|
| `docs` | Documentation files | `docs/**`, `**/*.md`, `**/*.rst` |
| `ci` | CI configuration | `.github/workflows/**`, `.github/filters.yaml`, `.github/actions/**` |
| `has_code_changes` | Source code changes | Benchmarks, components, containers, deploy, examples, lib, tests, etc. |
| `vllm` | vLLM-specific files | Dockerfile.vllm, vLLM components, requirements, tests |
| `sglang` | SGLang-specific files | Dockerfile.sglang, SGLang components, tests |
| `trtllm` | TensorRT-LLM files | Dockerfile.trtllm, TRT-LLM components, deps, tests |
| `sdk` | Deployment SDK | `deploy/**` |

### `release.yml`

Configures auto-generated release notes with categories:
- 🚀 **Features & Improvements** (`feat`, `perf`, `refactor`)
- 🐛 **Bug Fixes** (`fix`, `revert`)
- 📚 **Documentation** (`docs`)
- 🛠️ **Build, CI and Test** (`build`, `ci`, `test`)

### `dco.yml`

Developer Certificate of Origin configuration. All commits must be signed off (`Signed-off-by: Name <email>`).

---

## Custom Actions

### `docker-build`
Builds Dynamo container images with support for:
- Multiple frameworks (vLLM, SGLang, TRT-LLM)
- Multiple targets (dev, framework, runtime)
- Multi-architecture (amd64, arm64)
- sccache for faster Rust builds
- Build metrics collection

### `docker-login`
Authenticates with container registries:
- AWS ECR
- NVIDIA NGC
- Azure ACR

### `docker-tag-push`
Tags and pushes images to multiple registries in a single action.

### `pytest`
Runs pytest inside container images with:
- GPU detection and runtime configuration
- JUnit XML report generation
- Dry-run mode for test collection
- Artifact upload for test results

---

## GitLab CI Integration

Some workflows mirror the repository to GitLab and trigger internal CI pipelines for additional testing on NVIDIA infrastructure. **GitLab CI is not a required check** for merging PRs - it provides supplementary validation.

---

## Further Reading

- [PR Workflow Documentation](./PR_WORKFLOW.md) - Detailed PR CI flow
- [Nightly Workflow Documentation](./NIGHTLY_WORKFLOW.md) - Nightly build and test pipeline
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common CI issues and fixes
- [Test Documentation](../tests/README.md) - Test markers and pytest configuration

