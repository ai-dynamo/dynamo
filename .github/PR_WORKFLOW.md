# Pull Request CI Workflow

This document explains how CI runs on pull requests, what checks are required, and how different workflows are triggered based on the files you change.

## Required Checks

The following checks **must pass** before a PR can be merged:

| Check | Workflow | Description |
|-------|----------|-------------|
| `pre-commit` | `pre-merge.yml` | Code formatting and linting via pre-commit hooks |
| `copyright-checks` | `copyright-checks.yml` | Validates copyright headers on source files |
| `Build and Test - dynamo` | `container-validation-dynamo.yml` | Builds core Dynamo container and runs tests |
| `backend-status-check` | `container-validation-backends.yml` | Builds backend containers (vLLM, SGLang, TRT-LLM) and runs tests |

---

## PR Workflow Overview

```mermaid
flowchart TD
    subgraph PR["Pull Request Opened/Updated"]
        A[PR Created or Push to PR]
    end

    subgraph Always["Always Run"]
        B[pre-commit<br/>Code formatting & linting]
        C[copyright-checks<br/>Header validation]
        D[lint-pr-title<br/>Conventional commits]
        E[DCO Check<br/>Signed commits]
        F[CodeQL<br/>Security analysis]
        G[docs-link-check<br/>Link validation]
    end

    subgraph Conditional["Conditional - Based on Changed Files"]
        H{has_code_changes?}
        I[Build and Test - dynamo<br/>Core container build & tests]
        J[vLLM build & tests]
        K[SGLang build & tests]
        L[TRT-LLM build & tests]
        M[backend-status-check<br/>Aggregates backend results]
    end

    A --> B & C & D & E & F & G
    A --> H
    H -->|Yes| I
    H -->|Yes + vllm files| J
    H -->|Yes + sglang files| K
    H -->|Yes + trtllm files| L
    J & K & L --> M

    style B fill:#90EE90
    style C fill:#90EE90
    style I fill:#90EE90
    style M fill:#90EE90
```

---

## Workflow Details

### 1. Pre-Commit Checks (`pre-merge.yml`)

**Trigger**: All pull requests  
**Required**: ✅ Yes

Runs pre-commit hooks to validate code formatting and linting:
- Python formatting (black, isort)
- YAML/JSON validation
- Trailing whitespace removal
- End-of-file fixers
- And other configured hooks

### 2. Copyright Checks (`copyright-checks.yml`)

**Trigger**: All pull requests  
**Required**: ✅ Yes

Validates that all source files have proper NVIDIA copyright headers:
```
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

### 3. PR Title Linting (`lint-pr-title.yaml`)

**Trigger**: PR opened, edited, synchronized, reopened  
**Required**: ❌ No (but recommended)

Validates PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) format:
```
<type>: <description>

Types: feat, fix, docs, test, ci, refactor, perf, chore, revert, style, build
```

Valid examples:
- `feat: add new router component`
- `fix: resolve memory leak in KV cache`
- `docs: update installation guide`

Also automatically adds labels based on the PR type.

### 4. DCO Check

**Trigger**: All pull requests  
**Required**: ✅ Yes (enforced by GitHub)

All commits must be signed off with the Developer Certificate of Origin:
```bash
git commit -s -m "Your commit message"
```

If DCO fails, a bot will comment with instructions. See [DCO.md](../DCO.md) for details.

### 5. Rust Pre-Merge Checks (`pre-merge-rust.yml`)

**Trigger**: Changes to `*.rs`, `Cargo.toml`, `Cargo.lock`  
**Required**: ❌ No

Runs Rust-specific validation:
- `cargo fmt --check` - Code formatting
- `cargo clippy` - Linting
- `cargo test` - Unit and doc tests
- `cargo deny` - License and dependency checks

Runs on multiple workspace directories:
- Root workspace
- `lib/bindings/python`
- `lib/runtime/examples`
- `launch/dynamo-run`

### 6. Core Dynamo Build (`container-validation-dynamo.yml`)

**Trigger**: All PRs (push to main, release branches, or PRs)  
**Required**: ✅ Yes (`Build and Test - dynamo`)

```mermaid
flowchart LR
    A[Checkout] --> B[Build dynamo:latest<br/>--target dev --framework none]
    B --> C[Start nats-server & etcd]
    C --> D[Rust checks<br/>fmt, clippy, tests]
    D --> E[pytest parallel<br/>pre_merge + parallel]
    E --> F[pytest sequential<br/>pre_merge + mypy]
    F --> G[Upload test artifacts]
```

### 7. Backend Container Builds (`container-validation-backends.yml`)

**Trigger**: `has_code_changes` filter matches  
**Required**: ✅ Yes (`backend-status-check`)

Builds and tests framework-specific containers:

```mermaid
flowchart TD
    subgraph Parallel["Parallel Jobs"]
        subgraph vLLM["vLLM"]
            V1[Build amd64] --> V2[Run tests]
            V3[Build arm64]
        end
        
        subgraph SGLang["SGLang"]
            S1[Build amd64] --> S2[Run tests]
            S3[Build arm64]
        end
        
        subgraph TRT["TRT-LLM"]
            T1[Build amd64] --> T2[Run tests]
            T3[Build arm64]
        end
        
        subgraph Operator["Operator"]
            O1[Lint & Test]
            O2[Build container]
        end
    end
    
    V2 & V3 & S2 & S3 & T2 & T3 & O2 --> Check[backend-status-check]
    
    style Check fill:#90EE90
```

Each framework build:
1. Builds the container image for the target architecture
2. Runs sanity checks on the image
3. Pushes to container registry
4. Runs pytest with `pre_merge and <framework>` markers (amd64 only)

### 8. Documentation Link Check (`docs-link-check.yml`)

**Trigger**: All PRs and pushes to main  
**Required**: ❌ No

Two-part validation:
1. **Lychee**: External link checking (offline mode for PRs, full mode for main)
2. **Broken Links Script**: Internal markdown link validation

### 9. CodeQL Analysis (`codeql.yml`)

**Trigger**: All pull requests  
**Required**: ❌ No

Runs GitHub's CodeQL security analysis on Python code.

---

## File Filters and Conditional Execution

The `filters.yaml` file determines which workflows run based on changed files:

```mermaid
flowchart TD
    A[Changed Files] --> B{Match filters.yaml}
    
    B -->|"docs/**<br/>**/*.md"| C[docs-link-check]
    B -->|"*.rs<br/>Cargo.*"| D[pre-merge-rust]
    B -->|"has_code_changes<br/>(components, lib, tests, etc.)"| E[container-validation-backends]
    
    E --> F{Framework-specific?}
    F -->|"Dockerfile.vllm<br/>components/dynamo/vllm/**"| G[vLLM jobs only]
    F -->|"Dockerfile.sglang<br/>components/dynamo/sglang/**"| H[SGLang jobs only]
    F -->|"Dockerfile.trtllm<br/>components/dynamo/trtllm/**"| I[TRT-LLM jobs only]
```

### `has_code_changes` Filter

This filter matches when any of these paths change:
- `.github/workflows/**`, `.github/filters.yaml`, `.github/actions/**`
- `benchmarks/**`
- `components/**`
- `container/**`
- `deploy/**`
- `examples/**`
- `launch/**`
- `lib/**`
- `recipes/**`
- `tests/**`
- `*.toml`, `*.lock`, `*.py`, `*.rs`

---

## Understanding PR Check Results

### All Checks Passed ✅

Your PR is ready for review. All required checks have passed.

### Some Checks Failed ❌

1. **Check which required checks failed** (see table above)
2. **Click on the failed check** to see detailed logs
3. **Common fixes**:
   - `pre-commit`: Run `pre-commit run --all-files` locally
   - `copyright-checks`: Add copyright headers to new files
   - `DCO`: Amend commits with `-s` flag (see [Troubleshooting](./TROUBLESHOOTING.md))
   - Build failures: Check Docker build logs for errors

### Checks Not Running 🟡

See [Troubleshooting - CI Not Triggered](./TROUBLESHOOTING.md#ci-not-triggered-on-my-pr) for common causes.

---

## External Contributions

For PRs from forks (external contributors):

1. A welcome comment is posted with CI information
2. The `external-contribution` label is added automatically
3. **CI requires approval** from an NVIDIA maintainer via `copy-pr-bot`
4. Once approved, CI runs with full access to build infrastructure

See [Troubleshooting](./TROUBLESHOOTING.md#external-contribution-ci-not-running) for details.

---

## Post-Merge CI

After your PR is merged to `main` or a `release/*` branch, additional CI pipelines run automatically.

### What Runs After Merge

```mermaid
flowchart TD
    subgraph Trigger["🔀 PR Merged to main/release"]
        A[Push to main or release/*]
    end

    subgraph PostMerge["Post-Merge Jobs"]
        B[Rust Checks<br/>pre-merge-rust.yml]
        C[Core Dynamo Build & Tests<br/>container-validation-dynamo.yml]
        D[Backend Builds & Tests<br/>container-validation-backends.yml]
        E[Docs Link Check<br/>Full external link validation]
        F[GitLab CI Trigger<br/>Internal test infrastructure]
    end

    subgraph Extended["Extended Testing"]
        G[Fault Tolerance Tests<br/>Kubernetes deployment tests]
    end

    A --> B & C & D & E & F
    D --> G
```

### Post-Merge vs PR Checks

| Aspect | PR Checks | Post-Merge |
|--------|-----------|------------|
| **Rust checks** | Only on Rust file changes | Always runs on main |
| **Docs link check** | Offline mode (internal links only) | Full mode (external links too) |
| **Fault tolerance tests** | Not run | Runs Kubernetes deployment tests |
| **GitLab CI** | Triggered for internal PRs | Always triggered on main |
| **Image publishing** | Pushed to staging registry | Pushed to production registries |

### Fault Tolerance Tests

After backend containers are built, fault tolerance tests run on Kubernetes:

- **Frameworks tested**: vLLM, SGLang, TRT-LLM
- **Test scenarios**: Disaggregated prefill/decode with worker pod failures
- **Infrastructure**: Azure AKS cluster with GPU nodes
- **Duration**: ~30-60 minutes per framework

These tests validate that Dynamo deployments can recover from worker failures.

### GitLab CI Integration

The `trigger_ci.yml` workflow mirrors the repository to GitLab and triggers internal CI pipelines. This provides:

- Additional testing on NVIDIA internal infrastructure
- Extended test suites not available on GitHub runners
- Hardware-specific validation

> **Note**: GitLab CI results are informational and do not block releases.

### Why Post-Merge CI Matters

Even though your PR passed all checks, post-merge CI catches:

1. **Integration issues** - Conflicts with other recently merged PRs
2. **Extended test failures** - Tests that only run post-merge
3. **Infrastructure validation** - Kubernetes deployment correctness
4. **External link rot** - Broken external URLs in documentation

If post-merge CI fails, the team is notified and will address the issue. You may be contacted if your PR is identified as the cause.

---

## Related Documentation

- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Nightly CI Workflow](./NIGHTLY_WORKFLOW.md)
- [Test Documentation](../tests/README.md) - pytest markers and test configuration

