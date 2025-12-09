# PR Workflow

## Required Checks

| Check | Trigger |
|-------|---------|
| `pre-commit` | Direct |
| `copyright-checks` | Direct |
| `Build and Test - dynamo` | Direct |
| `backend-status-check` | Via `pull-request/N` branch |

---

## How copy-pr-bot Works

Some checks (backend builds, GitLab CI) don't run on direct PR events. They require a maintainer to trigger **copy-pr-bot**, which creates a `pull-request/N` branch. The **push to that branch** triggers the workflows.

```mermaid
flowchart LR
    A[PR Opened] --> B{Maintainer comments<br/>to trigger bot}
    B --> C[Bot creates<br/>pull-request/N branch]
    C --> D[Push triggers<br/>backend builds + GitLab CI]
```

This applies to both internal and external PRs.

---

## PR Flow

```mermaid
flowchart TD
    subgraph PR["Pull Request Opened/Updated"]
        A[PR Created or Push to PR]
    end

    subgraph Direct["Direct PR Checks"]
        B[pre-commit]
        C[copyright-checks]
        I[Build and Test - dynamo]
        D[lint-pr-title]
        E[DCO Check]
        F[CodeQL]
        G[docs-link-check]
    end

    subgraph CopyPR["Via pull-request/N branch"]
        H{has_code_changes?}
        J[vLLM build & tests]
        K[SGLang build & tests]
        L[TRT-LLM build & tests]
        M[backend-status-check]
        N[GitLab CI]
    end

    A --> B & C & I & D & E & F & G
    A -.->|copy-pr-bot| H & N
    H -->|Yes| J & K & L
    J & K & L --> M

    style B fill:#1f6feb,color:#fff
    style C fill:#1f6feb,color:#fff
    style I fill:#1f6feb,color:#fff
    style M fill:#1f6feb,color:#fff
    style N fill:#6e7681,color:#fff
```

---

## Core Dynamo Build (`container-validation-dynamo.yml`)

Runs on **all PRs** directly. Builds the core Dynamo container and runs Rust checks + pytest.

```mermaid
flowchart LR
    A[Checkout] --> B[Build dynamo:latest]
    B --> C[Start nats + etcd]
    C --> D[Rust checks]
    D --> E[pytest parallel]
    E --> F[pytest sequential]
    F --> G[Upload artifacts]
```

---

## Backend Builds (`container-validation-backends.yml`)

Only runs when code is pushed to `pull-request/N` branches or `main`/`release/*`. Uses `filters.yaml` to check if `has_code_changes` is true.

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
    
    V2 & V3 & S2 & S3 & T2 & T3 --> Check[backend-status-check]
    Check --> FT[Fault Tolerance Tests]
    
    style Check fill:#1f6feb,color:#fff
```

---

## Path Filters (`filters.yaml`)

| Filter | Used By | Paths |
|--------|---------|-------|
| `has_code_changes` | Backend builds | `components/**`, `lib/**`, `tests/**`, `container/**`, `*.py`, `*.rs`, etc. |
| `vllm` | GitLab CI | `Dockerfile.vllm`, `components/dynamo/vllm/**` |
| `sglang` | GitLab CI | `Dockerfile.sglang`, `components/dynamo/sglang/**` |
| `trtllm` | GitLab CI | `Dockerfile.trtllm`, `components/dynamo/trtllm/**` |

GitLab CI always runs (skips only `.md`/`.rst` changes) but uses framework filters to tell GitLab *which* frameworks to test.

---

## Post-Merge

After merge to `main` or `release/*`, workflows trigger on push (no copy-pr-bot needed).

```mermaid
flowchart TD
    subgraph Trigger["PR Merged to main/release"]
        A[Push to main or release/*]
    end

    subgraph PostMerge["Post-Merge Jobs"]
        B[Rust Checks]
        C[Core Dynamo Build]
        D[Backend Builds]
        E[Docs Link Check - Full]
        F[GitLab CI]
    end

    subgraph Extended["Extended Testing"]
        G[Fault Tolerance Tests]
    end

    A --> B & C & D & E & F
    D --> G
```

### Post-Merge vs PR

| Aspect | PR | Post-Merge |
|--------|-----|------------|
| Rust checks | On `*.rs` changes only | Always |
| Docs link check | Offline mode | Full external links |
| Fault tolerance | Via `pull-request/N` branch | ✅ Always |
| GitLab CI | Via `pull-request/N` branch | Always |

---

## Related

- [README](./README.md) - Workflow details and configuration
- [Nightly Workflow](./NIGHTLY_WORKFLOW.md) - Scheduled builds
- [Troubleshooting](./TROUBLESHOOTING.md) - Common CI issues
