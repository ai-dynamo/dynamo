# PR Workflow

## Required Checks

| Check | Trigger |
|-------|---------|
| `pre-commit` | Direct |
| `copyright-checks` | Direct |
| `DCO` | Direct |
| `dynamo-status-check` | Direct |
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
        E[DCO]
        I[dynamo-status-check]
        D[lint-pr-title]
        F[CodeQL]
        G[docs-link-check]
        P[label-pr]
        Q[pr-reminder]
    end

    subgraph CopyPR["Via pull-request/N branch"]
        H{changed-files}
        J[vLLM CUDA 12.9/13]
        K[SGLang CUDA 12.9/13]
        L[TRT-LLM CUDA 13]
        O[Operator]
        R[Frontend]
        M[backend-status-check]
        S[Deploy Tests]
        N[GitLab CI]
    end

    A --> B & C & I & D & E & F & G & P & Q
    A -.->|copy-pr-bot| H & N
    H -->|core/vllm| J
    H -->|core/sglang| K
    H -->|core/trtllm| L
    H -->|operator| O
    H -->|frontend| R
    J & K & L & O --> M
    M --> S

    style B fill:#1f6feb,color:#fff
    style C fill:#1f6feb,color:#fff
    style E fill:#1f6feb,color:#fff
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

## Backend Builds (`pr.yaml`)

Only runs when code is pushed to `pull-request/N` branches or `main`/`release/*`. Uses the `changed-files` action to determine which frameworks to build.

### Multi-CUDA Support

Both vLLM and SGLang now build for **CUDA 12.9** and **CUDA 13.0**. TRT-LLM builds for CUDA 13.0 only.

```mermaid
flowchart TD
    subgraph Parallel["Parallel Jobs"]
        subgraph vLLM["vLLM (CUDA 12.9 & 13.0)"]
            V1[Build amd64] --> V2[Run tests]
            V3[Build arm64]
        end

        subgraph SGLang["SGLang (CUDA 12.9 & 13.0)"]
            S1[Build amd64] --> S2[Run tests]
            S3[Build arm64]
        end

        subgraph TRT["TRT-LLM (CUDA 13.0)"]
            T1[Build amd64] --> T2[Run tests]
            T3[Build arm64]
        end

        subgraph Operator["Operator"]
            O1[Lint & Test]
            O2[Build container]
        end
        
        subgraph Frontend["Frontend"]
            F1[Build amd64]
            F2[Build arm64]
        end
    end

    V2 & V3 & S2 & S3 & T2 & T3 --> Check[backend-status-check]
    Check --> Deploy[Deploy Operator]
    Deploy --> DT[Deployment Tests]

    style Check fill:#1f6feb,color:#fff
```

### Deployment Tests

After backend builds complete, operator deployment tests run on Kubernetes:

- **Deploy Operator**: Installs Dynamo operator on AKS cluster
- **Deploy Tests**: Tests vLLM, SGLang, TRT-LLM deployments with profiles:
  - Aggregated (agg)
  - Aggregated with router (agg_router)  
  - Disaggregated (disagg)
  - Disaggregated with KV router (disagg_router)
- **Cleanup**: Removes deployments and namespace

---

## Path Filters (changed-files action)

The `changed-files` custom action determines which jobs run:

| Filter | Used By | Paths |
|--------|---------|-------|
| `core` | All backend builds | `components/**`, `lib/**`, `tests/**`, `container/**`, `*.py`, `*.rs` |
| `vllm` | vLLM builds & GitLab CI | `container/Dockerfile.vllm`, `components/src/dynamo/vllm/**`, `container/deps/requirements.vllm.txt` |
| `sglang` | SGLang builds & GitLab CI | `container/Dockerfile.sglang`, `components/src/dynamo/sglang/**` |
| `trtllm` | TRT-LLM builds & GitLab CI | `container/Dockerfile.trtllm`, `components/src/dynamo/trtllm/**`, `container/deps/trtllm/**` |
| `operator` | Operator builds | `deploy/operator/**`, `deploy/helm/**` |
| `deploy` | Deployment tests | `examples/backends/**/deploy/**` |
| `frontend` | Frontend builds | `components/src/dynamo/frontend/**`, `lib/llm/src/**` |

**Logic**: Backend jobs run if `core == true` OR framework-specific changes detected.

---

## Post-Merge

After merge to `main` or `release/*`, the `post-merge-ci.yml` workflow triggers automatically.

```mermaid
flowchart TD
    subgraph Trigger["PR Merged to main/release"]
        A[Push to main or release/*]
    end

    subgraph PostMerge["Post-Merge CI Pipeline"]
        B[ci-test-suite.yml]
    end

    subgraph Tests["Full Test Suite"]
        C[All Framework Builds]
        D[Unit Tests]
        E[Integration Tests]
        F[E2E Tests]
        G[Deployment Tests]
    end

    subgraph Notification
        H[Slack Notification]
    end

    A --> B
    B --> Tests
    Tests --> H
```

### Post-Merge vs PR

| Aspect | PR (pull-request/N) | Post-Merge |
|--------|---------------------|------------|
| Workflow | `pr.yaml` | `post-merge-ci.yml` → `ci-test-suite.yml` |
| Trigger | Push to `pull-request/N` branch | Push to `main`/`release/*` |
| Backend builds | Only changed frameworks | All frameworks (vLLM, SGLang, TRT-LLM) |
| CUDA versions | Both 12.9 and 13.0 | Both 12.9 and 13.0 |
| Test scope | Pre-merge marks only | Post-merge + nightly marks |
| Deployment tests | On main or manual trigger | Always |
| Rust checks | On `*.rs` changes only | Always |
| Docs link check | Offline mode | Full external links |
| Slack notifications | No | Yes |

---

## Related

- [README](./README.md) - Workflow details and configuration
- [Nightly Workflow](./NIGHTLY_WORKFLOW.md) - Scheduled builds
- [Troubleshooting](./TROUBLESHOOTING.md) - Common CI issues
