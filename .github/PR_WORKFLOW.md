# PR Workflow

## Required Checks

| Check | Trigger |
|-------|---------|
| `pre-commit` | Direct |
| `copyright-checks` | Direct |
| `Build and Test - dynamo` | Direct |
| `backend-status-check` | copy-pr-bot |

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

    subgraph CopyPR["Via copy-pr-bot"]
        H{has_code_changes?}
        J[vLLM build & tests]
        K[SGLang build & tests]
        L[TRT-LLM build & tests]
        M[backend-status-check]
        N[GitLab CI]
    end

    A --> B & C & I & D & E & F & G
    A -.->|copy-pr-bot| H
    H -->|Yes| J & K & L
    J & K & L --> M
    H -.-> N

    style B fill:#1f6feb,color:#fff
    style C fill:#1f6feb,color:#fff
    style I fill:#1f6feb,color:#fff
    style M fill:#1f6feb,color:#fff
    style N fill:#6e7681,color:#fff
```

---

## Core Dynamo Build

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

## Backend Builds

> ⚠️ Only runs via copy-pr-bot (`pull-request/N` branches)

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
    
    style Check fill:#1f6feb,color:#fff
```

---

## copy-pr-bot Flow

```mermaid
flowchart LR
    A[PR Opened] --> B{Maintainer triggers<br/>copy-pr-bot}
    B -->|Comment| C[Bot creates<br/>pull-request/N branch]
    C --> D[Backend builds +<br/>GitLab CI run]
    D --> E[Results reported<br/>to PR]

    style D fill:#1f6feb,color:#fff
```

---

## Post-Merge

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
| Rust checks | On `*.rs` changes | Always |
| Docs link check | Offline mode | Full external |
| Fault tolerance | ❌ | ✅ |
| GitLab CI | Via copy-pr-bot | Always |

---

## Related

- [README](./README.md) - Workflow details
- [Nightly Workflow](./NIGHTLY_WORKFLOW.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
