# Nightly CI Workflow

**Schedule**: Daily at 12:00 AM PST (08:00 UTC)

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph Trigger["⏰ Scheduled Trigger"]
        A["Daily at 12:00 AM PST"]
    end

    subgraph Build["🔨 Build Stage"]
        direction LR
        subgraph AMD64["amd64"]
            B1[vLLM]
            B2[SGLang]
            B3[TRT-LLM]
        end
        subgraph ARM64["arm64"]
            B4[vLLM]
            B5[SGLang]
            B6[TRT-LLM]
        end
    end

    subgraph Test["🧪 Test Stage"]
        T1[Unit Tests]
        T2[Integration Tests]
        T3[E2E Single GPU]
        T4[E2E Multi GPU]
    end

    subgraph Publish["📦 Publish"]
        P1[Push to ECR]
        P2[Push to ACR]
    end

    subgraph Report["📊 Results"]
        R1[Summary]
        R2[Slack Notification]
    end

    A --> Build
    Build --> Test
    Build --> Publish
    Test --> R1 --> R2
```

---

## Build Matrix

```mermaid
flowchart LR
    subgraph Frameworks
        F1[vLLM]
        F2[SGLang]
        F3[TRT-LLM]
    end

    subgraph Architectures
        A1[linux/amd64]
        A2[linux/arm64]
    end

    subgraph Images["6 Runtime Images"]
        I1[nightly-vllm-amd64]
        I2[nightly-vllm-arm64]
        I3[nightly-sglang-amd64]
        I4[nightly-sglang-arm64]
        I5[nightly-trtllm-amd64]
        I6[nightly-trtllm-arm64]
    end

    F1 --> A1 --> I1
    F1 --> A2 --> I2
    F2 --> A1 --> I3
    F2 --> A2 --> I4
    F3 --> A1 --> I5
    F3 --> A2 --> I6
```

---

## Test Matrix

```mermaid
flowchart TD
    subgraph Tests["Test Types"]
        U[Unit Tests]
        I[Integration Tests]
        E1[E2E gpu_1]
        E2[E2E gpu_2]
    end

    subgraph Frameworks["3 Frameworks × 2 Architectures"]
        F[vLLM, SGLang, TRT-LLM]
    end

    subgraph Runners
        R1[gpu-l40-amd64<br/>GPU tests]
        R2[cpu-arm-r8g-4xlarge<br/>ARM64 dry-run]
    end

    Tests --> Frameworks --> Runners
```

### Test Details

| Test Type | Timeout | pytest Markers |
|-----------|---------|----------------|
| Unit | 45 min | `unit and (nightly or post_merge or pre_merge)` |
| Integration | 90 min | `integration and (nightly or post_merge or pre_merge)` |
| E2E Single GPU | 120 min | `{framework} and e2e and gpu_1` |
| E2E Multi GPU | 150 min | `e2e and gpu_2` |

> ARM64 tests run in dry-run mode (collect-only) - no GPU runners available.

---

## Test Dependencies

```mermaid
flowchart LR
    subgraph Builds
        B1[Build vLLM amd64]
        B2[Build vLLM arm64]
    end

    subgraph Tests
        T1[vLLM-amd64-unit]
        T2[vLLM-amd64-integ]
        T3[vLLM-arm64-unit]
    end

    B1 -->|check status| T1
    B1 -->|check status| T2
    B2 -->|check status| T3
```

Tests fail immediately if their build failed (no wasted GPU time).

---

## Image Tags

| Tag Pattern | Example | Purpose |
|-------------|---------|---------|
| `nightly-{framework}-{arch}` | `nightly-vllm-amd64` | Latest nightly |
| `nightly-{framework}-{arch}-run-{id}` | `nightly-vllm-amd64-run-12345` | Specific run |
| `main-{framework}-framework-{arch}` | `main-vllm-framework-amd64` | Layer cache |

---

## Timing

| Stage | Duration |
|-------|----------|
| amd64 Builds | 60-90 min |
| arm64 Builds | 90-120 min |
| Unit Tests | 10-20 min |
| Integration Tests | 30-60 min |
| E2E Tests | 60-90 min |
| **Total** | **3-4 hours** |

---

## Complete Flow

```mermaid
flowchart TB
    subgraph Schedule["⏰ Daily 12:00 AM PST"]
        Cron["cron: 0 8 * * *"]
    end

    subgraph BuildAMD["🔨 Build AMD64"]
        BA1[vLLM] --> BA2[Runtime]
        BA3[SGLang] --> BA4[Runtime]
        BA5[TRT-LLM] --> BA6[Runtime]
    end

    subgraph BuildARM["🔨 Build ARM64"]
        BB1[vLLM] --> BB2[Runtime]
        BB3[SGLang] --> BB4[Runtime]
        BB5[TRT-LLM] --> BB6[Runtime]
    end

    subgraph TestAMD["🧪 AMD64 GPU Tests"]
        TA1[Unit]
        TA2[Integration]
        TA3[E2E gpu_1]
        TA4[E2E gpu_2]
    end

    subgraph TestARM["🧪 ARM64 Dry Run"]
        TB1[Unit - collect only]
        TB2[Integration - collect only]
    end

    subgraph Summary["📊 Summary"]
        S1[Results Summary]
        S2[Artifacts]
    end

    Cron --> BuildAMD & BuildARM
    BuildAMD --> TestAMD
    BuildARM --> TestARM
    TestAMD & TestARM --> Summary
```

---

## Related

- [README](./README.md) - Workflow details
- [PR Workflow](./PR_WORKFLOW.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
