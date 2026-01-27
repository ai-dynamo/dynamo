# Nightly CI Workflow

**Schedule**: Daily at 12:00 AM PST (08:00 UTC)  
**Workflow**: `nightly-ci.yml` → `ci-test-suite.yml`  
**Runners**: Production self-hosted runners (`prod-builder-*`, `prod-default-v1`)

The nightly pipeline builds all frameworks for both architectures and runs comprehensive test suites. Unlike PR builds, nightly builds all frameworks regardless of what changed.

## Reusable Workflow Architecture

The nightly CI uses the `ci-test-suite.yml` reusable workflow with these parameters:
- `pipeline_type`: `nightly`
- `include_nightly_marks`: `true`
- `image_prefix`: `nightly`
- `enable_slack_notification`: `true`

This same workflow is also used by `post-merge-ci.yml` with different parameters.

---

## Build Stage

Each framework builds for multiple CUDA versions and architectures:

### CUDA Version Support
- **vLLM**: CUDA 12.9 and CUDA 13.0
- **SGLang**: CUDA 12.9 and CUDA 13.0
- **TRT-LLM**: CUDA 13.0 only

Each build produces two images:
- **Framework image**: Build dependencies, used as cache for subsequent builds
- **Runtime image**: Deployable container with all components

```mermaid
flowchart LR
    subgraph Frameworks
        F1[vLLM<br/>CUDA 12.9 & 13]
        F2[SGLang<br/>CUDA 12.9 & 13]
        F3[TRT-LLM<br/>CUDA 13]
    end

    subgraph Architectures
        A1[linux/amd64]
        A2[linux/arm64]
    end

    subgraph Images["Runtime Images"]
        I1[nightly-vllm-cuda12-amd64]
        I1b[nightly-vllm-cuda13-amd64]
        I2[nightly-vllm-cuda12-arm64]
        I2b[nightly-vllm-cuda13-arm64]
        I3[nightly-sglang-cuda12-amd64]
        I3b[nightly-sglang-cuda13-amd64]
        I4[nightly-sglang-cuda12-arm64]
        I4b[nightly-sglang-cuda13-arm64]
        I5[nightly-trtllm-cuda13-amd64]
        I6[nightly-trtllm-cuda13-arm64]
    end

    F1 --> A1 --> I1 & I1b
    F1 --> A2 --> I2 & I2b
    F2 --> A1 --> I3 & I3b
    F2 --> A2 --> I4 & I4b
    F3 --> A1 --> I5
    F3 --> A2 --> I6
```

---

## Test Stage

Tests wait for their corresponding build to complete. If a build fails, tests fail immediately (no wasted GPU time).

```mermaid
flowchart TD
    subgraph Tests["Test Types"]
        U[Unit Tests]
        I[Integration Tests]
        E1[E2E gpu_1]
        E2[E2E gpu_2]
    end

    subgraph Frameworks["3 Frameworks × 2 CUDA × 2 Arch"]
        F[vLLM, SGLang, TRT-LLM]
    end

    subgraph Runners
        R1[prod-builder-amd-gpu-v1<br/>AMD64 GPU tests]
        R2[prod-builder-arm-v1<br/>ARM64 CPU-only]
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

> **Note**: ARM64 tests run in dry-run mode (collect-only) since no GPU runners are available for ARM64.

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

---

## Image Tags

Images are pushed to AWS ECR and Azure ACR with the following tag patterns:

| Tag Pattern | Example | Purpose |
|-------------|---------|---------|
| `nightly-{framework}-cuda{ver}-{arch}` | `nightly-vllm-cuda13-amd64` | Latest nightly by CUDA version |
| `nightly-{framework}-{arch}` | `nightly-vllm-amd64` | Latest nightly (primary CUDA) |
| `nightly-{framework}-cuda{ver}-{arch}-run-{id}` | `nightly-vllm-cuda13-amd64-run-12345` | Specific run by CUDA version |
| `main-{framework}-framework-{arch}` | `main-vllm-framework-amd64` | Layer cache |

**Note**: CUDA version in tags is the major version only (e.g., `cuda12` for CUDA 12.9, `cuda13` for CUDA 13.0).

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

## Deployment Tests

After all builds and tests complete successfully, deployment tests run on Kubernetes (AKS):

| Framework | Profiles Tested |
|-----------|-----------------|
| vLLM | agg, agg_router, disagg, disagg_router |
| SGLang | agg, agg_router, disagg, disagg_router |
| TRT-LLM | agg, agg_router, disagg, disagg_router |

Each test:
1. Deploys Dynamo operator
2. Creates DynamoGraphDeployment
3. Waits for pods to be ready
4. Sends test inference request
5. Validates response
6. Cleans up resources

---

## Complete Flow

```mermaid
flowchart TB
    subgraph Schedule["⏰ Daily 12:00 AM PST"]
        Cron["cron: 0 8 * * *<br/>via ci-test-suite.yml"]
    end

    subgraph BuildAMD["🔨 Build AMD64 (CUDA 12.9 & 13.0)"]
        BA1[vLLM] --> BA2[Runtime]
        BA3[SGLang] --> BA4[Runtime]
        BA5[TRT-LLM] --> BA6[Runtime]
    end

    subgraph BuildARM["🔨 Build ARM64 (CUDA 12.9 & 13.0)"]
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

    subgraph TestARM["🧪 ARM64 CPU-only"]
        TB1[Unit - collect only]
        TB2[Integration - collect only]
    end

    subgraph Deploy["🚀 Deployment Tests"]
        D1[Deploy Operator]
        D2[Test All Profiles]
        D3[Cleanup]
    end

    subgraph Summary["📊 Summary"]
        S1[Slack Notification]
        S2[Artifacts]
    end

    Cron --> BuildAMD & BuildARM
    BuildAMD --> TestAMD
    BuildARM --> TestARM
    TestAMD & TestARM --> Deploy
    Deploy --> Summary
```

---

## Related

- [README](./README.md) - Workflow details and configuration
- [PR Workflow](./PR_WORKFLOW.md) - Pull request CI
- [Troubleshooting](./TROUBLESHOOTING.md) - Common CI issues
