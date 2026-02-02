# UX Analysis: DGDR as Primary Kubernetes Entrypoint

## Executive Summary

The proposal to make **DynamoGraphDeploymentRequest (DGDR)** the primary entrypoint for Dynamo on Kubernetes has strong UX merit but requires careful phasing. The current experience forces users to understand low-level DGD (DynamoGraphDeployment) YAML before they can deploy anything, which is a significant onboarding barrier. DGDR abstracts this away behind SLA-driven intent, but the current DGDR implementation has gaps that need closing before it can serve as the sole entrypoint.

## Current UX Pain Points

### 1. Two Entrypoints, Unclear Hierarchy

Today, the Kubernetes README (`docs/kubernetes/README.md`) presents DGD first (Step 3: "Deploy Your First Model" uses raw `agg.yaml`), then introduces DGDR as a secondary concept under "Understanding Dynamo's Custom Resources." The framing is:

- **DGDR**: "recommended approach for generating optimal configurations"
- **DGD**: "lower-level interface... use when you need fine-grained control"

But the getting-started path uses DGD directly. This sends a mixed signal: the docs recommend DGDR but teach DGD first. Users who follow the happy path never encounter DGDR until they're already comfortable with DGD.

### 2. DGDR Requires Profiling Infrastructure

DGDR's value proposition is SLA-driven automation, but it requires:
- kube-prometheus-stack installed and running
- GPU resources available for profiling (even AIC mode needs the profiling job)
- Pre-existing familiarity with SLA concepts (TTFT, ITL, ISL, OSL)

For a first-time user who just wants to serve a model, this is a steep prerequisites cliff compared to `kubectl apply -f agg.yaml`.

### 3. DGD YAML is Verbose but Learnable

The current DGD YAML (e.g., `examples/backends/vllm/deploy/agg.yaml`) is ~30-50 lines and maps directly to Kubernetes concepts (services, replicas, images, commands). Platform engineers familiar with Kubernetes can read and modify it quickly. DGDR hides this but adds its own complexity (profiling configs, SLA tuning, configMapRef patterns).

### 4. Documentation Fragmentation

Planner documentation lives in `docs/planner/` with 4 files mixing operational and design content. The SLA quickstart is 500+ lines and covers DGDR end-to-end, but it's nested under "planner" rather than "kubernetes" or "getting-started." A user searching for "how to deploy on Kubernetes" won't find it.

## Analysis: Should DGDR Be the Entrypoint?

### Arguments For

| Factor | Assessment |
|--------|-----------|
| **Declarative intent** | DGDR lets users express *what* they want (model + SLA) instead of *how* to deploy it. This is the right abstraction level for most users. |
| **Automated optimization** | Profiling + auto-configuration eliminates manual TP/PP/replica tuning. Users consistently underperform here. |
| **Consistency** | One CR type means one mental model, one set of docs, one troubleshooting path. |
| **Progressive disclosure** | DGDR -> generates DGD -> users can inspect/modify. The abstraction layers compose well. |

### Arguments Against

| Factor | Assessment |
|--------|-----------|
| **Prerequisites cliff** | Prometheus + profiling infrastructure is non-trivial. Quick-start users will bounce. |
| **Not all use cases need SLA** | Dev/test deployments, demos, CI pipelines - these want `kubectl apply` and go. |
| **Immutability constraint** | DGDRs are immutable (delete + recreate to change). DGDs can be patched in-place. This is worse for iteration. |
| **Debugging opacity** | When DGDR fails, users must trace through DGDR controller -> profiling job -> DGD generation -> component deployment. Multiple indirection layers. |
| **AIC limitations** | AI Configurator doesn't support all model/GPU combinations. Users hit invisible walls. |

### Recommendation: Tiered Entrypoint

Make DGDR the **recommended** entrypoint but preserve DGD as a **first-class alternative**, not a fallback. Restructure the onboarding flow:

```
                    ┌──────────────────────┐
                    │   "I want to deploy   │
                    │    a model on K8s"    │
                    └─────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │  Do you have SLA      │
                    │  requirements?        │
                    └──┬───────────────┬───┘
                       │               │
                   YES │               │ NO / UNSURE
                       │               │
              ┌────────▼──────┐  ┌─────▼──────────┐
              │ DGDR Path     │  │ DGD Path        │
              │ (SLA-driven)  │  │ (Direct deploy) │
              │ Full profiling│  │ kubectl apply   │
              │ Auto-optimize │  │ Quick start     │
              └───────────────┘  └────────────────┘
```

## Specific UX Fixes

### Fix 1: Restructure `docs/kubernetes/README.md`

**Current**: Platform install -> Choose backend -> Deploy DGD -> (later) learn about DGDR

**Proposed**:
1. Platform install (unchanged)
2. **Quick Deploy** (DGD path, 3 commands, under 2 minutes)
3. **Production Deploy** (DGDR path, SLA-driven, links to full guide)
4. Understanding Custom Resources (reference, not tutorial)

This preserves the quick win while elevating DGDR for production use.

### Fix 2: Add a "Simple DGDR" Template

The current DGDR examples all require profiling configs. Create a minimal DGDR that uses sensible defaults:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B
  backend: vllm
  profilingConfig:
    profilerImage: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1
    config:
      sla:
        isl: 2048
        osl: 256
        ttft: 500
        itl: 50
      sweep:
        useAiConfigurator: true
        aicSystem: h100_sxm
  autoApply: true
```

This is still ~20 lines but expresses pure intent. Ship this as `examples/quickstart/dgdr-quickstart.yaml`.

### Fix 3: Create a "DGDR vs DGD" Decision Guide

Currently buried in prose. Make it a standalone page at `docs/kubernetes/dgdr-vs-dgd.md`:

| Scenario | Use DGDR | Use DGD |
|----------|:--------:|:-------:|
| Production deployment with SLA targets | X | |
| Quick dev/test deployment | | X |
| CI/CD pipeline | | X |
| First time trying Dynamo | | X |
| Optimizing GPU utilization | X | |
| Custom pipeline topology | | X |
| Iterative tuning | | X |
| Auto-scaling with SLA planner | X | |

### Fix 4: Surface DGDR Status UX

The DGDR status lifecycle (Pending -> Profiling -> Deploying -> Ready -> Failed) is good but not discoverable. Add:
- `kubectl` plugin or alias: `kubectl dynamo status`
- Better event messages during Profiling phase (progress %, ETA)
- Link to Grafana dashboard in DGDR status conditions

### Fix 5: Fix Planner Documentation Location

The SLA Planner quickstart (`docs/planner/sla_planner_quickstart.md`) is actually a DGDR deployment guide. It should be:
- **Linked from** `docs/kubernetes/README.md` as the production deployment path
- **Cross-referenced** from `docs/planner/` (which should focus on planner internals)
- The planner docs should separate "how to use the planner" (Tier 2) from "how the planner works" (Tier 3)

## Impact Assessment

| Change | Effort | User Impact |
|--------|--------|-------------|
| Restructure K8s README | Low | High - fixes first impression |
| Simple DGDR template | Low | Medium - reduces friction |
| Decision guide page | Low | Medium - reduces confusion |
| DGDR status UX | Medium | Medium - improves ongoing experience |
| Planner doc restructure | Medium | High - fixes discoverability |

## Conclusion

DGDR should be positioned as the **production-recommended** entrypoint, not the **only** entrypoint. The DGD path serves a real need for quick starts, development, and advanced users. The biggest UX wins come from restructuring the docs to present both paths clearly, rather than from changing the CRD architecture itself.

The Planner documentation restructure (Task 4 in this branch) demonstrates the template for fixing component docs across the board.
