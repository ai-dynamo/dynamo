# Draft GitHub Issue: Refactor K8s Documentation Structure and Content

**Title:** docs: refactor Kubernetes and update to include v1beta1 DGDR

**Labels:** `documentation`, `kubernetes`

---

## Summary

Nvidia QA provided several points of feedback on DGDR and docs for that and K8s:

> - **DGDR is effectively non-functional for most users.** The deploy-by-intent path supports only 6 GPU SKUs, all SXM variants plus L40S, with no published list, PCIe variants are excluded, blocking most cloud and colocation deployments. Beyond SKU restrictions, the AIC profiler crashes with a NaN handling bug in pareto_analysis.py, causing DGDR to reach Failed status after 4 retries on both minimal and explicit hardware configs. The supported SKU list should be published in DGDR docs, PCIe variants added, and the NaN bug fixed before GA.
> - **Documentation lacks a clear structure and navigation hierarchy.** The docs/ directory continues to mix high-level guides with component-specific implementation details, with no defined path from Concepts to Quick Start to Reference. Version mismatches across the container (1.0.0), PyPI (1.0.1), and GitHub README (sglang-runtime:1.0.1) add to the confusion. Critical configuration options remain explained in prose across multiple pages rather than consolidated in reference tables.
>
> **Needs improvement:**
>

> DGDR only supports 6 GPU SKUs, all SXM variants plus L40S. PCIe GPUs (H100-PCIe, A100-PCIe, A30, L4) are excluded, blocking DGDR for most cloud and colocation users. The supported list is only discoverable by hitting validation errors. The docs example uses a format (H100-SXM5-80GB) that doesn’t match what the webhook accepts (h100_sxm). 
> **Recommended Fix:**
> - Publish the supported GPU SKU list in the DGDR documentation
> - Add PCIe variants (h100_pcie, a100_pcie, l4, etc.)
> - Fix the docs example to use a valid value (e.g., h100_sxm not H100-SXM5-80GB)
> - DGDR hands-off deployment: Non-functional. GPU auto-discovery succeeds (H200 SXM correctly detected from node labels), and the profiling job starts running Pareto analysis across backends (vLLM, TRT-LLM) with multiple parallelism configs. However, the AIC profiler crashes with KeyError: "None of [Index([nan, nan, nan, nan]...)]" in pareto_analysis.py:510 — a NaN handling bug in the rapid search ranking. After 4 retries the DGDR reaches Failed status. Tested with both minimal spec and explicit hardware config (gpuSku: h200_sxm).


Our documentation lacks a clear structure and navigation hierarchy. For K8s docs, they should provide a comprehensive walkthrough of how to deploy Dynamo on Kubernetes from start to finish, with clear sections for prerequisites, installation, deployment, and troubleshooting. It should also include possible pitfalls or optional sections like provider specific guides, model caching, etc.

For the installation, it cover how to install Dynamo and prereqs like GPU operator. The installation docs should also cover optional things to install like Grove/KAI Scheduler, Prometheus, and how the values should be configured for Dynamo if you are installing those. It should also point out if you want to use model caching, you can set up a storage solution. It should also mention setting up RDMA/InfiniBand and when you should do so.

In deployment/quickstart, it should walk you through how to deploy a model using DGDR as the primary entrypoint. It should cover the two searchStrategy/profiling modes. It should point out that you should use model caching for a large model or one with a lot of pods bc of huggingface rate limits and how to do so. It should explain autoApply and how to use it. It should also cover the planner in the DGDR spec as an option to enable. We should explain that DGDR is the recommended way, but DGD is an option for deploying recipes that are hand crafted for specific models/hardware configs. They may be more optimal but don't generalize across different hardware configs and require understanding of what params are appropriate.

## Motivation

The current K8s docs are spread across `docs/kubernetes/`, `docs/getting-started/`, `examples/deployments/`, `docs/qwen-dgdr-demo-flow/`, and `recipes/`. A user trying to go from zero to a working K8s deployment has to piece together information from multiple pages, guess at prerequisites, and work around version mismatches.

The core problem is that **the docs read like an encyclopedia rather than a walkthrough.** Individual topics (model caching, RDMA, Grove, GPU Operator) each have their own page, but nothing ties them together into a coherent deployment flow. Users don't know they need model caching until they hit HuggingFace rate limits 30 minutes into a deployment. They don't know they need RDMA until NIXL fails. They don't know they need Grove until multinode scheduling breaks. Every one of these is a predictable requirement for common use cases — we already know users will hit them, but we make them discover it through errors instead of telling them upfront.

A good example of what we should aim for is [Cluster API's quickstart](https://cluster-api.sigs.k8s.io/user/quick-start), which provides a single linear flow from prerequisites to a working cluster. Cloud-provider-specific steps are interwoven at the relevant decision points using inline tabs (e.g., AWS/Azure/GCP tabs appear right where you need the provider-specific command) rather than relegated to separate pages. Prerequisites and optional steps are clearly demarcated with callouts, and the user never has to leave the page to figure out what to do next.

Key pain points with our current docs:

1. **No single K8s deployment path** — Users must jump between `docs/getting-started/quickstart.md`, `docs/kubernetes/installation-guide.md`, `docs/kubernetes/deployment/create-deployment.md`, and cloud-provider READMEs in `examples/deployments/` to assemble a working flow.

2. **Hidden prerequisites surface only as errors** — GPU Operator, RDMA/InfiniBand, Grove, model caching, and storage setup are all documented somewhere, but none are surfaced in the deployment flow at the point where they become relevant. For example:
   - GPU Operator installation is mentioned in `examples/deployments/AKS/AKS-deployment.md`, `deploy/pre-deployment/install-gpu-operator-rdma.sh`, and `docs/qwen-dgdr-demo-flow/deepseek-r1-sglang-disagg-deployment-guide.md` but not in the main K8s quickstart
   - Model caching (`docs/kubernetes/model-caching.md`) is a separate page never linked from the deployment flow — users discover they need it when HuggingFace rate-limits them or a 235B model download takes hours per pod
   - RDMA setup is only in cloud-specific docs and demo notes — users find out they need it when DGDR/NIXL fails
   - Grove is documented in `docs/kubernetes/grove.md` but the deployment guide doesn't tell you it's required for multinode

3. **Helm install commands are inconsistent** — `docs/kubernetes/installation-guide.md` uses tarball install, `docs/reference/release-artifacts.md` uses OCI registry, and `examples/deployments/EKS/README.md` uses yet another variant. Users don't know which is canonical.

4. **Grove/KAI Scheduler guidance is unclear** — `deploy/helm/charts/platform/values.yaml` has both `grove.install` and `kai-scheduler.install` defaulting to `false`, `deploy/pre-deployment/install-dynamo-platform.sh` sets both to `true`, and `docs/kubernetes/installation-guide.md` warns to install separately in production. Users need a clear decision tree: when do you need Grove? When do you need KAI Scheduler?

3. **DGDR is not a first-class entrypoint** — Disaggregated serving (DGDR) is buried under `docs/features/disaggregated-serving/` and demo notes in `docs/qwen-dgdr-demo-flow/`. DGDR should be the primary deployment path in the K8s docs, not an advanced feature footnote.

4. **Cloud provider docs are in examples/, not rendered on Fern** — AKS (`examples/deployments/AKS/`), EKS (`examples/deployments/EKS/`), ECS (`examples/deployments/ECS/`), and GKE (`examples/deployments/GKE/`) guides are hidden in `examples/` and not part of the docs site.

5. **Version mismatches** — Container images pin `1.0.0` in recipes, PyPI is at `1.0.1`, and some working manifests reference `sglang-runtime:1.0.1`. This is already noted in `docs/bugs/dgdr-thread.md` but not addressed in user-facing docs.

## Proposed Changes

### Design principle: walkthrough, not encyclopedia

Following the model of [Cluster API's quickstart](https://cluster-api.sigs.k8s.io/user/quick-start), the K8s docs should be a **single linear flow** that a user can follow from top to bottom to get a working deployment. Cloud-provider-specific and optional steps should be interwoven at the relevant decision points using inline tabs or callouts (e.g., AKS/EKS/GKE tabs appear right where you need the provider-specific command), not relegated to separate pages the user has to discover.

Prerequisites and optional components should be surfaced **at the point where they become relevant** in the flow, with clear guidance on _when_ and _why_ you need them — not buried in separate pages that users only find after hitting an error.

### 1. Installation Guide Refactor

Refactor `docs/kubernetes/installation-guide.md` into a comprehensive prerequisites + installation page:

**Cluster prerequisites:**
- Kubernetes cluster requirements (version, node types, GPU nodes)
- GPU Operator installation — consolidate the canonical install from `deploy/pre-deployment/install-gpu-operator-rdma.sh` and cloud-specific docs into one place, with provider-specific tabs (AKS/EKS/GKE) where commands differ
- Helm 3.x

**Dynamo platform install** — ONE canonical helm command:
```bash
helm install dynamo-platform oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform --version <VERSION>
```

**Optional components** surfaced with clear "when do I need this?" guidance:
- **Grove** — required for multinode/DGDR deployments. Explain what it does and helm flag: `--set global.grove.install=true`
- **KAI Scheduler** — what it provides and when to enable it
- **RDMA / InfiniBand / Network Operator** — required for DGDR with NIXL KV cache transfer. Include how to verify it's working. Provider-specific tabs for RDMA setup (AKS InfiniBand, AWS EFA, etc.)
- **Prometheus / observability stack** — optional, with helm values for Dynamo integration
- **Model caching / shared storage** — explain that large models (>70B) or deployments with many replicas will hit HuggingFace rate limits or take hours per pod to download. Show how to set up a shared `ReadWriteMany` PVC (`hf-model-cache` pattern), with provider-specific tabs for storage (EFS for EKS, Azure Files for AKS, Lustre, etc.)

### 2. Deployment Quickstart with DGDR as Primary Entrypoint

Create a deployment quickstart that walks through deploying a model using **DGDR as the recommended path**:

**DGDR deployment flow:**
1. Create a `DynamoGPUDisaggregatedRuntime` CR — explain the spec fields with a working example
2. **Search strategy / profiling modes** — explain the two modes (`rapid` vs `exhaustive`), when to use each, and what the profiling phase does
3. **`autoApply`** — explain what it does (automatically applies the profiler's recommended config) and when you might want `autoApply: false` to review the config first
4. **Planner** — explain the optional planner in the DGDR spec, what it optimizes, and how to enable it
5. **Model caching callout** — at this point in the flow, call out that if you're deploying a large model or one with many pods, you should have set up shared storage in the installation step. Link back to that section.
6. Verify the deployment is running and healthy
7. Send a test request

**DGD as an alternative:**
- Explain that `DynamoGPUDisaggregatedRuntime` (DGDR) is the recommended way to deploy because it automatically profiles and configures your deployment for your hardware
- `DynamoGPUDisaggregated` (DGD) is available for hand-crafted recipes targeting specific model/hardware combos — may be more optimal for known configurations but doesn't generalize across hardware and requires understanding of what parallelism params are appropriate
- Link to existing recipes in `recipes/`

**Pitfalls and need-to-knows** interwoven in the flow (not a separate page):
- Model caching: why it's needed, when to set it up (before deploying large models)
- GPU count and tensor parallelism alignment — how `gpuCount` in CRDs relates to TP and node GPU count
- Supported GPU SKUs for DGDR (currently only 6 SXM variants + L40S — publish the list, note PCIe exclusion per QA feedback)
- GPU SKU format — use valid values like `h100_sxm`, not `H100-SXM5-80GB`
- Version alignment: how to check container image versions match your Helm chart version

### 3. DGDR In-Depth Guide

Beyond the quickstart, add a dedicated DGDR guide page (e.g., `docs/kubernetes/dgdr-guide.md`) that explains how the DGDR system works end-to-end and how its components fit together. The individual components (profiler, planner) already have their own docs pages, but none of them explain how they're used **in the context of DGDR** — users reading `docs/components/profiler/` have no idea how profiling feeds into the DGDR lifecycle.

This page should cover:
- **DGDR lifecycle overview** — what happens when you create a `DynamoGPUDisaggregatedRuntime` CR: how it goes from submitted → profiling → configuring → running, and what each phase does
- **Profiler in DGDR context** — how the profiler is invoked by DGDR, what it profiles (backend × parallelism config combinations), how search strategies (`rapid` vs `exhaustive`) affect the profiling matrix, how results feed into the next phase. Link to the profiler component docs for algorithmic details.
- **Planner in DGDR context** — how the planner takes profiling results and computes an optimal deployment config (prefill/decode split, TP/PP, replica counts). When to enable vs. disable it. Link to planner component docs for SLA optimization details.
- **`autoApply` behavior** — what exactly gets applied, how to inspect the generated config before applying, how to manually apply or override
- **DGDR vs DGD** — detailed comparison: DGDR automates profiling and configuration, DGD takes a hand-crafted config. When each is appropriate, how to go from a DGDR-generated config to a DGD if you want to pin a known-good config.
- **CRD spec reference** — full field-by-field explanation of the `DynamoGPUDisaggregatedRuntime` spec, with valid values (e.g., supported GPU SKUs, search strategies)
- **Monitoring and troubleshooting** — how to check DGDR status, read profiling logs, diagnose common failures (profiler crash, unsupported SKU, etc.)

### 4. Cloud Provider Section


Move cloud provider guides from `examples/deployments/` into `docs/` so they render on the Fern docs site:
- `examples/deployments/AKS/` → `docs/kubernetes/cloud-providers/aks.md`
- `examples/deployments/EKS/` → `docs/kubernetes/cloud-providers/eks.md`
- `examples/deployments/ECS/` → `docs/kubernetes/cloud-providers/ecs.md` (or a separate non-K8s section)
- `examples/deployments/GKE/` → `docs/kubernetes/cloud-providers/gke.md`

In addition to standalone cloud provider pages, provider-specific steps should be **interwoven into the main installation and deployment flow** using tabs (like Cluster API does), so users don't have to context-switch to a separate page for their provider's storage command or RDMA setup.

### 5. Configuration Reference Tables

Consolidate critical K8s configuration options currently spread across prose into reference tables:
- Helm chart values (`deploy/helm/charts/platform/values.yaml`) — document key overrides
- `DynamoGPUDisaggregatedRuntime` CRD fields (with valid values, e.g., supported GPU SKUs)
- `DynamoGPUDisaggregated` CRD fields
- `DynamoDeployment` CRD fields
- Environment variables relevant to K8s deployments

## Out of Scope

- Non-K8s deployment docs (local, Docker Compose)
- Backend engine docs (vLLM, SGLang, TRT-LLM internals)
- Router/planner algorithm docs
- Fixing the AIC profiler NaN bug (separate issue, but should be noted in DGDR docs as a known issue)
- Adding PCIe GPU SKU support to DGDR (separate issue)
- Comprehensive version alignment fix across all artifacts (separate issue)

## References

- NVIDIA QA feedback on docs structure, DGDR functionality, and version mismatches
- [Cluster API quickstart](https://cluster-api.sigs.k8s.io/user/quick-start) — reference for walkthrough-style docs with inline provider tabs
- Demo deployment notes: `docs/qwen-dgdr-demo-flow/`
- Bug notes: `docs/bugs/dgdr-thread.md`, `docs/bugs/attention-dp-gpu-count-bug.md`
- Current K8s docs: `docs/kubernetes/`
- Cloud provider examples: `examples/deployments/{AKS,EKS,ECS,GKE}/`
