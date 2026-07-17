# Proposed Sidebar Restructure

**Goal:** Address QA feedback that docs mix high-level guides with component-specific implementation details, with no defined path from Concepts to Quick Start to Reference.

**Design principles:**
1. **Actionable vs knowledge bank** -- clear separation between docs that help you *do* something (guides, tutorials, config references) and docs that help you *understand* something (concepts, architecture, components).
2. **Talk flow for knowledge bank** -- Architecture Overview (big picture tl;dr) -> Concepts (deep dives) -> Components (the actual pieces).
3. **Two guide tracks** -- CLI Guide for local usage, Kubernetes Guide for K8s deployment/ops.
4. **Features as a first-class section** -- cross-cutting capabilities consolidated in one place rather than scattered across User Guides.

---

## Proposed Navigation

```
Getting Started
  ├── Introduction                        ← getting-started/introduction.md
  ├── CLI Quickstart                      ← getting-started/quickstart.md
  └── K8s Quickstart                      ← kubernetes/README.md

CLI Guide                                 ← NEW section (actionable, local usage)
  ├── Local Installation                  ← getting-started/local-installation.md
  ├── Disaggregated Serving               ← features/disaggregated-serving/README.md
  ├── KV Cache Aware Routing              ← components/router/router-guide.md
  └── KV Cache Offloading                 ← components/kvbm/kvbm-guide.md

Kubernetes Guide                          ← replaces "Kubernetes Deployment" (actionable, K8s ops)
  ├── Installation Guide                  ← kubernetes/installation-guide.md
  ├── Model Deployment Guide              ← kubernetes/model-deployment-guide.md
  ├── DGDR Reference                      ← kubernetes/dgdr.md
  ├── Observability
  │   ├── Metrics                         ← kubernetes/observability/metrics.md
  │   ├── Logging                         ← kubernetes/observability/logging.md
  │   └── Operator Metrics                ← kubernetes/observability/operator-metrics.md
  ├── Multinode
  │   ├── Multinode Deployments           ← kubernetes/deployment/multinode-deployment.md
  │   ├── Grove                           ← kubernetes/grove.md
  │   └── Topology Aware Scheduling       ← kubernetes/topology-aware-scheduling.md
  └── Cloud Providers
      ├── AWS
      │   ├── EKS Setup                   ← kubernetes/cloud-providers/eks/eks.md
      │   ├── EFS                         ← kubernetes/cloud-providers/eks/efs.md
      │   └── ECS                         ← kubernetes/cloud-providers/ecs/ecs.md
      ├── Azure
      │   ├── AKS Setup                   ← kubernetes/cloud-providers/aks/aks.md
      │   ├── RDMA / InfiniBand           ← kubernetes/cloud-providers/aks/rdma-infiniband.md
      │   ├── AKS Storage                 ← kubernetes/cloud-providers/aks/storage.md
      │   ├── Azure Lustre CSI            ← kubernetes/cloud-providers/aks/azure-lustre-csi.md
      │   └── Spot VMs                    ← kubernetes/cloud-providers/aks/spot-vms.md
      └── GCP
          └── GKE Setup                   ← kubernetes/cloud-providers/gke/gke.md

Architecture Overview                     ← design-docs/architecture.md (standalone page, big-picture tl;dr)

Concepts                                  ← replaces "Design Docs" (knowledge bank)
  ├── Architecture Flow                   ← design-docs/dynamo-flow.md
  ├── Disaggregated Serving               ← design-docs/disagg-serving.md
  ├── Distributed Runtime                 ← design-docs/distributed-runtime.md
  └── Communication Planes
      ├── Discovery Plane                 ← design-docs/discovery-plane.md
      ├── Request Plane                   ← design-docs/request-plane.md
      └── Event Plane                     ← design-docs/event-plane.md

Components                                ← knowledge bank, the actual software pieces
  ├── Operator
  │   ├── Dynamo Operator                 ← kubernetes/dynamo-operator.md
  │   ├── Service Discovery              ← kubernetes/service-discovery.md
  │   ├── Webhooks                        ← kubernetes/webhooks.md
  │   ├── Managing Models (DynamoModel)   ← kubernetes/deployment/dynamomodel-guide.md
  │   ├── Autoscaling                     ← kubernetes/autoscaling.md
  │   ├── Rolling Update                  ← kubernetes/rolling-update.md
  │   ├── Inference Gateway (GAIE)        ← kubernetes/inference-gateway.md
  │   ├── Snapshot                        ← kubernetes/snapshot.md
  │   └── Disagg Communication            ← kubernetes/disagg-communication-guide.md
  ├── Frontend
  │   ├── (landing)                       ← components/frontend/README.md
  │   ├── Frontend Guide                  ← components/frontend/frontend-guide.md
  │   └── Tokenizer                       ← components/frontend/Tokenizer.md
  ├── Router
  │   ├── (landing)                       ← components/router/README.md
  │   ├── Routing Concepts                ← components/router/router-concepts.md
  │   ├── Configuration and Tuning        ← components/router/router-configuration.md
  │   ├── Disaggregated Serving           ← components/router/router-disaggregated-serving.md
  │   ├── Router Operations               ← components/router/router-operations.md
  │   ├── Router Examples                 ← components/router/router-examples.md
  │   ├── Standalone Indexer              ← components/router/standalone-indexer.md
  │   ├── KV Event Replay Comparison      ← components/router/kv-event-replay-comparison.md
  │   └── Router Design                   ← design-docs/router-design.md  (moved from Design Docs)
  ├── Planner
  │   ├── (landing)                       ← components/planner/README.md
  │   ├── Planner Guide                   ← components/planner/planner-guide.md
  │   ├── Planner Examples                ← components/planner/planner-examples.md
  │   └── Planner Design                  ← design-docs/planner-design.md  (moved from Design Docs)
  ├── Profiler
  │   ├── (landing)                       ← components/profiler/README.md
  │   ├── Profiler Guide                  ← components/profiler/profiler-guide.md
  │   └── Profiler Examples               ← components/profiler/profiler-examples.md
  └── KVBM
      ├── (landing)                       ← components/kvbm/README.md
      └── KVBM Design                     ← design-docs/kvbm-design.md  (moved from Design Docs)

Features                                  ← NEW section (cross-cutting capabilities)
  ├── Benchmarking
  │   ├── Dynamo Benchmarking             ← benchmarks/benchmarking.md
  │   ├── Mocker Offline Trace Replay     ← benchmarks/mocker-trace-replay.md
  │   └── Planner Replay Benchmarking     ← benchmarks/planner-replay-benchmarking.md
  ├── Observability (Local)
  │   ├── (landing)                       ← observability/README.md
  │   ├── Prometheus + Grafana Setup      ← observability/prometheus-grafana.md
  │   ├── Metrics                         ← observability/metrics.md
  │   ├── Metrics Developer Guide         ← observability/metrics-developer-guide.md
  │   ├── Health Checks                   ← observability/health-checks.md
  │   ├── Tracing                         ← observability/tracing.md
  │   └── Logging                         ← observability/logging.md
  ├── Fault Tolerance
  │   ├── (landing)                       ← fault-tolerance/README.md
  │   ├── Request Migration               ← fault-tolerance/request-migration.md
  │   ├── Request Cancellation            ← fault-tolerance/request-cancellation.md
  │   ├── Request Rejection               ← fault-tolerance/request-rejection.md
  │   ├── Graceful Shutdown               ← fault-tolerance/graceful-shutdown.md
  │   └── Testing                         ← fault-tolerance/testing.md
  ├── Chat Processor Options              ← agents/chat-processor-options.md
  ├── Tool Calling                        ← agents/tool-calling.md
  ├── Reasoning                           ← agents/reasoning.md
  └── LoRA Adapters                       ← features/lora/README.md

Backends
  ├── SGLang                              ← backends/sglang/ (same subpages as current)
  ├── TensorRT-LLM                        ← backends/trtllm/ (same subpages as current)
  └── vLLM                               ← backends/vllm/README.md

Applications                              ← NEW section (use-case-oriented)
  ├── Multimodal
  │   ├── (landing)                       ← features/multimodal/README.md
  │   ├── Embedding Cache                 ← features/multimodal/embedding-cache.md
  │   ├── Encoder Disaggregation          ← features/multimodal/encoder-disaggregation.md
  │   └── Multimodal KV Routing           ← features/multimodal/multimodal-kv-routing.md
  ├── Diffusion (Preview)
  │   ├── (landing)                       ← features/diffusion/README.md
  │   ├── FastVideo                       ← features/diffusion/fastvideo.md
  │   ├── vLLM-Omni                       ← backends/vllm/vllm-omni.md
  │   ├── SGLang Diffusion                ← backends/sglang/sglang-diffusion.md
  │   └── TRT-LLM Diffusion              ← backends/trtllm/trtllm-diffusion.md
  └── Agents
      ├── (landing)                       ← features/agentic_workloads.md
      └── SGLang for Agentic Workloads    ← backends/sglang/agents.md

Integrations
  ├── LMCache                             ← integrations/lmcache-integration.md
  ├── FlexKV                              ← integrations/flexkv-integration.md
  └── KV Events for Custom Engines        ← integrations/kv-events-custom-engines.md

Developer's Guide                         ← NEW section (contributing & local dev tooling)
  ├── Contributing                        ← contribution-guide.md
  ├── Building from Source                ← getting-started/building-from-source.md
  ├── Writing Python Workers              ← development/backend-guide.md
  ├── Minikube Setup                      ← kubernetes/deployment/minikube.md
  └── Developing with Tilt                ← kubernetes/tilt-dev-setup.md

Resources
  ├── Support Matrix                      ← reference/support-matrix.md
  ├── Feature Matrix                      ← reference/feature-matrix.md
  ├── Release Artifacts                   ← reference/release-artifacts.md
  ├── Examples                            ← getting-started/examples.md
  └── Glossary                            ← reference/glossary.md

Blog (collapsed)                          ← same as current

Documentation
  └── Dynamo Docs Guide                   ← README.md

Hidden Pages                              ← same as current (accessible by URL, not in sidebar)
```

---

## What changed from current sidebar

| Current Section | What happens to it |
|---|---|
| **Getting Started** | Slimmed to intro + 2 quickstarts. Local Installation -> CLI Guide. Building from Source -> Developer's Guide. Contribution Guide -> Developer's Guide. |
| **Resources** | Unchanged |
| **Kubernetes Deployment** | Renamed to "Kubernetes Guide". Dev tools (Minikube/Tilt) moved to Developer's Guide. |
| **User Guides** | **Eliminated.** Contents distributed: disagg/routing/offloading -> CLI Guide, benchmarking/observability/fault-tolerance/chat/tool-calling/reasoning/LoRA -> Features, multimodal/diffusion/agents -> Applications, writing python workers -> Developer's Guide. |
| **Backends** | Unchanged |
| **Components** | Mostly unchanged. `router-guide` and `kvbm-guide` moved to CLI Guide (actionable how-to). Component design docs (`router-design`, `kvbm-design`, `planner-design`) moved here from Design Docs. |
| **Integrations** | Unchanged |
| **Design Docs** | **Eliminated.** `architecture.md` -> standalone Architecture Overview page. `dynamo-flow`, `disagg-serving`, `distributed-runtime`, communication planes -> Concepts. Component design docs -> their respective Components sections. |
| **Blog** | Unchanged |
| **Documentation** | Unchanged |

## New sections

| Section | Purpose |
|---|---|
| **CLI Guide** | Actionable local usage: installation + core feature how-to guides |
| **Architecture Overview** | Standalone big-picture page -- tl;dr of how Dynamo works, just enough concepts to understand the overview |
| **Concepts** | Deep dives into the ideas/theory behind Dynamo (disagg serving, distributed runtime, communication planes) |
| **Features** | Cross-cutting capabilities: benchmarking, observability, fault tolerance, chat processor, tool calling, reasoning, LoRA |
| **Applications** | Use-case-oriented: multimodal, diffusion, agents |
| **Developer's Guide** | Contributing, building from source, python workers, local K8s dev tooling |

## Pages that need deduplication

- `router-guide.md` -- currently in both User Guides and Components > Router. Proposed: CLI Guide only (actionable), removed from Components > Router.
- `kvbm-guide.md` -- currently in both User Guides and Components > KVBM. Proposed: CLI Guide only (actionable), removed from Components > KVBM.
- `sglang-diffusion.md` -- currently in both Backends > SGLang and User Guides > Diffusion. Proposed: keep in both Backends > SGLang and Applications > Diffusion (different navigation paths to same content).
- `agents.md` (SGLang) -- currently in both Backends > SGLang and User Guides > Agents. Proposed: keep in both.

## Landing pages that need writing (future work, not this PR)

- [ ] Concepts landing page
- [ ] Components landing page
- [ ] CLI Guide landing page
- [ ] Features landing page
- [ ] Applications landing page
- [ ] Developer's Guide landing page
- [ ] Architecture Overview expansion (current `architecture.md` may need a tl;dr rewrite)
