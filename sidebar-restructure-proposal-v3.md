# Proposed Sidebar Restructure (v3)

**Goal:** Address QA feedback that docs mix high-level guides with component-specific implementation details, with no defined path from Concepts to Quick Start to Reference.

**Design principles:**
1. **Two tutorial tracks** -- CLI Usage (run processes locally) and Kubernetes Usage (apply manifests on a cluster). These are the "do stuff" sections.
2. **Knowledge bank** follows a "giving a talk" flow -- Architecture Overview (big picture tl;dr) -> Concepts (deep dives) -> Components (the actual pieces).
3. **Features as knowledge bank** -- cross-cutting capabilities (tool calling, fault tolerance, observability details, etc.) are reference/explanation content, not tutorials. They live in the knowledge bank and get linked from tutorials.
4. **Usage sections are slim** -- each entry is a guide or examples page, not a deep-dive. Get the user to a working deployment, then link out.

---

## Changes from v2

1. **vLLM parity with SGLang/TRT-LLM** — CLI Usage now points to `vllm-examples.md` (not the README). Backends > vLLM gets its reference guide, KV cache offloading, and observability pages back.
2. **prometheus-grafana.md added to CLI Usage > Observability Setup** — was mentioned in v2 design notes but missing from the tree.
3. **agent-context.md placed in Features > Agents** — cross-cutting observability feature for agentic workloads.
4. **mocker.md placed in Developer's Guide** — it's a testing/simulation tool for development, not a user-facing feature.

---

## Proposed Navigation

```
Getting Started
  ├── Introduction                        ← getting-started/introduction.md
  ├── CLI Quickstart                      ← getting-started/quickstart.md
  └── K8s Quickstart                      ← kubernetes/README.md

CLI Usage                                 ← "do stuff locally" (run processes directly)
  ├── Installation                        ← getting-started/local-installation.md
  ├── SGLang Deployments                  ← backends/sglang/sglang-examples.md
  ├── TRT-LLM Deployments                ← backends/trtllm/trtllm-examples.md
  ├── vLLM Deployments                   ← backends/vllm/vllm-examples.md
  └── Observability Setup
      ├── Getting Started                 ← observability/README.md
      └── Prometheus + Grafana            ← observability/prometheus-grafana.md

Kubernetes Usage                          ← "do stuff on a cluster" (kubectl apply, helm install)
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

Concepts                                  ← deep dives into the ideas/theory
  ├── Architecture Flow                   ← design-docs/dynamo-flow.md
  ├── Disaggregated Serving               ← design-docs/disagg-serving.md
  ├── Distributed Runtime                 ← design-docs/distributed-runtime.md
  └── Communication Planes
      ├── Discovery Plane                 ← design-docs/discovery-plane.md
      ├── Request Plane                   ← design-docs/request-plane.md
      └── Event Plane                     ← design-docs/event-plane.md

Components                                ← knowledge bank: how each piece works
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
  │   ├── Router Guide                    ← components/router/router-guide.md
  │   ├── Routing Concepts                ← components/router/router-concepts.md
  │   ├── Configuration and Tuning        ← components/router/router-configuration.md
  │   ├── Disaggregated Serving           ← components/router/router-disaggregated-serving.md
  │   ├── Router Operations               ← components/router/router-operations.md
  │   ├── Router Examples                 ← components/router/router-examples.md
  │   ├── Standalone Indexer              ← components/router/standalone-indexer.md
  │   ├── KV Event Replay Comparison      ← components/router/kv-event-replay-comparison.md
  │   └── Router Design                   ← design-docs/router-design.md
  ├── Planner
  │   ├── (landing)                       ← components/planner/README.md
  │   ├── Planner Guide                   ← components/planner/planner-guide.md
  │   ├── Planner Examples                ← components/planner/planner-examples.md
  │   └── Planner Design                  ← design-docs/planner-design.md
  ├── Profiler
  │   ├── (landing)                       ← components/profiler/README.md
  │   ├── Profiler Guide                  ← components/profiler/profiler-guide.md
  │   └── Profiler Examples               ← components/profiler/profiler-examples.md
  └── KVBM
      ├── (landing)                       ← components/kvbm/README.md
      ├── KVBM Guide                      ← components/kvbm/kvbm-guide.md
      └── KVBM Design                     ← design-docs/kvbm-design.md

Backends                                  ← knowledge bank: backend-specific reference & explanation
  ├── SGLang
  │   ├── (landing)                       ← backends/sglang/README.md
  │   ├── Reference Guide                 ← backends/sglang/sglang-reference-guide.md
  │   ├── Chat Processor                  ← backends/sglang/sglang-chat-processor.md
  │   ├── Disaggregation                  ← backends/sglang/sglang-disaggregation.md
  │   ├── HiCache                         ← backends/sglang/sglang-hicache.md
  │   └── Observability                   ← backends/sglang/sglang-observability.md
  ├── TensorRT-LLM
  │   ├── (landing)                       ← backends/trtllm/README.md
  │   ├── Reference Guide                 ← backends/trtllm/trtllm-reference-guide.md
  │   ├── Observability                   ← backends/trtllm/trtllm-observability.md
  │   └── Known Issues                    ← backends/trtllm/trtllm-known-issues.md
  └── vLLM
      ├── (landing)                       ← backends/vllm/README.md
      ├── Reference Guide                 ← backends/vllm/vllm-reference-guide.md
      ├── KV Cache Offloading             ← backends/vllm/vllm-kv-offloading.md
      └── Observability                   ← backends/vllm/vllm-observability.md

Features                                  ← cross-cutting capabilities (reference/explanation, not tutorial)
  ├── Disaggregated Serving               ← features/disaggregated-serving/README.md
  ├── Agents
  │   ├── Chat Processor Options          ← agents/chat-processor-options.md
  │   ├── Tool Calling                    ← agents/tool-calling.md
  │   ├── Reasoning                       ← agents/reasoning.md
  │   └── Agent Context and Tracing       ← agents/agent-context.md
  ├── LoRA Adapters                       ← features/lora/README.md
  ├── Performance Tuning                  ← performance/tuning.md
  ├── Fault Tolerance
  │   ├── (landing)                       ← fault-tolerance/README.md
  │   ├── Request Migration               ← fault-tolerance/request-migration.md
  │   ├── Request Cancellation            ← fault-tolerance/request-cancellation.md
  │   ├── Request Rejection               ← fault-tolerance/request-rejection.md
  │   ├── Graceful Shutdown               ← fault-tolerance/graceful-shutdown.md
  │   └── Testing                         ← fault-tolerance/testing.md
  ├── Observability Reference
  │   ├── Metrics                         ← observability/metrics.md
  │   ├── Health Checks                   ← observability/health-checks.md
  │   ├── Tracing                         ← observability/tracing.md
  │   └── Logging                         ← observability/logging.md
  └── Benchmarking
      ├── Dynamo Benchmarking             ← benchmarks/benchmarking.md
      ├── Mocker Trace Replay             ← benchmarks/mocker-trace-replay.md
      └── Planner Replay Benchmarking     ← benchmarks/planner-replay-benchmarking.md

Applications                              ← use-case-oriented content
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

Developer's Guide
  ├── Contributing                        ← contribution-guide.md
  ├── Building from Source                ← getting-started/building-from-source.md
  ├── Writing Python Workers              ← development/backend-guide.md
  ├── Metrics Developer Guide             ← observability/metrics-developer-guide.md
  ├── Mocker                              ← mocker/mocker.md
  ├── Minikube Setup                      ← kubernetes/deployment/minikube.md
  └── Developing with Tilt                ← kubernetes/tilt-dev-setup.md

Resources
  ├── Support Matrix                      ← reference/support-matrix.md
  ├── Feature Matrix                      ← reference/feature-matrix.md
  ├── Release Artifacts                   ← reference/release-artifacts.md
  ├── Examples                            ← getting-started/examples.md
  └── Glossary                            ← reference/glossary.md

Blog (collapsed)
  ├── (landing)                           ← blogs/index.mdx
  ├── Full-Stack Optimizations...         ← blogs/agentic-inference/agentic-inference.md
  └── Flash Indexer...                    ← blogs/flash-indexer/flash-indexer.md

Documentation
  └── Dynamo Docs Guide                   ← README.md

Hidden Pages                              ← accessible by URL, not in sidebar
  (same as current index.yml hidden section)
```

---

## Key changes from v1

### CLI Usage replaces "CLI Guide"
v1 tried to make a "CLI Guide" with component how-tos (router guide, KVBM guide, etc.). But locally you don't use components independently -- you launch a frontend + backend worker and everything else is a flag. The local flow is **backend-centric**: pick SGLang/TRT-LLM/vLLM, pick a deployment pattern, go.

v2's CLI Usage is slim: Installation + one examples page per backend + observability setup. That's it. The examples pages (sglang-examples.md, trtllm-examples.md, vllm-examples.md) already walk through every deployment pattern (aggregated, disagg, routing, multimodal, diffusion) with exact commands.

### Backend examples move to CLI Usage, backend knowledge stays in Backends
The examples pages are tutorials (step-by-step commands) -> CLI Usage. The reference guides, deep-dives (disaggregation, HiCache, KV offloading, chat processor), and observability details are knowledge content -> Backends section.

### Observability splits across three sections
- **CLI Usage > Observability Setup** -- the docker-compose quickstart (observability/README.md) and Prometheus+Grafana visualization (prometheus-grafana.md)
- **Kubernetes Usage > Observability** -- K8s-specific metrics, logging, operator metrics
- **Features > Observability Reference** -- the detailed metrics catalog, health checks, tracing, logging config (env var tables, not setup steps)

### Features section is knowledge bank, not tutorials
v1 wasn't clear on this. v2 explicitly: Features contains reference/explanation content (what tool calling is, how fault tolerance works, config tables). Tutorials link to these pages but Features itself isn't a walkthrough.

### Agents content grouped under Features > Agents
Agent-related reference content (chat processors, tool calling, reasoning, agent context/tracing) is grouped together rather than scattered as top-level items in Features.

---

## What changed from current sidebar

| Current Section | What happens |
|---|---|
| **Getting Started** | Slimmed to intro + 2 quickstarts. Local Installation -> CLI Usage. Building from Source -> Developer's Guide. Contribution Guide -> Developer's Guide. |
| **Resources** | Unchanged |
| **Kubernetes Deployment** | Renamed to "Kubernetes Usage". Dev tools (Minikube/Tilt) -> Developer's Guide. |
| **User Guides** | **Eliminated.** Backend examples -> CLI Usage. Feature content -> Features. Multimodal/Diffusion/Agents -> Applications. Writing Python Workers -> Developer's Guide. Benchmarking -> Features. Mocker -> Developer's Guide. Observability (Local) split across CLI Usage + Features. |
| **Backends** | Examples pages moved to CLI Usage. Rest stays as knowledge-bank Backends (vLLM now has full parity with SGLang/TRT-LLM). |
| **Components** | Unchanged (was already knowledge bank). Design docs moved here from Design Docs. |
| **Integrations** | Unchanged |
| **Design Docs** | **Eliminated.** architecture.md -> standalone Architecture Overview. Conceptual docs -> Concepts. Component design docs -> their respective Components sections. |
| **Blog** | Unchanged |
| **Documentation** | Unchanged |

## New sections

| Section | Purpose | Content type |
|---|---|---|
| **CLI Usage** | Deploy and run Dynamo locally | Tutorial (slim: install + examples + observability setup) |
| **Architecture Overview** | Big-picture tl;dr of how Dynamo works | Explanation (standalone page) |
| **Concepts** | Deep dives into theory (disagg, distributed runtime, comm planes) | Explanation |
| **Features** | Cross-cutting capabilities: agents, fault tolerance, observability details, benchmarking, LoRA, perf tuning | Reference + Explanation |
| **Applications** | Use-case domains: multimodal, diffusion, agents | Explanation |
| **Developer's Guide** | Contributing, building, writing workers, mocker, local K8s dev | Tutorial + Reference |

---

## Pages that appear in multiple sections

- `sglang-diffusion.md` — Backends > SGLang (not listed, to keep it slim) AND Applications > Diffusion. Primary location is Applications > Diffusion.
- `backends/sglang/agents.md` — Applications > Agents only (removed from Backends > SGLang to avoid duplication).
- `observability/README.md` — CLI Usage > Observability Setup. The detailed metrics/logging/tracing/health-checks pages are in Features > Observability Reference.

---

## Landing pages that need writing (future work)

- [ ] CLI Usage landing page (or just use quickstart.md as the entry point)
- [ ] Concepts landing page
- [ ] Components landing page
- [ ] Features landing page
- [ ] Applications landing page
- [ ] Developer's Guide landing page
- [ ] Architecture Overview expansion (current architecture.md may need a tl;dr rewrite)
