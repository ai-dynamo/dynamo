# Dynamo Kubernetes Stack - Quick Reference

A one-page visual guide for understanding the Dynamo architecture.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│     YOU WRITE THIS:                      DYNAMO CREATES THIS:               │
│                                                                              │
│     ┌─────────────────┐                 ┌─────────────────────────────────┐ │
│     │ DynamoGraph     │                 │                                 │ │
│     │ Deployment      │   ──────────▶   │  Frontend Pod(s)                │ │
│     │                 │                 │       │                         │ │
│     │ services:       │                 │       ▼                         │ │
│     │   Frontend:     │                 │  Prefill Worker Pod(s)          │ │
│     │     replicas: 1 │                 │       │                         │ │
│     │   PrefillWorker:│                 │       ▼  KV-cache               │ │
│     │     replicas: 2 │                 │  Decode Worker Pod(s)           │ │
│     │   DecodeWorker: │                 │       │                         │ │
│     │     replicas: 4 │                 │       ▼                         │ │
│     └─────────────────┘                 │  + Services, RBAC, etc.         │ │
│                                         │                                 │ │
│                                         └─────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What Gets Created

```
DynamoGraphDeployment (DGD)          <-- You create this
    │
    ├── DynamoComponentDeployment    <-- Operator creates these
    │   └── Deployment/PodClique
    │       └── Pod (Frontend)
    │           └── Service
    │
    ├── DynamoComponentDeployment
    │   └── Deployment/PodClique
    │       └── Pod(s) (Prefill)
    │           └── Service
    │
    └── DynamoComponentDeployment
        └── Deployment/PodClique
            └── Pod(s) (Decode)
                └── Service
```

---

## Three Serving Patterns

### 1. Aggregated (Simple)
```
Client → Frontend → [Worker] [Worker] [Worker] → Response
                     ↑ Each worker does full inference
```

### 2. Disaggregated + KV-Routing (Production)
```
Client → Frontend → Router → Prefill ──KV──→ Decode → Response
                              ↑                  ↑
                        Processes prompt    Generates tokens
```

### 3. With Planner (Auto-scaling)
```
Client → Frontend → Planner monitors ──────→ Scales Workers
              │     latency/QPS              based on SLA
              └──→ Prefill/Decode Workers
```

---

## Core Components at a Glance

| Component | What It Does |
|-----------|--------------|
| **Frontend** | HTTP API entry point (OpenAI-compatible) |
| **Prefill Worker** | Processes prompts, caches KV states |
| **Decode Worker** | Generates tokens using cached KV |
| **Planner** | Auto-scales workers based on SLA |
| **Operator** | Watches DGDs, creates K8s resources |
| **etcd** | Stores state and configuration |
| **NATS** | Inter-component messaging, KV transfer |
| **Grove** | Multi-node orchestration (optional) |

---

## Essential Commands

```bash
# Install CRDs and Operator
helm install dynamo-crds dynamo-crds-${VERSION}.tgz -n default
helm install dynamo-platform dynamo-platform-${VERSION}.tgz -n dynamo

# Deploy your model
kubectl apply -f my-deployment.yaml

# Check status
kubectl get dgd                              # DynamoGraphDeployments
kubectl get dcd                              # DynamoComponentDeployments
kubectl describe dgd <name>                  # Detailed status

# View pods
kubectl get pods -l nvidia.com/dgd=<name>

# Port forward for testing
kubectl port-forward svc/<name>-frontend 8000:8000

# Test the API
curl http://localhost:8000/v1/models
```

---

## Minimal Example

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv

    VllmWorker:
      componentType: worker
      replicas: 2
      resources:
        limits:
          gpu: "1"
      envFromSecret: hf-token-secret
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.1
          command: [python3, -m, dynamo.vllm]
          args: [--model, meta-llama/Llama-3-8B]
```

---

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `DYN_ROUTER_MODE=kv` | Enable KV-cache routing |
| `HF_TOKEN` | Hugging Face authentication |
| `DYNAMO_NAMESPACE` | Service discovery namespace |

---

## Common Issues

| Problem | Check |
|---------|-------|
| Pods not starting | GPU availability, image pull |
| Workers not found | Service DNS, DYNAMO_NAMESPACE |
| KV-routing broken | NATS connectivity, DYN_ROUTER_MODE |
| DGD stuck | `kubectl describe dgd <name>` |

---

## Architecture Layers

```
┌───────────────────────────────────────────┐
│          Your DynamoGraphDeployment       │  ← Layer 4: User Config
├───────────────────────────────────────────┤
│          Dynamo Operator                  │  ← Layer 3: Automation
│     (Controllers + Webhooks)              │
├───────────────────────────────────────────┤
│    etcd │ NATS │ Grove │ KAI Scheduler    │  ← Layer 2: Infrastructure
├───────────────────────────────────────────┤
│            Kubernetes Cluster             │  ← Layer 1: Platform
│         (API Server, Scheduler, etc.)     │
└───────────────────────────────────────────┘
```

---

## Learn More

- Full architecture diagrams: `docs/kubernetes/architecture-diagrams.md`
- Installation guide: `docs/kubernetes/installation_guide.md`
- API reference: `docs/kubernetes/api_reference.md`
- Examples: `examples/backends/vllm/deploy/`
