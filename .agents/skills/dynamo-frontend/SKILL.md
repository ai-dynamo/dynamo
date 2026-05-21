---
name: dynamo-frontend
description: >-
  Configure the NVIDIA Dynamo Frontend service (the OpenAI-compatible
  HTTP layer), register models via the DynamoModel Custom Resource,
  set up multi-model serving, and wire in the gateway path (GAIE,
  kgateway, Istio, TLS, auth). Use when authoring Frontend args on a
  DGD, creating a DynamoModel CR, serving multiple models from one
  DGD, integrating the Gateway API Inference Extension (GAIE),
  diagnosing /v1/models or /v1/chat 4xx errors, or wiring TLS / auth
  in front of a Dynamo deployment.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - frontend
  - gateway
  - openai-api
  - gaie
  - dynamomodel
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "Dynamo Frontend", "DynamoModel", "GAIE", "/v1/models", "/v1/chat", "kgateway", "Istio sidecar", "Inference Gateway"
- Level 2 (this file): 4-phase request-path workflow
- Level 3: references/ for OpenAI API surface, DynamoModel CR, gateway integration
           scripts/  for the pre-apply Frontend validator and the post-apply API probe

Scope: the request-path layer between client and worker — Frontend service config,
       model registration via DynamoModel CR, multi-model serving, gateway wiring.
Out of scope: workload deployment (see dynamo-deploy), local-dev (see dynamo-serve),
              planning (see dynamo-plan), optimization (see dynamo-optimize),
              day-2 troubleshooting (see dynamo-troubleshoot), benchmarking (see dynamo-benchmark).
-->

# Dynamo Frontend (Request Path)

Own the layer between the client and the workers: the OpenAI HTTP
endpoints, model registration, multi-model serving, and the gateway
that fronts it. The deployment must already exist (`dynamo-deploy`
created the DGD); this skill configures how clients reach it.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Pre-Check → DGD running; Frontend pod ready; DynamoModel CRD available; gateway choice
Phase 2: Author    → Frontend args (--router, --enable-prefix-caching, ...); DynamoModel CRs; gateway resources
Phase 3: Apply     → kubectl apply / patch DGD; create DynamoModels; configure gateway
Phase 4: Verify    → /v1/models populated; sample requests through gateway; auth / TLS works
```

---

## Command Safety

Most commands are inspection. MUTATING applies to any DGD patch or
DynamoModel CR creation that affects live request routing.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `kubectl delete dynamomodel <name> -n <ns>` | Removes the model registration; `/v1/models` drops it; in-flight requests targeting it fail. |
| `kubectl delete gateway <name>` | Removes ingress; ALL traffic to deployments behind this gateway is dropped. |
| `kubectl delete validatingwebhookconfiguration <name>` | Removes admission validation; misconfigured manifests can now reach the API server. |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `kubectl patch dgd <name> --type=merge -p '{"spec":{"services":{"Frontend":{...}}}}'` | Rolls the Frontend pod. Brief `/v1/models` blip during the roll. |
| `kubectl apply -f <dynamomodel.yaml>` | Registers a new model; the Frontend picks it up; `/v1/models` reflects the addition. |
| `kubectl apply -f <gateway.yaml>` | Creates / updates the gateway. Existing routes may shift; verify with `kubectl get httproute`. |
| `kubectl apply -f <httproute.yaml>` | Adds / changes a route; existing requests using the old route may 404 during the transition. |

### SAFE — no confirmation needed

```
kubectl get dynamomodel -A
kubectl describe dynamomodel <name>
kubectl get gateway,httproute -A
kubectl get pods -n <ns> -l app.kubernetes.io/component=frontend
kubectl logs <frontend-pod> -n <ns>
kubectl logs <frontend-pod> -n <ns> --previous
curl http://<frontend>/v1/models
curl http://<gateway>/v1/models
kubectl exec <frontend-pod> -- env | grep -E "HF_TOKEN|MODEL"
```

---

## Phase 1: Pre-Check

**Goal.** Confirm the deployment is running and the request-path
surface is ready to configure.

**Inputs**:

| Input | How to get it |
|---|---|
| Target DGD name | `kubectl get dgd -A` |
| Frontend service | `kubectl get svc -n <ns> -l app.kubernetes.io/component=frontend` |
| DynamoModel CRD available | `kubectl get crd dynamomodels.nvidia.com` (per — v1alpha1-only) |
| Gateway choice | One of: GAIE (Inference Gateway Extension), kgateway, Istio, Contour, plain Service+Ingress |
| HF token Secret | `kubectl get secret hf-token-secret -n <ns>` if gated models |

**Commands** (SAFE):

```bash
# DGD is Ready.
kubectl get dgd <name> -n <ns> -o jsonpath='{.status.state}'

# Frontend pod is Ready.
kubectl get pods -n <ns> -l "nvidia.com/dgd-name=<name>,app.kubernetes.io/component=frontend"

# DynamoModel CRD is registered (per).
kubectl get crd dynamomodels.nvidia.com -o jsonpath='{.metadata.name}'

# Current /v1/models baseline.
kubectl port-forward -n <ns> svc/<frontend-svc> 8000:8000 &
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Run the bundled validator:

```bash
bash scripts/validate-frontend.sh -d <dgd-name> -n <ns>
```

The script applies the `pass/fail/warn` pattern from. See
[scripts/validate-frontend.sh](scripts/validate-frontend.sh).

**Decision point** — if the Frontend has no HF token mount and the
intended model is gated, fix this BEFORE any DynamoModel apply. The
 pattern (HF token required on Frontend, not just Worker) causes
the model registration to silently fail with `401 Unauthorized`.

**Verification gate.** DGD Ready, Frontend pod Ready, DynamoModel CRD
present, HF token Secret accessible if needed.

---

## Phase 2: Author

**Goal.** Produce the Frontend service config, DynamoModel manifests,
and gateway resources that match the intended user experience.

### 2.1 Frontend Service Args

The Frontend service is one of the DGD's services. Its args control
which OpenAI endpoints are active and how routing works. Authoritative
list: [references/api-surface.md](references/api-surface.md).

Common patterns:

| Pattern | Args on `Frontend.extraPodSpec.mainContainer.args` |
|---|---|
| Basic OpenAI surface | (defaults — `/v1/chat`, `/v1/completions`, `/v1/models` all on) |
| KV-aware routing in front of multi-worker | `[--router, kv-aware]` |
| Add /v1/embeddings | `[--enable-embeddings]` |
| Add /v1/realtime | `[--enable-realtime]` (per's reference to PR #9205 on main; check the release line) |
| KServe gRPC | `[--kserve-grpc]` |

Patch an existing DGD:

```bash
kubectl patch dgd <name> -n <ns> --type=merge -p '{
  "spec": {
    "services": {
      "Frontend": {
        "extraPodSpec": {
          "mainContainer": {
            "args": ["--router", "kv-aware"]
          }
        }
      }
    }
  }
}'
```

### 2.2 DynamoModel CR

Per, `DynamoModel` is **v1alpha1-only** — no v1beta1 schema exists.
The minimal CR:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: qwen3-06b
  namespace: <ns>
spec:
  baseModel: Qwen/Qwen3-0.6B
  type: chat
  # See references/dynamomodel-shape.md for the full field reference.
```

Use cases:

- Register a model the worker has loaded so `/v1/models` exposes it.
- Bind a model alias (so clients can refer to `Qwen3-0.6B` rather than
  `Qwen/Qwen3-0.6B`).
- Reserve a model name when the deployment is split across DGDs.

### 2.3 Multi-Model Serving

To serve multiple models from one Frontend, the DGD's Worker services
need to load multiple checkpoints, and each model needs a DynamoModel
CR registering it.

Pattern (sketch):

```yaml
# DGD with two workers each loading a different model.
spec:
  services:
    Frontend: {...}
    VllmDecodeWorker:
      replicas: 1
      extraPodSpec:
        mainContainer:
          args: [--model, Qwen/Qwen3-0.6B]
    VllmDecodeWorkerB:
      componentType: worker
      replicas: 1
      extraPodSpec:
        mainContainer:
          args: [--model, meta-llama/Llama-3-8B]
```

Plus DynamoModel CRs for both:

```yaml
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata: {name: qwen, namespace: <ns>}
spec: {baseModel: Qwen/Qwen3-0.6B, type: chat}
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata: {name: llama, namespace: <ns>}
spec: {baseModel: meta-llama/Llama-3-8B, type: chat}
```

The Frontend dispatches requests based on the `model` field in the
OpenAI request body.

### 2.4 Gateway Choice

See [references/gateway-integration.md](references/gateway-integration.md)
for the full per-gateway recipe. Summary:

| Gateway | When to use | Notes |
|---|---|---|
| **GAIE** (Gateway API Inference Extension) | Multi-deployment inference fleets; want SIG-aligned routing | Requires the EPP image per / `container/context.yaml`: `us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v1.5.0-rc.2` |
| **kgateway** | You already run kgateway | The known DYN-3077 / NVBug 6194957 P0 with Istio sidecar applies — see [references/known-issues.md](references/known-issues.md) |
| **Istio** | You already run Istio | Sidecar injection on the Frontend pod triggers DYN-3077 — explicit opt-out required |
| **Plain Service + Ingress** | Single-deployment, no SIG features | Simplest path; no smart routing |

**Decision:** The proposed configuration changes Frontend args to
`<X>`, creates DynamoModel CRs `<list>`, and configures the gateway
via `<resource list>`. This will affect the request-path during the
patch / apply. Proceed?

Wait for explicit yes.

**Verification gate.** All YAML manifests authored and parseable.

---

## Phase 3: Apply

**Goal.** Apply the manifests in dependency order; observe transitions.

Apply order:

1. **DynamoModel CRs first.** The CR is declarative and idempotent;
   creating it before the Frontend picks up new args avoids a
   transient state where the Frontend serves model traffic but the
   model is not yet registered.
2. **Frontend DGD patch second.** Patches roll the Frontend pod; the
   new args take effect after the new pod is Ready.
3. **Gateway resources last.** Gateway / HTTPRoute changes redirect
   request traffic; apply after the Frontend is serving the new
   surface.

```bash
# 1. DynamoModels
kubectl apply -f dynamomodel-*.yaml

# 2. Frontend DGD patch
kubectl patch dgd <name> -n <ns> --type=merge -p '{...}'

# 3. Gateway resources
kubectl apply -f gateway.yaml httproute.yaml
```

Watch the Frontend roll:

```bash
kubectl rollout status deployment/<frontend-deploy> -n <ns> --timeout=300s
kubectl get pods -n <ns> -l app.kubernetes.io/component=frontend -w
```

**Decision (MUTATING):** Patch will roll the Frontend pod. Brief
`/v1/models` blip during the roll. Proceed?

Wait for explicit yes.

**Verification gate.** Frontend pod Ready under the new spec;
DynamoModel CRs show in `kubectl get dynamomodel`; gateway resources
applied without conflicts.

---

## Phase 4: Verify

**Goal.** Confirm the request-path surface works end-to-end.

```bash
# 1. /v1/models reflects the registered models.
kubectl port-forward -n <ns> svc/<frontend-svc> 8000:8000 &
PF_PID=$!
trap "kill $PF_PID 2>/dev/null" EXIT
sleep 2
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 2. Sample inference per model.
for model in $(curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; [print(m["id"]) for m in json.load(sys.stdin).get("data",[])]'); do
  echo "=== $model ==="
  curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"$model\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
      \"max_tokens\": 16
    }" | python3 -m json.tool
done

# 3. Gateway path (when GAIE / kgateway / Istio).
GW_IP=$(kubectl get gateway <gw> -n <ns> -o jsonpath='{.status.addresses[0].value}')
curl -s -H "Host: <inference.example>" http://$GW_IP/v1/models | python3 -m json.tool
```

If `/v1/models` returns empty for >2 minutes after the patch, that's
the registration window — or the HF-token-on-Frontend issue
if the model is gated.

Run the bundled probe:

```bash
bash scripts/verify-api-surface.sh -d <dgd-name> -n <ns>
```

The script applies the `check()` pattern from.

**Verification gate.** `/v1/models` lists the expected models, sample
inference returns non-empty responses, gateway path returns the same
results.

---

## Refusal Conditions

- The deployment is in production and the gateway change would shift
  traffic between active models — refuse until a canary/staging path
  is confirmed.
- The DynamoModel CR is authored as `nvidia.com/v1beta1` — refuse and
  point at: DynamoModel is v1alpha1-only.
- The Frontend args reference a flag not present in the worker's
  `--help` for the target release — refuse and probe with
  `kubectl exec <frontend-pod> -- python3 -m dynamo.frontend --help`.
- Istio sidecar injection is enabled on the Frontend pod's namespace
  without the `kgateway` known-issue workaround — refuse and surface
  DYN-3077 / NVBug 6194957 from the per-release tracker.

---

## Cross-Skill Referencing

| Need | Sibling |
|---|---|
| Initial workload deploy | [dynamo-deploy](../dynamo-deploy/SKILL.md) — must run before this skill |
| Local-dev iteration on the same model | [dynamo-serve](../dynamo-serve/SKILL.md) — Frontend args translate (mostly) |
| Diagnose a Frontend 5xx or empty `/v1/models` | [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md) — Phase 3 signatures |
| Benchmark with multi-model traffic | [dynamo-benchmark](../dynamo-benchmark/SKILL.md) — AIPerf can target any model in `/v1/models` |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- A DGD created by `dynamo-deploy` and reporting Ready.
- Frontend pod Ready under that DGD.
- `DynamoModel` CRD registered (v1alpha1, per).
- Gateway controller installed if using GAIE / kgateway / Istio (the `dynamo-platform` chart does not pre-install these).
- HF token Secret present if the registered model is gated (per).

---

## Limitations

What this skill does NOT cover:

- Does NOT create the DGD. That is `dynamo-deploy`'s scope.
- Does NOT cover worker-side flags (parallelism, KV cache budget). See `dynamo-deploy` Phase 2.
- Does NOT measure end-to-end throughput. See `dynamo-benchmark`.
- Gateway choice and installation are not in scope; the skill configures the integration but expects the gateway controller to exist.

See `## Cross-Skill Referencing` for the appropriate sibling skill in each case.

---

## Troubleshooting

When a step in this workflow fails:

- **Per-skill known patterns** are catalogued in [references/known-issues.md](references/known-issues.md). Walk that list first.
- **Cross-skill day-2 patterns** (worker crashloops, conversion-webhook timeouts, KV-transfer fallback, gateway 500s) are owned by [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md). The 4-phase day-2 workflow there (Triage → Inspect → Diagnose → Remediate) applies.
- **Per-release bugs** live in the active QA tracker (per the skill's `version` field). Pull the tracker view for the matching release line.

---

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/validate-frontend.sh` | Pre-apply Frontend + DynamoModel validator | `-d <dgd-name> [-n <ns>] [--dm-file <file>]` |
| `scripts/verify-api-surface.sh` | Post-apply OpenAI surface probe (`/v1/models`, sample inference) | `-d <dgd-name> [-n <ns>] [-p <port>]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/validate-frontend.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/validate-frontend.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/api-surface.md](references/api-surface.md) — OpenAI
  endpoint reference, Frontend args, request/response shape per
  endpoint.
- [references/dynamomodel-shape.md](references/dynamomodel-shape.md) —
  DynamoModel CR field reference (v1alpha1).
- [references/gateway-integration.md](references/gateway-integration.md) —
  Per-gateway recipe (GAIE, kgateway, Istio, plain Ingress) with the
  known sidecar gotchas.
- [references/known-issues.md](references/known-issues.md) — Frontend-
  and gateway-specific stable patterns.
- [scripts/validate-frontend.sh](scripts/validate-frontend.sh) —
  Pre-apply Frontend + DynamoModel validator; PASS/FAIL/WARN per.
- [scripts/verify-api-surface.sh](scripts/verify-api-surface.sh) —
  Post-apply OpenAI surface probe; check() pattern per.
