---
name: dynamo-deploy
description: >-
  Deploy NVIDIA Dynamo inference workloads on Kubernetes. Covers
  DynamoGraphDeploymentRequest (DGDR) deploy-by-intent, DynamoGraphDeployment
  (DGD) manual authoring, recipe-based deployment, conversion-webhook
  semantics, and day-2 operations. Use when deploying a model on Dynamo,
  writing a DGDR or DGD, applying a recipe from ai-dynamo/dynamo, validating
  a manifest before apply, or running a Dynamo worker locally via
  python3 -m dynamo per backend.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - kubernetes
  - deploy
  - inference
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "DGDR", "DGD", "Dynamo deploy", "Qwen3", "dynamo-platform"
- Level 2 (this file): 4-phase workflow with decision branches
- Level 3: references/ for DGDR shape, deployment patterns, known issues
           scripts/  for deterministic validation

Scope: dynamo-platform deployment surface — DGDR + DGD + recipes + local-run via python3 -m dynamo.<backend>
Out of scope (separate skills): planning with AIConfigurator (see dynamo-plan), model optimization (see dynamo-optimize),
                                day-2 troubleshooting (see dynamo-troubleshoot), benchmarking (see dynamo-benchmark).
Platform install (helm install dynamo-platform, operator setup) is a one-time prerequisite; see dynamo-install (deferred).
-->

# Dynamo Deploy (Kubernetes)

Deploy a Dynamo inference workload on Kubernetes by authoring a
`DynamoGraphDeploymentRequest` (DGDR) or `DynamoGraphDeployment` (DGD), or by
applying a pre-tuned recipe from `ai-dynamo/dynamo`. The skill assumes the
`dynamo-platform` Helm chart is already installed and healthy.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Pre-Check  → verify dynamo-platform installed and ready
Phase 2: Author     → DGDR (deploy-by-intent) | DGD (manual) | recipe
Phase 3: Validate   → kubectl --dry-run=server, schema check, SKU match
Phase 4: Apply      → kubectl apply, watch lifecycle, /v1/models, test inference
```

---

## Command Safety

Before issuing any command that changes cluster state, classify it against
the tables below. The Human-in-the-Loop contract (per the authoring guide
§12) applies: present, then wait.

### DESTRUCTIVE — always require explicit confirmation

These commands cause immediate, hard-to-reverse impact. State what the
command does, what it destroys, ask "Do you want me to proceed? You are
responsible for the outcome." Wait for explicit yes.

| Command Pattern | Risk |
|---|---|
| `kubectl delete dgdr <name> -n <ns>` | Deletes the DGDR. The DGD it created is **not** removed (no owner reference); the workload keeps running. |
| `kubectl delete dgd <name> -n <ns>` | Deletes the running inference workload. All Frontend, Worker, Planner pods terminate. Inference traffic drops. |
| `kubectl delete crd dynamographdeployments.nvidia.com` | Removes the CRD and **all DGD instances cluster-wide**. Cannot be undone without operator reinstall. |
| `kubectl delete ns <ns>` | Removes the namespace and every resource in it. |
| `helm uninstall dynamo-platform -n <ns>` | Removes the operator, etcd, NATS, Grove, KAI. All Dynamo workloads in the cluster lose their controller. |

### MUTATING — require confirmation with explanation

State what changes, note any disruption (pod restarts, rolling updates),
ask "Should I proceed? This will affect running workloads."

| Command Pattern | Impact |
|---|---|
| `kubectl apply -f <dgdr.yaml>` | Creates a DGDR; operator starts a profiling Job. Profiling can run minutes (rapid) to hours (thorough) and consumes GPU. |
| `kubectl apply -f <dgd.yaml>` | Creates the DGD; operator pulls images, schedules Frontend + Workers + (optional) Planner, registers /v1/models. |
| `kubectl patch dgd <name> -n <ns> --type=merge -p '...'` | Modifies the live spec. Operator may roll workers depending on the field. |
| `kubectl edit dgdsa <name> -n <ns>` | Changes scaling behavior. Planner pod restart. |
| `helm upgrade dynamo-platform <chart>` | Rolls the operator. Active reconciliation pauses briefly; conversion webhook may glitch on in-flight applies. |

### SAFE — no confirmation needed

```
kubectl get dgdr/dgd/dgdsa/dcd/dynamomodel
kubectl describe dgdr/dgd <name>
kubectl get events --field-selector involvedObject.name=<name>
kubectl logs <pod>
kubectl get crd | grep nvidia.com
kubectl get apiservice | grep nvidia.com
helm status dynamo-platform
helm get values dynamo-platform
kubectl get pods -n <ns> -o wide
kubectl get nodes -L nvidia.com/gpu.product
```

---

## Phase 1: Pre-Check

**Goal.** Verify `dynamo-platform` is installed and reports healthy before
authoring any DGDR or DGD.

**Inputs.** Target namespace (default `dynamo-system` if installed cluster-
wide; `default` for namespace-restricted installs).

**Commands** (SAFE):

```bash
# Confirm the operator is running.
kubectl get deploy -n <ns> -l app.kubernetes.io/name=dynamo-operator

# Confirm all seven CRDs are registered.
kubectl get crd | grep nvidia.com
# Expect: dynamographdeployments, dynamographdeploymentrequests,
#         dynamocomponentdeployments, dynamographdeploymentscalingadapters,
#         dynamomodels, dynamocheckpoints, dynamoworkermetadatas.

# Confirm the conversion webhook is reachable for the CRDs with both v1alpha1
# and v1beta1 served (DGD, DGDR, DCD, DGDSA).
kubectl get apiservice | grep nvidia.com

# Confirm a GPU is visible to the cluster (via GPU Feature Discovery labels).
kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count
```

Run the bundled verifier:

```bash
bash scripts/verify-platform.sh -n <ns>
```

The script applies the `check()` pattern (per the authoring guide §8.4) and
exits non-zero if any required check fails. See
[scripts/verify-platform.sh](scripts/verify-platform.sh).

**Decision point** — if the platform is **not** installed:

**Decision:** `dynamo-platform` was not found in namespace `<ns>`. The
`dynamo-deploy` skill cannot proceed without it. The platform install is a
one-time cluster-level step owned by the `dynamo-install` skill (see
authoring guide §13 "Cross-Skill Referencing"). Should I refuse this
request and point you at `dynamo-install`?

Wait for explicit yes. Do not attempt a silent helm install from this
skill.

**Verification gate.** All seven CRDs present, operator pod Ready, GPU
nodes visible. Proceed to Phase 2 only when these are true.

---

## Phase 2: Author

**Goal.** Produce a valid DGDR, DGD, or recipe-based manifest the operator
will accept.

**Inputs.** Model identifier (HF ID or local path), serving mode preference
(aggregated vs disaggregated), SLA targets if known, available GPU SKU and
count.

### 2.1 Choose Authoring Path

Three paths. Pick one — do not mix.

| Path | Use when | Reference |
|---|---|---|
| **DGDR** (deploy-by-intent) | You have SLA targets but no hand-tuned config. Operator runs profiling, picks parallelism, writes the DGD. | [references/dgdr-shape.md](references/dgdr-shape.md) |
| **DGD** (manual) | You have a known-good config (your own or one not yet recipe-ised). | [references/deployment-patterns.md](references/deployment-patterns.md) |
| **Recipe** | A pre-tuned recipe exists for your model and hardware under `ai-dynamo/dynamo` `recipes/<model>/<framework>/<config>/`. | `ai-dynamo/dynamo` repo, `recipes/` |

**Decision:** Three valid authoring paths exist for the model you named.
Which do you want — DGDR, DGD, or recipe? Each is described in the table
above with a pointer to the reference content. Wait for explicit choice.

### 2.2 DGDR Authoring

The minimal DGDR is **two fields** — `model` and `image`:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: my-model
spec:
  model: Qwen/Qwen3-0.6B
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.0"
```

For SLA-driven deploys, add `sla`, `workload`, and optionally `hardware`:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-06b-sla
spec:
  model: Qwen/Qwen3-0.6B
  backend: trtllm
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.0"
  searchStrategy: thorough
  hardware:
    numGpusPerNode: 8
    gpuSku: h200_sxm
  workload:
    isl: 3000
    osl: 500
  sla:
    ttft: 50.0
    itl: 10.0
  autoApply: true
```

Full field reference: [references/dgdr-shape.md](references/dgdr-shape.md).

**Decision points for DGDR:**

- **`searchStrategy: rapid` vs `thorough`.** `rapid` uses the AIC simulator
  (~30 s) and is the default. `thorough` runs real-GPU sweeps over 2-4 h
  and consumes the GPUs for the duration. Confirm which the user wants.
- **`backend: auto`.** Auto picks an engine based on the model family. If
  you want determinism (e.g. for benchmarking), set the backend explicitly.
- **`hardware.gpuSku`.** Required in namespace-restricted operator
  installations (the operator cannot enumerate cluster nodes). Optional in
  cluster-wide installs (auto-detected from GPU Feature Discovery labels).
  See [references/dgdr-shape.md](references/dgdr-shape.md) for the enum and
  for the `h100_pcie` / `a100_pcie` / `v100_pcie` profiler caveat.
- **`autoApply: true` vs `false`.** `true` (the default) tells the operator
  to create the DGD as soon as profiling finishes. `false` holds the
  generated DGD spec on the DGDR status; the user reviews it before
  applying. Use `false` in regulated environments.

### 2.3 DGD Authoring

Three deployment patterns covered in
[references/deployment-patterns.md](references/deployment-patterns.md):

1. **Aggregated** — single Frontend + single Worker per replica. Default
   for small models and prototyping.
2. **Disaggregated** — Frontend + `prefill` Worker + `decode` Worker.
   Required for high-throughput serving and for any model where the
   prefill/decode compute profile diverges enough to benefit from
   independent scaling.
3. **KV-aware Routing** — Frontend + Router + Worker. Routes requests by
   KV-cache locality. Combine with disaggregated for production.

The `apiVersion` is `nvidia.com/v1alpha1` for current upstream examples.
Both `v1alpha1` and `v1beta1` are served (the conversion webhook handles
the bridge per the authoring guide §4); new manifests should target
`v1beta1`.

### 2.4 Recipe-Based Authoring

Recipes live under `ai-dynamo/dynamo` `recipes/<model>/<framework>/<config>/deploy/`.
Each recipe ships with a tested DGD and a benchmark companion. Use a recipe
when one exists for the user's model and hardware combination — they
encode tuning that DGDR/DGD authoring would have to re-derive.

The 1.2.0-line recipes include: `deepseek-r1`, `deepseek-v32-fp4`,
`deepseek-v4`, `glm-5-nvfp4`, `gpt-oss-120b`, `kimi-k2.5`, `llama-3-70b`,
`nemotron-3-nano-omni`, `nemotron-3-super-fp8`, `qwen3-32b`,
`qwen3-32b-fp8`, `qwen3-235b-a22b-fp8`, `qwen3-vl-30b`, `qwen3.6-35b`.
Recipe set varies per release; verify against the target release branch.

**Verification gate.** Authored manifest exists on disk and parses as YAML.

---

## Phase 3: Validate

**Goal.** Catch schema, hardware, and resource errors before they hit the
operator and waste a profiling Job's GPU time.

**Commands** (SAFE; the script defaults to `--dry-run=server`, per the
authoring guide §8.1):

```bash
# YAML parses.
python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" my-dgdr.yaml

# Server-side dry-run (the API server validates against the CRD schema
# without persisting the resource).
kubectl apply --dry-run=server -f my-dgdr.yaml

# Hardware spec matches a cluster node (only when hardware.gpuSku is set).
kubectl get nodes -L nvidia.com/gpu.product
```

Run the bundled validator:

```bash
bash scripts/validate-dgdr.sh -f my-dgdr.yaml -n <ns>
```

The script applies the `pass/fail/warn` pattern (per the authoring guide
§8.3) and exits non-zero if any failure is recorded. See
[scripts/validate-dgdr.sh](scripts/validate-dgdr.sh).

**Decision point** — if the server-side dry-run rejects the manifest:

**Decision:** The API server rejected `<name>` with `<error>`. Possible
causes: deprecated field, missing required field, schema mismatch, or
v1alpha1 manifest applied without conversion webhook reachable. The skill
will not retry blindly — what's the next step (edit the manifest, debug
the webhook, switch versions)?

Wait for explicit choice.

**Verification gate.** YAML parses, `--dry-run=server` succeeds, GPU SKU
present on at least one node (if specified), HF token Secret exists in
namespace (if the model is gated). Proceed to Phase 4 only when these are
true.

---

## Phase 4: Apply and Observe

**Goal.** Apply the manifest, watch the lifecycle, confirm the model is
serving requests.

### 4.1 Apply

**Decision (MUTATING):** Applying `<file>` will create a DGDR/DGD in
namespace `<ns>`. This will:

- (DGDR) Start a profiling Job that runs ~30 s (`rapid`) or 2-4 h
  (`thorough`) and consumes `<n>` GPU(s) for the duration.
- (DGD) Schedule Frontend + Worker pods that pull images and start
  serving traffic on `<service>:<port>`.

Should I proceed?

Wait for explicit yes. Then:

```bash
kubectl apply -f my-manifest.yaml
```

### 4.2 Watch Lifecycle (DGDR)

```bash
# Top-level phase progression.
kubectl get dgdr <name> -n <ns> -w

# Detailed status, conditions, events.
kubectl describe dgdr <name> -n <ns>

# Profiling sub-phase.
kubectl get dgdr <name> -n <ns> -o jsonpath='{.status.profilingPhase}'

# Profiling job logs (the Job is labeled by DGDR name).
kubectl get pods -n <ns> -l nvidia.com/dgdr-name=<name>
kubectl logs -f <pod> -n <ns>

# Generated DGD spec (when autoApply: false).
kubectl get dgdr <name> -n <ns> \
  -o jsonpath='{.status.profilingResults.selectedConfig}' | python3 -m json.tool
```

DGDR phase sequence: `Pending → Profiling → Ready → Deploying → Deployed`.
`Failed` is a terminal state from any prior phase. Profiling sub-phases:
`Initializing → SweepingPrefill → SweepingDecode → SelectingConfig →
BuildingCurves → GeneratingDGD → Done`.

### 4.3 Watch Lifecycle (DGD)

```bash
# DGD readiness.
kubectl get dgd <name> -n <ns> -w

# Pods.
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<name>

# Frontend endpoints.
kubectl get svc -n <ns> | grep frontend
```

### 4.4 Verify Model Registration

A DGD reporting `state: successful` does **not** imply the model is
registered with the Frontend. Workers must download weights and register
endpoints via NATS before `/v1/models` returns data. Poll until non-empty:

```bash
# Port-forward to the Frontend service (or use the cluster ingress).
kubectl port-forward -n <ns> svc/<frontend-svc> 8000:8000 &

# Poll until /v1/models is non-empty (typically 30-120 s after DGD Ready).
until curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")'; do
  echo "waiting for model registration..."
  sleep 10
done
```

This pattern is one of the stable known issues; see
[references/known-issues.md](references/known-issues.md) entry "DGD
successful but /v1/models empty".

### 4.5 Test Inference

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id-from-/v1/models>",
    "messages": [{"role": "user", "content": "Say hello."}]
  }' | python3 -m json.tool
```

A non-empty response with `choices[0].message.content` confirms the
deployment is serving end-to-end.

**Verification gate.** DGDR (or DGD) reports a healthy terminal state,
`/v1/models` returns the model, a test inference returns a non-empty
response.

---

## Phase 5 (optional): Local Development

For workstation development without a cluster, run a single-node Dynamo
worker directly. There is no `dynamo-run` CLI; the invocation is the
Python module form:

```bash
pip install 'ai-dynamo[vllm]==1.2.0.post1'   # or [trtllm], [sglang]
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

Backends: `dynamo.vllm`, `dynamo.trtllm`, `dynamo.sglang`. See the
authoring guide §4 for the full module inventory. Reference launch
scripts: `ai-dynamo/dynamo` `examples/backends/<backend>/launch/`.

This path does not exercise the operator, CRDs, or routing; it is for
local iteration only.

---

## Refusal Conditions

The skill refuses, even with user confirmation, when:

- The target namespace is listed as production-protected in the user's
  environment file.
- The operation is a DGD `kubectl delete` that would terminate a workload
  with live inference traffic (verify via `/v1/models` count + recent
  request rate before refusing the refusal).
- The DGDR `hardware.gpuSku` names a SKU not present on any cluster node
  (per `nvidia.com/gpu.product` labels) and the namespace is not
  explicitly tagged as a dry-run/test environment.
- The Dynamo image tag in `spec.image` does not exist in the target image
  registry (verify with `crane manifest` or equivalent).
- The user requests `v1alpha1` for a CRD that exists only as `v1beta1`.

---

## Cross-Skill Referencing

This skill does not absorb workflows owned by sibling skills (per
authoring guide §13):

| Need | Sibling | Hand-off |
|---|---|---|
| Decide parallelism / SLA targets before authoring | `dynamo-plan` | Skill emits suggested DGDR; user pastes into Phase 2. |
| Model optimization (FP8, FP4 quantization) before deploy | `dynamo-optimize` | Skill emits the optimized checkpoint identifier; user passes via `--load-format` in worker args. |
| Crashlooping worker, stuck Planner, `/v1/models` empty after 5 min | `dynamo-troubleshoot` | Skill takes a snapshot of the failing DGD and walks the day-2 playbook. |
| Benchmark a deployed workload | `dynamo-benchmark` | Skill points AIPerf at the Frontend service. |
| Install `dynamo-platform` itself | `dynamo-install` (deferred) | Phase 1 refuses if the platform is not installed; user runs `dynamo-install` first. |

If any of these needs surfaces during Phase 1-4 execution, pause and
hand off; do not attempt to absorb the workflow.

---

## Observability Hooks

When verification fails or the deployment misbehaves, the following are
the canonical sources:

| Signal | Source |
|---|---|
| Operator decisions | `kubectl logs -n <ns> deploy/dynamo-operator` |
| Profiling Job state | `kubectl logs <profiling-pod>` (selector: `nvidia.com/dgdr-name=<name>`) |
| KV-router metrics | Prometheus, scraped from the router pod (`/metrics`) |
| Frontend request metrics | Prometheus, scraped from the Frontend service (`/metrics`) |
| Conversion-webhook audit | API server audit log (cluster-admin) |
| etcd quorum | `kubectl exec` into an etcd pod; `etcdctl endpoint status` |
| NATS connection state | Operator logs; NATS box `nats stream report` |

The skill does not configure observability — that is per-cluster ops. It
names the sources so verification failures route to the right log first.

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- `dynamo-platform` Helm chart installed; operator pod Ready; all seven CRDs registered (per).
- GPU nodes visible to the cluster (`nvidia.com/gpu.product` labels populated via GPU Feature Discovery).
- HF token Secret in the target namespace if the model is gated (per).
- A target namespace; `kubectl` access to it.

---

## Limitations

What this skill does NOT cover:

- Does NOT install the `dynamo-platform` chart itself. That is the deferred `dynamo-install` skill's scope (per PLAN.md §2.1).
- Does NOT cover model optimization. See `dynamo-optimize`.
- Does NOT cover the API-surface configuration (Frontend args, DynamoModel CRs, gateway integration). See `dynamo-frontend`.
- Does NOT cover benchmarking. See `dynamo-benchmark`.

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
| `scripts/validate-dgdr.sh` | Pre-apply DGDR / DGD validator (kubectl --dry-run=server) | `-f <manifest.yaml> [-n <ns>]` |
| `scripts/verify-platform.sh` | Post-install dynamo-platform verifier (CRDs, webhooks, etcd, NATS) | `[-n <ns>]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/validate-dgdr.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/validate-dgdr.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/dgdr-shape.md](references/dgdr-shape.md) — Annotated `v1beta1`
  DGDR field reference: every top-level field, allowed values, enums,
  defaults, override semantics, lifecycle phases, conditions, event reasons.
- [references/deployment-patterns.md](references/deployment-patterns.md) —
  Aggregated, disaggregated, KV-aware routing, Planner-driven autoscaling,
  and recipe-based DGD patterns. Verbatim YAML blocks.
- [references/known-issues.md](references/known-issues.md) — Stable issue
  patterns recurring across releases (Helm v4 OCI, HF token wiring,
  `/v1/models` empty after Ready, base-model chat templates, disagg KV
  transfer config, `--dry-run=server` validation).
- [scripts/validate-dgdr.sh](scripts/validate-dgdr.sh) — Pre-apply DGDR
  validator using `kubectl --dry-run=server` and the `pass/fail/warn`
  helper.
- [scripts/verify-platform.sh](scripts/verify-platform.sh) — Post-install
  platform verifier using the `check()` pattern.
