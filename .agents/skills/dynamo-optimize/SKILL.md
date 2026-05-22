---
name: dynamo-optimize
description: >-
  Pick and apply a tested NVIDIA Dynamo recipe for a target model, hardware
  envelope, and workload shape. Wraps the existing recipes tree with a
  decision-heavy workflow: chain to AIConfigurator output when present,
  refuse outside the recipe tested envelope, capture a baseline, apply
  minimal patches, and validate against an AIPerf goodput SLO with
  per-dimension PASS/FAIL. Use when picking a recipe, patching one,
  or comparing aggregated vs disaggregated vs KV-aware routing.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - optimize
  - recipe
  - kubernetes
  - kv-router
  - disaggregated
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "optimize", "pick a recipe", "patch a recipe", "tune dynamo", "compare agg vs disagg", "KV router worth it for my workload"
- Level 2 (this file): 4-phase Gather → Pick → Patch → Apply with explicit decision points and refusals
- Level 3: references/ for SLO shape, inference-literature regressions, recipe workflow, known issues
           scripts/  for recipe discovery + SLO measurement

Scope: pick + apply + measure an existing recipe under ai-dynamo/dynamo recipes/. NOT a profiler, NOT a planner, NOT a benchmarker.
Out of scope (separate skills):
  - dynamo-plan      → AIConfigurator (offline best-config search). dynamo-optimize chains to its output, does not duplicate.
  - dynamo-deploy    → DGDR/DGD authoring without recipe context, manual TP/PP/EP choices, hand-rolled deploys.
  - dynamo-benchmark → AIPerf full sweep with the recipe's perf.yaml companion (the load Pod, dataset traces, post-hoc analysis).
  - dynamo-troubleshoot → day-2 failure triage.

Locked decisions (HANDOFF.md §2 + 2026-05-22 user resolution):
  - AIConfigurator chain: SOFT-recommend, branch on DGD/DGDR/neither. Reason: silent naive-fallback risk; DGDR immutability hostile to iteration; dead controller event constants suggest the "native CRD integration" claim is aspirational.
  - AIPerf pin: ALWAYS aiperf==0.8.0. Override per-recipe pin. Flag DYN-2878-style risk if base image ships incompatible transformers.
  - Workstation pre-validation via dynamo-serve: INCLUDED for agg-* modes; explicitly skipped for disagg-* modes with a correlation caveat in the skill body.
  - Skill keeps the name `dynamo-optimize`.

Every load-bearing claim in this body traces to a row in dynamo-skills/docs/citations.md Section G.
-->

# Dynamo Optimize (Recipe-Runner)

Optimize a Dynamo deployment by picking and applying a tested recipe from
`ai-dynamo/dynamo` `recipes/<model>/<framework>/<mode>/`, then validating it
against a declared AIPerf `--goodput` SLO. The skill does not invent new
configurations and does not duplicate AIConfigurator's job — it composes
existing parts (recipe tree, AIConfigurator output when present, AIPerf
`--goodput` grammar) into a workflow that ends in per-dimension PASS/FAIL
against a baseline.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Gather  → AIConfigurator output (if present), workload class, SLO, baseline
Phase 2: Pick    → recipe filtered by (model × framework × hardware envelope × workload class)
Phase 3: Patch   → minimal patches, optional agg-* pre-validation, server-side dry-run
Phase 4: Apply   → kubectl apply, pod-readiness wait, AIPerf with declared SLO, PASS/FAIL
```

---

## Command Safety

Classify before running. The Human-in-the-Loop contract (per
`dynamo-skill-author/SKILL.md`) applies: present, then wait.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `kubectl delete dgd <name> -n <ns>` | Deletes the running inference workload. Frontend, Worker, Planner pods terminate. Inference traffic drops. |
| `kubectl delete pvc model-cache -n <ns>` | Drops the cached model weights. Next deploy re-downloads (10s of GB). |
| `kubectl delete pvc perf-cache -n <ns>` | Drops captured AIPerf artifacts — no longer auditable against this run. |
| `git checkout -- recipes/<...>` | Overwrites local patches to the recipe tree. Cannot be undone if uncommitted. |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `kubectl apply -f recipes/<model>/model-cache/` | Starts the model-download Job. Pulls 10s of GB from Hugging Face; bandwidth + storage cost. |
| `kubectl apply -f recipes/<model>/<framework>/<mode>/deploy.yaml` | Creates the DGD; operator pulls images, schedules Frontend + Workers, registers `/v1/models`. |
| `kubectl apply -f recipes/<model>/<framework>/<mode>/perf.yaml` | Starts the AIPerf benchmark Pod. Generates load against the deployed model; consumes GPU cycles. |
| `kubectl patch dgd <name> --type=merge -p '...'` | Modifies live spec. Operator may roll workers depending on the field touched. |
| Patching local recipe files (storageClassName, image tag, GPU count) | Modifies your working copy; commit before to keep an undo path. |

### SAFE — no confirmation needed

```
kubectl get dgd/dgdr/pod/job/pvc/secret/storageclass
kubectl describe dgd <name>
kubectl logs -f <pod>
kubectl get events --field-selector involvedObject.name=<name>
kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count
git status / diff / log
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py list ...
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py validate ...
python3 .agents/skills/dynamo-optimize/scripts/measure_slo.py --artifact-dir ... --slo ...
curl -s http://localhost:8000/v1/models
```

---

## Phase 1: Gather

**Goal.** Collect every input the recipe selector needs, with explicit
provenance for the SLO and the AIConfigurator chain.

**Inputs.**

| Input | Source | Required |
|---|---|---|
| Model identifier (HF ID or local path) | User | yes |
| GPU SKU and count | `kubectl get nodes -L nvidia.com/gpu.product` or user | yes |
| Kubernetes context + namespace | `kubectl config current-context` + user | yes |
| HF token Secret name (default `hf-token-secret`) | `kubectl get secret -n <ns>` | yes (if gated model) |
| Storage class for PVCs | `kubectl get storageclass` | yes |
| AIConfigurator output: DGD spec or DGDR `.status.profilingResults.selectedConfig` | User paste, or `kubectl get dgdr <name> -o jsonpath='...'` | recommended (see Decision below) |
| Workload class: ISL, OSL, expected prefix reuse, request rate, latency-vs-throughput preference | Interview if AIC output is absent | yes (one or the other) |
| SLO declaration (AIPerf `--goodput` grammar) | User, or default from recipe | yes |
| Baseline endpoint (URL + AIPerf artifact dir, if comparing) | User | optional |

**Commands** (SAFE):

```bash
# Confirm cluster context and namespace exist.
kubectl config current-context
kubectl get namespace "${NAMESPACE}"

# Verify HF token Secret is present (gated models).
kubectl get secret "${HF_TOKEN_SECRET:-hf-token-secret}" -n "${NAMESPACE}"

# Inventory cluster GPUs (so we can refuse if the SKU isn't present).
kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count

# Inventory available storage classes.
kubectl get storageclass

# Discover recipes for this model (if any).
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py list \
  --query "${MODEL_SHORT_NAME}" --format table
```

**Decision points.**

- **AIConfigurator chain (G9, G10).** "Have you already run `dynamo-plan`
  (AIConfigurator) for this model + hardware + SLA combination? If yes,
  paste the DGD or the DGDR with a populated
  `status.profilingResults.selectedConfig`. If no, I will fall back to a
  workload interview + recipe-envelope filtering — but you lose
  AIConfigurator's configuration projection. AIConfigurator is integrated
  into the profiler / DGDR flow as a SDK-import inside the profiler Job
  (not a direct controller webhook), and it silently falls back to a
  memory-fit TP calculation when the model/hw/backend combination is not
  supported — so even a completed DGDR is not a guarantee that AIC ran.
  Which path do you want?"
  - If user opts in to AIC chain: extract the recipe constraints (backend,
    GPU count, parallelism) from the AIC output. These constrain Phase 2's
    candidate set.
  - If user opts out: continue with the interview. Record the opt-out
    explicitly in the output contract.

- **Workload class (G14, G16, G17, G21, G24).** "I need four numbers and
  one preference to filter the recipes. Average input sequence length
  (ISL)? Average output sequence length (OSL)? Expected prefix-reuse
  pattern (think: does your traffic repeat a system prompt, agent context,
  or RAG document — if so, roughly what fraction)? Request rate (requests
  per second at peak)? And: do you optimize for latency (TTFT/ITL) or for
  throughput at fixed SLO (goodput)?"

- **SLO declaration (G29-G32).** "Declare your SLO in AIPerf `--goodput`
  grammar — space-separated `KEY:VALUE` pairs where VALUE is in the
  metric's display unit (ms for latency, tokens/s for throughput). Example
  from the qwen3-32b recipe: `time_to_first_token:2000 inter_token_latency:25`.
  Supported metric tags include `time_to_first_token`, `inter_token_latency`,
  `request_latency`, `output_token_throughput_per_user`. Direction (lower
  vs higher is better) is inferred automatically. Unknown tags will raise
  `ValueError`. The skill always installs `aiperf==0.8.0`, regardless of
  what the recipe's `perf.yaml` pins. What is your SLO?"

- **Baseline (G22).** "Do you have a current endpoint we should measure as
  the baseline before deploying the new recipe? If yes, give me the URL —
  I will run AIPerf with the declared SLO against it first. If no, the
  output contract reports `baseline: null` and the PASS/FAIL is against
  the SLO only, not against a baseline delta."

**Refusal conditions** (decline even with user confirmation).

- HF token written to a file or printed to logs. Always via Kubernetes
  Secret.
- User declines AIConfigurator chain AND declines workload interview. Phase
  2 cannot pick without at least one signal.
- User declares an SLO with metric tags AIPerf doesn't recognise (skill
  hits `ValueError: Unknown metric tag(s) in --goodput: <tag>`).
- User insists on a recipe whose `model` and `framework` simply don't
  appear in the tree at the target commit.

**Verification gate.** AIC chain decision is recorded (chained-to or
opted-out); a workload class is captured; an SLO declaration parses
against the AIPerf 0.8.0 metric registry; cluster context, namespace, HF
token Secret, and storage class are confirmed.

---

## Phase 2: Pick

**Goal.** Filter the recipe tree to candidates that match the model,
framework (if AIC fixed it), hardware envelope, and workload class. Refuse
to deploy outside the recipe's tested envelope.

**Recipe tree shape (G1).** Recipes live at
`recipes/<model>/<framework>/<mode>/` for 14 model directories spanning 46
leaf recipes at this skill's `version`. Frameworks present (G2): `vllm`,
`sglang`, `trtllm`, `tokenspeed`. A small number of recipes break the
three-level shape — most notably `recipes/deepseek-v4/{deepseek-v4-flash,
deepseek-v4-pro}/<framework>/<mode>/` (extra sub-model level) and
`recipes/qwen3.6-35b/` (uses `deploy/{config}.yaml` + `hw/{target}.env` +
shared `perf.yaml` instead of the standard layout). See
[references/k8s-recipe-workflow.md](references/k8s-recipe-workflow.md) for
the exception list.

**Inputs.** All outputs from Phase 1.

**Commands** (SAFE):

```bash
# Filter recipes by model + framework + mode.
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py list \
  --query "${MODEL_SHORT_NAME}" \
  --framework "${FRAMEWORK}" \
  --format table

# Read the recipe's README and its perf.yaml to confirm tested envelope.
cat recipes/<model>/README.md
cat recipes/<model>/<framework>/<mode>/deploy.yaml
cat recipes/<model>/<framework>/<mode>/perf.yaml

# Confirm the recipe's tested GPU SKU is present on the cluster.
kubectl get nodes -L nvidia.com/gpu.product | grep "${RECIPE_GPU_SKU}"
```

### 2.1 Mode Taxonomy

18 unique modes across the tree (G3). The selector below is the canonical
mapping from workload signals to recipe mode. Every row has a verbatim
motivation quote in [references/k8s-recipe-workflow.md](references/k8s-recipe-workflow.md);
the inference-literature rationale lives in
[references/inference-literature.md](references/inference-literature.md).

| Workload signal | Mode candidates | Why |
|---|---|---|
| Baseline / simplest case | `agg`, `agg-round-robin` | Aggregated serving with round-robin routing. The baseline against which routing or disagg is measured. |
| High prefix overlap (≥ ~25-50% cache reuse), long inputs short outputs | `disagg-kv-router` | KV-aware routing leverages cache efficiency; disaggregation isolates decode from prefill injection. (Qwen3-32B README claims 36.64% cache efficiency justifies it.) |
| Single-GPU long-context / repeated prompts (KV memory pressure) | `agg-kvbm` | KV Block Manager offloads cold blocks to host memory; effective cache footprint extends beyond HBM. |
| Multimodal with repeated image inputs | `agg-embedding-cache` | vLLM ECConnector cache for prefix reuse on image embeddings. |
| EAGLE speculative decoding workload | `agg-eagle-kv-router`, `agg-eagle-round-robin`, `disagg-eagle-kv-router` | EAGLE + the relevant routing/disagg combination. Kimi-K2.5 README is the canonical reference. |
| Multi-node disagg | `disagg-multi-node`, `disagg-16gpu`, `disagg-gb200`, `disagg-b200` | Requires working KV transport (NIXL/UCX) between nodes. |
| Single-node disagg | `disagg`, `disagg-single-node`, `disagg-8gpu`, `disagg-eagle-kv-router` | No inter-node KV transfer. |

**Decision points.**

- **Mode selection (G12, G13, G14, G21, G22, G24).** "Workload signal:
  `<ISL=…, OSL=…, prefix_reuse=…, rate=…, opt=…>`. Recipe candidates that
  match the envelope: `<list>`. Recommended choice: `<mode>` because:
  `<rationale citing the table above>`. Should I proceed with `<mode>`?
  Alternative modes (less recommended, here's why): `<list with reasons>`."
  - Cite the regression conditions explicitly. Examples (each maps to a
    citation row in Section G):
    - **Don't pick `disagg-kv-router` if measured cache reuse is well
      below ~25%.** Mooncake's theoretical reuse ceiling for production
      workloads is ~50% (G12); below ~25% the KV-aware router's transfer
      cost rarely beats local recompute (G13). Use a non-KV mode.
    - **Don't pick a disagg mode if your workload is offline /
      non-latency-sensitive.** DistServe explicitly notes its
      effectiveness is compromised when the goodput target shifts to raw
      throughput (G14).
    - **Don't expect a KV router win below the "beneficial" threshold.**
      Dynamo's published criterion: TTFT improvement > 20% with no
      degradation, measurable prefix reuse (G21). Below 10% improvement,
      standard routing is better (G21).

- **Recipe-envelope match.** "The recipe is tested at `<SKU × GPU_count ×
  framework>`. Your cluster has `<observed SKU × count>`. Do they match?
  - If yes: proceed.
  - If no, the recipe's tested envelope does NOT hold for your hardware,
    and I am refusing to deploy under the recipe's claim unless you
    acknowledge this. Say `proceed at my own risk` to override; otherwise
    let's pick a different recipe or hand-author a DGD via `dynamo-deploy`."

**Refusal conditions.**

- No recipe exists for the requested `(model, framework, mode)` triple at
  the target commit.
- User's GPU SKU/count is outside the recipe's tested envelope AND user
  did not explicitly accept the risk.
- User requests `disagg-multi-node` but the cluster has only one GPU node.
- User asserts a workload-class signal that contradicts the chosen mode
  (e.g. "prefix reuse is zero" but picks `disagg-kv-router`).

**Verification gate.** Exactly one recipe path is selected; its tested
envelope is acknowledged (matched or risk-accepted); the workload class
maps to the mode per the table above; rationale citations to Section G are
recorded.

---

## Phase 3: Patch

**Goal.** Apply the minimum set of patches needed for this cluster, run
optional workstation pre-validation for agg-* modes, server-side dry-run
before any live apply.

**Inputs.** Selected recipe path; cluster context; AIPerf 0.8.0 SLO from
Phase 1.

**Commands** (SAFE first, then MUTATING in 3.3).

```bash
# Working-tree state: ensure the local recipe is clean before patching.
git status --short recipes/<model>/<framework>/<mode>/

# Validate the recipe lints and parses.
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py validate \
  recipes/<model>/<framework>/<mode>
```

### 3.1 Patch Matrix

Patch only these fields. Do not reformat whole YAML files. Each patch is
idempotent — run twice without effect.

| Field | When to patch | Source |
|---|---|---|
| `storageClassName` | Recipe's value doesn't match the cluster's available classes. | `kubectl get storageclass` |
| `image` repository/tag | Recipe pins a placeholder or stale tag. | Per-release Dynamo image tag (e.g. `nvcr.io/nvidia/ai-dynamo/dynamo-vllm-runtime:1.2.0`). |
| Model path / mount path | Recipe assumes a path you don't have. | User. |
| GPU resource requests/limits | Recipe pinned a count your cluster can't satisfy; if you're shrinking, this voids the tested envelope. | `kubectl get nodes -L nvidia.com/gpu.product`. |
| Frontend `DYN_ROUTER_MODE` env | You're testing a routing variant (see G4: round-robin / random / power-of-two / kv / direct / least-loaded / device-aware-weighted; default round-robin). | Reference [references/k8s-recipe-workflow.md](references/k8s-recipe-workflow.md). |
| `apiVersion: nvidia.com/v1alpha1` → `v1beta1` | **Don't.** The recipe was tested at `v1alpha1` (G38). Conversion-webhook details live in `docs/kubernetes/api-reference.md`, not in `recipes/`. Leaving the recipe's pinned apiVersion preserves the tested envelope. |
| HF token | **Never patched into files.** Always via `envFromSecret: { name: hf-token-secret }`. The recipe already uses this pattern. |
| AIPerf version pin (`perf.yaml` `pip install aiperf==…`) | Override to `aiperf==0.8.0` regardless of recipe pin. Skill design decision (locked 2026-05-22). If the recipe's base image conflicts with 0.8.0 (DYN-2878-style transformers-version conflict), warn and surface the conflict in the output contract; do not silently fall back. |

### 3.2 Optional Workstation Pre-Validation (agg-* modes only)

For `agg-*` recipes — and **only** for `agg-*` — the skill offers a 60-
second workstation shakedown via `dynamo-serve` before the cluster apply:

```bash
# Workstation pre-val (agg-* only). Validates: model loads, endpoint
# responds, declared SLO grammar is parseable. Does NOT validate cluster
# perf.
python3 -m dynamo.<framework> --model "${MODEL_NAME}" &
SERVE_PID=$!
sleep 30
python3 .agents/skills/dynamo-optimize/scripts/measure_slo.py \
  --url http://localhost:8000 \
  --slo "${SLO}" \
  --duration 60 \
  --artifact-dir /tmp/dynamo-optimize-preval-$(date +%s) \
  --mode preval
kill "${SERVE_PID}"
```

**Why agg-* only.** For aggregated single-node modes, workstation perf
correlates loosely with cluster perf at the same GPU count. For `disagg-*`
modes, the correlation **breaks** — disaggregation depends on the cross-
node KV transfer path (NIXL/UCX), which the workstation cannot exercise.
Running a workstation pre-val on a disagg recipe produces misleading
numbers, so the skill refuses.

**Refusal:** if the user requests pre-val on a `disagg-*` recipe, the
skill declines and explains the correlation gap.

### 3.3 Server-Side Dry-Run

Before the real `kubectl apply`, run the API-server-side validation:

```bash
kubectl apply --dry-run=server -f recipes/<model>/<framework>/<mode>/deploy.yaml
kubectl apply --dry-run=server -f recipes/<model>/<framework>/<mode>/perf.yaml
```

If the API server rejects a manifest, **stop**. Per the
`dynamo-deploy/SKILL.md` Phase 3 pattern, the skill does not retry
blindly — it surfaces the reject reason and asks the user whether to edit
the patch, debug the webhook, or switch versions.

### 3.4 Capture Baseline (if requested in Phase 1)

If the user gave a baseline endpoint URL, run AIPerf with the declared
SLO against it now, before applying the new recipe:

```bash
python3 .agents/skills/dynamo-optimize/scripts/measure_slo.py \
  --url "${BASELINE_URL}" \
  --slo "${SLO}" \
  --artifact-dir /tmp/dynamo-optimize-baseline-$(date +%s) \
  --mode baseline
```

The script writes `baseline.json` with the metric values from
`profile_export_aiperf.json` (G28: every metric is a `JsonMetricResult`
with `{unit, avg, p1, p5, p10, p25, p50, p75, p90, p95, p99, min, max, std,
count, sum}`). The output contract carries this snapshot for the final
delta comparison in Phase 4.

**Decision points.**

- **Image tag (MUTATING).** "Patching `spec.services.Frontend.image` from
  `<recipe-tag>` to `<your-tag>`. The recipe's tested envelope was
  measured at `<recipe-tag>`; this patch may change measured perf. OK to
  proceed?"
- **GPU count (MUTATING).** "Recipe is tested at `<recipe-count>`; you
  have `<cluster-count>`. Shrinking voids the tested envelope; expanding
  doesn't. OK to patch to `<cluster-count>`?"
- **AIPerf override.** "Recipe pins `aiperf==<recipe-pin>`; this skill
  always pins `aiperf==0.8.0`. The 0.8.0 release adds e2e_output_token_throughput,
  t-digest aggregation, and multi-run confidence reporting — none of
  which the recipe-pinned version has. If your recipe's base image ships
  a transformers-version that conflicts (DYN-2878-style), I'll surface
  the conflict, not silently fall back. OK to override the pin?"

**Refusal conditions.**

- User requests workstation pre-val on a `disagg-*` recipe.
- HF token would land in a file (recipe references a Secret; never inline
  the value).
- User insists on dropping the recipe's `envFromSecret` pattern.

**Verification gate.** All patches applied idempotently; recipe files
parse as YAML; both `deploy.yaml` and `perf.yaml` pass
`kubectl apply --dry-run=server`; baseline captured if requested;
workstation pre-val passed if invoked.

---

## Phase 4: Apply

**Goal.** Apply the patched recipe to the cluster, wait for the deployment
to become healthy, measure against the declared SLO, produce the per-
dimension PASS/FAIL output contract.

**Inputs.** Patched recipe path; SLO declaration; baseline snapshot (if
captured in 3.4).

**Commands** (MUTATING in 4.1-4.3, SAFE in 4.4-4.5):

### 4.1 Apply Model Cache

**Decision (MUTATING):** "Applying `recipes/<model>/model-cache/` will
start the model-download Job and create the `model-cache` PVC. The
download pulls 10s of GB from Hugging Face — bandwidth and storage cost.
OK to proceed?"

Wait for explicit yes. Then:

```bash
kubectl apply -f recipes/<model>/model-cache/ -n "${NAMESPACE}"
# Wait for the download Job to complete. Canonical timeout is 600s per
# the qwen3-32b README (G25). Anish's substrate used 6000s; the in-tree
# README is canonical, but very large models can exceed 600s — bump if
# you know the model is huge.
kubectl wait --for=condition=Complete job/model-download \
  -n "${NAMESPACE}" --timeout=600s
```

### 4.2 Apply DGD

**Decision (MUTATING):** "Applying `deploy.yaml` will create the DGD;
operator pulls images, schedules Frontend + Workers, starts serving on
`<service>:<port>`. OK to proceed?"

Wait for explicit yes. Then:

```bash
kubectl apply -f recipes/<model>/<framework>/<mode>/deploy.yaml -n "${NAMESPACE}"

# Wait for pods using the canonical label selector (G26).
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD_NAME}" \
  -n "${NAMESPACE}" --timeout=1200s
```

### 4.3 Smoke Test

```bash
# Port-forward to the frontend.
kubectl port-forward svc/"${DGD_NAME}"-frontend 8000:8000 -n "${NAMESPACE}" &

# Wait for /v1/models to be non-empty (per the dynamo-deploy known-issue
# entry: DGD Ready does NOT imply /v1/models populated — workers must
# register via NATS first).
until curl -s http://localhost:8000/v1/models \
  | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")' \
  2>/dev/null; do
  echo "waiting for model registration..."
  sleep 10
done

# Single chat completion.
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}]}" \
  | python3 -m json.tool
```

### 4.4 Validate Against SLO (and baseline if captured)

```bash
python3 .agents/skills/dynamo-optimize/scripts/measure_slo.py \
  --url http://localhost:8000 \
  --slo "${SLO}" \
  --artifact-dir /tmp/dynamo-optimize-postdeploy-$(date +%s) \
  --baseline /tmp/dynamo-optimize-baseline-*/baseline.json  # optional
```

The script emits `PASS|<dim>|<value> SLO=<threshold>` or
`FAIL|<dim>|<value> SLO=<threshold>` per declared metric, plus the
delta vs baseline if `--baseline` was passed. Stdout follows the
`PASS|name|detail` schema (per `dynamo-skill-author/references/body-shape.md`).

### 4.5 Output Contract (G39)

Always emit, even when refusing:

| Field | Value |
|---|---|
| `selected_recipe` | `recipes/<model>/<framework>/<mode>/` |
| `rationale` | Workload-class + envelope-match reason from Phase 2. |
| `ai_configurator_chain` | `chained: true` / `chained: false (opted_out: <reason>)`. |
| `workload_class` | ISL, OSL, prefix_reuse, rate, opt_target. |
| `slo` | The declared `--goodput` string + AIPerf version (`0.8.0`). |
| `baseline` | The captured snapshot, or `null`. |
| `patches_applied` | List of (field, before, after). |
| `workstation_preval` | `agg-only: <result>` / `disagg: skipped (correlation)`. |
| `dry_run_result` | `passed` / `failed: <reason>`. |
| `post_deploy_slo` | Per-metric PASS/FAIL vs SLO. |
| `delta_vs_baseline` | Per-metric percentage delta (if baseline present). |
| `rollback_action` | `none` / `kubectl delete dgd <name>` / `kubectl patch ...`. |
| `endpoint` | URL of the frontend service. |
| `smoke_test` | `passed` / `failed: <reason>`. |
| `chain_to` | `dynamo-benchmark` if user wants the full `perf.yaml` benchmark; `dynamo-troubleshoot` if smoke test failed. |

**Decision points.**

- **Any SLO dimension FAILs.** "The declared SLO failed on `<dim>`:
  measured `<value>`, threshold `<threshold>`. Rollback options:
  - `kubectl delete dgd <name>` (revert to before this Phase 4).
  - `kubectl patch dgd ...` (apply a smaller patch — e.g. revert
    `DYN_ROUTER_MODE` if you set it to `kv` and the workload didn't have
    enough prefix reuse, per G21).
  - Switch recipe mode (loop back to Phase 2 with the new workload class
    signal you just learned).
  - Accept the SLO miss explicitly and continue.
  Which do you want?"

- **Baseline regression.** "Post-deploy `<dim>` is `<delta>%` slower than
  baseline. Mode is `<mode>`. The closest published regression condition
  is `<G12 / G14 / G17 / G21 — whichever applies>`. Want to roll back, or
  proceed?"

**Refusal conditions.**

- **Claim "optimization" with no baseline AND no SLO from `dynamo-plan`.**
  The output reports a PASS/FAIL against the declared SLO; the skill does
  not claim it is "better" than anything if neither a baseline nor an
  AIConfigurator-driven SLO is present.
- **Declare success if any declared SLO dimension fails.** The output
  contract reports FAIL per dimension; the skill does not conflate
  partial success with success.
- **Silently exit on regression.** The skill always emits the output
  contract; "regression detected" is a contract field, not a hidden
  failure mode.
- **Compare across different AIPerf versions.** Skill always pins
  `aiperf==0.8.0` (locked decision); if a baseline was captured against a
  different version, the comparison is labeled "different AIPerf
  versions; comparison invalid" in `delta_vs_baseline`.
- **Skip the model-download `Complete` wait.** The skill always waits for
  the Job to complete before applying the DGD (avoids pod CrashLoop on
  missing weights).

**Verification gate.** DGD reports healthy terminal state; `/v1/models`
returns the model; chat completion smoke test passes; AIPerf with the
declared SLO produces a PASS or an explicit FAIL with rollback decision
recorded. Output contract emitted.

---

## Prerequisites

Phase 1 verifies the full readiness state. Short list:

- `dynamo-platform` Helm chart installed and operator pod Ready. (Refuses
  to Phase 2 otherwise — refers to `dynamo-deploy/SKILL.md` Phase 1 for
  the platform verification.)
- A target Kubernetes namespace with `kubectl` access.
- Hugging Face token Secret named `hf-token-secret` (or per-recipe
  override) in the target namespace, for gated models.
- A storage class for the model-cache and perf-cache PVCs.
- (Optional, soft-recommended) An AIConfigurator run via `dynamo-plan`,
  producing a DGD or a DGDR with populated `status.profilingResults.
  selectedConfig`.
- A clean working tree in `ai-dynamo/dynamo` (or willingness to patch
  recipe files locally; the skill records every patch in the output
  contract).

---

## Limitations

What this skill does NOT cover:

- Does NOT install the `dynamo-platform` chart. See `dynamo-deploy` Phase 1.
- Does NOT do AIConfigurator's job. It chains to `dynamo-plan` output;
  it does not search the configuration space itself.
- Does NOT do the full `perf.yaml` benchmark with the recipe's dataset
  trace. That is `dynamo-benchmark`. This skill runs a short AIPerf
  shakedown with the declared `--goodput` SLO, not the recipe's full
  benchmark companion.
- Does NOT model fault tolerance behavior. See `dynamo-troubleshoot`.
- Does NOT write Hugging Face tokens to files. Always Secret.
- Does NOT convert `nvidia.com/v1alpha1` to `v1beta1` mid-deploy. The
  recipe's pinned apiVersion is preserved.
- Does NOT validate the AIConfigurator chain end-to-end. It accepts the
  user-pasted DGD/DGDR at face value.

---

## Troubleshooting

When a step in this workflow fails:

- **Per-skill known patterns** live in [references/known-issues.md](references/known-issues.md). Walk that list first.
- **Cross-skill day-2 patterns** (worker crashloops, conversion-webhook timeouts, KV-transfer fallback) are owned by `dynamo-troubleshoot`.
- **Recipe-envelope mismatch** (the SKU/count doesn't match the recipe's tested envelope) usually surfaces as a SchedulingFailed event. See [references/known-issues.md](references/known-issues.md) entry "Recipe envelope mismatch".
- **AIPerf 0.8.0 vs recipe-pinned older AIPerf** can surface as `ValueError: Unknown metric tag(s) in --goodput: <tag>` if the user copied an SLO from a recipe authored against a pre-0.8.0 tag set. See [references/slo-shape.md](references/slo-shape.md) for the supported-tag inventory at 0.8.0.

---

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/recipe_tool.py` | List, query, and validate recipes against the local tree. Anish's substrate from PR #9782, landed verbatim. | `list [--query <substr>] [--framework <fw>] [--mode <mode>] [--format table\|json]` ; `validate <path>` |
| `scripts/measure_slo.py` | Wrap an AIPerf 0.8.0 invocation with the declared `--goodput` SLO; emit PASS/FAIL per dimension; optional delta vs baseline. | `--url <endpoint> --slo <goodput-string> --artifact-dir <path> [--baseline <baseline.json>] [--mode preval\|baseline\|postdeploy] [--duration <seconds>]` |

Both scripts emit the structured `PASS|<check>|<detail>` / `FAIL|<check>|<detail>` stdout schema per `dynamo-skill-author/references/body-shape.md` (the Python equivalent of the bash `pass/fail/warn` pattern). They exit non-zero on any FAIL. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/recipe_tool.py", args=["list", "--query", "qwen3", "--framework", "vllm", "--format", "table"])
run_script("scripts/measure_slo.py", args=["--url", "http://localhost:8000", "--slo", "time_to_first_token:2000 inter_token_latency:25", "--artifact-dir", "/tmp/dynamo-optimize-postdeploy"])
```

Equivalent direct invocation:

```bash
python3 scripts/recipe_tool.py list --query qwen3 --framework vllm --format table
python3 scripts/measure_slo.py --url http://localhost:8000 --slo "time_to_first_token:2000 inter_token_latency:25" --artifact-dir /tmp/dynamo-optimize-postdeploy
```

---

## Refusal Conditions

The skill refuses, even with user confirmation, when:

- The HF token would land in a file, environment variable in a manifest, or log line. Always via Kubernetes Secret.
- The user requests a recipe that doesn't exist for the target `(model, framework, mode)` at this commit.
- The user's cluster GPU SKU/count doesn't match the recipe's tested envelope AND the user has not explicitly said "proceed at my own risk".
- The user requests workstation pre-validation on a `disagg-*` recipe (correlation breaks).
- The user requests a comparison across different AIPerf versions (e.g. baseline at 0.6.0, post-deploy at 0.8.0). The comparison is invalid.
- The user requests conversion from `nvidia.com/v1alpha1` to `v1beta1` mid-deploy on a recipe that ships at `v1alpha1`.
- The user declares an SLO with metric tags AIPerf 0.8.0 doesn't recognise (skill would hit `ValueError`).
- The user opts out of both AIConfigurator chain and workload interview. Phase 2 cannot filter without at least one signal.

---

## Cross-Skill Referencing

This skill does not absorb workflows owned by sibling skills:

| Question | Skill to read |
|---|---|
| Where do I get the AIConfigurator output for the chain in Phase 1? | `dynamo-plan/SKILL.md` (soft prereq) |
| The recipe doesn't fit my hardware — how do I hand-author a DGD? | `dynamo-deploy/SKILL.md` (Phase 2.3 manual DGD authoring) |
| The smoke test failed in Phase 4. What now? | `dynamo-troubleshoot/SKILL.md` |
| I want the full benchmark with the recipe's dataset trace, not just `--goodput` validation. | `dynamo-benchmark/SKILL.md` |
| What's the API-surface configuration for the Frontend / DynamoModel CRs? | `dynamo-frontend/SKILL.md` |
| The `dynamo-platform` chart isn't installed. | `dynamo-install` (deferred) → today, see `dynamo-deploy/SKILL.md` Phase 1. |

If any of these surfaces during Phase 1-4, pause and hand off; do not absorb the workflow.

---

## References and Scripts

- [references/k8s-recipe-workflow.md](references/k8s-recipe-workflow.md) — Substrate from PR #9782, edited with citation hooks; includes the recipe tree layout, mode taxonomy detail, exception list (`deepseek-v4/*`, `qwen3.6-35b/`), and the canonical preflight/deploy/smoke command sequences.
- [references/slo-shape.md](references/slo-shape.md) — AIPerf 0.8.0 output schema (`JsonMetricResult` fields), `--goodput` grammar, supported metric tags, common pitfalls (`tokens_per_second` is NOT a real tag), per-dimension PASS/FAIL semantics, rollback policy.
- [references/inference-literature.md](references/inference-literature.md) — Regression-condition summaries from Mooncake, DistServe, SplitWise, vLLM, SGLang, Orca. Each citation traces to a verbatim quote in `dynamo-skills/corpus/papers/<short>/extracts.yaml`. Use this when justifying a mode choice or refusing a mode.
- [references/known-issues.md](references/known-issues.md) — Stable issue patterns: recipe-envelope mismatch, AIPerf-version-pin conflict (DYN-2878 transformers), DGDR immutability, AIC silent naive-fallback, `/v1/models` empty after Ready.
- [scripts/recipe_tool.py](scripts/recipe_tool.py) — Recipe discovery + lightweight validation. Anish's substrate from PR #9782, landed verbatim.
- [scripts/measure_slo.py](scripts/measure_slo.py) — AIPerf 0.8.0 wrapper that emits PASS/FAIL per declared metric and an optional delta vs baseline.
