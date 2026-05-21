---
name: dynamo-plan
description: >-
  Plan a NVIDIA Dynamo deployment before any container starts. Pick the
  right path between AIConfigurator (offline planning), DGDR
  searchStrategy (in-cluster profiling), and recipe selection; set SLA
  targets, parallelism strategy, and KV-cache budget for a model and
  hardware combination. Use when sizing a Dynamo deployment, picking
  TP/PP/EP parallelism, choosing rapid vs thorough profiling, setting
  TTFT/ITL SLAs, or deciding whether a recipe already fits.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - planning
  - sla
  - parallelism
  - aiconfigurator
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "plan deployment", "AIConfigurator", "searchStrategy", "SLA targets", "TP/PP/EP"
- Level 2 (this file): 4-phase planning workflow
- Level 3: references/ for AIConfigurator workflow, planner concepts, decision matrix
           scripts/  for the planning preflight

Scope: pre-deployment sizing decisions for a Dynamo workload
Out of scope: model optimization (see dynamo-optimize), the actual deploy (see dynamo-deploy),
              day-2 troubleshooting (see dynamo-troubleshoot), benchmarking (see dynamo-benchmark).
-->

# Dynamo Plan (Pre-Deployment Sizing)

Pick a planning path and produce the inputs the operator needs to start a
real deployment. Three valid paths exist: AIConfigurator (offline,
fastest), DGDR `searchStrategy` (in-cluster, more accurate), and recipe
selection (zero-effort if one fits).

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Gather   → model, workload (ISL/OSL), SLA targets, hardware available
Phase 2: Pick     → AIConfigurator | DGDR rapid | DGDR thorough | recipe
Phase 3: Execute  → run AIConfigurator | submit DGDR | select recipe
Phase 4: Capture  → write the planning result into a DGDR / DGD / recipe path
```

---

## Command Safety

This skill is mostly inspection and configuration. Two MUTATING cases:
submitting a DGDR (which kicks off a profiling Job and consumes GPU
during `thorough`), and starting an AIConfigurator run that hits a
remote API.

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `kubectl apply -f <dgdr.yaml>` | Creates a DGDR; operator may start a profiling Job that consumes GPU. `rapid` ~30s; `thorough` 2-4h. |
| `aiconfigurator plan ...` | Sends model/SLA to an NVIDIA inference endpoint; consumes API quota; takes ~30-90s. |

### SAFE — no confirmation needed

```
kubectl get dgdr <name> -n <ns>
kubectl describe dgdr <name> -n <ns>
ls ai-dynamo/dynamo/recipes/
cat recipes/<model>/<framework>/<config>/deploy/*.yaml
nvidia-smi
kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count
```

---

## Phase 1: Gather

**Goal.** Collect the inputs every planning path needs.

**Inputs**:

| Input | How to get it |
|---|---|
| Model identifier | HuggingFace ID (e.g. `Qwen/Qwen3-0.6B`) or local checkpoint path |
| Workload — ISL (input sequence length) | Mean tokens per request from production logs, or a workload assumption |
| Workload — OSL (output sequence length) | Mean output tokens; pick the realistic distribution mean, not the max |
| SLA — TTFT (time-to-first-token) | User-visible target; if unset, planner uses defaults |
| SLA — ITL (inter-token latency) | Streaming throughput target |
| Hardware — GPU SKU | One of the 16 SKUs per [references/decision-matrix.md](references/decision-matrix.md) |
| Hardware — count and topology | numGpusPerNode, totalGpus, interconnect (NVLink / PCIe / RoCE / IB) |
| Concurrency or request rate | One required when planner is disabled |

**Decision:** Some of these inputs are missing. Without them, planning produces a generic config that may not meet the user's actual SLA. Should I proceed with defaults or pause for the missing values?

Wait for explicit choice.

**Verification gate.** Model, workload (at least ISL/OSL), and GPU SKU are known.

---

## Phase 2: Pick a Planning Path

**Goal.** Choose the planning approach that matches the inputs and the time budget.

Three paths:

| Path | Cost | Accuracy | Use when |
|---|---|---|---|
| **Recipe** | Zero | Known-good for the tested model+hardware combo | A `recipes/<model>/<framework>/<config>/` exists for this exact combination |
| **AIConfigurator** | ~30-90s; API quota | Good (simulator-driven) | Recipe doesn't fit; need fast iteration; pre-cluster planning |
| **DGDR `rapid`** | ~30s in-cluster | Good (AIC simulator) | Cluster already up; want operator to author the DGD |
| **DGDR `thorough`** | 2-4h; consumes GPU | Best (real-GPU sweeps) | Production rollout; the planning cost amortises |

**Decision:** Based on the inputs and time budget, the recommended path is `<X>`. Other paths remain valid. Pick one — do not run multiple in parallel for the same model unless reconciling competing recommendations.

Wait for explicit choice.

**Verification gate.** A single path is selected.

---

## Phase 3: Execute the Plan

### 3.A Recipe Path

```bash
# Locate the recipe in the dynamo source tree.
RECIPE=$(ls -d /Users/dagil/dynamo/recipes/<model>/<framework>/<config>/ 2>/dev/null)
test -d "$RECIPE" || echo "no recipe found for <model>/<framework>/<config>"

# Inspect the published benchmark numbers and the DGD.
cat $RECIPE/README.md
cat $RECIPE/deploy/*.yaml
```

Output: the recipe path. Phase 4 will copy / reference it.

### 3.B AIConfigurator Path

```bash
# Pin per release: ai-dynamo[mocker] extra in pyproject.toml.
pip install 'ai-dynamo[mocker]==<release>'

aiconfigurator plan \
  --model <hf-id> \
  --backend <auto|vllm|sglang|trtllm> \
  --hardware.gpuSku <sku> \
  --hardware.numGpusPerNode <n> \
  --workload.isl <isl> --workload.osl <osl> \
  --sla.ttft <ms> --sla.itl <ms> \
  --output planning.json
```

Output: `planning.json` with the recommended parallelism (TP/PP/EP),
replica count, batch size, and KV-cache config.

### 3.C DGDR Path (`rapid` or `thorough`)

Submit a DGDR; the operator runs the profiling Job and writes the
recommended config to status.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: <model>-plan
spec:
  model: <hf-id>
  backend: <auto|vllm|sglang|trtllm>
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:<release>"
  searchStrategy: rapid          # or thorough
  hardware:
    numGpusPerNode: <n>
    gpuSku: <sku>
  workload:
    isl: <isl>
    osl: <osl>
  sla:
    ttft: <ms>
    itl: <ms>
  autoApply: false               # PLAN mode — review before deploy
```

Apply, then watch:

```bash
kubectl apply -f <model>-plan.yaml
kubectl get dgdr <model>-plan -n <ns> -w
```

When phase reaches `Ready`, the recommended config is at:

```bash
kubectl get dgdr <model>-plan -n <ns> \
  -o jsonpath='{.status.profilingResults.selectedConfig}' \
  | python3 -m json.tool
```

`autoApply: false` is deliberate — Phase 4 reviews the result before
the DGD is created. See [references/dgdr-planner-shape.md](references/dgdr-planner-shape.md).

**Decision (MUTATING — `thorough` only):** Thorough sweeps reserve the
requested GPUs for 2-4h. Should I proceed?

Wait for explicit yes for `thorough` mode.

**Verification gate.** A planning result is produced (`planning.json`,
recipe path, or DGDR `selectedConfig`).

---

## Phase 4: Capture the Plan

**Goal.** Hand the planning result off to `dynamo-deploy` in a form it
can consume.

| Source | Hand-off |
|---|---|
| Recipe | Pass the recipe path to `dynamo-deploy` Phase 2.4 (Recipe-Based Authoring). |
| AIConfigurator `planning.json` | Convert to a DGDR with `autoApply: true`, or to a DGD manifest. |
| DGDR `selectedConfig` (`autoApply: false`) | Flip `autoApply: true` on the same DGDR to deploy; or extract the DGD spec from status and apply directly. |

Run the bundled summary script:

```bash
bash scripts/summarize-plan.sh -i planning.json
# or
bash scripts/summarize-plan.sh -d <dgdr-name> -n <ns>
```

The script emits a one-screen summary of the planning decision: chosen
parallelism, replica count, KV-cache budget, expected TTFT/ITL, and the
hand-off command for `dynamo-deploy`.

**Verification gate.** The user has a single artifact (recipe path,
`planning.json`, or `selectedConfig` JSON) ready to hand off.

---

## Refusal Conditions

- The model is not visible to the planner (HF gating, private mirror) and
  no checkpoint path is provided.
- The hardware specified is not present on the target cluster (per
  `nvidia.com/gpu.product` node labels) AND the namespace is not a dry-
  run / test environment.
- SLA targets are physically infeasible for the named hardware (e.g. TTFT
  under 5ms on H100 PCIe). The skill flags this and refuses to forward.

---

## Cross-Skill Referencing

| Need | Sibling | Hand-off |
|---|---|---|
| Pre-quantization model optimization | [dynamo-optimize](../dynamo-optimize/SKILL.md) | Plan against the unoptimized model first; re-plan if optimization changes the shape. |
| Actual deployment | [dynamo-deploy](../dynamo-deploy/SKILL.md) | Pass the captured artifact from Phase 4. |
| Validating the planning result against real load | [dynamo-benchmark](../dynamo-benchmark/SKILL.md) | After deploy, benchmark to verify SLA. |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- A target model identified (HuggingFace ID or local checkpoint path).
- Workload assumptions: mean ISL / OSL, target concurrency or request rate.
- Hardware inventory: GPU SKU and count on the target cluster (auto-detected in cluster-wide operator installs; required explicitly in namespace-restricted installs per).
- Optional: an SLA envelope (`ttft`, `itl`, or `e2eLatency`).

---

## Limitations

What this skill does NOT cover:

- Does NOT deploy or run a workload. Output is a planning artifact (`planning.json`, recipe path, or DGDR `selectedConfig`) consumed by `dynamo-deploy`.
- Does NOT measure real performance. The output is profiler-driven; verify with `dynamo-benchmark`.
- Does NOT cover model optimization. For quantization, see `dynamo-optimize`.

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
| `scripts/summarize-plan.sh` | One-screen summary of a planning result; hand-off command for `dynamo-deploy` | `-i <planning.json>` OR `-d <dgdr-name> [-n <ns>]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/summarize-plan.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/summarize-plan.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/dgdr-planner-shape.md](references/dgdr-planner-shape.md) —
  The DGDR fields most relevant to planning: `searchStrategy`,
  `hardware`, `workload`, `sla`, `features.planner`.
- [references/decision-matrix.md](references/decision-matrix.md) —
  Path-selection matrix with worked examples.
- [references/known-issues.md](references/known-issues.md) — Planning-
  specific stable issue patterns.
- [scripts/summarize-plan.sh](scripts/summarize-plan.sh) — Emit a one-
  screen summary of a planning result for hand-off.
