---
name: dynamo-optimize
description: >-
  Optimize a model for NVIDIA Dynamo deployment via NVIDIA TensorRT Model
  Optimizer (modelopt). Pick a quantization technique (FP8, FP4, INT8,
  weight-only, AWQ), produce an optimized checkpoint, and verify it is
  loadable by a Dynamo worker. Use when quantizing a model for Dynamo,
  choosing between FP8 and FP4, preparing a checkpoint for NVFP4 recipes,
  validating an optimized checkpoint, or evaluating accuracy after
  quantization.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - quantization
  - modelopt
  - fp8
  - fp4
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "quantize", "FP8", "FP4", "modelopt", "TensorRT Model Optimizer"
- Level 2 (this file): 4-phase optimization workflow
- Level 3: references/ for technique matrix, modelopt CLI, quantization-aware recipes
           scripts/  for the post-quantization validator

Scope: pre-deployment model optimization via NVIDIA TensorRT Model Optimizer
Out of scope: deployment of the optimized checkpoint (see dynamo-deploy),
              planning (see dynamo-plan), benchmarking accuracy at scale (see dynamo-benchmark).
-->

# Dynamo Optimize (Quantization via Model Optimizer)

Produce an optimized model checkpoint that a Dynamo worker can load via
the backend's `--load-format` (vLLM) or `quant_config.json` (TensorRT-
LLM) path. The skill defers to NVIDIA TensorRT Model Optimizer
(`modelopt`) for the actual quantization; it owns the surrounding
workflow (technique selection, checkpoint validation, recipe alignment).

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Pre-Check → model accessible, modelopt installed, calibration data ready
Phase 2: Choose    → quantization technique (FP8 | NVFP4 | INT8 | weight-only | AWQ)
Phase 3: Run       → modelopt CLI produces an optimized checkpoint
Phase 4: Validate  → checkpoint loads in a Dynamo worker; sample accuracy probe
```

---

## Command Safety

Model optimization is local CPU/GPU work. Two DESTRUCTIVE-tier cases:
overwriting an existing checkpoint, and pushing the result to a shared
registry.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `modelopt quantize ... --output <dir>` (existing dir) | Overwrites the existing checkpoint in `<dir>`. Source weights are not modified, but the output dir contents are replaced. |
| `huggingface-cli upload <repo> <local>` | Uploads the optimized checkpoint to HuggingFace. Becomes immediately visible to anyone with read access to `<repo>`. |
| `aws s3 cp --recursive <local> s3://<bucket>/<path>` | Uploads to an S3 mirror; same visibility concern. |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `modelopt quantize ...` (new output dir) | Allocates GPU memory; quantization can take minutes (FP8 weight-only) to hours (NVFP4 with calibration). |
| `pip install --upgrade nvidia-modelopt` | Updates the modelopt package; may pull in new CUDA dependencies. |

### SAFE — no confirmation needed

```
modelopt --version
modelopt quantize --help
huggingface-cli download <repo> --local-dir <dir>
ls -la <checkpoint-dir>
python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('<path>'))"
```

---

## Phase 1: Pre-Check

**Goal.** Confirm the source model is accessible, modelopt is installed,
and any calibration data the chosen technique needs is on disk.

**Inputs**:

| Input | How to get it |
|---|---|
| Source model | HuggingFace ID (gated models need an HF token) or local path |
| modelopt | `pip install nvidia-modelopt` (or the version pinned by the target recipe) |
| Calibration dataset | C4, Pile, or a small slice of your production prompts. Needed for AWQ and PTQ techniques. Not needed for weight-only quantization. |
| Target hardware | The deployment hardware drives the precision choice (see Phase 2) |

**Commands** (SAFE):

```bash
# Verify modelopt.
python3 -c "import modelopt; print(modelopt.__version__)"

# Confirm the source model loads.
python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('<model-id-or-path>'))"

# Disk space for the output. Optimized checkpoints can be 50-80% smaller than fp16/bf16 source.
df -h <output-parent-dir>
```

**Decision point** — if a Dynamo recipe already exists for this model at
this hardware tier and quantization level, **prefer the recipe** over a
hand-rolled optimization. Recipes ship tested calibration data and known
accuracy numbers.

**Verification gate.** modelopt installed, source model loads, output
dir has space.

---

## Phase 2: Choose a Quantization Technique

**Goal.** Pick the technique that matches the target hardware, accuracy
tolerance, and deployment shape.

See [references/technique-matrix.md](references/technique-matrix.md) for
the full decision table. Summary:

| Technique | Best for | Hardware | Accuracy impact | Compute cost (quantize) |
|---|---|---|---|---|
| Weight-only FP8 | Quick wins, no calibration data | H100/H200/B200 | Minimal | Minutes |
| FP8 PTQ | Production deploys with calibration | H100/H200/B200 | Very low | 10s of minutes |
| NVFP4 | Maximum compression, Blackwell-tier hardware | B200/GB200 | Low (with calibration) | 1-2 hours |
| INT8 weight-only | Pre-Hopper hardware | A100/L40s | Modest | Minutes |
| AWQ (Activation-aware Weight Quantization) | Larger accuracy-sensitive models | Any | Lowest accuracy impact | 2-4 hours |

**Decision:** Several techniques are viable for `<model>` on `<hardware>`. Recommended: `<X>`. Alternatives: `<Y, Z>`. Confirm.

Wait for explicit choice.

**Verification gate.** A single technique selected.

---

## Phase 3: Run the Quantization

**Goal.** Produce an optimized checkpoint at the chosen precision.

The exact CLI varies by technique. Authoritative reference: NVIDIA
TensorRT Model Optimizer documentation. See
[references/modelopt-cli.md](references/modelopt-cli.md) for the verbatim
commands per technique.

Generic shape:

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode <fp8|nvfp4|int8|awq> \
  --calib-dataset <path>           # required for PTQ / AWQ
  --calib-samples 512 \             # typically 128-512
  --output <output-dir> \
  --device cuda
```

**Decision (MUTATING):** Quantization will run for `<estimated duration>`
and consume `<n>` GPU(s). Proceed?

Wait for explicit yes.

Output: `<output-dir>` containing the quantized weights, the tokenizer
files (copied verbatim from source), the model config, and a
`quant_config.json` describing the quantization (the format Dynamo
backends consume).

**Verification gate.** Quantization completes without error; output dir
contains weights, config, and `quant_config.json`.

---

## Phase 4: Validate the Optimized Checkpoint

**Goal.** Confirm the checkpoint loads in a Dynamo worker and produces
non-degraded output.

Run the validator:

```bash
bash scripts/validate-optimized.sh -m <output-dir> -b vllm
```

The script applies the `pass/fail/warn` pattern from:

1. Output dir present.
2. `config.json`, `tokenizer.json`, and weight shards present.
3. `quant_config.json` parses and matches the chosen technique.
4. A Dynamo worker can load the checkpoint without crashing
   (`python3 -m dynamo.vllm --model <output-dir> --load-format auto` with
   a 30 s timeout; success on first registered model).
5. A canned prompt returns a non-empty response (sample accuracy probe).

Compare to the recipe's published numbers if a recipe exists for the
same model+hardware (per lifecycle map). Significant accuracy
regression vs the recipe is a sign the calibration data or technique
choice was off.

**Verification gate.** Checkpoint loads, sample inference works.

---

## Refusal Conditions

- The source model is gated and no HF token is provided.
- The chosen technique is not supported on the target hardware (e.g.
  NVFP4 on H100 — Blackwell-only).
- Calibration data is missing or empty when the technique requires it.
- A recipe already exists for the target model+hardware+precision and
  the user has not articulated why the recipe is insufficient.

---

## Cross-Skill Referencing

| Need | Sibling |
|---|---|
| Plan parallelism / SLA for the optimized model | [dynamo-plan](../dynamo-plan/SKILL.md) — re-plan because the optimization changes the shape |
| Deploy the optimized checkpoint | [dynamo-deploy](../dynamo-deploy/SKILL.md) — pass the `<output-dir>` as the model in the DGD or DGDR |
| Benchmark accuracy at scale | [dynamo-benchmark](../dynamo-benchmark/SKILL.md) — run an accuracy harness against the deployed quantized model |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- Source model accessible (HuggingFace ID with token if gated, or local checkpoint).
- `nvidia-modelopt` installed at the version aligned with the target Dynamo release (per).
- Calibration dataset on disk (for PTQ / AWQ techniques).
- Target hardware identified: Blackwell for NVFP4, Hopper+ for FP8.

---

## Limitations

What this skill does NOT cover:

- Does NOT deploy the optimized checkpoint. Hand off to `dynamo-deploy`.
- Does NOT measure accuracy at scale. Hand off to `dynamo-benchmark`.
- Does NOT cover post-training fine-tuning. ModelOpt is quantization-only.

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
| `scripts/validate-optimized.sh` | Post-quantization checkpoint validator | `-m <output-dir> [-b <backend>] [--load-test]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/validate-optimized.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/validate-optimized.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/technique-matrix.md](references/technique-matrix.md) —
  Per-technique decision table with hardware support and accuracy notes.
- [references/modelopt-cli.md](references/modelopt-cli.md) — Verbatim CLI
  commands per technique and the `quant_config.json` schema Dynamo
  backends consume.
- [references/known-issues.md](references/known-issues.md) — Optimization-
  specific stable issue patterns.
- [scripts/validate-optimized.sh](scripts/validate-optimized.sh) — Post-
  quantization validator using the `pass/fail/warn` helper.
