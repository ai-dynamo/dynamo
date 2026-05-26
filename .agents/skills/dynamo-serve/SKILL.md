---
name: dynamo-serve
description: >-
  Run a single-node NVIDIA Dynamo inference worker locally on a
  workstation via the Python module form (`python3 -m dynamo.vllm` (or .trtllm / .sglang)).
  Iterate on model, backend, and engine flags without Kubernetes. Use
  when running Dynamo on a laptop or developer box, prototyping a model
  with vLLM / TensorRT-LLM / SGLang, debugging a worker before
  deployment, or comparing backends interactively.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - local
  - workstation
  - vllm
  - sglang
  - trtllm
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "run Dynamo locally", "python -m dynamo", "workstation iteration", "vllm local"
- Level 2 (this file): 4-phase local-dev workflow
- Level 3: references/launch-flags.md for per-backend flag matrix
           references/known-issues.md for local-dev stable patterns
           scripts/precheck-local.sh for environment readiness

Scope: workstation iteration with python3 -m dynamo.<backend>
Out of scope: anything requiring Kubernetes (operator, CRDs, multi-node, autoscaling) — see dynamo-deploy.
              Frontend service config, multi-model routing, gateway integration — see dynamo-frontend.
              Quantization — see dynamo-optimize.
              Benchmarking — see dynamo-benchmark.
              Day-2 troubleshooting — see dynamo-troubleshoot.

Per SKILL_AUTHORING.md §4, there is no `dynamo-run` CLI binary. The
invocation is `python3 -m dynamo.vllm` (or .trtllm / .sglang). This skill owns that
workflow end-to-end.
-->

# Dynamo Serve (Local Workstation Run)

Run one Dynamo worker against one model on a workstation, no Kubernetes
required. Use for fast iteration: try a model, try a backend, try a
flag, see the result, adjust.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Pre-Check  → Python, ai-dynamo[<backend>] extras, GPU visible, model accessible
Phase 2: Configure  → pick backend; pick model; pick engine flags
Phase 3: Run        → python3 -m dynamo.<backend> --model ... [flags]
Phase 4: Verify     → curl /v1/models, sample inference, check worker logs
```

---

## Command Safety

Local-dev is mostly client commands. One MUTATING case: the worker
itself binds a port and consumes GPU. DESTRUCTIVE cases are limited to
deleting cached weights or HF download directories.

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `rm -rf ~/.cache/huggingface` | Removes ALL HuggingFace weight caches across all models. Future model loads will re-download (can be tens of GB per model). |
| `rm -rf <model-dir>` | Removes a local model checkpoint. If quantization was applied, the optimized weights are gone too — see [dynamo-optimize](../dynamo-optimize/SKILL.md). |
| `kill -9 <worker-pid>` | Force-kills the worker. May leave GPU memory allocated until the driver reclaims it (~30s). |

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `python3 -m dynamo.<backend> --model ... --port <p>` | Starts a worker that consumes one or more GPUs and binds port `<p>`. Other GPU work on the box experiences contention. |
| `pip install 'ai-dynamo[<backend>]==<version>'` | Pulls the backend wheel + transitive deps (often 5-10 GB for vLLM with FlashInfer). |
| `huggingface-cli download <model>` | Downloads weights to `~/.cache/huggingface/` (can be tens of GB). |

### SAFE — no confirmation needed

```
python3 -m dynamo.<backend> --help
nvidia-smi
nvidia-smi -q -d MEMORY
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -d '...'
pip show ai-dynamo
pip show 'nixl[cu12]'
ls ~/.cache/huggingface/hub/
```

---

## Phase 1: Pre-Check

**Goal.** Confirm the workstation can run a Dynamo worker.

**Inputs**:

| Input | How to get it |
|---|---|
| Python 3.10+ | `python3 --version` |
| `ai-dynamo[<backend>]` wheel | `pip install 'ai-dynamo[<backend>]==<release>'` |
| GPU + driver | `nvidia-smi` reports at least one GPU; CUDA driver matches the backend's container CUDA pin (per) |
| HF token (gated models only) | `export HF_TOKEN=<token>` or `huggingface-cli login` |
| Model | HF ID or local path |

**Commands** (SAFE):

```bash
# Python.
python3 --version

# ai-dynamo wheel and backend extras.
pip show ai-dynamo
pip show ai-dynamo-runtime

# GPU.
nvidia-smi --query-gpu=name,memory.free --format=csv

# Confirm model loads (config-only, no weights yet).
python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('<model-id>'))"
```

Run the bundled pre-check:

```bash
bash scripts/precheck-local.sh -b <vllm|trtllm|sglang> -m <model-id-or-path>
```

The script applies the `pass/fail/warn` pattern from. See
[scripts/precheck-local.sh](scripts/precheck-local.sh).

**Decision point** — if the GPU driver or backend wheel is at a version
the recipe / target release does not pin to (per
`container/context.yaml`), point out the drift before continuing.
Workstation drift is the most common reason a locally-working
configuration fails to translate to a recipe-based deploy.

**Verification gate.** Python OK, ai-dynamo installed, GPU visible,
model config loadable.

---

## Phase 2: Configure

**Goal.** Pick the backend, model, and the launch flags that match the
intended iteration.

### 2.1 Backend Choice

| Module | When to use | Pin source |
|---|---|---|
| `dynamo.vllm` | Most models; the default in many recipes | `pyproject.toml` `[vllm]` extra (per) |
| `dynamo.trtllm` | TensorRT-LLM users; FP8/FP4 on Hopper/Blackwell | `pyproject.toml` `[trtllm]` extra |
| `dynamo.sglang` | SGLang users; multimodal Wan2.2 video via `dynamo.vllm.omni` | `pyproject.toml` `[sglang]` extra |
| `dynamo.mocker` | Testing without a real model; profiler-style workloads | `pyproject.toml` `[mocker]` extra |

There is no `dynamo-run` CLI (per). The `dynamo.<backend>` module
form is the only local-run path.

### 2.2 Flags

See [references/launch-flags.md](references/launch-flags.md) for the
per-backend flag matrix. Common patterns:

| Pattern | Flags |
|---|---|
| Single worker, single GPU, aggregated | `--model <id>` |
| Single worker, multi-GPU TP=N | `--model <id> --tensor-parallel-size N` |
| Disagg (prefill + decode) — requires two terminals | `--disaggregation-mode prefill --kv-transfer-config '...'` and `--disaggregation-mode decode` (per) |
| KV-aware routing — also requires a Router instance | not first-class in local-dev; use the deploy skill |
| Quantized checkpoint | `--model <path-to-optimized-dir> --load-format auto` |

**Decision:** The proposed configuration is `python3 -m dynamo.<backend>
--model <id> <flags...>`. This will consume `<n>` GPU(s) on this box.
Proceed?

Wait for explicit confirmation if the box is shared with other GPU work.

**Verification gate.** Backend, model, and flags fixed in one shell
command ready to run.

---

## Phase 3: Run

**Goal.** Start the worker and wait for it to bind the OpenAI port.

```bash
python3 -m dynamo.<backend> \
  --model <hf-id-or-path> \
  [flags...]
```

Default behaviour: binds `localhost:8000`, exposes
`/v1/chat/completions`, `/v1/completions`, `/v1/models`. Override the
port with `--port <N>` if 8000 is occupied.

For disagg mode (two terminals):

```bash
# Terminal 1: prefill (per).
python3 -m dynamo.vllm \
  --model <hf-id-or-path> \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Terminal 2: decode.
python3 -m dynamo.vllm \
  --model <hf-id-or-path> \
  --disaggregation-mode decode
```

The worker logs to stdout. First-token latency on a cold start is
dominated by weight load (seconds to minutes depending on size).

**Decision (MUTATING):** Starting the worker. It will consume `<n>`
GPU(s) until you Ctrl+C the process or the terminal closes. Proceed?

Wait for explicit yes.

**Verification gate.** Worker logs show "uvicorn running on
0.0.0.0:<port>" (or equivalent for the backend), and `nvidia-smi`
shows GPU memory consumed by the python process.

---

## Phase 4: Verify

**Goal.** Confirm the worker is serving and the model is registered.

```bash
# Model registration.
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Sample inference (chat).
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id-from-/v1/models>",
    "messages": [{"role": "user", "content": "Say hello in three words."}],
    "max_tokens": 16
  }' | python3 -m json.tool

# Sample inference (completions, for base models without chat template, per).
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model-id>",
    "prompt": "Once upon a time",
    "max_tokens": 16
  }' | python3 -m json.tool
```

If `/v1/models` returns empty for >2 minutes after the worker starts,
that matches the pattern (registration window). For local-dev
the typical cause is the weight download still in flight — check the
worker stdout for "downloading" lines.

**Verification gate.** A `/v1/chat/completions` or `/v1/completions`
request returns a non-empty response.

---

## Refusal Conditions

- The GPU does not match the backend's hardware requirements (e.g.
  NVFP4-quantized model on an H100).
- The model is gated on HuggingFace and no token is configured.
- The disagg configuration is missing the explicit `--kv-transfer-config`
  on prefill (per) — refuse and point at the fix.
- The user requests `dynamo-run --model ...` (legacy CLI that does not
  exist per) — refuse and propose the `python3 -m` form.

---

## Cross-Skill Referencing

| Need | Sibling |
|---|---|
| Quantize the model first | [dynamo-optimize](../dynamo-optimize/SKILL.md) |
| Move to K8s deployment | [dynamo-deploy](../dynamo-deploy/SKILL.md) — the configuration explored here informs the DGD spec |
| Multi-model serving or gateway integration | [dynamo-frontend](../dynamo-frontend/SKILL.md) — the Frontend service is K8s-only; local-dev is single-model only |
| Benchmark the local worker | [dynamo-benchmark](../dynamo-benchmark/SKILL.md) — AIPerf against `http://localhost:8000/...` |
| Debug a worker that won't start | [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md) Phase 2 — many of the day-2 signatures apply to local-dev too |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- Python 3.10+ installed.
- `ai-dynamo` wheel with the relevant backend extra (per `pyproject.toml` extras; see).
- A GPU visible to the process (skip for the mocker backend).
- HF token in the environment for gated models (per's local-dev variant).

---

## Limitations

What this skill does NOT cover:

- Single-node only. For multi-node, use `dynamo-deploy` on Kubernetes.
- No Frontend service config or multi-model routing. For those, use `dynamo-deploy` and `dynamo-frontend`.
- Does NOT use the `dynamo-platform` operator. Pods, CRDs, autoscaling are not in scope.
- The local-dev NIXL pin (PyPI `nixl[cu12]`) drifts from the container's NIXL build (per); disagg behavior may diverge from a K8s deploy.

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
| `scripts/precheck-local.sh` | Workstation readiness: Python, ai-dynamo, GPU, model, HF token | `-b <backend> -m <model> [--gated]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/precheck-local.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/precheck-local.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/launch-flags.md](references/launch-flags.md) — Per-
  backend flag matrix (vLLM, TRT-LLM, SGLang, Omni, mocker) with worked
  examples.
- [references/known-issues.md](references/known-issues.md) — Local-dev
  stable patterns: HF auth, port conflicts, GPU memory exhaustion,
  weights cache hygiene, disagg config drift.
- [scripts/precheck-local.sh](scripts/precheck-local.sh) — Workstation
  readiness check; PASS/FAIL/WARN per.
