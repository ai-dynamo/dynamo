<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kubernetes Recipe Workflow

Reference content for `dynamo-optimize/SKILL.md`. Substrate is Anish's
`dynamo-recipe-runner` from PR #9782, edited with citation hooks (every
load-bearing fact carries a Section-G citation key from
`dynamo-skills/docs/citations.md`).

## Recipe Tree Shape

Canonical layout at commit `6cf162e9a0`:

```
recipes/
  <model>/
    README.md
    model-cache/
      model-download.yaml
    <framework>/                 # vllm | sglang | trtllm | tokenspeed
      <mode>/
        deploy.yaml              # the DGD (nvidia.com/v1alpha1)
        perf.yaml                # AIPerf load Pod
        README.md                # per-mode (sometimes)
```

**Counts at this skill's `version`:** 14 model directories, 46 recipe
leaf directories, 4 frameworks (G1, G2).

### Exceptions to the three-level shape

Two model directories do NOT follow `recipes/<model>/<framework>/<mode>/`:

- **`recipes/deepseek-v4/{deepseek-v4-flash,deepseek-v4-pro}/<framework>/<mode>/`** —
  extra sub-model qualifier level. Treat
  `deepseek-v4-flash` and `deepseek-v4-pro` as separate models, not as
  `deepseek-v4/...` paths.

- **`recipes/qwen3.6-35b/`** — uses
  `deploy/{config}.yaml` + `hw/{target}.env` + a shared `perf.yaml`.
  Standard recipe-runner discovery will not find this. If the user asks
  for a `qwen3.6-35b` recipe, route them to the README in that directory;
  the standard patch matrix in `dynamo-optimize/SKILL.md` does not
  directly apply.

Other multi-level qualifier paths follow the standard shape but add a
fourth or fifth segment (e.g. `recipes/qwen3-235b-a22b-fp8/trtllm/agg/blackwell/`,
`recipes/llama-3-70b/vllm/disagg-single-node/gaie/`). The first three
segments still uniquely identify the recipe; the qualifier specializes
the hardware target.

## Required Inputs (Phase 1 reference)

Collect or infer before any patching:

- recipe target: model, framework, mode, GPU SKU + count.
- Kubernetes context and namespace.
- Hugging Face Secret name (default `hf-token-secret`).
- Storage class for the model-cache and perf-cache PVCs.
- Runtime image tag if the recipe pins a placeholder.
- Whether to run commands or only produce exact commands (offline mode).

If a required value is missing and cannot be inferred from the selected
recipe, ask for only that value — do not assume.

## Preflight Command Sequence

Run read-only checks first (SAFE tier):

```bash
git status --short
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py list --format table
kubectl config current-context
kubectl get storageclass
kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count
kubectl get namespace "${NAMESPACE}"
kubectl get secret hf-token-secret -n "${NAMESPACE}"
```

If `kubectl` is unavailable or the cluster is unreachable, continue with
selection and validation, then return exact commands instead of pretending
the deployment ran.

## Selection Command Sequence

Use the recipe matrix from `recipes/README.md` and the scanner:

```bash
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py list \
  --query qwen --framework vllm --mode disagg --format table
```

Prefer an exact existing recipe. Do not invent new manifests unless the
user explicitly asks to author a new recipe (in which case hand off to
`dynamo-deploy/SKILL.md` Phase 2.3 manual DGD authoring).

## Validation Command Sequence

Read the selected recipe README, model-cache manifests, `deploy.yaml`,
and `perf.yaml`. Then:

```bash
python3 .agents/skills/dynamo-optimize/scripts/recipe_tool.py validate \
  recipes/<model>/<framework>/<mode>
```

Resolve reported blockers before applying manifests:

- storage class
- model-cache PVC
- image tag
- HF token Secret
- GPU count
- frontend service name
- router mode

## Patch Matrix (Phase 3 reference)

Patch only these fields. Do not reformat whole YAML files.

| Field | Source | Recipe-Envelope Impact |
|---|---|---|
| `storageClassName` | `kubectl get storageclass` | None (purely cluster-local). |
| `image` repository/tag | Per-release Dynamo image | Voids the envelope; recipe perf was measured at the pinned tag. |
| Model path or mount path | User | None (refers to local PVC layout). |
| GPU resource requests/limits | `kubectl get nodes -L nvidia.com/gpu.product` | Voids the envelope if shrunk; safe to expand. |
| Frontend `DYN_ROUTER_MODE` env | Reference G4 — `round-robin` / `random` / `power-of-two` / `kv` / `direct` / `least-loaded` / `device-aware-weighted`; default `round-robin`. | Changes the routing variant; this IS the optimization knob in many cases. |
| Namespace when a manifest hardcodes it | User | None. |

**Never** write Hugging Face tokens into files or logs. Use Kubernetes
Secrets. The recipe already uses the `envFromSecret` pattern.

## Deploy Command Sequence

Follow the selected recipe's README when it differs from the default
sequence. The default (works for the standard three-level shape):

```bash
kubectl apply -f recipes/<model>/model-cache/ -n "${NAMESPACE}"

# G25: canonical model-download timeout is 600s.
kubectl wait --for=condition=Complete job/model-download \
  -n "${NAMESPACE}" --timeout=600s

kubectl apply -f recipes/<model>/<framework>/<mode>/deploy.yaml -n "${NAMESPACE}"

# G26: canonical pod-readiness selector.
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD_NAME}" \
  -n "${NAMESPACE}" --timeout=1200s

kubectl get dynamographdeployment -n "${NAMESPACE}"
kubectl get pods -n "${NAMESPACE}" -o wide
```

Wait for the frontend and workers to be ready before testing.

## Smoke-Test Command Sequence

Port-forward the frontend service, then verify `/v1/models` and one chat
completion:

```bash
kubectl port-forward svc/"${DGD_NAME}"-frontend 8000:8000 -n "${NAMESPACE}" &

# Worker registration via NATS takes 30-120s after pods report Ready.
until curl -s http://localhost:8000/v1/models \
  | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")'; do
  echo "waiting for model registration..."
  sleep 10
done

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}]}" \
  | python3 -m json.tool
```

If this fails, switch to `dynamo-troubleshoot/SKILL.md`.

## Mode Taxonomy Detail

The 18 unique modes observed across the tree (G3). Every motivation
quote below is verbatim from the cited README; see
`dynamo-skills/corpus/recipes/taxonomy.yaml` for the full extract.

### Aggregated modes

| Mode | Used by | Motivation source |
|---|---|---|
| `agg` | deepseek-v4-flash, deepseek-v4-pro, gpt-oss-120b, kimi-k2.5, llama-3-70b, nemotron-3-nano-omni, nemotron-3-super-fp8, qwen3-235b-a22b-fp8, qwen3-32b-fp8 | Plain aggregated serving — one process holds both prefill and decode. Baseline / lowest operational complexity. |
| `agg-round-robin` | deepseek-v32-fp4, kimi-k2.5, qwen3-32b | qwen3-32b README "Aggregated \| Round-robin \| 8x TP2 workers" — baseline against which KV-aware routing or disagg is measured. |
| `agg-kvbm` | qwen3-32b | qwen3-32b README: "Single-GPU aggregated deployment of `Qwen/Qwen3-32B` with the KV Block Manager (KVBM) enabled. KVBM offloads cold KV cache blocks to host memory so the effective cache footprint extends beyond GPU HBM, which improves prefix-reuse hit rate on long or repeated prompts without adding GPUs." |
| `agg-embedding-cache` | qwen3-vl-30b | qwen3-vl-30b README: enabling embedding cache on `Qwen3-VL-30B-A3B-Instruct-FP8` shows an average improvement of +16% throughput, -28% TTFT, -13% request latency on a single aggregated replica of GB200 using vLLM. |
| `agg-eagle-kv-router` | kimi-k2.5 | kimi-k2.5 README: agg + EAGLE speculative decoding + KV-aware routing. |
| `agg-eagle-round-robin` | kimi-k2.5 | kimi-k2.5 README: agg + EAGLE + round-robin (ablation point that drops KV-aware routing). |
| `agg-gb200`, `agg_b200`, `agg_gb200` | deepseek-v4 sub-models | No README motivation — hardware-qualifier variants of `agg`. Treat as platform-specialized `agg`. |

### Disaggregated modes

| Mode | Used by | Motivation source |
|---|---|---|
| `disagg` | deepseek-r1, deepseek-v4-pro, glm-5-nvfp4, gpt-oss-120b, nemotron-3-super-fp8, qwen3-235b-a22b-fp8, qwen3-32b-fp8 | recipes/README.md + llama-3-70b README: "Disagg (Single-Node) \| 8x H100/H200 \| ... Prefill + Decode separation". Generic disagg — prefill/decode in separate pods. |
| `disagg-8gpu`, `disagg-16gpu` | deepseek-r1 | deepseek-r1/sglang README: "The two deployment recipes are for 16x H200 (disagg-8gpu) and 32x H200 (disagg-16gpu). The folder names refer to GPUs per worker type (8 or 16), with separate prefill and decode workers each using that many GPUs." |
| `disagg-single-node` | llama-3-70b | recipes/README.md: "Disagg (Single-Node) \| 8x H100/H200 ... Prefill + Decode separation". Single-node disagg; no inter-node KV transfer. |
| `disagg-multi-node` | llama-3-70b | recipes/README.md: "Disagg (Multi-Node) \| 16x H100/H200 ... 2 nodes, 8 GPUs each". Requires working KV transport (NIXL/UCX) between nodes. |
| `disagg-kv-router` | deepseek-v32-fp4, qwen3-32b | qwen3-32b README: "Disaggregated \| KV-aware \| 6x prefill + 2x decode (TP2) ... KV-aware routing leverages the 36% cache efficiency to route requests to workers that already have relevant KV cache blocks, reducing redundant prefill computation and lowering TTFT. Disaggregated serving separates prefill and decode workers. With long input sequences (avg 12K tokens) and short outputs (avg 343 tokens), dedicated decode workers avoid 'prefill injection' — where a new long-context request interrupts ongoing decode operations, causing ITL spikes." |
| `disagg-eagle-kv-router` | kimi-k2.5 | kimi-k2.5 README: "The disaggregated configuration with KV-aware routing, Eagle decoding, and KV offloading achieves the best system throughput and interactivity." |
| `disagg-b200`, `disagg-gb200` | deepseek-v4-pro | No README motivation — hardware-qualifier variants of `disagg`. |

## When NOT to Use a Recipe

If the user's situation matches any of these, hand off rather than
patching a recipe:

- **No recipe matches the model.** Hand off to `dynamo-deploy` Phase 2.3
  for manual DGD authoring (or to `dynamo-plan` for AIConfigurator-led
  configuration).
- **Recipe's tested GPU SKU/count is unavailable.** Either pick a
  different recipe or hand off to `dynamo-deploy`. Do not patch GPU count
  down — that voids the envelope.
- **User wants AIConfigurator to search the config space.** That is
  `dynamo-plan`'s job; this skill applies a tested recipe.
- **User wants the full `perf.yaml` benchmark, not the `--goodput`
  shakedown.** That is `dynamo-benchmark`'s job; this skill validates
  against a declared SLO, not a full benchmark sweep.

## See Also

- [slo-shape.md](slo-shape.md) — AIPerf 0.8.0 SLO grammar + output schema.
- [inference-literature.md](inference-literature.md) — Regression conditions from the papers.
- [known-issues.md](known-issues.md) — Stable failure patterns.
