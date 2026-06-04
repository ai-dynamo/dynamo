# Qwen3.6-35B-A3B-FP8 — 3-way benchmark

Reproduces the `vllm-serve` vs `dynamo-fd` vs `dynamo-fd-ec` performance
comparison on a single GPU (H100 or GB200), measuring the throughput / TTFT
impact of Dynamo's frontend-decoding and embedding cache on a repeated-image
workload.

This harness **deploys the config it benchmarks** (so it's self-contained for
perf runs). If your goal is a production deployment rather than a benchmark,
use the standalone deploy path in [`../README.md`](../README.md) instead — it
applies the same `deploy/*.yaml` manifests without any benchmark tooling.

Prerequisites are the same as the top-level recipe: a `shared-model-cache`
PVC, a filled-in `../hw/<target>.env`, and `envsubst`. See
[`../README.md` → Pre-requisites](../README.md#pre-requisites).

## Quick start

```bash
export NAMESPACE=<your-namespace>
export HW=gb200   # or h100

# Run all three configs sequentially (prep + deploy + bench + retrieve + clean
# per config). Artifacts land under
# ~/workspace/dynamo-tmp/logs/<MM-DD>/qwen36-fp8-${HW}/{vllm-serve,dynamo-fd,dynamo-fd-ec}/.
./run-all-benchmarks.sh -n ${NAMESPACE} --hw ${HW}
```

Each config's `profile_export_aiperf.json` is retrieved into the matching
sub-directory; throughput / TTFT / ITL numbers can be read directly from
that file.

Or step-by-step for a single config:

```bash
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config vllm-serve
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config dynamo-fd
./run-benchmark.sh -n ${NAMESPACE} --hw ${HW} --config dynamo-fd-ec
```

`run-benchmark.sh` accepts `--step {pvc|download|dataset|deploy|bench|retrieve|clean}`
for granular control. `pvc`, `download`, and `dataset` are config-agnostic
(any `--config` works to run them once). The scripts read `deploy/`, `hw/`,
and `model-cache/` from the recipe root (`../`) and `perf.yaml` /
`data-gen-job.yaml` from this directory.

## How the pieces fit

| File | Role |
|------|------|
| `data-gen-job.yaml` | Generates the sliding-window jsonl dataset into `/perf-cache/datasets/`. |
| `perf.yaml`         | Shared aiperf bench Pod, templated across all three configs via `envsubst`. |
| `run-benchmark.sh`  | Single-config driver (prep → deploy → bench → retrieve → clean). |
| `run-all-benchmarks.sh` | Runs the three configs back-to-back on the same node. |

`perf.yaml` is shared because the only deltas across the configs (pod name,
frontend service, run-label) are exported as `${BENCH_POD}` /
`${BENCH_FRONTEND}` / `${BENCH_RUN_LABEL}` by `run-benchmark.sh` and resolved
via `envsubst` at apply time. The bench Pod runs the same
`vllm-runtime:1.2.0` image as the deployment, which bakes `aiperf` — no
install step needed.

## Dataset

`data-gen-job.yaml` generates `30u_8t_5w_8000word_base64.jsonl`:

- Sliding-window with `window=5`, `turns=8`, `users=30`,
  `image_size=2400x1080`, `user_text_tokens=8000`. Yields 240 requests
  (`users × turns`) over 12 unique images per user
  (`window + turns - 1`). Base64-inlined.
- Each jsonl row carries `session_id=user_<N>`. aiperf's `single_turn`
  dataset type honors session ordering (via
  [PR 824](https://github.com/ai-dynamo/aiperf/pull/824), now baked into
  the `1.2.0` image), so the 8 turns of any one user are sent in causal
  order — letting prefix-cache hits land.
- From turn 2 onwards each turn reuses 4-of-5 images from the previous turn's
  window, so repeated images dominate — the shape the embedding cache is
  designed for.

The `1.2.0` image bakes the sliding-window generator (dynamo PR 8201), so the
Job runs it straight from the baked tree — no git clone.

## Results

Per-config `profile_export_aiperf.json` holds the headline metrics (RPS, TTFT
percentiles, ITL). Published H100-vs-GB200 numbers and takeaways live in
[`docs/benchmarks/embedding_cache.md`](../../../docs/benchmarks/embedding_cache.md).

## Cleanup

The drivers clean each config (bench pod + deployment) between runs so they
share one node. PVCs are intentionally **not** deleted — that would force a
model re-download. To wipe everything:
`kubectl -n $NAMESPACE delete pvc shared-model-cache`.
