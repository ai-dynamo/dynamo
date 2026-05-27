# Qwen3.6-35B-A3B-FP8 — benchmark

aiperf-driven benchmark against an already-deployed config. Compares
the three deploy variants (`vllm-serve` / `dynamo-fd` / `dynamo-fd-ec`)
on a multimodal sliding-window workload.

**Prerequisite**: deploy a config first — see [`../deploy/README.md`](../deploy/README.md).
`benchmark.sh` will refuse to run if the target `Deployment` /
`DynamoGraphDeployment` isn't present in the namespace.

For shared pre-requisites (PVC, namespace, hostname fill-in), see the
[recipe root README](../README.md).

## aiperf

The bench Pod installs aiperf from PyPI:

```bash
pip install "aiperf>=0.8.0"
```

`>=0.8.0` is the floor because we need PR 824 (`session_id` for
single-turn causal ordering) + PR 903 (HF image-byte-dict fix). Both
shipped in upstream v0.8.0 — no fork or git pin required.

## Dataset

`data-gen-job.yaml` generates a sliding-window multimodal jsonl into
the shared PVC (`/perf-cache/datasets/30u_8t_5w_8000word_base64.jsonl`).
Parameters:

- 30 users × 8 turns = 240 requests
- window = 5 (each turn includes the last 5 images)
- 12 unique images per user (`window + turns − 1`), 2400×1080 base64-inlined
- 8000-token user-text per turn
- `session_id=user_<N>` per row — aiperf's `single_turn` mode honors
  that ordering, so the 8 turns of each user are sent in causal order
  and prefix-cache hits across turns actually land.

The Job is idempotent — `benchmark.sh dataset` skips it if a previous
run already Completed.

## Quick start

```bash
export NAMESPACE=<your-namespace>
export HW=gb200   # or h100; must match what deploy used

# `all` runs dataset → bench → retrieve.
./benchmark/benchmark.sh -n "$NAMESPACE" --hw "$HW" --config vllm-serve
./benchmark/benchmark.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd
./benchmark/benchmark.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd-ec
```

`benchmark.sh` accepts `--step {dataset|bench|retrieve|clean|all}`.
`dataset` is config-agnostic (any `--config` value works to run it once).

## Retrieve & interpret artifacts

`retrieve` (run by `all`) tars `/perf-cache/artifacts/` out of the bench
Pod to:

```
${BENCHMARK_RESULTS_DIR:-$HOME/workspace/dynamo-tmp/logs}/<MM-DD>/qwen36-fp8-<HW>/<CONFIG>/
```

The key file is `profile_export_aiperf.json` — throughput, TTFT,
inter-token latency (ITL), and per-request latency for the run live
there. `inputs.json` (the full prompt+image base64 record) is
excluded from the tar by default — it's ~4-12 GB per config and
trivially regenerable from the dataset jsonl. Set `KEEP_INPUTS_JSON=1`
on the bench Pod env if you need to keep it for individual-request
debugging.

## Cleanup

```bash
./benchmark/benchmark.sh -n "$NAMESPACE" --hw "$HW" --config <name> --step clean
```

Deletes the bench Pod only. The deployed config is left running — tear
it down via `../deploy/deploy.sh ... --step clean`.
