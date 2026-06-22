<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode Migration Test Recipe

This recipe brings up Dynamo's normal HTTP frontend with two migration-enabled
SGLang decode workers, then exercises active request migration through NIXL. It
covers a two-GPU correctness smoke test and the DeepSeek-V2-Lite DEP8 to DEP2
performance experiment.

## Validation Status

The TP1 to TP1 path is the correctness gate and has local coverage for exact
token parity, stream intervals, cancellation, finish races, cleanup, and
concurrent migrations.

The DEP8 to DEP2 command below is the target experiment, not yet a release
claim. It needs ten non-collocated GPUs and exercises multi-DP-rank migration.
The current host has eight GPUs, so run this part on a B200 or newer system with
at least ten free GPUs. Do not publish its Pareto point until all success gates
at the end of this recipe pass.

DeepSeek-V2-Lite is an MLA model, but it is not a thinking model. The static
experiment therefore treats the first 60% of each generated sequence as
reasoning and migrates at a sequence-length trigger. Use a token-ID trigger on
a model with an explicit end-of-thinking token.

## Branches

Use these branches together:

| Repository | Remote | Branch |
|---|---|---|
| SGLang | `git@github.com:aphoh/sglang.git` | `req-migration-2.0` |
| Dynamo | `git@github.com:ai-dynamo/dynamo.git` | `warnold/sglang-dd-2.0` |

For fresh checkouts:

```bash
mkdir -p /root/proj/decode-migration
cd /root/proj/decode-migration

git clone git@github.com:aphoh/sglang.git sglang
git -C sglang fetch origin req-migration-2.0
git -C sglang switch --track origin/req-migration-2.0

git clone git@github.com:ai-dynamo/dynamo.git dynamo
git -C dynamo fetch origin warnold/sglang-dd-2.0
git -C dynamo switch --track origin/warnold/sglang-dd-2.0
```

For existing checkouts, fetch and fast-forward instead of resetting local work:

```bash
git -C /root/proj/decode-migration/sglang fetch aphoh req-migration-2.0
git -C /root/proj/decode-migration/sglang switch req-migration-2.0
git -C /root/proj/decode-migration/sglang pull --ff-only aphoh req-migration-2.0

git -C /root/proj/decode-migration/dynamo fetch origin warnold/sglang-dd-2.0
git -C /root/proj/decode-migration/dynamo switch warnold/sglang-dd-2.0
git -C /root/proj/decode-migration/dynamo pull --ff-only
```

Record the tested revisions:

```bash
git -C /root/proj/decode-migration/sglang rev-parse HEAD
git -C /root/proj/decode-migration/dynamo rev-parse HEAD
```

## Prerequisites

The launch scripts expect:

- Linux with Docker, NVIDIA Container Toolkit, and `--gpus all` support;
- CUDA-capable GPUs with peer/NIXL connectivity;
- `maturin` and `protoc` on the host for the Dynamo runtime wheel;
- a local Hugging Face-format model checkpoint; and
- free ports `18000`, `18081`, `18082`, `18101`, `18102`, `18201`,
  and `18202`.

Install the build tools if needed:

```bash
uv tool install maturin
sudo apt-get install -y protobuf-compiler
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Set the two repository roots once per shell:

```bash
export DYNAMO_ROOT=/root/proj/decode-migration/dynamo
export SGLANG_ROOT=/root/proj/decode-migration/sglang
export RECIPE_DIR=$DYNAMO_ROOT/examples/backends/sglang/decode_migration
```

## Build the Runtime Image

The image contains the SGLang runtime dependencies and a Dynamo Python wheel
built from the selected Dynamo branch. The live source trees are mounted into
the container, so rebuild after Dynamo Rust or binding changes; Python-only
edits are visible immediately.

```bash
IMAGE=dynamo-sglang-decode-migration:dev \
DYNAMO_ROOT=$DYNAMO_ROOT \
  $RECIPE_DIR/build_image.sh
```

## Two-GPU Correctness Gate

Use the small Qwen checkpoint first. This runs the full scenario suite and
stops the deployment afterward:

```bash
IMAGE=dynamo-sglang-decode-migration:dev \
DYNAMO_ROOT=$DYNAMO_ROOT \
SGLANG_ROOT=$SGLANG_ROOT \
MODEL_ROOT=/root/models/qwen3-0.6b \
MODEL_PATH_IN_CONTAINER=/models/qwen3-0.6b \
SERVED_MODEL_NAME=Qwen/Qwen3-0.6B \
SOURCE_GPUS=0 \
DESTINATION_GPUS=1 \
SOURCE_TP=1 \
DESTINATION_TP=1 \
RESULT_DIR=/tmp/qwen-migration-smoke \
  $RECIPE_DIR/run_container.sh
```

Repeat with coalesced streaming. This specifically checks that a migration
trigger hidden inside a multi-token stream chunk is still observed:

```bash
STREAM_INTERVAL=4 \
RESULT_DIR=/tmp/qwen-migration-stream-4 \
  $RECIPE_DIR/run_container.sh
```

The second command inherits the repository, image, model, and topology
variables only if they were exported. Otherwise repeat those assignments.

## DEP8 to DEP2 Deployment

The local checkpoint used for the measurements is
`/root/models/deepseek-v2-lite`. It is BF16 DeepSeek-V2-Lite with MLA. MLA
does not use the GQA/MHA staging buffer, so leave
`SGLANG_DISAGG_STAGING_BUFFER=0`.

The source uses eight data/expert-parallel ranks, one per GPU. Its selected knee
is 16 concurrent requests per GPU. The destination uses two ranks at 128
concurrent requests per GPU. Start both once and sweep every offered load
against the same long-lived processes:

```bash
IMAGE=dynamo-sglang-decode-migration:dev \
DYNAMO_ROOT=$DYNAMO_ROOT \
SGLANG_ROOT=$SGLANG_ROOT \
MODEL_ROOT=/root/models/deepseek-v2-lite \
MODEL_PATH_IN_CONTAINER=/models/deepseek-v2-lite \
SERVED_MODEL_NAME=deepseek-ai/DeepSeek-V2-Lite \
SOURCE_GPUS=0,1,2,3,4,5,6,7 \
DESTINATION_GPUS=8,9 \
SOURCE_TP=8 SOURCE_DP=8 SOURCE_EP=8 \
DESTINATION_TP=2 DESTINATION_DP=2 DESTINATION_EP=2 \
SOURCE_ENABLE_DP_ATTENTION=1 \
DESTINATION_ENABLE_DP_ATTENTION=1 \
SOURCE_MOE_A2A_BACKEND=deepep \
DESTINATION_MOE_A2A_BACKEND=deepep \
SOURCE_MOE_RUNNER_BACKEND=deep_gemm \
DESTINATION_MOE_RUNNER_BACKEND=deep_gemm \
SOURCE_DEEPEP_MODE=low_latency \
DESTINATION_DEEPEP_MODE=low_latency \
SOURCE_ENABLE_JIT_DEEPGEMM=1 \
DESTINATION_ENABLE_JIT_DEEPGEMM=1 \
DEEPEP_MAX_DISPATCH_TOKENS=1024 \
SOURCE_MAX_RUNNING_REQUESTS=16 \
DESTINATION_MAX_RUNNING_REQUESTS=128 \
SOURCE_CUDA_GRAPH_BS="1 2 4 8 16" \
DESTINATION_CUDA_GRAPH_BS="1 2 4 8 16 32 64 128" \
SOURCE_MEM_FRACTION_STATIC=0.60 \
DESTINATION_MEM_FRACTION_STATIC=0.60 \
SGLANG_DISAGG_STAGING_BUFFER=0 \
ENABLE_DETERMINISTIC_INFERENCE=0 \
DISABLE_CUDA_GRAPH=0 \
STREAM_INTERVAL=1 \
TEST_MODE=serve \
RESULT_DIR=/tmp/deepseek-dep-migration \
  $RECIPE_DIR/run_container.sh
```

The frontend is the standard `python3 -m dynamo.frontend` path on port
`18000`. The source and destination retain their public model cards and
`generate` endpoints; migration is opt-in through `nvext.decode_migration`.
Overlap scheduling remains enabled.

Wait for:

```text
Dynamo frontend and migration workers are ready on port 18000
```

If CUDA graph capture at batch 128 exceeds available destination memory, reduce
`DESTINATION_CUDA_GRAPH_BS` and `DESTINATION_MAX_RUNNING_REQUESTS` together,
then remeasure the standalone DEP2 point. Do not compare results from different
graph or concurrency settings as if they were the same configuration.

## Run the Static Pareto Sweep

In a second terminal, run the included open-loop client. Its defaults are ISL
1, fixed OSL 512, a 60/40 source/destination split, 256 measured requests,
five seconds of warmup and cooldown traffic, and a 20 token/s visible-stage
gate. The migration trigger is sequence length 308 including the one-token
prompt.

```bash
export DYNAMO_ROOT=/root/proj/decode-migration/dynamo
export SGLANG_ROOT=/root/proj/decode-migration/sglang
export RECIPE_DIR=$DYNAMO_ROOT/examples/backends/sglang/decode_migration

MODEL=deepseek-ai/DeepSeek-V2-Lite \
MODES=migration \
RATES="40 50 60 70 78 82 86" \
REQUESTS=256 \
RESULT_DIR=/tmp/deepseek-dep-pareto \
  $RECIPE_DIR/run_static_pareto.sh
```

Run the source-only logical baseline on the same deployment with the
destination idle and disjoint:

```bash
MODES=baseline \
RATES="30 40 46 50 54" \
REQUESTS=256 \
RESULT_DIR=/tmp/deepseek-dep-baseline \
  $RECIPE_DIR/run_static_pareto.sh
```

The baseline reports throughput per eight active source GPUs. Migration reports
throughput per all ten active GPUs. For a truncated-normal OSL distribution
centered at 512, add:

```bash
OSL_STDDEV=64 OSL_MIN=256 OSL_MAX=768
```

Keep the 60/40 boundary request-relative; the client computes it independently
for every sampled OSL.

## Inspect Results

Summarize the sweep:

```bash
jq -r '
  [.summary.mode,
   .summary.arrival_rate_rps,
   .summary.completed,
   .summary.slo_compliant,
   .summary.p95_ttfnt_s,
   .summary.offered_goodput_per_gpu,
   .summary.p50_ttfnt_drift_s,
   .summary.min_visible_rate_tps] | @tsv
' /tmp/deepseek-dep-pareto/*.json
```

Confirm that migrations actually committed:

```bash
rg -c "decode migration committed" \
  /tmp/deepseek-dep-migration/stream-1/frontend.log
rg -n "ERROR|Traceback|timed out|declined|failed" \
  /tmp/deepseek-dep-migration/stream-1
```

Inspect `fast.log` and `slow.log` through the entire measured interval.
Running and queued request counts should be stationary rather than growing
between the early and late halves of the run.

## Success Gates

Accept a point only when all of these hold:

1. All 256 measured requests complete with exactly 512 output tokens.
2. All measured requests meet the 20 token/s visible-stage gate.
3. The frontend records one committed migration per migrated request.
4. There are no source-quiesce, prepare, sync, finalize, cancellation, or KV
   reservation leaks.
5. `p50_ttfnt_drift_s` is small enough to show a stationary queue.
6. The TP1 deterministic smoke has exact baseline/migration token fingerprints.
   For heterogeneous DEP8-to-DEP2, record fingerprints to diagnose loss or
   duplication and use paired task accuracy to bound expected numeric drift.
7. P95 TTFNT and throughput/GPU are reported from the same request set and GPU
   accounting described above.
8. No process restarts, OOMs, or scheduler invariant failures occur.

The measured standalone capacities that motivated this pairing were
approximately 222.8 token/s/user and 3201.1 output token/s/GPU for DEP8 at
16 concurrent requests/GPU, and 143.4 token/s/user and 15441.5 output
token/s/GPU for DEP2 at 128 concurrent requests/GPU. Treat these as a sanity
range, not an acceptance threshold for a different machine or build.

## Stop and Clean Up

Press Ctrl-C in the deployment terminal. The launcher terminates the frontend,
both workers, and its discovery keepalive process. Confirm no process remains:

```bash
docker ps --filter ancestor=dynamo-sglang-decode-migration:dev
nvidia-smi
```

Keep the JSON results, all three logs, both Git SHAs, image ID, GPU type, driver,
and CUDA versions with every reported Pareto plot.
