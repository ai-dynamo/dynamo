<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode Migration Test Recipe

This recipe defines the process arguments and validation workload for a
DeepSeek-V2-Lite DEP8-to-DEP2 decode-migration deployment. It deliberately
does not prescribe Kubernetes manifests, Services, GPU resource classes, or
pod placement. The deployment layer should translate the three process specs
below into the local Dynamo Kubernetes conventions.

## Revisions

Use these branches together:

| Repository | Remote | Branch | Minimum revision |
|---|---|---|---|
| SGLang | `git@github.com:Aphoh/sglang.git` | `req-migration-2.0` | `df1f4d7632` |
| Dynamo | `git@github.com:ai-dynamo/dynamo.git` | `warnold/sglang-dd-2.0` | `2914c20e23` |

The runtime image must contain the Dynamo revision and import the SGLang
revision above. Record both full SHAs and the image digest with every result.

## Container Image

A self-contained test image is published at:

```text
tag:    aphoh/not-sglang-dynamo:2914c20e23d4-df1f4d763224
digest: aphoh/not-sglang-dynamo@sha256:4ee5d05cf36b15545e63f87799a3b928087cfcfabc8d1366553071a75960365a
```

It contains Dynamo `2914c20e23d4c13fd00431c8d533f5214b0849d8` and
SGLang `df1f4d76322457eddd0fea97276b153fe00115bd`. Pin the digest in
Kubernetes manifests. The previous `fb1d8d5e0b3e-df1f4d763224` image is
incompatible with the SGLang source because it contains `sgl-deep-gemm==0.1.0`.

The replacement was validated on B200 with DeepSeek-V2-Lite BF16, DEP2,
DeepEP low-latency mode, the DeepGEMM runner, decode CUDA graphs through batch
size 8, and a successful generated response.

Build one immutable image after both worktrees are clean. The build archives
the committed Dynamo `components/src` and SGLang `python` trees on the matching
SGLang `v0.5.13.post1` CUDA 13 runtime, installs the Dynamo runtime wheel, and
records both full SHAs as OCI labels. A build-time check validates compiled
dependency versions and migration imports. It does not require source mounts
at runtime.

```bash
export DYNAMO_ROOT=/root/proj/decode-migration/dynamo
export SGLANG_ROOT=/root/proj/decode-migration/sglang
export DYNAMO_SHA=$(git -C "$DYNAMO_ROOT" rev-parse HEAD)
export SGLANG_SHA=$(git -C "$SGLANG_ROOT" rev-parse HEAD)
export IMAGE=aphoh/not-sglang-dynamo:${DYNAMO_SHA:0:12}-${SGLANG_SHA:0:12}

DYNAMO_ROOT=$DYNAMO_ROOT \
SGLANG_ROOT=$SGLANG_ROOT \
IMAGE=$IMAGE \
  $DYNAMO_ROOT/examples/backends/sglang/decode_migration/build_image.sh
```

Verify that the image resolves both feature trees without bind mounts:

```bash
docker run --rm --entrypoint python3 "$IMAGE" -c '
import dynamo.sglang.request_handlers.llm.decode_handler as dh
import sglang.srt.disaggregation.decode_migration as dm
print(dh.__file__)
print(dm.__file__)
'
docker inspect "$IMAGE" --format '{{json .Config.Labels}}'
```

Authenticate without placing the registry token in shell history, then push:

```bash
read -rsp "Docker PAT: " DOCKER_PAT; echo
printf "%s" "$DOCKER_PAT" | docker login -u aphoh --password-stdin
unset DOCKER_PAT
docker push "$IMAGE"
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}'
```

Use the resulting digest, rather than the mutable tag, in the Kubernetes
deployment.

## Deployment Shape

The minimum migration deployment is distributed across two worker pods:

| Role | Pods | GPUs per pod | SGLang topology | Taint |
|---|---:|---:|---|---|
| Fast reasoning source | 1 | 8 | TP8, DP8 attention, EP8 | `decode/fast` |
| Slow visible destination | 1 | 2 | TP2, DP2 attention, EP2 | `decode/slow` |

The frontend is a separate CPU process. The workers do not need to share a
node. Each pod sees only its assigned GPUs, so neither command uses
`CUDA_VISIBLE_DEVICES`.

DeepSeek-V2-Lite is an MLA model and does not need the GQA/MHA heterogeneous-TP
staging buffer. The source and destination use the same BF16 checkpoint, page
size, context length, KV dtype, pipeline layout, and NIXL transport.

DeepSeek-V2-Lite is not a thinking model. This controlled benchmark treats the
first 60% of generated tokens as the reasoning stage and uses a sequence-length
trigger. For a thinking model, replace that trigger with its end-of-thinking
token ID.

## Frontend

Run the normal Dynamo frontend:

```bash
python3 -m dynamo.frontend \
  --http-port 8000 \
  --namespace dynamo \
  --router-mode kv \
  --enable-decode-migration
```

Do not deploy the old Python migration frontend. Decode migration is a normal
frontend operator enabled by `--enable-decode-migration`.

## Worker Environment

Set these environment variables on both worker pods:

```bash
SGLANG_ENABLE_JIT_DEEPGEMM=1
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
SGLANG_DISAGG_STAGING_BUFFER=0
```

Use the deployment-provided Dynamo discovery and request-plane environment.
Do not force the file discovery backend used by the local smoke script.

## Fast Source Worker

Run this command in an 8-GPU pod:

```bash
python3 -m dynamo.sglang \
  --endpoint dyn://dynamo.backend.generate \
  --model-path /models/deepseek-v2-lite \
  --served-model-name deepseek-ai/DeepSeek-V2-Lite \
  --host 0.0.0.0 \
  --port 30000 \
  --disaggregation-bootstrap-port 8998 \
  --tensor-parallel-size 8 \
  --data-parallel-size 8 \
  --expert-parallel-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode low_latency \
  --dtype bfloat16 \
  --context-length 32768 \
  --page-size 16 \
  --max-running-requests 16 \
  --mem-fraction-static 0.60 \
  --cuda-graph-bs-decode 1 2 4 8 16 \
  --disaggregation-transfer-backend nixl \
  --enable-decode-migration \
  --worker-taint decode/fast \
  --stream-interval 1
```

The selected operating point is 16 concurrent requests per DP rank, or 128
requests across the DEP8 pod. Do not add `--disable-overlap-schedule` or
`--disable-cuda-graph`.

## Slow Destination Worker

Run this command in a 2-GPU pod:

```bash
python3 -m dynamo.sglang \
  --endpoint dyn://dynamo.backend.generate \
  --model-path /models/deepseek-v2-lite \
  --served-model-name deepseek-ai/DeepSeek-V2-Lite \
  --host 0.0.0.0 \
  --port 30000 \
  --disaggregation-bootstrap-port 8998 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --expert-parallel-size 2 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode low_latency \
  --dtype bfloat16 \
  --context-length 32768 \
  --page-size 16 \
  --max-running-requests 128 \
  --mem-fraction-static 0.60 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 128 \
  --disaggregation-transfer-backend nixl \
  --enable-decode-migration \
  --worker-taint decode/slow \
  --stream-interval 1
```

The selected operating point is 128 concurrent requests per DP rank, or 256
requests across the DEP2 pod. If graph capture at 128 exceeds memory, lower
both the graph sizes and `--max-running-requests`, remeasure standalone DEP2,
and label the new configuration separately.

## Kubernetes Requirements

The deployment agent may choose the concrete manifests, but it must preserve
these properties:

- source and destination pods have disjoint GPU allocations;
- all worker replicas register the same public model and `generate`,
  `migration_prepare`, `migration_sync`, and `migration_finalize` endpoints;
- worker taints are published in Dynamo discovery;
- pod IPs or advertised worker addresses are mutually routable;
- TCP port 8998 is reachable directly between worker pods for bootstrap;
- the selected NIXL data path and any required RDMA devices are available;
- source and destination ranks remain distinguishable in discovery metadata;
- frontend cancellation reaches both workers during handoff; and
- termination grace periods allow migration cleanup to run.

Do not encode rank identity in the bootstrap room. The room is opaque; source
and destination rank metadata travels explicitly in the migration protocol.

## Request Shape

A migration request is opt-in through `nvext`. For ISL 1, OSL 512, and a 60%
fast-stage fraction, migrate at total sequence length 308:

```json
{
  "model": "deepseek-ai/DeepSeek-V2-Lite",
  "prompt": [0],
  "temperature": 0,
  "max_tokens": 512,
  "ignore_eos": true,
  "stream": true,
  "nvext": {
    "decode_migration": {
      "source": {"required_taints": ["decode/fast"]},
      "destination": {"required_taints": ["decode/slow"]},
      "trigger": {"type": "sequence_length", "tokens": 308}
    },
    "extra_fields": ["completion_token_ids"]
  }
}
```

Every worker can send and receive. The taints express placement policy, not
one-way worker capabilities.

## Benchmark

Run the stdlib-only open-loop client from the SGLang checkout in a benchmark
pod or operator shell that can reach the frontend Service:

```bash
python3 examples/backends/sglang/decode_migration/static_decode_pareto.py \
  --base-url http://dynamo-frontend:8000 \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --mode migration \
  --arrival-rate 70 \
  --requests 256 \
  --warmup-seconds 5 \
  --cooldown-seconds 5 \
  --max-tokens 512 \
  --source-fraction 0.6 \
  --gpu-count 10 \
  --min-visible-rate 20 \
  --output /results/migration-rps-70.json
```

Sweep approximately 40, 50, 60, 70, 78, 82, and 86 requests/second without
restarting the worker pods. Refine around the first point with sustained queue
growth or TTFNT drift.

For the baseline, deploy only the DEP8 source worker, use `--mode baseline`,
`--gpu-count 8`, and sweep approximately 30, 40, 46, 50, and 54 requests/second.
Do not count an idle destination deployment as part of the baseline.

For a truncated-normal OSL distribution centered at 512, add:

```text
--osl-stddev 64 --osl-min 256 --osl-max 768
```

The trigger remains request-relative at 60% of each sampled OSL.

## Rate-Matched Scale Example

One DEP8 source pod and one DEP2 destination pod are useful for correctness and
locating queue knees, but they are not perfectly rate matched. A 64 decode-GPU
starting point is seven DEP8 source pods and four DEP2 destination pods:

```text
fast GPUs = 7 * 8 = 56
slow GPUs = 4 * 2 = 8
total     = 64
```

Scale whole worker replicas. Confirm balance from per-tier queue depth and
generated token rate rather than assuming the analytical ratio is exact.

## Acceptance Gates

Accept a migration point only when:

1. all measured requests complete with the expected output length;
2. the frontend records one committed migration per migrated request;
3. no prepare, sync, activation, finalize, cancellation, or reservation state
   leaks remain;
4. overlap scheduling and CUDA graphs remain enabled;
5. early-half and late-half TTFNT show stationary queues;
6. visible decode remains at or above 20 tokens/second/user;
7. no worker restarts, OOMs, or scheduler invariant failures occur; and
8. paired task accuracy remains within the accepted baseline tolerance.

Record token fingerprints for debugging loss or duplication. Require exact
fingerprint parity in deterministic same-topology smoke tests; heterogeneous
DEP8-to-DEP2 execution may have legitimate numerical drift.

The standalone measurements motivating these settings were approximately
222.8 token/s/user and 3201.1 output token/s/GPU for DEP8 at 16 concurrent
requests/GPU, and 143.4 token/s/user and 15441.5 output token/s/GPU for DEP2 at
128 concurrent requests/GPU. Treat them as sanity ranges, not acceptance
thresholds on different hardware or software.

Multi-DP-rank migration remains an experimental validation target until this
distributed deployment passes every gate above.
