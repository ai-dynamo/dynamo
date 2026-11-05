<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TokenSpeed Disaggregated Serving

**Experimental.** Run NVIDIA Dynamo with two TokenSpeed prefill replicas, one
decode replica, and KV-aware routing. TokenSpeed transfers the KV cache with
Mooncake; Dynamo selects workers and forwards the bootstrap rendezvous.

This example uses
[`meituan-longcat/LongCat-Flash-Chat-FP8`](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat-FP8).
TokenSpeed revision
[`1b859c107b1d`](https://github.com/lightseekorg/tokenspeed/tree/1b859c107b1d8450b750403f9c2e00251ecc59f3)
registers its `LongcatFlashForCausalLM` architecture. The
[official TokenSpeed recipe catalog](https://lightseek.org/tokenspeed/recipes/models)
does not list LongCat. This is an experimental Dynamo integration example.

## Prerequisites

- Three hosts, with either eight H200 GPUs or four GB300 GPUs per host: one
  host per replica. TokenSpeed lists both among its
  [supported GPUs](https://github.com/lightseekorg/tokenspeed/blob/1b859c107b1d8450b750403f9c2e00251ecc59f3/.github/ISSUE_TEMPLATE/1-bug-report.yml#L13-L15),
  with CUDA 13 and driver 580 or newer recommended. This is general hardware
  support, not a LongCat-specific recipe. The model owner's
  [deployment guide](https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/docs/deployment_guide.md)
  describes FP8 on eight 141 GB H20 GPUs with SGLang or vLLM.
- GB300 requires the native LongCat routing fix in
  [TokenSpeed PR #1582](https://github.com/lightseekorg/tokenspeed/pull/1582).
  The build instructions below apply the exact tested source.
- A working RDMA fabric between hosts, with GPU memory registration available to
  Mooncake. Expose `/dev/infiniband` to containers and use unlimited locked memory.
- Shared etcd and NATS services reachable from all processes. See
  [runtime services](../../README.md#getting-started).
- The same model checkpoint and TokenSpeed revision on every worker. Reserve at
  least 900 GB of disk per host; the pinned checkpoint contains about 748 GB of weights.

Attention data parallelism must be one inside each worker. Scale with independent
replicas. Both prefill and decode must use matching tensor/expert parallelism and
cache geometry. The launcher's H200 default uses `flashinfer_cutlass`. GB300 uses
`flashinfer_trtllm` with the native routing fix and the flags below.

## Build the Image

### H200

From the Dynamo repository root, use the shared TokenSpeed image builder:

```bash
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --target runtime \
  --build-arg BASE_IMAGE=lightseekorg/tokenspeed-runner@sha256:8187a564ba7fd3cb8b13dfa7fe697c61b894b518b5ee57fde7ca845ffb17c3ef \
  --build-arg TOKENSPEED_GIT_REF=1b859c107b1d8450b750403f9c2e00251ecc59f3 \
  -t dynamo-tokenspeed:longcat-flash .
```

This pins the upstream NVIDIA runner for x86 H200 hosts. It installs TokenSpeed
from source and builds Dynamo's Rust extension from the current checkout.
Make the resulting image available on all three hosts.

### GB300

Build on an ARM64 host with the matching runner:

```bash
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --target runtime \
  --build-arg BASE_IMAGE=lightseekorg/tokenspeed-runner@sha256:2d9477bc2417572be8740676e8e038b564938c9ab466421f23daca441c97b153 \
  --build-arg TOKENSPEED_GIT_REF=1b859c107b1d8450b750403f9c2e00251ecc59f3 \
  --build-arg CARGO_BUILD_JOBS=32 \
  -t dynamo-tokenspeed:longcat-flash-gb300-base .
```

Apply the pinned native LongCat fix. It selects precomputed SwiGLU expert
routing, which LongCat's zero-expert path requires. The base revision selects an
incompatible MoE plan for the Blackwell FP8 backend.

```bash
patch_dir=$(mktemp -d)
curl --fail --location \
  https://raw.githubusercontent.com/nv-yna/tokenspeed/a8a41686835e3b255670857424f9f8f9b70c0205/python/tokenspeed/runtime/models/longcat_flash.py \
  --output "$patch_dir/longcat_flash.py"
printf '%s  %s\n' \
  949c96de91cdc85312d2d6b9e596a02f8822a676d6dcdd454321483e91a411ee \
  "$patch_dir/longcat_flash.py" | sha256sum --check
cat > "$patch_dir/Dockerfile" <<'DOCKERFILE'
FROM dynamo-tokenspeed:longcat-flash-gb300-base
COPY longcat_flash.py /opt/tokenspeed/python/tokenspeed/runtime/models/longcat_flash.py
DOCKERFILE
docker build -t dynamo-tokenspeed:longcat-flash-gb300 "$patch_dir"
```

Make this patched image available on all three GB300 hosts.

## Download the Checkpoint

Run inside each worker's container with its model-storage directory mounted at
`/models`:

```bash
hf download meituan-longcat/LongCat-Flash-Chat-FP8 \
  --revision a373e08a1c4897ab12cbac2a540e497502d55118 \
  --local-dir /models/longcat-flash-fp8
```

The pinned checkpoint omits `model_type` from `config.json`, although its
[`LongcatFlashConfig`](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat-FP8/blob/a373e08a1c4897ab12cbac2a540e497502d55118/configuration_longcat_flash.py)
declares `longcat_flash`. Dynamo's frontend requires this field. Prepare the
metadata on every host before starting workers:

```bash
python3 examples/backends/tokenspeed/prepare_longcat.py /models/longcat-flash-fp8
```

The script saves the original config as `config.json.dynamo-original` and adds the
missing field. Restart existing workers after preparation so they publish the
updated model metadata.

## Launch

Run the commands below inside containers with all GPUs on the host exposed, host
networking, host IPC, RDMA devices, and `--ulimit memlock=-1`.

Set these variables in every process's environment:

| Variable | Value |
| --- | --- |
| `ETCD_ENDPOINTS` | Reachable etcd URL, including port |
| `NATS_SERVER` | Reachable NATS URL, including port |
| `DYN_NAMESPACE` | `tokenspeed` |
| `MODEL_PATH` | `/models/longcat-flash-fp8` |

On each prefill host, set `DYN_TOKENSPEED_BOOTSTRAP_HOST` to that host's reachable
address. Use `DISAGG_IB_DEVICE` to select an RDMA device when automatic selection
does not match the connected fabric. On hosts without `nvidia-peermem`, set
`WITH_NVIDIA_PEERMEM=0` in the worker containers to use Mooncake
[DMA-BUF registration](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md).

Start one prefill process on each of the first two hosts:

```bash
export DYN_SYSTEM_PORT=8081
bash examples/backends/tokenspeed/launch_disagg.sh prefill
```

Start decode on the third host:

```bash
export DYN_SYSTEM_PORT=8081
bash examples/backends/tokenspeed/launch_disagg.sh decode
```

Start the frontend in a separate process on the first host:

```bash
export DYN_SYSTEM_PORT=8080
bash examples/backends/tokenspeed/launch_disagg.sh frontend
```

The launcher sets attention TP=8, EP=8, an 8,192-token context limit, and prefix
granularity 64, with a 32,768-token cache capacity. It enables native KV events and gives each worker a unique local
IPC socket. Additional command-line arguments are forwarded to the selected
process. Use `ENGINE_PORT`, `BOOTSTRAP_PORT`, and `HTTP_PORT` to change ports.

### GB300 Worker Options

Use these settings for LongCat-Flash on 12 GB300 GPUs: two prefill workers and one
decode worker, each with TP=4 and EP=4. Use the shared environment above, then set
these on every GB300 worker:

```bash
export TENSOR_PARALLEL_SIZE=4
export MAX_TOTAL_TOKENS=917504
export DYN_SYSTEM_PORT=8081
```

Start each prefill worker:

```bash
bash examples/backends/tokenspeed/launch_disagg.sh prefill \
  --max-model-len 131072 --max-num-seqs 32 \
  --attention-backend trtllm_mla \
  --moe-backend flashinfer_trtllm --force-deterministic-rsag
```

Start decode:

```bash
bash examples/backends/tokenspeed/launch_disagg.sh decode \
  --max-model-len 131072 --max-num-seqs 32 \
  --attention-backend trtllm_mla \
  --moe-backend flashinfer_trtllm --force-deterministic-rsag \
  --disable-prefill-graph
```

`--force-deterministic-rsag` uses the native NCCL collective path. Disabling the
decode worker's prefill graph avoids a failure during its optional prefill graph
capture; decode CUDA graphs remain enabled. Start the frontend as above.

Apply the sequence and KV-cache limits to both roles. Prefill waits for decode-side
KV allocation, so insufficient decode capacity can queue prefill work. The
`trtllm_mla` attention backend addresses the measured long-context decode slowdown;
`flashinfer_trtllm` selects the separate MoE backend.

A recorded 20-minute AgentX run at concurrency 16 with these settings delivered
208.38 output tokens/s, 0.735 s median time to first token, and 11.7 ms median
inter-token latency. It completed 405 successful requests and recorded one
empty-response error. These are single-run measurements for this model and
topology. Separate functional checks verified KV transfer and cache-aware
prefill selection.

## Verify Generation and KV Routing

Run from the frontend host with the same etcd, NATS, and namespace settings:

```bash
python3 examples/backends/tokenspeed/validate_disagg.py \
  --tokenizer /models/longcat-flash-fp8 \
  --output longcat-disagg-results.json
```

The validator warms distinct prefixes on both prefill workers, waits for Dynamo
to observe the native KV events, then sends repeated requests without a worker
override. It checks that each request returns generated text, identifies both
prefill and decode workers, and selects the worker holding its cached prefix.
A third, uncached prefix provides a control. The JSON output contains responses,
worker IDs, overlap scores, timings, and a `passed` result.

## Configuration Notes

- `--disaggregation-mode prefill` registers a prefill worker and its bootstrap
  host/port. Decode registers separately under `backend.generate`.
- Prefill generates one token for the native handoff. Decode retains the client's
  generation budget.
- `--kv-events-config '{"enable_kv_cache_events":true}'` enables native cache
  mutation events. Prefix caching must remain enabled.
- When specifying an event endpoint manually, use `ipc://...` or
  `tcp://*:PORT`. TokenSpeed connects to a concrete IPv4 TCP address; Dynamo's
  subscriber requires TokenSpeed to bind.
- `DYN_TOKENSPEED_BOOTSTRAP_HOST` overrides the advertised host without changing
  TokenSpeed's listening address. It must be reachable from decode hosts.
