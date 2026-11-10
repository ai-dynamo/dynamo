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
[`0e6ee353d7bd`](https://github.com/lightseekorg/tokenspeed/tree/0e6ee353d7bdc7afeb1c5795d839533eab7ecb9a),
the November 2, 2026 nightly build, registers its `LongcatFlashForCausalLM`
architecture. The
[official TokenSpeed recipe catalog](https://lightseek.org/tokenspeed/recipes/models)
does not list LongCat. This is an experimental Dynamo integration example.

## Prerequisites

- Three hosts, with either eight H200 GPUs or four GB300 GPUs per host: one
  host per replica. TokenSpeed lists both among its
  [supported GPUs](https://github.com/lightseekorg/tokenspeed/blob/0e6ee353d7bdc7afeb1c5795d839533eab7ecb9a/.github/ISSUE_TEMPLATE/1-bug-report.yml#L13-L15),
  with CUDA 13 and driver 580 or newer recommended. This is general hardware
  support, not a LongCat-specific recipe. The model owner's
  [deployment guide](https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/docs/deployment_guide.md)
  describes FP8 on eight 141 GB H20 GPUs with SGLang or vLLM.
- GB300 requires the native LongCat routing fix in
  [TokenSpeed PR #1582](https://github.com/lightseekorg/tokenspeed/pull/1582),
  which the pinned revision includes.
- A working RDMA fabric between hosts, with GPU memory registration available to
  Mooncake. Expose `/dev/infiniband` to containers and use unlimited locked memory.
- Shared etcd and NATS services reachable from all processes. See
  [runtime services](../../README.md#getting-started).
- The same model checkpoint and TokenSpeed revision on every worker. Reserve at
  least 900 GB of disk per host; the pinned checkpoint contains about 748 GB of weights.

A worker can run attention data parallelism (`--data-parallel-size N
--attn-tp-size 1`), including across nodes; decode finds a request's prefill rank
from its bootstrap room, so the two roles' attention-DP sizes may differ. Both
must use matching attention tensor parallelism and cache geometry. The launcher's H200 default uses `flashinfer_cutlass`. GB300 uses
`flashinfer_trtllm` with the native routing fix and the flags below.

## Build the Image

From the Dynamo repository root, use the shared TokenSpeed image builder. It
installs the pinned TokenSpeed nightly wheels and builds Dynamo's Rust extension
from the current checkout.

For x86 H200 hosts:

```bash
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --target runtime \
  -t dynamo-tokenspeed:longcat-flash .
```

For GB300, build on an ARM64 host:

```bash
docker build \
  -f recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile \
  --target runtime \
  --build-arg ARCH=arm64 \
  --build-arg CARGO_BUILD_JOBS=32 \
  -t dynamo-tokenspeed:longcat-flash-gb300 .
```

The pinned revision includes the native LongCat routing fix, so no source patch is
needed. Make the image available on all three hosts.

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

The launcher sets attention TP=8, EP=8, an 8,192-token context limit, and a
32,768-token cache capacity, and tells Dynamo that the engine's KV prefix
granularity is 64 tokens (`KV_BLOCK_SIZE`; see the configuration notes). It
enables native KV events and gives each worker a unique local IPC socket.
Additional command-line arguments are forwarded to the selected process. Use `ENGINE_PORT`, `BOOTSTRAP_PORT`, and `HTTP_PORT` to change ports.

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

A recorded 20-minute AgentX run, taken with TokenSpeed revision `1b859c107b1d`
before the worker moved to TokenSpeed's msgpack wire, at concurrency 16 with these settings delivered
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
  mutation events. Prefix caching must remain enabled. KV events require attention
  DP=1. With attention DP > 1, turn them off on the workers and the router.
  `launch_disagg.sh` enables them by default, so set `KV_EVENTS_CONFIG` for the
  workers and pass `--no-router-kv-events` to the frontend:

  ```bash
  KV_EVENTS_CONFIG='{"enable_kv_cache_events":false}' \
    bash examples/backends/tokenspeed/launch_disagg.sh prefill --data-parallel-size 4
  KV_EVENTS_CONFIG='{"enable_kv_cache_events":false}' \
    bash examples/backends/tokenspeed/launch_disagg.sh decode --data-parallel-size 4
  bash examples/backends/tokenspeed/launch_disagg.sh frontend --no-router-kv-events
  ```

  A non-KV router mode also works.
- With attention DP > 1, the worker registers its DP size, and a KV-mode router
  picks a rank per request. A prefill request always runs on rank
  `bootstrap_room % dp_size`, where decode looks for its KV cache; the router mints
  rooms to match the rank it picks, and the worker drops a prefill rank that
  disagrees with its room. Decode runs on the router's rank. Without a router-chosen
  rank, TokenSpeed places prefill and decode by the room.
- A multi-node worker runs `python -m dynamo.tokenspeed` with the same arguments
  and its own `--node-rank` on every node. The first node registers the worker; the
  others run only their engine ranks. With the default loopback
  `--data-parallel-address`, every node dials `--dist-init-addr`'s host on
  `--data-parallel-rpc-port` (default 30500), so that port must be free on the first
  node.
- KV events also require `--kv-block-size`, set to the engine's KV prefix
  granularity. Each TokenSpeed model picks its granularity at startup and logs it
  as `prefix_granularity=N` in its `Allocated cache arena` line. Pass the same
  value to the frontend's `--kv-cache-block-size`. With a different value, Dynamo
  drops every KV event (`Block not published` warnings) and routing loses cache
  awareness.
- The worker starts TokenSpeed's headless engine
  (`python -m tokenspeed.runtime.entrypoints.engine`) as a child process and
  serves it over TokenSpeed's msgpack ZMQ wire. It binds the wire's sockets on
  `--data-parallel-address`, at free ports on a single node.
- Guided decoding (JSON schema, regex, grammar, structural tags) requires
  `--grammar-backend xgrammar` on checkpoints whose default backend is `none`.
  Without it, TokenSpeed refuses such requests with HTTP 400.
- When specifying an event endpoint manually, use `ipc://...` or
  `tcp://*:PORT`. TokenSpeed connects to a concrete IPv4 TCP address; Dynamo's
  subscriber requires TokenSpeed to bind.
- `DYN_TOKENSPEED_BOOTSTRAP_HOST` overrides the advertised host without changing
  TokenSpeed's listening address. It must be reachable from decode hosts.
