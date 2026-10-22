<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepInfer Backend

**Experimental.** `python -m dynamo.deepinfer` runs
[DeepInfer](https://github.com/deepseek-ai/DeepInfer) inside a Dynamo worker on the unified
backend. The target is DeepSeek-V4.1-Flash on Blackwell (GB200, GB300), aggregated.

One worker serves one DeepInfer engine. It runs one DeepInfer data-parallel (DP) rank per GPU:
rank 0 in the worker process and the others as child processes, stepped in lockstep. The worker
registers with Dynamo as one worker with `data_parallel_size` equal to the engine's rank count, so
a request can name its rank (`routing.dp_rank`); otherwise the least-loaded rank takes it. An
engine usually fills one node; it can also [span several nodes](#run-one-engine-on-several-nodes).

| Feature | Status |
|---|---|
| Aggregated serving | Supported |
| Data and expert parallelism (DEP) over the GPUs of one or more nodes | Supported |
| Reasoning and tool-call parsing | Through Dynamo (`deepseek_v41` parsers) |
| Logprobs | Supported |
| Prefix cache and KV events (`--kv-cache-manager prefix`) | Supported, 128-token blocks |
| Disaggregated prefill/decode (`--disaggregation-mode`) | Supported, KV and state read over NIXL (UCX) |
| Multimodal input | Not supported |

## Image

DeepInfer builds only for Blackwell (`sm_100a`). Build natively on arm64 for GB200 and GB300;
compiling FlashMLA needs a lot of RAM. From the repository root:

```bash
docker buildx build --platform linux/arm64 \
  -f examples/backends/deepinfer/Dockerfile -t <registry>/deepinfer-dynamo:<tag> .
```

The image contains:

| Path | Contents |
|---|---|
| `/opt/DeepInfer` | DeepInfer `774ce24` |
| `/opt/venv` | DeepInfer and the `ai-dynamo` runtime wheels. The worker runs here (`python`) |
| `/opt/dynamo-frontend` | A separate venv for `dynamo.frontend` with the full `ai-dynamo` dependency set |
| `/opt/dynamo-deepinfer` | This checkout's `dynamo.deepinfer`. Mount a newer copy here to iterate without rebuilding |

## Run on One Node

[`launch_agg.sh`](launch_agg.sh) starts the frontend and one aggregated worker in one container,
with file-based discovery, so it needs no etcd or NATS. Run it inside the image, from a checkout
of this repository. The script stops early when the checkpoint or the RDMA devices are missing.

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash examples/backends/deepinfer/launch_agg.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `DEEPINFER_HF_MODEL_DIR`, else a Hugging Face cache path | Checkpoint directory |
| `NUM_GPUS` | `4` | DP and EP size |
| `DYN_HTTP_PORT` | `8000` | Frontend port |
| `DYN_SYSTEM_PORT` | `8081` | Worker health and metrics port |
| `FRONTEND_PYTHON` | `/opt/dynamo-frontend/bin/python` | Interpreter for the frontend |

Arguments go to the worker after its defaults. Send a request once the worker logs that its
ranks are ready:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 64
}'
```

Expert parallelism fetches DeepSeek-V4's host-memory Engram tables over NCCL GIN, which needs RDMA
devices in the container. In a container that may use only some of the node's CPUs (a Kubernetes
pod), DeepInfer's CPU binding fails with `CPU number N is not eligible`: it assumes it owns the
whole node. `launch_agg.sh` puts [`cpu_affinity/sitecustomize.py`](cpu_affinity/sitecustomize.py)
on the worker's `PYTHONPATH`, which keeps each rank's binding inside the CPUs the container has.

## Run One Engine on Several Nodes

A DeepInfer engine with `--dp-size D` can span `--nnodes N` nodes, `D / N` ranks (GPUs) per node.
The head node (`--node-rank 0`) runs the Dynamo worker for the whole engine: it registers one
worker with `data_parallel_size=D` and drives every rank. The other nodes run
`python -m dynamo.deepinfer.node`, which starts that node's ranks; they connect to the head on
`--control-port`. Give every node the same DeepInfer flags: the head refuses a rank that was
started with different ones, and the node launcher ignores the Dynamo runtime flags, so it can
take the head's full command line. For example, a DEP16 engine on four 4-GPU nodes:

```bash
# Node 0 (the head). HEAD_IP is an address of node 0 that the other nodes reach.
python -m dynamo.deepinfer --model "$MODEL_PATH" \
  --dp-size 16 --nnodes 4 --node-rank 0 --master-addr "$HEAD_IP" \
  --master-port 23333 --control-port 23400

# Nodes 1, 2 and 3: the same flags and their own --node-rank.
python -m dynamo.deepinfer.node --model "$MODEL_PATH" \
  --dp-size 16 --nnodes 4 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP" \
  --master-port 23333 --control-port 23400
```

| Option | Default | Meaning |
|---|---|---|
| `--nnodes` | `1` | Nodes the engine spans; `--dp-size` must be a multiple |
| `--node-rank` | `0` | `0` on the head (`python -m dynamo.deepinfer`), `1..N-1` on the others |
| `--master-addr` | `MASTER_ADDR`, else `127.0.0.1` on one node | Head address for DeepInfer's rendezvous and the controller; required with `--nnodes` > 1 |
| `--master-port` | `MASTER_PORT`, else `23333` | DeepInfer's `MASTER_PORT`; its gloo store listens on this port minus 50 |
| `--control-port` | `23400` | Port on the head where the other nodes' ranks connect |
| `--node-timeout-s` | `30` | A silent connection to a node counts as lost after this long |
| `--shutdown-grace-s` | `90` | `dynamo.deepinfer.node` only: time its ranks get after SIGTERM to finish the head's shutdown |

Each rank uses one GPU: its position on the node in `CUDA_VISIBLE_DEVICES`, or GPUs `0..R-1`
when it is unset. Expert parallelism across nodes needs MNNVL (one NVLink domain, IMEX) or NCCL
GIN over RDMA, and `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` must name the interface the nodes
share. Non-head nodes have no Dynamo worker and no health port.

When a rank or node is lost (process exit, host down, or a connection silent for
`--node-timeout-s`), the head worker exits within a few seconds and the other nodes' launchers
stop their ranks and exit non-zero. A launcher that gets SIGTERM waits for the head's shutdown,
then kills its ranks after `--shutdown-grace-s`.

Two engines can share a node, for example two DEP2 engines on GPUs 0,1 and 2,3. Give each its own
`CUDA_VISIBLE_DEVICES`, `--master-port` and `DYN_SYSTEM_PORT`, and, for an engine that spans
nodes, its own `--control-port`:

```bash
CUDA_VISIBLE_DEVICES=0,1 DYN_SYSTEM_PORT=8081 python -m dynamo.deepinfer --model "$MODEL_PATH" \
  --dp-size 2 --master-port 23333 &
CUDA_VISIBLE_DEVICES=2,3 DYN_SYSTEM_PORT=8082 python -m dynamo.deepinfer --model "$MODEL_PATH" \
  --dp-size 2 --master-port 24333 &
```

## Run on SLURM with srt-slurm

[`srt-slurm/gb300-disagg-6p1d.yaml`](srt-slurm/gb300-disagg-6p1d.yaml) is an srt-slurm (`srtctl`)
config for disaggregated serving on 12 GB300 nodes. It runs six DEP2 prefill workers with the
prefix cache's host tier, and one DEP32 decode worker with expert-parallel load balancing. It is
tuned for agentic coding traffic at about 3,000 concurrent sessions, and needs an srt-slurm build
that supports `engine: deepinfer`.

## Worker Options

`python -m dynamo.deepinfer --help` lists every option. The main ones:

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | Checkpoint directory |
| `--dp-size`, `--ep-size` | local GPU count × `--nnodes`, `--dp-size` | DeepInfer DP and expert-parallel size over all nodes |
| `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` | DeepInfer defaults | Engine limits |
| `--engine-arg KEY=VALUE` | none | Any other DeepInfer `EngineArgs` field |
| `--no-rank0-in-process` | off | Run the lockstep controller and every rank of this node in a process of their own, so the worker's Python only serves requests and outputs |

The worker sets DeepInfer environment defaults before importing DeepInfer and keeps any value the
environment already has. It never sets `EP_DISABLE_GIN` or `NCCL_IB_HCA`; those belong to the
pod.

Debugging: `DYN_DEEPINFER_TRACE_LOCKSTEP=1` logs one line per lockstep iteration on the controller
and on every rank, plus a summary every 10 s.
