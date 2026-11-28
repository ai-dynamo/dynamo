---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: DeepInfer
---

> [!NOTE]
> The DeepInfer backend is experimental.

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) is DeepSeek's inference engine for
DeepSeek-V4 models on Blackwell GPUs. It runs a model offline and has no online serving API.
`python -m dynamo.deepinfer` runs a DeepInfer engine inside a Dynamo worker on the unified backend
(`dynamo.common.backend`), so Dynamo supplies the OpenAI-compatible frontend, discovery, request
routing to data-parallel ranks, cancellation and metrics.

## Feature Support Matrix

| Feature | Status | Notes |
| :--- | :---: | :--- |
| Aggregated serving | Ready | |
| Data and expert parallelism (DEP) | Ready | One DP rank per GPU, on one or more nodes |
| Reasoning and tool-call parsing | Ready | Dynamo's `deepseek_v41` parsers |
| Logprobs | Ready | |
| Request cancellation | Ready | |
| Prefix cache and KV events | Ready | `--kv-cache-manager prefix`; needs the DeepInfer bindings patch in the example image | |
| Disaggregated serving | Ready | `--disaggregation-mode prefill/decode`, KV read over NIXL |
| Multimodal input | Not supported | |
| `n` > 1 | Not supported | |

## Requirements

- Blackwell GPUs (`sm_100a`), such as GB200 or GB300. DeepInfer's kernels build for no other
  architecture.
- DeepInfer built from source. The
  [example Dockerfile](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/deepinfer/Dockerfile)
  builds DeepInfer `774ce24` with the Dynamo wheels.
- A DeepSeek-V4.1-Flash checkpoint.
- RDMA devices in the container: DeepInfer's expert parallelism fetches DeepSeek-V4's host-memory
  Engram tables over NCCL GIN. Across nodes, expert parallelism needs MNNVL or NCCL GIN over RDMA.

## Quick Start

The [DeepInfer example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/deepinfer)
builds the image and runs the frontend and an aggregated DEP4 worker on one node:

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash examples/backends/deepinfer/launch_agg.sh
```

## How It Works

### One worker per engine

A DeepInfer engine runs one data-parallel (DP) rank per GPU, and its ranks run collectives
together at every step: the `has_unfinished_requests` all-reduce, the prefill votes, the
expert-parallel forward and the `close()` barrier. Dynamo runs one worker per engine and registers
it with `data_parallel_size` equal to the engine's rank count. A request goes to the rank its
`routing.dp_rank` names, or else to the least-loaded rank.

Rank 0 runs in the worker process. The worker builds it, and starts ranks 1 to N-1 as child
processes, before the Rust worker and its Tokio threads exist. Child ranks talk to the worker over
ZMQ on `ipc://`. With `--no-rank0-in-process`, the controller and every rank run in a separate
process instead, so the worker's Python only serves requests and streams outputs.

### Lockstep

A controller drives every rank one iteration at a time. Each iteration it sends every rank one
command with that rank's new and aborted requests and the iteration's global flags: whether any
rank has unfinished requests, whether this is the closing iteration, and whether new prompts
arrived. The ranks take the flags from the command instead of running DeepInfer's per-step
all-reduce, so they all make the same step-or-flush choice and reach the same collectives. Each
rank replies with its outputs and statistics, and the controller checks that the ranks agree.
New prompts make the next step a prefill vote point on every rank, where a rank with new prompts
may prefill even when another rank cannot. While no rank has work, the controller waits on its
inbox instead of stepping.

A rank that dies or diverges leaves the others blocked in a collective, so the worker exits; the
deployment restarts it. Shutdown is one closing iteration in which every rank runs DeepInfer's
`close()`.

### Engines that span nodes

An engine with `--dp-size D` can span `--nnodes N` nodes with `D / N` ranks on each. The head
node runs `python -m dynamo.deepinfer`, which registers one worker for the whole engine; each
other node runs `python -m dynamo.deepinfer.node`, which starts that node's ranks. Remote ranks
connect to the head's controller over TCP on `--control-port`, with heartbeats: a node that stops
answering for `--node-timeout-s` is lost, and the engine stops. The head refuses a rank started
with different DeepInfer options or a different lockstep-relevant `DEEPINFER_*` environment.

## Configuration

`python -m dynamo.deepinfer --help` lists every option.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--model` | required | Checkpoint directory |
| `--dp-size`, `--ep-size` | local GPU count × `--nnodes`, `--dp-size` | DP and expert-parallel size over all nodes |
| `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens` | DeepInfer defaults | Engine limits |
| `--engine-arg KEY=VALUE` | none | Any other DeepInfer `EngineArgs` field |
| `--no-rank0-in-process` | off | Controller and ranks in a separate process |
| `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`, `--control-port`, `--node-timeout-s` | one node | Engines that span nodes |

The worker sets DeepInfer environment defaults (for example `DEEPINFER_GLOO_TIMEOUT=600`) before
it imports DeepInfer, and keeps any value the environment already has.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `DYN_DEEPINFER_CONTROLLER_PROCESS` | `1` | With `--no-rank0-in-process`, `0` keeps the controller on a thread of the worker |
| `DYN_DEEPINFER_TRACE_LOCKSTEP` | `0` | `1` logs one line per iteration on the controller and every rank, and a summary every 10 s |
| `DYN_DEEPINFER_EVICT_CHECKPOINT_CACHE` | `1` | Drop the checkpoint from the page cache before DeepEP pins host memory for the Engram tables, and retry that allocation after an out-of-memory error |

## Testing

The unit tests run on CPU against a fake of DeepInfer's Python surface
(`components/src/dynamo/deepinfer/testing/fake_deepinfer.py`), with real rank processes and
sockets; they need Linux but no GPU or DeepInfer:

```bash
pytest components/src/dynamo/deepinfer/tests
```

`python -m dynamo.deepinfer.testing.fake_load` runs one engine of up to 32 fake ranks under a
full batch and reports the lockstep period, which checks the controller's overhead without GPUs.

## Limitations

- Without a prefix cache, every request prefills its whole prompt, and the KV router has no cache
  view of the worker.
- DeepInfer stops a request only at EOS, `max_tokens` or `max_model_len`. Dynamo handles stop
  strings and stop token ids and aborts the request. A request without `max_tokens` may generate
  up to the remaining context length.
