<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mocker engine

Run `python3 -m dynamo.mocker` to launch live Mocker workers that register with the Dynamo runtime.
The command does not generate traffic or run virtual-clock replay. Use `aisimulate predict` or
`aisimulate recommend` for offline simulation.

The user-facing live deployment guide lives at
[Simulate a Kubernetes Deployment with Mocker](../../../../docs/fern/pages/kubernetes/operations/simulation-with-dynosim/mocker-live-simulation.mdx).

Useful adjacent references:

- Aggregated deployment example: [`examples/backends/mocker/deploy/agg.yaml`](../../../../examples/backends/mocker/deploy/agg.yaml)
- Disaggregated deployment example: [`examples/backends/mocker/deploy/disagg.yaml`](../../../../examples/backends/mocker/deploy/disagg.yaml)
- Global planner mocker example: [`examples/global_planner/global-planner-mocker-test.yaml`](../../../../examples/global_planner/global-planner-mocker-test.yaml)

## CPU state-cache regression recipe

Both the Rust and Python AISimulate dependencies are pinned to immutable commit
`c59a00a3fb7d531eea6bea9887235e92b535ec38` (`feat: add vLLM prefix_match_unit
(partial prefix hit) support (#314)`). From this Dynamo checkout, run:

```bash
cargo test -p dynamo-mocker --test state_cache_live --locked -- --nocapture
```

This exercises the public Dynamo `LiveEngine` request-stream API and its
AISimulate scheduler without GPUs, model downloads, a Python build, or external
services. Each case sends two serial 24300-token prompts with two output tokens
per request. The test checks actual output tokens and forward-pass work:

| State cache | Shared prefix | Restored prefix | Total committed prefill tokens | Output tokens |
|---|---:|---:|---:|---:|
| Enabled | 24192 | 24192 | 24408 | 4 |
| Enabled | 23700 | 23040 | 25560 | 4 |
| Enabled | 24191 | 23040 | 25560 | 4 |
| Enabled | 0 | 0 | 48600 | 4 |
| Disabled (legacy path) | 24192 | 23040 | 25560 | 4 |

With state cache, cold prefill steps end at `7680 / 15360 / 23040 / 24192 / 24300`;
the completed request retains reusable state checkpoints at `23040` and `24192`.
The legacy path retains its unaligned batch budget: `8192 / 16384 / 24300`.

## Manual state-cache and partial prefix matching

The AISimulate engine owns state-cache allocation and scheduling. Mocker forwards
`state_cache` as a complete configuration object (currently `bytes_per_request`),
`prefix_match_unit`, physical KV sizing, fixed capacity, and batch-token budget.
Build the Python runtime on a standard Dynamo development host with the
[contribution-guide prerequisites](../../../../docs/fern/pages/community/contributing/overview.md),
including Rust and `uv`. Linux binding builds can also require CUDA/NIXL libraries
through the default block-manager feature; GPU-free simulation does not remove
those build dependencies. Then launch a live aggregated vLLM worker:

```bash
# Also applies to AISimulate's source build outside this checkout.
export RUSTUP_TOOLCHAIN=1.96.1
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install 'maturin[patchelf]'
maturin develop --uv --manifest-path lib/bindings/python/Cargo.toml --locked
uv pip install -e .

# Supply a local tokenizer/config directory or a Hugging Face model ID.
export MODEL_PATH=Qwen/Qwen3-0.6B
python3 -m dynamo.mocker \
  --model-path "$MODEL_PATH" \
  --engine-type vllm \
  --disaggregation-mode agg \
  --router-mode round-robin \
  --num-gpu-blocks-override 128 \
  --block-size 1536 \
  --prefix-match-unit 128 \
  --state-cache '{"bytes_per_request":24576}' \
  --kv-cache-bytes-per-token 16 \
  --max-num-batched-tokens 8192 \
  --no-enable-kv-events
```

`MODEL_PATH` supplies tokenizer/config files; the worker does not load model
weights. Start the Dynamo frontend with the same discovery and transport settings
and round-robin routing. Partial prefix matching does not publish Dynamo KV events
or maintain the local KV index; `--no-enable-kv-events` explicitly disables both.
The default remains enabled for existing configurations. An explicit
`--router-mode kv` is rejected when KV events are disabled.

These values are synthetic per-rank memory assumptions, not measured K3/TP8/DCP8
sizes. `block_size=1536` is the physical allocation granularity;
`prefix_match_unit=128` is the matching granularity. Each complete recurrent state
occupies 24576 bytes. The fixed shared pool has 128 physical blocks, or
`128 * 1536 * 16 = 3145728` bytes per rank, used for token blocks and state storage.
No implicit TP/DCP division or native DCP cache-group layout is modeled.

`--kv-transfer-bytes-per-token` controls disaggregated transfer timing, independently
of `--kv-cache-bytes-per-token`. The legacy `--kv-bytes-per-token` spelling remains a
transfer-only alias. When state cache is enabled, Mocker neither estimates transport
bytes from the model nor supplies a default transfer bandwidth. Explicit transfer
configuration is rejected by the underlying engine for this mode.

The supported scope is manual sizing, aggregated vLLM, and the G1 state cache.
Explicit `prefix_match_unit` rejects speculative decoding, KV-event export, and
Belady replay. Internal prefill checkpoints, prefix-cache-retention intervals,
shared-prefix junction retention, and asynchronous overlapped forwards are not
implemented. Keep KV-aware routing disabled for these partial-prefix tests.
