<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# EXPERIMENT: real SGLang page-size-1 KV-event cadence

Campaign scratch (e2e indexer contention). Measures which KV events real SGLang emits for
AgentX-shaped multi-turn traffic at page size 1, and compares the SGLang-mode mocker on the
same request schedule.

1. `build_workload.py --trace <AgentX traces.jsonl> --out workload-s8.jsonl --chains 240`:
   flattens sessions into chains (main thread and subagents) and scales each 64-token trace
   block to 8 tokens and each output length by 1/8, so token ratios and per-lane KV pressure
   match the full-scale trace while prompts fit a 40k-context model.
2. `run.sh RUN LANES DURATION_S MAX_REQ MAX_TOTAL_TOKENS [CHAIN_OFFSET]`: starts SGLang 0.5.21
   (container) with `--page-size 1`, the radix cache, chunked prefill 8192, and the ZMQ KV-event
   publisher; records every ZMQ message (`record_events.py`) while `drive.py` runs LANES
   closed-loop lanes of chains against `/generate` with deterministic prompt token ids.
3. `analyze_real.py --run-dir out/RUN --workload workload-s8.jsonl --warmup-s 120`: blocks per
   Stored/Removed event, stored blocks per request split exactly into prompt vs output blocks
   (page size 1 lets every block's token path be rebuilt through parent hashes), and event rates.
4. `to_mooncake.py --mode lanes --requests out/RUN/requests.jsonl ...` then
   `cargo run --release -p dynamo-mocker --example sglang_event_cadence -- <trace> 8 <kv> 1`:
   the same requests on the same number of closed-loop lanes through the SGLang-mode mocker.
