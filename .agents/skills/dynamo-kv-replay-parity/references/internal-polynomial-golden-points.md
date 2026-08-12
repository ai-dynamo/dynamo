# Internal-polynomial replay golden-point seeds

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Use these configurations as starting seeds for offline replay qualification, not as
universal capacities, parity results, or performance results. They were qualified against
the contiguous first 5,000 rows of the canonical Mooncake trace:

- slice rule: rows 0 through 4,999 in source arrival order;
- slice SHA-256:
  `3892ae19ae480b643155f0c6b9d798591cbe2e73bec6a0fa5ae3d3bc0332fb8a`;
- model: internal polynomial mocker, without AIC profile arguments;
- routing: KV-aware;
- arrival speedup: 4;
- trace block size: 512; and
- model and decode speedups: 1.

Requalify on the pinned baseline, then run the candidate with exactly the same
configuration. Never tune the revisions separately.

## Native G1 seeds

| Engine and topology | Starting configuration | Expected pressure and nearest boundaries |
| --- | --- | --- |
| vLLM aggregated | 4 workers; engine block 64; G1 blocks 6,144; max sequences 16; batch tokens 8,192 | 1 preemption; 4,096 produced 21 and 8,192 produced 0 |
| vLLM disaggregated | 2 prefill + 2 decode; engine block 64; G1 blocks 16,000; max sequences 16; batch tokens 8,192; KV bytes/token 1; 100 GB/s full-prompt transfer | 1 fully readmitted preemption; 10 fresh-process repetitions produced one digest and identical counters; 18,000 produced 0 |
| SGLang aggregated | 4 workers; engine/page block 512; G1 blocks 1,536; max sequences 256; batch tokens 32,768 | 1 retraction; 1,024 produced 8 and 2,048 produced 0 |
| SGLang disaggregated | 2 prefill + 2 decode; engine/page block 512; G1 blocks 17,408; max sequences 256; batch tokens 32,768; KV bytes/token 262,144; 100 GB/s full-prompt transfer | 2 retractions; 16,384 produced 12 and 18,432 produced 0 |

The vLLM configurations rely on native/default G1 selection. An experiment-only
`--g1-backend` switch is not required to reproduce the native seeds.

The vLLM disaggregated row was requalified at commit
`d95e98aa732c50909dd2f7777172c604066c8307` after PR #13052 made prefill replay exactly
one output token. The former 40,964-block seed now produces zero preemptions and no longer
exercises the pressure edge. The requalified 16,000-block seed completed all requests with
exact token totals, 5,000 immediate and zero queued placements in each pool, 4,991 requests
with reuse, and canonical report SHA-256
`c60365af08f856d939fd87c30e1b55049630a0e22cf1b371a5c9a5acee82a6cd` in all ten runs.

For a post-qualification throttle soak, run the same 16,000-block configuration against
the complete 23,608-row Mooncake trace, SHA-256
`b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`. Three fresh
processes each completed all requests with exact totals of 202,791,701 input and 4,299,817
output tokens, 23 fully readmitted preemptions, 23,608 immediate and zero queued
placements in each pool, and canonical report SHA-256
`e4bb70b88b25986beda2df602fb5b813e57393101c10caa9eace3429667b3eab`.
The one-to-three pressure target applies to the 5,000-row parity fixture; the full-trace
soak may exceed it when every pressure event is bounded, readmitted, and followed by exact
completion.

### CLI templates

Set the artifact and trace paths, then reuse the common load-generation arguments:

```bash
BIN=/path/to/offline_replay_bench
TRACE_5000=/path/to/mooncake_trace_rows_000000_004999.jsonl
COMMON_ARGS=(
  --router-mode kv-router
  --arrival-speedup-ratio 4
  --trace-block-size 512
  --speedup-ratio 1
  --decode-speedup-ratio 1
  --iterations 1
)
```

vLLM aggregated:

```bash
"$BIN" "$TRACE_5000" \
  --serving-mode aggregated \
  --num-workers 4 \
  --engine-type vllm \
  --block-size 64 \
  --num-gpu-blocks 6144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  "${COMMON_ARGS[@]}"
```

vLLM disaggregated:

```bash
"$BIN" "$TRACE_5000" \
  --serving-mode disagg \
  --num-prefill-workers 2 \
  --num-decode-workers 2 \
  --engine-type vllm \
  --block-size 64 \
  --num-gpu-blocks 16000 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --kv-bytes-per-token 1 \
  --kv-transfer-bandwidth 100 \
  --kv-transfer-timing-mode full-prompt \
  "${COMMON_ARGS[@]}"
```

SGLang aggregated:

```bash
"$BIN" "$TRACE_5000" \
  --serving-mode aggregated \
  --num-workers 4 \
  --engine-type sglang \
  --block-size 512 \
  --num-gpu-blocks 1536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  "${COMMON_ARGS[@]}"
```

SGLang disaggregated:

```bash
"$BIN" "$TRACE_5000" \
  --serving-mode disagg \
  --num-prefill-workers 2 \
  --num-decode-workers 2 \
  --engine-type sglang \
  --block-size 512 \
  --num-gpu-blocks 17408 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --kv-bytes-per-token 262144 \
  --kv-transfer-bandwidth 100 \
  --kv-transfer-timing-mode full-prompt \
  "${COMMON_ARGS[@]}"
```

## Expected behavior

With the exact corpus and internal model above, use these observed values as drift
detectors:

| Engine and topology | Requests with reuse | Worker and handoff evidence |
| --- | --- | --- |
| vLLM aggregated | 4,918 | decode workers 0–3 |
| vLLM disaggregated | 4,991 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |
| SGLang aggregated | 4,840 | decode workers 0–3 |
| SGLang disaggregated | 4,992 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |

The offline replay CLI defaults `router_queue_threshold` to unset, so these templates do
not exercise router queueing. In the requalified vLLM disaggregated runs, every placement
in both pools was immediate and zero was queued. Require queued-placement evidence only
when an explicit queue-capable harness or forced fixture enables it; scheduler waiting is
not router queueing.

Every row must complete all 5,000 requests with no rejected, canceled, failed, or stranded
requests. A changed counter is not automatically a product failure, but it means the seed
must be requalified and the cause recorded before freezing the row.
