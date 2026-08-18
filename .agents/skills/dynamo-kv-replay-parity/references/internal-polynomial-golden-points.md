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
| vLLM disaggregated | 2 prefill + 2 decode; engine block 64; G1 blocks 5,500; max sequences 16; batch tokens 8,192; KV bytes/token 1; 100 GB/s full-prompt transfer | 14 fully readmitted preemptions; 10 fresh-process repetitions produced one digest and identical counters; nearby probes produced 11 at 5,000, 6 at 6,000, 16 at 4,500, and 24 at 4,000 |
| SGLang aggregated | 4 workers; engine/page block 512; G1 blocks 1,536; max sequences 256; batch tokens 32,768 | 1 retraction; 1,024 produced 8 and 2,048 produced 0 |
| SGLang disaggregated | 2 prefill + 2 decode; engine/page block 512; G1 blocks 11,264; max sequences 256; batch tokens 32,768; KV bytes/token 262,144; 100 GB/s full-prompt transfer | 14 fully readmitted retractions; 10 fresh-process repetitions produced one digest and identical counters; nearby probes produced 11 at 10,752, 7 at 11,776, 6 at 12,288, and 33 at 10,240 |

The configurations use the framework-local native G1 implementation. The removed
`--g1-backend` switch is not part of the current replay contract.

At the vLLM disaggregated seed, ten fresh processes completed all requests with exact
totals of 46,542,297 input and 922,544 output tokens, 14 fully readmitted preemptions,
5,000 immediate and zero queued placements in each pool, 4,988 requests with reuse, and
canonical report SHA-256
`dd693851bdc43cff67fe049ef77cf0f4d41e85c869a0bfd995a85d72147e6ec0` in every run.

At the SGLang disaggregated seed, ten fresh processes completed all requests with exact
totals of 46,542,297 input and 922,544 output tokens, 14 fully readmitted retractions,
5,000 immediate and zero queued placements in each pool, 4,990 requests with reuse, and
canonical report SHA-256
`c0488fcb82bb5c66b1c0d13b6b0b9c043fdadbf6f90ea171a763c38b70027afd` in every run.

The 5,000-row seeds are deliberately tight and are not universal full-trace capacities.
For a throttle soak against the complete 23,608-row Mooncake trace, SHA-256
`b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`, requalify a
separate capacity instead of copying these values automatically. A full-trace pressure
count may exceed the 10-to-20 target only when every event is bounded, readmitted, and
followed by exact completion without pathological replay-loop growth.

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
  --num-gpu-blocks 5500 \
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
  --num-gpu-blocks 11264 \
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
| vLLM disaggregated | 4,988 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |
| SGLang aggregated | 4,840 | decode workers 0–3 |
| SGLang disaggregated | 4,990 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |

The offline replay CLI defaults `router_queue_threshold` to unset, so these templates do
not exercise router queueing. At both qualified disaggregated seeds, every placement in
both pools was immediate and zero was queued. Require queued-placement evidence only when
an explicit queue-capable harness or forced fixture enables it; scheduler waiting is not
router queueing.

Every row must complete all 5,000 requests with no rejected, canceled, failed, or stranded
requests. A changed counter is not automatically a product failure, but it means the seed
must be requalified and the cause recorded before freezing the row.
