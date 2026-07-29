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
| vLLM disaggregated | 2 prefill + 2 decode; engine block 64; G1 blocks 40,964; max sequences 16; batch tokens 8,192; KV bytes/token 1; 100 GB/s full-prompt transfer | 4 preemptions; exact next integer capacity 40,965 produced 0. This is a documented nearest-feasible exception to the 1–3 target: max sequences 15 produced 9–13 near the edge and max sequences 17 produced 0 |
| SGLang aggregated | 4 workers; engine/page block 512; G1 blocks 1,536; max sequences 256; batch tokens 32,768 | 1 retraction; 1,024 produced 8 and 2,048 produced 0 |
| SGLang disaggregated | 2 prefill + 2 decode; engine/page block 512; G1 blocks 17,408; max sequences 256; batch tokens 32,768; KV bytes/token 262,144; 100 GB/s full-prompt transfer | 2 retractions; 16,384 produced 12 and 18,432 produced 0 |

The vLLM configurations rely on native/default G1 selection. An experiment-only
`--g1-backend` switch is not required to reproduce the native seeds.

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
  --num-gpu-blocks 40964 \
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

| Engine and topology | Immediate / queued | Requests with reuse | Worker and handoff evidence |
| --- | --- | --- | --- |
| vLLM aggregated | 8 / 4,992 | 4,918 | decode workers 0–3 |
| vLLM disaggregated | 4 / 4,996 | 4,991 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |
| SGLang aggregated | 5 / 4,995 | 4,840 | decode workers 0–3 |
| SGLang disaggregated | 2 / 4,998 | 4,992 | prefill and decode workers 0–1; 5,000 complete, backend-valid handoffs |

Every row must complete all 5,000 requests with no rejected, canceled, failed, or stranded
requests. A changed counter is not automatically a product failure, but it means the seed
must be requalified and the cause recorded before freezing the row.

## Derive and sanity-check KVBM rows

Derive each vLLM KVBM row from its corresponding native seed by enabling G2 and modestly
reducing or limiting G1. Keep the topology fixed; start disaggregated qualification at
exactly 2 prefill + 2 decode workers. Tune capacity or concurrency identically for both
revisions.

Require all of the following before freezing a KVBM row:

- all 5,000 requests complete;
- one to three bounded preemptions, without repeated preempt/re-admit cycling;
- nonzero G1-to-G2 eviction completion;
- nonzero G2-to-G1 restoration hits;
- identical lifecycle counts across repeated baseline and candidate runs;
- 5,000 complete, backend-valid handoffs for disaggregated replay; and
- one canonical digest per revision, with baseline and candidate digests matching.

Do not infer offload coverage from successful completion. Do not proceed to performance
when lifecycle counters match but a revision's canonical repetitions differ; that is an
internal determinism failure. SGLang KVBM offload is unsupported by the current harness
and must be reported as `UNSUPPORTED`, not simulated by toggling an ignored G1 option.
