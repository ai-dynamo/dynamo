# Qwen3-VL CustomEncoder benchmark

This workload exercises Dynamo's cross-request custom vision batching with a
public Qwen3-VL-2B vision tower and the full Qwen3-VL-2B checkpoint in vLLM.

Start the topology:

```bash
examples/custom_encoder/launch/agg_qwen3_vl.sh
```

In another shell, generate and run the workload:

```bash
python examples/custom_encoder/benchmark/generate_workload.py
examples/custom_encoder/benchmark/run_aiperf.sh
```

The generator creates five 299×299 and four 500×500 deterministic JPEGs, then
cycles them across 100 requests. Every request contains one image and the same
text prompt. It pads that prompt so the estimated mean server-side ISL is near
515 after the variable number of visual tokens is spliced in. The benchmark
uses concurrency 8 and forces 70 generated tokens with `ignore_eos:true`.

Artifacts are written below `logs/qwen3_vl_custom_encoder/`. The command uses
server token counts because client-side tokenization cannot see the visual-token
expansion performed by the custom encoder. Confirm the actual mean ISL and OSL
from the aiperf export rather than relying only on the generator's estimate.

The Qwen3-VL custom encoder captures CUDA graphs for exact image-grid and padded
batch-rung pairs. Defaults cover this workload's 299×299 and 500×500 images with
batch rungs 1, 2, 4, and 8. Override before server startup when needed:

```bash
DYN_QWEN3_VL_GRAPH_IMAGE_SIZES="299x299,500x500" \
DYN_QWEN3_VL_GRAPH_BATCH_BUCKETS="1,2,4,8" \
    examples/custom_encoder/launch/agg_qwen3_vl.sh
```

For the eager A/B control, set
``DYN_QWEN3_VL_DISABLE_CUDA_GRAPHS=1``. The independent
``DYN_QWEN3_VL_MAX_BATCH_COST`` defaults to 8 in either mode.

Unsupported image grids fail during preprocessing rather than triggering lazy
capture or an unbudgeted eager fallback. The launcher explicitly retains
``--enable-prefix-caching``. It currently keeps vLLM's native tower loaded:
the installed Qwen3-VL ``--language-model-only`` path fails compilation when
DeepStack-related tensors remain on the meta device.

Choose the ladder from observed batch costs. A rung above the configured
``max_batch_cost`` is unreachable, while extra rungs reserve one graph per
image grid. For this concurrency-8 workload, 1, 2, 4, and 8 covered every
batch; dense 1-through-8 capture did not improve throughput enough to justify
its additional graph memory. Re-benchmark if the workload or concurrency
changes.

H2D inputs use reusable pinned staging buffers, and each batch returns through
one D2H copy before it is split into per-image CPU views. These are always
enabled. If image sources are stable and repeat exactly, preprocessing can also
be cached by source string:

```bash
DYN_QWEN3_VL_PREPROCESS_CACHE_SIZE=16 \
    examples/custom_encoder/launch/agg_qwen3_vl.sh
```

The bounded cache stores decoded/patchified CPU tensors, not image embeddings.
It is disabled by default because the same HTTP URL or local path can later
refer to different bytes; enable it only when source-string identity is a valid
freshness policy. Data URLs and immutable content-addressed URLs are natural
candidates. Set the capacity at least as high as the expected working set (nine
for this benchmark). Capacity bounds entries, not bytes: processed tensors and
large source strings still consume host memory. Raw keys remain resident until
eviction or shutdown, so do not use credential-bearing URLs as cache keys.
Cached tensors are shared read-only values; concurrent misses may compute the
same key more than once before one result is retained. Shutdown waits for active
preprocessing before clearing the cache and logs final hit/miss counts.

The per-image CPU tensors returned by `forward_batch` are read-only views into
one fresh batch allocation. This is safe for the current assembler, which copies
them into `prompt_embeds`; callers must not mutate them or assume disjoint base
storage.

The CustomEncoder runtime uses immediate eager-drain by default. Graph backends
can opt into an experimental bounded coalescing delay before server startup:

```bash
DYN_CUSTOM_ENCODER_QUEUE_WAIT_MS=1 \
    examples/custom_encoder/launch/agg_qwen3_vl.sh
```

Collection dispatches before the deadline when one compatible `bucket_key`
group's summed cost reaches `max_batch_cost`; incompatible shapes never count
toward each other's fullness. The delay starts when the actor dequeues the first
item, so it does not bound time already spent behind an in-flight forward. It is
a latency/throughput tradeoff and must be swept for the deployment's concurrency
and shape distribution; it is not assumed to improve throughput. Nonzero values
are rejected for backends without CUDA graph buckets.

For this workload, a matched 1,000-request H100 sweep favored the zero-delay
default at every tested concurrency: 4.095/7.813/14.257/24.315 req/s at
concurrency 1/2/4/8, versus 4.094/7.708/14.152/24.144 req/s with a 1 ms delay.

## Timing and saturation sweep

Start the topology with stage timing enabled and retain its append-only log:

```bash
mkdir -p logs/qwen3_vl_custom_encoder
DYN_CUSTOM_ENCODER_TIMING=1 \
    examples/custom_encoder/launch/agg_qwen3_vl.sh \
    2>&1 | tee logs/qwen3_vl_custom_encoder/server.log
```

In another shell, run both an encoder-sensitive OSL=1 workload and the target
OSL=70 workload across concurrency 1, 2, 4, 8, 16, and 32:

```bash
SERVER_LOG="$PWD/logs/qwen3_vl_custom_encoder/server.log" \
    examples/custom_encoder/benchmark/run_sweep.sh
```

Override the matrix without editing the script, for example:

```bash
CONCURRENCIES="4 8 16" OSLS="1 70" REQUEST_COUNT=100 \
SERVER_LOG="$PWD/logs/qwen3_vl_custom_encoder/server.log" \
    examples/custom_encoder/benchmark/run_sweep.sh
```

Each run gets its corresponding slice of the server log. At completion,
`summarize_results.py` writes `aiperf_summary.csv`,
`stage_timing_summary.csv`, and `cuda_graph_summary.csv` into the sweep
directory. The timing summary consumes log entries with this stable grammar:

```text
custom_encoder_timing stage=vit_forward elapsed_ms=1.23 batch_size=8 bucket=8 cost=8
custom_encoder_graph selected_bucket=8 actual_cost=7 batch_size=7
```

The stage summary covers `preprocess`, `queue_wait`, `h2d`, `vit_forward`, and
`d2h`; the graph summary shows which captured bucket was selected and for what
actual batch size/cost. `DYN_CUSTOM_ENCODER_TIMING` must be set on the server
process, not only on the aiperf client.

Qwen3-VL's native path also injects DeepStack vision features into intermediate
language-model layers. The current CustomEncoder interface carries only primary
image embeddings, so this workload measures the real ViT/projector and Dynamo
batching path but is not a native-output parity test.
