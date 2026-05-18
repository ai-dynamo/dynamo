# Dynamo + LLaDA 2.0 (SGLang diffusion-LM): Where can KV routing and disaggregation actually help?

Author: deep-research session, 2026-05-11
Setup: `inclusionAI/LLaDA2.0-mini-preview` on SGLang 0.5.11, Dynamo `main`, dual RTX PRO 6000 (Blackwell).

> **⚠️ ERRATUM (2026-05-12 evening)**: The "ChunkCache structural blocker" thread that runs through this document is **wrong for SGLang 0.5.11**. The disable-radix-cache override for DLLM was removed in upstream before 0.5.11. The radix cache IS enabled by default for DLLM, IS emitting events when `--kv-events-config` is set, and contributes **+24% throughput on chat workloads, +149% throughput on RAG workloads** vs forcing `--disable-radix-cache`. Empirically verified with controlled benchmarks; see `aa-working-notes/2026-05-12-radix-cache-correction.md`.
>
> What's still correct in this doc:
> - The mental model of how Dynamo's KV router works (the cost function, event-driven vs approximate sub-modes, the role of `kv_block_size`).
> - The reason KV-aware routing modes don't beat round-robin at our 2-worker fleet — but for a **different reason than this doc claims**. The real reason is that with the cache active and only 4 prefixes × 2 workers, RR already achieves ~96% cache hit rate. "Perfect" routing would reach 98% — a 2-point gain swamped by the load-imbalance cost of pinning.
> - The compute-distribution analysis (prompt-encode is small fraction at OSL=64; larger at OSL=32 with long prompts).
>
> What's wrong:
> - Every claim that the radix cache is disabled, that `ChunkCache` is the active backend, that no `BlockStored` events flow, that engine-side caching is "out of scope" because it would need an upstream patch. None of these hold for SGLang 0.5.11.

## Scope constraint (Dynamo-only)

**No SGLang-side changes.** All recommendations must work against unmodified upstream SGLang 0.5.11 with `--dllm-algorithm` set. This rules out re-enabling the radix cache, hooking the scheduler to emit KV events, adding prompt-KV capture/replay hooks, KV quantization, and any change inside `process_batch_result_dllm`. Those remain valid future work — see the **Out of scope (future work)** section at the end of this doc — but are not in scope for any of the ranked recommendations below.

Net effect of the constraint: realistic per-request latency wins from Dynamo-side intervention are **bounded to single-digit percent**. The Dynamo-only wins are about *placement, capacity planning, and observability* — not engine-side KV reuse.

## TL;DR

Dynamo's router has **two KV-aware modes** that differ in where cache-state knowledge comes from:

- **Event-driven** (default, `--router-mode kv`): the indexer's prefix tree is filled from `BlockStored` / `BlockRemoved` events emitted by the engine. Requires the engine to actually publish those events.
- **Approximate** (`--router-mode kv --no-router-kv-events`): the indexer is filled from the router's *own* routing decisions, with TTL-based expiration. **Does not require any engine events.** (`lib/llm/src/kv_router.rs:357-366`, `lib/llm/src/kv_router/push_router.rs:541-570`.)

The earlier sections of this report focused on the event-driven mode. For LLaDA 2.0 today:

1. **Event-driven KV routing is broken for LLaDA**, because SGLang force-disables the radix cache when `--dllm-algorithm` is set (`server_args.py:2541-2545`) and the resulting `ChunkCache` emits zero events (`chunk_cache.py:25-87`). The prefix tree never gets populated from events.
2. **Approximate KV routing actually works for LLaDA today** — it never reads from `ChunkCache` or any engine cache. The router writes synthetic stored events on every successful routing decision (`push_router.rs:544-559`) and matches future requests against those. The only LLaDA-specific gotcha is `kv_block_size`: it flows from `server_args.page_size` through the MDC (`components/src/dynamo/sglang/register.py:64`, `lib/llm/src/discovery/watcher.rs:719-737`) and defaults to **1** for the diffusion path, so the router hashes per-token, not per 32-token semantic block. Adding `--page-size 32` to the launch script fixes the granularity.
3. The publisher's KV-event subscriber (`publisher.py:194-275`) is irrelevant in approximate mode; the indexer's `start_subscriber` is *not* started when `use_kv_events=false` (`kv_router.rs:357`).
4. The LLaDA mask predictor uses **bidirectional / encoder-only attention** (`models/llada2.py:490`, `attn_type=ENCODER_ONLY`). The original LLaDA paper says "LLaDA is incompatible with KV caching", but LLaDA 2.0's block diffusion does recover *clean-prefix* reuse on a single worker. That's exactly the surface approximate routing can exploit, **if the engine actually keeps the clean prefix's KV** — which `ChunkCache` does not across requests. So even approximate routing can only deliver wins when SGLang actually caches the prompt, which it does **not** today. See Thread 4 / Performance plan for the resulting honest upper bound.
5. SGLang's diffusion path has no prefill/decode split — `init_llm_diffusion()` ignores `DisaggregationMode.PREFILL` (`main.py:115-122`), there is no `init_dllm_prefill()`. **P/D disaggregation is not applicable** to LLaDA in the current stack.

The router still provides value as a **load balancer** (decode-block tracking) and **request gate** (queue / sequence tracking, independent of KV events). SLA-based autoscaling is a real win because LLaDA's per-request cost is unusually predictable. **Approximate routing is the highest-leverage Dynamo-only change**: it costs ~one flag flip plus `--page-size 32`, and yields cache-affinity routing *without touching SGLang*. The cache-hit benefit itself, though, is bounded by SGLang's lack of cross-request prompt caching — see the benchmark plan.

---

## Thread 1 — LLaDA 2.0 architecture facts

### Attention mask (bidirectional, with block-causal extension in LLaDA 2.0)

- **Original LLaDA (Nie et al., 2502.09992):** the mask predictor uses bidirectional attention. Quoted from the HTML version of the paper:
  > "LLaDA employs a Transformer as the mask predictor … LLaDA does not use a causal mask, as its formulation allows it to see the entire input for predictions."
  > "We use vanilla multi-head attention instead of grouped query attention for simplicity, as **LLaDA is incompatible with KV caching**" (Sec. 2 / Section discussing architecture).
  ([arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992))
- **LLaDA 2.0 (Ant Group, 2512.15745):** introduces a 3-phase WSD training scheme that includes a **block diffusion** phase with a structured attention mask combining (a) block-diagonal within the noisy block, (b) cross-attention from noisy to clean previous blocks, and (c) causal-block attention between clean blocks. In effect, "clean" blocks (already-decoded prior content) behave causally, while attention within the active block is bidirectional. The paper states:
  > "Blocks are produced auto-regressively … supporting KV-cache reuse during decoding, enhancing inference efficiency."
  ([arxiv.org/abs/2512.15745](https://arxiv.org/abs/2512.15745))
- **SGLang's implementation** of LLaDA2 sets `attn_type=AttentionType.ENCODER_ONLY` for every attention layer (`/home/ayush-lab/Work/sglang/python/sglang/srt/models/llada2.py:484-492`). The block-causal structure is *not* enforced inside the attention op itself; instead it is realized at the scheduling layer by feeding ragged "prefix_lens = seq_lens − block_size" into the FlashInfer prefill wrapper (`flashinfer_backend.py:645-672`). In other words, the model attends bidirectionally over the entire fed-in sequence, but the scheduler decides which tokens get committed to the KV pool each iteration. This is a *materialisation* of the block-causal structure rather than a masked attention op.

### Generation algorithm (LowConfidence)

`python/sglang/srt/dllm/algorithm/low_confidence.py` — the only currently shipped algorithm.

- **Block size**: hard-coded to 32 tokens for `LLaDA2MoeModelLM` (`dllm/config.py:34-41`).
- **Loop structure** (`low_confidence.py:23-101`):
  1. Append a fresh 32-mask-token block to the sequence (`schedule_batch.py:844-853`).
  2. Run up to `block_size` (=32) forward passes. Each pass:
     - Computes logits over the entire fed-in sequence (prompt + already-committed blocks + current masked block).
     - Greedily decodes per-position, computes softmax confidence.
     - Unmasks any positions whose confidence > `threshold` (default 0.95) — they are committed into `block_input_ids`.
     - If no token clears the threshold, force-commits the top-1.
  3. Exits early when no mask tokens remain in any block.
  4. One final forward pass with all positions committed → "save kv cache" path (lines 35-40).
- **Worst-case forward passes per output block**: 32 + 1 (final commit). Best case: 2 if greedy crosses the threshold immediately. Typical: 5-10 with threshold=0.95 per inclusionAI defaults.
- **`max_new_tokens` / generation length** therefore translates to ceil(N/32) outer block iterations, each with ~5-32 inner forward passes.

### KV-cache semantics

- During the *inner refinement loop* for the current block, the KV cache for that block is **invalidated every step** because the input tokens for those positions change between passes. The loop re-runs the full prefill over the active block each iteration. This is why SGLang uses `forward_mode=DLLM_EXTEND` with `prefix_lens = seq_lens − block_size` (`flashinfer_backend.py:666`) — the previous `seq_lens − block_size` tokens are "clean" and their KV is kept; the trailing 32 tokens are re-attended each iteration.
- Once a block is fully decoded, it becomes a "clean" prefix for subsequent blocks. Block diffusion training was specifically introduced (Section 3 of LLaDA2.0 paper) to make these clean prefixes' KV cache stable enough to reuse across block boundaries — i.e., **intra-sequence KV reuse**.
- **Cross-request prefix caching** (what Dynamo's router wants) requires that two different requests sharing the same prompt would produce **identical KV** if routed to the same worker. This is theoretically possible for LLaDA — the prompt portion gets a deterministic, position-stable bidirectional encoding — but it requires:
  1. The serving stack to actually populate a prefix tree / radix cache with the prompt blocks.
  2. The block hash to be over fixed-position semantic blocks (i.e., page_size aligned to LLaDA's 32-token block boundary).
- Neither condition holds today (see Thread 2).

### Sources

- `inclusionAI/LLaDA2.0-mini-preview` HF card: 16B total params (1.4B activated), 20 layers, 16 attention heads, 4096 context, RoPE, MoE. Defaults `block_length=32, steps=32, temperature=0` ([huggingface.co/inclusionAI/LLaDA2.0-mini-preview](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview)).
- LLaDA 2.0 paper: 100B flash variant, 535 tok/s with "Confidence-Aware Parallel" inference ([arxiv.org/abs/2512.15745](https://arxiv.org/abs/2512.15745)).
- LLaDA original: ([arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)).

---

## Thread 2 — SGLang's diffusion-LM KV event path (mostly broken)

### KV events in the standard autoregressive path

`mem_cache/radix_cache.py` is the producer:
- `_record_store_event` (lines 768-803) and `_record_remove_event` (lines 805-822) push `BlockStored` / `BlockRemoved` onto `self.kv_event_queue` whenever a node is inserted or evicted from the radix tree. Events are gated on `self.enable_kv_cache_events`.
- `take_events()` drains the queue (line 828-838).
- `scheduler_metrics_mixin.py:501-508`: `_publish_kv_events` reads the queue and sends a `KVEventBatch` over the configured `EventPublisher` (typically `ZmqEventPublisher`). Called from `log_prefill_stats` (line 276) and `log_decode_stats` (line 451) — i.e., once per scheduler step that touches prefill or decode.
- Publisher creation: `scheduler_metrics_mixin.py:128-132`, gated on `enable_kv_cache_events` which is `bool(server_args.kv_events_config and attn_tp_rank == 0)` (`scheduler.py:322-324`).

### Diffusion path → ChunkCache → no events

This is the load-bearing finding:

`server_args.py:_handle_dllm_inference()` (lines 2510-2550) forcibly sets:
```
if not self.disable_radix_cache:
    logger.warning("Radix cache is disabled because of using diffusion LLM inference")
    self.disable_radix_cache = True
```
plus `disable_overlap_schedule = True`, `pp_size = 1`, `cuda_graph_bs = [1]`, attention backend = `flashinfer` (CUDA path).

With `disable_radix_cache=True` and `chunked_prefill_size` non-None (it's auto-set to 4096-16384 based on GPU memory in `_handle_memory_settings()`), `scheduler.py:init_cache_with_memory_pool` (lines 642-653) selects `ChunkCache` instead of `RadixCache`. `ChunkCache` (`mem_cache/chunk_cache.py`):
- `match_prefix` returns an empty `MatchResult` always (line 50-55).
- `cache_finished_req` simply frees the KV slot (line 57-64) — **no insertion into a tree, no `_record_store_event`** (the method doesn't exist).
- `cache_unfinished_req` copies kv_indices into `req.prefix_indices` and exits (line 66-71).
- No `take_events()` method exists; the scheduler's `_publish_kv_events` calls `self.tree_cache.take_events()` which would `AttributeError` if `enable_kv_cache_events` were True — but `enable_kv_cache_events` is only True when `kv_events_config` is set, and even then no events would be produced because `ChunkCache` never calls `_record_*_event`.

**Net effect**: Even with `--kv-events-config` enabled (which the Dynamo launch does *not* do for `diffusion_llada.sh`), the LLaDA worker emits zero `BlockStored` events. The KvIndexer's prefix tree for this worker stays empty forever.

### Does the diffusion path otherwise touch the cache?

Yes — `process_batch_result_dllm` (`scheduler_output_processor_mixin.py:327-357`) calls `release_kv_cache` (which calls `cache_finished_req`) and `cache_unfinished_req` per block. These calls hit `ChunkCache`'s no-op insert. So KV memory is correctly accounted for, but nothing is indexed for routing.

### Dynamo's SGLang publisher (still wires up, but unused)

`components/src/dynamo/sglang/init_diffusion.py:62-64`: `init_llm_diffusion` does call `setup_sgl_metrics(engine, config, generate_endpoint)`, identical to `init_decode`. That function calls `publisher.init_kv_event_publish()` (`publisher.py:194-275`), which:
- Only creates a `KvEventPublisher` if `server_args.kv_events_config` is set (line 212).
- Subscribes to the SGLang ZMQ endpoint with `kv_block_size=self.server_args.page_size` (line 264).

So the wiring is *present*. Three problems compound:

1. **`kv_events_config` is unset** by default in `diffusion_llada.sh`. Even if you set it, …
2. **page_size defaults to 1** (`sglang/srt/server_args.py:1910-1912`) for the triton attention backend used by the launch script. `compute_block_hash_for_seq` hashes per-token, giving a 1-token-per-block prefix tree that is useless for sharing.
3. **No producer exists** in `ChunkCache` even if `kv_events_config` were set.

### What page_size=32 *would* mean

If a user set `--page-size 32` to match LLaDA's block size:
- The KV physical pool would be paginated in 32-token chunks (good alignment with LLaDA's block-causal structure).
- The Dynamo router's block hashes would be on 32-token windows of the prompt, which is plausible for prefix sharing.
- But still no events would flow, because `ChunkCache` doesn't emit them and the radix cache is forcibly disabled.

### What would it take to fix?

Three options, in order of effort:

1. **Cheapest**: re-enable radix cache for LLaDA. The disabling at `server_args.py:2541-2545` looks defensive — the comment is bare. Block-diffusion *should* be compatible with radix caching for the prompt portion (which is fixed-position bidirectional content). Investigating whether SGLang's `cache_unfinished_req` path corrupts state for `DLLM_EXTEND` forward mode is non-trivial — the masked-block tokens change every diffusion step, so they shouldn't be inserted into the radix tree until the block finishes. This is a real correctness question, not a no-op.
2. **Targeted**: add a `DllmRadixCache` (or extend `RadixCache`) that only inserts blocks once they're fully committed (transition from masked → clean). The `process_batch_result_dllm` path (`scheduler_output_processor_mixin.py:354`) already calls `cache_unfinished_req` after each block, which is the natural seam.
3. **Heavy**: implement direct publishing from the DLLM scheduler. After each block transitions to clean, push a `BlockStored` event using `dynamo.llm.KvEventPublisher.publish_stored` (the API documented in `docs/integrations/kv-events-custom-engines.md:47-109`). This bypasses SGLang's radix cache entirely and lets us tune block hash granularity to LLaDA's semantic 32-token block.

---

## Thread 3 — Dynamo's KV-aware router

### Signal

`docs/components/router/router-concepts.md:37-45` and `lib/kv-router/src/scheduling/` define the cost model:

```
cost = overlap_score_weight * prefill_blocks + decode_blocks
```

- `prefill_blocks` = remaining prompt tokens / kv_block_size, *after subtracting* the longest cached prefix on each worker. The router's prefix tree is the source of truth for "longest cached prefix".
- `decode_blocks` = an estimate of active decode load on the worker, updated as requests progress and complete.
- `overlap_score_weight` (default 1.0) trades cache-hit pursuit vs load smoothing.

### Events drive the prefix tree

`docs/integrations/kv-events-custom-engines.md`:
- `BlockStored` adds blocks to the per-worker prefix tree.
- `BlockRemoved` evicts them.
- The `KvIndexer` (Rust, `lib/kv-router/src/indexer/`) maintains a concurrent radix tree keyed by `LocalBlockHash` (XXH3 of the block's tokens) and rolling `SequenceHash` (chaining the parent), built by `compute_block_hash_for_seq` / `compute_seq_hash_for_block` in `lib/kv-router/src/protocols.rs:70-173`.

### Granularity

Block-granular, not token-granular. The router hashes **fixed-size windows** of the prompt (size = `kv_block_size` configured at publisher init time) and looks them up in each worker's prefix tree. This means routing only works when:

1. The publisher's `kv_block_size` matches the engine's actual cache block size.
2. The router (`--router-block-size` on the standalone router, or its default) matches.
3. Mismatch → hashes don't line up → zero overlap reported for everyone → degenerates to load-only routing. Per the docs:
   > "`kv_block_size` must match your engine's actual block size." (`kv-events-custom-engines.md:285`)

### Does the router care about attention pattern?

**No.** The router treats tokens as opaque and only cares that:
- Same prompt prefix → same block hash.
- The worker that previously processed those tokens has a `BlockStored` event in its tree.

This is *exactly* the property LLaDA 2.0's block-causal masking preserves for the prompt + clean-block portion of the sequence: bidirectional attention over a fixed prefix produces the same KV when fed the same tokens at the same positions. So in principle, a properly-wired LLaDA worker could participate in cross-request prefix sharing.

The mismatch isn't conceptual; it's just that SGLang doesn't currently populate the index.

---

---

## Routing modes: KV-aware vs approximate

Dynamo exposes six `--router-mode` values (`lib/runtime/src/pipeline/network/egress/push_router.rs:158-170`): `round_robin`, `random`, `power_of_two_choices`, `least_loaded`, `device_aware_weighted`, and `kv`. Only `kv` consults the cache-state indexer. Inside `kv`, there are **two sub-modes** controlled by `KvRouterConfig.use_kv_events` (default `true`; flipped by the CLI flag `--no-router-kv-events`, see `docs/components/router/router-configuration.md:27`).

### Mode 1 — Event-driven KV routing (`--router-mode kv`, default)

1. **Decision**: same cost function as documented in Thread 3 (`router-concepts.md:37-45`): `cost = overlap_score_weight * prefill_blocks + decode_blocks`. `prefill_blocks` is reduced by the longest cached-prefix match found in the per-worker prefix tree.
2. **State**: a concurrent radix tree keyed by `LocalBlockHash` (XXH3 of fixed-size token windows) and rolling `SequenceHash` (`lib/kv-router/src/protocols.rs:70-173`). One indexer instance covers all workers; per-worker membership is stored at tree-node level.
3. **Writes come from**: engine-emitted `BlockStored` / `BlockRemoved` events, received via the event-plane subscriber (`lib/llm/src/kv_router.rs:357-359`, started only when `should_subscribe_to_kv_events()` is true, i.e. `use_kv_events && overlap_score_weight > 0` — `lib/kv-router/src/scheduling/config.rs:420-422`).
4. **Reads** via `find_matches_for_request(tokens, lora_name, is_eagle)` (`lib/kv-router/src/indexer/thread_pool.rs:527-543`).
5. **Worker-side dependencies**: the engine MUST publish `BlockStored`/`BlockRemoved`. Block hashes MUST be computed on identical fixed-size windows on both sides — the router uses `kv_block_size` from the MDC (`card.kv_cache_block_size`), which for SGLang is just `server_args.page_size` (`components/src/dynamo/sglang/register.py:64`). Mismatch → no overlaps.
6. **Failure modes**: silent degradation to load-only when (a) no events flow, (b) `page_size` ≠ engine cache block, (c) engine evicts but doesn't emit `BlockRemoved` (router over-counts hits).

### Mode 2 — Approximate KV routing (`--router-mode kv --no-router-kv-events`)

1. **Decision**: identical cost function. The same indexer / scheduler / scoring code paths run.
2. **State**: same radix tree, but additionally a `PruneManager` with a TTL (`--router-ttl-secs`, default 120s — `lib/kv-router/src/indexer/pruning.rs:53-64`, `kv_router/indexer/mod.rs:82-114`). Entries auto-expire so the tree doesn't grow unboundedly without engine eviction events.
3. **Writes come from**: the **router's own routing decisions**. Every non-query-only request, after worker selection, calls `chooser.record_routing_decision(tokens_with_hashes, worker)` (`lib/llm/src/kv_router/push_router.rs:544-570`), which writes a synthetic stored-event into the indexer (`lib/kv-router/src/indexer/thread_pool.rs:438-475`). The router is assuming "if I just sent these tokens to worker W, then W now has the corresponding blocks cached for a while".
4. **Reads**: same `find_matches_for_request` path. The reader cannot tell whether an entry came from a real event or a synthetic one.
5. **Worker-side dependencies**: **none related to events**. The router still needs the worker's `KvMetrics` (decode-block load) over the standard metrics ZMQ socket to compute the `decode_blocks` term — but `KvMetrics` is emitted by SGLang in every batch result regardless of `dllm_algorithm` (consumed by `publisher.py:118-158`). The `kv_block_size` still has to match the engine's actual cache window for the cached *KV* to be reusable; if the engine's cache is finer/coarser than `page_size`, the router can still route correctly but reuse won't materialise.
6. **Failure modes**:
   - **Stale entries**: TTL expires before the request actually completes — request gets unbatched, less likely to be a cache hit on a re-route.
   - **Eviction blind spot**: if the engine evicts a block before TTL, the router thinks the worker still has it and routes anyway. For LLaDA this is *unusually benign* because the cache state being approximated is "this worker just touched these tokens, send the next-turn request there too" — i.e., locality, not strict cache contents.
   - **Block-size mismatch**: the router will still write coherent synthetic events at its configured `kv_block_size`, but matches will be on 1-token windows if `page_size=1`. Hashes are 32-bit-ish-collision-tolerant XXH3, so 1-token blocks aren't *wrong*, just much smaller than ideal — many small matches collide and overlap scores degrade to "this request and that request share any tokens at all".

### Other modes (no cache state)

- `round_robin`, `random`: stateless, fair.
- `least_loaded`, `power_of_two_choices`: use in-flight count from `RoutingOccupancyState` (no KV awareness; `push_router.rs:182-204`).
- `device_aware_weighted`: ratio-normalized least-loaded across CPU vs non-CPU groups. Irrelevant for a homogeneous LLaDA fleet.
- `direct`: explicit worker pin; used by P/D and sticky sessions, not a baseline.

### Sticky sessions

Activated automatically when requests carry `nvext.session_control` (`docs/components/router/router-configuration.md:50-55`). Pins `session_id -> worker_id` with sliding TTL. **Independent of KV events**, so it works for LLaDA today provided the SGLang backend has `--enable-streaming-session`. Different from approximate routing: sticky is per-session, approximate is per-token-pattern.

### LLaDA-specific implications

For LLaDA + SGLang today:
- Event-driven: dead (Thread 2). Empty prefix tree, `find_matches` returns nothing useful, scoring collapses to `decode_blocks` only.
- Approximate: **live**. The router will record every request's tokens against the worker that just served it. A second request with the same (or prefix-overlapping) tokens within `--router-ttl-secs` will preferentially route to that worker. **This is the affinity signal that survives `disable_radix_cache=True`.**
- The catch: even when the router perfectly co-routes "same prompt → same worker", the SGLang worker has no cross-request KV cache to reuse (it's running `ChunkCache`, which frees on completion — `chunk_cache.py:57-64`). The router will *prefer* a worker that *should* have hot weights/KV but doesn't.
- Useful real benefits in this regime: scheduler-level locality (active sequences, in-flight memory pressure), better TCP/HTTP keepalive reuse, possibly some CPU-side cache locality for tokenizer / preprocess work in the worker process. These are measurable but small.
- Full prefix-cache wins require the SGLang-side fix from Thread 2 (re-enable a controlled radix cache for DLLM). Approximate routing is the *router-side* half of the fix; the SGLang change is the other half.

---

## Thread 4 — Synthesis: where does Dynamo bring real value to LLaDA 2.0?

### Candidate 1: Cross-request prefix caching via KV-aware routing

**Theoretical value**: Real and high for chat/agent workloads where a system prompt and conversation history are shared. LLaDA's bidirectional attention does **not** break this — the prompt portion's KV is positionally deterministic.

**Today's state**: **Effectively zero**. The chain is broken at SGLang: `--dllm-algorithm` forces `disable_radix_cache=True` → `ChunkCache` → no `BlockStored` events → empty prefix tree → router defaults to load-only.

**What's missing/broken (file:line)**:
- `sglang/python/sglang/srt/server_args.py:2541-2545` — forced `disable_radix_cache=True` for DLLM. This is the single biggest blocker.
- `sglang/python/sglang/srt/mem_cache/chunk_cache.py:25-87` — no event emission. Even if radix were re-enabled, the diffusion path needs to be audited to avoid inserting half-decoded blocks.
- `dynamo/sglang/publisher.py:264` — `kv_block_size=self.server_args.page_size` will use 1 unless the launch script sets `--page-size 32`. The launch script `examples/backends/sglang/launch/diffusion_llada.sh:59-69` does not set it.
- `diffusion_llada.sh` also doesn't pass `--kv-events-config` to SGLang, so even if a tree existed there's no transport.

**Fix sketch**:
1. Add `--page-size 32` and a `--kv-events-config '{"endpoint": "tcp://*:5557"}'` to `diffusion_llada.sh`.
2. In SGLang, gate the `disable_radix_cache=True` override on a config flag (`--dllm-disable-radix-cache`, default True for safety) and audit `cache_unfinished_req` for the dllm path so that masked-block kv-indices are not inserted into the tree until the block is committed (the seam is the per-block iteration in `process_batch_result_dllm`).
3. Verify via the existing `kv-router-ab-testing` harness (`docs/benchmarks/kv-router-ab-testing.md`) that cache hits actually improve TTFT.

### Candidate 2: Disaggregated prefill/decode

**Theoretical value**: For autoregressive LLMs, P/D split lets prefill workers absorb bursts of long prompts while decode workers steady-state TPS. For LLaDA, "prefill" *is* still a thing — the prompt has to be encoded once and cached — but "decode" is a different beast: it's a sequence of `DLLM_EXTEND` forward passes, each at the full prompt-plus-current-block length, not the cheap 1-token decode of autoregressive models.

**Today's state**: **Doesn't apply**. SGLang's diffusion path has no prefill/decode boundary. `init_llm_diffusion` is the only DLLM init function — there is no `init_dllm_prefill` (`components/src/dynamo/sglang/main.py:115-122` shows the single dispatch branch). `_handle_dllm_inference` doesn't branch on `disaggregation_mode`. No bootstrap_room handshake on the DLLM path.

**Verdict**: P/D as currently implemented in Dynamo is a poor fit. A LLaDA-specific disaggregation would have a different structure:
- **Stage 1 (prompt encoding)**: one bidirectional forward pass over the prompt, producing the "clean" KV cache.
- **Stage 2 (iterative block refinement)**: repeated DLLM_EXTEND passes.

These two stages have the same KV requirements (same model, same cache layout). There's no efficiency win from splitting them across workers unless the prompt encoding can be parallelised more aggressively (TP) than the refinement loop. That's a model-parallelism question, not P/D.

**Could become real if**: someone implements LLaDA-specific stage splitting where prompt-encoding workers transfer encoded KV via NIXL to refinement workers. Cost/benefit is uncertain because the prompt forward pass is a small fraction of total work (refinement runs the model O(blocks × steps_per_block) times, with `steps_per_block ≈ 5-10` typical).

### Candidate 3: Request cancellation

**Value**: Modest. `DiffusionWorkerHandler.generate` inherits cancellation via `_process_token_stream` (`decode_handler.py:562-598`), which calls `context.is_stopped()` between yielded chunks. For LLaDA, a "chunk" is an entire 32-token block — so cancellation latency is one full block's worth of forward passes (potentially 5-32 × ~50ms = 250ms-1.6s).

**Today's state**: Works, but coarse. The cancellation is honoured at block boundaries, not at diffusion-step boundaries inside `LowConfidence.run()` (which is a tight torch loop — see `low_confidence.py:51-91`). Mid-block cancellation would require modifying `low_confidence.py` to accept a cancellation token.

**Verdict**: Existing cancellation is adequate for most workloads. Not a high-leverage area for Dynamo to invest.

### Candidate 4: SLA-based planning / autoscaling

**Value**: **Real and underappreciated.** LLaDA's per-request cost is unusually predictable:
- No early termination from EOS during iterative refinement (the model commits the whole block).
- `max_new_tokens` directly translates to ceil(N/32) blocks, each at predictable cost.
- `LowConfidence` may exit early *within* a block when all tokens cross threshold, but the worst-case ceiling is `block_size` forward passes — bounded and uniform.

Compare to AR LLMs where output length is highly variable. LLaDA workloads are easier to plan capacity for. The planner's autoscaler (in `docs/components/planner/`) can use the much tighter cost distribution to:
- Predict TTLT (time-to-last-token) accurately from `max_new_tokens` alone.
- Scale replicas based on expected work, not just observed queue depth.

**Today's state**: The planner does not have a LLaDA-specific cost model. It uses generic ITL/TTFT estimates derived from KV metrics. The current scheduler-metrics path *does* work for LLaDA (`publisher.py:118-158` consumes `KvMetrics` over ZMQ — non-event metrics, just cache usage and total blocks), so the planner does see basic occupancy.

**Verdict**: Genuine win. Building a LLaDA cost model into the planner is concrete, deterministic, and high-impact.

### Candidate 5: Multi-modal batch routing

LLaDA 2.0-mini-preview is text only. Skip.

### Candidate 6: Hierarchical KV cache (KVBM / hicache)

**Theoretical value**: For autoregressive models, hicache lets you offload cold KV to CPU/disk and reload on a hit. The hit rate is the lever.

**For LLaDA**: With `disable_radix_cache=True`, the cache is freed on request completion. There is no warm-cache surface for hicache to wrap. Even if radix were re-enabled, the natural cache footprint per request is `prompt_len + max_new_tokens` worth of KV — which is what AR models have too. No structural advantage.

`server_args.py:2541-2545` again — until the radix cache is back, hicache integration cannot help. `enable_hierarchical_cache` is selected in `scheduler.py:662` only when `disable_radix_cache=False`.

**Verdict**: Same blocker as Candidate 1. Hicache is a no-op for LLaDA today.

### Bonus: load-only routing still works

The router's `decode_blocks` term is fed by `WorkerMetricsPublisher.publish(dp_rank, kv_used_blocks=…)` in `publisher.py:142-153`, which receives data over the ZMQ scheduler-metrics socket. SGLang's `KvMetrics` is emitted in `process_batch_result` regardless of `dllm_algorithm`. So if you run multiple LLaDA workers behind a Dynamo frontend with `--router-mode kv`, the router will at least balance load between them based on `decode_blocks`, ignoring the empty `prefill_blocks` term. With `overlap_score_weight=0`, this is identical to least-active routing.

---

## Concrete recommendations, ranked (Dynamo-only)

Reminder: nothing here modifies SGLang. These are config flags, launch-script edits, and pure-Dynamo code paths.

1. **Approximate KV routing + `--page-size 32`** *(lowest effort, modest value)*. Run multiple LLaDA workers behind one Dynamo frontend, edit `examples/backends/sglang/launch/diffusion_llada.sh` to add `--page-size 32` to the worker `CMD`, and launch the frontend with `--router-mode kv --no-router-kv-events --router-kv-overlap-score-weight 2.0`. Affinity is real (router self-records its decisions, `push_router.rs:541-570`); cache-hit benefit is bounded because `ChunkCache` frees KV on every request completion (`chunk_cache.py:57-64`). Expected: **≤ 5% TTFB reduction** on prefix-heavy workloads, ≤ 1% on uncorrelated, zero regression.

2. **Sticky-session routing for multi-turn chat** *(low effort, real value for chat)*. Requests carrying `nvext.session_control` get pinned to a worker (`docs/components/router/router-configuration.md:50-55`). Pin lasts for a sliding TTL. Independent of KV events. For chat workloads where each user-turn within a session shares all prior turns, this is the single best Dynamo-only intervention because it forces all turns of one conversation to hit the same scheduler — the worker still doesn't keep KV across turns under `ChunkCache`, but tokenizer state, weights, and process locality are warm. Combine with recommendation 1; they're complementary (sticky pins sessions; approximate handles cross-session prefix overlap). Expected: similar single-digit wins, but stronger on long chats.

3. **LLaDA cost model in the planner** *(low effort, highest non-latency value)*. The planner today uses generic AR-LLM cost estimates. LLaDA's cost is `ceil(max_new_tokens / 32) * ~8 * forward_pass_time(seq_len)`, deterministic — no early-exit variability. Adding a LLaDA-aware cost path lets the autoscaler size replicas accurately from `max_new_tokens` alone, not after observing queue depth. Concrete file: `components/planner/` — add a `BackendKind::DLLM` cost branch using the formula. No latency win per-request, but accurate capacity = no over-provisioning under tail load.

4. **Tune `--router-ttl-secs` for the workload** *(trivial)*. Default 120s. For long chat sessions, raise to 600s. For high-churn one-shot workloads, lower to 30s to avoid stale affinity. `lib/kv-router/src/indexer/pruning.rs:53-64` controls eviction. Measurable via approximate-routing hit-rate metric.

5. **Use load-only routing as the negative-control baseline**. With `--router-mode least_loaded` you get balanced load with zero routing-state overhead. This is what `--router-mode kv` decays to today because `prefill_blocks` is always zero (empty prefix tree). Useful as a clean comparison point in benchmarks.

6. **Observability stays valuable.** The scheduler-metrics ZMQ path works for LLaDA (`publisher.py:118-158`), so Prometheus / Grafana dashboards function unchanged. Tracing via OpenTelemetry works. This is "free" value Dynamo brings regardless of routing mode.

### Explicit non-recommendations under this constraint

- ~~Re-enable SGLang radix cache for DLLM~~ — would change `sglang/srt/server_args.py`, out of scope.
- ~~Emit `BlockStored` from `process_batch_result_dllm`~~ — would change `sglang/srt/managers/scheduler_output_processor_mixin.py`, out of scope.
- ~~Prompt-KV capture-and-replay via KVBM~~ — requires new SGLang hooks, out of scope.
- ~~KV quantization for higher LLaDA concurrency~~ — SGLang config / kernel path, out of scope.
- ~~Cross-request prompt batching with shared KV~~ — needs SGLang `BatchInfo` audit / changes, out of scope.
- ~~P/D disaggregation~~ — requires SGLang-side `init_dllm_prefill` and bootstrap-room support, out of scope (and architecturally questionable for LLaDA anyway).
- ~~Hicache / KVBM integration~~ — depends on radix cache being enabled engine-side, out of scope.

## Performance benchmark plan

Goal: produce a defensible number for "can the Dynamo router measurably improve LLaDA throughput/latency on this stack today, in either mode?" Don't oversell — back-of-envelope below caps the win for the realistic shared-prompt case at single-digit percent.

### Back-of-envelope: where is the time actually spent?

Total forward passes per request ≈ `1 (prompt encode) + ceil(N/32) * ~8 (refinement)` (refinement loop in `low_confidence.py:23-101` runs ~5-10 passes per block at threshold 0.95):
- `P=2000, N=50`: ~17 passes; prompt encode ≈ 6% of compute.
- `P=2000, N=200`: ~57 passes; prompt encode ≈ 2%.
- `P=8000, N=50`: with near-linear cost in sequence length, prompt portion can be 15-20%.

**Dynamo-only ceiling** (the only relevant number given the scope constraint): you cannot save the prompt-encode pass. `ChunkCache` frees the KV at request completion (`chunk_cache.py:57-64`), so even when the router perfectly co-locates two requests sharing a 2000-token system prompt, the second request re-encodes from scratch. The wins come from:
- **Warm-process overhead**: tokenizer state, weights resident, JIT-compiled kernels resident.
- **HTTP / connection keep-alive** between the frontend and the same worker.
- **Scheduler-internal locality**: sequential requests on the same worker get co-batched if they arrive close enough, amortizing the prompt encode across the batch *within a single scheduling pass*.

Expect **1-3% latency improvement** on identical-prompt workloads, **0% on uncorrelated**. Zero regression in the uncorrelated case because `decode_blocks` keeps load balanced via the cost function.

(For reference only — out of scope under the no-SGLang-changes constraint: if the radix cache were re-enabled engine-side, this ceiling would rise to **5-15% TTFB reduction** because the prompt-encode pass itself could be skipped. See "Out of scope (future work)" below.)

### Setup

Two-worker setup on the dual RTX PRO 6000. Worker 0 runs from existing `diffusion_llada.sh` (GPU 0, port 8001). Add a second worker on GPU 1, same `dyn://` endpoint, so the frontend discovers both.

```bash
# Terminal 1: frontend + worker 0 (existing diffusion_llada.sh after editing for page-size)
# Edit diffusion_llada.sh: add `--page-size 32` to CMD, keep HTTP_PORT=8001
./examples/backends/sglang/launch/diffusion_llada.sh

# Terminal 2: second LLaDA worker, GPU 1, NO new frontend
CUDA_VISIBLE_DEVICES=1 python -m dynamo.sglang \
    --model-path inclusionAI/LLaDA2.0-mini-preview \
    --tp-size 1 --skip-tokenizer-init --trust-remote-code \
    --endpoint dyn://dynamo.backend.generate \
    --enable-metrics --disable-cuda-graph --disable-overlap-schedule \
    --attention-backend triton --dllm-algorithm LowConfidence \
    --page-size 32
```

For the router-mode sweep, restart the frontend with the right flag set in front of `python -m dynamo.frontend`:

```bash
# Baseline: round-robin
DYN_ROUTER_MODE=round_robin python -m dynamo.frontend --http-port 8001

# Approximate KV (no engine events needed)
DYN_ROUTER_MODE=kv DYN_ROUTER_NO_KV_EVENTS=1 \
  python -m dynamo.frontend --http-port 8001 \
  --router-kv-overlap-score-weight 2.0 --router-ttl-secs 120

# Event-driven KV (will degenerate to load-only; included as negative control)
DYN_ROUTER_MODE=kv python -m dynamo.frontend --http-port 8001
```

(If the env-var path doesn't exist, the same flags exist on the CLI: `--router-mode kv --no-router-kv-events --router-kv-overlap-score-weight 2.0`.)

### Workloads

Both via `aiperf` (already installed at `/home/ayush-lab/Work/aiperf`):

**(A) Prefix-heavy** — should benefit from approximate routing:

```bash
aiperf profile \
  -m inclusionAI/LLaDA2.0-mini-preview \
  --tokenizer inclusionAI/LLaDA2.0-mini-preview \
  --url http://localhost:8001 --endpoint /v1/chat/completions \
  --streaming --random-seed 42 \
  --concurrency 8 --request-count 200 \
  --num-prefix-prompts 4 --prefix-prompt-length 2000 \
  --osl 64 --isl 64 \
  --artifact-dir /tmp/aiperf_llada_prefix_<mode>
```

`--num-prefix-prompts 4 --prefix-prompt-length 2000` produces 4 distinct 2000-token system prompts; each request picks one and appends a small user turn. With 2 workers, a good router should pin each system-prompt group to one worker (cache affinity), reducing repeated prompt work. With round-robin, half the requests for a given prefix go to the wrong worker.

**(B) Uncorrelated** — should NOT benefit, must not hurt:

```bash
aiperf profile \
  -m inclusionAI/LLaDA2.0-mini-preview --tokenizer inclusionAI/LLaDA2.0-mini-preview \
  --url http://localhost:8001 --endpoint /v1/chat/completions \
  --streaming --random-seed 42 --concurrency 8 --request-count 200 \
  --osl 64 --isl 512 \
  --artifact-dir /tmp/aiperf_llada_random_<mode>
```

Pure synthetic random prompts (no `--num-prefix-prompts`). Verify approximate routing is within noise of round-robin.

### Metrics

`aiperf` reports TTFT, request latency, output-token throughput, ITL, p50/p95/p99. For LLaDA specifically:
- **TTFB (time-to-first-block)** maps onto `time_to_first_token` in `aiperf` — LLaDA streams once per block.
- **End-to-end latency** is the cleaner metric (TTFT vs ITL distinction is fuzzier for a block-diffusion model).
- **Output token throughput** at fixed concurrency — proxy for cache-hit benefit when prompt is amortised.
- Router-side `dynamo_component_router_kv_hit_rate` (`docs/observability/metrics.md` — router-metrics section). In approximate mode this measures *predicted* hits (how often the router thought it found a match); in event mode it measures real hits. For LLaDA on `ChunkCache`, real hits will be ~0 regardless.

### Pass/fail criteria (Dynamo-only)

- **Prefix workload, approximate vs round-robin**: expect **≤ 3% TTFB reduction** and ≤ 3% throughput uplift. Single-digit, period. Anything bigger than 5% would be surprising and worth understanding before celebrating (likely a measurement artifact). Statistical significance over 200 requests with `--random-seed 42`.
- **Prefix workload, sticky-session vs round-robin** (chat workload with same session_id per virtual user): expect similar **1-3% TTFB**, but more consistent because pinning is deterministic. Best signal is reduced p99 latency variance from co-located scheduling.
- **Prefix workload, event-driven vs round-robin**: expect **near-equality** (degenerates to load-only since the prefix tree is empty). If significantly worse, file a Dynamo bug.
- **Uncorrelated workload, all KV modes vs round-robin**: expect **near-equality (≤ 1% delta either way)**. If approximate routing is noticeably worse, the TTL is hoarding stale affinity — tune `--router-ttl-secs` down. This is the "do no harm" check.
- **Capacity planning (offline)**: with a LLaDA cost model in the planner, expected request-completion-time prediction error should drop from ~30% (AR-style estimate) to ~5%. Validate against `aiperf`'s observed-vs-estimated request latency.

If any of the prefix-workload results show > 5% TTFB reduction in approximate mode — that's evidence the affinity effect is bigger than the back-of-envelope predicts. Investigate (likely tokenizer-warmup or kernel-cache effects on the worker process) and report.

---

## What's "the value" today in one sentence (Dynamo-only)

If you run multiple LLaDA workers behind Dynamo *today, with no SGLang changes*:
- `--router-mode kv` (event-driven) gives you **load-aware routing** (decode-block balancing) and observability, but no prefix-cache awareness — the event source is empty.
- `--router-mode kv --no-router-kv-events` (approximate) additionally gives you **session/prefix affinity** routing that survives `ChunkCache`, costing zero SGLang change. Realised gain is **single-digit percent** (1-3%) latency on prefix-heavy workloads, primarily from process-level locality.
- **Sticky sessions** (request-driven, `nvext.session_control`) deliver the same magnitude of win for *chat* workloads with stronger predictability.
- The genuinely high-leverage non-latency win is **LLaDA-aware capacity planning** in the autoscaler.

The 10-30% prefix-cache TTFB wins that autoregressive LLMs see require engine-side KV retention. Under the no-SGLang-changes constraint, those are unreachable. The bound is set by physics (refinement loop dominates compute) and by SGLang's `ChunkCache` policy, neither of which Dynamo can override unilaterally.

## Out of scope (future work, requires SGLang changes)

Documented here only so the path forward is visible if the constraint is relaxed later. **Do not implement these under the current scope.**

1. **Re-enable a controlled radix cache for the DLLM path in SGLang.** Gate at `sglang/python/sglang/srt/server_args.py:2541-2545` behind a new flag. Audit `cache_unfinished_req` for the DLLM path so masked-block tokens are not inserted into the radix tree until the block transitions clean (seam: `process_batch_result_dllm` per-block loop). Pass `--page-size 32 --kv-events-config tcp://*:5557` from the launch script. Estimated win: **10-30% TTFB** on prefix-heavy workloads — the real prize. Effort: medium; risk: correctness audit of the masked-block insertion path.

2. **Emit `BlockStored` directly from the diffusion handler** using `KvEventPublisher.publish_stored` (`docs/integrations/kv-events-custom-engines.md:47-109`), bypassing SGLang's radix cache. Hooks would live in `process_batch_result_dllm`. Decouples Dynamo routing from SGLang cache decisions. Same ceiling as #1, more invasive scheduler-loop integration.

3. **Prompt-KV capture-and-replay via KVBM.** Extract the prompt-portion K,V tensors after the encode pass (when they're computed but before any masked-block work corrupts the working set), store in KVBM, inject on a future request with the same prompt hash. Architecturally cleaner than #1 because it doesn't require the engine to retain anything past request completion; it just needs `extract_prompt_kv_after_encode` and `inject_kv_before_generate` hooks. Cross-worker reuse via NIXL transport falls out for free. Same ~10-30% ceiling, but cleaner semantics for multi-worker fleets.

4. **KV quantization for higher LLaDA concurrency.** Independent of routing. SGLang already supports FP8/INT8 KV in its kernel paths; LLaDA's `ENCODER_ONLY` attention should tolerate it. Estimated win: **~2× concurrent requests per GPU**. Effort: medium (configuration + benchmarking).

5. **Cross-request prompt-share batching.** When N requests arrive within a short window sharing the same system prompt, pack them into one SGLang batch so the prompt is encoded once and reused N-way within that batch. Needs verification of `BatchInfo` semantics in the DLLM path. Estimated win: **near-linear with batch size** on the prompt-encode portion (which is still only 2-6% of compute, so capped at single-digit % per-request, but throughput wins are larger).

6. **LLaDA-specific disaggregation** (prompt-encode stage + refinement stage). Architecturally questionable because both stages run the same model on the same KV; only meaningful if prompt-encode parallelism (e.g., higher TP for the encode-only worker) opens up wins that uniform deployment can't. Defer until benchmarks show prompt-encode is a meaningful fraction of total latency.

The pattern across all six: every real cache-hit win requires SGLang to either retain KV across requests or expose hooks to extract/inject it. Pure Dynamo-side cannot get there.

## Sources

- LLaDA paper: [Large Language Diffusion Models (Nie et al., 2025)](https://arxiv.org/abs/2502.09992) — HTML: [arxiv.org/html/2502.09992v3](https://arxiv.org/html/2502.09992v3).
- LLaDA 2.0 paper: [LLaDA2.0: Scaling Up Diffusion Language Models to 100B (Ant Group)](https://arxiv.org/abs/2512.15745).
- LLaDA 2.0 GitHub: [github.com/inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X).
- HuggingFace card: [huggingface.co/inclusionAI/LLaDA2.0-mini-preview](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview).
- SGLang diffusion LM doc: `/home/ayush-lab/Work/sglang/docs/supported_models/diffusion_language_models.md`.
- Dynamo router concepts: `/home/ayush-lab/Work/dynamo/docs/components/router/router-concepts.md`.
- Dynamo custom-engine KV events guide: `/home/ayush-lab/Work/dynamo/docs/integrations/kv-events-custom-engines.md`.
