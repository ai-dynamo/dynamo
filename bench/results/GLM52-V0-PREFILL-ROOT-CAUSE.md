# GLM-5.2 GMS V0 vs V1 — prefill gap investigation

Working notes. **Supersedes the "MLA scales are the root cause" claim.**

## Retraction

An earlier conclusion held that the ~4x prefill gap was caused by GMS V0
skipping MLA `process_weights_after_loading` and zero-filling 312
`q_scale`/`k_scale`/`v_scale`/`prob_scale` tensors that the FP8 attention
kernel then consumed. **That is wrong.** Three checks kill it:

1. **The checkpoint contains no attention scales.** `model.safetensors.index.json`
   has 232,385 tensors and **0** ending in `q_scale`/`k_scale`/`v_scale`/`prob_scale`.
2. **Both paths land on 1.0.** On the reference path the staging params keep the
   `-1.0` sentinel, so `process_weights_after_loading` takes the "no scales
   loaded" branch and sets `k=v=1.0`
   (`vllm/model_executor/layers/quantization/kv_cache.py:111-116`). On the RO
   path the function is skipped, and the buffers the kernel actually reads
   (`_q_scale`, `_k_scale`, `_prob_scale`) were already created at 1.0
   (`layers/attention/attention.py:129-135`). The 312 zeroed tensors are
   **staging parameters the reference path reads and then deletes** — not what
   the kernel consumes.
3. **No scale-dependent kernel dispatch exists** on this backend. Scales fold
   into the scalar args `bmm1_scale`/`bmm2_scale`
   (`mla/flashinfer_mla_sparse.py:420-427`). A wrong scale is a numerics bug,
   not a speed bug.

The zero-fill is still a real **correctness** defect (it is why output is
garbage) but it is **not** the performance defect.

## Ruled out by measurement, not argument

| Hypothesis | How it died |
|---|---|
| GMS VMM memory slower than `cudaMalloc` | Microbenchmark in the engine container, same 576 B random-row gather as the paged-KV read: ordinary **1050.1 GB/s** vs VMM **1050.2 GB/s**; stream 6871 vs 6876 GB/s. Ratio **1.000**. |
| VMM fragmentation / TLB pressure | KV is **99 handles of ~830 MiB**, not thousands of 2 MiB fragments. 2 MiB is the *alignment*, not the handle size. |
| Sparse-attention misroute is V0-specific | Dispatch gates (`sparse_mla_attention.py:46-67`, `mla_attention.py:790-794`) are a pure function of model config + kv-cache dtype + vLLM version. Identical in both arms. |
| A dispatch cliff at `index_topk`=2048 | Measured TTFT from 540 to 32,400 tokens: per-token cost is **flat** (139 -> 178 us/token over a 27x length range). No cliff; the MQA path is always on, for both arms. |
| Different vLLM version / commit | Current V0 pod and the V1 reference both run image `7f9681b0`. |
| `CompilationMode.NONE` | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is set identically in the V0 DGD, V1 failover DGD, **and** the V1 dump DGD. |
| Prefix caching advantage for V1 | Both arms logged 0.0% hit rate for the entire run. |
| Workload difference | Both: ISL 32000, OSL ~1000, ~1250 s; gen/prompt token ratio 0.030 vs 0.027. |

## The actual finding

Scheduler windows where prefill and decode ran **in the same step**:

| Arm | windows | prefill+decode together | prefill-only |
|---|---:|---:|---:|
| **V1** | 118 | **113** | 4 |
| **V0** | 115 | **4** | 103 |

Restricted to comparable steady state (prompt >1000 tok/s and Running >=10):

| Arm | n | gen tok/s median | frac with gen >50 | per-running-req decode |
|---|---:|---:|---:|---:|
| **V1** | 115 | 510.0 | **0.99** | **10.25 tok/s** |
| **V0** | 102 | 37.1 | **0.04** | **0.69 tok/s** |

V0 is not losing per-kernel speed — it has **stopped co-scheduling decode with
chunked prefill**. Nearly every V0 step is consumed entirely by prefill, so
decode starves and the queue grows without bound (Waiting median 276, max 532,
vs V1's 22/73). This one mechanism explains both the ~3.7x prefill throughput
gap and the 6-18x ITL gap, with no GMS-specific kernel defect required.

### Live reproduction (same commit as V1)

24 concurrent pure-prefill requests against the current V0 engine:

```
prompt throughput: 4801.7 tok/s, gen: 2.2, Running: 4, Waiting: 27, KV 4.1%
prompt throughput: 7236.7 tok/s, gen: 1.9, Running: 4, Waiting: 28, KV 4.6%
```

Mean ~6,115 tok/s vs V1's 22,398 median = **3.66x**. `Running` is pinned at 3-4
with **28 requests queued and KV at 4%** — neither KV-limited nor
demand-limited. Per-step arithmetic is consistent: 8192 / 6115 = 1340 ms/step,
matching the profiled 1349 ms of self CUDA time for one 8192-token step.

### Single-stream sanity check

At batch 1 on an idle V0 engine, decode is **10.8 ms/token** (400 tokens in
4.31 s) — healthy. V0 collapses only under load: a scheduling/batching
signature, not a weight-path or memory signature.

## Caveats

- V1's scheduler config could not be read directly (CRIU-restored engines emit
  no init lines). It is inferred from the shared dump DGD. Neither arm passes
  `--max-num-batched-tokens`, so both inherit the vLLM default of 8192.
- The V0 arm carries a `PYTHONPATH=/vllm-cache/py` `sitecustomize.py` overlay
  that V1 never loads. **This is an uncontrolled variable** and should be
  removed before quoting any further V0-vs-V1 number.
- Why V0 fails to co-schedule decode is **not yet root-caused**. Next step is to
  instrument the scheduler decision (Waiting non-zero while Running stays at
  3-4), not to profile kernels again.

## Method note

Every hypothesis above that was killed by measurement had first been argued for
on plausible mechanism. The profile, the roofline, and the microbenchmark each
overturned a conclusion reached by reasoning alone. Measure first.
