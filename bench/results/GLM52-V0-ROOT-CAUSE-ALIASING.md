# GMS V0 vs V1 on GLM-5.2-NVFP4 — root cause

## Head-to-head, measured

Same model, same image `7f9681b0`, identical B200 hardware at identical
clocks, both engines idle, identical random prompts, prefix cache 0%.

| ~prompt tokens | V0 TTFT | V1 TTFT | V0/V1 |
|---:|---:|---:|---:|
| 1,215 | 0.169 s | 0.114 s | 1.48x |
| 4,050 | 0.573 s | 0.176 s | 3.26x |
| 8,100 | 1.301 s | 0.433 s | 3.00x |
| 16,200 | 2.704 s | 0.756 s | 3.58x |
| 32,400 | **5.753 s** | **1.440 s** | **3.99x** |

Output quality, same prompt "The capital of France is":

| arm | text | top logprob |
|---|---|---|
| V1 | `' Paris. Distance from Paris to Lyon is...'` | -0.81 |
| V0 | `'importimportimport...'` | -3.43 |

V0's pos1 and pos2 distributions are **bitwise identical** — appending a
token changed nothing, i.e. attention output does not depend on context.

## Root cause: GMS keys by name, vLLM shares storage

Measured directly from the live committed GMS layout (`DYN_GMS_ALIAS_REPORT=1`):

```
4209 published keys -> 2549 distinct (allocation_id, offset)
929 locations shared;  2589 of 4209 names are aliases
```

Concrete groups:

| shared location | keys | meaning |
|---|---:|---|
| `cos_sin_cache` | 198 | one rotary cache referenced by every layer |
| `topk_indices_buffer` | 147 | **DSA indexer -> attention channel** |
| `kv_b_proj.weight` ≡ `W_UK_T` | 4/layer | `W_UK_T`/`W_UV` are views into `kv_b_proj` |

`W_UK_T` sits at the *same address* as `kv_b_proj.weight` (e.g. layer 10,
`alloc=90f022f1 off=16252928`) because
`MLAAttention.process_weights_after_loading` uses
`replace_parameter(..., prefer_copy=True)`, whose documented behaviour is to
"preserve the parameter's storage address".

The reader (`materialize_module_from_gms`, `client/torch/module.py`) walks
keys independently and `.clone()`s every buffer/tensor_attr. That gives each
alias a private copy, so a shared producer/consumer buffer stops being
shared. For `topk_indices_buffer` the indexer writes its top-k indices into
one copy while sparse attention reads another — which explains *both*
symptoms at once: attention has no usable sparse index (garbage output), and
it degrades toward a dense gather (cost that grows with context).

## Why V1 is immune

`GMSV1Worker` (`v1/integrations/vllm/worker.py`) keeps vLLM's **normal**
loader and intercepts only the *allocator scope*
(`capture_weights`/`capture_kv_cache`). The real model object is built and
post-processed exactly as in stock vLLM, so storage identity — and therefore
aliasing — is preserved by construction.

V0 instead builds a **meta** model and rebinds tensors **by name**. A
name->address map cannot express "these two names are the same storage", so
the aliasing is unrecoverable at the reader.

## Hypotheses eliminated by measurement (not argument)

| Hypothesis | Verdict |
|---|---|
| CUDA VMM memory is slow | **Dead.** gather 1050.1 vs 1050.2 GB/s (1.000x); GEMM 0.996x, numerics exact |
| VMM page fragmentation | **Dead.** KV is 99 handles of ~830 MiB; 2 MiB is alignment, not handle size |
| Zeroed MLA q/k/v/prob scales | **Dead.** Measured live: `_q/_k/_v/_prob_scale = 1.0`. The 312 zero-filled tensors are unused *staging* params; the checkpoint contains **zero** attention scales, so the reference path also resolves to 1.0 |
| Corrupt MoE NVFP4 scales | **Dead.** `(256,2)->(32,)` is vLLM's own documented collapse (`modelopt.py:1550-1556`) plus correct EP sharding |
| Corrupt weights generally | **Dead.** Measured healthy: MLA `W_UV`/`W_UK_T` finite/non-zero, MoE weights+scales correct, embeddings correctly TP-sharded, rotary cache absmax=1, norms non-zero |
| Sparse-attention "misroute" is V0-specific | **Dead.** Dispatch gates are a pure function of model config + vLLM version; identical in both arms |
| Different commit / compile mode / prefix caching / workload | **Dead.** All verified identical |

## Fix status — NOT yet landed

Preserving aliasing for shared buffers is the right direction, and the
instrumentation confirms the reader can identify alias groups from
`(allocation_id, offset_bytes)`. Two attempts were made and **both reverted**:

1. Share the materialized tensor for every aliased location — engine reached
   CUDA-graph capture then failed to start.
2. Share only `topk_indices_buffer` — same failure.

So simply not-cloning is insufficient: something else in the RO path (most
likely graph capture against these buffers, or a later rebind that assumes a
private copy) depends on the clone. The cluster was returned to the previous
working configuration; both engines are Ready and V0 serves HTTP 200.

## Recommendation

The V0 name-keyed meta-model reconstruction cannot represent tensor aliasing,
and aliasing is load-bearing in vLLM. Rather than continue reconstructing
post-load state name by name, adopt the V1 ownership model for V0: run vLLM's
normal loader and intercept the allocator. That is empirically correct and
~4x faster on this exact model and node today.
