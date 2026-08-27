<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 NVFP4: GMS V0 reaches V1 steady-state parity

## Result

Same model, same image, identical B200 nodes clocked at 1965 MHz, both arms
idle, 0% prefix-cache hit. The V0 publishing engine was 3-4x slower than V1 and
emitted garbage; it now matches V1 on all three target metrics and produces
identical greedy text.

### Single request, TTFT vs prompt length

| ~tokens | V0 before | V0 after | V1 | after / V1 |
|--------:|----------:|---------:|-------:|-----------:|
| 1,215 | 0.169 s | 0.118 s | 0.110 s | 1.07x |
| 4,050 | 0.573 s | 0.200 s | 0.191 s | 1.05x |
| 8,100 | 1.301 s | 0.444 s | 0.395 s | 1.12x |
| 16,200 | 2.704 s | 0.800 s | 0.789 s | 1.01x |
| 32,400 | 5.753 s | 1.652 s | 1.631 s | 1.01x |

### Steady state, concurrency 8, ~3,240-token prompts, 120 output tokens

| Metric | V0 before | V0 after | V1 | after / V1 |
|--------|----------:|---------:|-------:|-----------:|
| TTFT p50 | ~1.75 s | 0.439 s | 0.430 s | 1.02x |
| ITL p50 | ~155 ms | 12.5 ms | 12.2 ms | 1.02x |
| output tok/s/user | ~6.4 | 55.17 | 56.42 | 0.98x |
| throughput | ~0.8 req/s | 3.08 req/s | 3.12 req/s | 0.99x |

Both arms ran 150 s at concurrency 8 with zero errors (464 and 473 completed
requests). Numbers above are from the final reviewed code; an earlier run of the
same configuration gave 0.438 s / 12.5 ms / 55.03 / 3.07, so the result is
stable across reloads.

Greedy continuation of `"The capital of France is"` is byte-identical between
arms: `" Paris. Distance from Paris to Lyon is 243 miles, ..."`. Before the fix
V0 emitted `"importimportimport..."`.

## Defect 1: name-based rebinding severs unreachable tensor holders

`rebind_nonparameter_tensors` moved GMS-resident non-parameter tensors onto
private storage by **name**: walk `module._modules`, clone each tensor, assign
the clone back onto the module attribute. That silently breaks any tensor whose
holders are not all reachable by module traversal.

The DSA top-k channel is exactly such a tensor. One `topk_indices_buffer` has
four holders:

| # | Holder | Reachable by name walk? |
|---|--------|-------------------------|
| 1 | `Indexer.topk_indices_buffer` | yes |
| 2 | `SparseAttnIndexer.topk_indices_buffer` (producer) | yes |
| 3 | `MultiHeadLatentAttentionWrapper.topk_indices_buffer` | yes |
| 4 | `MLAAttention.impl.topk_indices_buffer` (consumer) | **no** |

`MLAAttention.impl` is an `AttentionImpl`, not an `nn.Module`, so
`_iter_module_tensors` never visits it. After a name-based rebind the indexer
wrote top-k indices into the new buffer while attention read the old one, which
`torch.empty` had never initialized. Attention scored against arbitrary KV
slots: incoherent output, and destroyed KV-gather locality, which is where the
3-4x latency came from. It also explains the bitwise-identical consecutive
decode distributions observed earlier -- attention output was
context-independent.

**Fix.** Swap storage in place with `Tensor.set_()`, one private storage per
source storage. Preserving the TensorImpl makes every holder follow, including
holders that cannot be enumerated. GMS V1 gets this invariant for free
(`v1/client/parameter_storage.py` rebinds with `tensor.set_()` and discovers
tensors through `gc.get_objects()`), which is why V1 was never affected.

## Defect 2: per-name cloning splits deliberate aliases

`materialize_module_from_gms` cloned each published name separately, so vLLM's
intentional aliases became unrelated tensors: `W_UK_T`/`W_UV` are views over
`kv_b_proj.weight`, and one `cos_sin_cache` is shared by every layer.

**Fix.** One clone per `(allocation_id, offset_bytes)`; each name gets an
`as_strided` view of it, with dtype and extent asserted. Same fix in
`_clone_triton_incompatible_params_off_gms`, so `mlp.gate.e_score_correction_bias`
and `mlp.experts.routed_experts.e_score_correction_bias` keep sharing the
routing bias instead of being split into two copies.

## Defect 3: leftover meta scales zero-filled

Tensors still on the meta device after name-keyed materialization were filled
with **zeros**. 312 per rank are the MLA `q_scale`/`k_scale`/`v_scale`/
`prob_scale` Parameters: the writer registers these under their `_`-prefixed
buffer names, so the reader's un-prefixed Parameters find no match. A scale is a
multiplicative factor, so zero scales attention to nothing.

vLLM's `init_fp8_kv_scales` runs on wake-from-sleep and resets only
`k_scale`/`v_scale` to 1.0, so `q_scale` and `prob_scale` stayed at zero. That
is why the damage presented as gradual quality decay rather than outright
failure.

**Fix.** Fill leftover meta tensors with the neutral value: 1.0 for the four
known MLA scale leaves, 0.0 otherwise, matching vLLM's own
`set_default_quant_scales`.

## Method: whole-model tensor fingerprints

The decisive tool was a per-tensor fingerprint (shape, stride, dtype, and a
blake2b hash of a strided byte sample) dumped from both arms and diffed.

| Comparison | Result |
|------------|--------|
| Two different TP ranks (control) | 1760 / 4413 differ -- the fingerprint discriminates |
| Writer vs reader, at load | 0 / 4209 differ |
| Writer, load vs after sleep/wake | only KV caches differ, as expected |
| Reader, load vs after wake | exactly 156 differ, all `k_scale`/`v_scale`, hash of 0.0 -> hash of 1.0 |

That last row is what identified defect 3: it showed which tensors
`init_fp8_kv_scales` repaired, and by elimination which ones it did not.

## Ruled out by measurement

Recorded so these are not re-litigated:

| Hypothesis | Measurement |
|------------|-------------|
| VMM mappings are slower than ordinary CUDA memory | gather 1050.1 vs 1050.2 GB/s; bf16 GEMM 1596 vs 1590 TFLOP/s, numerics exact |
| VMM fragmentation | KV cache is 99 handles of ~830 MiB; 2 MiB is alignment, not handle size |
| Corrupt weights | all 4209 shared tensors byte-identical between arms |
| MoE scale shape collapse `(32,2)->(32,)` | correct per `modelopt.py` |
| Sparse-attention misroute | gates are pure config functions, identical in both arms; no cliff at `index_topk` |
| Scheduler starving the V0 arm | loaded-arm arithmetic matches single-request numbers; both arms decode every step |
| Different commit / compile mode / prefix cache / workload | all verified identical |
| Reader KV cache sized differently | same boot shows identical accounting and 1,890,624 tokens on both |

## Status

Publishing engine: at parity, verified above.
Read-only failover engine: defects 1-3 fixed and verified byte-identical to the
publisher after wake, but its output still degrades; the remaining cause is not
in tensor state and is tracked separately.
