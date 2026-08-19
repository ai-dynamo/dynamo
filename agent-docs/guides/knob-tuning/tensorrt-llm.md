<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM Runtime Knobs

Read this file only when the target runtime is TensorRT-LLM. Record the active runtime configuration before spending
GPU time. Target the TensorRT-LLM PyTorch backend. Do not steer agents toward the legacy C++ path; every hypothesis
must say which PyTorch-backend configuration surface owns the constraint.

## Triage Order

1. Record the PyTorch-backend LLM API configuration: `max_batch_size`, `max_num_tokens`, `max_seq_len`, MTP/max draft
   length, paged KV settings, and prefill/decode worker split when disaggregated.
2. Record runtime configuration: KV cache configuration, `free_gpu_memory_fraction`, explicit KV-token caps, CUDA
   graph configuration, overlap scheduler, attention DP, MoE backend, and disaggregation settings.
3. If runtime logs prove a capacity mismatch, run one focused capacity or graph ablation. Otherwise gather more runtime
   and AIPerf evidence before proposing a knob.

## Pruning Before GPU Time

- Co-include tensor parallelism with MoE expert parallelism, and require `tensor_parallel_size >=
  moe_expert_parallel_size` with divisibility.
- Keep `cuda_graph_config.max_batch_size` tied to the actual batch or concurrency upper bound for that experiment;
  oversizing graph capture wastes memory.
- Avoid extreme KV fractions such as `0.99` unless a memory ledger proves the workspace and graph buffers still fit.
- For synthetic or random prompts, do not count prefix or block reuse as a serving win unless the target traffic
  actually has shared prefixes.
- Keep speculative decoding out of general tuning unless it is the hypothesis under test.
- Leave headroom above `ISL + OSL` for chat-template overhead.
- If chunked prefill or context is off, the full prompt must fit in one scheduling step.
- `tensor_parallel_size` determines how many GPUs one model instance consumes. On a fixed-size node, `num_replicas =
  total_gpus / tp_size`. For throughput workloads, maximizing replicas (lower TP) almost always outperforms maximizing
  per-instance TP.

## Knob Families

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Larger-memory GPU does not improve throughput | `max_batch_size`, `max_num_tokens`, `max_seq_len`, `free_gpu_memory_fraction`, `max_tokens_in_paged_kv_cache`, MTP draft-token budget | higher actual batch/token budget or clear memory bottleneck | runtime log, runtime memory log, KV allocation, OOM boundary | HBM only helps if configuration admits more useful work |
| Runtime OOM or tiny KV cache despite valid launch configuration | `free_gpu_memory_fraction`, `max_tokens_in_paged_kv_cache`, KV cache type, paged vs continuous KV | recover memory fit, larger KV token budget, fewer stalls | memory ledger before/after model weights, KV blocks/tokens | free fraction is based on memory free at KV initialization, not total HBM |
| CUDA graph helps small shapes or obscures runtime behavior | `cuda_graph_config`, `use_cuda_graph`, overlap scheduler | lower host launch overhead or clear graph-related diagnosis | server graph logs, graph coverage, same-workload AIPerf run | disabling graph is a diagnostic fallback unless proven faster |
| MoE backend is wrong or version-sensitive | `moe_config.backend` such as `TRTLLM`, `CUTLASS`, `DeepGEMM`, autotuner flags | intended backend activation or correct fallback | runtime log, kernel names, correctness checks, AIPerf rerun | backend switches often need exact version and container support |
| PyTorch executor configuration caps admission | `max_num_tokens`, `max_batch_size`, autotuner or backend configuration | recover admission when runtime logs show underbatching | runtime configuration dump and admitted batch/token counts | do not change admission caps without queue/running-request evidence |
| Disaggregated prefill/decode rates mismatch | prefill `max_num_tokens`, decode `max_batch_size`, KV-transfer/transceiver configuration, overlap flags | balanced prefill/decode rates and less queueing | prefill/decode rate ledger, transfer timing, AIPerf | aggregated evidence does not prove disaggregated behavior |

## Evidence-Backed Entries

| Scenario | Knobs / config surface | Evidence signal | What to try first | Caveat |
|---|---|---|---|---|
| PyTorch-backend token budget underbatches prefill | `max_num_tokens`, `max_batch_size`, prefill/decode split | Long-context configurations size token budgets from observed ISL and concurrency rather than from a generic default. | If prefill underbatches, compute the target token budget from admitted batch and prompt lengths, then run one same-workload validation. | Wrong token cap can truncate the workload or underutilize prefill. |
| Runtime OOM or tiny KV cache after model load | `free_gpu_memory_fraction`, `max_tokens_in_paged_kv_cache`, `maxTokens`, memory check utility | Runtime memory pressure can appear only after weights, graph buffers, and KV/output tensors are allocated. | Use a memory check or runtime memory ledger before changing kernels. | Do not confuse runtime fit with model-load correctness. |
| Paged KV fraction semantics | `free_gpu_memory_fraction`, `maxTokens`, `max_tokens_in_paged_kv_cache` | The free fraction is applied to free memory at KV initialization, after model weights and buffers. If both explicit max tokens and free fraction exist, the effective allocation is the lower limit. | If KV looks tiny, record memory free at KV initialization and both caps before changing the fraction. | Raising the fraction can starve CUDA graph buffers or runtime workspaces. |
| Prefill-only or reranker workloads | lower KV allocation with `free_gpu_memory_fraction`, explicit max tokens sized to input length times batch | Prefill-only workloads can minimize KV instead of carrying a decode-sized cache. | For prefill-only goals, tune KV to the workload shape and label the result as prefill-only. | Not portable to decode or end-to-end serving. |
| Long-context prefill worker capacity | prefill `max_num_tokens`, KV fraction, chunked prefill, cache reuse | Prefill workers may need `max_num_tokens` sized to `concurrency * ISL`, with KV token capacity covering `concurrency * (ISL + 1)`. | If prefill underbatches long prompts, compute the token budget from target ISL and concurrency before changing kernels. | For 8k/1k and 32k/1k, caps can dominate any kernel conclusion. |
| CUDA graph behavior is unclear | `use_cuda_graph`, `cuda_graph_config` | Requested graph state and observed graph launches can differ when the wrong configuration surface is changed. | Prove graph state from runtime logs and counters before making graph conclusions. | Disabling graph can change timing and memory behavior. |
| PyTorch configuration caps capacity | PyTorch-backend configuration, `max_num_tokens`, attention backend, MoE backend, allreduce strategy, `use_cuda_graph`, FP8 KV | The TensorRT-LLM PyTorch backend exposes these fields directly. | Use the runtime configuration dump to identify the owning field before changing a CLI or manifest value. | Component-level behavior is not end-to-end serving proof. |
| Mamba/SSM module evidence | Mamba2 runtime components, prefill/decode measurement split | Runtime evidence can distinguish causal convolution, selective state update, and chunk-scan pressure. | If a hybrid model uses TensorRT-LLM, confirm the active Mamba/SSM path before changing one related cache or scheduler setting. | This evidence identifies a direction; it does not prove a runtime-knob improvement. |
| MoE backend and token-cap interplay | `moe_config.backend`, `moe_max_num_tokens`, graph mode | Runtime logs may show MoE token chunks spilling to an auxiliary stream in CUDA graph mode. | If MoE chunks spill or graph behavior changes, inspect the token cap and backend together. | Validate the interaction with serving logs and AIPerf. |

## Evidence Collection Checklist

- Save the launch command, LLM API configuration, runtime configuration, image, TensorRT-LLM version, and model
  revision.
- Save memory logs before and after model load and KV initialization.
- Save graph-state logs and graph coverage or capture counters.
- For backend changes, save activation logs, kernel names when available, and correctness checks before promotion.

If the symptom is missing, inspect the PyTorch-backend LLM API configuration, `max_batch_size`, `max_num_tokens`, KV
cache type and fraction, MTP configuration, CUDA graph configuration, overlap scheduler, autotuner, MoE backend,
attention DP, cache transceiver, and disaggregated serving configuration.

## Knob Semantics and Reference

Use this version-agnostic TensorRT-LLM configuration knowledge to read, choose, or validate one PyTorch-backend
configuration during runtime-knob triage. Verify release-specific names and defaults against the official TensorRT-LLM
documentation and repository listed in [`reference-repos.md`](../../references/reference-repos.md). Framework-agnostic
model-sizing and parallelism guidance is in [`parallelism.md`](../model-sizing/parallelism.md).

### Common Knobs

Exact API surfaces may differ by version.

- **Chunked prefill or chunked context** — splits long prefill into chunks so prefill and decode interleave. Strongly
  recommended `true` for production. The exact parameter name and prerequisites vary by version.
- **`max_batch_size`** — maximum concurrent requests in one batch; primary throughput/latency knob.
- **`max_num_tokens`** — maximum total scheduled tokens across all requests in one step.
- **`max_seq_len`** — per-request sequence cap. Generally fix it to the workload maximum rather than tuning it; memory
  impact differs by version and backend.
- **`kv_cache_config.free_gpu_memory_fraction`** — fraction of free GPU memory for the paged KV cache (default 0.9).
  **Warning:** lower `max_num_tokens` reduces profiled activation memory, so the KV allocator may grab more of the free
  memory, leaving less headroom for CUDA graphs and cuBLAS. Validate when combining low MNT with high fractions, such
  as MNT=2048 with a fraction of 0.90.
- **Tensor or pipeline parallelism** — `tensor_parallel_size` and `pipeline_parallel_size` exist in all versions. See
  the TP and replication policy in [`parallelism.md`](../model-sizing/parallelism.md). Whether parallelism is applied at
  startup depends on the version and backend.
- **`scheduler_config.capacity_scheduler_policy`** — admission policy when KV capacity is limited. Valid values are
  `"GUARANTEED_NO_EVICT"` (default and conservative: started requests run to completion), `"MAX_UTILIZATION"`
  (aggressive and may trigger recompute), and `"STATIC_BATCH"`. If admission behavior is the symptom,
  `GUARANTEED_NO_EVICT` and `MAX_UTILIZATION` are the two values worth comparing.
- **CUDA graph `max_batch_size` upper bound** — `cuda_graph_config.max_batch_size` sets the largest batch size for which
  CUDA graphs are pre-captured. Capturing graphs for batch sizes larger than the actual in-flight requests wastes GPU
  memory, approximately 200 MB per graph. If only `max_batch_size` is known, use it; if only concurrency is known, use
  `max(concurrencies)`; if both are known, use `min(max_batch_size, max(concurrencies))`. Round the chosen upper bound
  up to the nearest power of two, such as 100 to 128.
- **MoE expert parallelism** — `moe_expert_parallel_size` and `moe_tensor_parallel_size` exist in all versions. The hard
  constraint is that `moe_expert_parallel_size` must be less than or equal to `tensor_parallel_size`, and
  `tensor_parallel_size` must be divisible by `moe_expert_parallel_size`. The runtime computes `moe_tp_size =
  tensor_parallel_size / moe_expert_parallel_size` by floor division, so a larger EP silently floors to zero and
  crashes. Whenever you set `moe_expert_parallel_size`, set `tensor_parallel_size` explicitly because it otherwise
  defaults to one. EP=1 does not shard experts; it can maximize per-request throughput at very low concurrency but can
  become a serialization bottleneck as concurrency rises. Prefer EP=1 when optimizing low concurrency or when the
  workload is unknown.
- **Quantization or dtype** — `dtype` accepts `auto`, `float16`, `bfloat16`, or `float32`; generally leave it as `auto`.
  KV cache quantization exists in all versions, but its API and options differ. Verify that the selected KV-cache dtype
  is supported with the fixed weight quantization and validate correctness.
