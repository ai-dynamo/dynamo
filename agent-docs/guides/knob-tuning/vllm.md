<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Runtime Knobs

Read this file only when the target runtime is vLLM. Record the active runtime configuration before spending GPU time.
This is an evidence index, not a default recipe: use it to bind an observed symptom to one focused experiment.

## Triage Order

1. Confirm the run is workload-valid: model, precision, ISL/OSL, `--max-model-len`, `--max-num-seqs`,
   `--max-num-batched-tokens`, TP/EP/DP, attention backend, MoE backend, MTP/speculative configuration, and image or
   version.
2. Preserve iteration evidence when available: request and server logs, request counts, CUDA graph logs, accepted-token
   counters, and AIPerf results.
3. If the symptom is capacity or scheduling, run one minimal knob ablation. If it is not, gather more targeted runtime
   and AIPerf evidence before proposing a knob.
4. Never promote a configuration because its flag list looks familiar. Promotion requires same-workload AIPerf
   evidence, correctness checks when relevant, and proof that the intended backend ran.

## Pruning Before GPU Time

- Keep `tensor_parallel_size * pipeline_parallel_size * data_parallel_size` compatible with the physical GPU count and
  any external replication plan.
- Do not include `--enforce-eager` in production throughput comparisons unless it is the explicit control for a graph
  diagnosis.
- Prune combinations such as FlashAttention with FP8 KV when the active vLLM version rejects them at startup; verify
  the supported backend names for the active version.
- Keep speculative decoding out of general tuning unless it is the hypothesis under test.
- Size CUDA graph coverage to the expected decode batch shapes, not blindly to a large maximum that wastes memory.
- Leave headroom above `ISL + OSL` for chat-template tokens in `max_model_len`.
- If chunked prefill is off, the full prompt must fit in one scheduling step through `max_num_batched_tokens`.
- Leave activation and workspace headroom; avoid extreme `gpu_memory_utilization` values such as `0.99`.
- `tensor_parallel_size` determines how many GPUs one model instance consumes. On a fixed-size node, potential replicas
  equal `total_gpus / tp_size`. For throughput workloads, lower TP can allow more replicas than higher per-instance TP;
  validate the actual Kubernetes deployment rather than projecting throughput from an untested replica count.

## Knob Families

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Eager fallback, missing CUDA graph, or graph-size migration | `--enforce-eager`, `--max-cudagraph-capture-size`, `--cuda-graph-sizes`, `--compilation_config.max_cudagraph_capture_size`, `--compilation_config.cudagraph_mode` | lower decode CPU/launch overhead, higher output tok/s, fewer host gaps | server graph logs, graph coverage, same-workload AIPerf run | eager is usually a diagnostic path, not a production optimization |
| Decode batch is split even though GPU has headroom | `--max-num-batched-tokens`, `--max-num-seqs`, speculative lookahead or draft-token configuration | higher active decode requests and output tok/s | iteration logs showing prefill/decode tokens and admitted sequences | excessively large token budgets can OOM or hurt TTFT |
| KV/cache capacity is exhausted before compute is saturated | `--gpu-memory-utilization`, `--max-model-len`, `--kv-cache-dtype`, prefix-cache controls, KV offload | deployment fits without OOM, more KV blocks, higher sustainable concurrency | startup memory log, KV block count, actual running requests, OOM boundary | shrinking `--max-model-len` below the target workload creates invalid wins |
| Prefill dominates TTFT or long-context throughput | `--max-num-batched-tokens`, chunked-prefill controls, scheduler policy, `--max-model-len` | higher input tok/s or lower TTFT without decode regression | prefill token counts, TTFT, prompt throughput, completed request counts | a 1k/1k win may not transfer to 8k/1k |
| FP8/FP4 MoE or attention path is suspicious | `VLLM_USE_FLASHINFER_MOE_FP8`, `VLLM_USE_FLASHINFER_MOE_FP4`, `VLLM_FLASHINFER_MOE_BACKEND`, `VLLM_NVFP4_GEMM_BACKEND`, `--moe-backend`, `--attention-backend`, `VLLM_KV_CACHE_LAYOUT`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, `VLLM_ALLREDUCE_USE_FLASHINFER`, `VLLM_USE_NCCL_SYMM_MEM` | move onto specialized kernels or prove the default is correct | server backend log, kernel names when available, correctness checks, AIPerf | backend flags are highly version- and hardware-sensitive |
| MTP or speculative decode is active but output rate is weak | `--speculative_config`, draft-token count, accepted-token tuning, `--max-num-batched-tokens`, `--max-num-seqs`, `--kv-cache-dtype`, `--mamba_ssm_cache_dtype` | higher accepted output tok/s or recovered batch capacity | accepted-token rate, iteration logs, actual running requests | draft tokens and Mamba/SSM state consume capacity |
| Startup or load time is the only blocker | `--load-format fastsafetensors`, model parser plugins, cache or preload options | faster model startup | model-load timing | keep startup-only results separate from serving-throughput results |

## Evidence-Backed Entries

| Scenario | Knobs / config surface | Evidence signal | What to try first | Caveat |
|---|---|---|---|---|
| CUDA graph behavior changes after a vLLM update | `--max-cudagraph-capture-size`; older versions may use `--cuda_graph_sizes` | Version changes can rename graph arguments or alter configuration behavior, causing eager or per-shape launches. | If graph logs show unexpected eager or per-shape launches, verify the live flag name and graph capture size before deeper tuning. | Treat this as version-migration evidence, not a universal graph-size value. |
| Speculative decode splits high-concurrency batches | `--max-num-batched-tokens` sized for base tokens plus speculative tokens | A token budget that omits draft or lookahead tokens can split active requests across scheduler steps. | Compute the required token budget from active requests plus draft or lookahead tokens, then test one point and inspect iteration logs. | A higher token budget can hurt TTFT or memory fit, especially with long context. |
| DeepSeek-R1 FP4 needs specialized attention, MoE, or allreduce flags | `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`, `VLLM_FLASHINFER_MOE_BACKEND=latency`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_USE_NCCL_SYMM_MEM=1`, `NCCL_NVLS_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `--kv-cache-dtype fp8`, `--async-scheduling`, graph capture size | Missing specialized backend or environment flags can leave FP4 on an unintended execution path. | Prove each intended backend is active from logs and kernel names when available; then change one missing configuration surface. | Some older allreduce environment variables become obsolete after runtime fixes; do not carry stale flags forward. |
| DeepEP or A2A path causes accuracy or communication regressions | `VLLM_ALL2ALL_BACKEND=deepep_low_latency`, `VLLM_DEEPEP_*`, `VLLM_MOE_DP_CHUNK_SIZE`, `flashinfer_moe_a2a`, `flashinfer_all2allv` | Correctness failures, dispatch/combine timing, or exposed communication can identify an unhealthy A2A path. | Gate A2A changes on correctness, then inspect dispatch/combine and communication evidence before claiming an improvement. | Isolated communication gains can lose end to end if routing, quantization, scatter, or idle time grows. |
| Qwen3 long-context FP8 attention path | `VLLM_KV_CACHE_LAYOUT=HND`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `--attention-backend FLASHINFER`, `--kv-cache-dtype fp8` | Compatible Qwen3 configurations may benefit from the FlashInfer/XQA path for long-context attention. | If long-context attention dominates and backend logs show another path, test one FlashInfer/HND change. | Model-specific evidence must be validated on the exact Qwen3.5 or Qwen3-Next workload. |
| Qwen3-Next or Qwen3.5 MoE FP8 backend is uncertain | `VLLM_USE_FLASHINFER_MOE_FP8=1`, `--enable-expert-parallel`, `--async-scheduling`, `--no-enable-prefix-caching`, graph capture size, Qwen3-Next MTP speculative configuration | FP8 can appear slow when the intended MoE backend is not active. | Before changing capacity knobs, record launch arguments and prove MoE backend activation. | FlashInfer JIT, cache, or dependency failures can masquerade as performance problems. |
| Nemotron or Spark NVFP4 backend is unstable | `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `VLLM_FLASHINFER_MOE_BACKEND=throughput`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, `--attention-backend TRITON_ATTN`, `--mamba_ssm_cache_dtype float32`; fallback `VLLM_NVFP4_GEMM_BACKEND=marlin` | Runtime logs or failures show instability in the selected FlashInfer or CUTLASS path. | Confirm the intended flags are active, then change only one backend surface. | Do not infer Qwen behavior from Spark without matching kernels and correctness. |
| MiniMax-M2.5 H200 throughput lane | `--attention-backend FLASHINFER`, larger `--max-num-seqs`, CUDA graph, `--kv-cache-dtype fp8`, no prefix caching for random prompts | Compatible configurations use FlashInfer, higher sequence capacity, CUDA graphs, and FP8 KV together. | Verify backend activation and capacity, then select one surface that matches the observed symptom. | Model-specific; MTP support depends on the active version. |
| Random-workload cache overhead | disable prefix caching for random datasets, `VLLM_USE_FLASHINFER_MOE_INT4=1`, lower `--gpu-memory-utilization` in fit-sensitive paths | Random synthetic workloads do not benefit from prefix reuse and can expose unrelated cache overhead. | Disable prefix caching only when the target dataset lacks reusable prefixes. | Do not generalize this result to traffic with shared prefixes. |
| CUDA graph memory estimation starves KV cache | lower `--gpu-memory-utilization` or use `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` when graph estimation over-reserves memory | Startup memory logs show graph reservation shrinking the effective KV budget despite available memory. | Compare the estimated graph reserve with the observed capture footprint before changing the memory setting. | Use only when graph reservation is the blocker; otherwise it can hide a real OOM. |

## Evidence Collection Checklist

- Save the exact command, environment, image, vLLM version, and model revision.
- Save server logs proving graph mode, attention backend, MoE backend, KV dtype and layout, `--max-num-seqs`,
  `--max-num-batched-tokens`, and model length.
- Save iteration logs or equivalent counters for prefill tokens, decode tokens, running requests, accepted draft tokens,
  and per-iteration CPU time.
- For backend changes, collect correctness checks before promotion.
- If a runtime knob does not explain the symptom after one focused ablation, stop knob tuning and gather more targeted
  evidence.

## Knob Semantics and Reference

Use this version-agnostic vLLM configuration knowledge to read, choose, or validate one configuration during
runtime-knob triage. Verify release-specific names and defaults against the official vLLM documentation and repository
listed in [`reference-repos.md`](../../references/reference-repos.md). Framework-agnostic memory and parallelism
guidance is in [`memory.md`](../model-sizing/memory.md) and [`parallelism.md`](../model-sizing/parallelism.md).

### Common Knobs

Exact API surfaces may differ by version.

- **Chunked prefill** (`enable_chunked_prefill`) — splits long prefill into chunks so prefill and decode interleave.
  Strongly recommended `true` for throughput workloads. vLLM v0.18 scheduler defaults to `true` for standard generation
  models.
- **`max_num_seqs`** — maximum sequences per iteration; primary throughput/latency knob, analogous to
  `max_batch_size` in TensorRT-LLM.
- **`max_num_batched_tokens`** — maximum total tokens admitted per scheduling step, analogous to `max_num_tokens` in
  TensorRT-LLM. Without chunked prefill, the largest prompt must fit within this budget. An approximate sizing relation
  is `max_num_batched_tokens ≈ max_num_seqs × avg_seq_len`.
- **`max_model_len`** — per-request sequence cap. Fix it to the workload maximum rather than tuning it unless GPU memory
  is tight after weights; see [`memory.md`](../model-sizing/memory.md).
- **`gpu_memory_utilization`** — fraction of total GPU memory for weights, KV cache, and buffers, with a default of 0.9;
  it is analogous to `kv_cache_config.free_gpu_memory_fraction` in TensorRT-LLM. In tight-memory situations, values
  from 0.90 to 0.95 may help. **Warning:** lower `max_num_batched_tokens` reduces profiled activation memory, so the KV
  allocator may take more free memory and leave less headroom for CUDA graphs. Validate low batched-token limits with
  high utilization.
- **Tensor, pipeline, and data parallelism** — the hard constraint is `tensor_parallel_size × pipeline_parallel_size ×
  data_parallel_size = total_gpus`. Keep `pipeline_parallel_size` and `data_parallel_size` at one by default and vary
  only `tensor_parallel_size`; use external replication rather than in-process DP. See the TP and replication policy in
  [`parallelism.md`](../model-sizing/parallelism.md).
- **CUDA graph coverage** — `cudagraph_capture_sizes` or `max_cudagraph_capture_size` inside `compilation_config`. If
  only `max_num_seqs` is known, use it; if only concurrency is known, use `max(concurrencies)`; if both are known, use
  `min(max_num_seqs, max(concurrencies))`. Round up to the nearest power of two.
- **MoE expert parallelism** — `enable_expert_parallel` distributes MoE experts through EP rather than TP;
  `moe_backend` selects the GEMM implementation. Verify constraints for the active vLLM version.
- **Quantization or dtype** — `dtype` accepts `auto`, `float16`, `bfloat16`, or `float32`; generally leave it as `auto`.
  KV-cache quantization uses `kv_cache_dtype`, with options varying by hardware and version. Verify that the selected
  KV-cache dtype is supported with the fixed weight quantization and validate correctness.
