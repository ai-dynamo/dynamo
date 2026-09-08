<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Runtime Knobs

Read this file only when the target runtime is SGLang. Record the active runtime configuration before spending GPU
time. SGLang performance failures can look like kernel failures until admission, prefill scheduling, CUDA graph
coverage, and backend activation are checked.

## Triage Order

1. Identify the lane: aggregated or disaggregated, ISL/OSL, precision, MTP/spec settings, TP/EP/DP, DP attention, and
   whether the target is 1k/1k, 8k/1k, or longer context.
2. Save launch args and server logs proving `--max-running-requests`, `--chunked-prefill-size`, `--max-prefill-tokens`,
   `--mem-fraction-static`, CUDA graph batch sizes, attention/MoE/A2A backend, KV dtype, Mamba/SSM dtype, and
   speculative settings.
3. If throughput variation is high, enable iteration logs such as `--decode-log-interval 1` or equivalent and compare
   iterations with similar token/request shapes before blaming kernels.
4. Run one focused runtime ablation only when the symptom matches a row below. Otherwise gather more server and AIPerf
   evidence before proposing a knob.

## Pruning Before GPU Time

- Do not bind `--max-running-requests` to benchmark concurrency unless logs show admission is the bottleneck;
  long-context Mamba/KV state can make that worse.
- Keep 1k/1k and 8k/1k claims separate until both regimes rerun under the same intended backend and cache settings.
- Treat backend flags as branch-sensitive: require startup logs or kernel names proving the intended attention, MoE,
  GDN/Mamba, A2A, and allreduce path.
- Do not continue a broad scheduler or prefill sweep after one focused ablation fails to explain the symptom; gather
  more targeted evidence first.
- For disaggregated runs, measure prefill and decode rates separately. Aggregated evidence is not enough to justify
  transfer or admission changes.

## Knob Families

| Symptom | Candidate knobs | Expected metric effect | Validation | Caveat |
|---|---|---|---|---|
| Actual running requests are below intended concurrency | `--max-running-requests`, `--max-total-tokens`, TP/EP/DP per-rank admission, model max length | higher admitted batch or clear admission-capped diagnosis | request/running counters, queued requests, OOM boundary, same-workload rerun | blindly tying max running requests to benchmark concurrency can hurt 8k/1k |
| Prefill interferes with decode or TTFT dominates | `--enable-prefill-delayer`, `--prefill-delayer-queue-min-ratio`, `--prefill-delayer-max-delay-ms`, `--scheduler-recv-interval`, `--chunked-prefill-size`, `--max-prefill-tokens` | improved TTFT or reduced prefill/decode interference | TTFT, TPOT, prompt/output tok/s, prefill/decode iteration logs | delayer can improve TTFT while hurting throughput |
| CUDA graph capture misses decode shapes | `--cuda-graph-max-bs`, `--cuda-graph-bs`, `--disable-cuda-graph`, `--disable-piecewise-cuda-graph` | fewer host gaps and higher decode output tok/s | server graph logs, graph hit/miss counters, active batch sizes | disabling graph is a debug fallback unless proven faster |
| KV or Mamba/SSM state limits batch capacity | `--kv-cache-dtype fp8_e4m3`, `--mem-fraction-static`, `--max-mamba-cache-size`, `--mamba-ssm-dtype`, `--mamba-full-memory-ratio`, `--mamba-scheduler-strategy extra_buffer` | recover long-context fit or raise useful running requests | cache allocation logs, Mamba state usage, accepted batch, 1k/1k vs 8k/1k split | Mamba cache and KV cache trade memory with each other |
| Qwen3.5/GDN path needs backend confirmation | `--attention-backend`, `--mm-attention-backend`, GDN/linear-attention availability, `--moe-runner-backend`, `--fp4-gemm-backend`, DeepEP/FlashInfer A2A, `--enable-symm-mem` | intended GDN/MoE/communication backend runs, or default is proven | server logs, kernel names, correctness checks | backend flags are branch and image sensitive |
| MTP/spec decode changes capacity or acceptance | `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, `--speculative-num-steps`, `--speculative-num-draft-tokens`, `--speculative-eagle-topk`, max running requests | higher accepted output tok/s or lower TPOT | acceptance rate, draft path logs, memory/cache allocation | MTP can be worse at high concurrency if draft state reduces batch |
| Disaggregated transfer or rate matching is the symptom | `--disaggregation-transfer-backend nixl/mooncake`, connector env vars, per-worker `--max-running-requests`, prefill/decode admission caps, DeepEP mode | balanced prefill/decode rates and lower queueing | prefill/decode throughput, queue depth, transfer timing, AIPerf | aggregated evidence is not enough for disaggregated claims |

## Evidence-Backed Entries

| Scenario | Knobs / config surface | Evidence signal | What to try first | Caveat |
|---|---|---|---|---|
| Qwen3.5 FP8 aggregated settings | `--chunked-prefill-size 16384`, `--max-prefill-tokens 16384`, `--kv-cache-dtype fp8_e4m3`, `--quantization fp8`, `--attention-backend trtllm_mha`, `--moe-runner-backend flashinfer_trtllm`, `--mamba-ssm-dtype bfloat16`, `--disable-radix-cache`, `--tokenizer-worker-num 6`, `--stream-interval 50`, `--scheduler-recv-interval 10/30`, `--enable-symm-mem` when supported | Compatible Qwen3.5 recipes use this family of settings. | Use the observed symptom to select one relevant configuration surface to change. | Do not set `--max-running-requests` equal to benchmark concurrency without proving an admission limit. |
| Qwen3.5 FP8 MTP/spec decode | Add `SGLANG_ENABLE_SPEC_V2=1`, `--speculative-algorithm EAGLE`, `--speculative-num-steps 3`, `--speculative-eagle-topk 1`, `--speculative-num-draft-tokens 4`; often cap running requests separately | MTP extends the non-MTP configuration with speculative decoding and draft-token state. | Validate acceptance rate and memory pressure before treating MTP as a win. | If MTP becomes worse than non-MTP at a concurrency, do not extend that lane without evidence. |
| Qwen3.5 FP4/NVFP4 B300 aggregated lane | `--quantization modelopt_fp4`, `--fp4-gemm-backend flashinfer_cutlass`, `--kv-cache-dtype fp8_e4m3`, `--chunked-prefill-size 32768`, `--max-prefill-tokens 32768`, `--context-length ISL+OSL+20`, `--mm-attention-backend triton_attn`, `--scheduler-recv-interval 10/30` | Compatible FP4 recipes pair larger prefill limits with architecture-specific attention and GEMM backends. | Confirm the selected configuration is active before exploring new knobs. | An FP4 result is not evidence for FP8 unless kernel and backend paths match. |
| Scheduler receive interval | `--scheduler-recv-interval 10`, `30`, and lane-specific values | Iteration logs can reveal scheduler gaps or variation among similar token shapes. | Ablate one neighboring scheduler interval when the observed symptom matches. | Avoid a broad manual sweep; gather more evidence after one inconclusive ablation. |
| Prefill delayer | `--enable-prefill-delayer`, `--prefill-delayer-queue-min-ratio`, `--prefill-delayer-max-delay-ms` | A delayer can improve batching or TTFT but reduce output throughput. | Use when small fragments prevent useful prefill batching; validate both TTFT and output tok/s. | Delayer wait metrics may report zero in some versions, so validate with AIPerf timing too. |
| CUDA graph and piecewise CUDA graph | `--cuda-graph-max-bs`, `--cuda-graph-bs`, `--disable-piecewise-cuda-graph`, `--disable-cuda-graph` | Server logs and counters show graph coverage, misses, or modes that disable capture. | Compare graph coverage with actual decode batch shapes. | DeepEP normal mode and prefill servers may intentionally disable graph. |
| GDN/Mamba/linear-attention bottleneck | `--mamba-ssm-dtype bfloat16`, `--mamba-scheduler-strategy no_buffer/extra_buffer`, `--max-mamba-cache-size`, `--mamba-full-memory-ratio`, GDN prefill/decode backend availability | Long-context runs or server counters show Mamba state, cache pressure, or low useful batch occupancy. | First prove cache shape, dtype, and active GDN kernels; then try one cache or scheduler setting. | A modeled bottleneck is a priority hint, not measured speedup evidence. |
| Attention and MoE backend selection | `--attention-backend trtllm_mha/trtllm_mla/flashinfer`, `--mm-attention-backend triton_attn`, `--moe-runner-backend flashinfer_trtllm/flashinfer_cutlass/flashinfer_cutedsl`, DeepGEMM envs | Backend availability and performance vary by architecture, precision, image, and SGLang version. | Confirm backend activation and correctness before running one backend ablation. | Correctness comes first for generated or CuTeDSL kernels. |
| Symmetric memory and collectives | `--enable-symm-mem`, `--enable-flashinfer-allreduce-fusion`, `NCCL_NVLS_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `NCCL_MNNVL_ENABLE=1` | Runtime evidence shows exposed TP allreduce or collective overhead. | Try one communication knob with topology and backend activation evidence. | Do not use communication flags as defaults without topology proof. |
| Disaggregated NIXL vs Mooncake | `--disaggregation-transfer-backend nixl/mooncake`, `UCX_TLS=^cuda_ipc`, Mooncake `MC_*` envs, batch API or zero-copy variants | Transfer backend performance and warmup behavior vary with topology and NIC placement. | Run repeated prefill/decode rate tests and compare transfer timing. | Requires topology-specific setup and must not influence aggregated runs. |
| DeepEP/WideEP scale-out | `--moe-a2a-backend deepep`, `--deepep-mode normal/low_latency`, `--ep-dispatch-algorithm`, redundant experts, per-rank max dispatch tokens, transfer backend | EP and disaggregated deployments use explicit prefill/decode roles and admission limits. | Use only when the target requires disaggregated or EP scaling and records prefill/decode rates. | Some settings can corrupt results or hang; correctness and health checks are mandatory. |

## Disaggregated SGLang Optimization Notes

These notes are first-run priors for SGLang prefill/decode disaggregation. Treat them as model-agnostic heuristics:
apply one only when the same symptom appears in the target run, then validate on that model, topology, precision, and
container image before treating it as evidence.

| Scenario | Knobs / config surface | Evidence signal | What to try first | Caveat |
|---|---|---|---|---|
| Prefill/decode balance is unclear | Prefill/decode topology, worker count, per-worker admission caps, rate-matching inputs | Prefill and decode rates diverge, queues grow on one side, or results use different topologies. | Define and record the target topology, then hold it fixed while testing one runtime delta. | Rate matching changes coupled fields; record topology search separately from runtime-knob evidence. |
| Exposed communication or scheduler-sync cost in disaggregated DP attention | NCCL all-gather overlap envs, symmetric-memory flags, transfer backend envs | Logs or metrics show exposed collective, scheduler-sync, or transfer time. | Try one communication knob while holding the selected topology and cadence fixed. | Communication flags are topology and image sensitive; do not promote them as defaults without target-run proof. |
| Prefill cadence interferes with decode | prefill stream cadence, `--chunked-prefill-size`, `--max-prefill-tokens`, decode stream cadence | Decode is underfilled, prefill arrives in bursts, or TTFT and output throughput move in opposite directions. | Prefer a focused prefill-side cadence ablation before increasing prefill chunk size. | Cadence wins may not transfer across concurrency, sequence length, or model architecture. |
| Scheduler receive interval is tempting in DP-attention disaggregation | `--scheduler-recv-interval` and scheduler-receive skipper behavior | Live help or startup validation shows whether the image supports receive skipping with DP attention. | Check the active image before queuing the experiment. | Some images reject this combination at startup; fail fast instead of burning a benchmark slot. |
| A recipe uses hack or unsafe envs | `SGLANG_HACK_*`, `*HACK*`, `*UNSAFE*` env vars | Recipe env includes a truthy hack or unsafe flag. | Use only for diagnosis and tag the result as hack-carrying. | Hack-carrying results must not be promoted. |
| Custom allreduce looks attractive | custom allreduce envs, fused allreduce, symmetric-memory/NVLS/NCCL envs | Runtime evidence shows exposed allreduce and logs prove the requested backend is active. | Try cleaner communication knobs first; add custom allreduce only with runtime and log support. | Requires correctness and health checks, especially on multi-node or disaggregated decode lanes. |

## Measurement Hygiene

- Record `--decode-log-interval 1` or equivalent when trying to explain throughput variation. Compare iterations with
  similar token counts before assuming a runtime knob helped.
- Record `/metrics` or server counters for queue length, running requests, prefill delay, graph hit/miss, accepted
  speculative tokens, and cache usage.
- Keep 1k/1k and 8k/1k conclusions separate. Mamba/SSM and KV pressure can flip the winner across those regimes.
- If evidence points to a dominant kernel family, stop runtime sweeping and request a targeted backend investigation.

If the symptom is missing, inspect `python -m sglang.launch_server --help`, the official SGLang repository listed in
[`reference-repos.md`](../../references/reference-repos.md), compatible recipes, `max-running-requests`, chunked
prefill, max prefill/total tokens, memory fraction, CUDA graph batch sizes, prefill delayer, scheduler receive interval,
KV dtype, Mamba cache knobs, attention/MoE/A2A backend knobs, and the disaggregation transfer backend.
