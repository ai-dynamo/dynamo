<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rate Matching with AIPerf

Use rate matching only for deployments with separate prefill and decode workers and more than one feasible worker
allocation. For an aggregated deployment or a tiny allocation set, compare the complete configurations directly with
the target AIPerf workload.

Rate matching estimates how many prefill workers are needed to feed the decode workers without leaving either stage
consistently underused. AIPerf provides endpoint-level proxy measurements; it does not prove per-worker rates or the
cause of a bottleneck.

## Required Inputs

Keep these inputs fixed while measuring one workload:

- model, tokenizer, endpoint behavior, requested and observed ISL/OSL, EOS policy, seed, and request distribution
- AIPerf concurrency or request rate, request count or duration, warmup, and repetitions
- prefill and decode worker counts, GPUs per worker, and total GPU budget from `deploy.yaml`
- TP/EP/PP shape, runtime admission limits, KV-cache settings, and chunked-prefill settings for each worker type

AIPerf concurrency is client-side in-flight load. It does not prove that a worker admits or executes that many
requests.

## Proxy Measurements

Measure a prefill proxy with the target input shape and `OSL=1`:

```text
R_prefill_proxy = completed_requests / measurement_duration
```

Input-token throughput divided by observed input tokens per request may be used as a consistency check.

Measure a decode proxy at one or more AIPerf concurrencies `C` with a minimal fixed input and a fixed output length:

```text
R_decode_proxy(C) = output_token_throughput(C) / observed_output_tokens_per_request
```

Use fixed output controls when supported so early EOS does not distort the decode rate. Stop increasing concurrency
when errors, timeouts, or sharply worsening latency show that the point mainly measures queueing or memory pressure.

The prefill proxy, each decode-concurrency probe, and the target end-to-end workload use different AIPerf settings and
therefore belong to different benchmark series. Use the proxy series only for allocation planning, never as evidence
that a candidate improved the target workload.

## Candidate Allocation

When the measurements isolate one worker of the measured role and the opposite stage is not limiting the run, estimate:

```text
rate_ratio(C) = R_decode_proxy(C) / R_prefill_proxy
P * R_prefill_proxy ~= D * R_decode_proxy(C)
P * prefill_gpus_per_worker + D * decode_gpus_per_worker <= total_gpu_budget
```

`P` is the prefill-worker count and `D` is the decode-worker count. Prefer small integer allocations without large
excess capacity.

Do not treat ordinary multi-worker endpoint measurements as per-worker rates. If worker isolation or role-level
evidence is unavailable, use the proxy math only to rank possible allocations, then test the complete allocations
directly. A coordinated `P`/`D` adjustment may be one topology hypothesis when both fields must change to preserve the
GPU budget; record every changed field.

## End-to-End Validation

Deploy each selected allocation and run the unchanged target-workload AIPerf series. Record:

- worker allocation and active GPU count
- throughput, output throughput, TTFT, ITL, request latency, goodput when configured, and errors
- whether the result is valid and comparable under the existing benchmarking rules
- missing role-level evidence and any client-visible symptoms

Choose between allocations only from valid target-series measurements. AIPerf symptoms may suggest imbalance, but
AIPerf alone cannot establish per-role queues, runtime admission, KV-transfer engagement, cache pressure, graph
coverage, or which stage caused the result. State those signals as unavailable rather than inferring them.
