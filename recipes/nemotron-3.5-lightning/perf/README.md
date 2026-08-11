<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning Benchmark Results

This directory records full-trace benchmark results for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`.

## Workload

The benchmark replays the short 15% Mooncake-format agentic trace:

- Trace: `traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl`
- Requests: 3,541
- Input/output shape: 64k input tokens, 400 output tokens
- KV cache reuse: 90%
- Pass criteria: `tok/s/user >= 50` and TTFT p50 < 5s

The trace symlink points to the shared trace under
`recipes/kimi-k2.6/perf/traces/`. Some trace requests exceed the model's native
262144-token context length. Those requests returned errors and are excluded
from valid-request latency and throughput metrics.

Rows use this AIPerf shape with the request count shown in the table:

```text
aiperf profile --custom-dataset-type mooncake_trace --num-requests <Requests> \
  --endpoint-type chat --streaming --use-server-token-count \
  --extra-inputs ignore_eos:true --request-timeout-seconds 1200
```

## Trace Results

| Recipe | FW | Mode | GPU | Routing | Spec method | Spec tok | Synthetic AL | Concurrency | Requests | Valid requests | Errors | TTFT p50 (ms) | Output tok/s/GPU | Tok/s/user |
|--------|----|------|-----|---------|-------------|----------|--------------|-------------|----------|----------------|--------|---------------|------------------|------------|
| `vllm/agg-b200-dflash/deploy.yaml` | vLLM | agg | B200 | Single worker | DFlash | 5 | 3.18 | 20 | 3541 | 3411 | 130 | 212.57 | 2004.06 | 100.20 |
| `vllm/agg-b200-dspark/deploy.yaml` | vLLM | agg | B200 | Single worker | DSpark | 7 | 3.69 | 24 | 3541 | 3411 | 130 | 207.44 | 2471.12 | 102.96 |
| `vllm/agg-b200-dspark-kv-router/deploy.yaml` | vLLM | agg | B200 | KV router | DSpark | 7 | 3.69 | 96 | 3541 | 3411 | 130 | 295.03 | 2507.70 | 104.49 |
| `vllm/agg-b200-mtp/deploy.yaml` | vLLM | agg | B200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3411 | 130 | 217.12 | 1829.94 | 101.66 |
| `vllm/agg-gb200-dflash/deploy.yaml` | vLLM | agg | GB200 | Single worker | DFlash | 5 | 3.18 | 20 | 3541 | 3411 | 130 | 429.97 | 842.42 | 42.12 |
| `vllm/agg-gb200-dspark/deploy.yaml` | vLLM | agg | GB200 | Single worker | DSpark | 7 | 3.69 | 24 | 3541 | 3411 | 130 | 296.82 | 2860.20 | 119.18 |
| `vllm/agg-gb200-mtp/deploy.yaml` | vLLM | agg | GB200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3411 | 130 | 402.99 | 1758.22 | 97.68 |
| `vllm/agg-h100-dflash/deploy.yaml` | vLLM | agg | H100 | Single worker | DFlash | 5 | 3.18 | 16 | 3541 | 3411 | 130 | 216.44 | 1757.75 | 109.86 |
| `vllm/agg-h100-dspark/deploy.yaml` | vLLM | agg | H100 | Single worker | DSpark | 7 | 3.69 | 20 | 3541 | 3411 | 130 | 238.64 | 1878.11 | 93.91 |
| `vllm/agg-h100-dspark-kv-router/deploy.yaml` | vLLM | agg | H100 | KV router | DSpark | 7 | 3.69 | 80 | 3541 | 3411 | 130 | 303.84 | 2037.43 | 101.87 |
| `vllm/agg-h100-mtp/deploy.yaml` | vLLM | agg | H100 | Single worker | MTP | 7 | 3.687 | 12 | 3541 | 3411 | 130 | 241.21 | 1275.37 | 106.28 |
| `vllm/agg-h200-dflash/deploy.yaml` | vLLM | agg | H200 | Single worker | DFlash | 5 | 3.18 | 16 | 3541 | 3411 | 130 | 243.30 | 1744.04 | 109.00 |
| `vllm/agg-h200-dspark/deploy.yaml` | vLLM | agg | H200 | Single worker | DSpark | 7 | 3.69 | 20 | 3541 | 3411 | 130 | 243.97 | 2193.59 | 109.68 |
| `vllm/agg-h200-dspark-kv-router/deploy.yaml` | vLLM | agg | H200 | KV router | DSpark | 7 | 3.69 | 80 | 3541 | 3411 | 130 | 286.67 | 2171.00 | 108.55 |
| `vllm/agg-h200-mtp/deploy.yaml` | vLLM | agg | H200 | Single worker | MTP | 7 | 3.687 | 12 | 3541 | 3411 | 130 | 250.66 | 1478.67 | 123.22 |
| `vllm/disagg-b200-dspark/deploy.yaml` | vLLM | disagg | B200 | 1P1D | DSpark | 1 | 1.83 | 8 | 500 | 489 | 11 | 236.43 | 904.03 | 226.01 |
| `vllm/disagg-gb200-dflash/deploy.yaml` | vLLM | disagg | GB200 | 1P1D | DFlash | 7 | 3.41 | 4 | 3541 | 3411 | 130 | 491.40 | 584.85 | 292.43 |
| `vllm/disagg-gb200-dspark/deploy.yaml` | vLLM | disagg | GB200 | 1P1D | DSpark | 1 | 1.83 | 8 | 3541 | 3411 | 130 | 599.17 | 939.28 | 234.82 |
| `vllm/disagg-h100-dflash/deploy.yaml` | vLLM | disagg | H100 | 1P1D | DFlash | 3 | 2.73 | 12 | 3541 | 3404 | 137 | 542.71 | 1117.29 | 186.21 |
| `vllm/disagg-h100-dspark/deploy.yaml` | vLLM | disagg | H100 | 1P1D | DSpark | 1 | 1.83 | 24 | 3541 | 3411 | 130 | 2028.23 | 1445.29 | 120.44 |
| `vllm/disagg-h100-mtp/deploy.yaml` | vLLM | disagg | H100 | 1P1D | MTP | 5 | 3.421 | 12 | 3541 | 3398 | 143 | 493.52 | 1055.68 | 175.95 |
| `vllm/disagg-h200-dflash/deploy.yaml` | vLLM | disagg | H200 | 1P1D | DFlash | 3 | 2.73 | 12 | 3541 | 3401 | 140 | 408.30 | 1277.86 | 212.98 |
| `vllm/disagg-h200-dspark/deploy.yaml` | vLLM | disagg | H200 | 1P1D | DSpark | 1 | 1.83 | 24 | 3541 | 3411 | 130 | 834.80 | 1656.90 | 138.08 |
| `vllm/disagg-h200-mtp/deploy.yaml` | vLLM | disagg | H200 | 1P1D | MTP | 5 | 3.421 | 12 | 3541 | 3342 | 199 | 424.66 | 1203.97 | 200.66 |
| `trtllm/agg-b200/deploy.yaml` | TensorRT-LLM | agg | B200 | Single worker | None | 0 | 0.0 | 18 | 500 | 489 | 11 | 377.68 | 386.998 | 21.50 |
| `trtllm/agg-b200-mtp/deploy.yaml` | TensorRT-LLM | agg | B200 | Single worker | MTP | 3 | 2.874 | 18 | 500 | 489 | 11 | 446.88 | 427.82 | 23.77 |
| `trtllm/agg-gb200/deploy.yaml` | TensorRT-LLM | agg | GB200 | Single worker | None | 0 | 0.0 | 18 | 500 | 499 | 1 | 318.88 | 280.10 | 15.56 |
| `trtllm/agg-gb200-mtp/deploy.yaml` | TensorRT-LLM | agg | GB200 | Single worker | MTP | 3 | 2.874 | 18 | 500 | 499 | 1 | 400.88 | 580.09 | 32.23 |
| `trtllm/agg-h100-mtp/deploy.yaml` | TensorRT-LLM | agg | H100 | Single worker | MTP | 7 | 3.687 | 12 | 500 | 499 | 1 | 1757.00 | 264.87 | 22.07 |
| `trtllm/agg-h200-mtp/deploy.yaml` | TensorRT-LLM | agg | H200 | Single worker | MTP | 7 | 3.687 | 12 | 500 | 499 | 1 | 1630.96 | 269.26 | 22.44 |
