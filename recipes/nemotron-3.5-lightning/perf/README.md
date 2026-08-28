<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning Benchmark Results

This directory records full-trace benchmark results for the NVFP4 and BF16
variants of Nemotron 3.5 Lightning. BF16 B200 and GB200 rows are added as
results become available.

## Workload

The benchmark replays the short 15% Mooncake-format agentic trace:

- Trace: `traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl`
- Requests: 3,541
- Input/output shape: 64k input tokens, 400 output tokens
- KV cache reuse: 90%
- Pass criteria: `tok/s/user >= 50` and TTFT p50 < 5s

Rows use this AIPerf shape with the request count shown in the table:

```text
aiperf profile --custom-dataset-type mooncake_trace --num-requests <Requests> \
  --endpoint-type chat --streaming --use-server-token-count \
  --extra-inputs ignore_eos:true --request-timeout-seconds 1200
```

## Trace Results

### H100 and H200

| Precision | Recipe | FW | Mode | GPU | Routing | Spec method | Spec tok | Synthetic AL | Concurrency | Requests | Valid requests | Errors | TTFT p50 (ms) | Output tok/s/GPU | Tok/s/user |
|----------|--------|----|------|-----|---------|-------------|----------|--------------|-------------|----------|----------------|--------|---------------|------------------|------------|
| NVFP4 | `vllm/agg-h100-dflash/deploy.yaml` | vLLM | agg | H100 | Single worker | DFlash | 5 | 3.18 | 16 | 3541 | 3411 | 130 | 216.44 | 1757.75 | 109.86 |
| NVFP4 | `vllm/agg-h100-dspark/deploy.yaml` | vLLM | agg | H100 | Single worker | DSpark | 7 | 3.69 | 20 | 3541 | 3411 | 130 | 238.64 | 1878.11 | 93.91 |
| NVFP4 | `vllm/agg-h100-dspark-kv-router/deploy.yaml` | vLLM | agg | H100 | KV router | DSpark | 7 | 3.69 | 80 | 3541 | 3411 | 130 | 303.84 | 2037.43 | 101.87 |
| NVFP4 | `vllm/agg-h100-mtp/deploy.yaml` | vLLM | agg | H100 | Single worker | MTP | 7 | 3.687 | 12 | 3541 | 3411 | 130 | 241.21 | 1275.37 | 106.28 |
| NVFP4 | `vllm/agg-h200-dflash/deploy.yaml` | vLLM | agg | H200 | Single worker | DFlash | 5 | 3.18 | 16 | 3541 | 3411 | 130 | 243.30 | 1744.04 | 109.00 |
| NVFP4 | `vllm/agg-h200-dspark/deploy.yaml` | vLLM | agg | H200 | Single worker | DSpark | 7 | 3.69 | 20 | 3541 | 3411 | 130 | 243.97 | 2193.59 | 109.68 |
| NVFP4 | `vllm/agg-h200-dspark-kv-router/deploy.yaml` | vLLM | agg | H200 | KV router | DSpark | 7 | 3.69 | 80 | 3541 | 3411 | 130 | 286.67 | 2171.00 | 108.55 |
| NVFP4 | `vllm/agg-h200-mtp/deploy.yaml` | vLLM | agg | H200 | Single worker | MTP | 7 | 3.687 | 12 | 3541 | 3411 | 130 | 250.66 | 1478.67 | 123.22 |
| NVFP4 | `vllm/disagg-h100-dflash/deploy.yaml` | vLLM | disagg | H100 | 1P1D | DFlash | 3 | 2.73 | 12 | 3541 | 3404 | 137 | 542.71 | 1117.29 | 186.21 |
| NVFP4 | `vllm/disagg-h100-dspark/deploy.yaml` | vLLM | disagg | H100 | 1P1D | DSpark | 1 | 1.83 | 24 | 3541 | 3411 | 130 | 2028.23 | 1445.29 | 120.44 |
| NVFP4 | `vllm/disagg-h200-dflash/deploy.yaml` | vLLM | disagg | H200 | 1P1D | DFlash | 3 | 2.73 | 12 | 3541 | 3401 | 140 | 408.30 | 1277.86 | 212.98 |
| NVFP4 | `vllm/disagg-h200-dspark/deploy.yaml` | vLLM | disagg | H200 | 1P1D | DSpark | 1 | 1.83 | 24 | 3541 | 3411 | 130 | 834.80 | 1656.90 | 138.08 |
| NVFP4 | `trtllm/agg-h100-mtp/deploy.yaml` | TensorRT-LLM | agg | H100 | Single worker | MTP | 7 | 3.687 | 12 | 500 | 499 | 1 | 1757.00 | 264.87 | 22.07 |
| NVFP4 | `trtllm/agg-h200-mtp/deploy.yaml` | TensorRT-LLM | agg | H200 | Single worker | MTP | 7 | 3.687 | 12 | 3541 | 3527 | 14 | 1692.98 | 250.05 | 20.84 |

### B200 and GB200

The following table contains BF16 rows for B200 and GB200 recipes.

| Precision | Recipe | FW | Mode | GPU | Routing | Spec method | Spec tok | Synthetic AL | Concurrency | Requests | Valid requests | Errors | TTFT p50 (ms) | Output tok/s/GPU | Tok/s/user |
|----------|--------|----|------|-----|---------|-------------|----------|--------------|-------------|----------|----------------|--------|---------------|------------------|------------|
| BF16 | `vllm/agg-b200-dspark-bf16/deploy.yaml` | vLLM | agg | B200 | Single worker | DSpark | 7 | 3.69 | 50 | 3541 | 3410 | 131 | 221.49 | 4005.48 | 80.11 |
| BF16 | `vllm/agg-b200-dspark-kv-router-bf16/deploy.yaml` | vLLM | agg | B200 | KV router | DSpark | 7 | 3.69 | 192 | 3541 | 3410 | 131 | 254.26 | 4162.09 | 86.71 |
| BF16 | `vllm/agg-b200-mtp-bf16/deploy.yaml` | vLLM | agg | B200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3410 | 131 | 165.43 | 2454.72 | 136.37 |
| BF16 | `vllm/disagg-b200-dspark-bf16/deploy.yaml` | vLLM | disagg | B200 | 1P1D | DSpark | 1 | 1.83 | 80 | 3541 | 3410 | 131 | 1059.12 | 3711.37 | 92.78 |
| BF16 | `vllm/agg-gb200-dspark-bf16/deploy.yaml` | vLLM | agg | GB200 | Single worker | DSpark | 7 | 3.69 | 58 | 3541 | 3410 | 131 | 307.73 | 4327.90 | 74.62 |
| BF16 | `vllm/agg-gb200-mtp-bf16/deploy.yaml` | vLLM | agg | GB200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3410 | 131 | 228.77 | 2289.75 | 127.21 |
| BF16 | `vllm/disagg-gb200-dflash-bf16/deploy.yaml` | vLLM | disagg | GB200 | 1P1D | DFlash | 7 | 3.41 | 24 | 3541 | 3410 | 131 | 10641.10 | 1148.33 | 95.69 |
| BF16 | `vllm/disagg-gb200-dspark-bf16/deploy.yaml` | vLLM | disagg | GB200 | 1P1D | DSpark | 1 | 1.83 | 8 | 3541 | 3410 | 131 | 569.70 | 802.30 | 200.58 |
| BF16 | `trtllm/agg-b200-bf16/deploy.yaml` | TensorRT-LLM | agg | B200 | Single worker | None | 0 | 0.0 | 18 | 3541 | 3411 | 130 | 219.18 | 1869.54 | 103.86 |
| BF16 | `trtllm/agg-b200-mtp-bf16/deploy.yaml` | TensorRT-LLM | agg | B200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3411 | 130 | 244.95 | 3295.50 | 183.08 |
| BF16 | `trtllm/agg-gb200-bf16/deploy.yaml` | TensorRT-LLM | agg | GB200 | Single worker | None | 0 | 0.0 | 18 | 3541 | 3411 | 130 | 216.77 | 1985.84 | 110.32 |
| BF16 | `trtllm/agg-gb200-mtp-bf16/deploy.yaml` | TensorRT-LLM | agg | GB200 | Single worker | MTP | 3 | 2.874 | 18 | 3541 | 3411 | 130 | 231.34 | 3556.40 | 197.58 |
