---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Size with AIConfigurator
subtitle: Generate a tensor- and pipeline-parallel layout for your DGD worker from a latency target
---

This page is a detour from the **Determine topology and parallelism** step of [Deploy with DGD](dgd-guide.md). Rather than hand-pick tensor- and pipeline-parallel sizes, run AIConfigurator to search layouts against a latency target, then copy the recommended parallelism into your worker. It covers parallelism sizing only. For the full workflow — aggregated versus disaggregated comparison, generating complete manifests, and validating with AIPerf — see the [AIConfigurator](../features/disaggregated-serving/aiconfigurator.md) reference.

## Prerequisites

- A DGD in progress from [Deploy with DGD](dgd-guide.md), with the Frontend and worker defined and only parallelism left to set.
- AIConfigurator installed: `pip3 install aiconfigurator`.
- Your checkpoint's HuggingFace ID, GPU system (for example `h200_sxm`), backend (`vllm`, `sglang`, or `trtllm`), the total GPUs you can allocate, and your SLA: Time To First Token (TTFT), Time Per Output Token (TPOT), and typical input/output sequence lengths (ISL/OSL).

> [!IMPORTANT]
> AIConfigurator only models supported checkpoint + system + backend combinations. Confirm yours before sizing with `aiconfigurator cli support --model-path <model> --system <sku> --backend <backend>`. See the [support matrix](https://ai-dynamo.github.io/aiconfigurator/support-matrix/).

## Generate a configuration

Run `aiconfigurator cli default` with your GPU budget, system, backend, and SLA:

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm \
  --backend vllm \
  --backend-version 0.12.0 \
  --isl 4000 --osl 500 \
  --ttft 600 --tpot 16.67
```

- `--total-gpus` — GPUs available for the deployment; the search stays within this budget.
- `--system` — GPU SKU (`h200_sxm`, `h100_sxm`, `a100_sxm`).
- `--backend` / `--backend-version` — inference engine and version.
- `--isl` / `--osl` — input and output sequence lengths, in tokens.
- `--ttft` / `--tpot` — latency targets, in milliseconds; candidates that miss them are filtered out.

## Read the recommended parallelism

AIConfigurator ranks the layouts that meet your SLA. The `parallel` column is the layout to copy, and `gpus/worker` is the GPUs it needs (abridged):

```text
agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+--------+----------+--------------+-------------+----------+
| Rank | tokens/s/gpu |  TTFT  | replicas | gpus/replica | gpus/worker | parallel |
+------+--------------+--------+----------+--------------+-------------+----------+
|  1   |    322.69    | 546.92 |    2     |      4       | 4 (=4x1x1)  |  tp4pp1  |
|  2   |    293.94    | 593.10 |    4     |      2       | 2 (=2x1x1)  |  tp2pp1  |
+------+--------------+--------+----------+--------------+-------------+----------+
```

Read the top-ranked row:

- **`parallel`** — `tp4pp1` means tensor-parallel 4, pipeline-parallel 1.
- **`gpus/worker`** — GPUs one worker needs; equals TP × PP.
- **`replicas`** — how many copies of the worker to run.

## Apply it to your worker

Copy those three numbers into the worker you built in the DGD guide:

| AIConfigurator | DGD field |
|---|---|
| `parallel: tp4pp1` | `--tensor-parallel-size 4` (add `--pipeline-parallel-size N` when PP > 1) |
| `gpus/worker: 4` | `nvidia.com/gpu: "4"` — must equal TP × PP per node |
| `replicas: 2` | the worker's `replicas` |

```yaml
  - name: VllmWorker
    type: worker
    replicas: 2                         # recommended replicas
    podTemplate:
      spec:
        containers:
        - name: main
          command:
          - /bin/bash
          - -c
          - exec python3 -m dynamo.vllm --model $MODEL --tensor-parallel-size 4
          resources:
            limits:
              nvidia.com/gpu: "4"       # must equal TP × PP
```

> [!NOTE]
> Disaggregating? The `disagg` table sizes prefill and decode separately, as `(p)parallel` / `(p)gpus/worker` and `(d)parallel` / `(d)gpus/worker`. Apply each to the matching prefill and decode worker from the DGD guide's disaggregated step.

## Next steps

- Return to [Deploy with DGD](dgd-guide.md) to apply the deployment and send a request.
- For aggregated versus disaggregated comparison, generating complete Kubernetes manifests with `--deployment-target dynamo-j2`, and validating predictions with AIPerf, see the [AIConfigurator](../features/disaggregated-serving/aiconfigurator.md) reference.
