---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sizing with AIConfigurator
subtitle: Search parallelism and aggregated vs disaggregated layouts before you deploy
---

[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator/tree/main) (AIC) is a standalone command-line tool that sizes a NVIDIA Dynamo deployment before you run it. Given a model, GPU system, backend, and latency target, it searches aggregated and disaggregated layouts and reports the tensor- and pipeline-parallel configuration that meets your SLA — and can generate ready-to-apply Dynamo Kubernetes manifests. AIConfigurator is developed and versioned separately from Dynamo; install it from PyPI:

```bash
pip3 install aiconfigurator
```

> [!IMPORTANT]
> AIConfigurator only produces reliable estimates for model + system + backend + version combinations in its [support matrix](https://ai-dynamo.github.io/aiconfigurator/support-matrix/). Confirm yours before sizing with `aiconfigurator cli support --model-path <model> --system <sku> --backend <backend>`.

## When to use it

Reach for AIConfigurator when you need to decide how to lay out a deployment and want a validated starting point rather than a hand-picked guess:

- Compare aggregated versus disaggregated serving for your workload.
- Choose tensor- and pipeline-parallel sizes, worker counts, and replica counts against a Time To First Token (TTFT) and Time Per Output Token (TPOT) target.
- Generate Dynamo configuration files and Kubernetes manifests for the selected layout.

Treat the output as a starting point, then benchmark the generated configuration in your cluster with [AIPerf](aiperf.md).

## Where to go next

- To copy a parallelism layout into a DynamoGraphDeployment you are already authoring, follow the [Size with AIConfigurator](../kubernetes/dgd-aiconfigurator.md) tutorial.
- For the full workflow — aggregated versus disaggregated comparison, generating complete manifests, and validating predictions — see the [AIConfigurator Reference](../features/disaggregated-serving/aiconfigurator.md).
- For the upstream command reference, see the [AIConfigurator CLI Guide](https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/cli_user_guide.md).
