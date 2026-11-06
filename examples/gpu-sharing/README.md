<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Many Models per GPU: Reference Manifests and Scripts

This directory holds the manifests and scripts behind the
[GPU sharing results](../../docs/fern/pages/use-cases/gpu-sharing/overview.mdx)
for many-model serving. It reproduces three experiments on one node with 8x A100 40GB GPUs:

| Experiment | Directory | DGDs | Placement | Isolation |
|---|---|---:|---|---|
| Baseline | [`baseline/`](baseline/) | 8 | One model per GPU (`nvidia.com/gpu: "1"`) | Dedicated GPU |
| KAI + HAMi | [`kai-hami/`](kai-hami/) | 16 | Two models per GPU (`gpu-fraction: "0.5"`) | Memory cap only (HAMi); compute is time-sliced |
| KAI + GPU fractions | [`kai-gpu-fractions/`](kai-gpu-fractions/) | 16 | Two models per GPU (`gpu-fraction: "0.5"` + `sm-sharing`) | Memory cap (19,968 MiB) and 50% MPS active threads |

For prerequisites, setup, deployment, benchmarking, and known issues, see the step-by-step guides:
[baseline](docs/deploy-baseline.md),
[KAI + HAMi](docs/deploy-kai-hami.md), and
[KAI + GPU fractions](docs/deploy-kai-gpu-fractions.md), which
requires [the KAI-Scheduler and GPU fractioning forks](docs/build-gpu-fractioning-forks.md).

## Contents

| Path | Purpose |
|---|---|
| [`common/download-model.sh`](common/download-model.sh) | Downloads Qwen3-4B at the pinned revision into a hostPath cache (`HF_CACHE_DIR`, default `/opt/hf-cache`) with a Kubernetes Job |
| [`common/install-dynamo-platform.sh`](common/install-dynamo-platform.sh) | Installs `dynamo-platform` 1.4.2 with `global.etcd.install=true` and `global.nats.install=true` |
| [`common/setup-aiperf.sh`](common/setup-aiperf.sh) | Creates a venv with AIPerf 0.11.0 |
| [`common/run-sweep.sh`](common/run-sweep.sh) | Runs the concurrency sweep against `NUM_MODELS` DGD frontends at once |
| [`common/aiperf-client-pod.yaml`](common/aiperf-client-pod.yaml) | In-cluster load-generator pod that runs `run-sweep.sh` with `ENDPOINT_MODE=dns` |
| [`baseline/gen-dgds.sh`](baseline/gen-dgds.sh), [`baseline/dgds-8x.yaml`](baseline/dgds-8x.yaml) | Generator and rendered manifests for the 8-DGD baseline |
| [`kai-hami/install-kai-hami.sh`](kai-hami/install-kai-hami.sh) | Installs KAI-Scheduler v0.17.0 with the hamicore plugin, the HAMi resource isolator 1.1.0-chart, and opens the default queue quotas |
| [`kai-hami/gen-dgds.sh`](kai-hami/gen-dgds.sh), [`kai-hami/dgds-16x.yaml`](kai-hami/dgds-16x.yaml) | Generator and rendered manifests for the 16-DGD KAI + HAMi run |
| [`kai-gpu-fractions/gen-dgds.sh`](kai-gpu-fractions/gen-dgds.sh), [`kai-gpu-fractions/dgds-16x.yaml`](kai-gpu-fractions/dgds-16x.yaml) | Generator and rendered manifests for the 16-DGD GPU fractions run |
| [`kai-gpu-fractions/smoke-test-half-gpu.yaml`](kai-gpu-fractions/smoke-test-half-gpu.yaml) | Two idle pods sharing one GPU at 50% each, to verify the caps before deploying models |
| [`kai-gpu-fractions/microk8s-gpu-fractioning-config.yaml`](kai-gpu-fractions/microk8s-gpu-fractioning-config.yaml) | Reference copy of the MicroK8s `GpuFractioningConfig` that the chart's socket-path flags produce; not meant to be applied |
| [`docs/`](docs/) | Step-by-step guides for each experiment and the fork build, plus [Per-GPU Compute and Memory Limits with MPS](docs/mps-per-gpu-limits.md) |

The checked-in `dgds-*.yaml` files are the generators' default output: namespace
`default`, the model at `/opt/hf-cache/hub/models--Qwen--Qwen3-4B/snapshots/<revision>`,
and no node pin. To deploy, run the generator with `NODE_NAME` set, as below, so that
every pod lands on the node that holds the model cache.
