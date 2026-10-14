---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Common Setup
subtitle: Install the Dynamo platform and cache the model once for every GPU-sharing experiment
---

All three experiments (Baseline, KAI + HAMi, and KAI + GPU Fractions) start
from the same platform install and model cache. Do these steps once per cluster
before you deploy any experiment. Each experiment page lists its own cluster and
driver requirements, and the `GPU_NODE` variable it uses.

Run the commands from the root of a Dynamo repository checkout.

## Install the Dynamo Platform

```bash
cd examples/gpu-sharing
common/install-dynamo-platform.sh
```

The script installs the `dynamo-platform` Helm chart, version 1.4.2, into the `dynamo-system` namespace and waits for its pods to be Ready. It sets `global.etcd.install=true` and `global.nats.install=true`, which is deliberate. Platform 1.4.x installs without etcd and NATS by default and configures Kubernetes-based discovery, and with that default the `vllm-runtime:1.3.0` frontend never lists the model: `/v1/models` is empty and chat requests return 404. The generated DGDs set `DYN_DISCOVERY_BACKEND=etcd`, which needs the etcd and NATS the script installs.

Override `PLATFORM_NAMESPACE` or `PLATFORM_VERSION` to change the namespace or chart version.

## Download the Model to the GPU Node

```bash
NODE_NAME="$GPU_NODE" common/download-model.sh
```

The script runs a Kubernetes Job on `$GPU_NODE` that downloads `Qwen/Qwen3-4B` at the pinned revision `1cfa9a7208912126459214e8b04321603b3df60c` into a `hostPath` cache on that node, and prints the snapshot path. An init container running as root creates the directory and opens its permissions, because the download container, like the workers, runs as a non-root user. The workers then run fully offline from that cache.

| Variable | Default | Purpose |
|---|---|---|
| `NODE_NAME` | unset | Node to place the cache on. Set it to `$GPU_NODE` |
| `HF_CACHE_DIR` | `/opt/hf-cache` | `hostPath` cache directory on the node |
| `NAMESPACE` | `default` | Namespace for the download Job |
| `HF_TOKEN` | empty | Hugging Face token, if your network requires one |
| `RUNTIME_IMAGE` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` | Image that runs the download |

The model cache stays in `/opt/hf-cache` on `$GPU_NODE` across experiments, so
you only download it once.
