---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes"
subtitle: "[Schwinn Saereesitthipitak](https://developer.nvidia.com/blog/author/schwinnsaereesitthipitak/), [Dan Feigin](https://developer.nvidia.com/blog/author/danfeigin/), [Vikram Sharma Mailthody](https://developer.nvidia.com/blog/author/vmailthody/) and [Maksim Khadkevich](https://developer.nvidia.com/blog/author/mkhadkevich/) — May 2026"
description: "NVIDIA Dynamo Snapshot combines CUDA and host checkpointing to restore warm inference workers quickly on Kubernetes."
keywords: NVIDIA Dynamo Snapshot, Dynamo, Kubernetes, checkpoint restore, CRIU, cuda-checkpoint, GPU Memory Service, inference startup
last-updated: May 28, 2026
---

![Kubernetes checkpoint and restore lifecycle with NVIDIA Dynamo Snapshot.](./dynamo-snapshot-lifecycle.webp)

Cold-starting inference replicas on Kubernetes can take minutes while engines load weights, warm kernels, and compile graphs. In our blog post, [NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/), we introduce Dynamo Snapshot, a checkpoint/restore approach that combines `cuda-checkpoint`, CRIU, and a privileged `snapshot-agent` DaemonSet to restore warm workers from shared storage. We also walk through KV cache unmapping, CRIU restore optimizations, and GPU Memory Service (GMS), which bring the `gpt-oss-120b` prototype below five seconds and reduce startup time by 21x.
