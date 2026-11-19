<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (intro, support matrix, launch
example, Kubernetes example). Do not edit it unless the user explicitly asks
you to.
-->

# Sidecar common

Shared infrastructure for Rust sidecars:

- gRPC transport arguments and defaults
- plaintext endpoint validation
- connection pooling and startup retries
- gRPC-to-Dynamo error mapping

Engine protocols and request conversion stay in each sidecar crate.
