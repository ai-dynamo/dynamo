---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Writing Unified Backends
subtitle: Choose Python or Rust for Dynamo's shared backend contract
---

Dynamo's unified backend path lets custom engines implement the same lifecycle
contract used by the built-in backends. The engine owns inference; Dynamo owns
runtime registration, request serving, cancellation monitoring, signal handling,
drain, and graceful shutdown.

Use this path for new token-in-token-out engines unless you need a feature that
is still outside the unified contract.

## Choose an Implementation Language

| Path | Use when | Start here |
|---|---|---|
| Python unified backend | Your engine or serving library is Python-first, or you want the quickest path to integrate a custom engine. | [Python implementation](python-backend-guide.md) |
| Rust unified backend | You want a native Rust binary, tighter control of runtime dependencies, or no Python worker runtime. | [Rust implementation](rust-backend-guide.md) |
| Python workers (lower-level) | You need custom request handling or features not yet covered by the unified backend contract. | [Python workers](backend-guide.md) |

Both unified implementations follow the same shape:

```text
parse config -> start engine -> stream generated chunks -> abort/drain -> cleanup
```

The framework handles model registration, endpoint serving, cancellation
plumbing, and shutdown behavior around that engine contract.

## What the Unified Contract Covers

Supported today:

- aggregated token-in-token-out inference
- disaggregated serving modes for supported engines
- model registration through Dynamo discovery
- request cancellation
- structured backend errors
- graceful shutdown and drain hooks

Still use the lower-level Python worker path when you need features such as
multimodal requests, LoRA adapter management, logprobs, guided decoding,
engine-specific routes, custom request handling, or features that need direct
control of the request payload.

## Package and Deploy

After you implement the backend, package it into a runtime image with
[Runtime Containers](custom-containers.md). For Kubernetes deployment, place the
custom backend in a `DynamoGraphDeployment` and follow the
[Deployment Overview](../kubernetes/model-deployment-guide.md).
