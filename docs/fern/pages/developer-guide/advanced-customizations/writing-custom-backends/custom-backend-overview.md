---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Backend Overview
subtitle: Choose the right path for bringing your own engine to Dynamo
---

Dynamo supports custom backends through one preferred unified contract, a
lower-level worker path, and a packaging path:

| Path | Use when |
| --- | --- |
| [Writing Unified Backends](writing-unified-backends.md) | You are writing a new token-in-token-out engine in Python or Rust and want Dynamo to own the runtime lifecycle. |
| [Python Workers (lower-level)](writing-python-workers.md) | You need direct control of `register_model`, `serve_endpoint`, or the request payload. |
| [Runtime Containers](runtime-containers.md) | You need to package a built-in or custom backend into a deployable Dynamo image. |

The unified backend path is the preferred starting point for new custom engines.
It gives Python and Rust backends the same lifecycle shape: parse arguments,
start the engine, stream generated chunks, handle cancellation, drain, and clean
up. The Dynamo framework owns runtime registration, signal handling, model
registration, and graceful shutdown.

Use the lower-level Python worker path when your backend must own model
registration, endpoint serving, request handling, or lifecycle behavior.

If your custom engine uses KV-aware routing, [publish KV events](publish-kv-events.md)
so the Dynamo router can track which workers hold each prefix.
