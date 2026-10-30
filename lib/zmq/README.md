<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-zmq

One ZMQ socket helper: an async `ZmqSocket` over `tokio::io::unix::AsyncFd`, multipart send and receive, a process-wide context pool sized by `DYN_ZMQ_IO_THREADS`, the SUB, DEALER, and bound PUB constructors, and `validate_endpoint`. The router's ZMQ adapters (indexer listener, replica mesh, direct ingress) and the KV event publishers use it. It knows nothing about KV events or routing; the frame codec is `dynamo-kv-events`.

Dependencies: zmq, tokio, tracing, anyhow (the error type is inherited from the code's previous home and will become a `thiserror` enum when its callers move into the router adapter layer). No router crate depends on it except `dynamo-kv-router` behind its ZMQ features.

Part of the router split (see `lib/router/README.md`).
