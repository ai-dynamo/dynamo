<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-events

`dynamo-kv-events` defines the KV cache events engines publish when they store, remove, or clear blocks, and the `RouterEvent` envelope the router consumes them in. It also holds the msgpack codec for engine ZMQ streams (`zmq_wire`), the replay cursor a subscriber uses to resume a stream, and the residency vocabulary (storage tiers, domains, and owners). It opens no sockets; the ZMQ listener lives in `dynamo-kv-router` behind its `ingest-zmq` feature.
