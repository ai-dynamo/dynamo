<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-events

Everything about KV cache events, for both ends of the wire: the event types and the `RouterEvent` envelope (`event`), the engine ZMQ msgpack codec and its filters (`zmq_wire`), the replay cursor a subscriber keeps in step with a publisher (`cursor`), and the publisher-side placement helpers (`placement`). Workers, sidecars, and the C bindings publish with this crate; the router's indexer ingests with it.

Dependencies: `dynamo-router-protocol` (identities, residency vocabulary, block-hash newtypes, multimodal block info), `dynamo-kv-hashing` (the block hash contract the codec recomputes), serde, serde_json, rmp-serde, rustc-hash, thiserror, tracing. No sockets, tasks, or servers: the ZMQ socket is `dynamo-zmq` and the listener that drives it is a `dynamo-kv-router` adapter. The layering gate (`scripts/router_layering_check.sh`) checks the dependency set.

Part of the router split (see `lib/router/README.md`).
