// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM / TRT-LLM ZMQ ingress wire types (re-exports from `dynamo-kv-router::zmq_wire`).
//!
//! These are msgpack-over-ZMQ structures emitted by vLLM's Python `msgspec` codec.

pub use dynamo_kv_router::zmq_wire::{BlockHashValue, KvEventBatch, RawKvEvent};
