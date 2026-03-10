// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport abstraction for publishing batched KV cache events.
//!
//! Implementations handle the actual delivery mechanism (NATS event plane,
//! JetStream durable queue, direct indexer application, etc.). The trait lives
//! in this crate so that the batching processor and other routing logic can be
//! written generically; runtime-specific impls stay in `lib/llm`.

use std::future::Future;

use crate::protocols::RouterEvent;

/// Transport abstraction for publishing batched KV cache events.
pub trait EventSink: Send + Sync {
    fn publish_event(&self, event: &RouterEvent)
    -> impl Future<Output = anyhow::Result<()>> + Send;
}
