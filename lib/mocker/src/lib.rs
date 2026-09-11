// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock LLM scheduler and KV manager for testing.
//!
//! This crate provides a mock implementation of an LLM scheduler that simulates
//! KV cache management, request scheduling, and token generation timing without
//! requiring actual GPU resources or a full distributed runtime.

pub mod common;
pub mod engine;
pub(crate) mod engine_adapter;
pub(crate) mod engine_observations;
pub(crate) mod generalized_live;
pub mod grouped_scheduler;
pub mod live;
pub mod loadgen;
pub mod replay;
pub mod scheduler;
pub mod services;
pub mod sglang;

/// The Dynamo KV router as an aisimulate-core `PlacementPolicy`, plus the
/// `ReplayComposition` that wires it into a `Replayer`. The `kv_events`/
/// `kv_router` re-exports are the load-bearing ones -- their module chain is
/// `pub(crate)`, so this is their only public path. The rest are already
/// public elsewhere and are re-exported here for a single import site.
pub mod placement {
    pub use dynamo_kv_router::config::KvRouterConfig;

    pub use crate::common::protocols::{MockEngineArgs, MockEngineArgsBuilder};
    pub use crate::replay::ReplayPrefillLoadEstimator;
    pub use crate::replay::offline::extensions::kv_events::{
        RouterEventBatch, RouterEventObservation,
    };
    pub use crate::replay::offline::extensions::kv_router::composition::KvReplayComposition;
    pub use crate::replay::offline::extensions::kv_router::{
        KvReplayMetadata, KvRouterPlacement, provider_spec,
    };
}
