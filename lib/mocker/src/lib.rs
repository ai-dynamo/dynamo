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
/// `ReplayComposition` that wires it into a `Replayer` and the provider
/// identifier a `ReplaySpec` must name for that composition to accept it.
///
/// This is one import site for everything needed to drive the router, so a
/// caller does not assemble it from several paths at different stability.
///
/// The `kv_events`/`kv_router` re-exports below are the load-bearing ones:
/// their module chain is `pub(crate)`, so this module is their only public
/// path and the tree underneath stays free to move. The rest are already
/// public by their own paths (`common::protocols`, `replay`,
/// `dynamo_kv_router::config`) and appear here for convenience, not access.
pub mod placement {
    /// Router tuning the composition accepts, so a caller can vary scoring
    /// (overlap weight, temperature, queue policy) without reaching into
    /// dynamo-kv-router directly.
    pub use dynamo_kv_router::config::KvRouterConfig;

    /// The engine configuration both constructors take. Every field the
    /// router's predicted-load model reads comes from here, so a caller has to
    /// build one to construct either the policy or the composition.
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
