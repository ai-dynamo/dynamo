// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public entry point for embedding Dynamo's real KV Router placement in an
//! external AISimulate Replayer.
//!
//! These are the existing Dynamo-owned composition types; nothing here copies
//! Router logic. A host builds a [`KvReplayComposition`] and hands it to the
//! AISimulate Replayer, which drives [`KvRouterPlacement`] through the neutral
//! `ReplayComposition` / `PlacementPolicy` contract.

pub use crate::config::MockerConfig;
pub use crate::replay::ReplayPrefillLoadEstimator;
pub use crate::replay::offline::extensions::kv_events::{RouterEventBatch, RouterEventObservation};
pub use crate::replay::offline::extensions::kv_router::{
    KvReplayComposition, KvReplayMetadata, KvRouterPlacement,
};
pub use dynamo_kv_router::config::KvRouterConfig;
