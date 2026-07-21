// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod admission;
mod engine;
mod types;
mod worker_core;

#[cfg(test)]
pub(crate) use super::extensions::kv_router::OfflineRouterSnapshot;
pub(in crate::replay::offline) use admission::{
    AdmissionQueue, KvReplayMetadata, NoReplayMetadata, ReplayAdmissionMetadata,
};
pub(in crate::replay::offline) use engine::EngineComponent;
pub(in crate::replay) use types::ReplayMode;
pub use types::TrafficStats;
pub(in crate::replay::offline) use types::{
    EngineEffects, EnginePassMode, ScheduledWorkerCompletion, TrafficAccumulator,
};
pub(crate) use worker_core::ReplayWorkerCore;
