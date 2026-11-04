// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How `PushRouter` picks an instance for its routing modes.
//!
//! The runtime ships no selection algorithm for round-robin, random or least-loaded. Whoever builds a `PushRouter` passes [`ModeSelectors`] in
//! [`PushRouterOptions`](crate::pipeline::PushRouterOptions); Dynamo's hosts back them with
//! the router-mode plugins.

use std::sync::Arc;

use crate::pipeline::RouterMode;

/// Picks one instance for a `PushRouter` routing mode.
///
/// `PushRouter` calls [`select_instance`](Self::select_instance) inside its admission lock
/// when [`reads_active_requests`](Self::reads_active_requests) is true, so it must not block.
pub trait InstanceSelector: Send + Sync {
    /// Whether selection reads in-flight request counts. `PushRouter` then tracks them and
    /// reserves the chosen instance.
    fn reads_active_requests(&self) -> bool;

    /// Pick one of `instance_ids`, which are ascending and non-empty. A preview must not
    /// advance any state, such as a rotation.
    fn select_instance(
        &self,
        instance_ids: &[u64],
        active_requests: &dyn Fn(u64) -> u64,
        preview: bool,
    ) -> Option<u64>;
}

/// Builds a fresh selector for a routing mode, or `None` when it has none for that mode.
pub type ModeSelectors = Arc<dyn Fn(RouterMode) -> Option<Arc<dyn InstanceSelector>> + Send + Sync>;
