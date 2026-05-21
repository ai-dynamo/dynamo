// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod accessor;
mod blocks;
pub mod consolidator;
pub mod control;
mod describe_map;
pub mod discovery;
pub mod dispatch;
mod instance;
pub mod layout_compat;
mod onboarding;
pub mod parallelism;
mod remote_search;
pub mod search;
mod staging;
mod state;
mod transport;
mod types;
pub mod velo;

pub use accessor::{BlockAccessor, PolicyContext, TieredBlock};
pub use blocks::BlockHolder;
pub use consolidator::ConsolidatorParams;
pub use control::{ControlModule, ControlPlane, ControlPlaneBuilder};
pub use discovery::{RemoteBlockDiscovery, RemoteCandidates, RemoteDiscoveryHandle};
pub use instance::InstanceLeader;
pub use kvbm_consolidator::{ConsolidatorHandle, EventSource};
pub use onboarding::*;
pub use staging::{StagingResult, stage_g3_to_g2};
pub use state::{LeaderState, RemoteLeaderInfo, route_local_to_remote};
pub use transport::MetadataTransport;
pub use types::*;
pub use velo::VeloLeaderService;

use anyhow::Result;

use crate::SequenceHash;

/// Leader trait for distributed block onboarding operations.
pub trait Leader: Send + Sync {
    /// Find matching blocks with default options.
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<FindMatchesResult> {
        self.find_matches_with_options(sequence_hashes, FindMatchesOptions::default())
    }

    /// Find matching blocks with custom options.
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<FindMatchesResult>;
}
