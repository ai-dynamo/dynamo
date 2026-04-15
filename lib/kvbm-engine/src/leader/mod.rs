// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod accessor;
mod instance;
mod onboarding;
#[doc = include_str!("../../docs/session.md")]
pub mod session;
mod state;
mod types;
pub mod velo;

pub use accessor::{BlockAccessor, PolicyContext, TieredBlock};
pub use instance::InstanceLeader;
pub use onboarding::*;
pub use session::{
    ControllableSessionOptions, ControllableSessionResult, InitiatorSession, ResponderSession,
    ServerSession, ServerSessionHandle, ServerSessionOptions, SessionId,
};
pub use state::{LeaderState, RemoteLeaderInfo, route_local_to_remote};
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
