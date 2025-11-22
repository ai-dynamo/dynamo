// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod session;

pub use session::OnboardingSession;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    physical::{manager::LayoutHandle, transfer::TransferCompleteNotification},
    v2::{BlockId, InstanceId, SequenceHash, logical::LogicalLayoutHandle},
};

pub trait Leader: Send + Sync {
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<OnboardingSession> {
        self.find_matches_with_options(sequence_hashes, FindMatchesOptions::default())
    }

    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<OnboardingSession>;
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FindMatchesOptions {}
