// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod session_aware;

use serde::Deserialize;

use crate::protocols::WorkerWithDpRank;

pub use session_aware::SessionAwareConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchIntent {
    Any,
    Exact(WorkerWithDpRank),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueueAdmissionConfig {
    SessionAware(SessionAwareConfig),
}

impl QueueAdmissionConfig {
    pub(crate) fn validate(&self, location: &str) -> Result<(), String> {
        match self {
            Self::SessionAware(config) => config.validate(location),
        }
    }
}
