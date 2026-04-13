// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

pub type BlockId = usize;
pub type SequenceHash = dynamo_tokens::PositionalLineageHash;

pub use dynamo_tokens as tokens;

/// Logical layout handle type encoding the layout ID.
///
/// KVBM manages G1, G2 and G3 layouts directly. G4 is managed by an external service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicalLayoutHandle {
    /// Representation of GPU / Device Memory
    /// G1 is fixed sized and managed by either the framework or the local instance of KVBM.
    G1,
    /// Representation of CPU / Host Memory
    /// G2 is fixed sized and managed by the local instance of KVBM.
    G2,
    /// Representation of Disk Storage (Local or AttachedStorage)
    /// G3 is fixed sized and managed by the local instance of KVBM.
    G3,
    /// Representation of Blocks held in an external service
    /// outside the control of the KVBM system.
    G4,
}

/// Compatibility transfer routes used by KVBM metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvbmTransferRoute {
    OffloadD2H,
    OffloadH2D,
    OffloadD2D,
    OffloadD2O,
    OnboardH2D,
    OnboardD2D,
    OnboardO2D,
}

impl KvbmTransferRoute {
    /// Stable label used by additive metrics.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::OffloadD2H => "d2h",
            Self::OffloadH2D => "h2d",
            Self::OffloadD2D => "d2d",
            Self::OffloadD2O => "d2o",
            Self::OnboardH2D => "h2d",
            Self::OnboardD2D => "d2d",
            Self::OnboardO2D => "o2d",
        }
    }

    /// Stable operation label used by additive metrics.
    pub fn operation_label(&self) -> &'static str {
        match self {
            Self::OffloadD2H | Self::OffloadH2D | Self::OffloadD2D | Self::OffloadD2O => "offload",
            Self::OnboardH2D | Self::OnboardD2D | Self::OnboardO2D => "onboard",
        }
    }
}
