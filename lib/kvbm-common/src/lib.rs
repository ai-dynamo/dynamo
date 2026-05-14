// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::str::FromStr;

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

/// Device backend type selector.
///
/// storage layer (`dynamo-memory`) and the device layer (`kvbm-physical`)
/// can reference a single canonical enum.
///
/// Runtime probes (`is_available`, `detect_backend`, `list_available`)
/// live in `kvbm-physical`'s `DeviceBackendExt` trait because they must
/// call into feature-gated backend modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceBackend {
    /// NVIDIA CUDA backend.
    Cuda,
    /// SYCL backend (Intel XPU via SYCL).
    Sycl,
}

impl DeviceBackend {
    /// Human-readable name for logs and diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Sycl => "SYCL (XPU)",
        }
    }
}

impl FromStr for DeviceBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" | "nvidia" => Ok(Self::Cuda),
            "sycl" | "xpu" | "intel" => Ok(Self::Sycl),
            _ => Err(format!("Unknown device backend: {s}")),
        }
    }
}
