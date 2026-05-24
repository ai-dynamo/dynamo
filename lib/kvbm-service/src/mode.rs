// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Service-mode helpers. The wire form is the proto enum
//! [`crate::proto::ServiceMode`]; this module provides a hand-derived Rust
//! enum that's nicer for serde and pattern matching.

use serde::{Deserialize, Serialize};

use crate::proto;

/// Which engine backend the registered tenant intends to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceMode {
    /// Native KVBM cache management.
    Kvbm,
    /// SGLang-shaped cache management.
    Sgl,
}

impl ServiceMode {
    pub fn from_proto(value: proto::ServiceMode) -> Self {
        match value {
            proto::ServiceMode::Kvbm => Self::Kvbm,
            proto::ServiceMode::Sgl => Self::Sgl,
        }
    }

    pub fn to_proto(self) -> proto::ServiceMode {
        match self {
            Self::Kvbm => proto::ServiceMode::Kvbm,
            Self::Sgl => proto::ServiceMode::Sgl,
        }
    }
}

impl std::fmt::Display for ServiceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kvbm => f.write_str("kvbm"),
            Self::Sgl => f.write_str("sgl"),
        }
    }
}
