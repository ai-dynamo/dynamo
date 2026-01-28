// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type GlobalHash = u64;

/// Represents an active block beign built
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum UniqueBlock {
    /// Block identified by UUID
    PartialBlock(Uuid),
    /// Block identified by hash
    FullBlock(GlobalHash),
}

impl Default for UniqueBlock {
    fn default() -> Self {
        // Generate a random UUID when default is used
        Self::PartialBlock(Uuid::new_v4())
    }
}
