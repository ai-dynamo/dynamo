// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

pub mod blocks;
pub mod events;
pub mod manager;
pub mod pools;

// pub mod executor;

// Re-export for public use
pub use blocks::{
    BlockError, BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock, WeakBlock,
};

pub use super::BlockId;

pub use super::SequenceHash;

/// Logical layout handle type encoding the layout ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize, Deserialize)]
pub enum LogicalLayoutHandle {
    G1,
    G2,
    G3,
    G4,
}

#[cfg(test)]
pub(crate) mod tests;
