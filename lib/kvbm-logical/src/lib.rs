// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod blocks;
pub mod events;
pub mod manager;
pub mod metrics;
pub mod pools;
pub mod pubsub;
pub mod registry;
pub mod tinylfu;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

// Re-export common types and traits
pub use blocks::{
    BlockError, BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock, WeakBlock,
};
pub use manager::BlockManager;
pub use registry::BlockRegistry;

pub type BlockId = usize;
pub type SequenceHash = dynamo_tokens::PositionalLineageHash;

pub trait KvbmSequenceHashProvider {
    fn kvbm_sequence_hash(&self) -> SequenceHash;
}

impl KvbmSequenceHashProvider for dynamo_tokens::TokenBlock {
    fn kvbm_sequence_hash(&self) -> SequenceHash {
        self.positional_lineage_hash()
    }
}

/// Logical layout handle type encoding the layout ID.
///
/// KVBM manages G1, G2 and G3 layouts directly. G4 is managed by an external service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize, Deserialize)]
pub enum LogicalLayoutHandle {
    /// Representation of GPU / Device Memory
    G1,
    /// Representation of CPU / Host Memory
    G2,
    /// Representation of Disk Storage
    G3,
    /// Representation of Blocks held in an external service
    /// outside the control of the KVBM system.
    G4,
}
