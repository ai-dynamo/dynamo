// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical block lifecycle management for KVBM.
//!
//! This crate provides the core block lifecycle system:
//! - Type-safe state transitions (Reset -> Complete -> Registered)
//! - Block registry with deduplication and attachments
//! - Active/inactive/reset pool management
//! - Event pipeline for distributed coordination
//! - Block manager orchestration

pub mod blocks;
pub mod events;
pub mod manager;
pub mod pools;
pub mod pubsub;
pub mod registry;
pub mod tinylfu;

#[cfg(any(test, feature = "testing"))]
pub mod test_config;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Re-export common types and traits
pub use registry::BlockRegistry;
pub use blocks::{
    BlockError, BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock,
    WeakBlock,
};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

pub use dynamo_tokens::{PositionalLineageHash, SequenceHash as SequenceHashV1};

pub type BlockId = usize;
pub type SequenceHash = PositionalLineageHash;

pub trait KvbmSequenceHashProvider {
    fn kvbm_sequence_hash(&self) -> SequenceHash;
}

impl KvbmSequenceHashProvider for dynamo_tokens::TokenBlock {
    fn kvbm_sequence_hash(&self) -> SequenceHash {
        self.positional_lineage_hash()
    }
}

/// Logical layout handle type encoding the layout ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize, Deserialize)]
pub enum LogicalLayoutHandle {
    G1,
    G2,
    G3,
    G4,
}
