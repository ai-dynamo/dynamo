// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager V2 - EXPERIMENTAL
//!
//! A completely redesigned block management system with:
//! - Type-safe state transitions (Reset → Complete → Registered)
//! - Async batched return processing with controllable stepping
//! - Compile-time prevention of accessing registered mutable blocks
//! - Comprehensive testing support for race conditions
//!
//! NOTE: This module is currently experimental and under development.
//! It implements a simplified Block<T, State> API that differs from the
//! main codebase's Block<Storage, LocalityProvider, Metadata> API.

pub mod distributed;
pub mod logical;
pub mod physical;
pub mod runtime;
pub mod utils;

pub mod config;
pub mod types;

// Hub for distributed block coordination
// pub mod hub;

// // Integration modules for external frameworks
pub mod integrations;

// Re-export common types and traits
// pub use config::{AttentionConfig, ParallelConfig};
pub use runtime::{KvbmRuntime, KvbmRuntimeBuilder, RuntimeHandle};
pub use types::{CacheDtype, CacheLayout, ModelExecutorBackend};

// Test infrastructure
#[cfg(test)]
pub mod test_config;

// Testing utilities (available for tests and when testing feature is enabled)
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use dynamo_identity::InstanceId;
pub use dynamo_kvbm_config::KvbmConfig;
pub use dynamo_nova::{PeerInfo, WorkerAddress};
pub use dynamo_tokens::{PositionalLineageHash, SequenceHash as SequenceHashV1};

pub type BlockId = usize;
pub type SequenceHash = PositionalLineageHash;

/// G1 marker: GPU/device tier
#[derive(Clone, Copy, Debug)]
pub struct G1;

/// G2 marker: CPU/host tier
#[derive(Clone, Copy, Debug)]
pub struct G2;

/// G3 marker: Disk tier
#[derive(Clone, Copy, Debug)]
pub struct G3;

/// G4 marker: Object store tier
#[derive(Clone, Copy, Debug)]
pub struct G4;

pub trait KvbmSequenceHashProvider {
    fn kvbm_sequence_hash(&self) -> SequenceHash;
}

impl KvbmSequenceHashProvider for dynamo_tokens::TokenBlock {
    fn kvbm_sequence_hash(&self) -> SequenceHash {
        self.positional_lineage_hash()
    }
}
