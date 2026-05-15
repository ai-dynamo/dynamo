// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration modules for external frameworks.
//!
//! This module provides trait-based abstractions for integrating with
//! external serving frameworks like vLLM, allowing pure Rust code to
//! remain independent of framework-specific types.

pub mod common;
pub mod config;
pub mod connector;
pub mod vllm;

// Re-export key types for convenience
pub use common::{
    CachedRequestData, NewRequestData, Request, RequestMetadata, SchedulerConnectorState,
    SchedulerOutput,
};
pub use config::{AttentionConfig, IntegrationsConfig, ParallelConfig};

// Re-export workspace types used throughout this crate
pub use kvbm_common::{BlockId, SequenceHash};
pub use kvbm_engine::{G1, G2, G3, G4, InstanceId, KvbmRuntime};

// Re-exports for bindings — runtime construction
pub use kvbm_config::KvbmConfig;
pub use kvbm_engine::{KvbmRuntimeBuilder, PeerInfo, WorkerAddress};

// Re-exports for bindings — memory/tensor types (already in public API via ConnectorWorkerInterface)
pub use dynamo_memory::{MemoryDescriptor, StorageKind, TensorDescriptor};
pub mod memory {
    //! Re-exports from `dynamo-memory` for bindings convenience.
    pub use dynamo_memory::nixl;
}

#[cfg(feature = "testing")]
pub mod testing;
