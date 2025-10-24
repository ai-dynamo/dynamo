// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer module for copying blocks between layouts with different storage locations.
//!
//! This module provides functionality for transferring KV cache blocks between layouts
//! that may be backed by different storage types (GPU memory, pinned host memory, disk, etc.)
//! and potentially across NIXL-connected remote nodes.
//!
//! # Core Concepts
//!
//! - [`PhysicalLayout`]: Wraps a layout with its physical storage location and NIXL metadata
//! - [`LayoutDescriptor`]: Serializable representation for cross-node communication
//! - Transfer strategies: memcpy, CUDA, NIXL based on source/destination locations
//! - Block-wise and layer-wise transfer operations
//!
//! # Usage
//!
//! ```rust,ignore
//! use dynamo_kvbm::v2::transfer::{PhysicalLayout, transfer_blocks};
//!
//! // Create local physical layout with NIXL registration
//! let src = PhysicalLayout::new_local(src_layout, StorageKind::Device(0))
//!     .with_nixl_registration("local_agent".to_string())?;
//!
//! // Create remote physical layout
//! let dst = PhysicalLayout::new_remote(
//!     dst_layout,
//!     StorageKind::Pinned,
//!     "remote_agent".to_string()
//! );
//!
//! // Transfer blocks from local to remote
//! let src_block_ids = [0, 1, 2];
//! let dst_block_ids = [0, 1, 2];
//! let future = transfer_blocks(&src, &dst, &src_block_ids, &dst_block_ids, &ctx)?;
//! future.await?;
//! ```

pub mod capabilities;
pub mod checksum;
pub mod context;
pub mod executor;
pub mod fill;
pub mod nixl_agent;
pub mod notifications;
pub mod options;
pub mod preferences;
pub mod strategy;
pub mod validation;

#[cfg(test)]
mod tests;

// Re-export StorageKind
pub use crate::block_manager::v2::memory::StorageKind;

pub use capabilities::TransferCapabilities;
pub use checksum::{BlockChecksum, compute_block_checksums, compute_layer_checksums};
pub use fill::{FillPattern, fill_blocks, fill_layers};
pub use nixl_agent::{NixlAgent, NixlBackendConfig};
pub use options::{TransferOptions, TransferOptionsBuilder};
pub use preferences::{NativeVsNixlPolicy, TransferPreferences};
pub use strategy::{TransferPlan, TransferStrategy};
pub use validation::BlockValidationError;

// Internal - TransferContext is now managed by TransportManager
pub(crate) use context::TransferContext;

pub use super::layout::PhysicalLayout;

// Re-export manager types - TransportManager is the primary public API
pub use super::manager::{LayoutHandle, SerializedLayout, TransportManager, WorkerAddress};

// #[cfg(test)]
// pub use testing::{RoundTripTest, RoundTripTestResult};

use anyhow::Result;

/// Future representing an in-progress transfer operation.
///
/// The transfer completes when this future resolves.
pub type TransferFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>;

/// Specification for bounce buffer in multi-hop transfers.
///
/// This structure provides the layout and block IDs to use as an intermediate
/// staging area when direct transfers are not allowed.
pub trait BounceBufferSpec: Send + Sync {
    fn layout(&self) -> &PhysicalLayout;
    fn block_ids(&self) -> &[usize];
}

// #[cfg(all(test, feature = "testing-cuda"))]
// mod cuda_integration_tests {
//     use super::*;
//     use crate::block_manager::v2::layout::{
//         FullyContiguousLayout, Layout, LayoutConfig, MemoryRegion, OwnedMemoryRegion,
//     };
//     use cudarc::driver::CudaContext;
//     use std::sync::Arc;

//     // TODO: Add CUDA-specific integration tests
//     // These would test:
//     // - H2D transfers
//     // - D2H transfers
//     // - D2D transfers
//     // - Async completion via event synchronization
// }

// #[cfg(all(test, feature = "testing-nixl"))]
// mod nixl_integration_tests {
//     use super::*;

//     // TODO: Add NIXL-specific integration tests
//     // These would test:
//     // - Remote memory access via NIXL Read
//     // - Disk-backed transfers via NIXL Write
//     // - Cross-node serialization with LayoutDescriptor
// }
