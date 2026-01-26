// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
pub mod notifications;
pub mod options;
pub mod preferences;
pub mod strategy;
pub mod validation;

#[cfg(test)]
mod tests;

// Re-export StorageKind
pub use dynamo_memory::{StorageKind, pool as cuda_pool};

pub use capabilities::TransferCapabilities;
pub use checksum::{BlockChecksum, compute_block_checksums, compute_layer_checksums};
pub use context::{TransferCompleteNotification, TransferConfig};
pub use dynamo_memory::nixl::{NixlAgent, NixlBackendConfig};
pub use fill::{FillPattern, fill_blocks, fill_layers};
pub use options::{TransferOptions, TransferOptionsBuilder};
pub use preferences::{NativeVsNixlPolicy, TransferPreferences};
pub use strategy::{TransferPlan, TransferStrategy};
pub use validation::BlockValidationError;

// Internal - TransferContext is now managed by TransferManager
pub(crate) use context::TransferContext;

use crate::BlockId;

pub use super::layout::PhysicalLayout;

// Re-export manager types - TransferManager is the primary public API
pub use super::manager::{LayoutHandle, SerializedLayout, TransferManager, WorkerAddress};

// #[cfg(test)]
// pub use testing::{RoundTripTest, RoundTripTestResult};

// /// Specification for bounce buffer in multi-hop transfers.
// ///
// /// This structure provides the layout and block IDs to use as an intermediate
// /// staging area when direct transfers are not allowed.
// #[deprecated(since = "2025-11-25", note = "use TransferOptions instead")]
// pub trait BounceBufferSpec: Send + Sync {
//     fn layout(&self) -> &PhysicalLayout;
//     fn block_ids(&self) -> &[BlockId];
// }

#[derive(Clone)]
pub enum BounceBufferLayout {
    Layout(PhysicalLayout),
    Handle(LayoutHandle),
}

#[derive(Clone)]
pub struct BounceBuffer {
    layout: LayoutHandle,
    block_ids: Vec<BlockId>,
}

#[derive(Clone)]
pub struct BounceBufferInternal {
    layout: PhysicalLayout,
    block_ids: Vec<BlockId>,
}

impl BounceBuffer {
    pub fn from_handle(layout: LayoutHandle, block_ids: Vec<BlockId>) -> Self {
        Self { layout, block_ids }
    }

    pub(crate) fn into_parts(self) -> (LayoutHandle, Vec<BlockId>) {
        (self.layout, self.block_ids)
    }
}

impl BounceBufferInternal {
    pub fn from_layout(layout: PhysicalLayout, block_ids: Vec<BlockId>) -> Self {
        Self { layout, block_ids }
    }
}

// ============================================================================
// Layout Compatibility Helpers
// ============================================================================

use anyhow::anyhow;
use std::ops::Range;

/// Validate that layouts are compatible for transfer.
///
/// Returns an error if layouts require transformation, which is not yet supported.
/// This should be called early in transfer execution to fail fast.
pub fn validate_layout_compatibility(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
) -> anyhow::Result<()> {
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.block_layout().requires_transform(&dst_layout.block_layout()) {
        return Err(anyhow!(
            "Layout transformation not supported: src={:?}, dst={:?}",
            src_layout.block_layout(),
            dst_layout.block_layout()
        ));
    }

    Ok(())
}

/// Check if layouts support whole-block transfers.
///
/// Returns true when:
/// - Both src and dst are fully contiguous
/// - Transfer is full-block (layer_range covers all layers or is None)
///
/// Note: Caller must have already validated layout compatibility via
/// [`validate_layout_compatibility`].
pub fn can_use_whole_block_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    layer_range: Option<&Range<usize>>,
) -> bool {
    // Must be full-block transfer
    let is_full_block = match layer_range {
        None => true,
        Some(range) => range.start == 0 && range.end == src.layout().num_layers(),
    };
    if !is_full_block {
        return false;
    }

    // Both must be fully contiguous
    src.layout().is_fully_contiguous() && dst.layout().is_fully_contiguous()
}
