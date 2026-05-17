// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Block layout types and the [`KvBlocks`] collection wrapper.
//!
//! `BlockDim`, `KvBlockLayout`, and `InnerShape` live in `kvbm-common` so
//! kernels and label-inference code can use them without depending on
//! `kvbm-physical`. They are re-exported here for source-compat with
//! existing `crate::layout::KvBlockLayout` references.
//!
//! `KvBlocks` stays in this crate because it holds an `Arc<PhysicalLayout>`
//! and depends on `dynamo-memory`, which `kvbm-common` does not.

pub use kvbm_common::{BlockDim, InnerShape, KvBlockLayout};

use crate::BlockId;
use crate::layout::PhysicalLayout;
use std::sync::Arc;

/// A collection of blocks with a shared layout configuration and block layout type.
///
/// `KvBlocks` provides a convenient way to group blocks that should be treated
/// uniformly in transfer operations. All blocks in the collection share:
/// - The same [`PhysicalLayout`] (memory organization)
/// - The same [`KvBlockLayout`] interpretation (dimension ordering)
///
/// This enables efficient batch transfers with optional layout override.
///
/// # Example
///
/// ```ignore
/// // Create blocks with universal layout override
/// let blocks = KvBlocks::new(
///     physical_layout.clone(),
///     vec![0, 1, 2, 3],  // block IDs
///     Some(KvBlockLayout::Universal),
/// )?;
///
/// // Use in transfers - the override tells the transfer system
/// // to interpret these blocks as universal format
/// ```
#[derive(Debug, Clone)]
pub struct KvBlocks {
    /// The physical layout containing these blocks
    layout: Arc<PhysicalLayout>,
    /// Block IDs within the layout
    block_ids: Vec<BlockId>,
    /// Optional layout override (None = use layout's native block_layout)
    kv_layout_override: Option<KvBlockLayout>,
}

impl KvBlocks {
    /// Create a new KvBlocks collection.
    ///
    /// # Arguments
    /// * `layout` - The physical layout containing the blocks
    /// * `block_ids` - Block IDs to include in this collection
    /// * `kv_layout_override` - Optional override for the block layout interpretation.
    ///   If `None`, uses the layout's native `block_layout()`.
    ///   If `Some`, overrides the interpretation for transfers.
    ///
    /// # Validation
    /// - For layer-separate layouts, only operational layouts (NHD/HND) are valid overrides
    /// - For fully contiguous layouts, any layout is valid
    /// - If the override matches the native layout, it is normalized to None
    pub fn new(
        layout: Arc<PhysicalLayout>,
        block_ids: Vec<BlockId>,
        kv_layout_override: Option<KvBlockLayout>,
    ) -> anyhow::Result<Self> {
        // Validate block IDs are in range
        let num_blocks = layout.layout().num_blocks();
        for &id in &block_ids {
            if id >= num_blocks {
                return Err(anyhow::anyhow!(
                    "Block ID {} out of range (layout has {} blocks)",
                    id,
                    num_blocks
                ));
            }
        }

        // Validate layout override compatibility
        if let Some(ref override_layout) = kv_layout_override {
            // Layer-separate layouts can only use operational formats
            if !layout.layout().is_fully_contiguous() && !override_layout.is_operational() {
                return Err(anyhow::anyhow!(
                    "Layer-separate layouts only support operational block layouts (NHD/HND), got {:?}",
                    override_layout
                ));
            }
        }

        // Normalize: if override matches native layout, set to None
        let normalized_override = kv_layout_override.and_then(|override_layout| {
            if override_layout == layout.layout().block_layout() {
                None
            } else {
                Some(override_layout)
            }
        });

        Ok(Self {
            layout,
            block_ids,
            kv_layout_override: normalized_override,
        })
    }

    /// Create a KvBlocks collection without layout override.
    #[expect(dead_code)]
    pub fn from_layout(
        layout: Arc<PhysicalLayout>,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<Self> {
        Self::new(layout, block_ids, None)
    }

    /// Get the physical layout.
    #[expect(dead_code)]
    pub fn layout(&self) -> &Arc<PhysicalLayout> {
        &self.layout
    }

    /// Get the block IDs.
    #[expect(dead_code)]
    pub fn block_ids(&self) -> &[BlockId] {
        &self.block_ids
    }

    /// Get the effective block layout (override or native).
    pub fn effective_block_layout(&self) -> KvBlockLayout {
        self.kv_layout_override
            .unwrap_or_else(|| self.layout.layout().block_layout())
    }

    /// Get the layout override if set.
    #[expect(dead_code)]
    pub fn layout_override(&self) -> Option<KvBlockLayout> {
        self.kv_layout_override
    }

    /// Check if this collection has a layout override.
    #[expect(dead_code)]
    pub fn has_override(&self) -> bool {
        self.kv_layout_override.is_some()
    }

    /// Get the number of blocks in this collection.
    #[expect(dead_code)]
    pub fn len(&self) -> usize {
        self.block_ids.len()
    }

    /// Check if the collection is empty.
    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Check if a transfer between two KvBlocks collections requires transformation.
    ///
    /// Returns `true` if the effective layouts differ and a transformation kernel
    /// is needed rather than a simple copy.
    #[expect(dead_code)]
    pub fn requires_transform_to(&self, dst: &KvBlocks) -> bool {
        self.effective_block_layout()
            .requires_transform(&dst.effective_block_layout())
    }
}
