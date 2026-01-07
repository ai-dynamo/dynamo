// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Block layout types for describing dimension permutations within blocks.
//!
//! This module provides types for describing how dimensions are ordered within
//! a fully contiguous KV cache block, enabling type-driven kernel selection
//! for transfers between different layout formats.

use serde::{Deserialize, Serialize};

/// Symbolic dimensions that can be permuted within a block.
///
/// The head dimension (hd) is always innermost and not included here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockDim {
    /// Number of layers (nl)
    Layer,
    /// Outer dimension - typically 2 for K/V, 1 for MLA (no)
    Outer,
    /// Page size / tokens per block (nt)
    Page,
    /// Number of attention heads (nh)
    Head,
}

/// Block layout defined by dimension ordering.
///
/// Describes how the 4 permutable dimensions (layer, outer, page, head) are
/// ordered within a fully contiguous block. The head dimension (hd) is always
/// innermost and implicit.
///
/// The order specifies outer-to-inner dimensions, with head_dim always last.
///
/// # Examples
///
/// - `UniversalTP`: `[nh, nl, no, nt, hd]` - heads outermost for TP resharding
/// - `OperationalNHD`: `[nl, no, nt, nh, hd]` - inner is `[nt, nh, hd]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvBlockLayout {
    /// Universal format: `[nh, nl, no, nt, hd]`
    ///
    /// Heads are outermost to enable tensor parallelism (TP) resharding.
    /// Cache saved from one TP configuration can be loaded into another
    /// by simply slicing the head dimension differently.
    UniversalTP,

    /// Pipeline parallelism format: `[nl, nh, no, nt, hd]`
    ///
    /// Layers are outermost for pipeline parallelism scenarios.
    UniversalPP,

    /// Operational HND format: `[nl, no, nh, nt, hd]`
    ///
    /// Inner tensor shape is `[nh, nt, hd]` (heads, tokens, head_dim).
    OperationalHND,

    /// Operational NHD format: `[nl, no, nt, nh, hd]`
    ///
    /// Inner tensor shape is `[nt, nh, hd]` (tokens, heads, head_dim).
    /// This is the most common format used by vLLM and other frameworks.
    OperationalNHD,

    /// Custom ordering with explicit dimension list.
    ///
    /// The array specifies dimensions from outermost to innermost,
    /// with head_dim always implicitly last.
    Custom([BlockDim; 4]),

    /// Unknown layout - fallback when format cannot be determined.
    ///
    /// Operations involving Unknown layouts may fail or require explicit
    /// configuration.
    Unknown,
}

impl Default for KvBlockLayout {
    fn default() -> Self {
        // OperationalNHD is the most common format
        Self::Unknown
    }
}

impl KvBlockLayout {
    /// Get the dimension ordering as an array.
    ///
    /// Returns the 4 dimensions from outermost to innermost.
    /// Head dimension (hd) is implicit as the innermost dimension.
    ///
    /// # Returns
    /// `None` for `Unknown` layout, `Some([BlockDim; 4])` otherwise.
    pub fn dim_order(&self) -> Option<[BlockDim; 4]> {
        use BlockDim::*;
        match self {
            Self::UniversalTP => Some([Head, Layer, Outer, Page]),
            Self::UniversalPP => Some([Layer, Head, Outer, Page]),
            Self::OperationalHND => Some([Layer, Outer, Head, Page]),
            Self::OperationalNHD => Some([Layer, Outer, Page, Head]),
            Self::Custom(order) => Some(*order),
            Self::Unknown => None,
        }
    }

    /// Check if two layouts require transformation (not just copy).
    ///
    /// Returns `true` if the layouts have different dimension orderings,
    /// meaning a transformation kernel is needed rather than a simple copy.
    ///
    /// Returns `true` if either layout is `Unknown` (conservative).
    pub fn requires_transform(&self, other: &Self) -> bool {
        match (self.dim_order(), other.dim_order()) {
            (Some(a), Some(b)) => a != b,
            // Unknown always requires transform (conservative)
            _ => true,
        }
    }

    /// Check if this is an operational layout (NHD or HND).
    ///
    /// Operational layouts are used for direct computation and have
    /// layer/outer as the outermost dimensions.
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::OperationalNHD | Self::OperationalHND)
    }

    /// Check if this is a universal layout (TP or PP).
    ///
    /// Universal layouts are optimized for storage and transfer,
    /// with different parallelism-friendly orderings.
    pub fn is_universal(&self) -> bool {
        matches!(self, Self::UniversalTP | Self::UniversalPP)
    }

    /// Get the layout name as a string identifier.
    pub fn name(&self) -> &'static str {
        match self {
            Self::UniversalTP => "universal_tp",
            Self::UniversalPP => "universal_pp",
            Self::OperationalHND => "operational_hnd",
            Self::OperationalNHD => "operational_nhd",
            Self::Custom(_) => "custom",
            Self::Unknown => "unknown",
        }
    }

    /// Try to create a KvBlockLayout from an InnerShape.
    ///
    /// This provides compatibility with the existing InnerShape enum.
    pub fn from_inner_shape(inner_shape: super::InnerShape) -> Self {
        match inner_shape {
            super::InnerShape::NHD => Self::OperationalNHD,
            super::InnerShape::HND => Self::OperationalHND,
            super::InnerShape::Unknown => Self::Unknown,
        }
    }

    /// Convert to InnerShape if this is an operational layout.
    ///
    /// Returns `None` for universal or custom layouts.
    pub fn to_inner_shape(&self) -> Option<super::InnerShape> {
        match self {
            Self::OperationalNHD => Some(super::InnerShape::NHD),
            Self::OperationalHND => Some(super::InnerShape::HND),
            Self::Unknown => Some(super::InnerShape::Unknown),
            _ => None,
        }
    }
}

impl std::fmt::Display for KvBlockLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UniversalTP => write!(f, "Universal TP [nh, nl, no, nt, hd]"),
            Self::UniversalPP => write!(f, "Universal PP [nl, nh, no, nt, hd]"),
            Self::OperationalHND => write!(f, "Operational HND [nl, no, nh, nt, hd]"),
            Self::OperationalNHD => write!(f, "Operational NHD [nl, no, nt, nh, hd]"),
            Self::Custom(order) => write!(f, "Custom {:?}", order),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for BlockDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layer => write!(f, "nl"),
            Self::Outer => write!(f, "no"),
            Self::Page => write!(f, "nt"),
            Self::Head => write!(f, "nh"),
        }
    }
}

// ============================================================================
// KvBlocks - Collection wrapper for blocks with shared layout
// ============================================================================

use crate::BlockId;
use crate::v2::physical::layout::PhysicalLayout;
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
///     Some(KvBlockLayout::UniversalTP),
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
    pub fn from_layout(
        layout: Arc<PhysicalLayout>,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<Self> {
        Self::new(layout, block_ids, None)
    }

    /// Get the physical layout.
    pub fn layout(&self) -> &Arc<PhysicalLayout> {
        &self.layout
    }

    /// Get the block IDs.
    pub fn block_ids(&self) -> &[BlockId] {
        &self.block_ids
    }

    /// Get the effective block layout (override or native).
    pub fn effective_block_layout(&self) -> KvBlockLayout {
        self.kv_layout_override
            .unwrap_or_else(|| self.layout.layout().block_layout())
    }

    /// Get the layout override if set.
    pub fn layout_override(&self) -> Option<KvBlockLayout> {
        self.kv_layout_override
    }

    /// Check if this collection has a layout override.
    pub fn has_override(&self) -> bool {
        self.kv_layout_override.is_some()
    }

    /// Get the number of blocks in this collection.
    pub fn len(&self) -> usize {
        self.block_ids.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Check if a transfer between two KvBlocks collections requires transformation.
    ///
    /// Returns `true` if the effective layouts differ and a transformation kernel
    /// is needed rather than a simple copy.
    pub fn requires_transform_to(&self, dst: &KvBlocks) -> bool {
        self.effective_block_layout()
            .requires_transform(&dst.effective_block_layout())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_order() {
        use BlockDim::*;

        assert_eq!(
            KvBlockLayout::UniversalTP.dim_order(),
            Some([Head, Layer, Outer, Page])
        );
        assert_eq!(
            KvBlockLayout::OperationalNHD.dim_order(),
            Some([Layer, Outer, Page, Head])
        );
        assert_eq!(KvBlockLayout::Unknown.dim_order(), None);
    }

    #[test]
    fn test_requires_transform() {
        // Same layout - no transform
        assert!(!KvBlockLayout::OperationalNHD.requires_transform(&KvBlockLayout::OperationalNHD));

        // Different layouts - transform required
        assert!(KvBlockLayout::OperationalNHD.requires_transform(&KvBlockLayout::UniversalTP));
        assert!(KvBlockLayout::OperationalHND.requires_transform(&KvBlockLayout::OperationalNHD));

        // Unknown always requires transform
        assert!(KvBlockLayout::Unknown.requires_transform(&KvBlockLayout::OperationalNHD));
        assert!(KvBlockLayout::OperationalNHD.requires_transform(&KvBlockLayout::Unknown));
    }

    #[test]
    fn test_is_operational() {
        assert!(KvBlockLayout::OperationalNHD.is_operational());
        assert!(KvBlockLayout::OperationalHND.is_operational());
        assert!(!KvBlockLayout::UniversalTP.is_operational());
        assert!(!KvBlockLayout::Unknown.is_operational());
    }

    #[test]
    fn test_is_universal() {
        assert!(KvBlockLayout::UniversalTP.is_universal());
        assert!(KvBlockLayout::UniversalPP.is_universal());
        assert!(!KvBlockLayout::OperationalNHD.is_universal());
    }

    #[test]
    fn test_default() {
        assert_eq!(KvBlockLayout::default(), KvBlockLayout::Unknown);
    }

    #[test]
    fn test_serialization() {
        let layout = KvBlockLayout::UniversalTP;
        let json = serde_json::to_string(&layout).unwrap();
        let deserialized: KvBlockLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(layout, deserialized);

        // Test custom layout
        let custom = KvBlockLayout::Custom([
            BlockDim::Head,
            BlockDim::Page,
            BlockDim::Layer,
            BlockDim::Outer,
        ]);
        let json = serde_json::to_string(&custom).unwrap();
        let deserialized: KvBlockLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(custom, deserialized);
    }

    #[test]
    fn test_inner_shape_conversion() {
        use super::super::InnerShape;

        assert_eq!(
            KvBlockLayout::from_inner_shape(InnerShape::NHD),
            KvBlockLayout::OperationalNHD
        );
        assert_eq!(
            KvBlockLayout::from_inner_shape(InnerShape::HND),
            KvBlockLayout::OperationalHND
        );

        assert_eq!(
            KvBlockLayout::OperationalNHD.to_inner_shape(),
            Some(InnerShape::NHD)
        );
        assert_eq!(KvBlockLayout::UniversalTP.to_inner_shape(), None);
    }
}
