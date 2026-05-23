// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block layout types for describing dimension permutations *inside* a block.
//!
//! These describe how the four permutable dimensions (layer, outer, page,
//! head) are ordered within a fully contiguous KV cache block; the head
//! dimension (`hd`) is always innermost and implicit. Used by physical
//! layouts and transfer kernels to choose the right transformation
//! pipeline.
//!
//! These types live in `kvbm-common` so multiple crates (`kvbm-physical`,
//! a future `kvbm-kernels`, and connector-side label inference) can refer
//! to them without depending on the layout crate.

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
/// - `Universal`: `[nh, nl, no, nt, hd]` - heads outermost for TP resharding
/// - `OperationalNHD`: `[nl, no, nt, nh, hd]` - inner is `[nt, nh, hd]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum KvBlockLayout {
    /// Universal format: `[nh, nl, no, nt, hd]`
    ///
    /// Heads are outermost to enable tensor parallelism (TP) resharding.
    /// Cache saved from one TP configuration can be loaded into another
    /// by simply slicing the head dimension differently. This is the
    /// canonical storage/transfer layout selected by
    /// `BlockLayoutMode::Universal`.
    Universal,

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
    #[default]
    Unknown,
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
            Self::Universal => Some([Head, Layer, Outer, Page]),
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
    /// For Unknown→Unknown comparisons, returns `false` (compatible) but emits
    /// a warning so these cases can be tracked and fixed.
    ///
    /// Returns `true` if one is Unknown and the other is Known (conservative).
    pub fn requires_transform(&self, other: &Self) -> bool {
        match (self.dim_order(), other.dim_order()) {
            (Some(a), Some(b)) => a != b,
            (None, None) => {
                tracing::warn!("Unknown→Unknown KvBlockLayout comparison - this should be fixed");
                false
            }
            // Unknown→Known requires transform (conservative)
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

    /// Check if this is the universal layout.
    ///
    /// Universal is optimized for storage and transfer with heads outermost
    /// for tensor-parallel resharding (canonical axis order
    /// `[Head, Layer, Outer, Page]`).
    pub fn is_universal(&self) -> bool {
        matches!(self, Self::Universal)
    }

    /// Get the layout name as a string identifier.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Universal => "universal",
            Self::OperationalHND => "operational_hnd",
            Self::OperationalNHD => "operational_nhd",
            Self::Custom(_) => "custom",
            Self::Unknown => "unknown",
        }
    }

    /// Try to create a `KvBlockLayout` from an `InnerShape`.
    pub fn from_inner_shape(inner_shape: InnerShape) -> Self {
        match inner_shape {
            InnerShape::NHD => Self::OperationalNHD,
            InnerShape::HND => Self::OperationalHND,
            InnerShape::Unknown => Self::Unknown,
        }
    }

    /// Convert to `InnerShape` if this is an operational layout.
    ///
    /// Returns `None` for universal or custom layouts.
    pub fn to_inner_shape(self) -> Option<InnerShape> {
        match self {
            Self::OperationalNHD => Some(InnerShape::NHD),
            Self::OperationalHND => Some(InnerShape::HND),
            Self::Unknown => Some(InnerShape::Unknown),
            _ => None,
        }
    }
}

impl std::fmt::Display for KvBlockLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Universal => write!(f, "Universal [nh, nl, no, nt, hd]"),
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

/// Inner shape format for tensor layout.
///
/// Describes the per-block inner three dimensions for operational layouts.
/// `kvbm-physical` and the connector both consume this; keeping it here
/// (alongside `KvBlockLayout`) lets `from_inner_shape`/`to_inner_shape`
/// stay as inherent methods on `KvBlockLayout`.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnerShape {
    /// Unknown shape - fallback when we can't determine the format
    Unknown,
    /// NHD format: [block_size, num_heads, head_dim]
    /// Common for attention layers where N=tokens, H=heads, D=dimension
    NHD,
    /// HND format: [num_heads, block_size, head_dim]
    /// Alternative layout with heads first
    HND,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_order() {
        use BlockDim::*;

        assert_eq!(
            KvBlockLayout::Universal.dim_order(),
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
        assert!(KvBlockLayout::OperationalNHD.requires_transform(&KvBlockLayout::Universal));
        assert!(KvBlockLayout::OperationalHND.requires_transform(&KvBlockLayout::OperationalNHD));

        // Unknown→Known requires transform (conservative)
        assert!(KvBlockLayout::Unknown.requires_transform(&KvBlockLayout::OperationalNHD));
        assert!(KvBlockLayout::OperationalNHD.requires_transform(&KvBlockLayout::Unknown));

        // Unknown→Unknown is compatible (but emits warning)
        assert!(!KvBlockLayout::Unknown.requires_transform(&KvBlockLayout::Unknown));
    }

    #[test]
    fn test_is_operational() {
        assert!(KvBlockLayout::OperationalNHD.is_operational());
        assert!(KvBlockLayout::OperationalHND.is_operational());
        assert!(!KvBlockLayout::Universal.is_operational());
        assert!(!KvBlockLayout::Unknown.is_operational());
    }

    #[test]
    fn test_is_universal() {
        assert!(KvBlockLayout::Universal.is_universal());
        assert!(!KvBlockLayout::OperationalNHD.is_universal());
        assert!(!KvBlockLayout::OperationalHND.is_universal());
        assert!(!KvBlockLayout::Unknown.is_universal());
    }

    #[test]
    fn test_default() {
        assert_eq!(KvBlockLayout::default(), KvBlockLayout::Unknown);
    }

    /// Reproducer for c1: collapse UniversalTP + UniversalPP into a single
    /// Universal variant carrying today's UniversalTP semantics (axis order
    /// [Head, Layer, Outer, Page], serde tag "Universal", label "universal").
    #[test]
    fn universal_is_single_variant() {
        use BlockDim::*;
        assert_eq!(
            KvBlockLayout::Universal.dim_order(),
            Some([Head, Layer, Outer, Page])
        );
        assert_eq!(KvBlockLayout::Universal.name(), "universal");
        assert!(KvBlockLayout::Universal.is_universal());
        assert!(!KvBlockLayout::Universal.is_operational());
    }

    /// Forward wire-stability guard. `KvBlockLayout` flows through
    /// `#[bincode(with_serde)]` inside `LogicalLayoutDescriptor`
    /// (lib/kvbm-physical/src/manager/metadata.rs), which encodes enum
    /// variants by **declaration-order index** (u32 varint). Reordering
    /// or deleting any variant silently shifts surviving discriminants
    /// and turns a wire-version skew into an NHD↔HND swap on decode.
    ///
    /// Lock down today's post-c1 discriminants so any future refactor
    /// that touches the enum has to acknowledge the wire impact:
    ///   0 = Universal
    ///   1 = OperationalHND
    ///   2 = OperationalNHD
    ///   3 = Custom(...)
    ///   4 = Unknown
    ///
    /// Backcompat with pre-c1 bytes is **not** preserved — nothing
    /// shipped before c1, so there are no pre-c1 bytes in the wild.
    #[test]
    fn variant_discriminants_are_wire_stable() {
        let cfg = bincode::config::standard();
        let first_byte =
            |k: KvBlockLayout| -> u8 { bincode::serde::encode_to_vec(k, cfg).unwrap()[0] };
        assert_eq!(first_byte(KvBlockLayout::Universal), 0);
        assert_eq!(first_byte(KvBlockLayout::OperationalHND), 1);
        assert_eq!(first_byte(KvBlockLayout::OperationalNHD), 2);
        assert_eq!(
            first_byte(KvBlockLayout::Custom([
                BlockDim::Layer,
                BlockDim::Outer,
                BlockDim::Page,
                BlockDim::Head,
            ])),
            3
        );
        assert_eq!(first_byte(KvBlockLayout::Unknown), 4);
    }

    #[test]
    fn test_serialization() {
        let layout = KvBlockLayout::Universal;
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
        assert_eq!(KvBlockLayout::Universal.to_inner_shape(), None);
    }
}
