// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use kvbm_kernels::TensorDataType;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

/// Configuration for block layouts.
///
/// The `#[validate]` attributes on fields are checked during layout construction
/// (e.g., `FullyContiguousLayout::new_internal()`, `LayerSeparateLayout::new_internal()`),
/// not at builder `.build()` time.
#[derive(Debug, Clone, Builder, Validate, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutConfig {
    /// Number of blocks
    #[validate(range(min = 1))]
    pub num_blocks: usize,

    /// Number of layers
    #[validate(range(min = 1))]
    pub num_layers: usize,

    /// Number of outer dimensions
    #[validate(range(min = 1, max = 2))]
    pub outer_dim: usize,

    /// Page size (tokens per block).
    ///
    /// Must be a positive power of two. KVBM layout indexing assumes
    /// power-of-two page sizes for stride/alignment math; non-power-of-two
    /// values are rejected at layout construction time.
    #[validate(custom(function = "validate_page_size"))]
    pub page_size: usize,

    /// Inner dimension
    #[validate(range(min = 1))]
    pub inner_dim: usize,

    /// Alignment
    #[validate(custom(function = "validate_power_of_2"))]
    #[builder(default = "1")]
    pub alignment: usize,

    /// Data type
    #[validate(custom(function = "validate_dtype_width_bytes"))]
    #[builder(default = "2")]
    pub dtype_width_bytes: usize,

    /// Number of attention heads (optional).
    ///
    /// When provided, enables KvBlockLayout support for universal formats.
    /// The head dimension can be computed as: `inner_dim / (page_size * num_heads)`.
    ///
    /// Required for:
    /// - Universal layout transformations
    /// - Per-head memory region access
    #[builder(default = "None")]
    #[serde(default)]
    pub num_heads: Option<usize>,

    /// Tensor element dtype (`F16`, `BF16`, `F32`, `F64`).
    ///
    /// Required by the kernel catalog when projecting a layout with a
    /// known [`KvBlockLayout`] for transform dispatch — the kernels
    /// dispatch on dtype templates (F16 vs BF16 differ even at the
    /// same byte width). Validated at the projection site
    /// (`transfer::lower::layout_to_view`), not at `build()`, so legacy
    /// callers that don't enable `use_planner = true` are unaffected.
    ///
    /// `#[serde(skip)]` because [`TensorDataType`] does not implement
    /// serde traits in `kvbm-kernels`. Cross-process layout reconstruction
    /// will need to plumb this separately when wire transport lands.
    #[builder(default = "None")]
    #[serde(skip)]
    pub dtype: Option<TensorDataType>,
}

impl LayoutConfig {
    /// Builder for LayoutConfig
    pub fn builder() -> LayoutConfigBuilder {
        LayoutConfigBuilder::default()
    }

    pub fn required_bytes(&self) -> usize {
        self.num_blocks
            .saturating_mul(self.num_layers)
            .saturating_mul(self.outer_dim)
            .saturating_mul(self.page_size)
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }

    /// Get the number of bytes per block.
    ///
    /// This is the total size of a single block across all layers and outer dimensions.
    pub fn bytes_per_block(&self) -> usize {
        self.num_layers
            .saturating_mul(self.outer_dim)
            .saturating_mul(self.page_size)
            .saturating_mul(self.inner_dim)
            .saturating_mul(self.dtype_width_bytes)
    }

    /// Get the head dimension if `num_heads` is specified.
    ///
    /// In KVBM's layout model `inner_dim` is per-token bytes-per-head times
    /// `num_heads` — i.e. `inner_dim = num_heads * head_dim`. The page (token)
    /// axis is multiplied separately in `bytes_per_block`, so it must NOT
    /// appear in this divisor.
    ///
    /// # Returns
    /// `Some(head_dim)` if `num_heads` is set, `None` otherwise.
    pub fn head_dim(&self) -> Option<usize> {
        self.num_heads
            .map(|nh| self.inner_dim.checked_div(nh).unwrap_or(0))
    }

    /// The effective tensor dtype for this layout.
    ///
    /// Returns `self.dtype` when explicitly set; otherwise derives a
    /// best-effort `TensorDataType` from `dtype_width_bytes` via
    /// [`derive_tensor_dtype_from_width`]. Production layout builders
    /// (vLLM connector, engine parallelism, spmd, describe_map) set
    /// `dtype_width_bytes` but historically left `dtype` unset; the
    /// kernel catalog needs a concrete `TensorDataType` for dispatch,
    /// and deriving here keeps Universal-mode offload working without
    /// plumbing the typed dtype through every layout-construction site.
    /// Callers that need the explicit dtype (e.g. distinguishing F16 vs
    /// BF16 for arithmetic kernels) must still set it on the builder.
    ///
    /// `None` only when the width itself doesn't map to any known
    /// dtype (see [`derive_tensor_dtype_from_width`]).
    pub fn effective_dtype(&self) -> Option<TensorDataType> {
        self.dtype
            .or_else(|| derive_tensor_dtype_from_width(self.dtype_width_bytes))
    }

    /// True if `self` and `other` agree on every per-block shape
    /// field.
    ///
    /// `num_blocks` is excluded — it represents per-tier capacity, not
    /// per-block geometry, and may legitimately differ across tiers
    /// (e.g. G1 GPU HBM vs G2 pinned host typically hold different
    /// block counts even when the model's per-block shape is
    /// identical). Every other field is compared with `==`: kernel
    /// dispatch needs identical `num_layers`, `outer_dim`,
    /// `page_size`, `inner_dim`, `alignment`, `dtype_width_bytes`,
    /// `num_heads`, and `dtype` on both sides.
    pub fn has_same_block_shape(&self, other: &Self) -> bool {
        self.num_layers == other.num_layers
            && self.outer_dim == other.outer_dim
            && self.page_size == other.page_size
            && self.inner_dim == other.inner_dim
            && self.alignment == other.alignment
            && self.dtype_width_bytes == other.dtype_width_bytes
            && self.num_heads == other.num_heads
            && self.dtype == other.dtype
    }

    /// Check if this config supports KvBlockLayout operations.
    ///
    /// Returns `true` if `num_heads` is set and `inner_dim` is evenly
    /// divisible by `num_heads`.
    pub fn supports_kv_block_layout(&self) -> bool {
        match self.num_heads {
            Some(nh) => nh > 0 && self.inner_dim.is_multiple_of(nh),
            None => false,
        }
    }

    /// Validate that this config supports KvBlockLayout operations.
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err` with details otherwise.
    pub fn validate_for_kv_block_layout(&self) -> Result<(), ValidationError> {
        let nh = match self.num_heads {
            Some(nh) => nh,
            None => {
                return Err(ValidationError::new(
                    "num_heads_required_for_kv_block_layout",
                ));
            }
        };

        if nh == 0 {
            return Err(ValidationError::new("num_heads_must_be_positive"));
        }

        if !self.inner_dim.is_multiple_of(nh) {
            return Err(ValidationError::new(
                "inner_dim_must_be_divisible_by_num_heads",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod head_dim_tests {
    use super::*;

    fn cfg(inner_dim: usize, num_heads: Option<usize>) -> LayoutConfig {
        let mut b = LayoutConfig::builder();
        b.num_blocks(1)
            .num_layers(1)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(inner_dim)
            .dtype_width_bytes(2);
        if let Some(nh) = num_heads {
            b.num_heads(Some(nh));
        }
        b.build().unwrap()
    }

    /// `inner_dim = num_heads * head_dim`. For Llama-style `nh=8, hd=128`,
    /// `inner_dim=1024 → head_dim=128`. The old buggy formula divided by
    /// `page_size * nh = 16 * 8 = 128`, returning `8` — verifiably wrong
    /// because actual `head_dim` for Llama-3-8B is 128.
    #[test]
    fn head_dim_is_inner_dim_over_num_heads() {
        let c = cfg(8 * 128, Some(8));
        assert_eq!(c.head_dim(), Some(128));
    }

    #[test]
    fn head_dim_none_without_num_heads() {
        let c = cfg(1024, None);
        assert_eq!(c.head_dim(), None);
    }

    #[test]
    fn supports_kv_block_layout_requires_divisibility_by_num_heads() {
        assert!(cfg(1024, Some(8)).supports_kv_block_layout());
        // 1023 / 8 has a remainder — should be rejected.
        assert!(!cfg(1023, Some(8)).supports_kv_block_layout());
        assert!(!cfg(1024, None).supports_kv_block_layout());
    }

    #[test]
    fn validate_for_kv_block_layout_rejects_non_divisible_inner_dim() {
        let err = cfg(1023, Some(8))
            .validate_for_kv_block_layout()
            .unwrap_err();
        assert_eq!(err.code, "inner_dim_must_be_divisible_by_num_heads");
    }
}

/// The first two dimensions of the tensor, `shape[0]` and `shape[1]`, one of those corresponds to the
/// block dimension, while the other corresponds to the outer dimension.
///
/// The outer dimension is typically:
/// - 1: MLA or K and V stored together,
/// - 2: K and V stored separately,
///
/// The block dimension tell us the number of blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockDimension {
    /// The block dimension is the first dimension of the tensor, `[n_blocks, outer_dim, inner_dim]`
    BlockIsFirstDim,

    /// The block dimension is the second dimension of the tensor, `[outer_dim, n_blocks, inner_dim]`
    /// This is a replacement for v1's `outer_contiguous` is true.
    BlockIsSecondDim,
}

/// Validation function for Option<usize> to check if it's Some(power_of_2).
pub fn validate_power_of_2(alignment: usize) -> Result<(), ValidationError> {
    if !alignment.is_power_of_two() {
        // Return validation error if alignment is not a power of 2
        return Err(validator::ValidationError::new(
            "alignment_must_be_power_of_2",
        ));
    }
    // Passes validation if alignment is a power of 2
    Ok(())
}

/// Validation for `page_size`: must be a positive power of two.
///
/// KVBM layout indexing assumes power-of-two page sizes (stride/alignment
/// math, chunked transfers). Non-power-of-two values are not supported.
pub fn validate_page_size(page_size: usize) -> Result<(), ValidationError> {
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(validator::ValidationError::new(
            "page_size_must_be_positive_power_of_2",
        ));
    }
    Ok(())
}

pub fn validate_dtype_width_bytes(dtype_width_bytes: usize) -> Result<(), ValidationError> {
    if !dtype_width_bytes.is_power_of_two() || !(1..=8).contains(&dtype_width_bytes) {
        return Err(validator::ValidationError::new(
            "dtype_width_bytes_must_be_power_of_two_and_at_most_8_bytes",
        ));
    }
    Ok(())
}

/// Best-effort mapping from KV element byte width to a [`TensorDataType`].
///
/// Used by [`LayoutConfig::effective_dtype`] when the layout was built
/// without an explicit dtype. The mapping picks one canonical type per
/// width:
///
/// | width | dtype  | notes |
/// |-------|--------|-------|
/// |   1   | `FP8`  | 1-byte float. Handled by a dtype-agnostic byte-mover (uint8_t kernel template); catalog dispatches FP8 transforms like any other dtype. |
/// |   2   | `BF16` | Modern default. Permute kernels produce identical bytes for `F16` and `BF16`, so the choice only matters for arithmetic-bearing kernels (none today). |
/// |   4   | `F32`  | Sole 4-byte type today. |
/// |   8   | `F64`  | Sole 8-byte type today. |
///
/// Any other width returns `None` — `validate_dtype_width_bytes` caps
/// widths at the power-of-two set `{1,2,4,8}`, so callers reaching this
/// helper with an unsupported width have a deeper validation gap.
///
/// Callers who require disambiguation (`F16` vs `BF16`, or specific
/// FP8 formats) must set `LayoutConfig.dtype` explicitly at construction.
pub fn derive_tensor_dtype_from_width(dtype_width_bytes: usize) -> Option<TensorDataType> {
    match dtype_width_bytes {
        1 => Some(TensorDataType::FP8),
        2 => Some(TensorDataType::BF16),
        4 => Some(TensorDataType::F32),
        8 => Some(TensorDataType::F64),
        _ => None,
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod derive_dtype_tests {
    use super::*;

    #[test]
    fn derive_covers_validated_widths() {
        assert_eq!(derive_tensor_dtype_from_width(1), Some(TensorDataType::FP8));
        assert_eq!(
            derive_tensor_dtype_from_width(2),
            Some(TensorDataType::BF16)
        );
        assert_eq!(derive_tensor_dtype_from_width(4), Some(TensorDataType::F32));
        assert_eq!(derive_tensor_dtype_from_width(8), Some(TensorDataType::F64));
    }

    #[test]
    fn derive_rejects_widths_outside_validated_set() {
        assert_eq!(derive_tensor_dtype_from_width(0), None);
        assert_eq!(derive_tensor_dtype_from_width(3), None);
        assert_eq!(derive_tensor_dtype_from_width(16), None);
    }

    #[test]
    fn effective_dtype_prefers_explicit() {
        // dtype set explicitly → returned as-is, not derived from width.
        let cfg = LayoutConfig::builder()
            .num_blocks(1)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .dtype(Some(TensorDataType::F16))
            .build()
            .unwrap();
        assert_eq!(cfg.effective_dtype(), Some(TensorDataType::F16));
    }

    #[test]
    fn effective_dtype_falls_back_to_derived() {
        // dtype unset, width=2 → BF16 (the c6 derive default for 2 bytes).
        let cfg = LayoutConfig::builder()
            .num_blocks(1)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();
        assert_eq!(cfg.effective_dtype(), Some(TensorDataType::BF16));
    }
}
