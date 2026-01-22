// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

/// Configuration for block layouts
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

    /// Page size
    #[validate(range(min = 1))]
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

    /// Get the head dimension if `num_heads` is specified.
    ///
    /// Computes `inner_dim / (page_size * num_heads)`.
    ///
    /// # Returns
    /// `Some(head_dim)` if `num_heads` is set, `None` otherwise.
    pub fn head_dim(&self) -> Option<usize> {
        self.num_heads.map(|nh| {
            let divisor = self.page_size * nh;
            if divisor > 0 {
                self.inner_dim / divisor
            } else {
                0
            }
        })
    }

    /// Check if this config supports KvBlockLayout operations.
    ///
    /// Returns `true` if `num_heads` is set and the dimensions are valid
    /// (inner_dim is evenly divisible by page_size * num_heads).
    pub fn supports_kv_block_layout(&self) -> bool {
        if let Some(nh) = self.num_heads {
            let divisor = self.page_size * nh;
            divisor > 0 && self.inner_dim % divisor == 0
        } else {
            false
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

        let divisor = self.page_size * nh;
        if self.inner_dim % divisor != 0 {
            return Err(ValidationError::new(
                "inner_dim_must_be_divisible_by_page_size_times_num_heads",
            ));
        }

        Ok(())
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

pub fn validate_dtype_width_bytes(dtype_width_bytes: usize) -> Result<(), ValidationError> {
    if !dtype_width_bytes.is_power_of_two() || !(2..=8).contains(&dtype_width_bytes) {
        return Err(validator::ValidationError::new(
            "dtype_width_bytes_must_be_power_of_two_and_less_than_8_bytes",
        ));
    }
    Ok(())
}
