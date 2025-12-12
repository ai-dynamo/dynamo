// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use super::InnerShape;

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

    /// Inner shape format (NHD, HND, or Unknown)
    #[builder(default = "InnerShape::Unknown")]
    pub inner_shape: InnerShape,
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
